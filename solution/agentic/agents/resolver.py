"""
Resolver Agent

Uses RAG (Retrieval-Augmented Generation) to find relevant knowledge base
articles and generate responses. Calculates confidence scores to determine
if escalation is needed.
"""

import os
import sys
from typing import Dict, List
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from sqlalchemy import create_engine

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from agentic.tools.knowledge_search_tool import KnowledgeSearchTool, expand_query
from data.models import udahub
from utils import get_session


class ResolverAgent:
    """
    Agent responsible for resolving tickets using knowledge base retrieval.

    Uses RAG to find relevant articles and generate appropriate responses.
    Includes confidence scoring to determine when escalation is needed.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.3,
        db_path: str = "data/core/udahub.db",
        index_path: str = "data/core/faiss_index.bin",
        metadata_path: str = "data/core/faiss_metadata.pkl"
    ):
        """
        Initialize the resolver agent.

        Args:
            model: OpenAI model to use
            temperature: Model temperature (0.3 for balanced creativity)
            db_path: Path to UDA-Hub database
            index_path: Path to FAISS index
            metadata_path: Path to FAISS metadata
        """
        self.llm = ChatOpenAI(model=model, temperature=temperature)
        self.knowledge_tool = KnowledgeSearchTool(
            index_path=index_path,
            metadata_path=metadata_path
        )
        self.db_path = db_path

    def resolve(
        self,
        query: str,
        classification: Dict,
        account_id: str,
        tool_results: Dict = None
    ) -> Dict:
        """
        Attempt to resolve a ticket using knowledge base.

        Args:
            query: User's question/issue
            classification: Classification from classifier agent
            account_id: Account ID for knowledge base filtering
            tool_results: Optional results from tool execution

        Returns:
            Resolution dictionary with keys:
                - resolution: Response text
                - confidence: Confidence score (0-1)
                - sources: List of source article titles
                - needs_escalation: Boolean
                - recommended_action: Next step recommendation
        """
        # Expand query with domain-specific terms
        expanded_query = expand_query(query, classification)

        # Search knowledge base
        try:
            search_results = self.knowledge_tool.search(
                query=expanded_query,
                top_k=3,
                similarity_threshold=0.6  # Slightly lower for more results
            )
        except Exception as e:
            print(f"⚠️  Knowledge search error: {e}")
            search_results = []

        if not search_results:
            return {
                "resolution": None,
                "confidence": 0.0,
                "sources": [],
                "needs_escalation": True,
                "recommended_action": "escalate_no_knowledge",
                "reason": "No relevant knowledge articles found"
            }

        # Retrieve full article content from database
        articles = self._get_full_articles(search_results, account_id)

        # Generate response using LLM with retrieved context
        resolution_text, confidence = self._generate_response(
            query=query,
            classification=classification,
            articles=articles,
            tool_results=tool_results
        )

        # Determine if escalation is needed
        needs_escalation = confidence < 0.5 or self._detect_escalation_signals(resolution_text)

        return {
            "resolution": resolution_text,
            "confidence": confidence,
            "sources": [a['title'] for a in articles],
            "needs_escalation": needs_escalation,
            "recommended_action": "escalate" if needs_escalation else "resolve"
        }

    def _get_full_articles(self, search_results: List[Dict], account_id: str) -> List[Dict]:
        """
        Retrieve full article content from database.

        Args:
            search_results: List of search results with article_ids
            account_id: Account ID for filtering

        Returns:
            List of article dictionaries with full content
        """
        # Convert to absolute path if relative
        if not os.path.isabs(self.db_path):
            solution_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            db_path = os.path.join(solution_dir, self.db_path)
        else:
            db_path = self.db_path

        engine = create_engine(f"sqlite:///{db_path}", echo=False)

        articles = []

        try:
            with get_session(engine) as session:
                for result in search_results:
                    article = session.query(udahub.Knowledge)\
                        .filter_by(article_id=result['article_id'])\
                        .first()

                    if article:
                        articles.append({
                            'title': article.title,
                            'content': article.content,
                            'tags': article.tags,
                            'similarity': result['similarity']
                        })

        except Exception as e:
            print(f"⚠️  Database error retrieving articles: {e}")

        return articles

    def _generate_response(
        self,
        query: str,
        classification: Dict,
        articles: List[Dict],
        tool_results: Dict = None
    ) -> tuple[str, float]:
        """
        Generate response using LLM with retrieved articles.

        Args:
            query: User query
            classification: Ticket classification
            articles: Retrieved knowledge articles
            tool_results: Optional tool execution results

        Returns:
            Tuple of (response_text, confidence_score)
        """
        # Prepare context from articles
        context_parts = []
        for i, article in enumerate(articles, 1):
            context_parts.append(
                f"**Article {i}: {article['title']}** (Relevance: {article['similarity']:.2f})\n"
                f"{article['content']}\n"
            )

        context = "\n---\n\n".join(context_parts)

        # Add tool results if available
        tool_context = ""
        if tool_results:
            tool_context = f"\n\nAdditional Information from System:\n{tool_results}\n"

        system_prompt = """You are a helpful customer support agent for CultPass.

Your goal is to provide accurate, friendly, and helpful responses to customer inquiries.

Guidelines:
1. Base your response ONLY on the provided knowledge articles
2. Be concise but complete
3. Use a friendly, professional tone
4. Follow any "Suggested phrasing" in the articles
5. If the articles don't contain enough information, clearly state what you don't know
6. Don't make up information not in the articles
7. Be empathetic, especially if the customer seems frustrated

If you're uncertain or the knowledge articles don't fully address the question, include a brief note like:
"I may need to escalate this to a specialist for complete assistance."
"""

        user_prompt = f"""Knowledge Base Articles:
{context}
{tool_context}

Customer Issue Classification:
- Type: {classification.get('ticket_type', 'general')}
- Urgency: {classification.get('urgency', 'medium')}
- Sentiment: {classification.get('sentiment', 'neutral')}

Customer Question:
{query}

Provide a helpful response based on the knowledge articles above."""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]

        response = self.llm.invoke(messages)
        resolution_text = response.content

        # Calculate confidence score
        confidence = self._calculate_confidence(articles, resolution_text, classification)

        return resolution_text, confidence

    def _calculate_confidence(
        self,
        articles: List[Dict],
        resolution: str,
        classification: Dict
    ) -> float:
        """
        Calculate confidence score for the resolution.

        Args:
            articles: Retrieved articles with similarity scores
            resolution: Generated resolution text
            classification: Ticket classification

        Returns:
            Confidence score between 0 and 1
        """
        if not articles:
            return 0.0

        # Base confidence from top article similarity
        top_similarity = articles[0]['similarity']

        # Adjust for number of results (more results = more confidence)
        num_results_factor = min(len(articles) / 3.0, 1.0)

        # Penalty for uncertainty phrases
        uncertainty_phrases = [
            "i'm not sure",
            "i don't have enough information",
            "might be",
            "possibly",
            "unclear",
            "may need to escalate",
            "specialist",
            "not certain"
        ]

        uncertainty_penalty = 0.0
        resolution_lower = resolution.lower()
        for phrase in uncertainty_phrases:
            if phrase in resolution_lower:
                uncertainty_penalty = 0.3
                break

        # Boost for high urgency with high similarity
        urgency_boost = 0.0
        if classification.get('urgency') in ['critical', 'high'] and top_similarity > 0.85:
            urgency_boost = 0.1

        # Calculate final confidence
        confidence = (
            top_similarity * 0.7 +
            num_results_factor * 0.2 +
            urgency_boost * 0.1 -
            uncertainty_penalty
        )

        return max(0.0, min(1.0, confidence))

    def _detect_escalation_signals(self, resolution: str) -> bool:
        """
        Detect if resolution contains signals that escalation is needed.

        Args:
            resolution: Generated resolution text

        Returns:
            True if escalation signals detected
        """
        escalation_signals = [
            "escalate",
            "specialist",
            "human agent",
            "support team",
            "can't help with this",
            "beyond my capabilities"
        ]

        resolution_lower = resolution.lower()
        return any(signal in resolution_lower for signal in escalation_signals)
