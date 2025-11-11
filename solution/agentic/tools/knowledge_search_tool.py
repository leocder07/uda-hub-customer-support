"""
Knowledge Base Search Tool using RAG with FAISS Vector Store

This module implements semantic search over the CultPass knowledge base
using OpenAI embeddings and FAISS for efficient similarity search.
"""

import os
import json
import pickle
import numpy as np
from typing import List, Dict, Optional
from openai import OpenAI


class KnowledgeSearchTool:
    """
    Tool for semantic search over knowledge base articles.

    Uses FAISS for vector similarity search and OpenAI embeddings.
    """

    def __init__(
        self,
        index_path: str = "data/core/faiss_index.bin",
        metadata_path: str = "data/core/faiss_metadata.pkl",
        embedding_model: str = "text-embedding-3-small"
    ):
        """
        Initialize the knowledge search tool.

        Args:
            index_path: Path to FAISS index file
            metadata_path: Path to metadata pickle file
            embedding_model: OpenAI embedding model name
        """
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.embedding_model = embedding_model
        self.client = OpenAI()
        self.index = None
        self.metadata = None
        self.dimension = 1536  # text-embedding-3-small dimension

    def build_index(self, knowledge_articles: List[Dict]) -> None:
        """
        Build FAISS index from knowledge articles.

        Args:
            knowledge_articles: List of article dictionaries with keys:
                - article_id: Unique identifier
                - title: Article title
                - content: Article content
                - tags: Comma-separated tags
        """
        try:
            import faiss
        except ImportError:
            raise ImportError(
                "FAISS not installed. Install with: pip install faiss-cpu"
            )

        print(f"📚 Building FAISS index for {len(knowledge_articles)} articles...")

        # Prepare documents
        documents = []
        metadata = []

        for article in knowledge_articles:
            # Combine title, content, and tags for better semantic search
            combined_text = (
                f"{article['title']}\n\n"
                f"{article['content']}\n\n"
                f"Tags: {article['tags']}"
            )
            documents.append(combined_text)
            metadata.append({
                'article_id': article['article_id'],
                'title': article['title'],
                'tags': article['tags']
            })

        # Generate embeddings in batches
        embeddings = []
        batch_size = 100

        for i in range(0, len(documents), batch_size):
            batch = documents[i:i+batch_size]
            print(f"  Embedding batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1}...")

            response = self.client.embeddings.create(
                input=batch,
                model=self.embedding_model
            )

            batch_embeddings = [item.embedding for item in response.data]
            embeddings.extend(batch_embeddings)

        # Convert to numpy array
        embeddings_array = np.array(embeddings).astype('float32')

        # Create FAISS index (using L2 distance)
        dimension = embeddings_array.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings_array)

        # Create directories if they don't exist
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.metadata_path), exist_ok=True)

        # Save index and metadata
        faiss.write_index(index, self.index_path)
        with open(self.metadata_path, "wb") as f:
            pickle.dump(metadata, f)

        print(f"✅ FAISS index built successfully!")
        print(f"   - Index saved to: {self.index_path}")
        print(f"   - Metadata saved to: {self.metadata_path}")
        print(f"   - Indexed {len(documents)} articles with {dimension} dimensions")

        self.index = index
        self.metadata = metadata

    def load_index(self) -> None:
        """Load FAISS index and metadata from disk."""
        try:
            import faiss

            if not os.path.exists(self.index_path):
                raise FileNotFoundError(
                    f"FAISS index not found at {self.index_path}. "
                    "Please build the index first using build_index()."
                )

            if not os.path.exists(self.metadata_path):
                raise FileNotFoundError(
                    f"Metadata not found at {self.metadata_path}. "
                    "Please build the index first using build_index()."
                )

            self.index = faiss.read_index(self.index_path)
            with open(self.metadata_path, "rb") as f:
                self.metadata = pickle.load(f)

            print(f"✅ Loaded FAISS index with {self.index.ntotal} articles")

        except ImportError:
            raise ImportError(
                "FAISS not installed. Install with: pip install faiss-cpu"
            )

    def search(
        self,
        query: str,
        top_k: int = 3,
        similarity_threshold: float = 0.7
    ) -> List[Dict]:
        """
        Search knowledge base using semantic similarity.

        Args:
            query: User query string
            top_k: Number of top results to return
            similarity_threshold: Minimum similarity score (0-1)

        Returns:
            List of result dictionaries with keys:
                - article_id: Article identifier
                - title: Article title
                - tags: Article tags
                - similarity: Similarity score (0-1)
        """
        if self.index is None or self.metadata is None:
            self.load_index()

        # Generate query embedding
        response = self.client.embeddings.create(
            input=query,
            model=self.embedding_model
        )
        query_embedding = np.array([response.data[0].embedding]).astype('float32')

        # Search FAISS index
        distances, indices = self.index.search(query_embedding, top_k)

        # Convert L2 distance to similarity score (0-1)
        # L2 distance: 0 = identical, higher = more different
        # We normalize by a max distance threshold
        max_distance = 2.0  # Empirically determined
        similarities = 1 - np.minimum(distances[0] / max_distance, 1.0)

        # Filter by threshold and prepare results
        results = []
        for idx, sim in zip(indices[0], similarities):
            if sim >= similarity_threshold:
                results.append({
                    'article_id': self.metadata[idx]['article_id'],
                    'title': self.metadata[idx]['title'],
                    'tags': self.metadata[idx]['tags'],
                    'similarity': float(sim)
                })

        return results

    def add_article(self, article: Dict) -> None:
        """
        Add a single article to the existing index.

        Args:
            article: Article dictionary with keys:
                - article_id: Unique identifier
                - title: Article title
                - content: Article content
                - tags: Comma-separated tags
        """
        try:
            import faiss
        except ImportError:
            raise ImportError(
                "FAISS not installed. Install with: pip install faiss-cpu"
            )

        if self.index is None or self.metadata is None:
            self.load_index()

        # Prepare document
        combined_text = (
            f"{article['title']}\n\n"
            f"{article['content']}\n\n"
            f"Tags: {article['tags']}"
        )

        # Generate embedding
        response = self.client.embeddings.create(
            input=combined_text,
            model=self.embedding_model
        )
        embedding = np.array([response.data[0].embedding]).astype('float32')

        # Add to index
        self.index.add(embedding)

        # Update metadata
        self.metadata.append({
            'article_id': article['article_id'],
            'title': article['title'],
            'tags': article['tags']
        })

        # Save updated index and metadata
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, "wb") as f:
            pickle.dump(self.metadata, f)

        print(f"✅ Added article '{article['title']}' to index")


def expand_query(query: str, classification: Optional[Dict] = None) -> str:
    """
    Expand user query with domain-specific terms to improve retrieval.

    Args:
        query: Original user query
        classification: Optional classification dictionary with 'ticket_type'

    Returns:
        Expanded query string
    """
    if not classification:
        return query

    # Domain-specific term expansions
    expansions = {
        'login': ['password', 'authentication', 'access', 'account', 'sign in'],
        'billing': ['payment', 'subscription', 'refund', 'charges', 'credit card'],
        'reservation': ['booking', 'event', 'experience', 'cancel', 'reserve'],
        'technical': ['app', 'error', 'bug', 'crash', 'loading', 'issue'],
        'account': ['profile', 'settings', 'update', 'change', 'manage']
    }

    ticket_type = classification.get('ticket_type', '')

    if ticket_type in expansions:
        # Add related terms to query
        related_terms = ' '.join(expansions[ticket_type][:3])  # Add top 3 terms
        expanded = f"{query} {related_terms}"
        return expanded

    return query


# Convenience function for direct use
def search_knowledge_base(
    query: str,
    top_k: int = 3,
    similarity_threshold: float = 0.7,
    index_path: str = "data/core/faiss_index.bin",
    metadata_path: str = "data/core/faiss_metadata.pkl"
) -> List[Dict]:
    """
    Convenience function for searching knowledge base.

    Args:
        query: User query string
        top_k: Number of top results to return
        similarity_threshold: Minimum similarity score (0-1)
        index_path: Path to FAISS index file
        metadata_path: Path to metadata pickle file

    Returns:
        List of result dictionaries
    """
    tool = KnowledgeSearchTool(index_path=index_path, metadata_path=metadata_path)
    return tool.search(query, top_k=top_k, similarity_threshold=similarity_threshold)
