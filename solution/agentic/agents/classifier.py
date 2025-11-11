"""
Classifier Agent

Analyzes incoming support tickets and extracts structured information
including ticket type, urgency, sentiment, entities, and required expertise.
"""

import json
from typing import Dict
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage


class ClassifierAgent:
    """
    Agent responsible for classifying support tickets.

    Extracts ticket type, urgency, sentiment, entities, and determines
    which type of expertise is needed to resolve the ticket.
    """

    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.1):
        """
        Initialize the classifier agent.

        Args:
            model: OpenAI model to use
            temperature: Model temperature (lower = more deterministic)
        """
        self.llm = ChatOpenAI(model=model, temperature=temperature)

    def classify(self, user_message: str, customer_history: str = "") -> Dict:
        """
        Classify a support ticket.

        Args:
            user_message: User's support ticket message
            customer_history: Optional customer history context

        Returns:
            Classification dictionary with keys:
                - ticket_type: Type of issue
                - urgency: Urgency level
                - sentiment: Customer sentiment
                - entities: Extracted entities
                - required_expertise: List of required expertise types
                - keywords: List of relevant keywords
        """
        system_prompt = """You are a classification agent for CultPass customer support.
Your job is to analyze customer support tickets and extract structured information.

For each ticket, you must classify:

1. **Ticket Type** (choose ONE):
   - login: Login/authentication issues
   - billing: Payment, subscription, refund issues
   - reservation: Booking, cancellation, event-related issues
   - technical: App bugs, crashes, performance issues
   - account: Profile, settings, account management
   - general: General inquiries

2. **Urgency** (choose ONE):
   - critical: Account blocked, cannot access service, payment issues
   - high: Event starts soon, urgent reservation issues
   - medium: General issues that need resolution
   - low: General questions, feature requests

3. **Sentiment** (choose ONE):
   - positive: Happy, satisfied customer
   - neutral: Matter-of-fact, no strong emotion
   - negative: Frustrated but calm
   - frustrated: Clearly frustrated, may use strong language

4. **Entities**: Extract relevant information:
   - user_id: If mentioned (e.g., "my ID is abc123")
   - email: If mentioned
   - reservation_id: If mentioned
   - experience_name: If specific event mentioned
   - amount: If money mentioned

5. **Required Expertise** (choose ALL that apply):
   - knowledge_base: Can be answered from help articles
   - database_operations: Needs to lookup/modify user data
   - human_agent: Complex issue requiring human judgment

6. **Keywords**: Extract 3-5 relevant keywords from the message

Return your analysis as a JSON object with these exact keys:
{
  "ticket_type": "...",
  "urgency": "...",
  "sentiment": "...",
  "entities": {...},
  "required_expertise": [...],
  "keywords": [...]
}"""

        user_prompt = f"""Classify this support ticket:

{customer_history}

Customer Message:
{user_message}

Return ONLY the JSON classification object, no other text."""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]

        response = self.llm.invoke(messages)

        # Parse JSON response
        try:
            classification = json.loads(response.content)

            # Validate required keys
            required_keys = ['ticket_type', 'urgency', 'sentiment', 'entities', 'required_expertise', 'keywords']
            for key in required_keys:
                if key not in classification:
                    classification[key] = self._get_default_value(key)

            return classification

        except json.JSONDecodeError as e:
            # Fallback classification
            print(f"⚠️  Classification JSON parse error: {e}")
            return self._fallback_classification(user_message)

    def _get_default_value(self, key: str):
        """Get default value for missing classification key."""
        defaults = {
            'ticket_type': 'general',
            'urgency': 'medium',
            'sentiment': 'neutral',
            'entities': {},
            'required_expertise': ['knowledge_base'],
            'keywords': []
        }
        return defaults.get(key)

    def _fallback_classification(self, message: str) -> Dict:
        """
        Provide fallback classification if LLM fails.

        Uses simple keyword matching.
        """
        message_lower = message.lower()

        # Simple ticket type detection
        if any(word in message_lower for word in ['login', 'password', 'access', 'sign in']):
            ticket_type = 'login'
        elif any(word in message_lower for word in ['billing', 'payment', 'refund', 'charge']):
            ticket_type = 'billing'
        elif any(word in message_lower for word in ['reservation', 'booking', 'cancel', 'reserve', 'event']):
            ticket_type = 'reservation'
        elif any(word in message_lower for word in ['app', 'crash', 'bug', 'error', 'loading']):
            ticket_type = 'technical'
        elif any(word in message_lower for word in ['account', 'profile', 'settings']):
            ticket_type = 'account'
        else:
            ticket_type = 'general'

        # Simple urgency detection
        if any(word in message_lower for word in ['urgent', 'asap', 'immediately', 'blocked', 'cannot access']):
            urgency = 'high'
        elif any(word in message_lower for word in ['soon', 'event today', 'event tomorrow']):
            urgency = 'medium'
        else:
            urgency = 'medium'

        # Simple sentiment detection
        if any(word in message_lower for word in ['frustrated', 'angry', 'terrible', 'awful', 'worst']):
            sentiment = 'frustrated'
        elif any(word in message_lower for word in ['disappointed', 'unhappy', 'issue', 'problem']):
            sentiment = 'negative'
        else:
            sentiment = 'neutral'

        return {
            'ticket_type': ticket_type,
            'urgency': urgency,
            'sentiment': sentiment,
            'entities': {},
            'required_expertise': ['knowledge_base'],
            'keywords': []
        }
