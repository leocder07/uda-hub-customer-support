"""
Escalation Agent

Creates comprehensive escalation summaries for human agents when automated
resolution is not possible or appropriate.
"""

from typing import Dict, List
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage


class EscalationAgent:
    """
    Agent responsible for creating escalation summaries.

    When automated resolution fails or is inappropriate, this agent
    creates detailed summaries for human agents including context,
    attempted steps, and recommendations.
    """

    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.2):
        """
        Initialize the escalation agent.

        Args:
            model: OpenAI model to use
            temperature: Model temperature (low for factual summaries)
        """
        self.llm = ChatOpenAI(model=model, temperature=temperature)

    def escalate(
        self,
        original_query: str,
        classification: Dict,
        attempted_steps: List[str],
        resolver_output: Dict = None,
        tool_output: Dict = None,
        customer_history: str = ""
    ) -> Dict:
        """
        Create escalation summary for human agent.

        Args:
            original_query: Original customer query
            classification: Ticket classification
            attempted_steps: List of steps attempted
            resolver_output: Output from resolver agent (if any)
            tool_output: Output from tool executor (if any)
            customer_history: Customer's interaction history

        Returns:
            Escalation dictionary with keys:
                - escalation_summary: Comprehensive summary
                - escalation_reason: Primary reason for escalation
                - priority: Priority level for human queue
                - recommended_actions: List of recommended next steps
                - customer_sentiment: Sentiment assessment
        """
        # Determine escalation reason
        escalation_reason = self._determine_escalation_reason(
            classification,
            resolver_output,
            tool_output
        )

        # Determine priority
        priority = self._determine_priority(classification, escalation_reason)

        # Generate comprehensive summary using LLM
        summary = self._generate_summary(
            original_query=original_query,
            classification=classification,
            attempted_steps=attempted_steps,
            resolver_output=resolver_output,
            tool_output=tool_output,
            customer_history=customer_history,
            escalation_reason=escalation_reason
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(
            classification,
            escalation_reason,
            tool_output
        )

        return {
            "escalation_summary": summary,
            "escalation_reason": escalation_reason,
            "priority": priority,
            "recommended_actions": recommendations,
            "customer_sentiment": classification.get('sentiment', 'neutral')
        }

    def _determine_escalation_reason(
        self,
        classification: Dict,
        resolver_output: Dict = None,
        tool_output: Dict = None
    ) -> str:
        """
        Determine primary reason for escalation.

        Args:
            classification: Ticket classification
            resolver_output: Resolver output
            tool_output: Tool execution output

        Returns:
            Escalation reason category
        """
        # Check for critical urgency
        if classification.get('urgency') == 'critical':
            return "critical_urgency"

        # Check if customer explicitly requested human
        required_expertise = classification.get('required_expertise', [])
        if 'human_agent' in required_expertise:
            return "customer_request"

        # Check if resolver had low confidence
        if resolver_output and resolver_output.get('confidence', 1.0) < 0.5:
            return "low_confidence"

        # Check if no knowledge found
        if resolver_output and not resolver_output.get('sources'):
            return "no_knowledge"

        # Check if tool execution failed
        if tool_output and not tool_output.get('success'):
            return "tool_failure"

        # Check for policy exception needs (refunds, etc.)
        if tool_output and 'refund' in tool_output.get('tools_used', []):
            refund_result = tool_output.get('tool_results', {}).get('process_refund', {})
            if refund_result.get('requires_approval'):
                return "policy_exception"

        # Default
        return "complex_issue"

    def _determine_priority(self, classification: Dict, escalation_reason: str) -> str:
        """
        Determine priority level for human queue.

        Args:
            classification: Ticket classification
            escalation_reason: Reason for escalation

        Returns:
            Priority level: "critical", "high", "medium", "low"
        """
        urgency = classification.get('urgency', 'medium')
        sentiment = classification.get('sentiment', 'neutral')

        # Critical if urgent or critical reason
        if urgency == 'critical' or escalation_reason == 'critical_urgency':
            return 'critical'

        # High if frustrated customer or urgent issue
        if sentiment == 'frustrated' or urgency == 'high':
            return 'high'

        # High if policy exception needed
        if escalation_reason == 'policy_exception':
            return 'high'

        # Medium for most other cases
        if urgency == 'medium':
            return 'medium'

        # Low for simple questions
        return 'low'

    def _generate_summary(
        self,
        original_query: str,
        classification: Dict,
        attempted_steps: List[str],
        resolver_output: Dict,
        tool_output: Dict,
        customer_history: str,
        escalation_reason: str
    ) -> str:
        """
        Generate comprehensive escalation summary using LLM.

        Args:
            original_query: Customer's original query
            classification: Classification details
            attempted_steps: Steps attempted
            resolver_output: Resolver results
            tool_output: Tool execution results
            customer_history: Customer history
            escalation_reason: Escalation reason

        Returns:
            Comprehensive escalation summary
        """
        system_prompt = """You are an escalation specialist creating summaries for human support agents.

Create a clear, concise escalation summary that includes:
1. Customer's issue in plain language
2. Why automated resolution failed
3. What has been attempted
4. Key customer context (history, sentiment)
5. What the human agent should focus on

Be professional, factual, and highlight important details."""

        # Format attempted steps
        steps_text = "\n".join([f"- {step}" for step in attempted_steps])

        # Format resolver output
        resolver_text = "Not attempted"
        if resolver_output:
            confidence = resolver_output.get('confidence', 0)
            sources = resolver_output.get('sources', [])
            resolver_text = f"Attempted with {confidence:.1%} confidence. Sources: {', '.join(sources) if sources else 'None found'}"

        # Format tool output
        tool_text = "No tools used"
        if tool_output:
            tools = tool_output.get('tools_used', [])
            summary = tool_output.get('summary', '')
            tool_text = f"Tools used: {', '.join(tools)}\n{summary}"

        user_prompt = f"""Create an escalation summary for this ticket:

**Original Customer Query:**
{original_query}

**Classification:**
- Type: {classification.get('ticket_type', 'unknown')}
- Urgency: {classification.get('urgency', 'unknown')}
- Sentiment: {classification.get('sentiment', 'unknown')}

**Escalation Reason:** {escalation_reason}

**Customer History:**
{customer_history if customer_history else "No previous interactions"}

**Attempted Steps:**
{steps_text if steps_text else "None"}

**Knowledge Base Search:**
{resolver_text}

**Database Operations:**
{tool_text}

Provide a clear, action-oriented escalation summary for the human agent."""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]

        response = self.llm.invoke(messages)
        return response.content

    def _generate_recommendations(
        self,
        classification: Dict,
        escalation_reason: str,
        tool_output: Dict
    ) -> List[str]:
        """
        Generate recommended actions for human agent.

        Args:
            classification: Ticket classification
            escalation_reason: Reason for escalation
            tool_output: Tool execution results

        Returns:
            List of recommended actions
        """
        recommendations = []

        ticket_type = classification.get('ticket_type', '')

        # Reason-specific recommendations
        if escalation_reason == 'critical_urgency':
            recommendations.append("Prioritize immediate response within 1 hour")
            recommendations.append("Consider phone outreach for faster resolution")

        elif escalation_reason == 'customer_request':
            recommendations.append("Customer specifically requested human assistance")
            recommendations.append("Acknowledge request and provide personalized attention")

        elif escalation_reason == 'no_knowledge':
            recommendations.append("No relevant knowledge base articles found")
            recommendations.append("Consider creating new article after resolution")

        elif escalation_reason == 'policy_exception':
            recommendations.append("Review refund/exception request with support lead")
            recommendations.append("Verify customer account standing before approval")

        elif escalation_reason == 'tool_failure':
            recommendations.append("Database operation failed - check system status")
            recommendations.append("May need technical team involvement")

        # Type-specific recommendations
        if ticket_type == 'billing':
            recommendations.append("Review billing history and transaction logs")
            recommendations.append("Check for payment processing issues")

        elif ticket_type == 'technical':
            recommendations.append("Collect device/app version information")
            recommendations.append("Consider escalating to technical support team")

        elif ticket_type == 'reservation':
            recommendations.append("Check event timing and availability")
            recommendations.append("Verify reservation policy compliance")

        # Sentiment-based recommendations
        if classification.get('sentiment') == 'frustrated':
            recommendations.append("⚠️ Customer is frustrated - empathize and acknowledge issue")
            recommendations.append("Consider goodwill gesture or compensation")

        # Default if no specific recommendations
        if not recommendations:
            recommendations.append("Review ticket details and customer history")
            recommendations.append("Respond with personalized solution")

        return recommendations
