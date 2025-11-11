"""
Supervisor Agent

Orchestrates the entire workflow and makes routing decisions based on
ticket classification and agent outputs.
"""

from typing import Dict, List, Literal


class SupervisorAgent:
    """
    Supervisor agent responsible for workflow orchestration.

    Makes routing decisions based on classification and coordinates
    execution across specialized agents.
    """

    def __init__(self):
        """Initialize the supervisor agent."""
        pass

    def route_initial(self, classification: Dict, customer_history: Dict = None) -> Literal[
        "tool_executor", "resolver", "escalation"
    ]:
        """
        Make initial routing decision based on classification.

        Args:
            classification: Ticket classification from classifier
            customer_history: Optional customer history data

        Returns:
            Next agent to route to: "tool_executor", "resolver", or "escalation"
        """
        urgency = classification.get('urgency', 'medium')
        sentiment = classification.get('sentiment', 'neutral')
        required_expertise = classification.get('required_expertise', [])
        ticket_type = classification.get('ticket_type', 'general')

        # Immediate escalation conditions
        if urgency == 'critical' and 'human_agent' in required_expertise:
            return "escalation"

        # Check for frustrated customer with repeated issues
        if customer_history:
            repeated_issue_count = customer_history.get('repeated_issue_count', 0)
            if sentiment == 'frustrated' and repeated_issue_count >= 2:
                return "escalation"

        # Route to tool executor if database operations needed
        if 'database_operations' in required_expertise:
            return "tool_executor"

        # Default to resolver for knowledge base queries
        return "resolver"

    def should_escalate(
        self,
        classification: Dict,
        resolver_output: Dict = None,
        tool_output: Dict = None
    ) -> bool:
        """
        Determine if ticket should be escalated after resolution attempt.

        Args:
            classification: Ticket classification
            resolver_output: Output from resolver agent
            tool_output: Output from tool executor

        Returns:
            True if escalation needed
        """
        # Check resolver confidence
        if resolver_output:
            confidence = resolver_output.get('confidence', 1.0)
            needs_escalation = resolver_output.get('needs_escalation', False)

            if confidence < 0.5 or needs_escalation:
                return True

        # Check tool execution failure
        if tool_output and not tool_output.get('success', True):
            return True

        # Check for refund approval requirement
        if tool_output:
            tool_results = tool_output.get('tool_results', {})
            if 'process_refund' in tool_results:
                refund_result = tool_results['process_refund']
                if refund_result.get('requires_approval', False):
                    return True

        # Check classification flags
        required_expertise = classification.get('required_expertise', [])
        if 'human_agent' in required_expertise:
            return True

        return False

    def make_decision(
        self,
        classification: Dict,
        resolver_output: Dict = None,
        tool_output: Dict = None,
        customer_history: Dict = None
    ) -> Dict:
        """
        Make final decision on ticket routing and status.

        Args:
            classification: Ticket classification
            resolver_output: Resolver agent output
            tool_output: Tool executor output
            customer_history: Customer history data

        Returns:
            Decision dictionary with keys:
                - next_agent: Next agent to route to (or "end")
                - status: Ticket status ("resolved", "escalated", "in_progress")
                - decision_log: List of decision reasons
                - final_response: Response to send to customer (if resolved)
        """
        decision_log = []

        # Check if escalation is needed
        should_escalate = self.should_escalate(classification, resolver_output, tool_output)

        if should_escalate:
            decision_log.append("Escalation needed based on resolution confidence or requirements")
            return {
                "next_agent": "escalation",
                "status": "escalating",
                "decision_log": decision_log,
                "final_response": None
            }

        # If resolver provided good response, mark as resolved
        if resolver_output and resolver_output.get('confidence', 0) >= 0.5:
            decision_log.append(f"Resolved with {resolver_output['confidence']:.1%} confidence")

            # Combine resolver response with tool results if available
            final_response = resolver_output.get('resolution', '')

            if tool_output and tool_output.get('summary'):
                final_response = (
                    f"{tool_output['summary']}\n\n"
                    f"{final_response}"
                )

            return {
                "next_agent": "end",
                "status": "resolved",
                "decision_log": decision_log,
                "final_response": final_response
            }

        # Fallback to escalation if no clear resolution
        decision_log.append("No clear resolution path - escalating to human")
        return {
            "next_agent": "escalation",
            "status": "escalating",
            "decision_log": decision_log,
            "final_response": None
        }

    def get_decision_summary(self, decision_log: List[str]) -> str:
        """
        Format decision log into readable summary.

        Args:
            decision_log: List of decision log entries

        Returns:
            Formatted summary string
        """
        if not decision_log:
            return "No decisions recorded."

        summary = "Decision Path:\n"
        for i, entry in enumerate(decision_log, 1):
            summary += f"{i}. {entry}\n"

        return summary
