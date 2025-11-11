"""
Tool Executor Agent

Executes database operations on CultPass external system.
Handles user lookups, subscription checks, reservation management, and refund requests.
"""

import os
import sys
import json
from typing import Dict, List
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from agentic.tools.cultpass_mcp_server import CultPassTools


class ToolExecutorAgent:
    """
    Agent responsible for executing tools/operations on external systems.

    Determines which tool(s) to use based on the ticket classification
    and executes database operations.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.1,
        db_path: str = "data/external/cultpass.db"
    ):
        """
        Initialize the tool executor agent.

        Args:
            model: OpenAI model to use
            temperature: Model temperature (low for deterministic tool selection)
            db_path: Path to CultPass database
        """
        self.llm = ChatOpenAI(model=model, temperature=temperature)
        self.cultpass_tools = CultPassTools(db_path=db_path)

    def execute(self, query: str, classification: Dict) -> Dict:
        """
        Determine and execute appropriate tools based on ticket.

        Args:
            query: User's question/request
            classification: Classification from classifier agent

        Returns:
            Execution result dictionary with keys:
                - tools_used: List of tool names used
                - tool_results: Dictionary of tool results
                - summary: Human-readable summary
                - success: Boolean indicating overall success
        """
        # Determine which tools to use
        tools_to_use = self._determine_tools(query, classification)

        if not tools_to_use:
            return {
                "tools_used": [],
                "tool_results": {},
                "summary": "No database operations needed for this query.",
                "success": True
            }

        # Execute tools
        tool_results = {}
        tools_executed = []

        for tool_name, tool_args in tools_to_use:
            try:
                result = self._execute_tool(tool_name, tool_args)
                tool_results[tool_name] = result
                tools_executed.append(tool_name)
            except Exception as e:
                tool_results[tool_name] = {
                    "success": False,
                    "error": str(e)
                }

        # Generate summary
        summary = self._generate_summary(tools_executed, tool_results)

        # Check overall success
        all_success = all(
            result.get('success', False)
            for result in tool_results.values()
        )

        return {
            "tools_used": tools_executed,
            "tool_results": tool_results,
            "summary": summary,
            "success": all_success
        }

    def _determine_tools(self, query: str, classification: Dict) -> List[tuple]:
        """
        Determine which tools to use based on query and classification.

        Args:
            query: User query
            classification: Ticket classification

        Returns:
            List of (tool_name, tool_args) tuples
        """
        query_lower = query.lower()
        entities = classification.get('entities', {})
        ticket_type = classification.get('ticket_type', '')

        tools_to_use = []

        # Extract user identifier
        user_id = entities.get('user_id')
        email = entities.get('email')

        # Determine if we need user lookup
        needs_user_lookup = (
            'account' in query_lower or
            'subscription' in query_lower or
            'my details' in query_lower or
            'profile' in query_lower or
            ticket_type in ['account', 'billing']
        )

        if needs_user_lookup and (user_id or email):
            tools_to_use.append((
                'lookup_user',
                {'email': email, 'user_id': user_id}
            ))

        # Check if we need subscription info
        needs_subscription_check = (
            'subscription' in query_lower or
            'plan' in query_lower or
            'quota' in query_lower or
            'monthly' in query_lower or
            ticket_type == 'billing'
        )

        if needs_subscription_check and user_id:
            tools_to_use.append((
                'check_subscription',
                {'user_id': user_id}
            ))

        # Check if we need reservation management
        needs_reservation = (
            'reservation' in query_lower or
            'booking' in query_lower or
            'cancel' in query_lower and 'reservation' in query_lower or
            'my events' in query_lower or
            ticket_type == 'reservation'
        )

        if needs_reservation and user_id:
            # Determine action
            if 'cancel' in query_lower and entities.get('reservation_id'):
                action = 'cancel'
            else:
                action = 'view'

            tools_to_use.append((
                'manage_reservation',
                {
                    'user_id': user_id,
                    'reservation_id': entities.get('reservation_id'),
                    'action': action
                }
            ))

        # Check if we need refund processing
        needs_refund = (
            'refund' in query_lower or
            'money back' in query_lower or
            'charge' in query_lower and 'wrong' in query_lower
        )

        if needs_refund and user_id:
            # Determine transaction type
            if 'subscription' in query_lower or 'monthly' in query_lower:
                transaction_type = 'subscription'
            elif 'event' in query_lower or 'premium' in query_lower:
                transaction_type = 'premium_event'
            else:
                transaction_type = 'subscription'  # Default

            tools_to_use.append((
                'process_refund',
                {
                    'user_id': user_id,
                    'transaction_type': transaction_type,
                    'reason': query[:200],  # First 200 chars as reason
                    'amount': entities.get('amount')
                }
            ))

        return tools_to_use

    def _execute_tool(self, tool_name: str, tool_args: Dict) -> Dict:
        """
        Execute a specific tool.

        Args:
            tool_name: Name of tool to execute
            tool_args: Arguments for the tool

        Returns:
            Tool execution result
        """
        tool_map = {
            'lookup_user': self.cultpass_tools.lookup_user,
            'check_subscription': self.cultpass_tools.check_subscription,
            'manage_reservation': self.cultpass_tools.manage_reservation,
            'process_refund': self.cultpass_tools.process_refund
        }

        if tool_name not in tool_map:
            raise ValueError(f"Unknown tool: {tool_name}")

        tool_func = tool_map[tool_name]

        # Remove None values from args
        filtered_args = {k: v for k, v in tool_args.items() if v is not None}

        return tool_func(**filtered_args)

    def _generate_summary(self, tools_used: List[str], tool_results: Dict) -> str:
        """
        Generate human-readable summary of tool execution.

        Args:
            tools_used: List of tools that were executed
            tool_results: Dictionary of tool results

        Returns:
            Summary string
        """
        if not tools_used:
            return "No tools were executed."

        summary_parts = [f"Executed {len(tools_used)} operation(s):"]

        for tool_name in tools_used:
            result = tool_results.get(tool_name, {})

            if result.get('success'):
                if tool_name == 'lookup_user':
                    user = result.get('user', {})
                    summary_parts.append(
                        f"✅ User lookup: Found {user.get('full_name', 'user')} "
                        f"({user.get('email', 'N/A')})"
                    )

                elif tool_name == 'check_subscription':
                    sub = result.get('subscription', {})
                    usage = result.get('usage', {})
                    summary_parts.append(
                        f"✅ Subscription check: {sub.get('tier', 'N/A')} tier, "
                        f"status: {sub.get('status', 'N/A')}, "
                        f"quota: {usage.get('reservations_this_month', 0)}/{sub.get('monthly_quota', 0)}"
                    )

                elif tool_name == 'manage_reservation':
                    action = result.get('action', '')
                    if action == 'view':
                        count = result.get('count', 0)
                        summary_parts.append(
                            f"✅ Reservation lookup: Found {count} active reservation(s)"
                        )
                    elif action == 'cancel':
                        message = result.get('message', '')
                        summary_parts.append(f"✅ Reservation cancellation: {message}")

                elif tool_name == 'process_refund':
                    eligible = result.get('eligible', False)
                    message = result.get('message', '')
                    summary_parts.append(
                        f"✅ Refund check: {'Eligible' if eligible else 'Not eligible'}. {message}"
                    )

            else:
                error = result.get('error', 'Unknown error')
                summary_parts.append(f"❌ {tool_name}: {error}")

        return "\n".join(summary_parts)
