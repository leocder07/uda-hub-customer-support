"""
UDA-Hub Workflow Orchestration

LangGraph-based multi-agent workflow implementing the Supervisor Pattern.
Orchestrates ticket processing through classification, resolution, tool execution,
and escalation agents.
"""

import os
import sys
from typing import TypedDict, Annotated, Literal
from datetime import datetime
import operator

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from sqlalchemy import create_engine

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

from agents.classifier import ClassifierAgent
from agents.resolver import ResolverAgent
from agents.tool_executor import ToolExecutorAgent
from agents.escalation import EscalationAgent
from agents.supervisor import SupervisorAgent

# Import utilities
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import (
    get_customer_history,
    check_repeated_issue,
    format_history_summary,
    save_interaction,
    get_absolute_path
)


# Define the state
class AgentState(TypedDict):
    """State passed between agents in the workflow."""
    messages: Annotated[list[BaseMessage], operator.add]
    ticket_id: str
    account_id: str
    user_id: str
    classification: dict
    resolver_output: dict
    tool_output: dict
    escalation_output: dict
    confidence: float
    decision_log: list[str]
    final_response: str
    status: str  # "pending", "resolved", "escalated"
    customer_history: dict


# Initialize agents
classifier_agent = ClassifierAgent()
resolver_agent = ResolverAgent()
tool_executor_agent = ToolExecutorAgent()
escalation_agent = EscalationAgent()
supervisor_agent = SupervisorAgent()


def classify_node(state: AgentState) -> AgentState:
    """
    Classification node - analyzes ticket and extracts information.
    """
    print("🔍 Classifier Agent: Analyzing ticket...")

    # Get the last human message
    user_message = None
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            user_message = msg.content
            break

    if not user_message:
        user_message = "No message provided"

    # Get customer history if available
    customer_history_text = ""
    if state.get("customer_history"):
        customer_history_text = format_history_summary(
            state["customer_history"].get("interactions", [])
        )

    # Classify the ticket
    classification = classifier_agent.classify(user_message, customer_history_text)

    print(f"   ✓ Type: {classification['ticket_type']}, "
          f"Urgency: {classification['urgency']}, "
          f"Sentiment: {classification['sentiment']}")

    # Update decision log
    decision_log = state.get("decision_log", [])
    decision_log.append(f"Classified as {classification['ticket_type']} with {classification['urgency']} urgency")

    return {
        **state,
        "classification": classification,
        "decision_log": decision_log
    }


def supervisor_route_node(state: AgentState) -> Literal["tool_executor", "resolver", "escalation"]:
    """
    Supervisor routing node - decides which agent to invoke next.
    """
    print("🎯 Supervisor: Making routing decision...")

    classification = state.get("classification", {})
    customer_history = state.get("customer_history", {})

    # Make routing decision
    next_agent = supervisor_agent.route_initial(classification, customer_history)

    print(f"   → Routing to: {next_agent}")

    # Update decision log
    decision_log = state.get("decision_log", [])
    decision_log.append(f"Routed to {next_agent}")

    state["decision_log"] = decision_log

    return next_agent


def resolver_node(state: AgentState) -> AgentState:
    """
    Resolver node - uses RAG to find and generate responses.
    """
    print("💡 Resolver Agent: Searching knowledge base...")

    # Get user message
    user_message = None
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            user_message = msg.content
            break

    classification = state.get("classification", {})
    account_id = state.get("account_id", "cultpass")
    tool_output = state.get("tool_output")

    # Attempt resolution
    resolver_output = resolver_agent.resolve(
        query=user_message,
        classification=classification,
        account_id=account_id,
        tool_results=tool_output
    )

    print(f"   ✓ Confidence: {resolver_output['confidence']:.1%}, "
          f"Sources: {len(resolver_output['sources'])}")

    # Update decision log
    decision_log = state.get("decision_log", [])
    decision_log.append(
        f"Resolver found {len(resolver_output['sources'])} articles "
        f"with {resolver_output['confidence']:.1%} confidence"
    )

    return {
        **state,
        "resolver_output": resolver_output,
        "confidence": resolver_output.get("confidence", 0.0),
        "decision_log": decision_log
    }


def tool_executor_node(state: AgentState) -> AgentState:
    """
    Tool executor node - executes database operations.
    """
    print("🔧 Tool Executor Agent: Executing database operations...")

    # Get user message
    user_message = None
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            user_message = msg.content
            break

    classification = state.get("classification", {})

    # Execute tools
    tool_output = tool_executor_agent.execute(user_message, classification)

    print(f"   ✓ Executed {len(tool_output.get('tools_used', []))} tool(s)")

    # Update decision log
    decision_log = state.get("decision_log", [])
    decision_log.append(f"Executed tools: {', '.join(tool_output.get('tools_used', ['none']))}")

    return {
        **state,
        "tool_output": tool_output,
        "decision_log": decision_log
    }


def escalation_node(state: AgentState) -> AgentState:
    """
    Escalation node - creates escalation summary for human agent.
    """
    print("⚠️  Escalation Agent: Creating escalation summary...")

    # Get user message
    user_message = None
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            user_message = msg.content
            break

    classification = state.get("classification", {})
    resolver_output = state.get("resolver_output")
    tool_output = state.get("tool_output")
    decision_log = state.get("decision_log", [])

    # Get customer history text
    customer_history_text = ""
    if state.get("customer_history"):
        customer_history_text = format_history_summary(
            state["customer_history"].get("interactions", [])
        )

    # Create escalation
    escalation_output = escalation_agent.escalate(
        original_query=user_message,
        classification=classification,
        attempted_steps=decision_log,
        resolver_output=resolver_output,
        tool_output=tool_output,
        customer_history=customer_history_text
    )

    print(f"   ✓ Priority: {escalation_output['priority']}, "
          f"Reason: {escalation_output['escalation_reason']}")

    # Create escalation message
    escalation_message = (
        f"This ticket has been escalated to a human agent.\n\n"
        f"**Priority**: {escalation_output['priority'].upper()}\n"
        f"**Reason**: {escalation_output['escalation_reason']}\n\n"
        f"{escalation_output['escalation_summary']}\n\n"
        f"A support specialist will review your case shortly."
    )

    # Add AI message to conversation
    messages = state.get("messages", [])
    messages.append(AIMessage(content=escalation_message))

    # Update decision log
    decision_log.append(f"Escalated with {escalation_output['priority']} priority")

    return {
        **state,
        "escalation_output": escalation_output,
        "final_response": escalation_message,
        "status": "escalated",
        "messages": messages,
        "decision_log": decision_log
    }


def supervisor_decide_node(state: AgentState) -> Literal["escalation", "end"]:
    """
    Supervisor decision node - decides if escalation needed or can resolve.
    """
    print("🎯 Supervisor: Making final decision...")

    classification = state.get("classification", {})
    resolver_output = state.get("resolver_output")
    tool_output = state.get("tool_output")
    customer_history = state.get("customer_history")

    # Make decision
    decision = supervisor_agent.make_decision(
        classification=classification,
        resolver_output=resolver_output,
        tool_output=tool_output,
        customer_history=customer_history
    )

    print(f"   → Decision: {decision['status']}")

    # Update state with decision
    if decision["status"] == "resolved" and decision.get("final_response"):
        messages = state.get("messages", [])
        messages.append(AIMessage(content=decision["final_response"]))

        state["messages"] = messages
        state["final_response"] = decision["final_response"]

    state["status"] = decision["status"]
    state["decision_log"] = decision.get("decision_log", state.get("decision_log", []))

    return decision["next_agent"]


def after_tool_route(state: AgentState) -> Literal["resolver", "escalation"]:
    """
    Route after tool execution - to resolver for response formatting or escalation.
    """
    tool_output = state.get("tool_output", {})

    # If tools failed, escalate
    if not tool_output.get("success", False):
        print("   → Tools failed, routing to escalation")
        return "escalation"

    # Otherwise route to resolver to format response
    print("   → Tools succeeded, routing to resolver for response")
    return "resolver"


# Build the workflow graph
def create_workflow():
    """Create and compile the LangGraph workflow."""

    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("classify", classify_node)
    workflow.add_node("supervisor_route", lambda state: state)  # Dummy node for routing
    workflow.add_node("resolver", resolver_node)
    workflow.add_node("tool_executor", tool_executor_node)
    workflow.add_node("escalation", escalation_node)
    workflow.add_node("supervisor_decide", lambda state: state)  # Dummy node for decision

    # Set entry point
    workflow.set_entry_point("classify")

    # Add edges
    workflow.add_edge("classify", "supervisor_route")

    # Conditional routing from supervisor
    workflow.add_conditional_edges(
        "supervisor_route",
        supervisor_route_node,
        {
            "tool_executor": "tool_executor",
            "resolver": "resolver",
            "escalation": "escalation"
        }
    )

    # After tool executor, route to resolver or escalation
    workflow.add_conditional_edges(
        "tool_executor",
        after_tool_route,
        {
            "resolver": "resolver",
            "escalation": "escalation"
        }
    )

    # After resolver, supervisor decides
    workflow.add_edge("resolver", "supervisor_decide")

    # Conditional routing from supervisor decision
    workflow.add_conditional_edges(
        "supervisor_decide",
        supervisor_decide_node,
        {
            "escalation": "escalation",
            "end": END
        }
    )

    # Escalation ends the workflow
    workflow.add_edge("escalation", END)

    # Compile with memory saver for session management
    checkpointer = MemorySaver()
    compiled_workflow = workflow.compile(checkpointer=checkpointer)

    return compiled_workflow


# Create the orchestrator (main export)
orchestrator = create_workflow()


# Helper function to process a ticket with full context
def process_ticket(
    ticket_id: str,
    user_message: str,
    account_id: str = "cultpass",
    user_id: str = None,
    db_path: str = "data/core/udahub.db"
):
    """
    Process a support ticket through the workflow.

    Args:
        ticket_id: Unique ticket ID
        user_message: Customer's message
        account_id: Account ID (default: "cultpass")
        user_id: Optional user ID for history lookup
        db_path: Path to UDA-Hub database

    Returns:
        Final state after processing
    """
    # Load customer history if user_id provided
    customer_history = {}
    if user_id:
        db_absolute_path = get_absolute_path(db_path)
        if os.path.exists(db_absolute_path):
            engine = create_engine(f"sqlite:///{db_absolute_path}", echo=False)
            interactions = get_customer_history(engine, user_id, limit=5)
            repeated_count = check_repeated_issue(engine, user_id, "unknown", days=30)

            customer_history = {
                "interactions": interactions,
                "repeated_issue_count": repeated_count
            }

    # Initial state
    initial_state = {
        "messages": [HumanMessage(content=user_message)],
        "ticket_id": ticket_id,
        "account_id": account_id,
        "user_id": user_id or "unknown",
        "classification": {},
        "resolver_output": {},
        "tool_output": {},
        "escalation_output": {},
        "confidence": 0.0,
        "decision_log": [],
        "final_response": "",
        "status": "pending",
        "customer_history": customer_history
    }

    # Configuration for session management
    config = {
        "configurable": {
            "thread_id": ticket_id
        }
    }

    # Invoke workflow
    start_time = datetime.now()
    result = orchestrator.invoke(initial_state, config=config)
    end_time = datetime.now()

    # Save interaction to long-term memory
    if user_id and result.get("status") in ["resolved", "escalated"]:
        try:
            db_absolute_path = get_absolute_path(db_path)
            if os.path.exists(db_absolute_path):
                engine = create_engine(f"sqlite:///{db_absolute_path}", echo=False)

                save_interaction(
                    engine=engine,
                    ticket_id=ticket_id,
                    account_id=account_id,
                    user_id=user_id,
                    issue_type=result.get("classification", {}).get("ticket_type", "unknown"),
                    resolution_summary=result.get("final_response", "")[:500],
                    sentiment=result.get("classification", {}).get("sentiment", "neutral"),
                    was_escalated=(result.get("status") == "escalated"),
                    resolution_time_seconds=int((end_time - start_time).total_seconds())
                )
        except Exception as e:
            print(f"⚠️  Failed to save interaction: {e}")

    return result
