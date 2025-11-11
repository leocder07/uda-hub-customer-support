import os
import json
import pickle
from datetime import datetime
from sqlalchemy import create_engine, Engine, Column, String, Text, Integer, Float, Boolean, DateTime, BLOB
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.ext.declarative import declarative_base as legacy_declarative_base
from contextlib import contextmanager
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph.state import CompiledStateGraph

Base = legacy_declarative_base()


def reset_db(db_path: str, echo: bool = False):
    """Drops the existing database file and recreates all tables."""

    # Remove the file if it exists
    if os.path.exists(db_path):
        os.remove(db_path)
        print(f"✅ Removed existing {db_path}")

    # Create a new engine and recreate tables
    engine = create_engine(f"sqlite:///{db_path}", echo=echo)
    Base.metadata.create_all(engine)
    print(f"✅ Recreated {db_path} with fresh schema")


@contextmanager
def get_session(engine: Engine):
    """Context manager for database sessions with automatic commit/rollback."""
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        print(f"❌ Session error: {e}")
        raise
    finally:
        session.close()


def model_to_dict(instance):
    """Convert a SQLAlchemy model instance to a dictionary."""
    return {
        column.name: getattr(instance, column.name)
        for column in instance.__table__.columns
    }


def chat_interface(agent: CompiledStateGraph, ticket_id: str):
    """
    Simple chat interface for interacting with the agent system.

    Args:
        agent: Compiled LangGraph agent workflow
        ticket_id: Unique ticket ID for session management (thread_id)
    """
    print("="*60)
    print("UDA-Hub Customer Support Agent")
    print("="*60)
    print("Type 'quit', 'exit', or 'q' to end the conversation")
    print("="*60)

    is_first_iteration = True

    while True:
        user_input = input("\nUser: ").strip()

        if user_input.lower() in ["quit", "exit", "q"]:
            print("\nAssistant: Thank you for contacting CultPass support. Have a great day!")
            break

        if not user_input:
            continue

        # Prepare messages
        messages = [HumanMessage(content=user_input)]

        # Configuration with thread_id for session management
        config = {
            "configurable": {
                "thread_id": ticket_id,
            }
        }

        try:
            # Invoke agent
            result = agent.invoke(
                input={"messages": messages},
                config=config
            )

            # Extract final response
            final_message = result["messages"][-1]
            print(f"\nAssistant: {final_message.content}")

            # Show additional info if available
            if "status" in result:
                status = result["status"]
                if status == "escalated":
                    print("\n⚠️  This ticket has been escalated to a human agent.")
                elif status == "resolved":
                    print("\n✅ Ticket marked as resolved.")

        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Our system encountered an issue. A human agent will assist you shortly.")

        is_first_iteration = False


def log_decision(
    log_file: str,
    ticket_id: str,
    agent_name: str,
    input_data: dict,
    output_data: dict,
    execution_time: float
):
    """
    Log agent decisions to a JSONL file for observability.

    Args:
        log_file: Path to log file
        ticket_id: Ticket ID
        agent_name: Name of the agent
        input_data: Input to the agent
        output_data: Output from the agent
        execution_time: Execution time in seconds
    """
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "ticket_id": ticket_id,
        "agent_name": agent_name,
        "input": input_data,
        "output": output_data,
        "execution_time": execution_time
    }

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # Append to log file
    with open(log_file, 'a') as f:
        f.write(json.dumps(log_entry) + '\n')


# Long-term memory database models
class CustomerInteraction(Base):
    """Store historical customer interactions for long-term memory."""
    __tablename__ = 'customer_interactions'

    interaction_id = Column(String, primary_key=True)
    account_id = Column(String, nullable=False)
    user_id = Column(String, nullable=False)
    ticket_id = Column(String, nullable=False)
    interaction_date = Column(DateTime, nullable=False)
    issue_type = Column(String, nullable=False)
    resolution_summary = Column(Text, nullable=False)
    agent_notes = Column(Text)
    sentiment = Column(String)
    was_escalated = Column(Boolean)
    resolution_time_seconds = Column(Integer)
    customer_satisfaction = Column(Integer)  # 1-5 scale
    embedding = Column(BLOB)  # Pickled numpy array
    created_at = Column(DateTime, default=datetime.now)


class CustomerPreference(Base):
    """Store learned customer preferences."""
    __tablename__ = 'customer_preferences'

    preference_id = Column(String, primary_key=True)
    user_id = Column(String, nullable=False)
    preference_type = Column(String, nullable=False)
    preference_value = Column(Text, nullable=False)
    confidence = Column(Float)
    last_updated = Column(DateTime, default=datetime.now)


def get_customer_history(engine: Engine, user_id: str, limit: int = 5) -> list[dict]:
    """
    Retrieve customer's recent interaction history.

    Args:
        engine: SQLAlchemy engine
        user_id: User ID
        limit: Maximum number of interactions to return

    Returns:
        List of interaction dictionaries
    """
    with get_session(engine) as session:
        interactions = session.query(CustomerInteraction)\
            .filter_by(user_id=user_id)\
            .order_by(CustomerInteraction.interaction_date.desc())\
            .limit(limit)\
            .all()

        return [model_to_dict(i) for i in interactions]


def save_interaction(
    engine: Engine,
    ticket_id: str,
    account_id: str,
    user_id: str,
    issue_type: str,
    resolution_summary: str,
    sentiment: str,
    was_escalated: bool,
    resolution_time_seconds: int,
    agent_notes: str = None,
    embedding: bytes = None
):
    """
    Save a customer interaction to long-term memory.

    Args:
        engine: SQLAlchemy engine
        ticket_id: Ticket ID
        account_id: Account ID
        user_id: User ID
        issue_type: Type of issue
        resolution_summary: Summary of resolution
        sentiment: Customer sentiment
        was_escalated: Whether ticket was escalated
        resolution_time_seconds: Time to resolve
        agent_notes: Optional agent notes
        embedding: Optional pickled embedding vector
    """
    import uuid

    with get_session(engine) as session:
        interaction = CustomerInteraction(
            interaction_id=str(uuid.uuid4()),
            account_id=account_id,
            user_id=user_id,
            ticket_id=ticket_id,
            interaction_date=datetime.now(),
            issue_type=issue_type,
            resolution_summary=resolution_summary,
            agent_notes=agent_notes,
            sentiment=sentiment,
            was_escalated=was_escalated,
            resolution_time_seconds=resolution_time_seconds,
            embedding=embedding
        )
        session.add(interaction)


def check_repeated_issue(engine: Engine, user_id: str, issue_type: str, days: int = 30) -> int:
    """
    Check if user has reported the same issue multiple times recently.

    Args:
        engine: SQLAlchemy engine
        user_id: User ID
        issue_type: Type of issue
        days: Number of days to look back

    Returns:
        Count of repeated issues
    """
    from datetime import timedelta

    cutoff_date = datetime.now() - timedelta(days=days)

    with get_session(engine) as session:
        count = session.query(CustomerInteraction)\
            .filter_by(user_id=user_id, issue_type=issue_type)\
            .filter(CustomerInteraction.interaction_date >= cutoff_date)\
            .count()

        return count


def get_sentiment_trend(history: list[dict]) -> str:
    """
    Analyze sentiment trend from customer history.

    Args:
        history: List of interaction dictionaries

    Returns:
        Sentiment trend description
    """
    if not history:
        return "neutral"

    sentiments = [h.get('sentiment', 'neutral') for h in history]

    # Simple trend analysis
    negative_count = sentiments.count('negative') + sentiments.count('frustrated')
    positive_count = sentiments.count('positive')

    if negative_count >= len(sentiments) * 0.6:
        return "declining"
    elif positive_count >= len(sentiments) * 0.6:
        return "positive"
    else:
        return "neutral"


def format_history_summary(history: list[dict]) -> str:
    """
    Format customer history into a readable summary for agent context.

    Args:
        history: List of interaction dictionaries

    Returns:
        Formatted summary string
    """
    if not history:
        return "No previous interactions found."

    summary_lines = [
        f"📊 Customer History ({len(history)} recent interactions):",
        ""
    ]

    for i, interaction in enumerate(history[:5], 1):
        date = interaction.get('interaction_date', 'N/A')
        issue_type = interaction.get('issue_type', 'Unknown')
        was_escalated = interaction.get('was_escalated', False)
        sentiment = interaction.get('sentiment', 'neutral')

        escalated_icon = "⚠️" if was_escalated else "✅"
        summary_lines.append(
            f"{i}. {escalated_icon} {issue_type.upper()} - {date} (Sentiment: {sentiment})"
        )

    # Add trend analysis
    trend = get_sentiment_trend(history)
    summary_lines.append("")
    summary_lines.append(f"Sentiment Trend: {trend.upper()}")

    # Check for repeated issues
    issue_types = [h.get('issue_type') for h in history]
    if len(issue_types) != len(set(issue_types)):
        summary_lines.append("⚠️ ALERT: Repeated issues detected")

    return "\n".join(summary_lines)


def ensure_db_initialized(db_path: str):
    """
    Ensure database exists and has required tables.
    Creates database if it doesn't exist.

    Args:
        db_path: Path to database file
    """
    if not os.path.exists(db_path):
        print(f"⚠️  Database not found at {db_path}")
        print(f"Creating new database...")
        engine = create_engine(f"sqlite:///{db_path}", echo=False)
        Base.metadata.create_all(engine)
        print(f"✅ Database created at {db_path}")
        return engine

    return create_engine(f"sqlite:///{db_path}", echo=False)


def get_absolute_path(relative_path: str) -> str:
    """
    Convert relative path to absolute path from solution/ directory.

    Args:
        relative_path: Relative path from solution/ directory

    Returns:
        Absolute path
    """
    solution_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(solution_dir, relative_path)
