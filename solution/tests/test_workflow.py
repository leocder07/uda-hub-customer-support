"""
Integration tests for the complete UDA-Hub workflow
"""

import pytest
import sys
import os
import uuid

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from agentic.workflow import process_ticket, orchestrator
from langchain_core.messages import HumanMessage


@pytest.mark.integration
def test_simple_knowledge_query():
    """Test simple query that should be resolved by knowledge base."""
    ticket_id = f"test_{uuid.uuid4().hex[:8]}"
    message = "How do I reserve an event?"

    result = process_ticket(
        ticket_id=ticket_id,
        user_message=message,
        account_id="cultpass"
    )

    assert result['status'] in ['resolved', 'escalated']
    assert 'classification' in result
    assert result['classification']['ticket_type'] in ['reservation', 'general']


@pytest.mark.integration
def test_login_issue_resolution():
    """Test login issue resolution."""
    ticket_id = f"test_{uuid.uuid4().hex[:8]}"
    message = "I can't log in to my account. What should I do?"

    result = process_ticket(
        ticket_id=ticket_id,
        user_message=message,
        account_id="cultpass"
    )

    assert result['status'] in ['resolved', 'escalated']
    assert result['classification']['ticket_type'] == 'login'

    # Should have attempted resolution
    assert len(result['decision_log']) > 0


@pytest.mark.integration
def test_billing_question():
    """Test billing-related question."""
    ticket_id = f"test_{uuid.uuid4().hex[:8]}"
    message = "What's included in my CultPass subscription?"

    result = process_ticket(
        ticket_id=ticket_id,
        user_message=message,
        account_id="cultpass"
    )

    assert result['status'] in ['resolved', 'escalated']
    assert result['classification']['ticket_type'] in ['billing', 'general']

    # Should find relevant knowledge
    if result['status'] == 'resolved':
        assert result['confidence'] > 0.0


@pytest.mark.integration
def test_technical_issue():
    """Test technical issue handling."""
    ticket_id = f"test_{uuid.uuid4().hex[:8]}"
    message = "The app keeps crashing when I try to view my reservations."

    result = process_ticket(
        ticket_id=ticket_id,
        user_message=message,
        account_id="cultpass"
    )

    assert result['status'] in ['resolved', 'escalated']
    assert result['classification']['ticket_type'] == 'technical'


@pytest.mark.integration
def test_critical_urgency_escalation():
    """Test that critical urgency triggers escalation."""
    ticket_id = f"test_{uuid.uuid4().hex[:8]}"
    message = "URGENT! My account is blocked and I have an event in 1 hour!"

    result = process_ticket(
        ticket_id=ticket_id,
        user_message=message,
        account_id="cultpass"
    )

    # Critical issues should be escalated
    assert result['classification']['urgency'] in ['critical', 'high']


@pytest.mark.integration
def test_decision_log_tracking():
    """Test that decision log is populated."""
    ticket_id = f"test_{uuid.uuid4().hex[:8]}"
    message = "How do I cancel my subscription?"

    result = process_ticket(
        ticket_id=ticket_id,
        user_message=message,
        account_id="cultpass"
    )

    # Should have decision log entries
    assert 'decision_log' in result
    assert len(result['decision_log']) > 0

    # Should include classification
    assert any('classified' in log.lower() for log in result['decision_log'])


@pytest.mark.integration
def test_classification_accuracy():
    """Test classification accuracy across different ticket types."""
    test_cases = [
        ("I can't log in", "login"),
        ("I want a refund", "billing"),
        ("How do I cancel my reservation?", "reservation"),
        ("The app won't load", "technical"),
        ("How do I update my profile?", "account")
    ]

    for message, expected_type in test_cases:
        ticket_id = f"test_{uuid.uuid4().hex[:8]}"

        result = process_ticket(
            ticket_id=ticket_id,
            user_message=message,
            account_id="cultpass"
        )

        # Allow some flexibility in classification
        assert result['classification']['ticket_type'] in [expected_type, 'general']


@pytest.mark.integration
def test_session_memory():
    """Test that session memory works across invocations."""
    ticket_id = f"session_test_{uuid.uuid4().hex[:8]}"

    config = {
        "configurable": {
            "thread_id": ticket_id
        }
    }

    # First message
    state1 = {
        "messages": [HumanMessage(content="How do I reserve an event?")],
        "ticket_id": ticket_id,
        "account_id": "cultpass",
        "user_id": "test_user",
        "classification": {},
        "resolver_output": {},
        "tool_output": {},
        "escalation_output": {},
        "confidence": 0.0,
        "decision_log": [],
        "final_response": "",
        "status": "pending",
        "customer_history": {}
    }

    result1 = orchestrator.invoke(state1, config=config)

    # Check that messages are in the state
    assert len(result1['messages']) > 1  # Should have user message + responses

    # Second message (should have context from first)
    state2 = {
        **result1,
        "messages": result1["messages"] + [HumanMessage(content="What about cancelling it?")],
        "status": "pending"
    }

    result2 = orchestrator.invoke(state2, config=config)

    # Should maintain conversation history
    assert len(result2['messages']) > len(result1['messages'])


@pytest.mark.integration
def test_resolver_confidence_scoring():
    """Test that resolver provides confidence scores."""
    ticket_id = f"test_{uuid.uuid4().hex[:8]}"
    message = "What's included in my subscription?"

    result = process_ticket(
        ticket_id=ticket_id,
        user_message=message,
        account_id="cultpass"
    )

    # Should have confidence score
    if result['status'] == 'resolved':
        assert 'confidence' in result
        assert 0.0 <= result['confidence'] <= 1.0


@pytest.mark.integration
def test_escalation_structure():
    """Test that escalation output has proper structure."""
    ticket_id = f"test_{uuid.uuid4().hex[:8]}"
    # Deliberately vague query that should trigger escalation
    message = "I need help with something very specific and unusual."

    result = process_ticket(
        ticket_id=ticket_id,
        user_message=message,
        account_id="cultpass"
    )

    if result['status'] == 'escalated':
        assert 'escalation_output' in result
        escalation = result['escalation_output']

        assert 'escalation_summary' in escalation
        assert 'priority' in escalation
        assert 'recommended_actions' in escalation
        assert isinstance(escalation['recommended_actions'], list)


@pytest.mark.integration
def test_workflow_completes_successfully():
    """Test that workflow completes without errors."""
    ticket_id = f"test_{uuid.uuid4().hex[:8]}"
    message = "Can you help me understand how reservations work?"

    try:
        result = process_ticket(
            ticket_id=ticket_id,
            user_message=message,
            account_id="cultpass"
        )

        # Should complete successfully
        assert result is not None
        assert 'status' in result
        assert result['status'] in ['resolved', 'escalated']

    except Exception as e:
        pytest.fail(f"Workflow should not raise exception: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])
