"""
Unit tests for Classifier Agent
"""

import pytest
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from agentic.agents.classifier import ClassifierAgent


@pytest.fixture
def classifier():
    """Create classifier agent instance."""
    return ClassifierAgent()


def test_classifier_login_issue(classifier):
    """Test classification of login issue."""
    message = "I can't log in to my account. I forgot my password."
    result = classifier.classify(message)

    assert result['ticket_type'] == 'login'
    assert result['urgency'] in ['low', 'medium', 'high']
    assert 'required_expertise' in result
    assert 'sentiment' in result


def test_classifier_billing_issue(classifier):
    """Test classification of billing issue."""
    message = "I was charged twice for my subscription this month!"
    result = classifier.classify(message)

    assert result['ticket_type'] == 'billing'
    assert result['sentiment'] in ['negative', 'frustrated']
    assert 'urgency' in result


def test_classifier_reservation_issue(classifier):
    """Test classification of reservation issue."""
    message = "How do I cancel my reservation for tomorrow's event?"
    result = classifier.classify(message)

    assert result['ticket_type'] == 'reservation'
    assert isinstance(result['keywords'], list)


def test_classifier_technical_issue(classifier):
    """Test classification of technical issue."""
    message = "The app keeps crashing when I try to open it."
    result = classifier.classify(message)

    assert result['ticket_type'] == 'technical'
    assert 'required_expertise' in result


def test_classifier_critical_urgency(classifier):
    """Test classification of critical urgency."""
    message = "URGENT: My account is blocked and I can't access my reservations!"
    result = classifier.classify(message)

    assert result['urgency'] in ['critical', 'high']
    assert result['sentiment'] in ['negative', 'frustrated']


def test_classifier_general_question(classifier):
    """Test classification of general question."""
    message = "What events are available this weekend?"
    result = classifier.classify(message)

    assert result['ticket_type'] in ['general', 'reservation']
    assert result['sentiment'] in ['neutral', 'positive']


def test_classifier_output_structure(classifier):
    """Test that classifier output has required structure."""
    message = "I need help with my account."
    result = classifier.classify(message)

    required_keys = ['ticket_type', 'urgency', 'sentiment', 'entities',
                      'required_expertise', 'keywords']

    for key in required_keys:
        assert key in result, f"Missing required key: {key}"

    assert isinstance(result['entities'], dict)
    assert isinstance(result['required_expertise'], list)
    assert isinstance(result['keywords'], list)


def test_classifier_with_customer_history(classifier):
    """Test classifier with customer history context."""
    message = "I'm having the same login problem again."
    history = "Previous tickets: login issue (2 weeks ago), billing question (1 month ago)"

    result = classifier.classify(message, history)

    assert result['ticket_type'] == 'login'
    # Should recognize this as potentially repeated issue
    assert result is not None


def test_classifier_fallback(classifier):
    """Test classifier fallback with unparseable response."""
    message = "Random text that might confuse the classifier"

    try:
        result = classifier.classify(message)
        # Should return valid classification even if confused
        assert 'ticket_type' in result
        assert 'urgency' in result
    except Exception as e:
        pytest.fail(f"Classifier should not raise exception: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
