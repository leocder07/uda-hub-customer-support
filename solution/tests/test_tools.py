"""
Unit tests for CultPass Tools
"""

import pytest
import sys
import os
import tempfile
import shutil
from sqlalchemy import create_engine

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from agentic.tools.cultpass_mcp_server import CultPassTools
from data.models import cultpass
from utils import get_session


@pytest.fixture
def test_db():
    """Create a temporary test database."""
    # Create temp directory
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, "test_cultpass.db")

    # Create engine and tables
    engine = create_engine(f"sqlite:///{db_path}", echo=False)
    cultpass.Base.metadata.create_all(engine)

    # Add test data
    with get_session(engine) as session:
        # Create test user
        user = cultpass.User(
            user_id="test123",
            full_name="Test User",
            email="test@example.com",
            is_blocked=False
        )

        # Create subscription
        subscription = cultpass.Subscription(
            subscription_id="sub123",
            user_id="test123",
            status="active",
            tier="premium",
            monthly_quota=8
        )

        # Create experience
        experience = cultpass.Experience(
            experience_id="exp123",
            title="Test Experience",
            description="Test description",
            location="Test Location",
            when=None,  # Set manually if needed
            slots_available=10,
            is_premium=False
        )

        # Create reservation
        reservation = cultpass.Reservation(
            reservation_id="res123",
            user_id="test123",
            experience_id="exp123",
            status="reserved"
        )

        session.add_all([user, subscription, experience, reservation])

    yield db_path

    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def tools(test_db):
    """Create CultPassTools instance with test database."""
    return CultPassTools(db_path=test_db)


def test_lookup_user_by_email(tools):
    """Test user lookup by email."""
    result = tools.lookup_user(email="test@example.com")

    assert result['success'] is True
    assert result['user']['email'] == "test@example.com"
    assert result['user']['full_name'] == "Test User"
    assert result['subscription']['status'] == "active"


def test_lookup_user_by_id(tools):
    """Test user lookup by user ID."""
    result = tools.lookup_user(user_id="test123")

    assert result['success'] is True
    assert result['user']['user_id'] == "test123"
    assert result['subscription'] is not None


def test_lookup_user_not_found(tools):
    """Test user lookup when user doesn't exist."""
    result = tools.lookup_user(email="nonexistent@example.com")

    assert result['success'] is False
    assert 'error' in result


def test_lookup_user_missing_params(tools):
    """Test user lookup with missing parameters."""
    result = tools.lookup_user()

    assert result['success'] is False
    assert 'error' in result


def test_check_subscription(tools):
    """Test subscription check."""
    result = tools.check_subscription(user_id="test123")

    assert result['success'] is True
    assert result['subscription']['tier'] == "premium"
    assert result['subscription']['monthly_quota'] == 8
    assert 'usage' in result


def test_check_subscription_user_not_found(tools):
    """Test subscription check for non-existent user."""
    result = tools.check_subscription(user_id="nonexistent")

    assert result['success'] is False


def test_manage_reservation_view(tools):
    """Test viewing reservations."""
    result = tools.manage_reservation(user_id="test123", action="view")

    assert result['success'] is True
    assert result['action'] == "view"
    assert 'reservations' in result
    assert len(result['reservations']) > 0


def test_manage_reservation_cancel_missing_id(tools):
    """Test cancelling reservation without reservation_id."""
    result = tools.manage_reservation(user_id="test123", action="cancel")

    assert result['success'] is False
    assert 'error' in result


def test_manage_reservation_invalid_action(tools):
    """Test invalid action."""
    result = tools.manage_reservation(user_id="test123", action="invalid")

    assert result['success'] is False
    assert 'Unknown action' in result['error']


def test_process_refund_basic(tools):
    """Test basic refund processing."""
    result = tools.process_refund(
        user_id="test123",
        transaction_type="subscription",
        reason="Not satisfied"
    )

    assert result['success'] is True
    assert 'eligible' in result
    assert 'requires_approval' in result


def test_process_refund_blocked_user(test_db):
    """Test refund for blocked user."""
    # Create a blocked user
    engine = create_engine(f"sqlite:///{test_db}", echo=False)

    with get_session(engine) as session:
        blocked_user = cultpass.User(
            user_id="blocked123",
            full_name="Blocked User",
            email="blocked@example.com",
            is_blocked=True
        )
        session.add(blocked_user)

    tools = CultPassTools(db_path=test_db)
    result = tools.process_refund(
        user_id="blocked123",
        transaction_type="subscription",
        reason="Want refund"
    )

    assert result['success'] is True
    assert result['eligible'] is False
    assert 'blocked' in result['message'].lower()


def test_process_refund_premium_event(tools):
    """Test premium event refund processing."""
    result = tools.process_refund(
        user_id="test123",
        transaction_type="premium_event",
        reason="Cannot attend"
    )

    assert result['success'] is True
    assert 'eligible' in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
