"""
CultPass MCP Server

FastMCP-based tools for interacting with the CultPass external database.
Provides tools for user lookup, subscription management, reservation operations,
and refund processing.
"""

import os
from datetime import datetime, timedelta
from typing import Dict, Optional, List
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import sys

# Add parent directories to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from data.models import cultpass
from utils import get_session, model_to_dict


class CultPassTools:
    """
    Tools for interacting with CultPass database.

    These tools abstract database operations for the agent system.
    """

    def __init__(self, db_path: str = "data/external/cultpass.db"):
        """
        Initialize CultPass tools.

        Args:
            db_path: Path to CultPass SQLite database
        """
        self.db_path = db_path
        self.engine = None

    def _get_engine(self):
        """Get or create database engine."""
        if self.engine is None:
            # Convert to absolute path if relative
            if not os.path.isabs(self.db_path):
                solution_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
                db_path = os.path.join(solution_dir, self.db_path)
            else:
                db_path = self.db_path

            self.engine = create_engine(f"sqlite:///{db_path}", echo=False)

        return self.engine

    def lookup_user(self, email: Optional[str] = None, user_id: Optional[str] = None) -> Dict:
        """
        Lookup user information by email or user_id.

        Args:
            email: User's email address
            user_id: User's external ID

        Returns:
            Dictionary containing user details and subscription status:
            {
                "success": bool,
                "user": {...},
                "subscription": {...},
                "error": str (if failed)
            }
        """
        if not email and not user_id:
            return {
                "success": False,
                "error": "Either email or user_id must be provided"
            }

        engine = self._get_engine()

        try:
            with get_session(engine) as session:
                # Query user
                query = session.query(cultpass.User)
                if email:
                    user = query.filter_by(email=email).first()
                else:
                    user = query.filter_by(user_id=user_id).first()

                if not user:
                    return {
                        "success": False,
                        "error": "User not found"
                    }

                # Get subscription
                subscription = user.subscription

                return {
                    "success": True,
                    "user": {
                        "user_id": user.user_id,
                        "full_name": user.full_name,
                        "email": user.email,
                        "is_blocked": user.is_blocked,
                        "created_at": user.created_at.isoformat() if user.created_at else None
                    },
                    "subscription": {
                        "subscription_id": subscription.subscription_id if subscription else None,
                        "status": subscription.status if subscription else "none",
                        "tier": subscription.tier if subscription else "none",
                        "monthly_quota": subscription.monthly_quota if subscription else 0,
                        "started_at": subscription.started_at.isoformat() if subscription and subscription.started_at else None,
                        "ended_at": subscription.ended_at.isoformat() if subscription and subscription.ended_at else None
                    } if subscription else None
                }

        except Exception as e:
            return {
                "success": False,
                "error": f"Database error: {str(e)}"
            }

    def check_subscription(self, user_id: str) -> Dict:
        """
        Check subscription details and usage for a user.

        Args:
            user_id: User's external ID

        Returns:
            Dictionary containing subscription details and reservation count:
            {
                "success": bool,
                "subscription": {...},
                "usage": {...},
                "error": str (if failed)
            }
        """
        engine = self._get_engine()

        try:
            with get_session(engine) as session:
                # Get user
                user = session.query(cultpass.User).filter_by(user_id=user_id).first()

                if not user:
                    return {
                        "success": False,
                        "error": "User not found"
                    }

                # Get subscription
                subscription = user.subscription

                if not subscription:
                    return {
                        "success": True,
                        "subscription": None,
                        "message": "User has no active subscription"
                    }

                # Count reservations for current month
                now = datetime.now()
                month_start = datetime(now.year, now.month, 1)
                reservations_this_month = session.query(cultpass.Reservation)\
                    .filter_by(user_id=user_id)\
                    .filter(cultpass.Reservation.created_at >= month_start)\
                    .filter(cultpass.Reservation.status == 'reserved')\
                    .count()

                return {
                    "success": True,
                    "subscription": {
                        "subscription_id": subscription.subscription_id,
                        "status": subscription.status,
                        "tier": subscription.tier,
                        "monthly_quota": subscription.monthly_quota,
                        "started_at": subscription.started_at.isoformat() if subscription.started_at else None,
                        "ended_at": subscription.ended_at.isoformat() if subscription.ended_at else None
                    },
                    "usage": {
                        "reservations_this_month": reservations_this_month,
                        "quota_remaining": max(0, subscription.monthly_quota - reservations_this_month),
                        "quota_exceeded": reservations_this_month >= subscription.monthly_quota
                    }
                }

        except Exception as e:
            return {
                "success": False,
                "error": f"Database error: {str(e)}"
            }

    def manage_reservation(
        self,
        user_id: str,
        reservation_id: Optional[str] = None,
        action: str = "view"
    ) -> Dict:
        """
        View or manage user reservations.

        Args:
            user_id: User's external ID
            reservation_id: Specific reservation ID (required for cancel)
            action: Action to perform ("view", "cancel")

        Returns:
            Dictionary containing reservation details or action result:
            {
                "success": bool,
                "action": str,
                "reservations": [...] (for view),
                "message": str (for cancel),
                "error": str (if failed)
            }
        """
        engine = self._get_engine()

        try:
            with get_session(engine) as session:
                # Get user
                user = session.query(cultpass.User).filter_by(user_id=user_id).first()

                if not user:
                    return {
                        "success": False,
                        "error": "User not found"
                    }

                if action == "view":
                    # Get all active reservations
                    reservations = session.query(cultpass.Reservation)\
                        .filter_by(user_id=user_id)\
                        .filter(cultpass.Reservation.status == 'reserved')\
                        .all()

                    reservation_list = []
                    for res in reservations:
                        experience = res.experience
                        reservation_list.append({
                            "reservation_id": res.reservation_id,
                            "status": res.status,
                            "created_at": res.created_at.isoformat() if res.created_at else None,
                            "experience": {
                                "experience_id": experience.experience_id,
                                "title": experience.title,
                                "location": experience.location,
                                "when": experience.when.isoformat() if experience.when else None,
                                "is_premium": experience.is_premium
                            }
                        })

                    return {
                        "success": True,
                        "action": "view",
                        "reservations": reservation_list,
                        "count": len(reservation_list)
                    }

                elif action == "cancel":
                    if not reservation_id:
                        return {
                            "success": False,
                            "error": "reservation_id required for cancel action"
                        }

                    # Get reservation
                    reservation = session.query(cultpass.Reservation)\
                        .filter_by(reservation_id=reservation_id, user_id=user_id)\
                        .first()

                    if not reservation:
                        return {
                            "success": False,
                            "error": "Reservation not found or does not belong to this user"
                        }

                    # Check if already cancelled
                    if reservation.status == 'cancelled':
                        return {
                            "success": False,
                            "error": "Reservation is already cancelled"
                        }

                    # Check cancellation timing (24 hour policy)
                    experience = reservation.experience
                    hours_until_event = (experience.when - datetime.now()).total_seconds() / 3600

                    if hours_until_event < 24:
                        credit_returned = False
                        message = "Reservation cancelled. Note: Cancelled within 24 hours, experience credit not returned."
                    else:
                        credit_returned = True
                        message = "Reservation cancelled successfully. Experience credit returned to your monthly quota."

                    # Update reservation status
                    reservation.status = 'cancelled'
                    reservation.updated_at = datetime.now()

                    # Increment experience slots
                    experience.slots_available += 1

                    session.commit()

                    return {
                        "success": True,
                        "action": "cancel",
                        "message": message,
                        "credit_returned": credit_returned,
                        "reservation_id": reservation_id
                    }

                else:
                    return {
                        "success": False,
                        "error": f"Unknown action: {action}. Must be 'view' or 'cancel'"
                    }

        except Exception as e:
            return {
                "success": False,
                "error": f"Database error: {str(e)}"
            }

    def process_refund(
        self,
        user_id: str,
        transaction_type: str,
        reason: str,
        amount: Optional[float] = None
    ) -> Dict:
        """
        Process refund request (requires human approval).

        Args:
            user_id: User's external ID
            transaction_type: Type of transaction ("subscription", "premium_event")
            reason: Reason for refund request
            amount: Optional refund amount

        Returns:
            Dictionary containing refund eligibility and approval requirement:
            {
                "success": bool,
                "eligible": bool,
                "requires_approval": bool,
                "message": str,
                "error": str (if failed)
            }
        """
        engine = self._get_engine()

        try:
            with get_session(engine) as session:
                # Get user
                user = session.query(cultpass.User).filter_by(user_id=user_id).first()

                if not user:
                    return {
                        "success": False,
                        "error": "User not found"
                    }

                # Check if account is in good standing
                if user.is_blocked:
                    return {
                        "success": True,
                        "eligible": False,
                        "message": "Account is blocked. Refunds not available for blocked accounts."
                    }

                # Get subscription info
                subscription = user.subscription

                if not subscription:
                    return {
                        "success": True,
                        "eligible": False,
                        "message": "User has no subscription to refund."
                    }

                # Refund policy checks
                if transaction_type == "subscription":
                    # Check subscription age
                    days_since_start = (datetime.now() - subscription.started_at).days if subscription.started_at else 0

                    if days_since_start < 7:
                        eligible = True
                        message = "User is within 7-day window. Refund eligible but requires support lead approval."
                        requires_approval = True
                    else:
                        eligible = False
                        message = "Subscription refunds only available within 7 days of purchase per policy."
                        requires_approval = False

                elif transaction_type == "premium_event":
                    # Premium events may be refundable
                    eligible = True
                    message = "Premium event refund eligible if cancelled 48+ hours before event. Requires verification of event timing."
                    requires_approval = True

                else:
                    eligible = False
                    message = f"Unknown transaction type: {transaction_type}"
                    requires_approval = False

                return {
                    "success": True,
                    "eligible": eligible,
                    "requires_approval": requires_approval,
                    "message": message,
                    "user_info": {
                        "user_id": user_id,
                        "full_name": user.full_name,
                        "email": user.email
                    },
                    "refund_request": {
                        "transaction_type": transaction_type,
                        "reason": reason,
                        "amount": amount
                    }
                }

        except Exception as e:
            return {
                "success": False,
                "error": f"Database error: {str(e)}"
            }


# Convenience functions for LangChain tool integration
def create_cultpass_tools(db_path: str = "data/external/cultpass.db") -> Dict:
    """
    Create CultPass tools for use with LangChain agents.

    Args:
        db_path: Path to CultPass database

    Returns:
        Dictionary of tool functions
    """
    tools = CultPassTools(db_path=db_path)

    return {
        "lookup_user": tools.lookup_user,
        "check_subscription": tools.check_subscription,
        "manage_reservation": tools.manage_reservation,
        "process_refund": tools.process_refund
    }
