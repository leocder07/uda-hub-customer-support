#!/usr/bin/env python3
"""
UDA-Hub Agentic Application

Main entry point for running the UDA-Hub customer support agent system.
Provides a command-line interface for testing the multi-agent workflow.
"""

import os
import sys
import uuid
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Verify OpenAI API key is set
if not os.getenv("OPENAI_API_KEY"):
    print("❌ Error: OPENAI_API_KEY not found in environment variables")
    print("Please create a .env file with your OpenAI API key:")
    print("OPENAI_API_KEY=sk-...")
    sys.exit(1)

# Import utilities and workflow
from utils import chat_interface, ensure_db_initialized
from agentic.workflow import orchestrator, process_ticket


def main():
    """Main entry point for the application."""

    print("="*60)
    print("UDA-Hub: Universal Decision Agent for Customer Support")
    print("="*60)
    print()

    # Ensure databases are initialized
    print("🔍 Checking database setup...")
    udahub_db = ensure_db_initialized("data/core/udahub.db")
    print()

    # Check if FAISS index exists
    if not os.path.exists("data/core/faiss_index.bin"):
        print("⚠️  FAISS index not found!")
        print("Please run 02_core_db_setup.ipynb to build the knowledge base index.")
        print()
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(0)
        print()

    print("✅ System ready!")
    print()
    print("Choose an option:")
    print("1. Interactive chat mode (with session memory)")
    print("2. Process single ticket (no session)")
    print("3. Run test cases")
    print("4. Exit")
    print()

    choice = input("Enter choice (1-4): ").strip()
    print()

    if choice == "1":
        run_interactive_mode()
    elif choice == "2":
        run_single_ticket()
    elif choice == "3":
        run_test_cases()
    elif choice == "4":
        print("Goodbye!")
    else:
        print("Invalid choice. Exiting.")


def run_interactive_mode():
    """Run interactive chat mode with session memory."""
    print("="*60)
    print("Interactive Chat Mode")
    print("="*60)
    print()

    # Generate a ticket ID for this session
    ticket_id = f"ticket_{uuid.uuid4().hex[:8]}"
    print(f"Ticket ID: {ticket_id}")
    print()

    # Optional: Get user ID for customer history
    user_id = input("Enter user ID (or press Enter to skip): ").strip()
    if not user_id:
        user_id = None

    print()
    print("Starting chat interface...")
    print("Type 'quit', 'exit', or 'q' to end the conversation")
    print("="*60)
    print()

    # Start chat interface
    chat_interface(orchestrator, ticket_id)


def run_single_ticket():
    """Process a single ticket without session continuity."""
    print("="*60)
    print("Single Ticket Processing")
    print("="*60)
    print()

    # Get ticket details
    ticket_message = input("Enter customer message: ").strip()
    if not ticket_message:
        print("No message provided. Exiting.")
        return

    user_id = input("Enter user ID (optional): ").strip() or None

    ticket_id = f"ticket_{uuid.uuid4().hex[:8]}"

    print()
    print(f"Processing ticket {ticket_id}...")
    print("="*60)
    print()

    # Process ticket
    try:
        result = process_ticket(
            ticket_id=ticket_id,
            user_message=ticket_message,
            account_id="cultpass",
            user_id=user_id
        )

        # Display results
        print()
        print("="*60)
        print("RESULTS")
        print("="*60)
        print()
        print(f"Status: {result['status'].upper()}")
        print(f"Confidence: {result.get('confidence', 0):.1%}")
        print()
        print("Final Response:")
        print("-" * 60)
        print(result.get('final_response', 'No response generated'))
        print("-" * 60)
        print()
        print("Decision Log:")
        for i, decision in enumerate(result.get('decision_log', []), 1):
            print(f"  {i}. {decision}")
        print()

        if result.get('classification'):
            print("Classification:")
            print(f"  Type: {result['classification'].get('ticket_type')}")
            print(f"  Urgency: {result['classification'].get('urgency')}")
            print(f"  Sentiment: {result['classification'].get('sentiment')}")
            print()

    except Exception as e:
        print(f"❌ Error processing ticket: {e}")
        import traceback
        traceback.print_exc()


def run_test_cases():
    """Run predefined test cases."""
    print("="*60)
    print("Running Test Cases")
    print("="*60)
    print()

    test_cases = [
        {
            "name": "Login Issue",
            "message": "I can't log in to my account. I forgot my password.",
            "user_id": None
        },
        {
            "name": "Reservation Cancellation",
            "message": "I need to cancel my reservation for tomorrow's event.",
            "user_id": "f556c0"
        },
        {
            "name": "Billing Question",
            "message": "What's included in my subscription?",
            "user_id": None
        },
        {
            "name": "Technical Issue",
            "message": "The app keeps crashing when I try to open it.",
            "user_id": None
        }
    ]

    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest Case {i}: {test_case['name']}")
        print("-" * 60)
        print(f"Message: {test_case['message']}")
        print()

        ticket_id = f"test_{i}_{uuid.uuid4().hex[:6]}"

        try:
            result = process_ticket(
                ticket_id=ticket_id,
                user_message=test_case['message'],
                account_id="cultpass",
                user_id=test_case['user_id']
            )

            print(f"✓ Status: {result['status']}")
            print(f"✓ Confidence: {result.get('confidence', 0):.1%}")
            print(f"✓ Type: {result['classification'].get('ticket_type')}")
            print()

        except Exception as e:
            print(f"✗ Error: {e}")
            print()

    print("="*60)
    print("Test cases completed!")
    print("="*60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
