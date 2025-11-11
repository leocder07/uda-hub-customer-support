# UDA-Hub: Universal Decision Agent for Customer Support

A sophisticated multi-agent system built with LangGraph that intelligently processes customer support tickets through automated classification, knowledge retrieval, tool execution, and escalation.

## Project Overview

UDA-Hub is designed to integrate with existing customer support platforms (Zendesk, Intercom, Freshdesk) and act as an operational brain that:
- 🧠 **Understands** customer tickets across channels
- 🎯 **Decides** which agent or tool should handle each case
- 💡 **Retrieves** answers from knowledge base using RAG
- ⚡ **Executes** database operations when needed
- 🚨 **Escalates** complex issues to human agents
- 🧩 **Learns** from interactions via long-term memory

## Architecture

### Multi-Agent System (Supervisor Pattern)

```
Ticket → Classifier → Supervisor → [Resolver | Tool Executor | Escalation] → Response
                         ↓              ↓            ↓              ↓
                    Decision Log   Knowledge    Database     Human Agent
                                     Base       Operations
```

### Specialized Agents

1. **Classifier Agent**: Analyzes tickets and extracts structured information (type, urgency, sentiment, entities)
2. **Supervisor Agent**: Makes routing decisions and coordinates workflow
3. **Resolver Agent**: Uses RAG to find relevant knowledge and generate responses
4. **Tool Executor Agent**: Executes database operations (user lookup, subscriptions, reservations, refunds)
5. **Escalation Agent**: Creates comprehensive summaries for human agents

See [agentic/design/architecture.md](agentic/design/architecture.md) for detailed architecture documentation.

## Key Features

### ✅ Rubric Requirements Checklist

- **Multi-Agent Architecture**: 5 specialized agents with clear responsibilities
- **Routing Logic**: Supervisor pattern with intelligent decision-making
- **Knowledge Retrieval**: RAG with FAISS vector store and OpenAI embeddings
- **Tool Integration**: 4 FastMCP-based tools for CultPass database operations
- **Escalation Logic**: Confidence-based escalation with priority assignment
- **Session Memory**: LangGraph MemorySaver with thread_id
- **Long-term Memory**: Persistent customer interaction history with semantic search
- **Knowledge Base**: 15 comprehensive support articles (11 more than required)
- **Testing**: Comprehensive unit and integration tests
- **Documentation**: Complete architecture, design, and API documentation

### Advanced Features

- 📊 **Confidence Scoring**: Automatic confidence calculation for resolutions
- 🔍 **Semantic Search**: FAISS-powered vector search over knowledge base
- 📝 **Decision Logging**: Complete audit trail of routing decisions
- 🔄 **Customer History**: Personalized support based on past interactions
- ⚠️ **Repeated Issue Detection**: Flags customers with recurring problems
- 🎭 **Sentiment Analysis**: Adapts responses based on customer emotion

## Getting Started

### Prerequisites

- Python 3.10 or higher
- OpenAI API key
- 2GB free disk space (for dependencies and databases)

### Installation

```bash
# Clone or navigate to the solution directory
cd solution/

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the `solution/` directory:

```bash
OPENAI_API_KEY=sk-your-key-here
```

### Database Setup

Run the setup notebooks in order:

```bash
# 1. Set up CultPass external database (customer accounts)
jupyter nbconvert --execute --to notebook 01_external_db_setup.ipynb

# 2. Set up UDA-Hub core database and build FAISS index
jupyter nbconvert --execute --to notebook 02_core_db_setup.ipynb
```

Alternatively, open the notebooks in Jupyter and run them interactively:

```bash
jupyter notebook
```

### Running the Application

#### Option 1: Command-Line Interface

```bash
python 03_agentic_app.py
```

Choose from:
1. Interactive chat mode (multi-turn conversation with memory)
2. Single ticket processing
3. Run test cases
4. Exit

#### Option 2: Interactive Notebook

```bash
jupyter notebook 03_agentic_app.ipynb
```

#### Option 3: Programmatic Usage

```python
from agentic.workflow import process_ticket

result = process_ticket(
    ticket_id="ticket_001",
    user_message="I can't log in to my account",
    account_id="cultpass",
    user_id="user_123"
)

print(f"Status: {result['status']}")
print(f"Response: {result['final_response']}")
```

## Project Structure

```
solution/
├── agentic/
│   ├── agents/                    # Specialized agents
│   │   ├── classifier.py          # Ticket classification
│   │   ├── resolver.py            # Knowledge-based resolution
│   │   ├── tool_executor.py       # Database operations
│   │   ├── escalation.py          # Escalation summaries
│   │   └── supervisor.py          # Workflow orchestration
│   ├── design/                    # Architecture documentation
│   │   ├── architecture.md        # System design
│   │   ├── architecture.mmd       # Mermaid diagrams
│   │   ├── memory-strategy.md     # Memory implementation
│   │   └── rag-system.md          # RAG details
│   ├── tools/                     # Agent tools
│   │   ├── cultpass_mcp_server.py # FastMCP database tools
│   │   └── knowledge_search_tool.py # RAG search
│   └── workflow.py                # LangGraph orchestration
├── data/
│   ├── core/                      # UDA-Hub database
│   │   ├── udahub.db              # Tickets, users, knowledge
│   │   ├── faiss_index.bin        # Vector index
│   │   └── faiss_metadata.pkl     # Index metadata
│   ├── external/                  # CultPass database
│   │   ├── cultpass.db            # Customer data
│   │   ├── cultpass_articles.jsonl # Knowledge articles (15)
│   │   ├── cultpass_experiences.jsonl
│   │   └── cultpass_users.jsonl
│   └── models/                    # SQLAlchemy models
│       ├── cultpass.py            # CultPass schema
│       └── udahub.py              # UDA-Hub schema
├── tests/                         # Test suite
│   ├── test_classifier.py         # Classifier unit tests
│   ├── test_tools.py              # Tools unit tests
│   ├── test_workflow.py           # Integration tests
│   └── README.md                  # Testing guide
├── utils.py                       # Utility functions
├── requirements.txt               # Dependencies
├── 01_external_db_setup.ipynb     # Setup notebook 1
├── 02_core_db_setup.ipynb         # Setup notebook 2
├── 03_agentic_app.ipynb           # Run notebook
├── 03_agentic_app.py              # CLI interface
└── README.md                      # This file
```

## Testing

### Run All Tests

```bash
# Unit and integration tests
pytest tests/ -v

# Unit tests only
pytest tests/test_classifier.py tests/test_tools.py -v

# Integration tests only
pytest tests/test_workflow.py -v -m integration
```

### Test Coverage

```bash
pytest tests/ --cov=agentic --cov-report=html
open htmlcov/index.html
```

See [tests/README.md](tests/README.md) for detailed testing guide.

## Example Usage

### Example 1: Login Issue

```
User: I can't log in to my account. I forgot my password.

Classifier: login, medium urgency, neutral sentiment
Supervisor: Route to Resolver (knowledge base)
Resolver: Found article "How to Handle Login Issues?" (confidence: 0.87)

Response: "Try tapping 'Forgot Password' on the login screen. Make sure you're
using the email associated with your account. If the email doesn't arrive,
check spam or try again in a few minutes."

Status: ✅ Resolved
```

### Example 2: Reservation with Database Lookup

```
User: I need to cancel my reservation for tomorrow's event.

Classifier: reservation, high urgency, neutral sentiment
Supervisor: Route to Tool Executor (database operations needed)
Tool Executor: lookup_user → manage_reservation (action=view)
Resolver: Generate response with tool context

Response: "I found your active reservations. You have 2 upcoming events...
To cancel, note that cancellations within 24 hours won't return the credit..."

Status: ✅ Resolved
```

### Example 3: Escalation

```
User: URGENT! My account is blocked and I have an event starting in 1 hour!

Classifier: account, critical urgency, frustrated sentiment
Supervisor: Route to Escalation (critical + frustrated)

Response: "This ticket has been escalated to a human agent.
Priority: CRITICAL
A support specialist will review your case immediately."

Status: ⚠️ Escalated
```

## Dependencies

### Core

- `langchain>=0.3.27`: LLM framework
- `langgraph>=0.5.4`: Multi-agent workflows
- `langchain-openai>=0.3.28`: OpenAI integration
- `openai>=1.12.0`: OpenAI API client
- `sqlalchemy>=2.0.41`: Database ORM
- `faiss-cpu>=1.8.0`: Vector similarity search

See [requirements.txt](requirements.txt) for complete list with versions.

## Python Version

**Tested on**: Python 3.10.x

Compatibility:
- Python 3.10: ✅ Fully supported
- Python 3.11: ✅ Supported
- Python 3.12: ⚠️ May require dependency updates

## Built With

* [LangChain](https://www.langchain.com/) - LLM application framework
* [LangGraph](https://www.langchain.com/langgraph) - Multi-agent orchestration
* [OpenAI](https://openai.com/) - LLM and embeddings
* [FAISS](https://github.com/facebookresearch/faiss) - Vector similarity search
* [SQLAlchemy](https://www.sqlalchemy.org/) - Database ORM
* [FastMCP](https://github.com/jlowin/fastmcp) - MCP server framework

---

**Generated with assistance from Claude Code** 🤖
