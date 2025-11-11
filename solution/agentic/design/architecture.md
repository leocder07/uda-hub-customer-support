# UDA-Hub Multi-Agent Architecture

## Overview
UDA-Hub implements a **Supervisor Pattern** multi-agent system designed to intelligently process customer support tickets for CultPass through automated decision-making, knowledge retrieval, and escalation logic.

## Architecture Pattern: Supervisor Pattern

We chose the Supervisor Pattern because:
- **Clear routing logic**: Supervisor makes all routing decisions based on classifier output
- **Maintainability**: Each agent has a single, well-defined responsibility
- **Scalability**: Easy to add new specialized agents without changing core logic
- **Observability**: Centralized decision tracking through the supervisor
- **Error handling**: Supervisor can catch agent failures and route to escalation

## System Components

### 1. Agents (5 Specialized Agents)

#### 1.1 Classifier Agent
**Purpose**: Analyzes incoming tickets and extracts structured information

**Responsibilities**:
- Classify ticket type (login, billing, reservation, technical, account)
- Determine urgency level (low, medium, high, critical)
- Extract relevant entities (user_id, reservation_id, etc.)
- Analyze sentiment (positive, neutral, negative, frustrated)
- Identify required expertise (knowledge_base, database_operations, human_agent)

**Output Schema**:
```python
{
  "ticket_type": str,
  "urgency": str,
  "sentiment": str,
  "entities": dict,
  "required_expertise": list[str],
  "keywords": list[str]
}
```

#### 1.2 Resolver Agent
**Purpose**: Attempts to resolve tickets using RAG-powered knowledge base search

**Responsibilities**:
- Perform semantic search over knowledge base using vector embeddings
- Retrieve top 3 most relevant articles
- Generate responses based on retrieved knowledge
- Calculate confidence score (0-1) for the proposed solution
- Determine if resolution is sufficient or needs escalation

**Confidence Scoring**:
- 0.8-1.0: High confidence, provide resolution
- 0.5-0.8: Medium confidence, provide resolution with caveats
- 0.0-0.5: Low confidence, escalate to human or tool execution

**Output Schema**:
```python
{
  "resolution": str,
  "confidence": float,
  "sources": list[str],  # Article titles used
  "needs_escalation": bool,
  "recommended_action": str
}
```

#### 1.3 Tool Executor Agent
**Purpose**: Executes database operations on CultPass external system

**Responsibilities**:
- Lookup user information (subscription status, account details)
- Check and manage reservations
- Verify billing and payment information
- Execute approved actions (cancel reservation, pause subscription)
- Handle refund requests with approval workflow

**Available Tools**:
- `lookup_user`: Get user details from CultPass DB
- `check_subscription`: Verify subscription status, tier, quota
- `manage_reservation`: View/modify user reservations
- `process_refund`: Handle refund requests (requires approval)

**Output Schema**:
```python
{
  "tool_used": str,
  "tool_result": dict,
  "action_taken": str,
  "requires_followup": bool,
  "followup_message": str
}
```

#### 1.4 Escalation Agent
**Purpose**: Creates comprehensive escalation summaries for human agents

**Responsibilities**:
- Summarize the ticket history and context
- List all attempted resolution steps
- Highlight blocking issues or missing information
- Provide recommendations for human agent
- Categorize escalation reason (complex, policy_exception, technical_limitation, customer_request)
- Assign priority level for human queue

**Output Schema**:
```python
{
  "escalation_summary": str,
  "attempted_steps": list[str],
  "escalation_reason": str,
  "priority": str,
  "recommended_actions": list[str],
  "customer_sentiment": str
}
```

#### 1.5 Supervisor Agent
**Purpose**: Orchestrates the entire workflow and makes routing decisions

**Responsibilities**:
- Receive classifier output and route to appropriate agent
- Monitor agent execution and handle failures
- Make final resolution decisions
- Track decision path for observability
- Handle multi-step workflows (e.g., tool execution → resolution)

**Routing Logic**:
```
IF urgency == "critical" AND required_expertise contains "human_agent":
  → Escalation Agent (immediate)

ELIF required_expertise contains "database_operations":
  → Tool Executor Agent → Resolver Agent (for final response)

ELIF required_expertise contains "knowledge_base":
  → Resolver Agent
  → IF confidence < 0.5: Escalation Agent

ELSE:
  → Resolver Agent (default)
```

### 2. Tools (FastMCP-based)

#### 2.1 CultPass MCP Server
Located at: `solution/agentic/tools/cultpass_mcp_server.py`

**Tools Provided**:

1. **lookup_user**
   - Input: email or external_user_id
   - Output: User details, subscription status, account standing
   - Database: CultPass DB (external)

2. **check_subscription**
   - Input: user_id
   - Output: Subscription tier, status, monthly quota, usage
   - Database: CultPass DB (external)

3. **manage_reservation**
   - Input: user_id, reservation_id, action (view/cancel)
   - Output: Reservation details or cancellation confirmation
   - Database: CultPass DB (external)

4. **process_refund**
   - Input: user_id, transaction_type, amount, reason
   - Output: Refund eligibility check, approval requirement
   - Note: Actual refund requires human approval for compliance

#### 2.2 Knowledge Search Tool
Located at: `solution/agentic/tools/knowledge_search_tool.py`

- Semantic search over knowledge base using FAISS vector store
- Uses OpenAI embeddings (text-embedding-3-small)
- Returns top-k relevant articles with similarity scores
- Threshold: 0.7 for relevance filtering

### 3. RAG System

**Vector Store**: FAISS (Facebook AI Similarity Search)
- Lightweight, no external dependencies
- Fast similarity search
- Persisted to disk for reuse

**Embedding Model**: OpenAI text-embedding-3-small
- 1536 dimensions
- Cost-effective
- Good performance for short documents

**Indexing Process**:
1. Load all 15 knowledge articles from database
2. Generate embeddings for article title + content + tags
3. Store in FAISS index with metadata
4. Save index to disk for persistence

**Retrieval Process**:
1. User query → Generate embedding
2. FAISS similarity search → Top 3 results
3. Filter by similarity threshold (0.7)
4. Return article content and metadata

**See**: `solution/agentic/design/rag-system.md` for detailed implementation

### 4. Memory Systems

**Short-term Memory (Session-based)**:
- Implementation: LangGraph MemorySaver with thread_id
- Scope: Single conversation/ticket
- Stores: Message history, agent decisions, tool calls
- Purpose: Context continuity within a ticket

**Long-term Memory (Persistent)**:
- Implementation: New table `customer_interactions` in udahub.db
- Scope: Cross-session customer history
- Stores: Previous resolutions, preferences, common issues
- Purpose: Personalized support based on history

**See**: `solution/agentic/design/memory-strategy.md` for detailed implementation

## Workflow Execution Flow

```
1. Ticket arrives → Supervisor receives it
2. Supervisor → Classifier Agent
3. Classifier analyzes → Returns classification
4. Supervisor makes routing decision based on:
   - Ticket type
   - Urgency
   - Required expertise
   - Customer history (long-term memory)

5a. Route: Knowledge Base Path
    Supervisor → Resolver Agent
    Resolver performs RAG search
    IF confidence >= 0.5:
      Return resolution
    ELSE:
      Supervisor → Escalation Agent

5b. Route: Tool Execution Path
    Supervisor → Tool Executor Agent
    Tool Executor calls appropriate MCP tool
    Tool Executor → Resolver Agent (for response formatting)
    Return resolution with tool results

5c. Route: Immediate Escalation Path
    Supervisor → Escalation Agent
    Return escalation summary

6. Log all decisions and actions to database
7. Update long-term memory with resolution
8. Return final response to user
```

## State Management

The LangGraph state contains:
```python
class AgentState(TypedDict):
    messages: list[BaseMessage]
    ticket_id: str
    classification: dict
    resolver_output: dict
    tool_output: dict
    escalation_output: dict
    confidence: float
    decision_log: list[str]
    final_response: str
    status: str  # "resolved", "escalated", "pending"
```

## Error Handling

1. **Agent Failures**: Supervisor catches exceptions and routes to escalation
2. **Tool Failures**: Tool Executor returns error message, Supervisor escalates
3. **Low Confidence**: Resolver flags for escalation if confidence < 0.5
4. **Timeout**: Max execution time 30 seconds, then escalate
5. **Invalid Input**: Classifier validates input, requests clarification if needed

## Logging and Observability

All agent decisions are logged with:
- Timestamp
- Agent name
- Input/Output
- Confidence scores
- Routing decisions
- Tool calls
- Execution time

Logs stored in: `solution/logs/agent_decisions.jsonl`

## Performance Considerations

- **Token Optimization**: Use gpt-4o-mini for cost efficiency
- **Caching**: FAISS index cached in memory, embeddings cached
- **Parallel Execution**: Independent agents can run in parallel (not implemented in v1)
- **Timeout**: 30-second max per agent to prevent hanging

## Future Enhancements

1. **Parallel Agent Execution**: Use LangGraph's parallelization features
2. **Feedback Loop**: Collect human agent edits to improve RAG
3. **A/B Testing**: Test different routing strategies
4. **Sentiment-based Routing**: Priority for frustrated customers
5. **Multi-language Support**: Detect and route to language-specific agents
