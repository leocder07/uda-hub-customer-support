# Memory Strategy for UDA-Hub

## Overview
UDA-Hub implements a dual-memory architecture to provide both contextual conversation continuity and personalized support based on historical customer interactions.

## Memory Types

### 1. Short-term Memory (Session/Conversation-based)

**Purpose**: Maintain context within a single ticket/conversation session

**Implementation**: LangGraph MemorySaver with thread_id

**Scope**: Single ticket lifecycle (from creation to resolution/escalation)

**What's Stored**:
- All messages in the conversation (user + agent responses)
- Agent decisions and routing choices
- Tool calls and their results
- Classifier outputs
- Confidence scores
- Intermediate state between agent invocations

**Storage Location**: In-memory during execution, can be persisted to SQLite checkpointer

**Access Pattern**:
```python
config = {
    "configurable": {
        "thread_id": ticket_id  # Each ticket has unique thread_id
    }
}

# Retrieve full conversation history
state_history = orchestrator.get_state_history(config)
```

**Lifecycle**:
- Created: When ticket is first received
- Updated: After each agent invocation
- Accessed: By all agents to maintain context
- Retained: For 7 days after ticket closure (configurable)
- Archived: Moved to long-term storage after retention period

**Use Cases**:
1. **Context Continuity**: Resolver agent references previous messages to avoid repetition
2. **Multi-turn Conversations**: User asks follow-up questions, agent remembers context
3. **Error Recovery**: If agent fails, supervisor can review history and retry
4. **Decision Tracking**: Audit trail of all routing decisions within a ticket

**Example**:
```
User: "I can't log in to my account"
Classifier: [login, medium urgency, user_id=a4ab87]
Resolver: "Try resetting your password..."

User: "That didn't work"
Resolver: [accesses short-term memory, sees password reset already suggested]
Resolver: "Let me check your account status..." [routes to Tool Executor]
```

### 2. Long-term Memory (Persistent Cross-session)

**Purpose**: Store and recall customer history across multiple tickets and sessions

**Implementation**: New table `customer_interactions` in udahub.db + semantic search

**Scope**: Entire customer lifetime with the system

**Database Schema**:
```sql
CREATE TABLE customer_interactions (
    interaction_id TEXT PRIMARY KEY,
    account_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    ticket_id TEXT NOT NULL,
    interaction_date DATETIME NOT NULL,
    issue_type TEXT NOT NULL,
    resolution_summary TEXT NOT NULL,
    agent_notes TEXT,
    sentiment TEXT,
    was_escalated BOOLEAN,
    resolution_time_seconds INTEGER,
    customer_satisfaction INTEGER,  -- 1-5 scale, optional
    embedding BLOB,  -- Stored vector embedding for semantic search
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (account_id) REFERENCES accounts(account_id),
    FOREIGN KEY (user_id) REFERENCES users(user_id),
    FOREIGN KEY (ticket_id) REFERENCES tickets(ticket_id)
);

CREATE INDEX idx_user_interactions ON customer_interactions(user_id, interaction_date DESC);
CREATE INDEX idx_issue_type ON customer_interactions(issue_type);
```

**Additional Table for Customer Preferences**:
```sql
CREATE TABLE customer_preferences (
    preference_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    preference_type TEXT NOT NULL,  -- e.g., 'communication_style', 'issue_category'
    preference_value TEXT NOT NULL,
    confidence FLOAT,  -- How confident we are about this preference
    last_updated DATETIME,
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);
```

**What's Stored**:
1. **Resolved Issues**: Summary of past ticket resolutions
2. **Common Issues**: Patterns in customer's repeat issues
3. **Preferences**: Communication style, preferred resolution methods
4. **Escalation History**: Previous escalations and their outcomes
5. **Sentiment Trends**: How customer sentiment has changed over time
6. **Tool Usage**: What database operations were performed for this customer

**Storage Location**: UDA-Hub database (data/core/udahub.db)

**Access Patterns**:

**1. Retrieve Customer History (most recent 5 interactions)**:
```python
def get_customer_history(user_id: str, limit: int = 5):
    with get_session(engine) as session:
        interactions = session.query(CustomerInteraction)\
            .filter_by(user_id=user_id)\
            .order_by(CustomerInteraction.interaction_date.desc())\
            .limit(limit)\
            .all()
        return [interaction_to_dict(i) for i in interactions]
```

**2. Semantic Search for Similar Past Issues**:
```python
def find_similar_past_issues(user_id: str, current_issue: str, top_k: int = 3):
    # Generate embedding for current issue
    current_embedding = openai.embeddings.create(
        input=current_issue,
        model="text-embedding-3-small"
    )

    # Retrieve all user's past interactions
    past_interactions = get_customer_history(user_id, limit=50)

    # Compute similarity scores
    similarities = []
    for interaction in past_interactions:
        similarity = cosine_similarity(current_embedding, interaction['embedding'])
        similarities.append((interaction, similarity))

    # Return top_k most similar
    return sorted(similarities, key=lambda x: x[1], reverse=True)[:top_k]
```

**3. Check for Repeated Issues (potential account problem)**:
```python
def check_repeated_issue(user_id: str, issue_type: str, days: int = 30):
    """Check if user has reported same issue type recently"""
    cutoff_date = datetime.now() - timedelta(days=days)

    with get_session(engine) as session:
        count = session.query(CustomerInteraction)\
            .filter_by(user_id=user_id, issue_type=issue_type)\
            .filter(CustomerInteraction.interaction_date >= cutoff_date)\
            .count()

        return count  # If > 2, flag as repeated issue
```

**Lifecycle**:
- Created: After each ticket is resolved or escalated
- Updated: Never (append-only for audit trail)
- Accessed: At ticket start (load history) and during resolution (check patterns)
- Retained: Indefinitely (compliance permitting) or per data retention policy

**Use Cases**:

1. **Personalized Greetings**:
   ```
   History shows: User had login issue 2 weeks ago, resolved
   Supervisor: "Welcome back! I see we helped you with login recently. How can I assist today?"
   ```

2. **Pattern Detection**:
   ```
   History shows: User reported billing issue 3 times in 2 months
   Classifier: [flags as repeated issue, escalate priority]
   Escalation: "Customer has recurring billing issues, may indicate systemic problem"
   ```

3. **Proactive Solutions**:
   ```
   History shows: User always forgets to cancel reservations 24h in advance
   Resolver: "Based on your past reservations, here's a reminder about our 24h cancellation policy..."
   ```

4. **Sentiment Tracking**:
   ```
   History shows: Sentiment degrading from neutral → frustrated → angry
   Supervisor: [auto-escalate to human agent with high priority]
   ```

5. **Resolution Optimization**:
   ```
   History shows: User prefers step-by-step instructions over links
   Resolver: [formats response with numbered steps instead of article link]
   ```

## Memory Integration in Workflow

### On Ticket Arrival:
```python
def handle_new_ticket(ticket_id: str, user_id: str, message: str):
    # 1. Load short-term memory (create new thread)
    config = {"configurable": {"thread_id": ticket_id}}

    # 2. Load long-term memory
    customer_history = get_customer_history(user_id, limit=5)
    repeated_issues = check_repeated_issue(user_id, issue_type)

    # 3. Inject history into context
    context_message = f"""
    Customer History Summary:
    - Previous tickets: {len(customer_history)}
    - Recent issues: {[h['issue_type'] for h in customer_history]}
    - Sentiment trend: {get_sentiment_trend(customer_history)}
    - Repeated issue alert: {repeated_issues > 2}
    """

    # 4. Pass to supervisor with enriched context
    state = {
        "messages": [SystemMessage(content=context_message),
                     HumanMessage(content=message)],
        "ticket_id": ticket_id,
        "customer_history": customer_history
    }

    result = orchestrator.invoke(state, config=config)
    return result
```

### On Ticket Resolution:
```python
def save_interaction(ticket_id: str, user_id: str, resolution: dict):
    # Generate embedding for semantic search
    text_to_embed = f"{resolution['issue_type']} {resolution['summary']}"
    embedding = openai.embeddings.create(
        input=text_to_embed,
        model="text-embedding-3-small"
    ).data[0].embedding

    # Store in long-term memory
    with get_session(engine) as session:
        interaction = CustomerInteraction(
            interaction_id=str(uuid.uuid4()),
            account_id=get_account_id(user_id),
            user_id=user_id,
            ticket_id=ticket_id,
            interaction_date=datetime.now(),
            issue_type=resolution['issue_type'],
            resolution_summary=resolution['summary'],
            agent_notes=resolution.get('agent_notes'),
            sentiment=resolution['sentiment'],
            was_escalated=resolution['status'] == 'escalated',
            resolution_time_seconds=resolution['time_taken'],
            embedding=pickle.dumps(embedding)  # Store as blob
        )
        session.add(interaction)
```

## Memory Cleanup and Maintenance

### Short-term Memory Cleanup:
```python
# Run daily job to archive old threads
def cleanup_old_threads(days_to_keep: int = 7):
    cutoff_date = datetime.now() - timedelta(days=days_to_keep)

    # Archive threads older than cutoff to long-term storage
    # Delete from active checkpointer
    pass
```

### Long-term Memory Analytics:
```python
# Weekly job to extract insights
def analyze_customer_patterns():
    # Identify customers with repeated issues
    # Flag potential systemic problems
    # Update customer preferences based on interactions
    # Generate reports for support team
    pass
```

## Privacy and Compliance

### Data Retention:
- Short-term: 7 days after ticket closure
- Long-term: Configurable per data protection regulations (e.g., GDPR)
- Sensitive data (PII): Redacted or encrypted

### User Rights:
- Right to be forgotten: Implement deletion of customer_interactions on request
- Data export: Provide function to export all customer interactions

### Anonymization:
- Production: Store actual user data
- Analytics: Use anonymized user_id hashes
- Sharing with ML team: Only anonymized, aggregated data

## Performance Considerations

### Caching:
- Cache recent customer histories (last 100 active users) in Redis
- Cache embeddings for frequent issue types
- TTL: 1 hour for history cache

### Indexing:
- Index on (user_id, interaction_date) for fast history retrieval
- Index on issue_type for pattern analysis
- Consider vector index (e.g., pgvector) for production semantic search at scale

### Lazy Loading:
- Don't load full history unless needed
- Load summary first, then full details on demand
- Paginate history for users with many interactions

## Metrics and Monitoring

Track memory system effectiveness:
- **Cache hit rate**: % of history queries served from cache
- **Personalization impact**: Resolution rate difference with/without history
- **Pattern detection accuracy**: % of repeated issues caught
- **Storage growth**: Size of customer_interactions table over time
- **Query performance**: P95 latency for history retrieval

## Future Enhancements

1. **Federated Learning**: Learn patterns across all customers without centralizing PII
2. **Memory Compression**: Summarize old interactions to save space
3. **Multi-modal Memory**: Store interaction type (chat, email, phone) and adapt style
4. **Predictive Escalation**: Use history to predict escalation before low confidence
5. **Memory Sharing**: Allow agents to share insights about common issues (anonymized)
