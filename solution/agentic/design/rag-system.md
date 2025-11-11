# RAG System Implementation for UDA-Hub

## Overview
UDA-Hub uses Retrieval-Augmented Generation (RAG) to provide accurate, knowledge-based responses to customer support tickets by semantically searching through the CultPass knowledge base.

## Architecture

```
User Query → Embedding Generation → Vector Similarity Search → Top-K Retrieval → Response Generation
```

## Components

### 1. Vector Store: FAISS (Facebook AI Similarity Search)

**Why FAISS?**
- **Lightweight**: No external database dependencies (unlike Chroma/Pinecone)
- **Fast**: Optimized for similarity search with millions of vectors
- **Persistent**: Can save/load index from disk
- **Production-ready**: Used by Meta in production systems
- **Cost-effective**: No API costs, runs locally

**Index Type**:
- `IndexFlatL2` for <10K documents (exact search, high accuracy)
- Can upgrade to `IndexIVFFlat` if knowledge base grows >10K (approximate search, faster)

**Storage**:
- Index file: `solution/data/core/faiss_index.bin`
- Metadata: `solution/data/core/faiss_metadata.pkl`

### 2. Embedding Model: OpenAI text-embedding-3-small

**Why OpenAI embeddings?**
- **Quality**: State-of-the-art semantic understanding
- **Dimensions**: 1536 (good balance of quality and size)
- **Cost**: $0.02 per 1M tokens (very affordable)
- **Speed**: Fast API response times
- **Consistency**: Same model for indexing and querying

**Alternative Considered**:
- Sentence-transformers (all-MiniLM-L6-v2): Free but lower quality
- OpenAI text-embedding-3-large: Higher quality but 3072 dimensions (overkill)

### 3. Document Processing

**Document Structure**:
Each knowledge article is processed as:
```python
{
    "article_id": str,
    "title": str,
    "content": str,
    "tags": str,
    "combined_text": f"{title}\n\n{content}\n\nTags: {tags}"  # What gets embedded
}
```

**Why combine title + content + tags?**
- Title contains key topic words (e.g., "Login Issues")
- Content has detailed information
- Tags provide keyword matching (e.g., "password, access")
- Combined embedding captures all semantic signals

### 4. Chunking Strategy

**Current Implementation**: No chunking (whole article)
- **Why**: Knowledge articles are short (<500 words each)
- **Benefit**: Preserves full context in each retrieval

**Future**: If articles grow >1000 words:
- Chunk size: 512 tokens with 128 token overlap
- Use sliding window to preserve context across chunks
- Store chunk_id and parent_article_id for reconstruction

## Implementation Details

### Indexing Pipeline

Located at: `solution/agentic/tools/knowledge_search_tool.py`

```python
def build_faiss_index(knowledge_articles: list[dict]):
    """
    Build FAISS index from knowledge articles
    """
    import faiss
    import openai
    import numpy as np
    import pickle

    # 1. Prepare documents
    documents = []
    metadata = []

    for article in knowledge_articles:
        combined_text = f"{article['title']}\n\n{article['content']}\n\nTags: {article['tags']}"
        documents.append(combined_text)
        metadata.append({
            'article_id': article['article_id'],
            'title': article['title'],
            'tags': article['tags']
        })

    # 2. Generate embeddings (batch for efficiency)
    embeddings = []
    batch_size = 100

    for i in range(0, len(documents), batch_size):
        batch = documents[i:i+batch_size]
        response = openai.embeddings.create(
            input=batch,
            model="text-embedding-3-small"
        )
        batch_embeddings = [item.embedding for item in response.data]
        embeddings.extend(batch_embeddings)

    # 3. Convert to numpy array
    embeddings_array = np.array(embeddings).astype('float32')

    # 4. Create FAISS index
    dimension = embeddings_array.shape[1]  # 1536 for text-embedding-3-small
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_array)

    # 5. Save index and metadata
    faiss.write_index(index, "solution/data/core/faiss_index.bin")
    with open("solution/data/core/faiss_metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)

    print(f"✅ Indexed {len(documents)} articles")
    return index, metadata
```

### Query Pipeline

```python
def search_knowledge_base(query: str, top_k: int = 3, similarity_threshold: float = 0.7):
    """
    Search knowledge base using semantic similarity
    """
    import faiss
    import openai
    import numpy as np
    import pickle

    # 1. Load index and metadata
    index = faiss.read_index("solution/data/core/faiss_index.bin")
    with open("solution/data/core/faiss_metadata.pkl", "rb") as f:
        metadata = pickle.load(f)

    # 2. Generate query embedding
    response = openai.embeddings.create(
        input=query,
        model="text-embedding-3-small"
    )
    query_embedding = np.array([response.data[0].embedding]).astype('float32')

    # 3. Search FAISS index
    distances, indices = index.search(query_embedding, top_k)

    # 4. Convert L2 distance to similarity score (0-1)
    # L2 distance: 0 = identical, higher = more different
    # Convert to similarity: higher = more similar
    max_distance = 2.0  # Empirically determined threshold
    similarities = 1 - (distances[0] / max_distance)
    similarities = np.clip(similarities, 0, 1)

    # 5. Filter by threshold and prepare results
    results = []
    for idx, sim in zip(indices[0], similarities):
        if sim >= similarity_threshold:
            results.append({
                'article_id': metadata[idx]['article_id'],
                'title': metadata[idx]['title'],
                'tags': metadata[idx]['tags'],
                'similarity': float(sim)
            })

    return results
```

### Integration with Resolver Agent

```python
class ResolverAgent:
    def resolve(self, query: str, classification: dict):
        # 1. Search knowledge base
        results = search_knowledge_base(
            query=query,
            top_k=3,
            similarity_threshold=0.7
        )

        if not results:
            return {
                "resolution": None,
                "confidence": 0.0,
                "needs_escalation": True,
                "reason": "No relevant knowledge articles found"
            }

        # 2. Retrieve full article content from database
        articles = []
        for result in results:
            article = get_article_by_id(result['article_id'])
            articles.append({
                'title': article['title'],
                'content': article['content'],
                'similarity': result['similarity']
            })

        # 3. Generate response using LLM with retrieved articles
        context = "\n\n---\n\n".join([
            f"Article: {a['title']}\n{a['content']}"
            for a in articles
        ])

        prompt = f"""
        You are a customer support agent for CultPass. Use the following knowledge articles to answer the customer's question.

        Knowledge Articles:
        {context}

        Customer Question:
        {query}

        Provide a helpful, accurate response based ONLY on the knowledge articles provided.
        If the articles don't contain enough information, say so clearly.
        """

        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful customer support agent."},
                {"role": "user", "content": prompt}
            ]
        )

        resolution_text = response.choices[0].message.content

        # 4. Calculate confidence score
        confidence = self._calculate_confidence(results, resolution_text)

        return {
            "resolution": resolution_text,
            "confidence": confidence,
            "sources": [a['title'] for a in articles],
            "needs_escalation": confidence < 0.5
        }

    def _calculate_confidence(self, search_results: list, resolution: str) -> float:
        """
        Calculate confidence score based on:
        - Top result similarity
        - Number of relevant results
        - Resolution contains uncertainty phrases
        """
        if not search_results:
            return 0.0

        # Base confidence from top result similarity
        top_similarity = search_results[0]['similarity']

        # Adjust for number of results (more results = more confidence)
        num_results_factor = min(len(search_results) / 3, 1.0)

        # Penalize uncertainty phrases in resolution
        uncertainty_phrases = [
            "I'm not sure",
            "might be",
            "possibly",
            "don't have enough information",
            "unclear"
        ]
        uncertainty_penalty = 0
        for phrase in uncertainty_phrases:
            if phrase.lower() in resolution.lower():
                uncertainty_penalty = 0.3
                break

        # Final confidence
        confidence = (top_similarity * 0.7 + num_results_factor * 0.3) - uncertainty_penalty
        return max(0.0, min(1.0, confidence))
```

## Similarity Thresholds

Based on empirical testing with CultPass knowledge base:

| Similarity Score | Interpretation | Action |
|-----------------|----------------|--------|
| 0.9 - 1.0 | Highly relevant | High confidence response |
| 0.8 - 0.9 | Very relevant | Confident response |
| 0.7 - 0.8 | Relevant | Cautious response |
| 0.5 - 0.7 | Somewhat relevant | Flag for review |
| 0.0 - 0.5 | Not relevant | Escalate |

**Default threshold: 0.7** (balances precision and recall)

## Query Enhancement

To improve retrieval quality:

### 1. Query Expansion
```python
def expand_query(query: str, classification: dict) -> str:
    """
    Expand user query with domain-specific terms
    """
    expansions = {
        'login': ['password', 'authentication', 'access', 'account'],
        'billing': ['payment', 'subscription', 'refund', 'charges'],
        'reservation': ['booking', 'event', 'experience', 'cancel']
    }

    issue_type = classification['ticket_type']
    if issue_type in expansions:
        expanded = f"{query} {' '.join(expansions[issue_type])}"
        return expanded
    return query
```

### 2. Reranking (Future Enhancement)
- Use cross-encoder model to rerank top 10 results
- More accurate but slower than bi-encoder (current approach)

## Index Maintenance

### When to Rebuild Index:
- New knowledge articles added
- Articles updated or deleted
- Index corruption detected
- Embedding model upgraded

### Incremental Updates:
```python
def add_article_to_index(article: dict):
    """
    Add single article to existing index
    """
    # Load existing index
    index = faiss.read_index("solution/data/core/faiss_index.bin")
    with open("solution/data/core/faiss_metadata.pkl", "rb") as f:
        metadata = pickle.load(f)

    # Generate embedding
    combined_text = f"{article['title']}\n\n{article['content']}\n\nTags: {article['tags']}"
    embedding = openai.embeddings.create(
        input=combined_text,
        model="text-embedding-3-small"
    ).data[0].embedding

    # Add to index
    index.add(np.array([embedding]).astype('float32'))

    # Update metadata
    metadata.append({
        'article_id': article['article_id'],
        'title': article['title'],
        'tags': article['tags']
    })

    # Save
    faiss.write_index(index, "solution/data/core/faiss_index.bin")
    with open("solution/data/core/faiss_metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)
```

## Performance Metrics

### Retrieval Metrics:
- **Precision@3**: % of top 3 results that are relevant
- **Recall@3**: % of all relevant articles in top 3
- **MRR (Mean Reciprocal Rank)**: 1 / rank of first relevant result
- **Latency**: Time from query to results (target: <100ms)

### End-to-end Metrics:
- **Resolution Rate**: % of tickets resolved without escalation
- **User Satisfaction**: Explicit feedback on resolutions
- **Human Override Rate**: % of times human changes RAG response

## Testing Strategy

### 1. Unit Tests (test_rag.py)
```python
def test_exact_match():
    """Test that exact title match returns similarity > 0.9"""
    query = "How to Reserve a Spot for an Event"
    results = search_knowledge_base(query, top_k=1)
    assert results[0]['similarity'] > 0.9
    assert "Reserve" in results[0]['title']

def test_semantic_match():
    """Test semantic understanding"""
    query = "I can't book a concert"  # Should match reservation article
    results = search_knowledge_base(query, top_k=3)
    assert any("Reserve" in r['title'] or "Reservation" in r['title']
               for r in results)

def test_no_match():
    """Test that irrelevant query returns empty or low similarity"""
    query = "What's the weather today?"  # Not in knowledge base
    results = search_knowledge_base(query, top_k=3, similarity_threshold=0.7)
    assert len(results) == 0 or results[0]['similarity'] < 0.5
```

### 2. Integration Tests
```python
def test_end_to_end_resolution():
    """Test full pipeline from query to resolution"""
    resolver = ResolverAgent()
    result = resolver.resolve(
        query="I forgot my password",
        classification={'ticket_type': 'login', 'urgency': 'medium'}
    )
    assert result['confidence'] > 0.7
    assert 'password' in result['resolution'].lower()
    assert len(result['sources']) > 0
```

### 3. Human Evaluation
- Sample 50 queries across different categories
- Compare RAG responses to ground truth (human-written responses)
- Measure accuracy, helpfulness, and tone

## Error Handling

### Index Not Found:
```python
try:
    index = faiss.read_index("solution/data/core/faiss_index.bin")
except Exception as e:
    print("Index not found, rebuilding...")
    build_faiss_index(load_knowledge_articles())
```

### OpenAI API Errors:
```python
try:
    embedding = openai.embeddings.create(...)
except openai.RateLimitError:
    # Exponential backoff
    time.sleep(2 ** retry_count)
except openai.APIError:
    # Fallback to keyword search
    return keyword_search_fallback(query)
```

### Empty Results:
```python
if not results:
    # Try relaxed threshold
    results = search_knowledge_base(query, similarity_threshold=0.5)
    if not results:
        return escalate_to_human("No relevant knowledge found")
```

## Future Enhancements

1. **Hybrid Search**: Combine semantic search with keyword (BM25) search
2. **Query Intent Detection**: Classify query type before search
3. **Multi-vector Retrieval**: Separate embeddings for title, content, tags
4. **Contextual Embeddings**: Include customer history in query embedding
5. **Active Learning**: Collect feedback to fine-tune retrieval
6. **Multilingual Support**: Translate query and articles for global support
7. **Real-time Updates**: Use streaming index updates instead of batch rebuild

## Cost Analysis

For 15 articles, indexing cost:
- Embedding generation: ~2K tokens × $0.02/1M tokens = $0.00004

For 1000 queries/day:
- Query embeddings: ~10K tokens × $0.02/1M tokens = $0.0002/day = $0.06/month

**Total**: ~$0.10/month for RAG system (negligible)

LLM calls for response generation dominate cost (~$5-20/month depending on volume).
