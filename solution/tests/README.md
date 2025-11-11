# UDA-Hub Test Suite

This directory contains comprehensive unit and integration tests for the UDA-Hub system.

## Test Files

### Unit Tests

- **test_classifier.py**: Tests for the Classifier Agent
  - Ticket type classification
  - Urgency detection
  - Sentiment analysis
  - Output structure validation

- **test_tools.py**: Tests for CultPass tools and database operations
  - User lookup functionality
  - Subscription management
  - Reservation operations
  - Refund processing
  - Error handling

### Integration Tests

- **test_workflow.py**: End-to-end workflow tests
  - Complete ticket processing pipeline
  - Multi-agent coordination
  - Session memory
  - Decision logging
  - Escalation scenarios

## Running Tests

### Prerequisites

```bash
# Install test dependencies
pip install pytest pytest-asyncio

# Ensure OpenAI API key is set (required for LLM tests)
export OPENAI_API_KEY=your_key_here
```

### Run All Tests

```bash
# From the solution/ directory
pytest tests/ -v
```

### Run Specific Test Files

```bash
# Unit tests only
pytest tests/test_classifier.py -v
pytest tests/test_tools.py -v

# Integration tests only
pytest tests/test_workflow.py -v -m integration
```

### Run with Coverage

```bash
pytest tests/ --cov=agentic --cov-report=html
```

## Test Categories

Tests are marked with pytest markers:

- `@pytest.mark.integration`: Integration tests (require full system)
- No marker: Unit tests (test individual components)

## Expected Behavior

### Unit Tests
- Should run quickly (<30 seconds total)
- Test individual components in isolation
- Use mock data where appropriate
- Should pass without external dependencies (except OpenAI API)

### Integration Tests
- May take longer (1-2 minutes)
- Require databases to be set up
- Test complete workflows
- Validate agent coordination

## Test Coverage Goals

- **Agents**: >80% coverage
  - All classification paths
  - All routing decisions
  - Error handling

- **Tools**: >90% coverage
  - All database operations
  - Success and failure cases
  - Edge cases (missing data, blocked users)

- **Workflow**: >70% coverage
  - All routing paths
  - Escalation scenarios
  - Resolution scenarios

## Known Limitations

1. **LLM Variability**: Tests involving LLMs may have slight variations in output
2. **API Dependencies**: Tests require OpenAI API access
3. **Database State**: Some tests create temporary databases

## Adding New Tests

When adding new tests:

1. Follow existing naming conventions (`test_*.py`)
2. Use descriptive test names (`test_classifier_handles_urgent_tickets`)
3. Add docstrings explaining what is being tested
4. Mark integration tests appropriately
5. Clean up resources (databases, files) in teardown

## Continuous Integration

These tests are designed to run in CI/CD pipelines:

```yaml
# Example GitHub Actions
- name: Run tests
  run: |
    pytest tests/ -v --junitxml=test-results.xml
```

## Troubleshooting

### Tests Failing Due to Missing Database
```bash
# Run database setup notebooks first
jupyter nbconvert --execute --to notebook 01_external_db_setup.ipynb
jupyter nbconvert --execute --to notebook 02_core_db_setup.ipynb
```

### Tests Failing Due to Missing FAISS Index
```bash
# Build the FAISS index in the setup notebook
# Or skip RAG tests temporarily
pytest tests/ -v -k "not rag"
```

### API Rate Limiting
```bash
# Run tests with delays between API calls
pytest tests/ --durations=10 -v
```
