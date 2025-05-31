"""
End-to-end integration tests for the Analyst Agent.

This module tests the complete flow from API endpoints through to metrics collection,
ensuring that the system works correctly as a whole.
"""

from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock

from backend.main import app
from backend.core.metrics import registry, llm_tokens_used_total
from backend.auth.dependencies import get_current_user

client = TestClient(app)

# Sample graph result to mimic fraud investigation output
_FAKE_GRAPH = {
    "nodes": [
        {"id": "1", "labels": ["Person"], "properties": {"name": "Alice", "risk_score": 0.7}},
        {"id": "2", "labels": ["Account"], "properties": {"balance": 10000}},
    ],
    "relationships": [
        {"id": "r1", "type": "OWNS", "startNode": "1", "endNode": "2", "properties": {}}
    ]
}


async def _fake_run_crew(*args, **kwargs):  # pylint: disable=unused-argument
    """Return deterministic fake result and simulate token usage."""
    return {
        "success": True,
        "result": {
            "graph_data": _FAKE_GRAPH,
            "fraud_patterns": [],
            "risk_score": 42,
        },
        "llm_usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "model": "models/gemini-1.5-flash-latest",
        },
    }


# Mock the auth dependency to bypass authentication
async def override_get_current_user():
    """Mock authenticated user for testing."""
    return {"username": "testuser", "is_active": True}


@patch("backend.agents.factory.CrewFactory.connect", new_callable=AsyncMock)
@patch("backend.agents.factory.CrewFactory.run_crew", side_effect=_fake_run_crew)
@patch("backend.auth.dependencies.get_current_user", override_get_current_user)
def test_crew_run_and_metrics(mock_run, mock_connect, _):  # noqa: D103  # pylint: disable=unused-argument
    """
    Test that crew execution works and updates Prometheus metrics.
    
    This test verifies:
    1. The /api/v1/crew/run endpoint returns successful results
    2. The Prometheus LLM token counters increment appropriately
    """
    # Capture counters before call
    before_prompt = llm_tokens_used_total.labels(model="models/gemini-1.5-flash-latest", type="input")._value.get()  # type: ignore  # pylint: disable=protected-access
    before_completion = llm_tokens_used_total.labels(model="models/gemini-1.5-flash-latest", type="output")._value.get()  # type: ignore  # pylint: disable=protected-access

    response = client.post(
        "/api/v1/crew/run",
        json={"crew_name": "fraud_investigation", "inputs": {}, "async_execution": False},
        headers={"Authorization": "Bearer test"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is True
    assert payload["result"]["graph_data"]["nodes"], "Graph nodes missing"
    assert payload["result"]["risk_score"] == 42, "Risk score incorrect"

    # Ensure factory methods invoked
    mock_connect.assert_awaited_once()
    mock_run.assert_awaited_once()

    # Check Prometheus token counter incremented
    after_prompt = llm_tokens_used_total.labels(model="models/gemini-1.5-flash-latest", type="input")._value.get()  # type: ignore  # pylint: disable=protected-access
    after_completion = llm_tokens_used_total.labels(model="models/gemini-1.5-flash-latest", type="output")._value.get()  # type: ignore  # pylint: disable=protected-access
    
    assert after_prompt > before_prompt, "Input token counter did not increase"
    assert after_completion > before_completion, "Output token counter did not increase"

    # Registry exposable
    metrics_text = registry.collect()  # ensure no crash
    assert any(m.name == "llm_tokens_used_total" for m in metrics_text)
