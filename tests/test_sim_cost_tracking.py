"""
Unit tests for SIM client cost tracking functionality.

This module tests that the SIM client properly tracks API costs,
integrates with the backpressure middleware for budget protection,
and correctly calculates costs per endpoint.
"""

import asyncio
import json
import os
import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from prometheus_client import REGISTRY, Counter

from backend.integrations.sim_client import SimClient, SimApiError
from backend.core.backpressure import BackpressureManager, TaskPriority, CircuitState
from backend.core.metrics import ApiMetrics, external_api_credit_used_total
from backend.core.redis_client import RedisClient, RedisDb, SerializationFormat


# --- Test Fixtures ---

@pytest.fixture
def mock_provider_registry():
    """Mock provider registry with SIM configuration."""
    sim_provider = {
        "id": "sim",
        "name": "SIM Blockchain API",
        "description": "Core data provider for real-time multi-chain blockchain data.",
        "connection_uri": "https://api.sim-blockchain.com/v1",
        "auth": {
            "api_key_env_var": "SIM_API_KEY"
        },
        "budget": {
            "monthly_usd": 100.00
        },
        "rate_limits": {
            "requests_per_minute": 60
        },
        "cost_rules": {
            "default_cost_per_request": 0.01,
            "endpoints": {
                "activity": 0.02,
                "balances": 0.01,
                "token-info": 0.01,
                "transactions": 0.05,
                "graph-data": 0.10
            }
        },
        "retry_policy": {
            "attempts": 3,
            "backoff_factor": 0.2
        }
    }
    
    with patch("backend.integrations.sim_client.get_provider", return_value=sim_provider):
        yield sim_provider


@pytest.fixture
def mock_redis_client():
    """Mock Redis client for backpressure tests."""
    mock_client = AsyncMock(spec=RedisClient)
    mock_client.get.return_value = None
    mock_client.set.return_value = True
    return mock_client


@pytest.fixture
def mock_backpressure_manager(mock_redis_client):
    """Mock backpressure manager for budget tests."""
    with patch("backend.core.backpressure.RedisClient", return_value=mock_redis_client):
        manager = BackpressureManager(redis_client=mock_redis_client)
        manager.provider_budgets = {"sim": 100.0}
        manager.provider_costs = {"sim": 0.0}
        manager.provider_rate_limits = {"sim": {"requests_per_minute": 60}}
        yield manager


@pytest.fixture
def mock_aiohttp_session():
    """Mock aiohttp ClientSession for API call tests."""
    mock_session = AsyncMock()
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.json.return_value = {"result": "success", "data": []}
    mock_response.__aenter__.return_value = mock_response
    mock_session.request.return_value = mock_response
    
    with patch("aiohttp.ClientSession", return_value=mock_session):
        yield mock_session, mock_response


@pytest.fixture
def sim_client(mock_provider_registry):
    """Create a SIM client with mocked provider config."""
    os.environ["SIM_API_KEY"] = "test_api_key"
    client = SimClient()
    yield client
    # Clean up
    if hasattr(client, "_session") and client._session:
        asyncio.run(client.close())


# --- Reset metrics between tests ---

@pytest.fixture(autouse=True)
def reset_metrics():
    """Reset Prometheus metrics between tests."""
    for collector in list(REGISTRY._collector_to_names.keys()):
        if isinstance(collector, Counter) and collector._name == "external_api_credit_used_total":
            collector._value = {}
    yield


# --- Tests ---

@pytest.mark.asyncio
async def test_sim_client_cost_tracking_basic(sim_client, mock_aiohttp_session):
    """Test that basic API calls emit cost metrics."""
    # Arrange
    mock_session, mock_response = mock_aiohttp_session
    
    # Create a mock for the ApiMetrics.record_api_cost method
    with patch("backend.core.metrics.ApiMetrics.record_api_cost") as mock_record_cost:
        # Act
        result = await sim_client.get_balances("0x123456789")
        
        # Assert
        assert result == {"result": "success", "data": []}
        mock_session.request.assert_called_once()
        mock_record_cost.assert_called_once_with("sim", "balances", 0.01)


@pytest.mark.asyncio
async def test_sim_client_different_endpoint_costs(sim_client, mock_aiohttp_session):
    """Test that different endpoints have different costs."""
    # Arrange
    mock_session, mock_response = mock_aiohttp_session
    
    # Create a dictionary to store called costs
    recorded_costs = {}
    
    def mock_record_cost(provider, endpoint, cost):
        recorded_costs[endpoint] = cost
    
    with patch("backend.core.metrics.ApiMetrics.record_api_cost", side_effect=mock_record_cost):
        # Act - Call different endpoints
        await sim_client.get_activity("0x123456789")
        await sim_client.get_balances("0x123456789")
        await sim_client.get_token_info("0xtoken")
        await sim_client.get_transactions(address="0x123456789")
        
        # Assert - Each endpoint should have its specific cost
        assert recorded_costs["activity"] == 0.02
        assert recorded_costs["balances"] == 0.01
        assert recorded_costs["token-info"] == 0.01
        assert recorded_costs["transactions"] == 0.05


@pytest.mark.asyncio
async def test_sim_client_prometheus_metrics(sim_client, mock_aiohttp_session):
    """Test that API calls update Prometheus metrics."""
    # Arrange
    mock_session, mock_response = mock_aiohttp_session
    
    # Act
    await sim_client.get_activity("0x123456789")
    
    # Assert - Check that the Prometheus counter was incremented
    # We need to get the metric from the registry and check its value
    metric_value = 0
    for sample in REGISTRY.get_sample_value("external_api_credit_used_total", {"provider": "sim", "credit_type": "activity"}):
        metric_value = sample[2]
    
    assert metric_value == 0.02


@pytest.mark.asyncio
async def test_backpressure_budget_check(mock_backpressure_manager):
    """Test that budget checks work in the backpressure manager."""
    # Arrange
    manager = mock_backpressure_manager
    manager.provider_costs["sim"] = 0.0  # Reset costs
    
    # Act & Assert - With low cost, should be allowed
    can_proceed, error = await manager.check_budget("sim", 1.0)
    assert can_proceed is True
    assert error is None
    
    # Act & Assert - With cost exceeding budget, should be denied
    can_proceed, error = await manager.check_budget("sim", 101.0)
    assert can_proceed is False
    assert "Budget exceeded" in error


@pytest.mark.asyncio
async def test_backpressure_emergency_protection(mock_backpressure_manager):
    """Test emergency budget protection in backpressure manager."""
    # Arrange
    manager = mock_backpressure_manager
    # Set current cost to 95% of budget (emergency threshold)
    manager.provider_costs["sim"] = 95.0
    
    # Act & Assert - Small cost should be denied due to emergency threshold
    can_proceed, error = await manager.check_budget("sim", 1.0)
    assert can_proceed is False
    assert "Emergency budget threshold" in error


@pytest.mark.asyncio
async def test_sim_client_with_backpressure(sim_client, mock_aiohttp_session, mock_backpressure_manager):
    """Test integration between SIM client and backpressure middleware."""
    # Arrange
    mock_session, mock_response = mock_aiohttp_session
    manager = mock_backpressure_manager
    
    # Set current cost to 99% of budget
    manager.provider_costs["sim"] = 99.0
    
    # Mock the process_request method to use our mock manager
    with patch("backend.core.backpressure.BackpressureManager.process_request", 
               side_effect=manager.process_request) as mock_process:
        
        # Act - This should be denied due to budget
        can_proceed, error, task_id = await manager.process_request(
            provider_id="sim",
            endpoint="graph-data",
            params={"address": "0x123456789"},
            priority=TaskPriority.NORMAL
        )
        
        # Assert
        assert can_proceed is False
        assert "Budget exceeded" in error
        assert task_id is not None  # Task should be queued


@pytest.mark.asyncio
async def test_sim_client_rate_limiting(sim_client, mock_aiohttp_session):
    """Test rate limiting scenarios."""
    # Arrange
    mock_session, mock_response = mock_aiohttp_session
    
    # Simulate rate limiting response
    mock_response.status = 429
    mock_response.headers = {"Retry-After": "1"}
    
    # Mock sleep to avoid actual waiting
    with patch("asyncio.sleep", return_value=None) as mock_sleep:
        # Act
        await sim_client.get_balances("0x123456789")
        
        # Assert
        assert mock_sleep.called
        assert mock_sleep.call_args[0][0] == 1  # Should sleep for 1 second


@pytest.mark.asyncio
async def test_sim_client_retry_logic(sim_client, mock_aiohttp_session):
    """Test retry logic for failed requests."""
    # Arrange
    mock_session, mock_response = mock_aiohttp_session
    
    # Simulate server error then success
    mock_response.status = 500
    
    # Set up side effect to return error first, then success
    response_success = AsyncMock()
    response_success.status = 200
    response_success.json.return_value = {"result": "success after retry"}
    response_success.__aenter__.return_value = response_success
    
    mock_session.request.side_effect = [mock_response, response_success]
    
    # Mock sleep to avoid actual waiting
    with patch("asyncio.sleep", return_value=None) as mock_sleep:
        # Act
        result = await sim_client.get_balances("0x123456789")
        
        # Assert
        assert mock_sleep.called  # Should have slept between retries
        assert mock_session.request.call_count == 2  # Should have retried once
        assert result == {"result": "success after retry"}


@pytest.mark.asyncio
async def test_sim_client_circuit_breaker(mock_backpressure_manager):
    """Test circuit breaker functionality in backpressure manager."""
    # Arrange
    manager = mock_backpressure_manager
    
    # Act - Record multiple failures to trigger circuit breaker
    for _ in range(5):  # Assuming threshold is 5
        await manager.record_failure("sim", "API error")
    
    # Check circuit breaker state
    can_proceed, error = await manager.check_circuit_breaker("sim")
    
    # Assert
    assert can_proceed is False
    assert "Circuit breaker is OPEN" in error


@pytest.mark.asyncio
async def test_enqueue_task_when_budget_exceeded(mock_backpressure_manager, mock_redis_client):
    """Test that tasks are enqueued when budget is exceeded."""
    # Arrange
    manager = mock_backpressure_manager
    manager.provider_costs["sim"] = 99.0  # Near budget limit
    
    # Act
    can_proceed, error, task_id = await manager.process_request(
        provider_id="sim",
        endpoint="graph-data",
        params={"address": "0x123456789"},
        priority=TaskPriority.NORMAL
    )
    
    # Assert
    assert can_proceed is False
    assert task_id is not None
    # Verify task was enqueued in Redis
    mock_redis_client.set.assert_called_once()
    # The first arg should be the Redis key
    assert mock_redis_client.set.call_args[0][0].startswith("backpressure:queue:sim")


@pytest.mark.asyncio
async def test_critical_priority_bypasses_budget(mock_backpressure_manager):
    """Test that critical priority tasks can bypass budget restrictions."""
    # Arrange
    manager = mock_backpressure_manager
    manager.provider_costs["sim"] = 99.0  # Near budget limit
    
    # Act - Normal priority should be denied
    can_proceed_normal, _, _ = await manager.process_request(
        provider_id="sim",
        endpoint="balances",
        params={"address": "0x123456789"},
        priority=TaskPriority.NORMAL
    )
    
    # Critical priority should be allowed
    can_proceed_critical, _, _ = await manager.process_request(
        provider_id="sim",
        endpoint="balances",
        params={"address": "0x123456789"},
        priority=TaskPriority.CRITICAL
    )
    
    # Assert
    assert can_proceed_normal is False
    assert can_proceed_critical is True


@pytest.mark.asyncio
async def test_cost_accumulation(mock_backpressure_manager):
    """Test that costs accumulate correctly in the backpressure manager."""
    # Arrange
    manager = mock_backpressure_manager
    manager.provider_costs["sim"] = 0.0  # Reset costs
    
    # Act - Record multiple API costs
    await manager.record_success("sim", 0.01)
    await manager.record_success("sim", 0.02)
    await manager.record_success("sim", 0.05)
    
    # Assert
    assert manager.provider_costs["sim"] == 0.08  # 0.01 + 0.02 + 0.05


@pytest.mark.asyncio
async def test_sim_client_error_handling(sim_client, mock_aiohttp_session):
    """Test error handling in SIM client."""
    # Arrange
    mock_session, mock_response = mock_aiohttp_session
    mock_response.status = 400
    mock_response.text = AsyncMock(return_value='{"error": "Invalid request"}')
    
    # Act & Assert
    with pytest.raises(SimApiError) as exc_info:
        await sim_client.get_balances("0x123456789")
    
    assert "SIM API request failed: 400" in str(exc_info.value)
