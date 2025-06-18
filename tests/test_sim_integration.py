"""
Tests for Sim API integration with the Analyst Augmentation Agent.

This module contains comprehensive tests for the Sim API client, tools, and
integration with the application's workflow. It includes both mock tests and
optional real API tests when credentials are available.
"""

import os
import json
import pytest
from unittest.mock import patch, MagicMock, ANY
from datetime import datetime

# Import Sim client and tools
from backend.integrations.sim_client import SimClient, SimApiError
from backend.agents.tools.sim_balances_tool import SimBalancesTool
from backend.agents.tools.sim_activity_tool import SimActivityTool

# Import event system for testing graph events
from backend.core.events import GraphAddEvent

# Test constants
TEST_WALLET_ETH = "0xd8da6bf26964af9d7eed9e03e53415d37aa96045"  # vitalik.eth
TEST_WALLET_SOL = "DYw8jCTfwHNRJhhmFcbXvVDTqWMEVFBX6ZKUmG5CNSKK"
TEST_TOKEN_ADDRESS = "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"  # USDC on Ethereum
TEST_CHAIN_ID = 1  # Ethereum mainnet

# Flag to determine if real API tests should run
REAL_API_TESTS = os.environ.get("SIM_API_KEY") is not None


# ======= Fixtures =======

@pytest.fixture
def mock_balances_response():
    """Fixture providing a mock response for the balances endpoint."""
    return {
        "balances": [
            {
                "address": "native",
                "amount": "605371497350928252303",
                "chain": "ethereum",
                "chain_id": 1,
                "decimals": 18,
                "price_usd": 3042.82,
                "symbol": "ETH",
                "value_usd": 1842034.66,
                "token_metadata": {
                    "symbol": "ETH",
                    "name": "Ethereum",
                    "decimals": 18,
                    "logo": "https://example.com/eth.png"
                }
            },
            {
                "address": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
                "amount": "1000000000",
                "chain": "ethereum",
                "chain_id": 1,
                "decimals": 6,
                "price_usd": 1.0,
                "symbol": "USDC",
                "value_usd": 1000.0,
                "token_metadata": {
                    "symbol": "USDC",
                    "name": "USD Coin",
                    "decimals": 6,
                    "logo": "https://example.com/usdc.png"
                }
            }
        ],
        "wallet_address": TEST_WALLET_ETH,
        "next_offset": "abc123",
        "request_time": "2025-06-18T10:00:00Z",
        "response_time": "2025-06-18T10:00:01Z"
    }


@pytest.fixture
def mock_activity_response():
    """Fixture providing a mock response for the activity endpoint."""
    return {
        "activity": [
            {
                "id": "tx1",
                "type": "send",
                "chain": "ethereum",
                "chain_id": 1,
                "block_number": 18000000,
                "block_time": "2025-06-17T15:30:00Z",
                "transaction_hash": "0x123abc",
                "from_address": TEST_WALLET_ETH,
                "to_address": "0x742d35Cc6634C0532925a3b844Bc454e4438f44e",
                "asset_type": "native",
                "amount": "1000000000000000000",
                "value": "1000000000000000000",
                "value_usd": 3042.82,
                "token_metadata": {
                    "symbol": "ETH",
                    "name": "Ethereum",
                    "decimals": 18
                }
            },
            {
                "id": "tx2",
                "type": "receive",
                "chain": "ethereum",
                "chain_id": 1,
                "block_number": 17999900,
                "block_time": "2025-06-17T14:30:00Z",
                "transaction_hash": "0x456def",
                "from_address": "0x742d35Cc6634C0532925a3b844Bc454e4438f44e",
                "to_address": TEST_WALLET_ETH,
                "asset_type": "erc20",
                "token_address": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
                "amount": "1000000000",
                "value": "1000000000",
                "value_usd": 1000.0,
                "token_metadata": {
                    "symbol": "USDC",
                    "name": "USD Coin",
                    "decimals": 6
                }
            },
            {
                "id": "tx3",
                "type": "call",
                "chain": "ethereum",
                "chain_id": 1,
                "block_number": 17999800,
                "block_time": "2025-06-17T13:30:00Z",
                "transaction_hash": "0x789ghi",
                "from_address": TEST_WALLET_ETH,
                "to_address": "0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D",
                "function": {
                    "name": "swapExactTokensForETH",
                    "signature": "swapExactTokensForETH(uint256,uint256,address[],address,uint256)",
                    "parameters": [
                        {
                            "name": "amountIn",
                            "type": "uint256",
                            "value": "1000000000"
                        },
                        {
                            "name": "amountOutMin",
                            "type": "uint256",
                            "value": "900000000000000000"
                        }
                    ]
                }
            }
        ],
        "wallet_address": TEST_WALLET_ETH,
        "next_offset": "def456",
        "request_time": "2025-06-18T10:00:00Z",
        "response_time": "2025-06-18T10:00:01Z"
    }


@pytest.fixture
def mock_token_info_response():
    """Fixture providing a mock response for the token info endpoint."""
    return {
        "tokens": [
            {
                "address": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
                "chain": "ethereum",
                "chain_id": 1,
                "decimals": 6,
                "name": "USD Coin",
                "symbol": "USDC",
                "price_usd": 1.0,
                "total_supply": "45738409759336",
                "circulating_supply": "45738409759336",
                "pool_size": 500000000.0,
                "low_liquidity": False,
                "logo": "https://example.com/usdc.png",
                "url": "https://www.circle.com/usdc"
            }
        ],
        "next_offset": None,
        "request_time": "2025-06-18T10:00:00Z",
        "response_time": "2025-06-18T10:00:01Z"
    }


@pytest.fixture
def mock_supported_chains_response():
    """Fixture providing a mock response for the supported chains endpoint."""
    return {
        "chains": [
            {
                "name": "ethereum",
                "chain_id": 1,
                "tags": ["mainnet", "l1"],
                "balances": {"supported": True},
                "activity": {"supported": True},
                "collectibles": {"supported": True},
                "transactions": {"supported": True},
                "token_info": {"supported": True},
                "token_holders": {"supported": True}
            },
            {
                "name": "optimism",
                "chain_id": 10,
                "tags": ["l2", "optimism"],
                "balances": {"supported": True},
                "activity": {"supported": True},
                "collectibles": {"supported": True},
                "transactions": {"supported": True},
                "token_info": {"supported": True},
                "token_holders": {"supported": True}
            }
        ]
    }


@pytest.fixture
def mock_sim_client():
    """Fixture providing a mocked Sim client."""
    with patch('backend.integrations.sim_client.SimClient') as mock_client:
        client_instance = mock_client.return_value
        client_instance.api_key = "test_api_key"
        client_instance.base_url = "https://api.sim.dune.com"
        yield client_instance


@pytest.fixture
def mock_requests():
    """Fixture providing a mocked requests module."""
    with patch('backend.integrations.sim_client.requests') as mock_req:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {}
        mock_req.request.return_value = mock_response
        mock_req.get.return_value = mock_response
        mock_req.post.return_value = mock_response
        yield mock_req


@pytest.fixture
def mock_emit_event():
    """Fixture providing a mocked emit_event function."""
    with patch('backend.agents.tools.sim_balances_tool.emit_event') as mock_emit:
        yield mock_emit


# ======= Unit Tests: Sim Client =======

class TestSimClient:
    """Tests for the Sim API client."""

    def test_client_initialization(self):
        """Test that the client initializes correctly with API key and base URL."""
        client = SimClient(api_key="test_key", base_url="https://test.api.com")
        assert client.api_key == "test_key"
        assert client.base_url == "https://test.api.com"
        assert client.headers["X-Sim-Api-Key"] == "test_key"
        assert client.headers["Content-Type"] == "application/json"

    def test_client_initialization_missing_key(self):
        """Test that the client raises an error when API key is missing."""
        with patch('backend.integrations.sim_client.settings', MagicMock(SIM_API_KEY=None)):
            with pytest.raises(ValueError, match="Sim API key is required"):
                SimClient()

    def test_request_success(self, mock_requests, mock_balances_response):
        """Test successful API request."""
        mock_requests.request.return_value.json.return_value = mock_balances_response
        
        client = SimClient(api_key="test_key")
        result = client._request("GET", "/v1/evm/balances/test")
        
        assert result == mock_balances_response
        mock_requests.request.assert_called_once_with(
            method="GET",
            url="https://api.sim.dune.com/v1/evm/balances/test",
            headers=client.headers,
            params=None,
            json=None,
            timeout=30
        )

    def test_request_http_error(self, mock_requests):
        """Test handling of HTTP errors."""
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.json.return_value = {
            "error": {
                "message": "Bad request",
                "code": "BAD_REQUEST"
            }
        }
        mock_requests.request.return_value = mock_response
        
        client = SimClient(api_key="test_key")
        with pytest.raises(SimApiError) as excinfo:
            client._request("GET", "/v1/evm/balances/test")
        
        assert "Sim API error: 400 - Bad request" in str(excinfo.value)
        assert excinfo.value.status_code == 400
        assert excinfo.value.error_code == "BAD_REQUEST"

    def test_request_rate_limit(self, mock_requests):
        """Test handling of rate limit errors."""
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.json.return_value = {
            "error": {
                "message": "Rate limit exceeded",
                "code": "RATE_LIMIT_EXCEEDED"
            }
        }
        mock_response.headers = {
            'X-Rate-Limit-Remaining': '0',
            'X-Rate-Limit-Reset': str(int(datetime.now().timestamp()) + 60)
        }
        mock_requests.request.return_value = mock_response
        
        client = SimClient(api_key="test_key")
        with pytest.raises(SimApiError) as excinfo:
            client._request("GET", "/v1/evm/balances/test")
        
        assert "Rate limit exceeded" in str(excinfo.value)
        assert excinfo.value.status_code == 429
        assert excinfo.value.error_code == "RATE_LIMIT_EXCEEDED"
        assert client.rate_limit_remaining == 0

    def test_get_balances(self, mock_sim_client, mock_balances_response):
        """Test get_balances method."""
        mock_sim_client._request.return_value = mock_balances_response
        
        result = mock_sim_client.get_balances(
            wallet=TEST_WALLET_ETH,
            limit=100,
            chain_ids="1,10",
            metadata="url,logo"
        )
        
        mock_sim_client._request.assert_called_once_with(
            "GET",
            f"/v1/evm/balances/{TEST_WALLET_ETH}",
            params={"limit": 100, "chain_ids": "1,10", "metadata": "url,logo"}
        )
        assert result == mock_balances_response

    def test_get_activity(self, mock_sim_client, mock_activity_response):
        """Test get_activity method."""
        mock_sim_client._request.return_value = mock_activity_response
        
        result = mock_sim_client.get_activity(
            wallet=TEST_WALLET_ETH,
            limit=25
        )
        
        mock_sim_client._request.assert_called_once_with(
            "GET",
            f"/v1/evm/activity/{TEST_WALLET_ETH}",
            params={"limit": 25}
        )
        assert result == mock_activity_response

    def test_get_token_info(self, mock_sim_client, mock_token_info_response):
        """Test get_token_info method."""
        mock_sim_client._request.return_value = mock_token_info_response
        
        result = mock_sim_client.get_token_info(
            token_address=TEST_TOKEN_ADDRESS,
            chain_ids="all"
        )
        
        mock_sim_client._request.assert_called_once_with(
            "GET",
            f"/v1/evm/token-info/{TEST_TOKEN_ADDRESS}",
            params={"chain_ids": "all"}
        )
        assert result == mock_token_info_response

    def test_get_supported_chains(self, mock_sim_client, mock_supported_chains_response):
        """Test get_supported_chains method."""
        mock_sim_client._request.return_value = mock_supported_chains_response
        
        result = mock_sim_client.get_supported_chains()
        
        mock_sim_client._request.assert_called_once_with(
            "GET",
            "/v1/evm/supported-chains",
            params=None
        )
        assert result == mock_supported_chains_response


# ======= Unit Tests: Sim Tools =======

class TestSimBalancesTool:
    """Tests for the Sim Balances Tool."""

    def test_tool_initialization(self):
        """Test that the tool initializes correctly."""
        with patch('backend.agents.tools.sim_balances_tool.settings', MagicMock(SIM_API_KEY="test_key", SIM_API_URL="https://test.api.com")):
            tool = SimBalancesTool()
            assert tool.name == "sim_balances_tool"
            assert "Fetches token balances" in tool.description
            assert tool.api_key == "test_key"
            assert tool.api_url == "https://test.api.com"

    def test_run_success(self, mock_requests, mock_balances_response, mock_emit_event):
        """Test successful execution of the tool."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_balances_response
        mock_requests.get.return_value = mock_response
        
        with patch('backend.agents.tools.sim_balances_tool.settings', MagicMock(SIM_API_KEY="test_key", SIM_API_URL="https://test.api.com")):
            tool = SimBalancesTool()
            result = tool.run(wallet=TEST_WALLET_ETH, limit=100, chains="all", metadata="url,logo")
            
            assert result["wallet_address"] == TEST_WALLET_ETH
            assert len(result["balances"]) == 2
            assert result["balances"][0]["symbol"] == "ETH"
            assert result["balances"][1]["symbol"] == "USDC"
            
            # Verify graph events were emitted
            mock_emit_event.assert_called_once()
            event_arg = mock_emit_event.call_args[0][0]
            assert isinstance(event_arg, GraphAddEvent)
            assert event_arg.type == "wallet_balances"
            assert event_arg.data["wallet"] == TEST_WALLET_ETH

    def test_run_invalid_wallet(self):
        """Test that the tool validates wallet address."""
        tool = SimBalancesTool()
        with pytest.raises(ValueError, match="Wallet address must be a non-empty string"):
            tool.run(wallet="")

    def test_run_api_error(self, mock_requests):
        """Test handling of API errors."""
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.json.return_value = {
            "error": {
                "message": "Wallet not found",
                "code": "NOT_FOUND"
            }
        }
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("404 Client Error")
        mock_requests.get.return_value = mock_response
        
        with patch('backend.agents.tools.sim_balances_tool.settings', MagicMock(SIM_API_KEY="test_key", SIM_API_URL="https://test.api.com")):
            tool = SimBalancesTool()
            with pytest.raises(Exception, match="Error fetching balances"):
                tool.run(wallet=TEST_WALLET_ETH)


class TestSimActivityTool:
    """Tests for the Sim Activity Tool."""

    def test_tool_initialization(self):
        """Test that the tool initializes correctly."""
        with patch('backend.agents.tools.sim_activity_tool.settings', MagicMock(SIM_API_KEY="test_key", SIM_API_URL="https://test.api.com")):
            tool = SimActivityTool()
            assert tool.name == "sim_activity_tool"
            assert "Fetches transaction activity" in tool.description
            assert tool.api_key == "test_key"
            assert tool.api_url == "https://test.api.com"

    def test_run_success(self, mock_requests, mock_activity_response, mock_emit_event):
        """Test successful execution of the tool."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_activity_response
        mock_requests.get.return_value = mock_response
        
        with patch('backend.agents.tools.sim_activity_tool.settings', MagicMock(SIM_API_KEY="test_key", SIM_API_URL="https://test.api.com")):
            with patch('backend.agents.tools.sim_activity_tool.emit_event') as mock_emit:
                tool = SimActivityTool()
                result = tool.run(wallet=TEST_WALLET_ETH, limit=25)
                
                assert result["wallet_address"] == TEST_WALLET_ETH
                assert len(result["activity"]) == 3
                assert result["activity"][0]["type"] == "send"
                assert result["activity"][1]["type"] == "receive"
                assert result["activity"][2]["type"] == "call"
                
                # Verify graph events were emitted (one per activity)
                assert mock_emit.call_count == 3
                for call_args in mock_emit.call_args_list:
                    event_arg = call_args[0][0]
                    assert isinstance(event_arg, GraphAddEvent)
                    assert event_arg.type == "wallet_activity"
                    assert "wallet_address" in event_arg.data
                    assert event_arg.data["wallet_address"] == TEST_WALLET_ETH

    def test_run_invalid_wallet(self):
        """Test that the tool validates wallet address."""
        tool = SimActivityTool()
        with pytest.raises(ValueError, match="Wallet address must be a non-empty string"):
            tool.run(wallet="")


# ======= Integration Tests =======

@pytest.mark.integration
class TestSimIntegration:
    """Integration tests for Sim API tools with the application workflow."""

    def test_balances_to_graph_flow(self, mock_requests, mock_balances_response):
        """Test the complete flow from balances API call to graph event emission."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_balances_response
        mock_requests.get.return_value = mock_response
        
        with patch('backend.agents.tools.sim_balances_tool.emit_event') as mock_emit:
            with patch('backend.agents.tools.sim_balances_tool.settings', MagicMock(SIM_API_KEY="test_key", SIM_API_URL="https://test.api.com")):
                # Execute the tool
                tool = SimBalancesTool()
                result = tool.run(wallet=TEST_WALLET_ETH)
                
                # Verify API call
                mock_requests.get.assert_called_once()
                assert TEST_WALLET_ETH in mock_requests.get.call_args[0][0]
                
                # Verify result
                assert result["wallet_address"] == TEST_WALLET_ETH
                assert len(result["balances"]) == 2
                
                # Verify graph event
                mock_emit.assert_called_once()
                event_arg = mock_emit.call_args[0][0]
                assert isinstance(event_arg, GraphAddEvent)
                assert event_arg.type == "wallet_balances"
                assert "wallet" in event_arg.data
                assert "balances" in event_arg.data
                assert "timestamp" in event_arg.data
                assert len(event_arg.data["balances"]) == 2

    @patch('backend.core.events.emit_event')
    @patch('backend.agents.tools.sim_activity_tool.settings', MagicMock(SIM_API_KEY="test_key", SIM_API_URL="https://test.api.com"))
    def test_activity_graph_edge_creation(self, mock_emit, mock_requests, mock_activity_response):
        """Test that activity events create appropriate graph edges."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_activity_response
        mock_requests.get.return_value = mock_response
        
        # Execute the tool
        tool = SimActivityTool()
        result = tool.run(wallet=TEST_WALLET_ETH)
        
        # Verify different edge types were created
        edge_types = []
        for call_args in mock_emit.call_args_list:
            event_arg = call_args[0][0]
            edge_types.append(event_arg.data["edge_type"])
        
        assert "SEND" in edge_types
        assert "RECEIVE" in edge_types
        assert "CALL" in edge_types


# ======= Real API Tests (Optional) =======

@pytest.mark.skipif(not REAL_API_TESTS, reason="SIM_API_KEY not available")
class TestRealSimApi:
    """Tests that run against the real Sim API when credentials are available."""

    def test_real_balances(self):
        """Test fetching real balances from the Sim API."""
        client = SimClient()
        result = client.get_balances(wallet=TEST_WALLET_ETH, limit=5)
        
        assert "balances" in result
        assert isinstance(result["balances"], list)
        assert len(result["balances"]) > 0
        assert "wallet_address" in result
        assert result["wallet_address"] == TEST_WALLET_ETH

    def test_real_activity(self):
        """Test fetching real activity from the Sim API."""
        client = SimClient()
        result = client.get_activity(wallet=TEST_WALLET_ETH, limit=5)
        
        assert "activity" in result
        assert isinstance(result["activity"], list)
        assert len(result["activity"]) > 0
        assert "wallet_address" in result
        assert result["wallet_address"] == TEST_WALLET_ETH

    def test_real_token_info(self):
        """Test fetching real token info from the Sim API."""
        client = SimClient()
        result = client.get_token_info(token_address=TEST_TOKEN_ADDRESS, chain_ids="1")
        
        assert "tokens" in result
        assert isinstance(result["tokens"], list)
        assert len(result["tokens"]) > 0
        assert result["tokens"][0]["address"] == TEST_TOKEN_ADDRESS.lower()
        assert result["tokens"][0]["chain_id"] == 1

    def test_real_supported_chains(self):
        """Test fetching real supported chains from the Sim API."""
        client = SimClient()
        result = client.get_supported_chains()
        
        assert "chains" in result
        assert isinstance(result["chains"], list)
        assert len(result["chains"]) > 0
        # Ethereum should always be supported
        ethereum = next((chain for chain in result["chains"] if chain["chain_id"] == 1), None)
        assert ethereum is not None
        assert ethereum["name"] == "ethereum"
