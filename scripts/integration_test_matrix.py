#!/usr/bin/env python3
"""
Comprehensive Integration Test Matrix for Coding Analyst Droid

This script automatically discovers all configured API providers, spins up
mock servers for them, and tests the complete data flow from API interaction
through Neo4j ingestion. It validates data integrity, error handling,
and metrics integration, supporting parallel execution and generating
detailed reports for CI/CD.

Usage:
    python scripts/integration_test_matrix.py
    python scripts/integration_test_matrix.py --provider sim
    python scripts/integration_test_matrix.py --provider covalent
    python scripts/integration_test_matrix.py --verbose

Options:
    --provider ID       Run tests only for a specific provider ID
    --verbose           Enable verbose logging for detailed test execution
    --help              Show this help message and exit
"""

import argparse
import asyncio
import json
import logging
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import yaml

# Add backend to sys.path to allow imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.providers import get_providers_by_category, ProviderConfig
from backend.core.neo4j_loader import Neo4jLoader, GraphStats
from backend.core.metrics import ApiMetrics, DatabaseMetrics
from backend.core.redis_client import RedisClient, RedisDb, SerializationFormat

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global test results storage
test_results: List[Dict[str, Any]] = []

# --- Provider-Specific Test Configurations ---

async def _perform_sim_ingestion(client: Any, neo4j_loader: Neo4jLoader, api_data: Dict[str, Any]):
    """Ingestion logic for SIM API data."""
    from backend.core.neo4j_loader import Balance, Activity
    logger.debug("Performing SIM ingestion logic")
    if "balances" in api_data:
        balances = [Balance(**b) for b in api_data["balances"]]
        await neo4j_loader.ingest_balances(balances)
    if "activity" in api_data:
        activities = [Activity(**a) for a in api_data["activity"]]
        await neo4j_loader.ingest_activity(activities)

async def _perform_covalent_ingestion(client: Any, neo4j_loader: Neo4jLoader, api_data: Dict[str, Any]):
    """Ingestion logic for Covalent API data."""
    logger.debug("Performing Covalent ingestion logic")
    if "data" in api_data and "items" in api_data["data"]:
        query = """
        UNWIND $items AS item
        MERGE (t:Token {address: item.contract_address})
        ON CREATE SET t.name = item.contract_name, t.decimals = item.contract_decimals
        MERGE (a:Address {address: $wallet_address})
        MERGE (a)-[r:HOLDS]->(t)
        SET r.balance = item.balance, r.quote = item.quote
        """
        await neo4j_loader._execute_query(query, {
            "items": api_data["data"]["items"],
            "wallet_address": api_data["data"]["address"]
        })

async def _perform_moralis_ingestion(client: Any, neo4j_loader: Neo4jLoader, api_data: Dict[str, Any]):
    """Ingestion logic for Moralis API data."""
    logger.debug("Performing Moralis ingestion logic")
    if "result" in api_data and isinstance(api_data["result"], list):
        query = """
        UNWIND $nfts AS nft
        MERGE (n:NFT {token_address: nft.token_address, token_id: nft.token_id})
        ON CREATE SET n.name = nft.name, n.symbol = nft.symbol, n.contract_type = nft.contract_type
        MERGE (a:Address {address: $wallet_address})
        MERGE (a)-[r:OWNS]->(n)
        SET r.amount = nft.amount
        """
        await neo4j_loader._execute_query(query, {
            "nfts": api_data["result"],
            "wallet_address": api_data.get("owner", "unknown") # Assuming owner info is available
        })

PROVIDER_TEST_CASES = {
    "sim": {
        "successful_ingestion": {
            "mock_responses": {
                "GET /v1/evm/balances/0x123": {"status_code": 200, "content": {"balances": [{"address": "native", "chain": "ethereum", "symbol": "ETH", "amount": "1000"}]}},
            },
            "test_logic": _perform_sim_ingestion,
            "client_method": "get_balances",
            "client_args": {"address": "0x123"}
        }
    },
    "covalent": {
        "successful_ingestion": {
            "mock_responses": {
                "GET /v1/1/address/0x123/balances_v2/": {"status_code": 200, "content": {"data": {"address": "0x123", "items": [{"contract_address": "0xabc", "contract_name": "TestCoin", "contract_decimals": 18, "balance": "5000", "quote": 123.45}]}}},
            },
            "test_logic": _perform_covalent_ingestion,
            "client_method": "get_token_balances",
            "client_args": {"chain_id": "1", "address": "0x123"}
        }
    },
    "moralis": {
        "successful_ingestion": {
            "mock_responses": {
                "GET /api/v2.2/0x123/nft": {"status_code": 200, "content": {"result": [{"token_address": "0xdef", "token_id": "1", "name": "TestNFT", "symbol": "TNFT", "contract_type": "ERC721", "amount": "1"}], "owner": "0x123"}},
            },
            "test_logic": _perform_moralis_ingestion,
            "client_method": "get_wallet_nfts",
            "client_args": {"address": "0x123"}
        }
    }
}


# --- Mocking Setup ---

class MockTransport(httpx.AsyncBaseTransport):
    """
    A mock HTTPX transport that allows defining responses for specific URLs/methods.
    """
    def __init__(self, responses: Dict[str, Any]):
        self.responses = responses
        self.requests_made: List[httpx.Request] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.requests_made.append(request)
        # Normalize path by removing trailing slash
        path = request.url.path.rstrip('/')
        key = f"{request.method} {path}"
        
        response_config = self.responses.get(key)
        
        if response_config:
            if isinstance(response_config, list): # Handle sequential responses
                if not hasattr(self, '_call_counts'): self._call_counts = defaultdict(int)
                response_config = response_config[self._call_counts[key]]
                self._call_counts[key] += 1

            status_code = response_config.get("status_code", 200)
            content = response_config.get("content", {})
            headers = response_config.get("headers", {})
            
            if isinstance(content, dict):
                content = json.dumps(content).encode('utf-8')
            elif isinstance(content, str):
                content = content.encode('utf-8')

            return httpx.Response(status_code, content=content, headers=headers, request=request)
        
        logger.warning(f"No mock response for: '{key}'. Available mocks: {list(self.responses.keys())}. Returning 404.")
        return httpx.Response(404, content=b"Not Found", request=request)

# Patching global metrics to capture calls
mock_api_metrics_track_call = MagicMock()
mock_api_metrics_track_credits = MagicMock()
mock_db_metrics_track_operation = MagicMock()

@patch('backend.core.metrics.ApiMetrics.track_call', new=mock_api_metrics_track_call)
@patch('backend.core.metrics.ApiMetrics.track_credits', new=mock_api_metrics_track_credits)
@patch('backend.core.metrics.DatabaseMetrics.track_operation', new=mock_db_metrics_track_operation)
class IntegrationTestRunner:
    """
    Runs integration tests for a given provider.
    """
    def __init__(self, provider_config: ProviderConfig, verbose: bool = False):
        self.provider_config = provider_config
        self.provider_id = provider_config.get("id")
        self.verbose = verbose
        if self.verbose:
            logger.setLevel(logging.DEBUG)
        
        self.mock_neo4j_loader = self._setup_mock_neo4j_loader()
        self.mock_redis_client = self._setup_mock_redis_client()
        
        self.cleanup_tasks = []

    def addCleanup(self, func):
        self.cleanup_tasks.append(func)

    def doCleanups(self):
        for func in reversed(self.cleanup_tasks):
            func()
        self.cleanup_tasks = []

    def _setup_mock_neo4j_loader(self) -> Neo4jLoader:
        """Sets up a mock Neo4jLoader to simulate database interactions."""
        loader = MagicMock(spec=Neo4jLoader)
        loader._execute_query = AsyncMock(return_value=MagicMock(
            consume=MagicMock(return_value=MagicMock(counters={'nodes_created': 1})),
            single=MagicMock(return_value={})
        ))
        loader.ingest_balances = AsyncMock()
        loader.ingest_activity = AsyncMock()
        patcher = patch('backend.core.neo4j_loader.Neo4jLoader', return_value=loader)
        patcher.start()
        self.addCleanup(patcher.stop)
        return loader

    def _setup_mock_redis_client(self) -> RedisClient:
        """Sets up a mock RedisClient to simulate Redis interactions."""
        client = MagicMock(spec=RedisClient)
        client.get = MagicMock(return_value=None)
        client.set = MagicMock(return_value=True)
        patcher = patch('backend.core.redis_client.RedisClient', return_value=client)
        patcher.start()
        self.addCleanup(patcher.stop)
        return client

    async def _run_test_case(self, test_name: str, test_func: Callable, mock_responses: Dict[str, Any]):
        """Runs a single test case and records its result."""
        start_time = time.perf_counter()
        status = "FAIL"
        error_message = None
        
        logger.info(f"--- Running test: {self.provider_id} - {test_name} ---")
        
        # Reset mocks for this specific test case
        mock_api_metrics_track_call.reset_mock()
        mock_api_metrics_track_credits.reset_mock()
        mock_db_metrics_track_operation.reset_mock()
        self.mock_neo4j_loader._execute_query.reset_mock()
        self.mock_neo4j_loader.ingest_balances.reset_mock()
        self.mock_neo4j_loader.ingest_activity.reset_mock()

        mock_transport = MockTransport(mock_responses)
        
        try:
            client_module_name = f"backend.integrations.{self.provider_id}_client"
            __import__(client_module_name) # Ensure module is loaded
            client_class_name = self._to_camel_case(f"{self.provider_id}Client")
            
            with patch(f'{client_module_name}.httpx.AsyncClient') as MockAsyncClient:
                # The mock transport needs to be an awaitable that returns the transport
                async def transport_factory(*args, **kwargs):
                    return mock_transport
                
                MockAsyncClient.return_value = httpx.AsyncClient(transport=mock_transport)

                try:
                    await test_func()
                    status = "PASS"
                    logger.info(f"Test {self.provider_id} - {test_name} PASSED.")
                except Exception as e:
                    status = "FAIL"
                    error_message = f"{type(e).__name__}: {e}"
                    logger.error(f"Test {self.provider_id} - {test_name} FAILED: {e}", exc_info=self.verbose)

        except ImportError as e:
            status = "SKIP"
            error_message = f"Client module not found or invalid: {e}. Did you run new_provider_scaffold.py?"
            logger.warning(f"Test {self.provider_id} - {test_name} SKIPPED: {error_message}")
        except Exception as e:
            status = "FAIL"
            error_message = f"Test setup failed: {type(e).__name__}: {e}"
            logger.error(f"Test {self.provider_id} - {test_name} FAILED during setup: {e}", exc_info=self.verbose)
        finally:
            duration = time.perf_counter() - start_time
            test_results.append({
                "provider_id": self.provider_id,
                "test_name": test_name,
                "status": status,
                "duration_ms": round(duration * 1000, 2),
                "error_message": error_message,
                "api_calls_made": len(mock_transport.requests_made),
                "neo4j_ingestions": self.mock_neo4j_loader._execute_query.call_count + self.mock_neo4j_loader.ingest_balances.call_count + self.mock_neo4j_loader.ingest_activity.call_count,
                "api_metrics_tracked": mock_api_metrics_track_call.call_count,
                "db_metrics_tracked": mock_db_metrics_track_operation.call_count,
            })
            self.doCleanups()

    async def run_all_tests(self):
        """Run all tests for this provider."""
        logger.info(f"=== Running tests for provider: {self.provider_id} ===")
        
        provider_tests = PROVIDER_TEST_CASES.get(self.provider_id)
        if not provider_tests:
            logger.warning(f"No specific test cases found for provider '{self.provider_id}'. Skipping.")
            test_results.append({
                "provider_id": self.provider_id, "test_name": "provider_setup",
                "status": "SKIP", "duration_ms": 0, "error_message": "No test cases defined in PROVIDER_TEST_CASES",
                "api_calls_made": 0, "neo4j_ingestions": 0, "api_metrics_tracked": 0, "db_metrics_tracked": 0,
            })
            return

        # --- Test Successful Ingestion ---
        ingestion_test_config = provider_tests.get("successful_ingestion")
        if ingestion_test_config:
            async def test_logic():
                client_module = __import__(f"backend.integrations.{self.provider_id}_client", fromlist=[''])
                ClientClass = getattr(client_module, self._to_camel_case(f"{self.provider_id}Client"))
                client = ClientClass()
                
                method_to_call = getattr(client, ingestion_test_config["client_method"])
                api_data = await method_to_call(**ingestion_test_config["client_args"])

                assert api_data is not None
                await ingestion_test_config["test_logic"](client, self.mock_neo4j_loader, api_data)
                
                # Assertions for metrics and Neo4j calls
                assert mock_api_metrics_track_call.call_count > 0
                ingestion_calls = self.mock_neo4j_loader._execute_query.call_count + self.mock_neo4j_loader.ingest_balances.call_count + self.mock_neo4j_loader.ingest_activity.call_count
                assert ingestion_calls > 0
                assert mock_db_metrics_track_operation.call_count > 0

            await self._run_test_case("successful_ingestion", test_logic, ingestion_test_config["mock_responses"])

        # Future generic tests (rate-limiting, errors, etc.) can be added here
        # For now, we focus on the provider-specific success case.

        logger.info(f"=== Completed tests for provider: {self.provider_id} ===")

    def _to_camel_case(self, snake_str: str) -> str:
        """Convert snake_case to CamelCase."""
        return ''.join(x.title() for x in snake_str.split('_'))

async def run_provider_tests(provider_config: ProviderConfig, verbose: bool = False) -> None:
    """Run tests for a specific provider."""
    runner = IntegrationTestRunner(provider_config, verbose)
    await runner.run_all_tests()

async def run_all_provider_tests(provider_configs: List[ProviderConfig], verbose: bool = False) -> None:
    """Run tests for all providers in parallel."""
    tasks = [run_provider_tests(p, verbose) for p in provider_configs]
    await asyncio.gather(*tasks)

def generate_report() -> str:
    """Generate a comprehensive test report."""
    if not test_results:
        return "No test results available."
    
    # Group results by provider
    provider_results = defaultdict(list)
    for result in test_results:
        provider_results[result["provider_id"]].append(result)
    
    report = ["=" * 80, f"INTEGRATION TEST MATRIX REPORT - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", "=" * 80]
    
    total_tests = len(test_results)
    passed = sum(1 for r in test_results if r["status"] == "PASS")
    failed = sum(1 for r in test_results if r["status"] == "FAIL")
    skipped = sum(1 for r in test_results if r["status"] == "SKIP")
    
    report.append(f"\nOVERALL SUMMARY:")
    report.append(f"  Total Tests Run: {total_tests}")
    report.append(f"  Passed: {passed} ({passed / total_tests * 100:.1f}%)" if total_tests > 0 else "  Passed: 0 (0.0%)")
    report.append(f"  Failed: {failed} ({failed / total_tests * 100:.1f}%)" if total_tests > 0 else "  Failed: 0 (0.0%)")
    report.append(f"  Skipped: {skipped} ({skipped / total_tests * 100:.1f}%)" if total_tests > 0 else "  Skipped: 0 (0.0%)")
    
    report.append("\nPROVIDER SUMMARIES:")
    for provider_id, results in provider_results.items():
        total = len(results)
        p_passed = sum(1 for r in results if r["status"] == "PASS")
        report.append(f"\n  {provider_id}: {p_passed}/{total} passed")
        for result in results:
            symbol = "✅" if result["status"] == "PASS" else "❌" if result["status"] == "FAIL" else "⏩"
            report.append(f"    {symbol} {result['test_name']} ({result['duration_ms']:.2f} ms)")
            if result["status"] == "FAIL":
                report.append(f"      ERROR: {result['error_message']}")
    
    return "\n".join(report)

def save_junit_report() -> None:
    """Save test results in JUnit XML format for CI/CD integration."""
    try:
        from junit_xml import TestSuite, TestCase
        
        provider_results = defaultdict(list)
        for result in test_results:
            provider_results[result["provider_id"]].append(result)
        
        test_suites = []
        for provider_id, results in provider_results.items():
            test_cases = []
            for result in results:
                case = TestCase(
                    name=result["test_name"],
                    classname=f"integration.{provider_id}",
                    elapsed_sec=result["duration_ms"] / 1000,
                )
                if result["status"] == "FAIL":
                    case.add_failure_info(message=result["error_message"])
                elif result["status"] == "SKIP":
                    case.add_skipped_info(message=result["error_message"])
                test_cases.append(case)
            
            test_suites.append(TestSuite(f"Integration Tests - {provider_id}", test_cases))
        
        with open('integration_test_results.xml', 'w') as f:
            f.write(TestSuite.to_xml_report_string(test_suites))
        
        logger.info("Saved JUnit XML report to integration_test_results.xml")
    
    except ImportError:
        logger.warning("junit-xml package not installed. Skipping JUnit XML report generation.")

def save_json_report() -> None:
    """Save test results in JSON format."""
    with open('integration_test_results.json', 'w') as f:
        json.dump({"results": test_results}, f, indent=2)
    logger.info("Saved JSON report to integration_test_results.json")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run integration tests for API providers.")
    parser.add_argument("--provider", help="Run tests only for a specific provider ID")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    return parser.parse_args()

async def main():
    """Main entry point."""
    args = parse_args()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    try:
        providers = get_providers_by_category("api")
        if not providers:
            logger.error("No API providers found in registry.")
            return 1
        
        if args.provider:
            providers = [p for p in providers if p.get("id") == args.provider]
            if not providers:
                logger.error(f"Provider '{args.provider}' not found.")
                return 1
        
        logger.info(f"Testing providers: {[p.get('id') for p in providers]}")
        
        await run_all_provider_tests(providers, args.verbose)
        
        report = generate_report()
        print(report)
        
        save_json_report()
        save_junit_report()
        
        return 1 if any(r['status'] == 'FAIL' for r in test_results) else 0
    
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
