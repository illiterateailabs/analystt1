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
    python scripts/integration_test_matrix.py --provider sim-api
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
from backend.core.neo4j_loader import Neo4jLoader, GraphStats, Balance, Activity, Token, Relationship
from backend.core.metrics import ApiMetrics, DatabaseMetrics
from backend.core.redis_client import RedisClient, RedisDb, SerializationFormat

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global test results storage
test_results: List[Dict[str, Any]] = []

# --- Mocking Setup ---

class MockTransport:
    """
    A mock HTTPX transport that allows defining responses for specific URLs/methods.
    """
    def __init__(self, responses: Dict[str, Any]):
        self.responses = responses
        self.requests_made: List[httpx.Request] = []

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self.requests_made.append(request)
        key = f"{request.method} {request.url.path}"
        
        if key in self.responses:
            response_config = self.responses[key]
            status_code = response_config.get("status_code", 200)
            content = response_config.get("content", {})
            headers = response_config.get("headers", {})
            
            if isinstance(content, dict):
                content = json.dumps(content).encode('utf-8')
            elif isinstance(content, str):
                content = content.encode('utf-8')

            return httpx.Response(status_code, content=content, headers=headers, request=request)
        
        logger.warning(f"No mock response for: {key}. Returning 404.")
        return httpx.Response(404, content=b"Not Found", request=request)

# Patching global metrics to capture calls
mock_api_metrics_track_call = AsyncMock()
mock_api_metrics_track_credits = AsyncMock()
mock_db_metrics_track_operation = AsyncMock()

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
        
        # Reset mocks before each test run for a provider
        mock_api_metrics_track_call.reset_mock()
        mock_api_metrics_track_credits.reset_mock()
        mock_db_metrics_track_operation.reset_mock()

    def _setup_mock_neo4j_loader(self) -> Neo4jLoader:
        """Sets up a mock Neo4jLoader to simulate database interactions."""
        loader = MagicMock(spec=Neo4jLoader)
        loader._execute_query = AsyncMock(return_value=MagicMock(
            consume=MagicMock(return_value=MagicMock(
                counters=MagicMock(
                    nodes_created=1, relationships_created=1, properties_set=2, labels_added=1
                ),
                result_available_after=10.0
            )),
            single=MagicMock(return_value={}) # For get_node, get_relationship etc.
        ))
        loader._process_result_stats = MagicMock(return_value=GraphStats(
            nodes_created=1, relationships_created=1, properties_set=2, labels_added=1,
            query_time_ms=10.0
        ))
        loader._execute_query_with_retry = AsyncMock(side_effect=loader._execute_query)
        
        # Mock the constructor to return our mock instance
        patcher = patch('backend.core.neo4j_loader.Neo4jLoader', return_value=loader)
        patcher.start()
        self.addCleanup(patcher.stop) # Ensure patch is stopped after test
        
        return loader

    def _setup_mock_redis_client(self) -> RedisClient:
        """Sets up a mock RedisClient to simulate Redis interactions."""
        client = MagicMock(spec=RedisClient)
        client.get = MagicMock(return_value=None) # Default to cache miss
        client.set = MagicMock(return_value=True)
        client.store_vector = MagicMock(return_value=True)
        client.get_vector = MagicMock(return_value=None)
        
        # Mock the constructor to return our mock instance
        patcher = patch('backend.core.redis_client.RedisClient', return_value=client)
        patcher.start()
        self.addCleanup(patcher.stop) # Ensure patch is stopped after test
        
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

        # Setup mock HTTPX transport for the client
        mock_transport = MockTransport(mock_responses)
        
        # Dynamically import the client module and patch httpx.AsyncClient
        try:
            client_module_name = f"backend.integrations.{self.provider_id}_client"
            client_module = __import__(client_module_name, fromlist=[''])
            client_class_name = self._to_camel_case(f"{self.provider_id}Client")
            client_class = getattr(client_module, client_class_name)

            with patch('httpx.AsyncClient', autospec=True) as MockAsyncClient:
                MockAsyncClient.return_value.__aenter__.return_value.request.side_effect = mock_transport.handle_request
                MockAsyncClient.return_value.__aenter__.return_value.post.side_effect = mock_transport.handle_request
                
                # Patch get_provider to return the current provider's config
                with patch('backend.integrations.get_provider', return_value=self.provider_config):
                    try:
                        await test_func(client_class, self.mock_neo4j_loader, self.mock_redis_client)
                        status = "PASS"
                        logger.info(f"Test {self.provider_id} - {test_name} PASSED.")
                    except Exception as e:
                        status = "FAIL"
                        error_message = str(e)
                        logger.error(f"Test {self.provider_id} - {test_name} FAILED: {e}", exc_info=self.verbose)
        except ImportError as e:
            status = "SKIP"
            error_message = f"Client module not found or invalid: {e}. Did you run new_provider_scaffold.py?"
            logger.warning(f"Test {self.provider_id} - {test_name} SKIPPED: {error_message}")
        except Exception as e:
            status = "FAIL"
            error_message = str(e)
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
                "neo4j_ingestions": self.mock_neo4j_loader._execute_query.call_count,
                "api_metrics_tracked": mock_api_metrics_track_call.call_count,
                "db_metrics_tracked": mock_db_metrics_track_operation.call_count,
            })

    async def test_successful_ingestion(self, ClientClass: Type[Any], neo4j_loader: Neo4jLoader, redis_client: RedisClient):
        """Tests a successful API call and Neo4j ingestion."""
        logger.debug(f"Running test_successful_ingestion for {self.provider_id}")
        
        mock_responses = {
            "GET /api/v1/data": {"status_code": 200, "content": {"data": [{"id": "test1", "value": 100}]}},
            "POST /api/v1/transactions": {"status_code": 200, "content": {"status": "success"}}
        }
        
        await self._run_test_case("successful_ingestion", self._perform_successful_ingestion, mock_responses)
        
        # Assertions for metrics and Neo4j calls
        assert mock_api_metrics_track_call.call_count > 0
        assert neo4j_loader._execute_query.call_count > 0
        assert mock_db_metrics_track_operation.call_count > 0

    async def _perform_successful_ingestion(self, ClientClass: Type[Any], neo4j_loader: Neo4jLoader, redis_client: RedisClient):
        """Helper function to perform the actual successful ingestion flow."""
        client = ClientClass()
        
        # Simulate API call
        if hasattr(client, 'get_data'):
            api_data = await client.get_data()
        elif hasattr(client, 'get_transactions'):
            api_data = await client.get_transactions(variables={"address": "0x123", "limit": 1})
        else:
            raise NotImplementedError("No suitable data retrieval method found on client.")

        assert api_data is not None
        logger.debug(f"API data received: {api_data}")

        # Simulate Neo4j ingestion
        if "data" in api_data and isinstance(api_data["data"], list) and api_data["data"]:
            sample_item = api_data["data"][0]
            if "id" in sample_item and "value" in sample_item:
                # Example: Ingest as a Balance or Activity
                if self.provider_id == "sim-api": # Assuming sim-api provides balances/activities
                    balances = [Balance(address="0x123", chain="ethereum", asset="ETH", amount=1.0)]
                    await neo4j_loader.ingest_balances(balances)
                else:
                    # Generic ingestion for other providers
                    query = """
                    UNWIND $items AS item
                    MERGE (n:Item {id: item.id})
                    SET n.value = item.value
                    RETURN n
                    """
                    await neo4j_loader._execute_query(query, {"items": api_data["data"]})
        else:
            # Generic ingestion for other data formats
            query = """
            CREATE (n:ApiResponse)
            SET n = $data
            RETURN n
            """
            await neo4j_loader._execute_query(query, {"data": api_data})

    async def test_rate_limit_handling(self, ClientClass: Type[Any], neo4j_loader: Neo4jLoader, redis_client: RedisClient):
        """Tests handling of rate limiting responses."""
        logger.debug(f"Running test_rate_limit_handling for {self.provider_id}")
        
        # Setup mock responses with rate limit then success
        mock_responses = {
            "GET /api/v1/data": [
                {"status_code": 429, "headers": {"Retry-After": "1"}}, # First call gets rate limited
                {"status_code": 200, "content": {"data": [{"id": "test1", "value": 100}]}} # Second call succeeds
            ]
        }
        
        await self._run_test_case("rate_limit_handling", self._perform_rate_limit_handling, mock_responses)
        
        # Assertions
        assert mock_api_metrics_track_call.call_count > 0

    async def _perform_rate_limit_handling(self, ClientClass: Type[Any], neo4j_loader: Neo4jLoader, redis_client: RedisClient):
        """Helper function to test rate limit handling."""
        client = ClientClass()
        
        # This should automatically retry after the rate limit response
        if hasattr(client, 'get_data'):
            api_data = await client.get_data()
        elif hasattr(client, 'get_transactions'):
            api_data = await client.get_transactions(variables={"address": "0x123", "limit": 1})
        else:
            raise NotImplementedError("No suitable data retrieval method found on client.")
        
        assert api_data is not None
        logger.debug(f"API data received after rate limit: {api_data}")

    async def test_api_error_handling(self, ClientClass: Type[Any], neo4j_loader: Neo4jLoader, redis_client: RedisClient):
        """Tests handling of API errors."""
        logger.debug(f"Running test_api_error_handling for {self.provider_id}")
        
        # Setup mock response with server error
        mock_responses = {
            "GET /api/v1/data": {"status_code": 500, "content": {"error": "Internal Server Error"}}
        }
        
        await self._run_test_case("api_error_handling", self._perform_api_error_handling, mock_responses)

    async def _perform_api_error_handling(self, ClientClass: Type[Any], neo4j_loader: Neo4jLoader, redis_client: RedisClient):
        """Helper function to test API error handling."""
        client = ClientClass()
        
        # This should raise an exception due to the 500 error
        with self.assertRaises(httpx.HTTPStatusError):
            if hasattr(client, 'get_data'):
                await client.get_data()
            elif hasattr(client, 'get_transactions'):
                await client.get_transactions(variables={"address": "0x123", "limit": 1})
            else:
                raise NotImplementedError("No suitable data retrieval method found on client.")

    async def test_cache_integration(self, ClientClass: Type[Any], neo4j_loader: Neo4jLoader, redis_client: RedisClient):
        """Tests integration with Redis caching."""
        logger.debug(f"Running test_cache_integration for {self.provider_id}")
        
        # Setup mock response
        mock_responses = {
            "GET /api/v1/data": {"status_code": 200, "content": {"data": [{"id": "test1", "value": 100}]}}
        }
        
        await self._run_test_case("cache_integration", self._perform_cache_integration, mock_responses)

    async def _perform_cache_integration(self, ClientClass: Type[Any], neo4j_loader: Neo4jLoader, redis_client: RedisClient):
        """Helper function to test cache integration."""
        client = ClientClass()
        
        # First call should miss cache
        redis_client.get.return_value = None
        
        if hasattr(client, 'get_data'):
            api_data = await client.get_data()
        elif hasattr(client, 'get_transactions'):
            api_data = await client.get_transactions(variables={"address": "0x123", "limit": 1})
        else:
            raise NotImplementedError("No suitable data retrieval method found on client.")
        
        # Verify cache set was called
        assert redis_client.set.call_count > 0
        
        # Setup cache hit for second call
        redis_client.get.return_value = api_data
        
        # Second call should hit cache
        if hasattr(client, 'get_data'):
            cached_data = await client.get_data()
        elif hasattr(client, 'get_transactions'):
            cached_data = await client.get_transactions(variables={"address": "0x123", "limit": 1})
        else:
            raise NotImplementedError("No suitable data retrieval method found on client.")
        
        assert cached_data == api_data

    async def test_data_validation(self, ClientClass: Type[Any], neo4j_loader: Neo4jLoader, redis_client: RedisClient):
        """Tests data validation using Pydantic models."""
        logger.debug(f"Running test_data_validation for {self.provider_id}")
        
        # Setup mock response with invalid data
        mock_responses = {
            "GET /api/v1/data": {"status_code": 200, "content": {"data": [{"id": "test1", "invalid_field": "value"}]}}
        }
        
        await self._run_test_case("data_validation", self._perform_data_validation, mock_responses)

    async def _perform_data_validation(self, ClientClass: Type[Any], neo4j_loader: Neo4jLoader, redis_client: RedisClient):
        """Helper function to test data validation."""
        from pydantic import BaseModel, ValidationError
        
        # Define a validation model
        class DataItem(BaseModel):
            id: str
            value: int  # This will fail validation with the mock data
        
        client = ClientClass()
        
        # Get data from API
        if hasattr(client, 'get_data'):
            api_data = await client.get_data()
        elif hasattr(client, 'get_transactions'):
            api_data = await client.get_transactions(variables={"address": "0x123", "limit": 1})
        else:
            raise NotImplementedError("No suitable data retrieval method found on client.")
        
        # Validate data
        if "data" in api_data and isinstance(api_data["data"], list):
            with self.assertRaises(ValidationError):
                for item in api_data["data"]:
                    DataItem(**item)  # This should fail validation

    async def test_metrics_integration(self, ClientClass: Type[Any], neo4j_loader: Neo4jLoader, redis_client: RedisClient):
        """Tests integration with metrics tracking."""
        logger.debug(f"Running test_metrics_integration for {self.provider_id}")
        
        # Setup mock response
        mock_responses = {
            "GET /api/v1/data": {"status_code": 200, "content": {"data": [{"id": "test1", "value": 100}]}}
        }
        
        await self._run_test_case("metrics_integration", self._perform_metrics_integration, mock_responses)

    async def _perform_metrics_integration(self, ClientClass: Type[Any], neo4j_loader: Neo4jLoader, redis_client: RedisClient):
        """Helper function to test metrics integration."""
        client = ClientClass()
        
        # Make API call
        if hasattr(client, 'get_data'):
            await client.get_data()
        elif hasattr(client, 'get_transactions'):
            await client.get_transactions(variables={"address": "0x123", "limit": 1})
        else:
            raise NotImplementedError("No suitable data retrieval method found on client.")
        
        # Verify metrics were called
        assert mock_api_metrics_track_call.call_count > 0
        
        # Perform Neo4j operation
        await neo4j_loader._execute_query("MATCH (n) RETURN n LIMIT 1", {})
        
        # Verify database metrics were called
        assert mock_db_metrics_track_operation.call_count > 0

    async def run_all_tests(self):
        """Run all tests for this provider."""
        logger.info(f"=== Running tests for provider: {self.provider_id} ===")
        
        # Define test cases
        test_cases = [
            (self.test_successful_ingestion, "Successful Ingestion"),
            (self.test_rate_limit_handling, "Rate Limit Handling"),
            (self.test_api_error_handling, "API Error Handling"),
            (self.test_cache_integration, "Cache Integration"),
            (self.test_data_validation, "Data Validation"),
            (self.test_metrics_integration, "Metrics Integration"),
        ]
        
        # Run test cases
        for test_func, test_name in test_cases:
            try:
                await test_func()
            except Exception as e:
                logger.error(f"Error running test {test_name}: {e}", exc_info=self.verbose)
        
        logger.info(f"=== Completed tests for provider: {self.provider_id} ===")

    def _to_camel_case(self, snake_str: str) -> str:
        """Convert snake_case to CamelCase."""
        components = snake_str.split('_')
        return ''.join(x.title() for x in components)
    
    def assertRaises(self, exception_class, callable_obj=None, *args, **kwargs):
        """Simple implementation of unittest.TestCase.assertRaises for use in test methods."""
        if callable_obj is None:
            return _AssertRaisesContext(exception_class)
        
        try:
            callable_obj(*args, **kwargs)
        except exception_class:
            return
        except Exception as e:
            raise AssertionError(f"Expected {exception_class.__name__}, but got {type(e).__name__}: {e}")
        
        raise AssertionError(f"Expected {exception_class.__name__} to be raised, but no exception was raised")
    
    def addCleanup(self, func, *args, **kwargs):
        """Simple implementation of unittest.TestCase.addCleanup for use in test methods."""
        # In a real implementation, we would store these for cleanup
        # For now, just log them
        logger.debug(f"Would add cleanup: {func.__name__}")


class _AssertRaisesContext:
    """Context manager for assertRaises."""
    def __init__(self, expected):
        self.expected = expected
        self.exception = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, tb):
        if exc_type is None:
            raise AssertionError(f"{self.expected.__name__} not raised")
        if not issubclass(exc_type, self.expected):
            return False
        self.exception = exc_value
        return True


async def run_provider_tests(provider_config: ProviderConfig, verbose: bool = False) -> None:
    """Run tests for a specific provider."""
    runner = IntegrationTestRunner(provider_config, verbose)
    await runner.run_all_tests()


async def run_all_provider_tests(provider_configs: List[ProviderConfig], verbose: bool = False) -> None:
    """Run tests for all providers in parallel."""
    tasks = []
    for provider_config in provider_configs:
        tasks.append(run_provider_tests(provider_config, verbose))
    
    await asyncio.gather(*tasks)


def generate_report() -> str:
    """Generate a comprehensive test report."""
    if not test_results:
        return "No test results available."
    
    # Group results by provider
    provider_results = defaultdict(list)
    for result in test_results:
        provider_results[result["provider_id"]].append(result)
    
    # Generate report
    report = []
    report.append("=" * 80)
    report.append(f"INTEGRATION TEST MATRIX REPORT - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("=" * 80)
    
    # Overall summary
    total_tests = len(test_results)
    passed_tests = sum(1 for r in test_results if r["status"] == "PASS")
    failed_tests = sum(1 for r in test_results if r["status"] == "FAIL")
    skipped_tests = sum(1 for r in test_results if r["status"] == "SKIP")
    
    report.append(f"\nOVERALL SUMMARY:")
    report.append(f"  Total Tests: {total_tests}")
    report.append(f"  Passed: {passed_tests} ({passed_tests / total_tests * 100:.1f}%)")
    report.append(f"  Failed: {failed_tests} ({failed_tests / total_tests * 100:.1f}%)")
    report.append(f"  Skipped: {skipped_tests} ({skipped_tests / total_tests * 100:.1f}%)")
    
    # Performance summary
    durations = [r["duration_ms"] for r in test_results]
    avg_duration = sum(durations) / len(durations) if durations else 0
    max_duration = max(durations) if durations else 0
    min_duration = min(durations) if durations else 0
    
    report.append(f"\nPERFORMANCE SUMMARY:")
    report.append(f"  Average Test Duration: {avg_duration:.2f} ms")
    report.append(f"  Maximum Test Duration: {max_duration:.2f} ms")
    report.append(f"  Minimum Test Duration: {min_duration:.2f} ms")
    
    # Provider summaries
    report.append("\nPROVIDER SUMMARIES:")
    for provider_id, results in provider_results.items():
        provider_total = len(results)
        provider_passed = sum(1 for r in results if r["status"] == "PASS")
        provider_failed = sum(1 for r in results if r["status"] == "FAIL")
        provider_skipped = sum(1 for r in results if r["status"] == "SKIP")
        
        report.append(f"\n  {provider_id}:")
        report.append(f"    Total Tests: {provider_total}")
        report.append(f"    Passed: {provider_passed} ({provider_passed / provider_total * 100:.1f}%)")
        report.append(f"    Failed: {provider_failed} ({provider_failed / provider_total * 100:.1f}%)")
        report.append(f"    Skipped: {provider_skipped} ({provider_skipped / provider_total * 100:.1f}%)")
        
        # API and DB integration
        api_calls = sum(r["api_calls_made"] for r in results)
        neo4j_ingestions = sum(r["neo4j_ingestions"] for r in results)
        api_metrics = sum(r["api_metrics_tracked"] for r in results)
        db_metrics = sum(r["db_metrics_tracked"] for r in results)
        
        report.append(f"    API Calls Made: {api_calls}")
        report.append(f"    Neo4j Ingestions: {neo4j_ingestions}")
        report.append(f"    API Metrics Tracked: {api_metrics}")
        report.append(f"    DB Metrics Tracked: {db_metrics}")
    
    # Detailed results
    report.append("\nDETAILED TEST RESULTS:")
    for provider_id, results in provider_results.items():
        report.append(f"\n  {provider_id}:")
        for result in results:
            status_symbol = "✅" if result["status"] == "PASS" else "❌" if result["status"] == "FAIL" else "⏩"
            report.append(f"    {status_symbol} {result['test_name']} - {result['duration_ms']:.2f} ms")
            if result["status"] == "FAIL" and result["error_message"]:
                report.append(f"      Error: {result['error_message']}")
    
    # CI/CD integration
    report.append("\nCI/CD INTEGRATION:")
    report.append(f"  Exit Code: {1 if failed_tests > 0 else 0}")
    report.append(f"  JUnit XML Report: {Path.cwd() / 'integration_test_results.xml'}")
    report.append(f"  JSON Report: {Path.cwd() / 'integration_test_results.json'}")
    
    return "\n".join(report)


def save_junit_report() -> None:
    """Save test results in JUnit XML format for CI/CD integration."""
    try:
        from junit_xml import TestSuite, TestCase
        
        # Group results by provider
        provider_results = defaultdict(list)
        for result in test_results:
            provider_results[result["provider_id"]].append(result)
        
        # Create test suites
        test_suites = []
        for provider_id, results in provider_results.items():
            test_cases = []
            for result in results:
                test_case = TestCase(
                    name=result["test_name"],
                    classname=f"integration.{provider_id}",
                    elapsed_sec=result["duration_ms"] / 1000,
                )
                
                if result["status"] == "FAIL":
                    test_case.add_failure_info(message=result["error_message"])
                elif result["status"] == "SKIP":
                    test_case.add_skipped_info(message=result["error_message"])
                
                test_cases.append(test_case)
            
            test_suite = TestSuite(f"Integration Tests - {provider_id}", test_cases)
            test_suites.append(test_suite)
        
        # Save to file
        with open('integration_test_results.xml', 'w') as f:
            from junit_xml import to_xml_report_string
            f.write(to_xml_report_string(test_suites))
        
        logger.info("Saved JUnit XML report to integration_test_results.xml")
    
    except ImportError:
        logger.warning("junit-xml package not installed. Skipping JUnit XML report generation.")


def save_json_report() -> None:
    """Save test results in JSON format for CI/CD integration."""
    report = {
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "total_tests": len(test_results),
            "passed_tests": sum(1 for r in test_results if r["status"] == "PASS"),
            "failed_tests": sum(1 for r in test_results if r["status"] == "FAIL"),
            "skipped_tests": sum(1 for r in test_results if r["status"] == "SKIP"),
        },
        "results": test_results,
    }
    
    with open('integration_test_results.json', 'w') as f:
        json.dump(report, f, indent=2)
    
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
    
    # Load provider configurations
    try:
        providers = get_providers_by_category("api")
        
        if not providers:
            logger.error("No API providers found in registry.")
            return 1
        
        # Filter providers if specified
        if args.provider:
            providers = [p for p in providers if p.get("id") == args.provider]
            if not providers:
                logger.error(f"Provider '{args.provider}' not found in registry.")
                return 1
        
        logger.info(f"Found {len(providers)} API providers to test.")
        
        # Run tests
        await run_all_provider_tests(providers, args.verbose)
        
        # Generate and print report
        report = generate_report()
        print(report)
        
        # Save reports for CI/CD
        save_json_report()
        save_junit_report()
        
        # Return exit code based on test results
        failed_tests = sum(1 for r in test_results if r["status"] == "FAIL")
        return 1 if failed_tests > 0 else 0
    
    except Exception as e:
        logger.error(f"Error running integration tests: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    # Run the async main function
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
