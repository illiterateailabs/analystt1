#!/usr/bin/env python3
"""
Health Endpoints Test Script

This script tests all health endpoints in the API to verify they are working correctly.
It makes HTTP requests to each endpoint and validates the response structure.

Usage:
    python test_health_endpoints.py [--base-url URL] [--verbose]

Options:
    --base-url URL    Base URL of the API (default: http://localhost:8000)
    --verbose         Show detailed response data
"""

import argparse
import json
import sys
import time
from typing import Dict, Any, List, Tuple

import requests
from requests.exceptions import RequestException


# ANSI color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_colored(message: str, color: str) -> None:
    """Print a message with color."""
    print(f"{color}{message}{Colors.ENDC}")


def print_success(message: str) -> None:
    """Print a success message."""
    print_colored(f"✅ {message}", Colors.GREEN)


def print_warning(message: str) -> None:
    """Print a warning message."""
    print_colored(f"⚠️ {message}", Colors.YELLOW)


def print_error(message: str) -> None:
    """Print an error message."""
    print_colored(f"❌ {message}", Colors.RED)


def print_info(message: str) -> None:
    """Print an info message."""
    print_colored(message, Colors.BLUE)


def print_header(message: str) -> None:
    """Print a header message."""
    print("\n" + "=" * 80)
    print_colored(f"{message}", Colors.HEADER + Colors.BOLD)
    print("=" * 80)


def validate_response(response: Dict[str, Any], required_fields: List[str]) -> Tuple[bool, List[str]]:
    """
    Validate that the response contains all required fields.
    
    Args:
        response: The response data to validate
        required_fields: List of required field names
        
    Returns:
        Tuple of (is_valid, missing_fields)
    """
    missing = []
    for field in required_fields:
        if field not in response:
            missing.append(field)
    return len(missing) == 0, missing


def test_endpoint(base_url: str, endpoint: str, required_fields: List[str], verbose: bool = False) -> bool:
    """
    Test a health endpoint.
    
    Args:
        base_url: Base URL of the API
        endpoint: Endpoint path to test
        required_fields: List of required fields in the response
        verbose: Whether to print detailed response data
        
    Returns:
        True if the test passed, False otherwise
    """
    url = f"{base_url}{endpoint}"
    print_info(f"Testing {url}...")
    
    try:
        start_time = time.time()
        response = requests.get(url, timeout=10)
        elapsed = (time.time() - start_time) * 1000
        
        if response.status_code == 200:
            data = response.json()
            is_valid, missing = validate_response(data, required_fields)
            
            if is_valid:
                print_success(f"Endpoint {endpoint} responded in {elapsed:.2f}ms")
                if verbose:
                    print(json.dumps(data, indent=2))
                return True
            else:
                print_error(f"Endpoint {endpoint} missing required fields: {', '.join(missing)}")
                if verbose:
                    print(json.dumps(data, indent=2))
                return False
        else:
            print_error(f"Endpoint {endpoint} returned status code {response.status_code}")
            print_error(response.text)
            return False
    except RequestException as e:
        print_error(f"Failed to connect to {endpoint}: {str(e)}")
        return False
    except json.JSONDecodeError:
        print_error(f"Endpoint {endpoint} returned invalid JSON")
        print_error(response.text)
        return False
    except Exception as e:
        print_error(f"Unexpected error testing {endpoint}: {str(e)}")
        return False


def test_basic_health(base_url: str, verbose: bool = False) -> bool:
    """Test the basic health endpoint."""
    print_header("Testing Basic Health Endpoint")
    return test_endpoint(
        base_url=base_url,
        endpoint="/health",
        required_fields=["status", "timestamp"],
        verbose=verbose
    )


def test_database_health(base_url: str, verbose: bool = False) -> bool:
    """Test the database health endpoint."""
    print_header("Testing Database Health Endpoint")
    return test_endpoint(
        base_url=base_url,
        endpoint="/api/v1/health/database",
        required_fields=["status", "connection_pool", "latency_ms"],
        verbose=verbose
    )


def test_redis_health(base_url: str, verbose: bool = False) -> bool:
    """Test the Redis health endpoint."""
    print_header("Testing Redis Health Endpoint")
    return test_endpoint(
        base_url=base_url,
        endpoint="/api/v1/health/redis",
        required_fields=["status", "components"],
        verbose=verbose
    )


def test_graph_health(base_url: str, verbose: bool = False) -> bool:
    """Test the Neo4j graph health endpoint."""
    print_header("Testing Graph Health Endpoint")
    return test_endpoint(
        base_url=base_url,
        endpoint="/api/v1/health/graph",
        required_fields=["status", "node_count", "latency_ms"],
        verbose=verbose
    )


def test_worker_health(base_url: str, verbose: bool = False) -> bool:
    """Test the Celery worker health endpoint."""
    print_header("Testing Worker Health Endpoint")
    return test_endpoint(
        base_url=base_url,
        endpoint="/api/v1/health/workers",
        required_fields=["overall_status", "workers", "queues"],
        verbose=verbose
    )


def test_system_health(base_url: str, verbose: bool = False) -> bool:
    """Test the comprehensive system health endpoint."""
    print_header("Testing System Health Endpoint")
    return test_endpoint(
        base_url=base_url,
        endpoint="/api/v1/health/system",
        required_fields=["status", "components", "response_time_ms"],
        verbose=verbose
    )


def main() -> None:
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Test API health endpoints")
    parser.add_argument("--base-url", default="http://localhost:8000", help="Base URL of the API")
    parser.add_argument("--verbose", action="store_true", help="Show detailed response data")
    args = parser.parse_args()
    
    print_header(f"Testing Health Endpoints at {args.base_url}")
    
    results = []
    results.append(("Basic Health", test_basic_health(args.base_url, args.verbose)))
    results.append(("Database Health", test_database_health(args.base_url, args.verbose)))
    results.append(("Redis Health", test_redis_health(args.base_url, args.verbose)))
    results.append(("Graph Health", test_graph_health(args.base_url, args.verbose)))
    results.append(("Worker Health", test_worker_health(args.base_url, args.verbose)))
    results.append(("System Health", test_system_health(args.base_url, args.verbose)))
    
    # Print summary
    print_header("Test Results Summary")
    passed = 0
    failed = 0
    
    for name, result in results:
        if result:
            print_success(f"{name}: PASSED")
            passed += 1
        else:
            print_error(f"{name}: FAILED")
            failed += 1
    
    print("\n" + "-" * 40)
    print(f"Total: {len(results)}, Passed: {passed}, Failed: {failed}")
    
    # Exit with appropriate status code
    if failed > 0:
        print_error("Some tests failed!")
        sys.exit(1)
    else:
        print_success("All tests passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()
