#!/usr/bin/env python3
"""
Smoke Test Runner for Analyst Droid One Platform

This script runs the smoke tests to validate that the entire platform is working
correctly. It checks the environment, runs the tests, and reports the results in
a user-friendly way.

Usage:
    python smoke_test.py [--verbose] [--quick] [--skip-env-check]

Options:
    --verbose       Show detailed test output
    --quick         Run only essential tests for a quick check
    --skip-env-check Skip environment checks (useful if you know services are running)
"""

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# Try to import required packages
try:
    import pytest
    import redis
    import requests
    from neo4j import GraphDatabase
except ImportError as e:
    print(f"Error: Missing required package: {e}")
    print("Please install required packages: pip install pytest redis requests neo4j")
    sys.exit(1)

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


def print_header(message: str) -> None:
    """Print a header message."""
    print("\n" + "=" * 80)
    print(f"{Colors.HEADER}{Colors.BOLD}{message}{Colors.ENDC}")
    print("=" * 80)


def print_success(message: str) -> None:
    """Print a success message."""
    print(f"{Colors.GREEN}✅ {message}{Colors.ENDC}")


def print_warning(message: str) -> None:
    """Print a warning message."""
    print(f"{Colors.YELLOW}⚠️ {message}{Colors.ENDC}")


def print_error(message: str) -> None:
    """Print an error message."""
    print(f"{Colors.RED}❌ {message}{Colors.ENDC}")


def print_info(message: str) -> None:
    """Print an info message."""
    print(f"{Colors.BLUE}ℹ️ {message}{Colors.ENDC}")


def check_environment() -> bool:
    """
    Check that all required services are running.
    
    Returns:
        True if all checks pass, False otherwise
    """
    print_header("Checking Environment")
    all_passed = True
    
    # Check Redis
    print_info("Checking Redis...")
    try:
        r = redis.Redis(
            host=os.environ.get("REDIS_HOST", "localhost"),
            port=int(os.environ.get("REDIS_PORT", 6379)),
            password=os.environ.get("REDIS_PASSWORD", None),
            socket_timeout=2,
        )
        r.ping()
        print_success("Redis is running")
    except Exception as e:
        print_error(f"Redis check failed: {e}")
        all_passed = False
    
    # Check Neo4j
    print_info("Checking Neo4j...")
    try:
        uri = f"neo4j://{os.environ.get('NEO4J_HOST', 'localhost')}:{os.environ.get('NEO4J_PORT', '7687')}"
        driver = GraphDatabase.driver(
            uri,
            auth=(
                os.environ.get("NEO4J_USERNAME", "neo4j"),
                os.environ.get("NEO4J_PASSWORD", "password")
            )
        )
        with driver.session() as session:
            result = session.run("RETURN 1 AS test")
            assert result.single()["test"] == 1
        driver.close()
        print_success("Neo4j is running")
    except Exception as e:
        print_error(f"Neo4j check failed: {e}")
        all_passed = False
    
    # Check FastAPI backend
    print_info("Checking FastAPI backend...")
    try:
        response = requests.get(
            f"http://{os.environ.get('API_HOST', 'localhost')}:{os.environ.get('API_PORT', '8000')}/health",
            timeout=5
        )
        if response.status_code == 200:
            print_success("FastAPI backend is running")
        else:
            print_error(f"FastAPI backend returned status {response.status_code}")
            all_passed = False
    except Exception as e:
        print_error(f"FastAPI backend check failed: {e}")
        all_passed = False
    
    # Check Celery workers
    print_info("Checking Celery workers...")
    try:
        response = requests.get(
            f"http://{os.environ.get('API_HOST', 'localhost')}:{os.environ.get('API_PORT', '8000')}/api/v1/health/workers",
            timeout=5
        )
        if response.status_code == 200:
            data = response.json()
            if data["overall_status"] == "HEALTHY":
                print_success("Celery workers are running")
            else:
                print_warning(f"Celery workers status: {data['overall_status']}")
                all_passed = False
        else:
            print_error(f"Celery workers check returned status {response.status_code}")
            all_passed = False
    except Exception as e:
        print_error(f"Celery workers check failed: {e}")
        all_passed = False
    
    return all_passed


def run_smoke_tests(verbose: bool = False, quick: bool = False) -> Tuple[bool, Dict]:
    """
    Run the smoke tests.
    
    Args:
        verbose: Whether to show detailed test output
        quick: Whether to run only essential tests
        
    Returns:
        Tuple of (all_passed, results)
    """
    print_header("Running Smoke Tests")
    
    # Prepare pytest arguments
    pytest_args = ["-xvs" if verbose else "-x"]
    
    # Add test file
    test_file = "tests/test_smoke_flow.py"
    pytest_args.append(test_file)
    
    # Add markers for quick test if needed
    if quick:
        pytest_args.extend(["-k", "test_health_endpoints or test_full_system_flow"])
    
    # Run pytest
    print_info(f"Running pytest with args: {' '.join(pytest_args)}")
    
    # Capture start time
    start_time = time.time()
    
    # Run pytest and capture output
    result = pytest.main(pytest_args)
    
    # Calculate duration
    duration = time.time() - start_time
    
    # Parse results
    all_passed = result == 0
    
    # Create results dictionary
    results = {
        "passed": all_passed,
        "exit_code": result,
        "duration_seconds": duration,
        "timestamp": datetime.now().isoformat(),
    }
    
    return all_passed, results


def print_troubleshooting_tips(results: Dict) -> None:
    """
    Print troubleshooting tips based on test results.
    
    Args:
        results: Test results dictionary
    """
    print_header("Troubleshooting Tips")
    
    if results["passed"]:
        print_success("All tests passed! No troubleshooting needed.")
        return
    
    print_info("Here are some tips to fix common issues:")
    
    # General tips
    print(f"\n{Colors.BOLD}General Tips:{Colors.ENDC}")
    print("1. Make sure all services are running:")
    print("   - Redis")
    print("   - Neo4j")
    print("   - FastAPI backend")
    print("   - Celery workers")
    print("\n2. Check environment variables:")
    print("   - REDIS_HOST, REDIS_PORT, REDIS_PASSWORD")
    print("   - NEO4J_HOST, NEO4J_PORT, NEO4J_USERNAME, NEO4J_PASSWORD")
    print("   - API_HOST, API_PORT")
    print("\n3. Check logs for errors:")
    print("   - Backend logs: docker logs analyst-droid-backend")
    print("   - Worker logs: docker logs analyst-droid-worker")
    print("   - Redis logs: docker logs analyst-droid-redis")
    print("   - Neo4j logs: docker logs analyst-droid-neo4j")
    
    # Specific tips based on exit code
    exit_code = results["exit_code"]
    if exit_code == 1:
        print(f"\n{Colors.BOLD}Specific Tips for Test Failures:{Colors.ENDC}")
        print("- Check that the database has some data (empty database can cause test failures)")
        print("- Verify API endpoints are accessible and returning expected data")
        print("- Ensure Celery workers are processing tasks")
    elif exit_code == 2:
        print(f"\n{Colors.BOLD}Specific Tips for Test Errors:{Colors.ENDC}")
        print("- Check for syntax errors or import errors in the codebase")
        print("- Verify that all dependencies are installed")
    elif exit_code == 3:
        print(f"\n{Colors.BOLD}Specific Tips for Test Collection Errors:{Colors.ENDC}")
        print("- Ensure test files are properly named and located")
        print("- Check for syntax errors in test files")
    elif exit_code == 4:
        print(f"\n{Colors.BOLD}Specific Tips for Usage Errors:{Colors.ENDC}")
        print("- Check pytest configuration")
        print("- Verify command line arguments")
    
    print(f"\n{Colors.BOLD}For More Help:{Colors.ENDC}")
    print("- Check the documentation in memory-bank/")
    print("- Run tests with --verbose for more detailed output")
    print("- Inspect the specific test failures above")


def quick_validation() -> bool:
    """
    Perform a quick validation of the platform.
    
    Returns:
        True if validation passes, False otherwise
    """
    print_header("Quick Platform Validation")
    
    try:
        # Check basic health endpoint
        print_info("Checking API health...")
        response = requests.get(
            f"http://{os.environ.get('API_HOST', 'localhost')}:{os.environ.get('API_PORT', '8000')}/health",
            timeout=5
        )
        if response.status_code != 200:
            print_error(f"API health check failed with status {response.status_code}")
            return False
        
        health_data = response.json()
        if health_data["status"] != "healthy":
            print_warning(f"API health status is {health_data['status']}")
        else:
            print_success("API health check passed")
        
        # Check system health
        print_info("Checking system health...")
        response = requests.get(
            f"http://{os.environ.get('API_HOST', 'localhost')}:{os.environ.get('API_PORT', '8000')}/api/v1/health/system",
            timeout=5
        )
        if response.status_code != 200:
            print_error(f"System health check failed with status {response.status_code}")
            return False
        
        system_health = response.json()
        if system_health["status"] != "healthy":
            print_warning(f"System health status is {system_health['status']}")
            if "issues" in system_health and system_health["issues"]:
                for issue in system_health["issues"]:
                    print_warning(f"  - {issue}")
        else:
            print_success("System health check passed")
        
        # Check component health
        components = system_health.get("components", {})
        all_healthy = True
        
        for component, status in components.items():
            component_status = status.get("status", "unknown")
            if component_status == "healthy":
                print_success(f"Component '{component}' is healthy")
            elif component_status == "degraded":
                print_warning(f"Component '{component}' is degraded")
                all_healthy = False
            else:
                print_error(f"Component '{component}' is {component_status}")
                all_healthy = False
        
        return all_healthy
    
    except Exception as e:
        print_error(f"Quick validation failed: {e}")
        return False


def print_summary(env_check_passed: bool, test_passed: bool, validation_passed: bool) -> None:
    """
    Print a summary of all checks.
    
    Args:
        env_check_passed: Whether environment checks passed
        test_passed: Whether smoke tests passed
        validation_passed: Whether quick validation passed
    """
    print_header("Summary")
    
    if env_check_passed:
        print_success("Environment checks: PASSED")
    else:
        print_error("Environment checks: FAILED")
    
    if test_passed:
        print_success("Smoke tests: PASSED")
    else:
        print_error("Smoke tests: FAILED")
    
    if validation_passed:
        print_success("Quick validation: PASSED")
    else:
        print_error("Quick validation: FAILED")
    
    # Overall status
    print("\n")
    if env_check_passed and test_passed and validation_passed:
        print_success("OVERALL STATUS: 🚀 PLATFORM IS FULLY OPERATIONAL 🚀")
    elif test_passed or validation_passed:
        print_warning("OVERALL STATUS: ⚠️ PLATFORM IS PARTIALLY OPERATIONAL ⚠️")
    else:
        print_error("OVERALL STATUS: ❌ PLATFORM IS NOT OPERATIONAL ❌")


def main():
    """Main entry point."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run smoke tests for Analyst Droid One platform")
    parser.add_argument("--verbose", action="store_true", help="Show detailed test output")
    parser.add_argument("--quick", action="store_true", help="Run only essential tests")
    parser.add_argument("--skip-env-check", action="store_true", help="Skip environment checks")
    args = parser.parse_args()
    
    # Print banner
    print_header("Analyst Droid One - Smoke Test Runner")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Mode: {'Quick' if args.quick else 'Full'}")
    print(f"Verbose: {'Yes' if args.verbose else 'No'}")
    
    # Check environment
    env_check_passed = True
    if not args.skip_env_check:
        env_check_passed = check_environment()
        if not env_check_passed and not args.quick:
            print_warning("Environment checks failed. Tests may not run correctly.")
            response = input("Continue anyway? (y/n): ")
            if response.lower() != "y":
                print_info("Exiting...")
                return
    
    # Run smoke tests
    test_passed, test_results = run_smoke_tests(verbose=args.verbose, quick=args.quick)
    
    # Quick validation
    validation_passed = quick_validation()
    
    # Print troubleshooting tips if needed
    if not test_passed:
        print_troubleshooting_tips(test_results)
    
    # Print summary
    print_summary(env_check_passed, test_passed, validation_passed)
    
    # Exit with appropriate status code
    sys.exit(0 if test_passed else 1)


if __name__ == "__main__":
    main()
