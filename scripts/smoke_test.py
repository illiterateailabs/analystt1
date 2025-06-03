#!/usr/bin/env python3
"""
Comprehensive Smoke Test for Analystt1 Platform

This script performs an end-to-end test of the Analystt1 platform, verifying:
1. Authentication & JWT handling
2. Template creation
3. Analysis execution
4. Results retrieval
5. All critical integration points (GraphQuery, GNN, CodeGen, PolicyDocs)
6. HITL workflow
7. Redis JWT blacklist persistence

Usage:
    python scripts/smoke_test.py [--host HOST] [--port PORT]

Requirements:
    pip install requests redis colorama
"""

import argparse
import json
import os
import sys
import time
import unittest
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

import requests
import redis
from colorama import Fore, Style, init

# Initialize colorama
init(autoreset=True)

# Default configuration
DEFAULT_HOST = "localhost"
DEFAULT_PORT = 8000
DEFAULT_USERNAME = "admin"
DEFAULT_PASSWORD = "admin123"
DEFAULT_REDIS_HOST = "localhost"
DEFAULT_REDIS_PORT = 6379
DEFAULT_REDIS_PASSWORD = "analyst123"
POLL_INTERVAL = 5  # seconds
MAX_POLL_TIME = 300  # seconds (5 minutes)

# Test template for fraud investigation
TEST_TEMPLATE = """
name: smoke_test_template
description: Template created by smoke test script
version: 1.0.0
agents:
  - id: investigator
    name: Lead Investigator
    goal: Orchestrate the fraud investigation workflow
    backstory: You are a senior financial crime analyst tasked with coordinating complex investigations.
    tools:
      - GraphQueryTool
      - GNNFraudDetectionTool
      - PatternLibraryTool
  - id: code_analyst
    name: Code Analyst
    goal: Generate visualizations and analysis code
    backstory: You are a data scientist who specializes in creating visual representations of fraud patterns.
    tools:
      - CodeGenTool
      - SandboxExecTool
  - id: compliance_officer
    name: Compliance Officer
    goal: Ensure all findings comply with regulations
    backstory: You ensure all investigations follow proper regulatory guidelines.
    tools:
      - PolicyDocsTool
  - id: report_writer
    name: Report Writer
    goal: Create a comprehensive investigation report
    backstory: You specialize in creating clear, actionable reports from complex investigations.
workflow:
  - agent: investigator
    tasks:
      - description: Extract relevant transaction subgraph
        expected_output: JSON with graph structure
  - agent: investigator
    tasks:
      - description: Run GNN fraud detection on the subgraph
        expected_output: JSON with fraud predictions
  - agent: code_analyst
    tasks:
      - description: Generate visualization code for the fraud patterns
        expected_output: Python code and visualization artifacts
  - agent: compliance_officer
    tasks:
      - description: Check findings against AML and KYC policies
        expected_output: Compliance report with policy references
  - agent: report_writer
    tasks:
      - description: Compile all findings into a comprehensive report
        expected_output: Markdown report with executive summary
"""


class SmokeTestResult:
    """Class to track test results and generate a report"""

    def __init__(self):
        self.start_time = datetime.now()
        self.end_time = None
        self.tests = []
        self.passed = 0
        self.failed = 0
        self.warnings = 0

    def add_result(self, name: str, passed: bool, message: str, warning: bool = False):
        """Add a test result"""
        self.tests.append({
            "name": name,
            "passed": passed,
            "message": message,
            "warning": warning,
            "timestamp": datetime.now()
        })
        if passed:
            if warning:
                self.warnings += 1
            else:
                self.passed += 1
        else:
            self.failed += 1

    def finish(self):
        """Mark the test suite as complete"""
        self.end_time = datetime.now()

    def print_report(self):
        """Print a formatted report of test results"""
        print("\n" + "=" * 80)
        print(f"{Fore.CYAN}ANALYSTT1 SMOKE TEST REPORT{Style.RESET_ALL}")
        print("=" * 80)
        
        duration = (self.end_time - self.start_time).total_seconds()
        print(f"Started: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Completed: {self.end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Duration: {duration:.2f} seconds")
        print(f"Results: {Fore.GREEN}{self.passed} passed{Style.RESET_ALL}, "
              f"{Fore.YELLOW}{self.warnings} warnings{Style.RESET_ALL}, "
              f"{Fore.RED}{self.failed} failed{Style.RESET_ALL}")
        print("-" * 80)
        
        for i, test in enumerate(self.tests, 1):
            if test["passed"]:
                if test["warning"]:
                    status = f"{Fore.YELLOW}WARNING{Style.RESET_ALL}"
                else:
                    status = f"{Fore.GREEN}PASS{Style.RESET_ALL}"
            else:
                status = f"{Fore.RED}FAIL{Style.RESET_ALL}"
                
            print(f"{i:2d}. [{status}] {test['name']}")
            print(f"    {test['message']}")
            
        print("=" * 80)
        if self.failed == 0:
            if self.warnings > 0:
                print(f"{Fore.YELLOW}SMOKE TEST PASSED WITH WARNINGS{Style.RESET_ALL}")
            else:
                print(f"{Fore.GREEN}SMOKE TEST PASSED SUCCESSFULLY{Style.RESET_ALL}")
        else:
            print(f"{Fore.RED}SMOKE TEST FAILED{Style.RESET_ALL}")
        print("=" * 80)

    def save_report(self, filename: str = "smoke_test_report.json"):
        """Save the test results to a JSON file"""
        report = {
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "duration_seconds": (self.end_time - self.start_time).total_seconds(),
            "tests_passed": self.passed,
            "tests_warnings": self.warnings,
            "tests_failed": self.failed,
            "tests": [{
                "name": t["name"],
                "passed": t["passed"],
                "warning": t["warning"],
                "message": t["message"],
                "timestamp": t["timestamp"].isoformat()
            } for t in self.tests]
        }
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Report saved to {filename}")


class APIClient:
    """Client for interacting with the Analystt1 API"""

    def __init__(self, host: str, port: int):
        self.base_url = f"http://{host}:{port}"
        self.session = requests.Session()
        self.access_token = None
        self.refresh_token = None

    def login(self, username: str, password: str) -> Tuple[bool, str]:
        """Authenticate with the API and get JWT tokens"""
        try:
            response = self.session.post(
                f"{self.base_url}/api/v1/auth/login",
                json={"username": username, "password": password}
            )
            
            if response.status_code == 200:
                data = response.json()
                self.access_token = data.get("access_token")
                self.refresh_token = data.get("refresh_token")
                return True, "Authentication successful"
            else:
                return False, f"Authentication failed: {response.status_code} - {response.text}"
        except Exception as e:
            return False, f"Authentication error: {str(e)}"

    def create_template(self, yaml_content: str) -> Tuple[bool, str, Optional[str]]:
        """Create a new investigation template"""
        if not self.access_token:
            return False, "Not authenticated", None
            
        try:
            headers = {"Authorization": f"Bearer {self.access_token}"}
            response = self.session.post(
                f"{self.base_url}/api/v1/templates",
                headers=headers,
                files={"yaml": ("template.yaml", yaml_content, "text/yaml")}
            )
            
            if response.status_code in (200, 201):
                data = response.json()
                template_id = data.get("id") or data.get("template_id")
                template_name = data.get("name") or "smoke_test_template"
                return True, f"Template created with ID: {template_id}", template_name
            else:
                return False, f"Template creation failed: {response.status_code} - {response.text}", None
        except Exception as e:
            return False, f"Template creation error: {str(e)}", None

    def run_analysis(self, template_name: str) -> Tuple[bool, str, Optional[str]]:
        """Start an analysis using the specified template"""
        if not self.access_token:
            return False, "Not authenticated", None
            
        try:
            headers = {"Authorization": f"Bearer {self.access_token}"}
            response = self.session.post(
                f"{self.base_url}/api/v1/analysis",
                headers=headers,
                json={"template": template_name}
            )
            
            if response.status_code in (200, 201):
                data = response.json()
                task_id = data.get("task_id")
                return True, f"Analysis started with task ID: {task_id}", task_id
            else:
                return False, f"Analysis start failed: {response.status_code} - {response.text}", None
        except Exception as e:
            return False, f"Analysis start error: {str(e)}", None

    def get_analysis_status(self, task_id: str) -> Tuple[bool, str, Optional[str]]:
        """Get the status of an analysis task"""
        if not self.access_token:
            return False, "Not authenticated", None
            
        try:
            headers = {"Authorization": f"Bearer {self.access_token}"}
            response = self.session.get(
                f"{self.base_url}/api/v1/analysis/{task_id}/status",
                headers=headers
            )
            
            if response.status_code == 200:
                data = response.json()
                status = data.get("status")
                return True, f"Analysis status: {status}", status
            else:
                return False, f"Status check failed: {response.status_code} - {response.text}", None
        except Exception as e:
            return False, f"Status check error: {str(e)}", None

    def get_analysis_results(self, task_id: str) -> Tuple[bool, str, Optional[Dict]]:
        """Get the results of a completed analysis"""
        if not self.access_token:
            return False, "Not authenticated", None
            
        try:
            headers = {"Authorization": f"Bearer {self.access_token}"}
            response = self.session.get(
                f"{self.base_url}/api/v1/analysis/{task_id}/results",
                headers=headers
            )
            
            if response.status_code == 200:
                data = response.json()
                return True, "Results retrieved successfully", data
            else:
                return False, f"Results retrieval failed: {response.status_code} - {response.text}", None
        except Exception as e:
            return False, f"Results retrieval error: {str(e)}", None

    def create_hitl_review(self, task_id: str, review_type: str = "compliance", 
                         risk_level: str = "medium") -> Tuple[bool, str, Optional[str]]:
        """Create a HITL review for an analysis task"""
        if not self.access_token:
            return False, "Not authenticated", None
            
        try:
            headers = {"Authorization": f"Bearer {self.access_token}"}
            response = self.session.post(
                f"{self.base_url}/api/v1/hitl/reviews",
                headers=headers,
                json={
                    "task_id": task_id,
                    "review_type": review_type,
                    "risk_level": risk_level
                }
            )
            
            if response.status_code in (200, 201):
                data = response.json()
                review_id = data.get("id")
                return True, f"HITL review created with ID: {review_id}", review_id
            else:
                return False, f"HITL review creation failed: {response.status_code} - {response.text}", None
        except Exception as e:
            return False, f"HITL review creation error: {str(e)}", None

    def approve_hitl_review(self, review_id: str, comments: str = "Approved by smoke test") -> Tuple[bool, str]:
        """Approve a HITL review"""
        if not self.access_token:
            return False, "Not authenticated"
            
        try:
            headers = {"Authorization": f"Bearer {self.access_token}"}
            response = self.session.post(
                f"{self.base_url}/api/v1/hitl/reviews/{review_id}/approve",
                headers=headers,
                json={"comments": comments}
            )
            
            if response.status_code == 200:
                return True, "HITL review approved successfully"
            else:
                return False, f"HITL review approval failed: {response.status_code} - {response.text}"
        except Exception as e:
            return False, f"HITL review approval error: {str(e)}"

    def blacklist_token(self) -> Tuple[bool, str]:
        """Blacklist the current access token by logging out"""
        if not self.access_token:
            return False, "Not authenticated"
            
        try:
            headers = {"Authorization": f"Bearer {self.access_token}"}
            response = self.session.post(
                f"{self.base_url}/api/v1/auth/logout",
                headers=headers
            )
            
            if response.status_code == 200:
                old_token = self.access_token
                self.access_token = None
                return True, f"Token blacklisted: {old_token[:10]}..."
            else:
                return False, f"Token blacklist failed: {response.status_code} - {response.text}"
        except Exception as e:
            return False, f"Token blacklist error: {str(e)}"

    def verify_token_blacklisted(self, token: str) -> Tuple[bool, str]:
        """Verify a token is blacklisted by trying to use it"""
        try:
            headers = {"Authorization": f"Bearer {token}"}
            response = self.session.get(
                f"{self.base_url}/api/v1/auth/verify",
                headers=headers
            )
            
            # If the token is properly blacklisted, we should get a 401 Unauthorized
            if response.status_code == 401:
                return True, "Token correctly blacklisted (401 response)"
            else:
                return False, f"Token not properly blacklisted: got {response.status_code} instead of 401"
        except Exception as e:
            return False, f"Token blacklist verification error: {str(e)}"


class RedisClient:
    """Client for interacting with Redis to check JWT blacklist"""
    
    def __init__(self, host: str, port: int, password: str = None):
        self.redis = redis.Redis(
            host=host,
            port=port,
            password=password,
            decode_responses=True
        )
        
    def check_connection(self) -> Tuple[bool, str]:
        """Check if Redis connection is working"""
        try:
            if self.redis.ping():
                return True, "Redis connection successful"
            return False, "Redis ping failed"
        except Exception as e:
            return False, f"Redis connection error: {str(e)}"
    
    def check_aof_enabled(self) -> Tuple[bool, str]:
        """Check if Redis AOF is enabled"""
        try:
            config = self.redis.config_get("appendonly")
            if config.get("appendonly") == "yes":
                return True, "Redis AOF is enabled"
            return False, f"Redis AOF is not enabled: {config}"
        except Exception as e:
            return False, f"Redis AOF check error: {str(e)}"
    
    def check_token_blacklisted(self, token: str) -> Tuple[bool, str]:
        """Check if a token is in the blacklist"""
        try:
            # The blacklist key format might vary based on implementation
            # Common formats: "blacklist:{token}", "token:{token}:blacklisted", etc.
            formats = [
                f"blacklist:{token}",
                f"token:{token}:blacklisted",
                f"jwt:blacklist:{token}"
            ]
            
            for key_format in formats:
                if self.redis.exists(key_format):
                    return True, f"Token found in Redis blacklist: {key_format}"
            
            # If we can't find the exact format, try a pattern search
            keys = self.redis.keys("*blacklist*")
            if keys:
                return True, f"Blacklist keys exist in Redis: {keys[:3]}..."
            
            return False, "Token not found in Redis blacklist"
        except Exception as e:
            return False, f"Redis blacklist check error: {str(e)}"


def run_smoke_test(args):
    """Run the complete smoke test suite"""
    results = SmokeTestResult()
    
    print(f"{Fore.CYAN}Starting Analystt1 Smoke Test{Style.RESET_ALL}")
    print(f"Host: {args.host}, Port: {args.port}")
    
    # Initialize API client
    api = APIClient(args.host, args.port)
    
    # Step 1: Authentication
    print(f"\n{Fore.CYAN}[1/8] Testing Authentication{Style.RESET_ALL}")
    success, message = api.login(args.username, args.password)
    results.add_result("Authentication", success, message)
    
    if not success:
        print(f"{Fore.RED}Authentication failed. Aborting smoke test.{Style.RESET_ALL}")
        results.finish()
        results.print_report()
        return False
    
    # Step 2: Template Creation
    print(f"\n{Fore.CYAN}[2/8] Testing Template Creation{Style.RESET_ALL}")
    success, message, template_name = api.create_template(TEST_TEMPLATE)
    results.add_result("Template Creation", success, message)
    
    if not success or not template_name:
        print(f"{Fore.RED}Template creation failed. Aborting smoke test.{Style.RESET_ALL}")
        results.finish()
        results.print_report()
        return False
    
    # Step 3: Analysis Execution
    print(f"\n{Fore.CYAN}[3/8] Testing Analysis Execution{Style.RESET_ALL}")
    success, message, task_id = api.run_analysis(template_name)
    results.add_result("Analysis Execution", success, message)
    
    if not success or not task_id:
        print(f"{Fore.RED}Analysis execution failed. Aborting smoke test.{Style.RESET_ALL}")
        results.finish()
        results.print_report()
        return False
    
    # Step 4: Status Polling
    print(f"\n{Fore.CYAN}[4/8] Polling Analysis Status{Style.RESET_ALL}")
    start_time = time.time()
    final_status = None
    
    while time.time() - start_time < MAX_POLL_TIME:
        success, message, status = api.get_analysis_status(task_id)
        
        if not success:
            print(f"{Fore.RED}Status polling failed: {message}{Style.RESET_ALL}")
            break
        
        print(f"Current status: {status} (polling for {int(time.time() - start_time)}s)")
        
        if status in ("completed", "done", "finished"):
            final_status = status
            break
        elif status in ("failed", "error"):
            final_status = status
            break
        
        time.sleep(POLL_INTERVAL)
    
    if final_status in ("completed", "done", "finished"):
        results.add_result("Analysis Completion", True, f"Analysis completed successfully: {final_status}")
    elif final_status in ("failed", "error"):
        results.add_result("Analysis Completion", False, f"Analysis failed with status: {final_status}")
    else:
        results.add_result("Analysis Completion", False, f"Analysis timed out after {MAX_POLL_TIME}s, last status: {status}")
        
    # Step 5: Results Validation
    if final_status in ("completed", "done", "finished"):
        print(f"\n{Fore.CYAN}[5/8] Validating Results{Style.RESET_ALL}")
        success, message, result_data = api.get_analysis_results(task_id)
        
        if success and result_data:
            # Check for expected result structure
            validation_results = []
            
            # Check for executive summary or risk score
            if "risk_score" in result_data or "executive_summary" in result_data:
                validation_results.append("Has risk score or executive summary")
            
            # Check for graph data
            if "graph_data" in result_data or "nodes" in result_data:
                validation_results.append("Has graph data")
            
            # Check for visualizations
            if "visualizations" in result_data:
                validation_results.append("Has visualizations")
                
            # Check for findings
            if "findings" in result_data:
                validation_results.append("Has findings")
                
            # Check for compliance data
            if "compliance" in result_data or "policy_references" in result_data:
                validation_results.append("Has compliance data")
            
            if len(validation_results) >= 3:
                results.add_result("Results Structure", True, 
                                 f"Results contain expected data: {', '.join(validation_results)}")
            else:
                results.add_result("Results Structure", False, 
                                 f"Results missing expected data. Found only: {', '.join(validation_results)}")
        else:
            results.add_result("Results Retrieval", False, message)
    
    # Step 6: Test HITL Workflow
    print(f"\n{Fore.CYAN}[6/8] Testing HITL Workflow{Style.RESET_ALL}")
    success, message, review_id = api.create_hitl_review(task_id)
    results.add_result("HITL Review Creation", success, message)
    
    if success and review_id:
        success, message = api.approve_hitl_review(review_id)
        results.add_result("HITL Review Approval", success, message)
    
    # Step 7: Test JWT Blacklist via API
    print(f"\n{Fore.CYAN}[7/8] Testing JWT Blacklist via API{Style.RESET_ALL}")
    
    # Store the token before blacklisting
    old_token = api.access_token
    
    if old_token:
        success, message = api.blacklist_token()
        results.add_result("JWT Blacklist Creation", success, message)
        
        if success:
            # Try to use the blacklisted token
            success, message = api.verify_token_blacklisted(old_token)
            results.add_result("JWT Blacklist Validation", success, message)
            
            # Login again to get a new token for remaining operations
            success, _ = api.login(args.username, args.password)
            if not success:
                results.add_result("Re-authentication", False, "Failed to re-authenticate after token blacklisting")
    
    # Step 8: Test Redis AOF for JWT Blacklist
    print(f"\n{Fore.CYAN}[8/8] Testing Redis AOF for JWT Blacklist{Style.RESET_ALL}")
    
    # Connect to Redis
    redis_client = RedisClient(args.redis_host, args.redis_port, args.redis_password)
    
    # Check Redis connection
    success, message = redis_client.check_connection()
    results.add_result("Redis Connection", success, message)
    
    if success:
        # Check if AOF is enabled
        success, message = redis_client.check_aof_enabled()
        results.add_result("Redis AOF Enabled", success, message)
        
        # Check if the blacklisted token is in Redis
        if old_token:
            success, message = redis_client.check_token_blacklisted(old_token)
            results.add_result("Redis Blacklist Storage", success, message, warning=not success)
    
    # Complete the test and print report
    results.finish()
    results.print_report()
    results.save_report()
    
    return results.failed == 0


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Analystt1 Smoke Test")
    parser.add_argument("--host", default=DEFAULT_HOST, help=f"API host (default: {DEFAULT_HOST})")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"API port (default: {DEFAULT_PORT})")
    parser.add_argument("--username", default=DEFAULT_USERNAME, help=f"Username (default: {DEFAULT_USERNAME})")
    parser.add_argument("--password", default=DEFAULT_PASSWORD, help=f"Password (default: {DEFAULT_PASSWORD})")
    parser.add_argument("--redis-host", default=DEFAULT_REDIS_HOST, help=f"Redis host (default: {DEFAULT_REDIS_HOST})")
    parser.add_argument("--redis-port", type=int, default=DEFAULT_REDIS_PORT, help=f"Redis port (default: {DEFAULT_REDIS_PORT})")
    parser.add_argument("--redis-password", default=DEFAULT_REDIS_PASSWORD, help=f"Redis password (default: {DEFAULT_REDIS_PASSWORD})")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    success = run_smoke_test(args)
    sys.exit(0 if success else 1)
