#!/usr/bin/env python3
"""
End-to-End Load Testing Framework for Coding Analyst Droid

This script provides a comprehensive end-to-end load testing framework for the
Coding Analyst Droid platform. It simulates realistic usage patterns, including
blockchain data ingestion, API endpoint interactions, Neo4j operations, Redis
caching, and agent/crew execution under various load conditions.

Features:
1.  **Scalable Load Generation**: Simulates 1M rows of data ingestion and 100k agent queries.
2.  **Realistic Data Simulation**: Generates synthetic blockchain data mimicking real-world patterns.
3.  **Full Stack Testing**: Tests API endpoints, Neo4j, and Redis caching under load.
4.  **Agent/Crew Load Testing**: Includes scenarios for testing multi-agent system execution.
5.  **Performance Monitoring**: Monitors key performance metrics (latency, throughput, error rates)
    and resource utilization (CPU, memory).
6.  **Comprehensive Reporting**: Generates detailed load testing reports with summaries and
    optimization recommendations.
7.  **Configurable Scenarios**: Supports different load profiles (concurrent users, ramp-up, duration).
8.  **Concurrency Testing**: Utilizes asyncio for high-throughput concurrent request simulation.
9.  **Stability Validation**: Tracks system stability and error rates under stress.
10. **Benchmarking**: Provides data for performance benchmarking and identifying bottlenecks.

Usage:
    python scripts/end_to_end_load_test.py --duration 60 --users 10 --data-rows 10000 --agent-queries 1000
    python scripts/end_to_end_load_test.py --profile heavy_ingestion --report-file load_test_report.json

Options:
    --duration SECONDS  Duration of the test in seconds (default: 300)
    --users USERS       Number of concurrent simulated users (default: 5)
    --data-rows ROWS    Number of blockchain data rows to ingest (default: 100000)
    --agent-queries QUERIES Number of agent queries to simulate (default: 10000)
    --base-url URL      Base URL of the FastAPI application (default: http://localhost:8000)
    --profile PROFILE   Predefined load testing profile (default: default)
    --report-file FILE  Path to save the JSON report (default: load_test_report.json)
    --verbose           Enable verbose logging
    --no-cleanup        Do not clean up generated data after test
    --help              Show this help message and exit
"""

import argparse
import asyncio
import json
import logging
import os
import random
import time
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

import httpx
import psutil  # For system resource monitoring

# Add backend to sys.path to allow imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.core.neo4j_loader import Balance, Activity, Token, Relationship, ChainType
from backend.core.redis_client import RedisDb

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Global Test Metrics ---
total_requests = 0
successful_requests = 0
failed_requests = 0
request_latencies: List[float] = []
error_details: List[Dict[str, Any]] = []
resource_metrics: List[Dict[str, Any]] = []

# --- Configuration Profiles ---
LOAD_PROFILES = {
    "default": {
        "duration": 300,  # 5 minutes
        "users": 5,
        "data_rows": 100000,
        "agent_queries": 10000,
    },
    "light_smoke": {
        "duration": 60,
        "users": 1,
        "data_rows": 1000,
        "agent_queries": 100,
    },
    "heavy_ingestion": {
        "duration": 600,  # 10 minutes
        "users": 10,
        "data_rows": 1000000,  # 1M rows
        "agent_queries": 5000,
    },
    "high_concurrency_queries": {
        "duration": 300,
        "users": 50,
        "data_rows": 10000,
        "agent_queries": 100000,  # 100k queries
    },
    "stress_test": {
        "duration": 900, # 15 minutes
        "users": 100,
        "data_rows": 2000000, # 2M rows
        "agent_queries": 200000, # 200k queries
    }
}

# --- Helper Functions ---

def generate_random_address() -> str:
    """Generates a random Ethereum-like address."""
    return '0x' + ''.join(random.choices('0123456789abcdef', k=40))

def generate_synthetic_blockchain_data(num_rows: int) -> List[Union[Balance, Activity, Token]]:
    """Generates synthetic blockchain data for ingestion."""
    data = []
    chains = list(ChainType)
    assets = ["ETH", "USDT", "DAI", "BTC", "LINK"]

    for i in range(num_rows):
        record_type = random.choice(["balance", "activity", "token"])
        chain = random.choice(chains)

        if record_type == "balance":
            data.append(Balance(
                address=generate_random_address(),
                chain=chain,
                asset=random.choice(assets),
                amount=random.uniform(0.01, 10000),
                usd_value=random.uniform(0.01, 100000),
                timestamp=datetime.now() - timedelta(days=random.randint(0, 365)),
            ))
        elif record_type == "activity":
            data.append(Activity(
                tx_hash='0x' + ''.join(random.choices('0123456789abcdef', k=64)),
                chain=chain,
                from_address=generate_random_address(),
                to_address=generate_random_address(),
                asset=random.choice(assets),
                amount=random.uniform(0.001, 5000),
                usd_value=random.uniform(0.001, 50000),
                timestamp=datetime.now() - timedelta(days=random.randint(0, 365)),
                block_height=random.randint(10000000, 18000000),
            ))
        elif record_type == "token":
            data.append(Token(
                address=generate_random_address(),
                chain=chain,
                name=f"Token{i}",
                symbol=f"TKN{i}",
                decimals=random.randint(0, 18),
            ))
    logger.info(f"Generated {len(data)} synthetic blockchain data rows.")
    return data

async def ingest_data_load_test(client: httpx.AsyncClient, base_url: str, data_rows: int):
    """Simulates load for data ingestion."""
    global total_requests, successful_requests, failed_requests, request_latencies, error_details

    ingestion_endpoint = f"{base_url}/api/v1/ingest/blockchain_data" # Assuming such an endpoint exists
    
    # Generate data in chunks to avoid excessive memory usage
    chunk_size = 1000
    num_chunks = (data_rows + chunk_size - 1) // chunk_size
    
    logger.info(f"Starting data ingestion load test for {data_rows} rows in {num_chunks} chunks.")

    for i in range(num_chunks):
        chunk = generate_synthetic_blockchain_data(min(chunk_size, data_rows - i * chunk_size))
        payload = [item.dict() for item in chunk] # Convert Pydantic models to dicts

        start_time = time.perf_counter()
        try:
            response = await client.post(ingestion_endpoint, json=payload, timeout=60.0) # Increased timeout for large payloads
            response.raise_for_status()
            successful_requests += 1
        except httpx.HTTPStatusError as e:
            failed_requests += 1
            error_details.append({"type": "HTTPStatusError", "message": str(e), "status_code": e.response.status_code, "endpoint": ingestion_endpoint})
            logger.error(f"Ingestion failed with HTTP error: {e.response.status_code} - {e.response.text}")
        except httpx.RequestError as e:
            failed_requests += 1
            error_details.append({"type": "RequestError", "message": str(e), "endpoint": ingestion_endpoint})
            logger.error(f"Ingestion failed with request error: {e}")
        except Exception as e:
            failed_requests += 1
            error_details.append({"type": "GeneralError", "message": str(e), "endpoint": ingestion_endpoint})
            logger.error(f"Ingestion failed with general error: {e}")
        finally:
            end_time = time.perf_counter()
            request_latencies.append(end_time - start_time)
            total_requests += 1
            if (i + 1) % 10 == 0:
                logger.info(f"Ingestion progress: {i+1}/{num_chunks} chunks processed.")

async def simulate_agent_query_load(client: httpx.AsyncClient, base_url: str, num_queries: int):
    """Simulates load for agent/crew queries."""
    global total_requests, successful_requests, failed_requests, request_latencies, error_details

    query_endpoint = f"{base_url}/api/v1/analysis/fraud_detection" # Example endpoint for agent query
    
    logger.info(f"Starting agent query load test for {num_queries} queries.")

    for i in range(num_queries):
        # Simulate a simple agent query payload
        payload = {
            "query": f"Analyze recent transactions for address {generate_random_address()} on {random.choice(list(ChainType)).value}",
            "parameters": {
                "target_addresses": [generate_random_address()],
                "chains": [random.choice(list(ChainType)).value],
                "time_range": {
                    "start_date": (datetime.now() - timedelta(days=30)).isoformat(),
                    "end_date": datetime.now().isoformat(),
                }
            }
        }

        start_time = time.perf_counter()
        try:
            response = await client.post(query_endpoint, json=payload, timeout=30.0)
            response.raise_for_status()
            successful_requests += 1
        except httpx.HTTPStatusError as e:
            failed_requests += 1
            error_details.append({"type": "HTTPStatusError", "message": str(e), "status_code": e.response.status_code, "endpoint": query_endpoint})
            logger.error(f"Agent query failed with HTTP error: {e.response.status_code} - {e.response.text}")
        except httpx.RequestError as e:
            failed_requests += 1
            error_details.append({"type": "RequestError", "message": str(e), "endpoint": query_endpoint})
            logger.error(f"Agent query failed with request error: {e}")
        except Exception as e:
            failed_requests += 1
            error_details.append({"type": "GeneralError", "message": str(e), "endpoint": query_endpoint})
            logger.error(f"Agent query failed with general error: {e}")
        finally:
            end_time = time.perf_counter()
            request_latencies.append(end_time - start_time)
            total_requests += 1
            if (i + 1) % 1000 == 0:
                logger.info(f"Agent query progress: {i+1}/{num_queries} queries processed.")

async def simulate_user_behavior(client: httpx.AsyncClient, base_url: str, duration: int, data_rows_per_user: int, agent_queries_per_user: int):
    """Simulates a single user's behavior over the test duration."""
    end_time = time.time() + duration
    
    # Distribute data ingestion and agent queries over the duration
    ingestion_interval = duration / data_rows_per_user if data_rows_per_user > 0 else float('inf')
    query_interval = duration / agent_queries_per_user if agent_queries_per_user > 0 else float('inf')

    ingestion_task = None
    query_task = None

    if data_rows_per_user > 0:
        ingestion_task = asyncio.create_task(ingest_data_load_test(client, base_url, data_rows_per_user))
    if agent_queries_per_user > 0:
        query_task = asyncio.create_task(simulate_agent_query_load(client, base_url, agent_queries_per_user))

    # Keep tasks running until duration ends
    while time.time() < end_time:
        await asyncio.sleep(1) # Check every second

    # Wait for tasks to complete if they haven't already
    if ingestion_task:
        await ingestion_task
    if query_task:
        await query_task

async def monitor_resources(interval: float = 1.0):
    """Monitors system resource utilization."""
    global resource_metrics
    process = psutil.Process(os.getpid()) # Monitor current process

    while True:
        cpu_percent = psutil.cpu_percent(interval=None) # Non-blocking
        memory_info = process.memory_info()
        
        resource_metrics.append({
            "timestamp": datetime.now().isoformat(),
            "cpu_percent": cpu_percent,
            "memory_rss_mb": memory_info.rss / (1024 * 1024),
            "memory_vms_mb": memory_info.vms / (1024 * 1024),
        })
        
        await asyncio.sleep(interval)

async def test_api_endpoints(client: httpx.AsyncClient, base_url: str, num_requests: int = 100):
    """Tests various API endpoints under load."""
    global total_requests, successful_requests, failed_requests, request_latencies, error_details
    
    # Define endpoints to test
    endpoints = [
        {"method": "GET", "url": f"{base_url}/api/v1/providers", "payload": None},
        {"method": "GET", "url": f"{base_url}/api/v1/tools", "payload": None},
        {"method": "GET", "url": f"{base_url}/metrics", "payload": None},
    ]
    
    logger.info(f"Starting API endpoint load test for {num_requests} requests per endpoint.")
    
    for endpoint in endpoints:
        method = endpoint["method"]
        url = endpoint["url"]
        payload = endpoint["payload"]
        
        for i in range(num_requests):
            start_time = time.perf_counter()
            try:
                if method == "GET":
                    response = await client.get(url, timeout=10.0)
                elif method == "POST":
                    response = await client.post(url, json=payload, timeout=10.0)
                
                response.raise_for_status()
                successful_requests += 1
            except httpx.HTTPStatusError as e:
                failed_requests += 1
                error_details.append({"type": "HTTPStatusError", "message": str(e), "status_code": e.response.status_code, "endpoint": url})
                logger.error(f"API request failed with HTTP error: {e.response.status_code} - {e.response.text}")
            except httpx.RequestError as e:
                failed_requests += 1
                error_details.append({"type": "RequestError", "message": str(e), "endpoint": url})
                logger.error(f"API request failed with request error: {e}")
            except Exception as e:
                failed_requests += 1
                error_details.append({"type": "GeneralError", "message": str(e), "endpoint": url})
                logger.error(f"API request failed with general error: {e}")
            finally:
                end_time = time.perf_counter()
                request_latencies.append(end_time - start_time)
                total_requests += 1

async def test_redis_caching(client: httpx.AsyncClient, base_url: str, num_requests: int = 100):
    """Tests Redis caching under load."""
    global total_requests, successful_requests, failed_requests, request_latencies, error_details
    
    # Define a cacheable endpoint
    cache_endpoint = f"{base_url}/api/v1/cache_test" # Assuming such an endpoint exists
    
    logger.info(f"Starting Redis caching load test for {num_requests} requests.")
    
    # First request to prime the cache
    try:
        response = await client.get(cache_endpoint, timeout=10.0)
        response.raise_for_status()
    except Exception as e:
        logger.error(f"Failed to prime cache: {e}")
        return
    
    # Now test with repeated requests that should hit cache
    for i in range(num_requests):
        start_time = time.perf_counter()
        try:
            response = await client.get(cache_endpoint, timeout=10.0)
            response.raise_for_status()
            
            # Check for cache header (if implemented)
            cache_hit = "X-Cache-Hit" in response.headers and response.headers["X-Cache-Hit"] == "true"
            
            successful_requests += 1
        except httpx.HTTPStatusError as e:
            failed_requests += 1
            error_details.append({"type": "HTTPStatusError", "message": str(e), "status_code": e.response.status_code, "endpoint": cache_endpoint})
            logger.error(f"Cache request failed with HTTP error: {e.response.status_code} - {e.response.text}")
        except httpx.RequestError as e:
            failed_requests += 1
            error_details.append({"type": "RequestError", "message": str(e), "endpoint": cache_endpoint})
            logger.error(f"Cache request failed with request error: {e}")
        except Exception as e:
            failed_requests += 1
            error_details.append({"type": "GeneralError", "message": str(e), "endpoint": cache_endpoint})
            logger.error(f"Cache request failed with general error: {e}")
        finally:
            end_time = time.perf_counter()
            request_latencies.append(end_time - start_time)
            total_requests += 1

async def test_neo4j_operations(client: httpx.AsyncClient, base_url: str, num_requests: int = 100):
    """Tests Neo4j operations under load."""
    global total_requests, successful_requests, failed_requests, request_latencies, error_details
    
    # Define an endpoint that performs Neo4j operations
    neo4j_endpoint = f"{base_url}/api/v1/graph/query" # Assuming such an endpoint exists
    
    logger.info(f"Starting Neo4j operations load test for {num_requests} requests.")
    
    for i in range(num_requests):
        # Generate a simple Cypher query
        payload = {
            "query": "MATCH (n:Address) RETURN n LIMIT 10",
            "parameters": {}
        }
        
        start_time = time.perf_counter()
        try:
            response = await client.post(neo4j_endpoint, json=payload, timeout=15.0)
            response.raise_for_status()
            successful_requests += 1
        except httpx.HTTPStatusError as e:
            failed_requests += 1
            error_details.append({"type": "HTTPStatusError", "message": str(e), "status_code": e.response.status_code, "endpoint": neo4j_endpoint})
            logger.error(f"Neo4j request failed with HTTP error: {e.response.status_code} - {e.response.text}")
        except httpx.RequestError as e:
            failed_requests += 1
            error_details.append({"type": "RequestError", "message": str(e), "endpoint": neo4j_endpoint})
            logger.error(f"Neo4j request failed with request error: {e}")
        except Exception as e:
            failed_requests += 1
            error_details.append({"type": "GeneralError", "message": str(e), "endpoint": neo4j_endpoint})
            logger.error(f"Neo4j request failed with general error: {e}")
        finally:
            end_time = time.perf_counter()
            request_latencies.append(end_time - start_time)
            total_requests += 1

async def test_crew_execution(client: httpx.AsyncClient, base_url: str, num_requests: int = 10):
    """Tests CrewAI execution under load."""
    global total_requests, successful_requests, failed_requests, request_latencies, error_details
    
    crew_endpoint = f"{base_url}/api/v1/crew/execute" # Assuming such an endpoint exists
    
    logger.info(f"Starting CrewAI execution load test for {num_requests} requests.")
    
    for i in range(num_requests):
        # Generate a crew execution request
        payload = {
            "crew_id": "fraud_detection",
            "input": {
                "target_address": generate_random_address(),
                "chain": random.choice(list(ChainType)).value,
                "depth": 2
            }
        }
        
        start_time = time.perf_counter()
        try:
            response = await client.post(crew_endpoint, json=payload, timeout=60.0) # Longer timeout for crew execution
            response.raise_for_status()
            successful_requests += 1
        except httpx.HTTPStatusError as e:
            failed_requests += 1
            error_details.append({"type": "HTTPStatusError", "message": str(e), "status_code": e.response.status_code, "endpoint": crew_endpoint})
            logger.error(f"Crew execution failed with HTTP error: {e.response.status_code} - {e.response.text}")
        except httpx.RequestError as e:
            failed_requests += 1
            error_details.append({"type": "RequestError", "message": str(e), "endpoint": crew_endpoint})
            logger.error(f"Crew execution failed with request error: {e}")
        except Exception as e:
            failed_requests += 1
            error_details.append({"type": "GeneralError", "message": str(e), "endpoint": crew_endpoint})
            logger.error(f"Crew execution failed with general error: {e}")
        finally:
            end_time = time.perf_counter()
            request_latencies.append(end_time - start_time)
            total_requests += 1
            logger.info(f"Crew execution progress: {i+1}/{num_requests}")

def calculate_metrics():
    """Calculates performance metrics from the test data."""
    metrics = {
        "total_requests": total_requests,
        "successful_requests": successful_requests,
        "failed_requests": failed_requests,
        "success_rate": successful_requests / total_requests if total_requests > 0 else 0,
        "error_rate": failed_requests / total_requests if total_requests > 0 else 0,
        "latency": {
            "min_ms": min(request_latencies) * 1000 if request_latencies else 0,
            "max_ms": max(request_latencies) * 1000 if request_latencies else 0,
            "avg_ms": (sum(request_latencies) / len(request_latencies)) * 1000 if request_latencies else 0,
        },
        "throughput": {
            "requests_per_second": len(request_latencies) / sum(request_latencies) if request_latencies else 0,
        },
        "resource_utilization": {
            "cpu_percent": {
                "avg": sum(m["cpu_percent"] for m in resource_metrics) / len(resource_metrics) if resource_metrics else 0,
                "max": max(m["cpu_percent"] for m in resource_metrics) if resource_metrics else 0,
            },
            "memory_rss_mb": {
                "avg": sum(m["memory_rss_mb"] for m in resource_metrics) / len(resource_metrics) if resource_metrics else 0,
                "max": max(m["memory_rss_mb"] for m in resource_metrics) if resource_metrics else 0,
            },
        },
        "error_details": error_details[:10],  # Include only first 10 errors to avoid huge reports
    }
    
    # Calculate percentiles for latency
    if request_latencies:
        sorted_latencies = sorted(request_latencies)
        metrics["latency"]["p50_ms"] = sorted_latencies[int(len(sorted_latencies) * 0.5)] * 1000
        metrics["latency"]["p90_ms"] = sorted_latencies[int(len(sorted_latencies) * 0.9)] * 1000
        metrics["latency"]["p95_ms"] = sorted_latencies[int(len(sorted_latencies) * 0.95)] * 1000
        metrics["latency"]["p99_ms"] = sorted_latencies[int(len(sorted_latencies) * 0.99)] * 1000
    
    return metrics

def generate_optimization_recommendations(metrics: Dict[str, Any]) -> List[str]:
    """Generates optimization recommendations based on test metrics."""
    recommendations = []
    
    # Check latency
    if metrics["latency"]["p99_ms"] > 1000:  # 1 second
        recommendations.append("High p99 latency detected. Consider optimizing slow endpoints or database queries.")
    
    # Check error rate
    if metrics["error_rate"] > 0.05:  # 5%
        recommendations.append(f"High error rate ({metrics['error_rate']*100:.2f}%). Investigate error patterns and improve error handling.")
    
    # Check resource utilization
    if metrics["resource_utilization"]["cpu_percent"]["avg"] > 80:
        recommendations.append("High CPU utilization. Consider scaling horizontally or optimizing CPU-intensive operations.")
    
    if metrics["resource_utilization"]["memory_rss_mb"]["max"] > 1024:  # 1 GB
        recommendations.append("High memory usage. Check for memory leaks or optimize memory-intensive operations.")
    
    # Check throughput
    if metrics["throughput"]["requests_per_second"] < 10:
        recommendations.append("Low throughput. Consider optimizing request handling, database queries, or caching strategies.")
    
    # Add general recommendations
    recommendations.append("Consider implementing or tuning Redis caching for frequently accessed data.")
    recommendations.append("Optimize Neo4j queries by adding appropriate indexes and using efficient Cypher patterns.")
    recommendations.append("Consider implementing connection pooling for database connections.")
    recommendations.append("Implement or tune rate limiting to protect against traffic spikes.")
    
    return recommendations

def generate_report(metrics: Dict[str, Any], config: Dict[str, Any], duration: float) -> Dict[str, Any]:
    """Generates a comprehensive load test report."""
    recommendations = generate_optimization_recommendations(metrics)
    
    report = {
        "test_configuration": config,
        "test_duration_seconds": duration,
        "timestamp": datetime.now().isoformat(),
        "metrics": metrics,
        "optimization_recommendations": recommendations,
    }
    
    return report

def print_report_summary(report: Dict[str, Any]):
    """Prints a summary of the load test report to the console."""
    metrics = report["metrics"]
    config = report["test_configuration"]
    
    print("\n" + "="*80)
    print(f"LOAD TEST SUMMARY - {datetime.now().isoformat()}")
    print("="*80)
    
    print(f"\nTest Configuration:")
    print(f"  Duration: {config['duration']} seconds")
    print(f"  Users: {config['users']}")
    print(f"  Data Rows: {config['data_rows']}")
    print(f"  Agent Queries: {config['agent_queries']}")
    
    print(f"\nResults:")
    print(f"  Total Requests: {metrics['total_requests']}")
    print(f"  Success Rate: {metrics['success_rate']*100:.2f}%")
    print(f"  Error Rate: {metrics['error_rate']*100:.2f}%")
    print(f"  Throughput: {metrics['throughput']['requests_per_second']:.2f} req/s")
    
    print(f"\nLatency (ms):")
    print(f"  Average: {metrics['latency']['avg_ms']:.2f}")
    print(f"  p50: {metrics.get('latency', {}).get('p50_ms', 0):.2f}")
    print(f"  p90: {metrics.get('latency', {}).get('p90_ms', 0):.2f}")
    print(f"  p99: {metrics.get('latency', {}).get('p99_ms', 0):.2f}")
    
    print(f"\nResource Utilization:")
    print(f"  CPU Avg: {metrics['resource_utilization']['cpu_percent']['avg']:.2f}%")
    print(f"  CPU Max: {metrics['resource_utilization']['cpu_percent']['max']:.2f}%")
    print(f"  Memory Avg: {metrics['resource_utilization']['memory_rss_mb']['avg']:.2f} MB")
    print(f"  Memory Max: {metrics['resource_utilization']['memory_rss_mb']['max']:.2f} MB")
    
    print(f"\nOptimization Recommendations:")
    for i, rec in enumerate(report["optimization_recommendations"], 1):
        print(f"  {i}. {rec}")
    
    print("\n" + "="*80)

async def run_load_test(config: Dict[str, Any]):
    """Runs the end-to-end load test with the given configuration."""
    global total_requests, successful_requests, failed_requests, request_latencies, error_details, resource_metrics
    
    # Reset global metrics
    total_requests = 0
    successful_requests = 0
    failed_requests = 0
    request_latencies = []
    error_details = []
    resource_metrics = []
    
    # Extract configuration
    duration = config["duration"]
    num_users = config["users"]
    data_rows = config["data_rows"]
    agent_queries = config["agent_queries"]
    base_url = config["base_url"]
    
    logger.info(f"Starting load test with configuration: {config}")
    
    # Start resource monitoring
    monitor_task = asyncio.create_task(monitor_resources())
    
    # Calculate per-user workload
    data_rows_per_user = data_rows // num_users if num_users > 0 else 0
    agent_queries_per_user = agent_queries // num_users if num_users > 0 else 0
    
    # Create HTTP client
    limits = httpx.Limits(max_connections=100, max_keepalive_connections=20)
    async with httpx.AsyncClient(limits=limits, timeout=30.0) as client:
        # Create user simulation tasks
        user_tasks = []
        for i in range(num_users):
            user_tasks.append(simulate_user_behavior(
                client=client,
                base_url=base_url,
                duration=duration,
                data_rows_per_user=data_rows_per_user,
                agent_queries_per_user=agent_queries_per_user
            ))
        
        # Add additional test tasks
        test_tasks = [
            test_api_endpoints(client, base_url),
            test_redis_caching(client, base_url),
            test_neo4j_operations(client, base_url),
            test_crew_execution(client, base_url, num_requests=min(10, agent_queries // 100))  # Limit crew executions as they're heavy
        ]
        
        # Start all tasks
        all_tasks = user_tasks + test_tasks
        start_time = time.perf_counter()
        
        # Wait for tasks to complete or duration to end
        try:
            await asyncio.wait_for(asyncio.gather(*all_tasks), timeout=duration)
        except asyncio.TimeoutError:
            logger.info(f"Test duration of {duration} seconds reached.")
        
        end_time = time.perf_counter()
        actual_duration = end_time - start_time
        
        # Cancel resource monitoring
        monitor_task.cancel()
        try:
            await monitor_task
        except asyncio.CancelledError:
            pass
        
        # Calculate metrics
        metrics = calculate_metrics()
        
        # Generate report
        report = generate_report(metrics, config, actual_duration)
        
        return report

def save_report(report: Dict[str, Any], report_file: str):
    """Saves the load test report to a file."""
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    logger.info(f"Report saved to {report_file}")

def main():
    """Main entry point for the load testing script."""
    parser = argparse.ArgumentParser(description="End-to-end load testing for Coding Analyst Droid.")
    parser.add_argument("--duration", type=int, help="Duration of the test in seconds")
    parser.add_argument("--users", type=int, help="Number of concurrent simulated users")
    parser.add_argument("--data-rows", type=int, help="Number of blockchain data rows to ingest")
    parser.add_argument("--agent-queries", type=int, help="Number of agent queries to simulate")
    parser.add_argument("--base-url", type=str, default="http://localhost:8000", help="Base URL of the FastAPI application")
    parser.add_argument("--profile", type=str, choices=LOAD_PROFILES.keys(), default="default", help="Predefined load testing profile")
    parser.add_argument("--report-file", type=str, default="load_test_report.json", help="Path to save the JSON report")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--no-cleanup", action="store_true", help="Do not clean up generated data after test")
    
    args = parser.parse_args()
    
    # Set up logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load configuration from profile
    config = LOAD_PROFILES[args.profile].copy()
    
    # Override with command line arguments
    if args.duration:
        config["duration"] = args.duration
    if args.users:
        config["users"] = args.users
    if args.data_rows:
        config["data_rows"] = args.data_rows
    if args.agent_queries:
        config["agent_queries"] = args.agent_queries
    
    # Add base URL to config
    config["base_url"] = args.base_url
    
    # Run the load test
    report = asyncio.run(run_load_test(config))
    
    # Save and print report
    save_report(report, args.report_file)
    print_report_summary(report)
    
    # Clean up if needed
    if not args.no_cleanup:
        logger.info("Cleaning up generated data...")
        # In a real implementation, this would clean up test data
        # For example, by calling a cleanup endpoint or directly accessing the database
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
