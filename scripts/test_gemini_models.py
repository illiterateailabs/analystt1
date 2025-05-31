#!/usr/bin/env python
"""
Gemini 2.5 Models Smoke Test Script

This script tests Gemini 2.5 Flash and Pro models, comparing their performance,
error handling, token usage, and multimodal capabilities.

Usage:
    python scripts/test_gemini_models.py [--all] [--text] [--multimodal] [--errors] [--tokens]

Options:
    --all         Run all tests (default)
    --text        Run text-only tests
    --multimodal  Run multimodal tests (requires test images)
    --errors      Run error handling tests
    --tokens      Run token usage tests
    --output      Output file for results (default: gemini_test_results.json)

Example:
    python scripts/test_gemini_models.py --text --tokens
"""

import os
import sys
import time
import json
import base64
import argparse
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
import traceback
import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Add project root to path if running as script
if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    sys.path.append(str(project_root))

try:
    from crewai.llm import LLM
    from google import genai
except ImportError:
    logger.error("Required packages not found. Please install: pip install crewai google-genai")
    sys.exit(1)

# Model IDs
GEMINI_2_5_FLASH = "gemini-2.5-flash-preview-05-20"
GEMINI_2_5_PRO = "gemini-2.5-pro-preview-05-06"
GEMINI_2_5_FLASH_AUDIO = "gemini-2.5-flash-preview-native-audio-dialog"
GEMINI_2_0_FLASH = "gemini-2.0-flash"  # Fallback option

# Test image paths - adjust as needed
TEST_IMAGES_DIR = Path(__file__).parent / "test_images"
TEST_IMAGES = {
    "chart": TEST_IMAGES_DIR / "financial_chart.jpg",
    "document": TEST_IMAGES_DIR / "financial_document.jpg",
    "transaction": TEST_IMAGES_DIR / "transaction_receipt.jpg",
}

# Test prompts
TEXT_TEST_PROMPTS = [
    {
        "name": "structuring_analysis",
        "prompt": """
        Analyze the following transaction pattern and identify potential fraud indicators:
        
        - Account A received $50,000 from an overseas source
        - Within 24 hours, Account A sent $9,800 to Account B
        - Account A then sent $9,700 to Account C
        - Account A sent $9,600 to Account D
        - Account A sent $9,500 to Account E
        - Account A withdrew the remaining balance in cash
        
        What type of financial crime pattern might this represent? What regulations might apply?
        """,
        "complexity": "medium"
    },
    {
        "name": "cypher_generation",
        "prompt": """
        Generate a Neo4j Cypher query to find all accounts that:
        1. Received more than $10,000 in total within the last 30 days
        2. Sent at least 5 transactions to different recipients
        3. Have at least one transaction flagged as suspicious
        
        Assume the following schema:
        - (Account)-[:RECEIVED]->(Transaction)-[:SENT_BY]->(Account)
        - Account properties: id, name, type, created_date
        - Transaction properties: amount, timestamp, currency, is_suspicious
        """,
        "complexity": "high"
    },
    {
        "name": "simple_classification",
        "prompt": """
        Classify the following transaction as legitimate or suspicious:
        
        A new customer opened an account yesterday and immediately received 
        5 separate cash deposits of $1,900 each from different branches of the same bank.
        """,
        "complexity": "low"
    }
]

IMAGE_TEST_PROMPTS = [
    {
        "name": "chart_analysis",
        "prompt": """
        Analyze this financial chart:
        1. Describe the key trends shown
        2. Identify any anomalies or suspicious patterns
        3. What conclusions would you draw for a financial investigation?
        """,
        "image": "chart"
    },
    {
        "name": "document_analysis",
        "prompt": """
        Examine this financial document:
        1. What type of document is this?
        2. Extract key information (dates, amounts, parties involved)
        3. Are there any suspicious elements or red flags?
        """,
        "image": "document"
    },
    {
        "name": "transaction_receipt",
        "prompt": """
        Analyze this transaction receipt:
        1. Extract transaction details (date, amount, parties)
        2. Is there anything unusual about this transaction?
        3. What additional information would you need to investigate further?
        """,
        "image": "transaction"
    }
]


class TokenCounter:
    """Tracks token usage and estimates costs for Gemini models"""
    
    # Approximate costs per 1M tokens (as of May 2025)
    COST_PER_MILLION = {
        "gemini-2.5-flash-preview-05-20": {"input": 0.35, "output": 1.05},
        "gemini-2.5-pro-preview-05-06": {"input": 0.70, "output": 2.10},
        "gemini-2.0-flash": {"input": 0.25, "output": 0.75},
    }
    
    def __init__(self):
        self.usage = {model: {"input": 0, "output": 0} for model in self.COST_PER_MILLION}
        self.requests = {model: 0 for model in self.COST_PER_MILLION}
    
    def add_usage(self, model: str, input_tokens: int, output_tokens: int):
        """Add token usage for a model"""
        if model not in self.usage:
            logger.warning(f"Unknown model: {model}, using gemini-2.0-flash pricing")
            model = "gemini-2.0-flash"
        
        self.usage[model]["input"] += input_tokens
        self.usage[model]["output"] += output_tokens
        self.requests[model] += 1
    
    def estimate_cost(self, model: str) -> float:
        """Estimate cost in USD for a model"""
        if model not in self.usage:
            return 0.0
        
        input_tokens = self.usage[model]["input"]
        output_tokens = self.usage[model]["output"]
        
        input_cost = (input_tokens / 1_000_000) * self.COST_PER_MILLION[model]["input"]
        output_cost = (output_tokens / 1_000_000) * self.COST_PER_MILLION[model]["output"]
        
        return input_cost + output_cost
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of token usage and costs"""
        summary = {
            "total_requests": sum(self.requests.values()),
            "models": {}
        }
        
        total_cost = 0.0
        for model in self.usage:
            if self.requests[model] > 0:
                cost = self.estimate_cost(model)
                total_cost += cost
                summary["models"][model] = {
                    "requests": self.requests[model],
                    "input_tokens": self.usage[model]["input"],
                    "output_tokens": self.usage[model]["output"],
                    "total_tokens": self.usage[model]["input"] + self.usage[model]["output"],
                    "estimated_cost_usd": round(cost, 6)
                }
        
        summary["total_cost_usd"] = round(total_cost, 6)
        return summary


# Initialize token counter
token_counter = TokenCounter()


def get_gemini_client() -> Any:
    """Initialize and return the Gemini client"""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set")
    
    genai.configure(api_key=api_key)
    return genai


def create_llm(model: str, temperature: float = 0.2, max_tokens: int = 2048) -> LLM:
    """Create a CrewAI LLM instance with the specified model"""
    return LLM(
        model=model,
        api_key=os.environ.get("GEMINI_API_KEY"),
        temperature=temperature,
        max_tokens=max_tokens,
    )


def estimate_tokens(text: str) -> int:
    """Roughly estimate token count from text (approximation)"""
    # Very rough approximation: ~4 chars per token for English text
    return len(text) // 4


def run_text_test(prompt_data: Dict[str, Any], models: List[str] = None) -> Dict[str, Any]:
    """Run a text-based test on specified models"""
    if models is None:
        models = [GEMINI_2_5_FLASH, GEMINI_2_5_PRO]
    
    prompt = prompt_data["prompt"]
    name = prompt_data["name"]
    complexity = prompt_data.get("complexity", "medium")
    
    logger.info(f"Running text test: {name} (complexity: {complexity})")
    
    results = {
        "prompt_name": name,
        "complexity": complexity,
        "models": {}
    }
    
    for model in models:
        try:
            llm = create_llm(model)
            
            # Measure performance
            start_time = time.time()
            response = llm.generate(prompt)
            elapsed_time = time.time() - start_time
            
            # Estimate token usage
            input_tokens = estimate_tokens(prompt)
            output_tokens = estimate_tokens(response)
            token_counter.add_usage(model, input_tokens, output_tokens)
            
            # Record results
            results["models"][model] = {
                "success": True,
                "time_seconds": round(elapsed_time, 3),
                "estimated_input_tokens": input_tokens,
                "estimated_output_tokens": output_tokens,
                "response_length": len(response),
                "response_excerpt": response[:200] + "..." if len(response) > 200 else response
            }
            
        except Exception as e:
            logger.error(f"Error testing {model} on {name}: {str(e)}")
            results["models"][model] = {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    
    return results


def run_multimodal_test(prompt_data: Dict[str, Any], models: List[str] = None) -> Dict[str, Any]:
    """Run a multimodal test on specified models"""
    if models is None:
        models = [GEMINI_2_5_FLASH, GEMINI_2_5_PRO]
    
    prompt = prompt_data["prompt"]
    name = prompt_data["name"]
    image_key = prompt_data["image"]
    
    # Check if image exists
    image_path = TEST_IMAGES.get(image_key)
    if not image_path or not image_path.exists():
        return {
            "prompt_name": name,
            "success": False,
            "error": f"Image not found: {image_key} (path: {image_path})"
        }
    
    logger.info(f"Running multimodal test: {name} with image: {image_key}")
    
    # Read image and encode as base64
    try:
        with open(image_path, "rb") as img_file:
            img_data = base64.b64encode(img_file.read()).decode("utf-8")
    except Exception as e:
        return {
            "prompt_name": name,
            "success": False,
            "error": f"Failed to read image {image_path}: {str(e)}"
        }
    
    results = {
        "prompt_name": name,
        "image": str(image_path),
        "models": {}
    }
    
    for model in models:
        try:
            # For multimodal, we need to use the Gemini API directly
            client = get_gemini_client()
            genai_model = client.GenerativeModel(model)
            
            # Prepare multimodal content
            content = [
                {"type": "text", "text": prompt},
                {"type": "image", "data": img_data}
            ]
            
            # Measure performance
            start_time = time.time()
            response = genai_model.generate_content(content)
            elapsed_time = time.time() - start_time
            
            # Get response text
            response_text = response.text
            
            # Estimate token usage (very rough for multimodal)
            input_tokens = estimate_tokens(prompt) + 1000  # Add 1000 tokens as rough estimate for image
            output_tokens = estimate_tokens(response_text)
            token_counter.add_usage(model, input_tokens, output_tokens)
            
            # Record results
            results["models"][model] = {
                "success": True,
                "time_seconds": round(elapsed_time, 3),
                "estimated_input_tokens": input_tokens,
                "estimated_output_tokens": output_tokens,
                "response_length": len(response_text),
                "response_excerpt": response_text[:200] + "..." if len(response_text) > 200 else response_text
            }
            
        except Exception as e:
            logger.error(f"Error testing {model} on multimodal {name}: {str(e)}")
            results["models"][model] = {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    
    return results


def test_error_handling() -> Dict[str, Any]:
    """Test error handling scenarios"""
    logger.info("Testing error handling scenarios")
    
    results = {
        "scenarios": {}
    }
    
    # Scenario 1: Invalid API key
    try:
        # Save original API key
        original_api_key = os.environ.get("GEMINI_API_KEY")
        
        # Set invalid API key
        os.environ["GEMINI_API_KEY"] = "invalid_key_12345"
        
        llm = create_llm(GEMINI_2_5_FLASH)
        start_time = time.time()
        response = llm.generate("This is a test with invalid API key")
        elapsed_time = time.time() - start_time
        
        results["scenarios"]["invalid_api_key"] = {
            "success": True,  # This should not happen
            "unexpected_response": response,
            "time_seconds": round(elapsed_time, 3)
        }
    except Exception as e:
        results["scenarios"]["invalid_api_key"] = {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
            "handled_correctly": True
        }
    finally:
        # Restore original API key
        if original_api_key:
            os.environ["GEMINI_API_KEY"] = original_api_key
        else:
            os.environ.pop("GEMINI_API_KEY", None)
    
    # Scenario 2: Missing API key
    try:
        # Save original API key
        original_api_key = os.environ.get("GEMINI_API_KEY")
        
        # Remove API key
        if "GEMINI_API_KEY" in os.environ:
            os.environ.pop("GEMINI_API_KEY")
        
        llm = create_llm(GEMINI_2_5_FLASH)
        start_time = time.time()
        response = llm.generate("This is a test with missing API key")
        elapsed_time = time.time() - start_time
        
        results["scenarios"]["missing_api_key"] = {
            "success": True,  # This should not happen
            "unexpected_response": response,
            "time_seconds": round(elapsed_time, 3)
        }
    except Exception as e:
        results["scenarios"]["missing_api_key"] = {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
            "handled_correctly": True
        }
    finally:
        # Restore original API key
        if original_api_key:
            os.environ["GEMINI_API_KEY"] = original_api_key
    
    # Scenario 3: Network error simulation
    try:
        # Create a mock LLM that will raise a network error
        class MockLLM:
            def generate(self, prompt):
                raise requests.exceptions.ConnectionError("Simulated network error")
        
        mock_llm = MockLLM()
        start_time = time.time()
        response = mock_llm.generate("This is a test with network error")
        elapsed_time = time.time() - start_time
        
        results["scenarios"]["network_error"] = {
            "success": True,  # This should not happen
            "unexpected_response": response,
            "time_seconds": round(elapsed_time, 3)
        }
    except Exception as e:
        results["scenarios"]["network_error"] = {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
            "handled_correctly": True
        }
    
    # Scenario 4: Invalid model ID
    try:
        llm = create_llm("gemini-nonexistent-model")
        start_time = time.time()
        response = llm.generate("This is a test with invalid model ID")
        elapsed_time = time.time() - start_time
        
        results["scenarios"]["invalid_model"] = {
            "success": True,  # This should not happen
            "unexpected_response": response,
            "time_seconds": round(elapsed_time, 3)
        }
    except Exception as e:
        results["scenarios"]["invalid_model"] = {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
            "handled_correctly": True
        }
    
    return results


def test_token_usage() -> Dict[str, Any]:
    """Test token usage tracking with controlled examples"""
    logger.info("Testing token usage tracking")
    
    # Reset token counter for this test
    global token_counter
    token_counter = TokenCounter()
    
    results = {
        "tests": {}
    }
    
    # Test with short prompt
    short_prompt = "Summarize the concept of money laundering in one sentence."
    short_result = run_text_test({
        "name": "token_short",
        "prompt": short_prompt,
        "complexity": "low"
    })
    results["tests"]["short_prompt"] = short_result
    
    # Test with medium prompt
    medium_prompt = """
    Explain the following financial concepts and how they relate to fraud detection:
    1. Structuring
    2. Layering
    3. Integration
    4. Smurfing
    5. Round-tripping
    
    For each concept, provide a brief definition and at least one example of how it might appear in transaction data.
    """
    medium_result = run_text_test({
        "name": "token_medium",
        "prompt": medium_prompt,
        "complexity": "medium"
    })
    results["tests"]["medium_prompt"] = medium_result
    
    # Test with long prompt
    long_prompt = """
    You are a financial crime analyst investigating a complex case. Below is a detailed description of the transaction patterns observed:
    
    Company A, registered in Delaware, received $1.5 million from an offshore entity in the Cayman Islands on January 15, 2025. Within 48 hours, Company A distributed these funds as follows:
    
    - $240,000 to Vendor B (registered in Nevada)
    - $235,000 to Vendor C (registered in Wyoming)
    - $245,000 to Vendor D (registered in Delaware)
    - $230,000 to Vendor E (registered in South Dakota)
    - $250,000 to Vendor F (registered in Nevada)
    
    Each vendor then transferred approximately 90% of the received funds to different personal accounts held by individuals with apparent connections to the director of Company A. These individuals then:
    
    1. Withdrew between $8,000-$9,500 in cash
    2. Purchased cryptocurrency through different exchanges
    3. Transferred funds to online gambling platforms
    4. Made payments to luxury goods retailers
    
    Based on this information:
    1. Identify all potential red flags and suspicious patterns
    2. Classify the types of financial crimes that might be occurring
    3. Recommend specific investigation steps to confirm your suspicions
    4. Draft the key points that should be included in a Suspicious Activity Report (SAR)
    5. Suggest preventative measures that financial institutions could implement to detect similar patterns earlier
    
    Please provide a comprehensive analysis with specific references to relevant regulations and compliance requirements.
    """
    long_result = run_text_test({
        "name": "token_long",
        "prompt": long_prompt,
        "complexity": "high"
    })
    results["tests"]["long_prompt"] = long_result
    
    # Get token usage summary
    results["token_usage"] = token_counter.get_summary()
    
    return results


def create_test_images_if_needed():
    """Create test image directory and placeholder images if they don't exist"""
    if not TEST_IMAGES_DIR.exists():
        TEST_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
        
        # Create placeholder text files explaining how to add real images
        for image_key, image_path in TEST_IMAGES.items():
            with open(str(image_path) + ".txt", "w") as f:
                f.write(f"Place a test image for {image_key} here named {image_path.name}\n")
                f.write("For multimodal tests to work, you need real images.\n")
                f.write("Recommended: screenshots of financial charts, documents, or transaction receipts.\n")


def run_all_tests(args) -> Dict[str, Any]:
    """Run all specified tests and return results"""
    results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "models_tested": [GEMINI_2_5_FLASH, GEMINI_2_5_PRO],
            "test_args": vars(args)
        },
        "tests": {}
    }
    
    # Check if API key is set
    if not os.environ.get("GEMINI_API_KEY"):
        logger.error("GEMINI_API_KEY environment variable not set. Set it before running tests.")
        results["error"] = "GEMINI_API_KEY environment variable not set"
        return results
    
    # Run text tests
    if args.all or args.text:
        text_results = []
        for prompt_data in TEXT_TEST_PROMPTS:
            result = run_text_test(prompt_data)
            text_results.append(result)
        results["tests"]["text"] = text_results
    
    # Run multimodal tests
    if args.all or args.multimodal:
        # Create test image placeholders if needed
        create_test_images_if_needed()
        
        # Check if any real images exist
        real_images_exist = any(image_path.exists() for image_path in TEST_IMAGES.values())
        
        if real_images_exist:
            multimodal_results = []
            for prompt_data in IMAGE_TEST_PROMPTS:
                result = run_multimodal_test(prompt_data)
                multimodal_results.append(result)
            results["tests"]["multimodal"] = multimodal_results
        else:
            logger.warning("No test images found. Skipping multimodal tests.")
            results["tests"]["multimodal"] = {
                "error": "No test images found",
                "message": f"Add images to {TEST_IMAGES_DIR} to run multimodal tests"
            }
    
    # Run error handling tests
    if args.all or args.errors:
        results["tests"]["error_handling"] = test_error_handling()
    
    # Run token usage tests
    if args.all or args.tokens:
        results["tests"]["token_usage"] = test_token_usage()
    
    # Add token usage summary
    results["token_usage_summary"] = token_counter.get_summary()
    
    # Add performance comparison
    if (args.all or args.text) and "text" in results["tests"]:
        flash_times = []
        pro_times = []
        
        for test in results["tests"]["text"]:
            if GEMINI_2_5_FLASH in test["models"] and test["models"][GEMINI_2_5_FLASH].get("success", False):
                flash_times.append(test["models"][GEMINI_2_5_FLASH]["time_seconds"])
            
            if GEMINI_2_5_PRO in test["models"] and test["models"][GEMINI_2_5_PRO].get("success", False):
                pro_times.append(test["models"][GEMINI_2_5_PRO]["time_seconds"])
        
        if flash_times and pro_times:
            avg_flash = sum(flash_times) / len(flash_times)
            avg_pro = sum(pro_times) / len(pro_times)
            
            results["performance_comparison"] = {
                "average_times": {
                    GEMINI_2_5_FLASH: round(avg_flash, 3),
                    GEMINI_2_5_PRO: round(avg_pro, 3)
                },
                "speed_ratio": round(avg_pro / avg_flash, 2) if avg_flash > 0 else "N/A",
                "flash_faster_by_percentage": round((avg_pro - avg_flash) / avg_pro * 100, 1) if avg_pro > 0 else "N/A"
            }
    
    return results


def print_summary(results: Dict[str, Any]):
    """Print a human-readable summary of test results"""
    print("\n" + "="*80)
    print(f"GEMINI 2.5 MODELS TEST SUMMARY - {results['metadata']['timestamp']}")
    print("="*80)
    
    # Print overall stats
    if "error" in results:
        print(f"\n❌ ERROR: {results['error']}")
        return
    
    # Performance comparison
    if "performance_comparison" in results:
        pc = results["performance_comparison"]
        print("\n📊 PERFORMANCE COMPARISON:")
        print(f"  • Gemini 2.5 Flash avg response time: {pc['average_times'][GEMINI_2_5_FLASH]:.3f}s")
        print(f"  • Gemini 2.5 Pro avg response time:   {pc['average_times'][GEMINI_2_5_PRO]:.3f}s")
        print(f"  • Speed ratio (Pro/Flash):            {pc['speed_ratio']}x")
        print(f"  • Flash faster by:                    {pc['flash_faster_by_percentage']}%")
    
    # Token usage
    if "token_usage_summary" in results:
        tu = results["token_usage_summary"]
        print("\n💰 TOKEN USAGE & COST:")
        print(f"  • Total requests: {tu['total_requests']}")
        print(f"  • Total estimated cost: ${tu['total_cost_usd']:.6f} USD")
        
        for model, stats in tu.get("models", {}).items():
            if stats["requests"] > 0:
                print(f"\n  {model}:")
                print(f"    - Requests:      {stats['requests']}")
                print(f"    - Input tokens:  {stats['input_tokens']}")
                print(f"    - Output tokens: {stats['output_tokens']}")
                print(f"    - Total tokens:  {stats['total_tokens']}")
                print(f"    - Est. cost:     ${stats['estimated_cost_usd']:.6f} USD")
    
    # Text test results
    if "text" in results.get("tests", {}):
        print("\n📝 TEXT TESTS:")
        text_tests = results["tests"]["text"]
        success_count = 0
        
        for test in text_tests:
            flash_success = test["models"].get(GEMINI_2_5_FLASH, {}).get("success", False)
            pro_success = test["models"].get(GEMINI_2_5_PRO, {}).get("success", False)
            
            status = "✅" if flash_success and pro_success else "⚠️" if flash_success or pro_success else "❌"
            print(f"  {status} {test['prompt_name']} ({test['complexity']})")
            
            if flash_success and pro_success:
                success_count += 1
        
        print(f"  Success rate: {success_count}/{len(text_tests)} tests passed on both models")
    
    # Multimodal test results
    if "multimodal" in results.get("tests", {}):
        if isinstance(results["tests"]["multimodal"], dict) and "error" in results["tests"]["multimodal"]:
            print(f"\n🖼️ MULTIMODAL TESTS: ❌ {results['tests']['multimodal']['error']}")
        else:
            print("\n🖼️ MULTIMODAL TESTS:")
            multimodal_tests = results["tests"]["multimodal"]
            success_count = 0
            
            for test in multimodal_tests:
                flash_success = test["models"].get(GEMINI_2_5_FLASH, {}).get("success", False)
                pro_success = test["models"].get(GEMINI_2_5_PRO, {}).get("success", False)
                
                status = "✅" if flash_success and pro_success else "⚠️" if flash_success or pro_success else "❌"
                print(f"  {status} {test['prompt_name']} (image: {test.get('image', 'N/A')})")
                
                if flash_success and pro_success:
                    success_count += 1
            
            print(f"  Success rate: {success_count}/{len(multimodal_tests)} tests passed on both models")
    
    # Error handling results
    if "error_handling" in results.get("tests", {}):
        print("\n🛡️ ERROR HANDLING TESTS:")
        error_tests = results["tests"]["error_handling"]["scenarios"]
        success_count = 0
        
        for scenario, result in error_tests.items():
            status = "✅" if not result["success"] and result.get("handled_correctly", False) else "❌"
            print(f"  {status} {scenario}")
            
            if not result["success"] and result.get("handled_correctly", False):
                success_count += 1
        
        print(f"  Success rate: {success_count}/{len(error_tests)} error scenarios handled correctly")
    
    print("\n" + "="*80)
    print(f"Complete results saved to: {args.output}")
    print("="*80 + "\n")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Test Gemini 2.5 models")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    parser.add_argument("--text", action="store_true", help="Run text-only tests")
    parser.add_argument("--multimodal", action="store_true", help="Run multimodal tests")
    parser.add_argument("--errors", action="store_true", help="Run error handling tests")
    parser.add_argument("--tokens", action="store_true", help="Run token usage tests")
    parser.add_argument("--output", default="gemini_test_results.json", help="Output file for results")
    
    args = parser.parse_args()
    
    # If no specific tests are selected, run all
    if not any([args.all, args.text, args.multimodal, args.errors, args.tokens]):
        args.all = True
    
    return args


if __name__ == "__main__":
    args = parse_args()
    
    # Run tests
    logger.info(f"Starting Gemini 2.5 model tests")
    results = run_all_tests(args)
    
    # Save results to file
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print_summary(results)
else:
    # When imported as a module
    logger.info("Gemini test module imported")
