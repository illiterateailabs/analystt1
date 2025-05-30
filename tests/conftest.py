"""
Pytest configuration for the Analyst's Augmentation Agent.

This module configures pytest for testing the backend, including:
- Setting up import paths
- Configuring environment variables
- Setting up common fixtures
- Configuring asyncio for testing
"""

import os
import sys
from pathlib import Path

import pytest
from dotenv import load_dotenv

# Add the parent directory to sys.path so imports work correctly
# This ensures that 'import backend.xyz' works in test files
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables from .env.test if it exists, otherwise use .env
env_file = Path(__file__).parent / '.env.test'
if not env_file.exists():
    env_file = Path(__file__).parent.parent / '.env'

load_dotenv(dotenv_path=env_file)

# Set up test environment variables if not already set
if not os.environ.get("TESTING"):
    # Set testing flag
    os.environ["TESTING"] = "true"
    
    # Set test database connections
    os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
    os.environ.setdefault("NEO4J_USERNAME", "neo4j")
    os.environ.setdefault("NEO4J_PASSWORD", "analyst123")
    os.environ.setdefault("NEO4J_DATABASE", "neo4j")
    
    # Set test API keys (dummy values for testing)
    os.environ.setdefault("GOOGLE_API_KEY", "dummy_key_for_tests")
    os.environ.setdefault("E2B_API_KEY", "dummy_key_for_tests")
    os.environ.setdefault("E2B_TEMPLATE_ID", "python3-default")
    
    # Set test JWT configuration
    os.environ.setdefault("SECRET_KEY", "test_secret_key_for_jwt_not_for_production")
    os.environ.setdefault("JWT_ALGORITHM", "HS256")
    os.environ.setdefault("JWT_EXPIRATION_MINUTES", "60")
    os.environ.setdefault("JWT_REFRESH_EXPIRATION_MINUTES", "10080")
    os.environ.setdefault("JWT_AUDIENCE", "analyst-agent-api-test")
    os.environ.setdefault("JWT_ISSUER", "analyst-agent-test")
    
    # Set test application configuration
    os.environ.setdefault("APP_NAME", "Analyst Augmentation Agent Test")
    os.environ.setdefault("APP_VERSION", "test-version")
    os.environ.setdefault("DEBUG", "true")
    os.environ.setdefault("LOG_LEVEL", "DEBUG")
    os.environ.setdefault("CORS_ORIGINS", "http://localhost:3000,http://localhost:8000")


# Configure pytest
def pytest_configure(config):
    """Configure pytest."""
    # Register markers
    config.addinivalue_line("markers", "unit: mark a test as a unit test")
    config.addinivalue_line("markers", "integration: mark a test as an integration test")
    config.addinivalue_line("markers", "slow: mark a test as slow")
    
    # Set asyncio mode
    config.option.asyncio_mode = "auto"


# Cleanup after tests
@pytest.fixture(scope="session", autouse=True)
def cleanup_after_tests():
    """Perform cleanup after all tests have run."""
    yield
    # Any cleanup code goes here
