"""Unit tests for integration clients.

This module contains tests for the GeminiClient, Neo4jClient, and E2BClient
which integrate with external services.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set required environment variables before importing backend modules
os.environ.setdefault("SECRET_KEY", "test_secret_key")
os.environ.setdefault("GOOGLE_API_KEY", "dummy_key")
os.environ.setdefault("E2B_API_KEY", "dummy_key") 
os.environ.setdefault("NEO4J_PASSWORD", "test_password")

import asyncio
import base64
import io
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
from PIL import Image

import google.generativeai as genai
from neo4j import AsyncGraphDatabase, AsyncSession, AsyncTransaction
from neo4j.exceptions import ServiceUnavailable, AuthError

from backend.integrations.gemini_client import GeminiClient
from backend.integrations.neo4j_client import Neo4jClient
from backend.integrations.e2b_client import E2BClient
from backend.config import settings


# ---- Fixtures ----

@pytest.fixture
def sample_image():
    """Create a simple test image."""
    # Create a small red square image
    img = Image.new('RGB', (100, 100), color='red')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    return img_byte_arr.getvalue()


@pytest.fixture
def sample_image_base64(sample_image):
    """Create a base64 encoded test image."""
    return base64.b64encode(sample_image).decode('utf-8')


@pytest.fixture
def mock_gemini_response():
    """Mock response from Gemini API."""
    mock_response = MagicMock()
    mock_response.text = "This is a test response from Gemini."
    return mock_response


@pytest.fixture
def mock_neo4j_driver():
    """Mock Neo4j driver."""
    mock_driver = AsyncMock(spec=AsyncGraphDatabase.driver)
    
    # Mock session
    mock_session = AsyncMock(spec=AsyncSession)
    
    # Mock transaction
    mock_tx = AsyncMock(spec=AsyncTransaction)
    mock_tx.run = AsyncMock(return_value=MagicMock())
    
    # Setup session to return transaction
    mock_session.__aenter__.return_value = mock_session
    mock_session.begin_transaction.return_value = AsyncMock()
    mock_session.begin_transaction().__aenter__.return_value = mock_tx
    
    # Setup driver to return session
    mock_driver.session.return_value = mock_session
    
    return mock_driver


@pytest.fixture
def mock_e2b_sdk():
    """Mock e2b.dev SDK."""
    mock_sdk = MagicMock()
    mock_session = MagicMock()
    
    # Mock session methods
    mock_session.process = MagicMock()
    mock_session.filesystem = MagicMock()
    mock_session.filesystem.write = AsyncMock()
    mock_session.filesystem.read = AsyncMock(return_value=b"file content")
    mock_session.process.start = AsyncMock(return_value=MagicMock(id="process123"))
    mock_session.process.wait = AsyncMock(return_value=MagicMock(exit_code=0, stdout="success", stderr=""))
    
    # Mock SDK methods
    mock_sdk.Session = MagicMock(return_value=mock_session)
    
    return mock_sdk


# ---- GeminiClient Tests ----

@pytest.mark.asyncio
async def test_gemini_client_init():
    """Test GeminiClient initialization."""
    with patch('google.generativeai.configure') as mock_configure:
        with patch('google.generativeai.GenerativeModel') as mock_model_class:
            # Create client
            client = GeminiClient()
            
            # Verify API was configured
            mock_configure.assert_called_once_with(api_key=settings.google_api_key)
            
            # Verify model was created
            mock_model_class.assert_called_once()
            assert client.model == mock_model_class.return_value


@pytest.mark.asyncio
async def test_generate_text_success(mock_gemini_response):
    """Test successful text generation."""
    with patch.object(GeminiClient, 'model', new_callable=PropertyMock) as mock_model:
        # Setup mock
        mock_model.return_value.generate_content_async = AsyncMock(return_value=mock_gemini_response)
        
        # Create client and call method
        client = GeminiClient()
        response = await client.generate_text("Test prompt")
        
        # Verify response
        assert response == mock_gemini_response.text
        mock_model.return_value.generate_content_async.assert_called_once_with("Test prompt")


@pytest.mark.asyncio
async def test_generate_text_with_context(mock_gemini_response):
    """Test text generation with context."""
    with patch.object(GeminiClient, 'model', new_callable=PropertyMock) as mock_model:
        # Setup mock
        mock_model.return_value.generate_content_async = AsyncMock(return_value=mock_gemini_response)
        
        # Create client and call method
        client = GeminiClient()
        response = await client.generate_text("Test prompt", context="Some context")
        
        # Verify response
        assert response == mock_gemini_response.text
        mock_model.return_value.generate_content_async.assert_called_once()
        # Verify context was included in prompt
        call_args = mock_model.return_value.generate_content_async.call_args[0][0]
        assert "Context: Some context" in call_args
        assert "Query: Test prompt" in call_args


@pytest.mark.asyncio
async def test_generate_text_error():
    """Test error handling in text generation."""
    with patch.object(GeminiClient, 'model', new_callable=PropertyMock) as mock_model:
        # Setup mock to raise exception
        mock_model.return_value.generate_content_async = AsyncMock(side_effect=Exception("API error"))
        
        # Create client
        client = GeminiClient()
        
        # Verify exception is propagated
        with pytest.raises(Exception) as exc_info:
            await client.generate_text("Test prompt")
        
        assert "API error" in str(exc_info.value)


@pytest.mark.asyncio
async def test_analyze_image_bytes(mock_gemini_response, sample_image):
    """Test image analysis with byte data."""
    with patch.object(GeminiClient, 'model', new_callable=PropertyMock) as mock_model:
        # Setup mock
        mock_model.return_value.generate_content_async = AsyncMock(return_value=mock_gemini_response)
        
        # Create client and call method
        client = GeminiClient()
        response = await client.analyze_image(sample_image)
        
        # Verify response
        assert response == mock_gemini_response.text
        mock_model.return_value.generate_content_async.assert_called_once()
        
        # Verify Image and prompt were passed
        args = mock_model.return_value.generate_content_async.call_args[0][0]
        assert len(args) == 2
        assert isinstance(args[0], str)  # Prompt
        assert isinstance(args[1], Image.Image)  # PIL Image


@pytest.mark.asyncio
async def test_analyze_image_base64(mock_gemini_response, sample_image_base64):
    """Test image analysis with base64 data."""
    with patch.object(GeminiClient, 'model', new_callable=PropertyMock) as mock_model:
        # Setup mock
        mock_model.return_value.generate_content_async = AsyncMock(return_value=mock_gemini_response)
        
        # Create client and call method
        client = GeminiClient()
        response = await client.analyze_image(sample_image_base64)
        
        # Verify response
        assert response == mock_gemini_response.text
        mock_model.return_value.generate_content_async.assert_called_once()


@pytest.mark.asyncio
async def test_analyze_image_error(sample_image):
    """Test error handling in image analysis."""
    with patch.object(GeminiClient, 'model', new_callable=PropertyMock) as mock_model:
        # Setup mock to raise exception
        mock_model.return_value.generate_content_async = AsyncMock(side_effect=Exception("API error"))
        
        # Create client
        client = GeminiClient()
        
        # Verify exception is propagated
        with pytest.raises(Exception) as exc_info:
            await client.analyze_image(sample_image)
        
        assert "API error" in str(exc_info.value)


@pytest.mark.asyncio
async def test_generate_cypher_query(mock_gemini_response):
    """Test Cypher query generation."""
    with patch.object(GeminiClient, 'model', new_callable=PropertyMock) as mock_model:
        # Setup mock
        mock_gemini_response.text = "MATCH (n:Person) RETURN n LIMIT 10"
        mock_model.return_value.generate_content_async = AsyncMock(return_value=mock_gemini_response)
        
        # Create client and call method
        client = GeminiClient()
        response = await client.generate_cypher_query(
            "Find all people", 
            "Schema: (:Person {name, age})-[:KNOWS]->(:Person)"
        )
        
        # Verify response
        assert response == mock_gemini_response.text
        mock_model.return_value.generate_content_async.assert_called_once()
        
        # Verify schema was included in prompt
        call_args = mock_model.return_value.generate_content_async.call_args[0][0]
        assert "Graph Schema Context:" in call_args
        assert "Natural Language Query: Find all people" in call_args


@pytest.mark.asyncio
async def test_generate_python_code(mock_gemini_response):
    """Test Python code generation."""
    with patch.object(GeminiClient, 'model', new_callable=PropertyMock) as mock_model:
        # Setup mock with code response
        mock_gemini_response.text = "```python\ndef hello():\n    print('Hello')\n```"
        mock_model.return_value.generate_content_async = AsyncMock(return_value=mock_gemini_response)
        
        # Create client and call method
        client = GeminiClient()
        response = await client.generate_python_code(
            "Create a hello function", 
            libraries=["pandas", "numpy"]
        )
        
        # Verify response (should strip markdown)
        assert response == "def hello():\n    print('Hello')"
        mock_model.return_value.generate_content_async.assert_called_once()
        
        # Verify libraries were included in prompt
        call_args = mock_model.return_value.generate_content_async.call_args[0][0]
        assert "Available libraries: pandas, numpy" in call_args


# ---- Neo4jClient Tests ----

@pytest.mark.asyncio
async def test_neo4j_client_connect(mock_neo4j_driver):
    """Test Neo4j client connection."""
    with patch('neo4j.AsyncGraphDatabase.driver', return_value=mock_neo4j_driver):
        # Create client
        client = Neo4jClient()
        await client.connect()
        
        # Verify driver was created
        assert client.driver == mock_neo4j_driver


@pytest.mark.asyncio
async def test_neo4j_client_connect_error():
    """Test Neo4j connection error handling."""
    with patch('neo4j.AsyncGraphDatabase.driver', side_effect=ServiceUnavailable("Connection refused")):
        # Create client
        client = Neo4jClient()
        
        # Verify exception is propagated
        with pytest.raises(ServiceUnavailable) as exc_info:
            await client.connect()
        
        assert "Connection refused" in str(exc_info.value)


@pytest.mark.asyncio
async def test_neo4j_client_auth_error():
    """Test Neo4j authentication error handling."""
    with patch('neo4j.AsyncGraphDatabase.driver', side_effect=AuthError("Invalid credentials")):
        # Create client
        client = Neo4jClient()
        
        # Verify exception is propagated
        with pytest.raises(AuthError) as exc_info:
            await client.connect()
        
        assert "Invalid credentials" in str(exc_info.value)


@pytest.mark.asyncio
async def test_neo4j_run_query(mock_neo4j_driver):
    """Test running a Cypher query."""
    with patch('neo4j.AsyncGraphDatabase.driver', return_value=mock_neo4j_driver):
        # Mock query result
        mock_result = MagicMock()
        mock_result.data.return_value = [{"n": {"name": "Test"}}]
        
        # Setup transaction mock
        mock_tx = mock_neo4j_driver.session.return_value.__aenter__.return_value.begin_transaction.return_value.__aenter__.return_value
        mock_tx.run.return_value = mock_result
        
        # Create client and connect
        client = Neo4jClient()
        await client.connect()
        
        # Run query
        result = await client.run_query("MATCH (n:Test) RETURN n")
        
        # Verify query was executed
        mock_tx.run.assert_called_once_with("MATCH (n:Test) RETURN n", {})
        assert result == [{"n": {"name": "Test"}}]


@pytest.mark.asyncio
async def test_neo4j_run_query_with_params(mock_neo4j_driver):
    """Test running a Cypher query with parameters."""
    with patch('neo4j.AsyncGraphDatabase.driver', return_value=mock_neo4j_driver):
        # Mock query result
        mock_result = MagicMock()
        mock_result.data.return_value = [{"n": {"name": "Test"}}]
        
        # Setup transaction mock
        mock_tx = mock_neo4j_driver.session.return_value.__aenter__.return_value.begin_transaction.return_value.__aenter__.return_value
        mock_tx.run.return_value = mock_result
        
        # Create client and connect
        client = Neo4jClient()
        await client.connect()
        
        # Run query with parameters
        params = {"name": "Test"}
        result = await client.run_query("MATCH (n:Test {name: $name}) RETURN n", params)
        
        # Verify query was executed with parameters
        mock_tx.run.assert_called_once_with("MATCH (n:Test {name: $name}) RETURN n", params)
        assert result == [{"n": {"name": "Test"}}]


@pytest.mark.asyncio
async def test_neo4j_run_query_error(mock_neo4j_driver):
    """Test error handling when running a Cypher query."""
    with patch('neo4j.AsyncGraphDatabase.driver', return_value=mock_neo4j_driver):
        # Setup transaction mock to raise exception
        mock_tx = mock_neo4j_driver.session.return_value.__aenter__.return_value.begin_transaction.return_value.__aenter__.return_value
        mock_tx.run.side_effect = Exception("Query error")
        
        # Create client and connect
        client = Neo4jClient()
        await client.connect()
        
        # Verify exception is propagated
        with pytest.raises(Exception) as exc_info:
            await client.run_query("MATCH (n:Invalid) RETURN n")
        
        assert "Query error" in str(exc_info.value)


@pytest.mark.asyncio
async def test_neo4j_close(mock_neo4j_driver):
    """Test closing Neo4j connection."""
    with patch('neo4j.AsyncGraphDatabase.driver', return_value=mock_neo4j_driver):
        # Create client and connect
        client = Neo4jClient()
        await client.connect()
        
        # Close connection
        await client.close()
        
        # Verify driver was closed
        mock_neo4j_driver.close.assert_called_once()


# ---- E2BClient Tests ----

@pytest.mark.asyncio
async def test_e2b_client_init(mock_e2b_sdk):
    """Test E2BClient initialization."""
    with patch('backend.integrations.e2b_client.e2b', mock_e2b_sdk):
        # Create client
        client = E2BClient()
        
        # Verify client was initialized
        assert client.api_key == settings.e2b_api_key
        assert client.template_id == settings.e2b_template_id


@pytest.mark.asyncio
async def test_e2b_create_sandbox(mock_e2b_sdk):
    """Test creating an e2b sandbox."""
    with patch('backend.integrations.e2b_client.e2b', mock_e2b_sdk):
        # Create client
        client = E2BClient()
        
        # Create sandbox
        sandbox = await client.create_sandbox()
        
        # Verify sandbox was created
        mock_e2b_sdk.Session.assert_called_once_with(
            id=None,
            api_key=settings.e2b_api_key,
            template=settings.e2b_template_id
        )
        assert sandbox == mock_e2b_sdk.Session.return_value


@pytest.mark.asyncio
async def test_e2b_execute_code(mock_e2b_sdk):
    """Test executing code in e2b sandbox."""
    with patch('backend.integrations.e2b_client.e2b', mock_e2b_sdk):
        # Setup process mock
        mock_process = MagicMock()
        mock_process.exit_code = 0
        mock_process.stdout = "Hello, World!"
        mock_process.stderr = ""
        
        # Setup session mock
        mock_session = mock_e2b_sdk.Session.return_value
        mock_session.process.start.return_value = MagicMock(id="process123")
        mock_session.process.wait.return_value = mock_process
        
        # Create client
        client = E2BClient()
        
        # Execute code
        code = "print('Hello, World!')"
        result = await client.execute_code(code, sandbox=mock_session)
        
        # Verify code was executed
        mock_session.filesystem.write.assert_called_once()
        mock_session.process.start.assert_called_once()
        mock_session.process.wait.assert_called_once()
        
        # Verify result
        assert result["success"] is True
        assert result["stdout"] == "Hello, World!"
        assert result["stderr"] == ""
        assert result["exit_code"] == 0


@pytest.mark.asyncio
async def test_e2b_execute_code_error(mock_e2b_sdk):
    """Test error handling when executing code."""
    with patch('backend.integrations.e2b_client.e2b', mock_e2b_sdk):
        # Setup process mock with error
        mock_process = MagicMock()
        mock_process.exit_code = 1
        mock_process.stdout = ""
        mock_process.stderr = "SyntaxError: invalid syntax"
        
        # Setup session mock
        mock_session = mock_e2b_sdk.Session.return_value
        mock_session.process.start.return_value = MagicMock(id="process123")
        mock_session.process.wait.return_value = mock_process
        
        # Create client
        client = E2BClient()
        
        # Execute code with syntax error
        code = "print('Hello, World!'"  # Missing closing quote
        result = await client.execute_code(code, sandbox=mock_session)
        
        # Verify result contains error
        assert result["success"] is False
        assert result["stderr"] == "SyntaxError: invalid syntax"
        assert result["exit_code"] == 1


@pytest.mark.asyncio
async def test_e2b_install_package(mock_e2b_sdk):
    """Test installing a package in e2b sandbox."""
    with patch('backend.integrations.e2b_client.e2b', mock_e2b_sdk):
        # Setup process mock
        mock_process = MagicMock()
        mock_process.exit_code = 0
        mock_process.stdout = "Successfully installed pandas-2.0.0"
        mock_process.stderr = ""
        
        # Setup session mock
        mock_session = mock_e2b_sdk.Session.return_value
        mock_session.process.start.return_value = MagicMock(id="process123")
        mock_session.process.wait.return_value = mock_process
        
        # Create client
        client = E2BClient()
        
        # Install package
        result = await client.install_package("pandas", sandbox=mock_session)
        
        # Verify package was installed
        mock_session.process.start.assert_called_once()
        assert "pip install pandas" in mock_session.process.start.call_args[0][0]
        
        # Verify result
        assert result["success"] is True
        assert "Successfully installed pandas" in result["stdout"]


@pytest.mark.asyncio
async def test_e2b_upload_file(mock_e2b_sdk):
    """Test uploading a file to e2b sandbox."""
    with patch('backend.integrations.e2b_client.e2b', mock_e2b_sdk):
        # Setup session mock
        mock_session = mock_e2b_sdk.Session.return_value
        
        # Create client
        client = E2BClient()
        
        # Upload file
        file_content = b"test file content"
        await client.upload_file(file_content, "test.txt", sandbox=mock_session)
        
        # Verify file was uploaded
        mock_session.filesystem.write.assert_called_once_with("/home/user/test.txt", file_content)


@pytest.mark.asyncio
async def test_e2b_download_file(mock_e2b_sdk):
    """Test downloading a file from e2b sandbox."""
    with patch('backend.integrations.e2b_client.e2b', mock_e2b_sdk):
        # Setup session mock
        mock_session = mock_e2b_sdk.Session.return_value
        mock_session.filesystem.read.return_value = b"file content"
        
        # Create client
        client = E2BClient()
        
        # Download file
        content = await client.download_file("test.txt", sandbox=mock_session)
        
        # Verify file was downloaded
        mock_session.filesystem.read.assert_called_once_with("/home/user/test.txt")
        assert content == b"file content"


@pytest.mark.asyncio
async def test_e2b_close_sandbox(mock_e2b_sdk):
    """Test closing an e2b sandbox."""
    with patch('backend.integrations.e2b_client.e2b', mock_e2b_sdk):
        # Setup session mock
        mock_session = mock_e2b_sdk.Session.return_value
        mock_session.close = AsyncMock()
        
        # Create client
        client = E2BClient()
        
        # Close sandbox
        await client.close_sandbox(mock_session)
        
        # Verify sandbox was closed
        mock_session.close.assert_called_once()
