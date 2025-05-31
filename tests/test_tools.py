"""
Tests for agent tools.

This module contains tests for the various tools used by agents, including
TemplateEngineTool, PolicyDocsTool, CodeGenTool, and others. It verifies
that they function correctly, handle errors properly, and return valid
JSON responses.
"""

import json
import pytest
import os
from unittest.mock import AsyncMock, MagicMock, patch, mock_open

from backend.agents.tools.template_engine_tool import TemplateEngineTool
from backend.agents.tools.policy_docs_tool import PolicyDocsTool
from backend.agents.tools.code_gen_tool import CodeGenTool
from backend.agents.tools.graph_query_tool import GraphQueryTool
from backend.agents.tools.sandbox_exec_tool import SandboxExecTool
from backend.agents.tools.pattern_library_tool import PatternLibraryTool
from backend.agents.tools.neo4j_schema_tool import Neo4jSchemaTool
from backend.integrations.gemini_client import GeminiClient
from backend.integrations.neo4j_client import Neo4jClient
from backend.integrations.e2b_client import E2BClient


# ---- Fixtures ----

@pytest.fixture
def mock_gemini_client():
    """Fixture for mocked GeminiClient."""
    with patch("backend.agents.tools.policy_docs_tool.GeminiClient", autospec=True) as mock_client:
        mock_instance = mock_client.return_value
        mock_instance.generate_text = AsyncMock(return_value="Generated policy response")
        mock_instance.generate_cypher_query = AsyncMock(return_value="MATCH (n) RETURN n LIMIT 10")
        mock_instance.generate_python_code = AsyncMock(return_value="def test(): return 'Hello'")
        yield mock_instance


@pytest.fixture
def mock_neo4j_client():
    """Fixture for mocked Neo4jClient."""
    with patch("backend.agents.tools.graph_query_tool.Neo4jClient", autospec=True) as mock_client:
        mock_instance = mock_client.return_value
        mock_instance.run_query = AsyncMock(return_value=[{"result": "test"}])
        yield mock_instance


@pytest.fixture
def mock_e2b_client():
    """Fixture for mocked E2BClient."""
    with patch("backend.agents.tools.sandbox_exec_tool.E2BClient", autospec=True) as mock_client:
        mock_instance = mock_client.return_value
        mock_instance.create_sandbox = AsyncMock()
        mock_instance.execute_code = AsyncMock(return_value={
            "success": True,
            "stdout": "Test output",
            "stderr": "",
            "exit_code": 0
        })
        mock_instance.close_sandbox = AsyncMock()
        yield mock_instance


@pytest.fixture
def template_engine_tool():
    """Fixture for TemplateEngineTool."""
    # Mock jinja2 environment
    with patch("backend.agents.tools.template_engine_tool.jinja2", autospec=True) as mock_jinja:
        # Mock template loading
        mock_env = MagicMock()
        mock_template = MagicMock()
        mock_template.render.return_value = "Rendered template content"
        mock_env.get_template.return_value = mock_template
        mock_env.from_string.return_value = mock_template
        mock_jinja.Environment.return_value = mock_env
        mock_jinja.FileSystemLoader.return_value = MagicMock()
        mock_jinja.select_autoescape.return_value = ["html", "xml"]
        
        # Set JINJA2_AVAILABLE to True
        with patch("backend.agents.tools.template_engine_tool.JINJA2_AVAILABLE", True):
            # Create tool
            tool = TemplateEngineTool()
            yield tool


@pytest.fixture
def policy_docs_tool(mock_gemini_client):
    """Fixture for PolicyDocsTool."""
    tool = PolicyDocsTool(gemini_client=mock_gemini_client)
    yield tool


@pytest.fixture
def code_gen_tool(mock_gemini_client):
    """Fixture for CodeGenTool."""
    tool = CodeGenTool(gemini_client=mock_gemini_client)
    yield tool


@pytest.fixture
def graph_query_tool(mock_neo4j_client):
    """Fixture for GraphQueryTool."""
    tool = GraphQueryTool(neo4j_client=mock_neo4j_client)
    yield tool


@pytest.fixture
def sandbox_exec_tool(mock_e2b_client):
    """Fixture for SandboxExecTool."""
    tool = SandboxExecTool(e2b_client=mock_e2b_client)
    yield tool


# ---- Tests for TemplateEngineTool ----

@pytest.mark.asyncio
async def test_template_engine_named_template(template_engine_tool):
    """Test TemplateEngineTool with a named template."""
    # Test data
    data = {
        "title": "Test Report",
        "summary": "This is a test summary",
        "findings": [
            {"title": "Finding 1", "description": "Description 1", "risk_level": "high"},
            {"title": "Finding 2", "description": "Description 2", "risk_level": "medium"}
        ],
        "analysis": "Detailed analysis goes here",
        "recommendations": ["Recommendation 1", "Recommendation 2"]
    }
    
    # Run tool
    result = await template_engine_tool._arun(
        template_name="markdown_report",
        template_format="markdown",
        data=data
    )
    
    # Parse result
    result_json = json.loads(result)
    
    # Verify result
    assert result_json["success"] is True
    assert "content" in result_json
    assert result_json["template_format"] == "markdown"
    assert "template_source" in result_json
    assert "markdown_report" in result_json["template_source"]


@pytest.mark.asyncio
async def test_template_engine_custom_template(template_engine_tool):
    """Test TemplateEngineTool with a custom template."""
    # Test data
    data = {
        "title": "Test Report",
        "items": ["Item 1", "Item 2", "Item 3"]
    }
    
    # Custom template
    template_content = """# {{ title }}

## Items
{% for item in items %}
- {{ item }}
{% endfor %}
"""
    
    # Run tool
    result = await template_engine_tool._arun(
        template_content=template_content,
        template_format="markdown",
        data=data
    )
    
    # Parse result
    result_json = json.loads(result)
    
    # Verify result
    assert result_json["success"] is True
    assert "content" in result_json
    assert result_json["template_format"] == "markdown"
    assert result_json["template_source"] == "custom"


@pytest.mark.asyncio
async def test_template_engine_html_format(template_engine_tool):
    """Test TemplateEngineTool with HTML format."""
    # Test data
    data = {
        "title": "Test Report",
        "summary": "This is a test summary",
        "findings": [
            {"title": "Finding 1", "description": "Description 1", "risk_level": "high"},
            {"title": "Finding 2", "description": "Description 2", "risk_level": "medium"}
        ]
    }
    
    # Run tool
    result = await template_engine_tool._arun(
        template_name="html_report",
        template_format="html",
        data=data
    )
    
    # Parse result
    result_json = json.loads(result)
    
    # Verify result
    assert result_json["success"] is True
    assert "content" in result_json
    assert result_json["template_format"] == "html"


@pytest.mark.asyncio
async def test_template_engine_json_format(template_engine_tool):
    """Test TemplateEngineTool with JSON format."""
    # Test data
    data = {
        "title": "Test Report",
        "summary": "This is a test summary",
        "findings": [
            {"title": "Finding 1", "description": "Description 1", "risk_level": "high"},
            {"title": "Finding 2", "description": "Description 2", "risk_level": "medium"}
        ],
        "data": {"metric1": 100, "metric2": 200}
    }
    
    # Run tool
    result = await template_engine_tool._arun(
        template_name="json_report",
        template_format="json",
        data=data
    )
    
    # Parse result
    result_json = json.loads(result)
    
    # Verify result
    assert result_json["success"] is True
    assert "content" in result_json
    assert result_json["template_format"] == "json"
    
    # Verify content is valid JSON
    content = json.loads(result_json["content"])
    assert "report" in content
    assert content["report"]["title"] == "Test Report"


@pytest.mark.asyncio
async def test_template_engine_error_handling(template_engine_tool):
    """Test TemplateEngineTool error handling."""
    # Mock template rendering to raise an exception
    with patch.object(template_engine_tool.env.get_template("markdown_report.j2"), "render") as mock_render:
        mock_render.side_effect = Exception("Template rendering error")
        
        # Run tool
        result = await template_engine_tool._arun(
            template_name="markdown_report",
            template_format="markdown",
            data={}
        )
        
        # Parse result
        result_json = json.loads(result)
        
        # Verify result
        assert result_json["success"] is False
        assert "error" in result_json
        assert "Template rendering error" in result_json["error"]


@pytest.mark.asyncio
async def test_template_engine_no_jinja(template_engine_tool):
    """Test TemplateEngineTool when Jinja2 is not available."""
    # Set JINJA2_AVAILABLE to False
    with patch("backend.agents.tools.template_engine_tool.JINJA2_AVAILABLE", False):
        # Run tool
        result = await template_engine_tool._arun(
            template_name="markdown_report",
            template_format="markdown",
            data={}
        )
        
        # Parse result
        result_json = json.loads(result)
        
        # Verify result
        assert result_json["success"] is False
        assert "error" in result_json
        assert "Jinja2 is not available" in result_json["error"]


# ---- Tests for PolicyDocsTool ----

@pytest.mark.asyncio
async def test_policy_docs_direct_match(policy_docs_tool):
    """Test PolicyDocsTool with a direct match in documents."""
    # Run tool
    result = await policy_docs_tool._arun(
        query="Politically Exposed Person",
        document_type="kyc"
    )
    
    # Parse result
    result_json = json.loads(result)
    
    # Verify result
    assert result_json["success"] is True
    assert "results" in result_json
    assert len(result_json["results"]) > 0
    assert "PEP" in result_json["results"][0]["title"]
    assert result_json["results"][0]["type"] == "kyc"


@pytest.mark.asyncio
async def test_policy_docs_document_type_filter(policy_docs_tool):
    """Test PolicyDocsTool with document type filtering."""
    # Run tool
    result = await policy_docs_tool._arun(
        query="transaction",
        document_type="aml"
    )
    
    # Parse result
    result_json = json.loads(result)
    
    # Verify result
    assert result_json["success"] is True
    assert "results" in result_json
    assert len(result_json["results"]) > 0
    
    # Verify all results are of type "aml"
    for item in result_json["results"]:
        assert item["type"] == "aml"


@pytest.mark.asyncio
async def test_policy_docs_no_match_gemini_fallback(policy_docs_tool, mock_gemini_client):
    """Test PolicyDocsTool with no direct match, falling back to Gemini."""
    # Run tool with a query that won't match directly
    result = await policy_docs_tool._arun(
        query="obscure regulatory requirement",
        document_type=None
    )
    
    # Parse result
    result_json = json.loads(result)
    
    # Verify result
    assert result_json["success"] is True
    assert "generated_response" in result_json
    assert result_json["results"] == []
    assert "Generated policy response" in result_json["generated_response"]
    
    # Verify Gemini was called
    mock_gemini_client.generate_text.assert_called_once()


@pytest.mark.asyncio
async def test_policy_docs_max_results(policy_docs_tool):
    """Test PolicyDocsTool with max_results parameter."""
    # Run tool with max_results=1
    result = await policy_docs_tool._arun(
        query="transaction",
        max_results=1
    )
    
    # Parse result
    result_json = json.loads(result)
    
    # Verify result
    assert result_json["success"] is True
    assert "results" in result_json
    assert len(result_json["results"]) == 1


@pytest.mark.asyncio
async def test_policy_docs_error_handling(policy_docs_tool):
    """Test PolicyDocsTool error handling."""
    # Mock policy_docs to raise an exception
    with patch.object(policy_docs_tool, "policy_docs", side_effect=Exception("Policy docs error")):
        # Run tool
        result = await policy_docs_tool._arun(
            query="transaction"
        )
        
        # Parse result
        result_json = json.loads(result)
        
        # Verify result
        assert result_json["success"] is False
        assert "error" in result_json
        assert "Policy docs error" in result_json["error"]


# ---- Tests for CodeGenTool ----

@pytest.mark.asyncio
async def test_code_gen_basic(code_gen_tool, mock_gemini_client):
    """Test CodeGenTool basic functionality."""
    # Run tool
    result = await code_gen_tool._arun(
        task_description="Create a function to calculate the Fibonacci sequence",
        libraries=["numpy", "pandas"],
        code_style="standard"
    )
    
    # Parse result
    result_json = json.loads(result)
    
    # Verify result
    assert result_json["success"] is True
    assert "code" in result_json
    assert result_json["language"] == "python"
    assert result_json["libraries_used"] == ["numpy", "pandas"]
    
    # Verify Gemini was called
    mock_gemini_client.generate_python_code.assert_called_once()


@pytest.mark.asyncio
async def test_code_gen_security_levels(code_gen_tool):
    """Test CodeGenTool with different security levels."""
    # Test security levels
    security_levels = ["standard", "high", "paranoid"]
    
    for level in security_levels:
        # Run tool
        result = await code_gen_tool._arun(
            task_description="Create a function to read a file",
            security_level=level
        )
        
        # Parse result
        result_json = json.loads(result)
        
        # Verify result
        assert result_json["success"] is True
        assert "code" in result_json
        
        # For paranoid level, check for security wrapper
        if level == "paranoid":
            assert "limit_resources" in result_json["code"]
            assert "signal.alarm" in result_json["code"]


@pytest.mark.asyncio
async def test_code_gen_code_styles(code_gen_tool, mock_gemini_client):
    """Test CodeGenTool with different code styles."""
    # Test code styles
    code_styles = ["standard", "functional", "object-oriented"]
    
    for style in code_styles:
        # Run tool
        result = await code_gen_tool._arun(
            task_description="Create a program to sort a list",
            code_style=style
        )
        
        # Parse result
        result_json = json.loads(result)
        
        # Verify result
        assert result_json["success"] is True
        assert "code" in result_json
        
        # Verify Gemini was called with the correct style
        args, kwargs = mock_gemini_client.generate_python_code.call_args
        assert style in kwargs.get("task_description", "")


@pytest.mark.asyncio
async def test_code_gen_security_checks(code_gen_tool):
    """Test CodeGenTool security checks."""
    # Mock generate_python_code to return code with security issues
    with patch.object(code_gen_tool.gemini_client, "generate_python_code") as mock_generate:
        mock_generate.return_value = """
def unsafe_function():
    user_input = input("Enter command: ")
    result = eval(user_input)  # Unsafe eval
    return result
"""
        
        # Run tool with high security
        result = await code_gen_tool._arun(
            task_description="Create a function to evaluate user input",
            security_level="high"
        )
        
        # Parse result
        result_json = json.loads(result)
        
        # Verify result
        assert result_json["success"] is True
        assert "code" in result_json
        assert "WARNING" in result_json["code"]
        assert "eval(" in result_json["code"]
        
        # Run tool with paranoid security
        result = await code_gen_tool._arun(
            task_description="Create a function to evaluate user input",
            security_level="paranoid"
        )
        
        # Parse result
        result_json = json.loads(result)
        
        # Verify result
        assert result_json["success"] is True
        assert "code" in result_json
        assert "WARNING" in result_json["code"]
        assert "SECURITY RISK REMOVED" in result_json["code"]
        assert "limit_resources" in result_json["code"]


@pytest.mark.asyncio
async def test_code_gen_error_handling(code_gen_tool, mock_gemini_client):
    """Test CodeGenTool error handling."""
    # Mock generate_python_code to raise an exception
    mock_gemini_client.generate_python_code.side_effect = Exception("Code generation error")
    
    # Run tool
    result = await code_gen_tool._arun(
        task_description="Create a function that will fail"
    )
    
    # Parse result
    result_json = json.loads(result)
    
    # Verify result
    assert result_json["success"] is False
    assert "error" in result_json
    assert "Code generation error" in result_json["error"]
    assert "code" in result_json
    assert "# Error generating code" in result_json["code"]


# ---- Tests for GraphQueryTool ----

@pytest.mark.asyncio
async def test_graph_query_direct_cypher(graph_query_tool, mock_neo4j_client):
    """Test GraphQueryTool with direct Cypher query."""
    # Run tool
    result = await graph_query_tool._arun(
        query="MATCH (n) RETURN n LIMIT 10",
        use_gemini=False
    )
    
    # Parse result
    result_json = json.loads(result)
    
    # Verify result
    assert result_json["success"] is True
    assert "results" in result_json
    assert result_json["results"] == [{"result": "test"}]
    assert "cypher_query" in result_json
    assert result_json["cypher_query"] == "MATCH (n) RETURN n LIMIT 10"
    
    # Verify Neo4j client was called
    mock_neo4j_client.run_query.assert_called_once_with("MATCH (n) RETURN n LIMIT 10")


@pytest.mark.asyncio
async def test_graph_query_with_gemini(graph_query_tool, mock_neo4j_client):
    """Test GraphQueryTool with Gemini-generated Cypher."""
    # Mock GeminiClient
    with patch("backend.agents.tools.graph_query_tool.GeminiClient") as mock_gemini_class:
        mock_gemini_instance = mock_gemini_class.return_value
        mock_gemini_instance.generate_cypher_query = AsyncMock(return_value="MATCH (n) RETURN n LIMIT 5")
        
        # Run tool
        result = await graph_query_tool._arun(
            query="Find all nodes",
            use_gemini=True
        )
        
        # Parse result
        result_json = json.loads(result)
        
        # Verify result
        assert result_json["success"] is True
        assert "results" in result_json
        assert "cypher_query" in result_json
        assert result_json["cypher_query"] == "MATCH (n) RETURN n LIMIT 5"
        assert "natural_language_query" in result_json
        assert result_json["natural_language_query"] == "Find all nodes"
        
        # Verify Gemini was called
        mock_gemini_instance.generate_cypher_query.assert_called_once()
        
        # Verify Neo4j client was called with the generated query
        mock_neo4j_client.run_query.assert_called_once_with("MATCH (n) RETURN n LIMIT 5")


@pytest.mark.asyncio
async def test_graph_query_with_parameters(graph_query_tool, mock_neo4j_client):
    """Test GraphQueryTool with query parameters."""
    # Run tool
    result = await graph_query_tool._arun(
        query="MATCH (n) WHERE n.name = $name RETURN n",
        parameters={"name": "John"},
        use_gemini=False
    )
    
    # Parse result
    result_json = json.loads(result)
    
    # Verify result
    assert result_json["success"] is True
    assert "results" in result_json
    assert "parameters" in result_json
    assert result_json["parameters"] == {"name": "John"}
    
    # Verify Neo4j client was called with parameters
    mock_neo4j_client.run_query.assert_called_once_with(
        "MATCH (n) WHERE n.name = $name RETURN n",
        {"name": "John"}
    )


@pytest.mark.asyncio
async def test_graph_query_error_handling(graph_query_tool, mock_neo4j_client):
    """Test GraphQueryTool error handling."""
    # Mock run_query to raise an exception
    mock_neo4j_client.run_query.side_effect = Exception("Neo4j query error")
    
    # Run tool
    result = await graph_query_tool._arun(
        query="MATCH (n) RETURN n",
        use_gemini=False
    )
    
    # Parse result
    result_json = json.loads(result)
    
    # Verify result
    assert result_json["success"] is False
    assert "error" in result_json
    assert "Neo4j query error" in result_json["error"]


# ---- Tests for SandboxExecTool ----

@pytest.mark.asyncio
async def test_sandbox_exec_basic(sandbox_exec_tool, mock_e2b_client):
    """Test SandboxExecTool basic functionality."""
    # Run tool
    result = await sandbox_exec_tool._arun(
        code="print('Hello, World!')",
        timeout_seconds=10
    )
    
    # Parse result
    result_json = json.loads(result)
    
    # Verify result
    assert result_json["success"] is True
    assert "stdout" in result_json
    assert result_json["stdout"] == "Test output"
    assert "stderr" in result_json
    assert result_json["stderr"] == ""
    assert "exit_code" in result_json
    assert result_json["exit_code"] == 0
    
    # Verify E2B client was called
    mock_e2b_client.execute_code.assert_called_once()


@pytest.mark.asyncio
async def test_sandbox_exec_with_dependencies(sandbox_exec_tool, mock_e2b_client):
    """Test SandboxExecTool with dependencies."""
    # Run tool
    result = await sandbox_exec_tool._arun(
        code="import numpy as np\nprint(np.array([1, 2, 3]))",
        dependencies=["numpy"],
        timeout_seconds=10
    )
    
    # Parse result
    result_json = json.loads(result)
    
    # Verify result
    assert result_json["success"] is True
    assert "dependencies" in result_json
    assert result_json["dependencies"] == ["numpy"]
    
    # Verify E2B client was called with install command
    calls = mock_e2b_client.execute_code.call_args_list
    assert len(calls) >= 2
    install_call = calls[0]
    assert "pip install" in install_call[0][0]
    assert "numpy" in install_call[0][0]


@pytest.mark.asyncio
async def test_sandbox_exec_with_environment_variables(sandbox_exec_tool, mock_e2b_client):
    """Test SandboxExecTool with environment variables."""
    # Run tool
    result = await sandbox_exec_tool._arun(
        code="import os\nprint(os.environ.get('TEST_VAR'))",
        environment_variables={"TEST_VAR": "test_value"},
        timeout_seconds=10
    )
    
    # Parse result
    result_json = json.loads(result)
    
    # Verify result
    assert result_json["success"] is True
    assert "environment_variables" in result_json
    assert result_json["environment_variables"] == {"TEST_VAR": "test_value"}


@pytest.mark.asyncio
async def test_sandbox_exec_error_handling(sandbox_exec_tool, mock_e2b_client):
    """Test SandboxExecTool error handling."""
    # Mock execute_code to raise an exception
    mock_e2b_client.execute_code.side_effect = Exception("Sandbox execution error")
    
    # Run tool
    result = await sandbox_exec_tool._arun(
        code="print('This will fail')",
        timeout_seconds=10
    )
    
    # Parse result
    result_json = json.loads(result)
    
    # Verify result
    assert result_json["success"] is False
    assert "error" in result_json
    assert "Sandbox execution error" in result_json["error"]


@pytest.mark.asyncio
async def test_sandbox_exec_code_error(sandbox_exec_tool, mock_e2b_client):
    """Test SandboxExecTool with code that has errors."""
    # Mock execute_code to return error
    mock_e2b_client.execute_code.return_value = {
        "success": True,
        "stdout": "",
        "stderr": "SyntaxError: invalid syntax",
        "exit_code": 1
    }
    
    # Run tool
    result = await sandbox_exec_tool._arun(
        code="print('Hello, World!'",  # Missing closing parenthesis
        timeout_seconds=10
    )
    
    # Parse result
    result_json = json.loads(result)
    
    # Verify result
    assert result_json["success"] is True  # Tool execution succeeded, but code had errors
    assert "stderr" in result_json
    assert "SyntaxError" in result_json["stderr"]
    assert "exit_code" in result_json
    assert result_json["exit_code"] == 1


# ---- Tests for PatternLibraryTool ----

@pytest.mark.asyncio
async def test_pattern_library_load_patterns(mock_neo4j_client):
    """Test PatternLibraryTool loading patterns."""
    # Mock open to return pattern library YAML
    pattern_yaml = """
patterns:
  STRUCTURING:
    name: Structuring
    description: Multiple transactions just below reporting thresholds
    cypher_template: |
      MATCH (a:Account)-[:SENT]->(t:Transaction)
      WHERE t.amount >= $min_amount AND t.amount <= $threshold
      WITH a, count(t) as tx_count
      WHERE tx_count >= $min_transactions
      RETURN a.id as account_id, tx_count, "STRUCTURING" as pattern_type
    parameters:
      threshold: 10000
      min_amount: 9000
      min_transactions: 3
    risk_score: 0.8
    """
    
    with patch("builtins.open", mock_open(read_data=pattern_yaml)):
        with patch("os.path.exists", return_value=True):
            # Create tool
            from backend.agents.tools.pattern_library_tool import PatternLibraryTool
            tool = PatternLibraryTool(neo4j_client=mock_neo4j_client)
            
            # Verify patterns were loaded
            assert "STRUCTURING" in tool.patterns
            assert tool.patterns["STRUCTURING"]["name"] == "Structuring"
            assert "cypher_template" in tool.patterns["STRUCTURING"]
            assert tool.patterns["STRUCTURING"]["risk_score"] == 0.8


@pytest.mark.asyncio
async def test_pattern_library_match_pattern(mock_neo4j_client):
    """Test PatternLibraryTool matching patterns."""
    # Mock open to return pattern library YAML
    pattern_yaml = """
patterns:
  STRUCTURING:
    name: Structuring
    description: Multiple transactions just below reporting thresholds
    cypher_template: |
      MATCH (a:Account)-[:SENT]->(t:Transaction)
      WHERE t.amount >= $min_amount AND t.amount <= $threshold
      WITH a, count(t) as tx_count
      WHERE tx_count >= $min_transactions
      RETURN a.id as account_id, tx_count, "STRUCTURING" as pattern_type
    parameters:
      threshold: 10000
      min_amount: 9000
      min_transactions: 3
    risk_score: 0.8
    """
    
    with patch("builtins.open", mock_open(read_data=pattern_yaml)):
        with patch("os.path.exists", return_value=True):
            # Create tool
            from backend.agents.tools.pattern_library_tool import PatternLibraryTool
            tool = PatternLibraryTool(neo4j_client=mock_neo4j_client)
            
            # Mock run_query to return pattern matches
            mock_neo4j_client.run_query.return_value = [
                {"account_id": "A123", "tx_count": 5, "pattern_type": "STRUCTURING"}
            ]
            
            # Run tool
            result = await tool._arun(
                pattern_type="STRUCTURING",
                parameters={"threshold": 9500}
            )
            
            # Parse result
            result_json = json.loads(result)
            
            # Verify result
            assert result_json["success"] is True
            assert "matches" in result_json
            assert len(result_json["matches"]) == 1
            assert result_json["matches"][0]["account_id"] == "A123"
            assert result_json["pattern_type"] == "STRUCTURING"
            assert "cypher_query" in result_json
            
            # Verify Neo4j client was called
            mock_neo4j_client.run_query.assert_called_once()
            
            # Verify parameters were merged
            args, kwargs = mock_neo4j_client.run_query.call_args
            assert "parameters" in kwargs
            assert kwargs["parameters"]["threshold"] == 9500  # Custom parameter
            assert kwargs["parameters"]["min_amount"] == 9000  # Default parameter


@pytest.mark.asyncio
async def test_pattern_library_llm_generation(mock_neo4j_client):
    """Test PatternLibraryTool with LLM-generated patterns."""
    # Create tool with empty patterns
    with patch("os.path.exists", return_value=False):
        # Mock GeminiClient
        with patch("backend.agents.tools.pattern_library_tool.GeminiClient") as mock_gemini_class:
            mock_gemini_instance = mock_gemini_class.return_value
            mock_gemini_instance.generate_cypher_query = AsyncMock(return_value="MATCH (n) RETURN n")
            
            # Create tool
            from backend.agents.tools.pattern_library_tool import PatternLibraryTool
            tool = PatternLibraryTool(neo4j_client=mock_neo4j_client)
            
            # Run tool with custom pattern
            result = await tool._arun(
                pattern_type="CUSTOM_PATTERN",
                pattern_description="Funds moving in circles",
                use_llm=True
            )
            
            # Parse result
            result_json = json.loads(result)
            
            # Verify result
            assert result_json["success"] is True
            assert "matches" in result_json
            assert "pattern_type" in result_json
            assert result_json["pattern_type"] == "CUSTOM_PATTERN"
            assert "llm_generated" in result_json
            assert result_json["llm_generated"] is True
            
            # Verify Gemini was called
            mock_gemini_instance.generate_cypher_query.assert_called_once()


@pytest.mark.asyncio
async def test_pattern_library_error_handling(mock_neo4j_client):
    """Test PatternLibraryTool error handling."""
    # Create tool with empty patterns
    with patch("os.path.exists", return_value=False):
        # Create tool
        from backend.agents.tools.pattern_library_tool import PatternLibraryTool
        tool = PatternLibraryTool(neo4j_client=mock_neo4j_client)
        
        # Run tool with non-existent pattern
        result = await tool._arun(
            pattern_type="NON_EXISTENT_PATTERN",
            use_llm=False
        )
        
        # Parse result
        result_json = json.loads(result)
        
        # Verify result
        assert result_json["success"] is False
        assert "error" in result_json
        assert "Pattern not found" in result_json["error"]


# ---- Tests for Neo4jSchemaTool ----

@pytest.mark.asyncio
async def test_neo4j_schema_get_schema(mock_neo4j_client):
    """Test Neo4jSchemaTool getting schema."""
    # Mock run_query to return schema
    mock_neo4j_client.run_query.return_value = [
        {"label": "Person", "properties": ["name", "id"]},
        {"label": "Account", "properties": ["number", "balance"]}
    ]
    
    # Create tool
    from backend.agents.tools.neo4j_schema_tool import Neo4jSchemaTool
    tool = Neo4jSchemaTool(neo4j_client=mock_neo4j_client)
    
    # Run tool
    result = await tool._arun()
    
    # Parse result
    result_json = json.loads(result)
    
    # Verify result
    assert result_json["success"] is True
    assert "schema" in result_json
    assert "nodes" in result_json["schema"]
    assert len(result_json["schema"]["nodes"]) == 2
    assert result_json["schema"]["nodes"][0]["label"] == "Person"
    assert "properties" in result_json["schema"]["nodes"][0]
    
    # Verify Neo4j client was called
    mock_neo4j_client.run_query.assert_called()


@pytest.mark.asyncio
async def test_neo4j_schema_get_relationships(mock_neo4j_client):
    """Test Neo4jSchemaTool getting relationships."""
    # Mock run_query to return relationships
    mock_neo4j_client.run_query.side_effect = [
        [{"label": "Person", "properties": ["name", "id"]}],  # First call for nodes
        [{"type": "OWNS", "source": "Person", "target": "Account"}]  # Second call for relationships
    ]
    
    # Create tool
    from backend.agents.tools.neo4j_schema_tool import Neo4jSchemaTool
    tool = Neo4jSchemaTool(neo4j_client=mock_neo4j_client)
    
    # Run tool
    result = await tool._arun()
    
    # Parse result
    result_json = json.loads(result)
    
    # Verify result
    assert result_json["success"] is True
    assert "schema" in result_json
    assert "relationships" in result_json["schema"]
    assert len(result_json["schema"]["relationships"]) == 1
    assert result_json["schema"]["relationships"][0]["type"] == "OWNS"
    assert result_json["schema"]["relationships"][0]["source"] == "Person"
    assert result_json["schema"]["relationships"][0]["target"] == "Account"


@pytest.mark.asyncio
async def test_neo4j_schema_error_handling(mock_neo4j_client):
    """Test Neo4jSchemaTool error handling."""
    # Mock run_query to raise an exception
    mock_neo4j_client.run_query.side_effect = Exception("Neo4j schema error")
    
    # Create tool
    from backend.agents.tools.neo4j_schema_tool import Neo4jSchemaTool
    tool = Neo4jSchemaTool(neo4j_client=mock_neo4j_client)
    
    # Run tool
    result = await tool._arun()
    
    # Parse result
    result_json = json.loads(result)
    
    # Verify result
    assert result_json["success"] is False
    assert "error" in result_json
    assert "Neo4j schema error" in result_json["error"]


# ---- Tests for JSON Response Format ----

@pytest.mark.parametrize("tool_fixture", [
    "template_engine_tool",
    "policy_docs_tool",
    "code_gen_tool",
    "graph_query_tool",
    "sandbox_exec_tool"
])
@pytest.mark.asyncio
async def test_tool_json_response_format(request, tool_fixture):
    """Test that all tools return properly formatted JSON responses."""
    # Get the tool fixture
    tool = request.getfixturevalue(tool_fixture)
    
    # Run the tool with minimal arguments
    if tool_fixture == "template_engine_tool":
        result = await tool._arun(template_name="markdown_report", data={"title": "Test"})
    elif tool_fixture == "policy_docs_tool":
        result = await tool._arun(query="test")
    elif tool_fixture == "code_gen_tool":
        result = await tool._arun(task_description="Print hello world")
    elif tool_fixture == "graph_query_tool":
        result = await tool._arun(query="MATCH (n) RETURN n LIMIT 1", use_gemini=False)
    elif tool_fixture == "sandbox_exec_tool":
        result = await tool._arun(code="print('hello')")
    
    # Verify result is valid JSON
    try:
        result_json = json.loads(result)
        
        # Verify common fields
        assert "success" in result_json
        assert isinstance(result_json["success"], bool)
        
        # If success is False, verify error field
        if not result_json["success"]:
            assert "error" in result_json
            assert isinstance(result_json["error"], str)
        
    except json.JSONDecodeError:
        pytest.fail(f"{tool_fixture} did not return valid JSON: {result}")
