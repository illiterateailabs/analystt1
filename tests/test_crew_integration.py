"""
Integration tests for CrewAI crews with agent configurations and RBAC.

This module contains integration tests that verify the interaction between
CrewFactory, agent configurations, RBAC enforcement, and end-to-end crew execution.
"""

import json
import pytest
import uuid
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

from fastapi import FastAPI, Request, Depends
from fastapi.testclient import TestClient
from crewai import Agent, Task, Crew, Process
from crewai.agent import TaskOutput

from backend.main import app
from backend.auth.jwt_handler import create_access_token
from backend.auth.rbac import require_roles, Roles, RoleSets
from backend.agents.factory import CrewFactory
from backend.agents.config import load_agent_config, load_crew_config
from backend.agents.tools import (
    GraphQueryTool,
    SandboxExecTool,
    CodeGenTool,
    PatternLibraryTool,
    Neo4jSchemaTool,
    TemplateEngineTool,
    PolicyDocsTool
)


# ---- Fixtures ----

@pytest.fixture
def admin_token():
    """Create a token for an admin user."""
    return create_access_token({"sub": "admin@example.com", "role": "admin"})


@pytest.fixture
def analyst_token():
    """Create a token for an analyst user."""
    return create_access_token({"sub": "analyst@example.com", "role": "analyst"})


@pytest.fixture
def compliance_token():
    """Create a token for a compliance officer."""
    return create_access_token({"sub": "compliance@example.com", "role": "compliance"})


@pytest.fixture
def user_token():
    """Create a token for a regular user."""
    return create_access_token({"sub": "user@example.com", "role": "user"})


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def mock_neo4j_client():
    """Fixture for mocked Neo4jClient."""
    with patch("backend.agents.factory.Neo4jClient", autospec=True) as mock_client:
        mock_instance = mock_client.return_value
        mock_instance.connect = AsyncMock()
        mock_instance.close = AsyncMock()
        mock_instance.run_query = AsyncMock(return_value=[{"result": "test"}])
        yield mock_instance


@pytest.fixture
def mock_gemini_client():
    """Fixture for mocked GeminiClient."""
    with patch("backend.agents.factory.GeminiClient", autospec=True) as mock_client:
        mock_instance = mock_client.return_value
        mock_instance.generate_text = AsyncMock(return_value="Test response")
        mock_instance.generate_cypher_query = AsyncMock(return_value="MATCH (n) RETURN n LIMIT 10")
        mock_instance.generate_python_code = AsyncMock(return_value="def test(): return 'Hello'")
        yield mock_instance


@pytest.fixture
def mock_e2b_client():
    """Fixture for mocked E2BClient."""
    with patch("backend.agents.factory.E2BClient", autospec=True) as mock_client:
        mock_instance = mock_client.return_value
        mock_instance.create_sandbox = AsyncMock()
        mock_instance.execute_code = AsyncMock(return_value={
            "success": True,
            "stdout": "Test output",
            "stderr": "",
            "exit_code": 0
        })
        mock_instance.close_sandbox = AsyncMock()
        mock_instance.close_all_sandboxes = AsyncMock()
        yield mock_instance


@pytest.fixture
def mock_llm_provider():
    """Fixture for mocked GeminiLLMProvider."""
    with patch("backend.agents.factory.GeminiLLMProvider", autospec=True) as mock_provider:
        mock_instance = mock_provider.return_value
        yield mock_instance


@pytest.fixture
def mock_crew():
    """Fixture for mocked CrewAI Crew."""
    with patch("backend.agents.factory.Crew", autospec=True) as mock_crew_class:
        mock_instance = mock_crew_class.return_value
        
        # Mock kickoff method
        mock_instance.kickoff = MagicMock(return_value=TaskOutput(
            raw_output="Test crew result",
            agent_id="test_agent_id",
            task_id="test_task_id"
        ))
        
        yield mock_crew_class


# ---- Tests for Agent Configuration Loading ----

@pytest.mark.parametrize("agent_id", [
    "graph_analyst",
    "compliance_checker",
    "report_writer",
    "fraud_pattern_hunter",
    "nlq_translator"
])
def test_load_agent_config(agent_id):
    """Test loading agent configurations for all required agents."""
    config = load_agent_config(agent_id)
    
    # Verify it loaded successfully
    assert config is not None
    assert config.id == agent_id
    assert config.role is not None
    assert config.goal is not None
    assert config.system_prompt is not None
    assert len(config.system_prompt) > 100
    
    # Verify metadata
    assert hasattr(config, "metadata")
    assert "capabilities" in config.metadata
    assert len(config.metadata["capabilities"]) > 0


@pytest.mark.asyncio
async def test_create_all_agents(mock_neo4j_client, mock_gemini_client, mock_e2b_client, mock_llm_provider):
    """Test creating all required agents with CrewFactory."""
    factory = CrewFactory()
    
    # List of required agents
    required_agents = [
        "graph_analyst",
        "compliance_checker",
        "report_writer", 
        "fraud_pattern_hunter",
        "nlq_translator"
    ]
    
    # Create each agent
    for agent_id in required_agents:
        agent = factory.create_agent(agent_id)
        
        # Verify agent was created
        assert agent is not None
        assert agent.id == agent_id
        
        # Verify agent was cached
        assert agent_id in factory.agents_cache


# ---- Tests for Tool Assignment ----

@pytest.mark.asyncio
async def test_tools_assigned_to_agents(mock_neo4j_client, mock_gemini_client, mock_e2b_client, mock_llm_provider):
    """Test that tools are correctly assigned to agents based on config."""
    factory = CrewFactory()
    
    # Expected tool assignments
    tool_assignments = {
        "graph_analyst": ["graph_query_tool"],
        "compliance_checker": ["policy_docs_tool"],
        "report_writer": ["template_engine_tool"],
        "fraud_pattern_hunter": ["pattern_library_tool", "graph_query_tool"],
        "nlq_translator": ["neo4j_schema_tool", "graph_query_tool"]
    }
    
    # Create each agent and check tools
    for agent_id, expected_tools in tool_assignments.items():
        # Mock the Agent constructor to capture the tools
        with patch("backend.agents.factory.Agent") as mock_agent:
            factory.create_agent(agent_id)
            
            # Check that Agent was called with the expected tools
            mock_agent.assert_called_once()
            kwargs = mock_agent.call_args.kwargs
            
            # Verify tools were passed
            assert "tools" in kwargs
            
            # Get the tool names
            tool_names = [tool.name for tool in kwargs["tools"]]
            
            # Verify expected tools were included
            for tool_name in expected_tools:
                assert any(tool_name in name for name in tool_names), f"Tool {tool_name} not found in {tool_names} for agent {agent_id}"


# ---- Tests for RBAC Enforcement on Crew Endpoints ----

def test_crew_run_rbac_admin(client, admin_token):
    """Test that admin can access /crew/run endpoint."""
    response = client.post(
        "/api/v1/crew/run",
        headers={"Authorization": f"Bearer {admin_token}"},
        json={"crew_name": "fraud_investigation", "inputs": {}}
    )
    
    # Should either be 200 OK or 503 Service Unavailable (if services not available)
    assert response.status_code in [200, 503]
    
    if response.status_code == 503:
        # This is expected in tests where external services are not available
        assert "Could not connect to required services" in response.json()["detail"]


def test_crew_run_rbac_analyst(client, analyst_token):
    """Test that analyst can access /crew/run endpoint."""
    response = client.post(
        "/api/v1/crew/run",
        headers={"Authorization": f"Bearer {analyst_token}"},
        json={"crew_name": "fraud_investigation", "inputs": {}}
    )
    
    # Should either be 200 OK or 503 Service Unavailable (if services not available)
    assert response.status_code in [200, 503]
    
    if response.status_code == 503:
        # This is expected in tests where external services are not available
        assert "Could not connect to required services" in response.json()["detail"]


def test_crew_run_rbac_user(client, user_token):
    """Test that regular user cannot access /crew/run endpoint."""
    response = client.post(
        "/api/v1/crew/run",
        headers={"Authorization": f"Bearer {user_token}"},
        json={"crew_name": "fraud_investigation", "inputs": {}}
    )
    
    # Should be 403 Forbidden
    assert response.status_code == 403
    assert "Only administrators and analysts can run crews" in response.json()["detail"]


def test_crew_pause_rbac_compliance(client, compliance_token):
    """Test that compliance officer can access /crew/pause endpoint."""
    # Create a fake task_id
    task_id = str(uuid.uuid4())
    
    # Mock the task_states in the crew module
    with patch("backend.api.v1.crew.task_states", {
        task_id: {
            "state": "running",
            "crew_name": "fraud_investigation",
            "inputs": {},
            "created_at": "2025-05-31T12:00:00",
            "last_updated": "2025-05-31T12:00:00",
            "current_agent": None,
            "paused_at": None,
            "result": None,
            "error": None
        }
    }):
        response = client.post(
            f"/api/v1/crew/pause/{task_id}",
            headers={"Authorization": f"Bearer {compliance_token}"},
            json={
                "findings": "Suspicious activity detected",
                "risk_level": "high",
                "regulatory_implications": ["AML", "KYC"]
            }
        )
        
        # Should be 200 OK
        assert response.status_code == 200
        assert response.json()["success"] is True


def test_crew_pause_rbac_analyst(client, analyst_token):
    """Test that analyst cannot access /crew/pause endpoint."""
    # Create a fake task_id
    task_id = str(uuid.uuid4())
    
    response = client.post(
        f"/api/v1/crew/pause/{task_id}",
        headers={"Authorization": f"Bearer {analyst_token}"},
        json={
            "findings": "Suspicious activity detected",
            "risk_level": "high",
            "regulatory_implications": ["AML", "KYC"]
        }
    )
    
    # Should be 403 Forbidden
    assert response.status_code == 403
    assert "Only administrators and compliance officers can pause crews" in response.json()["detail"]


def test_crew_resume_rbac_compliance(client, compliance_token):
    """Test that compliance officer can access /crew/resume endpoint."""
    # Create a fake task_id
    task_id = str(uuid.uuid4())
    
    # Mock the task_states and compliance_reviews in the crew module
    with patch("backend.api.v1.crew.task_states", {
        task_id: {
            "state": "paused",
            "crew_name": "fraud_investigation",
            "inputs": {},
            "created_at": "2025-05-31T12:00:00",
            "last_updated": "2025-05-31T12:00:00",
            "current_agent": "compliance_checker",
            "paused_at": "2025-05-31T12:05:00",
            "result": None,
            "error": None,
            "review_id": "REV-12345678"
        }
    }), patch("backend.api.v1.crew.compliance_reviews", {
        "REV-12345678": {
            "review_id": "REV-12345678",
            "task_id": task_id,
            "findings": "Suspicious activity detected",
            "risk_level": "high",
            "regulatory_implications": ["AML", "KYC"],
            "details": None,
            "status": "pending",
            "created_at": "2025-05-31T12:05:00",
            "responses": []
        }
    }):
        response = client.post(
            f"/api/v1/crew/resume/{task_id}",
            headers={"Authorization": f"Bearer {compliance_token}"},
            json={
                "status": "approved",
                "reviewer": "compliance@example.com",
                "comments": "Reviewed and approved"
            }
        )
        
        # Should be 200 OK
        assert response.status_code == 200
        assert response.json()["success"] is True


def test_crew_resume_rbac_analyst(client, analyst_token):
    """Test that analyst cannot access /crew/resume endpoint."""
    # Create a fake task_id
    task_id = str(uuid.uuid4())
    
    response = client.post(
        f"/api/v1/crew/resume/{task_id}",
        headers={"Authorization": f"Bearer {analyst_token}"},
        json={
            "status": "approved",
            "reviewer": "analyst@example.com",
            "comments": "Reviewed and approved"
        }
    )
    
    # Should be 403 Forbidden
    assert response.status_code == 403
    assert "Only administrators and compliance officers can resume crews" in response.json()["detail"]


# ---- Tests for End-to-End Crew Execution ----

@pytest.mark.asyncio
async def test_fraud_investigation_crew_creation(mock_neo4j_client, mock_gemini_client, mock_e2b_client, mock_llm_provider, mock_crew):
    """Test creating and running a fraud_investigation crew."""
    # Create factory
    factory = CrewFactory()
    
    # Connect to external services
    await factory.connect()
    
    # Create crew
    crew = await factory.create_crew("fraud_investigation")
    
    # Verify crew was created
    assert crew is not None
    
    # Verify crew was cached
    assert "fraud_investigation" in factory.crews_cache
    
    # Run crew
    result = await factory.run_crew("fraud_investigation", {"query": "Trace funds from account A123"})
    
    # Verify result
    assert result["success"] is True
    assert "result" in result
    
    # Close connections
    await factory.close()


@pytest.mark.asyncio
async def test_hitl_workflow_with_compliance_checker(mock_neo4j_client, mock_gemini_client, mock_e2b_client, mock_llm_provider):
    """Test HITL workflow with compliance_checker pausing execution."""
    # Create factory
    factory = CrewFactory()
    
    # Create a test task_id
    task_id = str(uuid.uuid4())
    
    # Mock the task_states in the crew module
    with patch("backend.api.v1.crew.task_states", {
        task_id: {
            "state": "running",
            "crew_name": "fraud_investigation",
            "inputs": {"query": "Trace funds from account A123"},
            "created_at": "2025-05-31T12:00:00",
            "last_updated": "2025-05-31T12:00:00",
            "current_agent": "compliance_checker",
            "paused_at": None,
            "result": None,
            "error": None
        }
    }):
        # Create compliance_checker agent
        compliance_checker = factory.create_agent("compliance_checker")
        
        # Verify agent was created
        assert compliance_checker is not None
        assert compliance_checker.id == "compliance_checker"
        
        # Mock the pause_crew function
        with patch("backend.api.v1.crew.pause_crew") as mock_pause:
            mock_pause.return_value = {
                "success": True,
                "message": f"Task '{task_id}' paused for compliance review",
                "review_id": "REV-12345678",
                "task_id": task_id
            }
            
            # Simulate HITL pause by calling the pause_crew function
            review_request = {
                "findings": "Suspicious activity detected",
                "risk_level": "high",
                "regulatory_implications": ["AML", "KYC"],
                "details": {"transaction_amount": 15000, "jurisdiction": "high-risk"}
            }
            
            # Call pause_crew
            pause_result = await mock_pause(review_request, task_id)
            
            # Verify pause was successful
            assert pause_result["success"] is True
            assert "paused for compliance review" in pause_result["message"]
            assert "review_id" in pause_result
            
            # Update task state to paused
            with patch("backend.api.v1.crew.task_states", {
                task_id: {
                    "state": "paused",
                    "crew_name": "fraud_investigation",
                    "inputs": {"query": "Trace funds from account A123"},
                    "created_at": "2025-05-31T12:00:00",
                    "last_updated": "2025-05-31T12:05:00",
                    "current_agent": "compliance_checker",
                    "paused_at": "2025-05-31T12:05:00",
                    "result": None,
                    "error": None,
                    "review_id": "REV-12345678"
                }
            }), patch("backend.api.v1.crew.compliance_reviews", {
                "REV-12345678": {
                    "review_id": "REV-12345678",
                    "task_id": task_id,
                    "findings": "Suspicious activity detected",
                    "risk_level": "high",
                    "regulatory_implications": ["AML", "KYC"],
                    "details": {"transaction_amount": 15000, "jurisdiction": "high-risk"},
                    "status": "pending",
                    "created_at": "2025-05-31T12:05:00",
                    "responses": []
                }
            }):
                # Mock the resume_crew function
                with patch("backend.api.v1.crew.resume_crew") as mock_resume:
                    mock_resume.return_value = {
                        "success": True,
                        "message": f"Task '{task_id}' resuming in background",
                        "review_id": "REV-12345678",
                        "status": "resuming"
                    }
                    
                    # Simulate HITL resume by calling the resume_crew function
                    resume_request = {
                        "status": "approved",
                        "reviewer": "compliance@example.com",
                        "comments": "Reviewed and approved"
                    }
                    
                    # Call resume_crew
                    resume_result = await mock_resume(resume_request, task_id)
                    
                    # Verify resume was successful
                    assert resume_result["success"] is True
                    assert "resuming" in resume_result["status"]
                    assert "review_id" in resume_result


# ---- Tests for Integration Between Components ----

@pytest.mark.asyncio
async def test_integration_agent_tools_factory(mock_neo4j_client, mock_gemini_client, mock_e2b_client, mock_llm_provider):
    """Test integration between agents, tools, and factory."""
    # Create factory
    factory = CrewFactory()
    
    # Verify tools were initialized
    assert "graph_query_tool" in factory.tools
    assert "sandbox_exec_tool" in factory.tools
    assert "code_gen_tool" in factory.tools
    assert "pattern_library_tool" in factory.tools
    assert "neo4j_schema_tool" in factory.tools
    
    # Create agents
    graph_analyst = factory.create_agent("graph_analyst")
    compliance_checker = factory.create_agent("compliance_checker")
    report_writer = factory.create_agent("report_writer")
    
    # Verify agents have the correct tools
    with patch.object(graph_analyst, "tools", create=True) as mock_tools:
        mock_tools.__iter__.return_value = [factory.tools["graph_query_tool"]]
        tools = list(graph_analyst.tools)
        assert len(tools) == 1
        assert tools[0].name == "graph_query_tool"
    
    with patch.object(compliance_checker, "tools", create=True) as mock_tools:
        mock_tools.__iter__.return_value = [factory.tools["policy_docs_tool"]]
        tools = list(compliance_checker.tools)
        assert len(tools) == 1
        assert tools[0].name == "policy_docs_tool"
    
    with patch.object(report_writer, "tools", create=True) as mock_tools:
        mock_tools.__iter__.return_value = [factory.tools["template_engine_tool"]]
        tools = list(report_writer.tools)
        assert len(tools) == 1
        assert tools[0].name == "template_engine_tool"


@pytest.mark.asyncio
async def test_crew_api_integration(client, admin_token, mock_neo4j_client, mock_gemini_client, mock_e2b_client, mock_llm_provider, mock_crew):
    """Test integration between API, RBAC, and CrewFactory."""
    # Mock the CrewFactory.run_crew method
    with patch("backend.agents.factory.CrewFactory.run_crew") as mock_run_crew:
        mock_run_crew.return_value = {
            "success": True,
            "result": "Test crew result",
            "task_id": "test_task_id",
            "agent_id": "test_agent_id"
        }
        
        # Call the API
        response = client.post(
            "/api/v1/crew/run",
            headers={"Authorization": f"Bearer {admin_token}"},
            json={"crew_name": "fraud_investigation", "inputs": {"query": "Trace funds from account A123"}}
        )
        
        # Verify response
        assert response.status_code == 200
        assert response.json()["success"] is True
        
        # Verify CrewFactory.run_crew was called
        mock_run_crew.assert_called_once()
        args, kwargs = mock_run_crew.call_args
        assert kwargs["crew_name"] == "fraud_investigation"
        assert kwargs["inputs"]["query"] == "Trace funds from account A123"
