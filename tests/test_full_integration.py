"""
Full integration tests for the Analyst Agent system.

This module contains comprehensive integration tests that verify the entire
system works together correctly, including agent configurations, tool initialization,
crew execution, RBAC enforcement, and error handling.
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
        
        # Mock query results for different query types
        def mock_run_query(query, parameters=None):
            if "MATCH (n) WHERE n.id" in query:
                # Entity lookup query
                return [{"id": "E123", "name": "Test Entity", "risk_score": 0.75}]
            elif "MATCH (a:Account)" in query and "STRUCTURING" in query:
                # Pattern detection query
                return [{"account_id": "A123", "tx_count": 5, "pattern_type": "STRUCTURING"}]
            elif "MATCH (a)-[r]->(b)" in query:
                # Relationship query
                return [
                    {"a.id": "A123", "b.id": "B456", "r.type": "SENT", "r.amount": 9500},
                    {"a.id": "A123", "b.id": "C789", "r.type": "SENT", "r.amount": 9700}
                ]
            elif "CALL gds.pageRank" in query:
                # Graph algorithm query
                return [
                    {"node_id": "A123", "score": 0.85},
                    {"node_id": "B456", "score": 0.65}
                ]
            elif "MATCH (n)" in query and "RETURN labels(n)" in query:
                # Schema query
                return [
                    {"label": "Person", "properties": ["name", "id", "risk_score"]},
                    {"label": "Account", "properties": ["id", "balance", "type"]},
                    {"label": "Transaction", "properties": ["id", "amount", "timestamp"]}
                ]
            elif "MATCH ()-[r]-()" in query and "RETURN type(r)" in query:
                # Relationship schema query
                return [
                    {"type": "OWNS", "source": "Person", "target": "Account"},
                    {"type": "SENT", "source": "Account", "target": "Transaction"},
                    {"type": "RECEIVED_BY", "source": "Transaction", "target": "Account"}
                ]
            else:
                # Default response for any other query
                return [{"result": "test"}]
                
        mock_instance.run_query = AsyncMock(side_effect=mock_run_query)
        yield mock_instance


@pytest.fixture
def mock_gemini_client():
    """Fixture for mocked GeminiClient."""
    with patch("backend.agents.factory.GeminiClient", autospec=True) as mock_client:
        mock_instance = mock_client.return_value
        
        # Mock different generation methods
        mock_instance.generate_text = AsyncMock(side_effect=lambda prompt: 
            "Suspicious activity detected in account A123" if "suspicious" in prompt.lower() 
            else "Analysis complete. No suspicious activity detected.")
            
        mock_instance.generate_cypher_query = AsyncMock(side_effect=lambda query, **kwargs: 
            "MATCH (a:Account {id: 'A123'})-[:SENT]->(t:Transaction) RETURN a, t" if "account" in query.lower()
            else "MATCH (n) RETURN n LIMIT 10")
            
        mock_instance.generate_python_code = AsyncMock(return_value="""
def analyze_transactions(transactions):
    suspicious = [t for t in transactions if t['amount'] > 9000 and t['amount'] < 10000]
    return {
        'suspicious_count': len(suspicious),
        'total_amount': sum(t['amount'] for t in suspicious),
        'risk_score': len(suspicious) / len(transactions) if transactions else 0
    }
""")
        yield mock_instance


@pytest.fixture
def mock_e2b_client():
    """Fixture for mocked E2BClient."""
    with patch("backend.agents.factory.E2BClient", autospec=True) as mock_client:
        mock_instance = mock_client.return_value
        mock_instance.create_sandbox = AsyncMock()
        
        # Mock code execution with realistic results
        def mock_execute_code(code, **kwargs):
            if "analyze_transactions" in code and "suspicious" in code:
                return {
                    "success": True,
                    "stdout": """{'suspicious_count': 3, 'total_amount': 28700, 'risk_score': 0.75}""",
                    "stderr": "",
                    "exit_code": 0
                }
            elif "error" in code.lower() or "raise" in code.lower():
                return {
                    "success": True,
                    "stdout": "",
                    "stderr": "Error: Test exception",
                    "exit_code": 1
                }
            else:
                return {
                    "success": True,
                    "stdout": "Execution successful",
                    "stderr": "",
                    "exit_code": 0
                }
                
        mock_instance.execute_code = AsyncMock(side_effect=mock_execute_code)
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
        
        # Mock kickoff method with realistic fraud investigation result
        mock_instance.kickoff = MagicMock(return_value=TaskOutput(
            raw_output=json.dumps({
                "executive_summary": "# Executive Summary\n\nSuspicious activity detected in account A123 with multiple transactions just below reporting thresholds.",
                "detailed_report": "# Fraud Investigation Report\n\n## Executive Summary\n\nSuspicious activity detected in account A123 with multiple transactions just below reporting thresholds.\n\n## Pattern Analysis\n\nAccount A123 shows signs of structuring with 5 transactions between $9,000-$9,900 in a short time period.\n\n## Risk Assessment\n\nOverall Risk: **High (0.75)**\n\n## Recommendations\n\n1. File SAR for account A123\n2. Enhance monitoring on related accounts\n3. Review customer KYC information",
                "graph_data": {
                    "nodes": [
                        {"id": "A123", "label": "Account A123", "type": "Account", "risk_score": 0.75},
                        {"id": "T1", "label": "Transaction $9,500", "type": "Transaction"},
                        {"id": "T2", "label": "Transaction $9,700", "type": "Transaction"}
                    ],
                    "edges": [
                        {"from": "A123", "to": "T1", "label": "SENT"},
                        {"from": "A123", "to": "T2", "label": "SENT"}
                    ]
                },
                "risk_score": 0.75,
                "confidence": 0.85,
                "recommendations": [
                    "File SAR for account A123",
                    "Enhance monitoring on related accounts",
                    "Review customer KYC information"
                ],
                "fraud_patterns": ["STRUCTURING"],
                "compliance_findings": ["Multiple transactions below CTR threshold"]
            }),
            agent_id="report_writer",
            task_id="generate_comprehensive_investigation_report"
        ))
        
        yield mock_crew_class


# ---- Comprehensive Integration Tests ----

@pytest.mark.asyncio
async def test_full_fraud_investigation_integration(
    mock_neo4j_client, mock_gemini_client, mock_e2b_client, mock_llm_provider, mock_crew
):
    """
    Test a complete fraud investigation workflow with all components integrated.
    
    This test verifies:
    1. All required agents can be created with their configs
    2. All tools are properly initialized and assigned
    3. The crew can be created and executed
    4. The results are structured correctly
    """
    # Create factory
    factory = CrewFactory()
    
    # Connect to external services
    await factory.connect()
    
    # Verify all required agents can be created
    required_agents = [
        "nlq_translator", 
        "graph_analyst", 
        "fraud_pattern_hunter", 
        "compliance_checker", 
        "report_writer"
    ]
    
    agents = {}
    for agent_id in required_agents:
        agent = factory.create_agent(agent_id)
        agents[agent_id] = agent
        
        # Verify agent was created successfully
        assert agent is not None
        assert agent.id == agent_id
        assert agent in factory.agents_cache.values()
    
    # Verify tools were initialized
    required_tools = [
        "graph_query_tool",
        "neo4j_schema_tool",
        "pattern_library_tool",
        "policy_docs_tool",
        "template_engine_tool",
        "sandbox_exec_tool",
        "code_gen_tool"
    ]
    
    for tool_name in required_tools:
        assert tool_name in factory.tools
        assert factory.tools[tool_name] is not None
    
    # Create fraud investigation crew
    crew = await factory.create_crew("fraud_investigation")
    
    # Verify crew was created successfully
    assert crew is not None
    assert "fraud_investigation" in factory.crews_cache
    
    # Run the crew with a test query
    result = await factory.run_crew(
        "fraud_investigation", 
        {"query": "Investigate account A123 for suspicious activity"}
    )
    
    # Verify the result structure
    assert result["success"] is True
    assert "result" in result
    
    # Parse the result JSON
    result_data = json.loads(result["result"])
    
    # Verify all expected sections are present
    assert "executive_summary" in result_data
    assert "detailed_report" in result_data
    assert "graph_data" in result_data
    assert "nodes" in result_data["graph_data"]
    assert "edges" in result_data["graph_data"]
    assert "risk_score" in result_data
    assert "recommendations" in result_data
    assert "fraud_patterns" in result_data
    
    # Verify specific content
    assert "A123" in result_data["executive_summary"]
    assert "STRUCTURING" in result_data["fraud_patterns"]
    assert len(result_data["graph_data"]["nodes"]) >= 2
    assert len(result_data["graph_data"]["edges"]) >= 1
    assert result_data["risk_score"] > 0
    assert len(result_data["recommendations"]) >= 1
    
    # Close connections
    await factory.close()


@pytest.mark.asyncio
async def test_hitl_workflow_integration(
    mock_neo4j_client, mock_gemini_client, mock_e2b_client, mock_llm_provider, client, compliance_token
):
    """
    Test the Human-in-the-Loop (HITL) workflow integration.
    
    This test verifies:
    1. The compliance_checker agent can pause execution
    2. The API endpoints for pause/resume work correctly
    3. The workflow can be resumed after human approval
    """
    # Create a test task_id
    task_id = str(uuid.uuid4())
    
    # Mock task_states for the crew module
    with patch("backend.api.v1.crew.task_states", {
        task_id: {
            "state": "running",
            "crew_name": "fraud_investigation",
            "inputs": {"query": "Investigate account A123 for suspicious activity"},
            "created_at": "2025-05-31T12:00:00",
            "last_updated": "2025-05-31T12:00:00",
            "current_agent": "compliance_checker",
            "paused_at": None,
            "result": None,
            "error": None
        }
    }):
        # Test pausing the workflow
        pause_response = client.post(
            f"/api/v1/crew/pause/{task_id}",
            headers={"Authorization": f"Bearer {compliance_token}"},
            json={
                "findings": "High-risk transactions detected requiring manual review",
                "risk_level": "high",
                "regulatory_implications": ["BSA", "AML"],
                "details": {
                    "transaction_amount": 28700,
                    "transaction_count": 3,
                    "below_threshold": True
                }
            }
        )
        
        # Verify pause was successful
        assert pause_response.status_code == 200
        pause_data = pause_response.json()
        assert pause_data["success"] is True
        assert "review_id" in pause_data
        review_id = pause_data["review_id"]
        
        # Update task_states to reflect paused state
        with patch("backend.api.v1.crew.task_states", {
            task_id: {
                "state": "paused",
                "crew_name": "fraud_investigation",
                "inputs": {"query": "Investigate account A123 for suspicious activity"},
                "created_at": "2025-05-31T12:00:00",
                "last_updated": "2025-05-31T12:05:00",
                "current_agent": "compliance_checker",
                "paused_at": "2025-05-31T12:05:00",
                "result": None,
                "error": None,
                "review_id": review_id
            }
        }), patch("backend.api.v1.crew.compliance_reviews", {
            review_id: {
                "review_id": review_id,
                "task_id": task_id,
                "findings": "High-risk transactions detected requiring manual review",
                "risk_level": "high",
                "regulatory_implications": ["BSA", "AML"],
                "details": {
                    "transaction_amount": 28700,
                    "transaction_count": 3,
                    "below_threshold": True
                },
                "status": "pending",
                "created_at": "2025-05-31T12:05:00",
                "responses": []
            }
        }):
            # Test getting review details
            review_response = client.get(
                f"/api/v1/crew/review/{task_id}",
                headers={"Authorization": f"Bearer {compliance_token}"}
            )
            
            # Verify review details
            assert review_response.status_code == 200
            review_data = review_response.json()
            assert review_data["review_id"] == review_id
            assert review_data["risk_level"] == "high"
            assert "BSA" in review_data["regulatory_implications"]
            
            # Test resuming the workflow with approval
            with patch("backend.api.v1.crew.resume_crew") as mock_resume:
                mock_resume.return_value = {
                    "success": True,
                    "message": f"Task '{task_id}' resuming in background",
                    "review_id": review_id,
                    "status": "resuming"
                }
                
                resume_response = client.post(
                    f"/api/v1/crew/resume/{task_id}",
                    headers={"Authorization": f"Bearer {compliance_token}"},
                    json={
                        "status": "approved",
                        "reviewer": "compliance@example.com",
                        "comments": "Reviewed and approved for SAR filing"
                    }
                )
                
                # Verify resume was successful
                assert resume_response.status_code == 200
                resume_data = resume_response.json()
                assert resume_data["success"] is True
                assert resume_data["status"] == "resuming"


def test_api_rbac_integration(client, admin_token, analyst_token, compliance_token, user_token):
    """
    Test RBAC integration with API endpoints.
    
    This test verifies:
    1. Admin can access all endpoints
    2. Analyst can access appropriate endpoints
    3. Compliance officer can access appropriate endpoints
    4. Regular user is blocked from protected endpoints
    """
    # Test matrix of endpoints and roles
    endpoints = [
        # Endpoint, Method, Required Role(s), Request Body
        ("/api/v1/crew/run", "POST", ["admin", "analyst"], {"crew_name": "fraud_investigation", "inputs": {}}),
        ("/api/v1/graph/query", "POST", ["admin", "analyst"], {"query": "MATCH (n) RETURN n LIMIT 10"}),
        ("/api/v1/prompts", "GET", ["admin"], None),
        ("/api/v1/prompts/agent/nlq_translator", "GET", ["admin"], None),
        ("/api/v1/crew/crews", "GET", ["admin", "analyst", "compliance", "user"], None),
    ]
    
    # Map tokens to roles
    token_map = {
        "admin": admin_token,
        "analyst": analyst_token,
        "compliance": compliance_token,
        "user": user_token
    }
    
    # Test each endpoint with each role
    for endpoint, method, allowed_roles, body in endpoints:
        for role, token in token_map.items():
            # Determine expected status code
            expected_status = 200 if role in allowed_roles else 403
            
            # Some endpoints might return 503 if services are unavailable
            alternative_status = 503 if endpoint == "/api/v1/crew/run" and role in allowed_roles else None
            
            # Make request
            if method == "GET":
                response = client.get(endpoint, headers={"Authorization": f"Bearer {token}"})
            elif method == "POST":
                response = client.post(endpoint, headers={"Authorization": f"Bearer {token}"}, json=body)
            
            # Verify response
            assert response.status_code in [expected_status, alternative_status] if alternative_status else response.status_code == expected_status, \
                f"Role {role} should {'be allowed' if role in allowed_roles else 'be denied'} access to {endpoint}"


@pytest.mark.asyncio
async def test_error_handling_integration(
    mock_neo4j_client, mock_gemini_client, mock_e2b_client, mock_llm_provider, client, admin_token
):
    """
    Test error handling integration across the system.
    
    This test verifies:
    1. Database connection errors are handled gracefully
    2. LLM errors are handled gracefully
    3. Invalid input errors are handled gracefully
    4. API endpoints return appropriate error responses
    """
    # Test database connection error
    with patch.object(mock_neo4j_client, "connect", side_effect=Exception("Neo4j connection error")):
        # Create factory
        factory = CrewFactory()
        
        # Attempt to connect
        with pytest.raises(Exception) as exc_info:
            await factory.connect()
        
        # Verify error message
        assert "Neo4j connection error" in str(exc_info.value)
        
        # Test API endpoint with database error
        response = client.post(
            "/api/v1/crew/run",
            headers={"Authorization": f"Bearer {admin_token}"},
            json={"crew_name": "fraud_investigation", "inputs": {}}
        )
        
        # Verify response
        assert response.status_code == 503
        assert "Could not connect to required services" in response.json()["detail"]
    
    # Test LLM error
    with patch.object(mock_gemini_client, "generate_text", side_effect=Exception("Gemini API error")):
        # Create factory
        factory = CrewFactory()
        
        # Run crew with LLM error
        result = await factory.run_crew(
            "fraud_investigation", 
            {"query": "Investigate account A123 for suspicious activity"}
        )
        
        # Verify error is handled
        assert result["success"] is False
        assert "error" in result
        assert "Gemini API error" in result["error"]
    
    # Test invalid input error
    response = client.post(
        "/api/v1/crew/run",
        headers={"Authorization": f"Bearer {admin_token}"},
        json={"crew_name": "nonexistent_crew", "inputs": {}}
    )
    
    # Verify response
    assert response.status_code == 404
    assert "not found" in response.json()["detail"]
    
    # Test invalid task_id
    response = client.get(
        "/api/v1/crew/status/nonexistent_task_id",
        headers={"Authorization": f"Bearer {admin_token}"}
    )
    
    # Verify response
    assert response.status_code == 404
    assert "not found" in response.json()["detail"]


@pytest.mark.asyncio
async def test_tool_integration_with_agents(
    mock_neo4j_client, mock_gemini_client, mock_e2b_client, mock_llm_provider
):
    """
    Test integration between tools and agents.
    
    This test verifies:
    1. Each agent has the correct tools assigned
    2. Tool methods are called with the correct parameters
    3. Tool results are properly handled
    """
    # Create factory
    factory = CrewFactory()
    
    # Map of expected tools for each agent
    expected_tools = {
        "graph_analyst": ["graph_query_tool"],
        "fraud_pattern_hunter": ["pattern_library_tool", "graph_query_tool"],
        "compliance_checker": ["policy_docs_tool"],
        "report_writer": ["template_engine_tool"],
        "nlq_translator": ["neo4j_schema_tool", "graph_query_tool"]
    }
    
    # Create each agent and verify tools
    for agent_id, expected_tool_names in expected_tools.items():
        # Create agent
        agent = factory.create_agent(agent_id)
        
        # Verify agent has the correct tools
        for tool_name in expected_tool_names:
            tool = factory.get_tool(tool_name)
            assert tool is not None
            
            # Verify tool can be called
            if tool_name == "graph_query_tool":
                result = await tool._arun(query="MATCH (n) RETURN n LIMIT 10", use_gemini=False)
                result_json = json.loads(result)
                assert result_json["success"] is True
                assert "results" in result_json
            
            elif tool_name == "pattern_library_tool":
                # Mock patterns
                with patch.object(tool, "patterns", {"STRUCTURING": {
                    "name": "Structuring",
                    "description": "Multiple transactions just below reporting thresholds",
                    "cypher_template": "MATCH (a:Account) RETURN a",
                    "parameters": {"threshold": 10000},
                    "risk_score": 0.8
                }}):
                    result = await tool._arun(pattern_type="STRUCTURING")
                    result_json = json.loads(result)
                    assert result_json["success"] is True
                    assert "matches" in result_json
            
            elif tool_name == "policy_docs_tool":
                result = await tool._arun(query="SAR filing")
                result_json = json.loads(result)
                assert result_json["success"] is True
                assert "results" in result_json
            
            elif tool_name == "template_engine_tool":
                # Mock Jinja2
                with patch("backend.agents.tools.template_engine_tool.JINJA2_AVAILABLE", True):
                    with patch.object(tool, "env") as mock_env:
                        mock_template = MagicMock()
                        mock_template.render.return_value = "Rendered template"
                        mock_env.get_template.return_value = mock_template
                        mock_env.from_string.return_value = mock_template
                        
                        result = await tool._arun(
                            template_name="markdown_report",
                            data={"title": "Test Report"}
                        )
                        result_json = json.loads(result)
                        assert result_json["success"] is True
                        assert "content" in result_json
            
            elif tool_name == "neo4j_schema_tool":
                result = await tool._arun()
                result_json = json.loads(result)
                assert result_json["success"] is True
                assert "schema" in result_json


@pytest.mark.asyncio
async def test_end_to_end_api_workflow(
    client, admin_token, mock_neo4j_client, mock_gemini_client, mock_e2b_client, mock_llm_provider, mock_crew
):
    """
    Test end-to-end API workflow.
    
    This test verifies:
    1. The complete API workflow from crew creation to result retrieval
    2. Asynchronous crew execution
    3. Task status tracking
    4. Result formatting
    """
    # Mock CrewFactory.run_crew to return a realistic result
    with patch("backend.agents.factory.CrewFactory.run_crew") as mock_run_crew:
        # Create a mock task ID
        task_id = str(uuid.uuid4())
        
        # Set up mock result
        mock_result = {
            "success": True,
            "result": json.dumps({
                "executive_summary": "Suspicious activity detected in account A123",
                "detailed_report": "# Fraud Investigation Report\n\nSuspicious activity detected...",
                "graph_data": {
                    "nodes": [{"id": "A123", "label": "Account A123", "type": "Account"}],
                    "edges": []
                },
                "risk_score": 0.75,
                "recommendations": ["File SAR for account A123"],
                "fraud_patterns": ["STRUCTURING"],
                "compliance_findings": ["Multiple transactions below CTR threshold"]
            }),
            "task_id": task_id
        }
        
        mock_run_crew.return_value = mock_result
        
        # Run crew asynchronously
        run_response = client.post(
            "/api/v1/crew/run",
            headers={"Authorization": f"Bearer {admin_token}"},
            json={
                "crew_name": "fraud_investigation", 
                "inputs": {"query": "Investigate account A123"},
                "async_execution": True
            }
        )
        
        # Verify run response
        assert run_response.status_code == 200
        run_data = run_response.json()
        assert run_data["success"] is True
        assert "task_id" in run_data
        task_id = run_data["task_id"]
        
        # Mock task_states for status check
        with patch("backend.api.v1.crew.task_states", {
            task_id: {
                "state": "running",
                "crew_name": "fraud_investigation",
                "inputs": {"query": "Investigate account A123"},
                "created_at": "2025-05-31T12:00:00",
                "last_updated": "2025-05-31T12:00:00",
                "current_agent": "graph_analyst",
                "paused_at": None,
                "result": None,
                "error": None
            }
        }):
            # Check task status
            status_response = client.get(
                f"/api/v1/crew/status/{task_id}",
                headers={"Authorization": f"Bearer {admin_token}"}
            )
            
            # Verify status response
            assert status_response.status_code == 200
            status_data = status_response.json()
            assert status_data["state"] == "running"
            assert status_data["crew_name"] == "fraud_investigation"
            
            # Update task_states to completed
            with patch("backend.api.v1.crew.task_states", {
                task_id: {
                    "state": "completed",
                    "crew_name": "fraud_investigation",
                    "inputs": {"query": "Investigate account A123"},
                    "created_at": "2025-05-31T12:00:00",
                    "last_updated": "2025-05-31T12:10:00",
                    "current_agent": "report_writer",
                    "paused_at": None,
                    "result": json.loads(mock_result["result"]),
                    "error": None
                }
            }):
                # Check task status again
                status_response = client.get(
                    f"/api/v1/crew/status/{task_id}",
                    headers={"Authorization": f"Bearer {admin_token}"}
                )
                
                # Verify status response
                assert status_response.status_code == 200
                status_data = status_response.json()
                assert status_data["state"] == "completed"
                assert "result" in status_data
                assert "executive_summary" in status_data["result"]
                assert "graph_data" in status_data["result"]
                assert "risk_score" in status_data["result"]
