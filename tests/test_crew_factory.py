"""
Tests for the CrewFactory class.

This module contains tests for the CrewFactory class, which is responsible
for building and managing CrewAI crews, including agent creation, tool
assignment, task definition, and crew orchestration.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

from crewai import Agent, Task, Crew, Process
from crewai.agent import TaskOutput

from backend.agents.factory import CrewFactory
from backend.agents.config import AgentConfig, CrewConfig
from backend.agents.tools import (
    GraphQueryTool,
    SandboxExecTool,
    CodeGenTool,
    PatternLibraryTool,
    Neo4jSchemaTool,
)
from backend.integrations.neo4j_client import Neo4jClient
from backend.integrations.gemini_client import GeminiClient
from backend.integrations.e2b_client import E2BClient
from backend.agents.llm import GeminiLLMProvider


# ---- Fixtures ----

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
def mock_agent():
    """Fixture for mocked CrewAI Agent."""
    with patch("backend.agents.factory.Agent", autospec=True) as mock_agent_class:
        mock_instance = mock_agent_class.return_value
        mock_instance.id = "test_agent_id"
        mock_instance.role = "test_role"
        mock_instance.goal = "test_goal"
        yield mock_agent_class


@pytest.fixture
def mock_task():
    """Fixture for mocked CrewAI Task."""
    with patch("backend.agents.factory.Task", autospec=True) as mock_task_class:
        mock_instance = mock_task_class.return_value
        mock_instance.description = "test_description"
        mock_instance.expected_output = "test_output"
        yield mock_task_class


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


@pytest.fixture
def mock_load_agent_config():
    """Fixture for mocked load_agent_config function."""
    with patch("backend.agents.factory.load_agent_config") as mock_load:
        # Create a mock agent config
        mock_config = MagicMock(spec=AgentConfig)
        mock_config.id = "test_agent"
        mock_config.role = "Test Agent"
        mock_config.goal = "Test the system"
        mock_config.backstory = "A test agent for unit testing"
        mock_config.verbose = True
        mock_config.allow_delegation = True
        mock_config.tools = ["graph_query_tool", "code_gen_tool"]
        mock_config.max_iter = 10
        mock_config.max_rpm = 10
        mock_config.memory = True
        
        mock_load.return_value = mock_config
        yield mock_load


@pytest.fixture
def mock_load_crew_config():
    """Fixture for mocked load_crew_config function."""
    with patch("backend.agents.factory.load_crew_config") as mock_load:
        # Create a mock crew config
        mock_config = MagicMock(spec=CrewConfig)
        mock_config.name = "test_crew"
        mock_config.description = "A test crew for unit testing"
        mock_config.agents = ["test_agent", "test_agent2"]
        mock_config.process_type = "sequential"
        mock_config.verbose = True
        mock_config.max_rpm = 10
        mock_config.memory = True
        mock_config.cache = True
        mock_config.manager = "test_agent"
        
        mock_load.return_value = mock_config
        yield mock_load


@pytest.fixture
def mock_tools():
    """Fixture for mocked tools."""
    # Mock all tool classes
    with patch("backend.agents.factory.GraphQueryTool", autospec=True) as mock_graph_tool:
        with patch("backend.agents.factory.SandboxExecTool", autospec=True) as mock_sandbox_tool:
            with patch("backend.agents.factory.CodeGenTool", autospec=True) as mock_code_tool:
                with patch("backend.agents.factory.PatternLibraryTool", autospec=True) as mock_pattern_tool:
                    with patch("backend.agents.factory.Neo4jSchemaTool", autospec=True) as mock_schema_tool:
                        with patch("backend.agents.factory.create_crypto_tools") as mock_crypto_tools:
                            # Setup mock instances
                            mock_graph_instance = mock_graph_tool.return_value
                            mock_sandbox_instance = mock_sandbox_tool.return_value
                            mock_code_instance = mock_code_tool.return_value
                            mock_pattern_instance = mock_pattern_tool.return_value
                            mock_schema_instance = mock_schema_tool.return_value
                            
                            # Setup crypto tools
                            mock_crypto_tools.return_value = {
                                "dune_analytics_tool": MagicMock(),
                                "defillama_tool": MagicMock(),
                                "etherscan_tool": MagicMock()
                            }
                            
                            yield {
                                "graph_query_tool": mock_graph_instance,
                                "sandbox_exec_tool": mock_sandbox_instance,
                                "code_gen_tool": mock_code_instance,
                                "pattern_library_tool": mock_pattern_instance,
                                "neo4j_schema_tool": mock_schema_instance,
                                "crypto_tools": mock_crypto_tools
                            }


# ---- Tests for CrewFactory initialization ----

@pytest.mark.asyncio
async def test_crew_factory_init(mock_neo4j_client, mock_gemini_client, mock_e2b_client, mock_llm_provider, mock_tools):
    """Test CrewFactory initialization."""
    # Create factory
    factory = CrewFactory()
    
    # Verify clients were initialized
    assert factory.neo4j_client is not None
    assert factory.gemini_client is not None
    assert factory.e2b_client is not None
    assert factory.llm_provider is not None
    
    # Verify tools were initialized
    assert "graph_query_tool" in factory.tools
    assert "sandbox_exec_tool" in factory.tools
    assert "code_gen_tool" in factory.tools
    assert "pattern_library_tool" in factory.tools
    assert "neo4j_schema_tool" in factory.tools
    
    # Verify caches were initialized
    assert factory.agents_cache == {}
    assert factory.crews_cache == {}


@pytest.mark.asyncio
async def test_crew_factory_connect(mock_neo4j_client):
    """Test connecting to external services."""
    # Create factory
    factory = CrewFactory()
    
    # Connect
    await factory.connect()
    
    # Verify Neo4j connection was attempted
    mock_neo4j_client.connect.assert_called_once()


@pytest.mark.asyncio
async def test_crew_factory_connect_error(mock_neo4j_client):
    """Test error handling when connecting to external services."""
    # Setup Neo4j client to raise exception
    mock_neo4j_client.connect.side_effect = Exception("Connection error")
    
    # Create factory
    factory = CrewFactory()
    
    # Verify exception is propagated
    with pytest.raises(Exception) as exc_info:
        await factory.connect()
    
    assert "Connection error" in str(exc_info.value)


@pytest.mark.asyncio
async def test_crew_factory_close(mock_neo4j_client, mock_e2b_client):
    """Test closing connections to external services."""
    # Create factory
    factory = CrewFactory()
    
    # Set up Neo4j client with driver
    factory.neo4j_client.driver = MagicMock()
    
    # Close connections
    await factory.close()
    
    # Verify Neo4j connection was closed
    mock_neo4j_client.close.assert_called_once()


@pytest.mark.asyncio
async def test_crew_factory_close_no_driver(mock_neo4j_client):
    """Test closing connections when no driver exists."""
    # Create factory
    factory = CrewFactory()
    
    # Set up Neo4j client without driver
    factory.neo4j_client.driver = None
    
    # Close connections
    await factory.close()
    
    # Verify Neo4j connection was not closed
    mock_neo4j_client.close.assert_not_called()


# ---- Tests for tool management ----

def test_get_tool_existing(mock_tools):
    """Test getting an existing tool."""
    # Create factory
    factory = CrewFactory()
    
    # Get tool
    tool = factory.get_tool("graph_query_tool")
    
    # Verify tool was returned
    assert tool is not None
    assert tool == factory.tools["graph_query_tool"]


def test_get_tool_nonexistent(mock_tools):
    """Test getting a nonexistent tool."""
    # Create factory
    factory = CrewFactory()
    
    # Get tool
    tool = factory.get_tool("nonexistent_tool")
    
    # Verify None was returned
    assert tool is None


# ---- Tests for agent creation ----

def test_create_agent_new(mock_agent, mock_load_agent_config, mock_tools, mock_llm_provider):
    """Test creating a new agent."""
    # Create factory
    factory = CrewFactory()
    
    # Create agent
    agent = factory.create_agent("test_agent")
    
    # Verify agent was created
    assert agent is not None
    
    # Verify agent was cached
    assert "test_agent" in factory.agents_cache
    assert factory.agents_cache["test_agent"] == agent
    
    # Verify Agent constructor was called with correct parameters
    mock_agent.assert_called_once()
    args, kwargs = mock_agent.call_args
    assert kwargs["id"] == "test_agent"
    assert kwargs["role"] == "Test Agent"
    assert kwargs["goal"] == "Test the system"
    assert kwargs["backstory"] == "A test agent for unit testing"
    assert kwargs["verbose"] is True
    assert kwargs["allow_delegation"] is True
    assert len(kwargs["tools"]) > 0
    assert kwargs["llm"] == factory.llm_provider


def test_create_agent_cached(mock_agent, mock_load_agent_config):
    """Test creating an agent that's already cached."""
    # Create factory
    factory = CrewFactory()
    
    # Create a mock cached agent
    mock_cached_agent = MagicMock()
    factory.agents_cache["test_agent"] = mock_cached_agent
    
    # Create agent
    agent = factory.create_agent("test_agent")
    
    # Verify cached agent was returned
    assert agent == mock_cached_agent
    
    # Verify Agent constructor was not called
    mock_agent.assert_not_called()


def test_create_agent_missing_tool(mock_agent, mock_load_agent_config):
    """Test creating an agent with a missing tool."""
    # Create factory
    factory = CrewFactory()
    
    # Modify agent config to include a nonexistent tool
    mock_load_agent_config.return_value.tools = ["nonexistent_tool", "graph_query_tool"]
    
    # Create agent
    agent = factory.create_agent("test_agent")
    
    # Verify agent was created
    assert agent is not None
    
    # Verify Agent constructor was called with only the existing tool
    mock_agent.assert_called_once()
    args, kwargs = mock_agent.call_args
    assert len(kwargs["tools"]) == 1  # Only graph_query_tool should be included


# ---- Tests for task creation ----

def test_create_tasks_fraud_investigation(mock_task):
    """Test creating tasks for fraud investigation crew."""
    # Create factory
    factory = CrewFactory()
    
    # Create mock agents
    agents = {
        "nlq_translator": MagicMock(),
        "graph_analyst": MagicMock(),
        "fraud_pattern_hunter": MagicMock(),
        "sandbox_coder": MagicMock(),
        "compliance_checker": MagicMock(),
        "report_writer": MagicMock()
    }
    
    # Create tasks
    tasks = factory.create_tasks("fraud_investigation", agents)
    
    # Verify tasks were created
    assert len(tasks) == 6
    
    # Verify Task constructor was called for each task
    assert mock_task.call_count == 6
    
    # Verify task agents
    task_agents = [call.kwargs["agent"] for call in mock_task.call_args_list]
    assert agents["nlq_translator"] in task_agents
    assert agents["graph_analyst"] in task_agents
    assert agents["fraud_pattern_hunter"] in task_agents
    assert agents["sandbox_coder"] in task_agents
    assert agents["compliance_checker"] in task_agents
    assert agents["report_writer"] in task_agents


def test_create_tasks_alert_enrichment(mock_task):
    """Test creating tasks for alert enrichment crew."""
    # Create factory
    factory = CrewFactory()
    
    # Create mock agents
    agents = {
        "nlq_translator": MagicMock(),
        "graph_analyst": MagicMock(),
        "fraud_pattern_hunter": MagicMock(),
        "compliance_checker": MagicMock(),
        "report_writer": MagicMock()
    }
    
    # Create tasks
    tasks = factory.create_tasks("alert_enrichment", agents)
    
    # Verify tasks were created
    assert len(tasks) == 5
    
    # Verify Task constructor was called for each task
    assert mock_task.call_count == 5


def test_create_tasks_red_blue_simulation(mock_task):
    """Test creating tasks for red-blue team simulation crew."""
    # Create factory
    factory = CrewFactory()
    
    # Create mock agents
    agents = {
        "red_team_adversary": MagicMock(),
        "graph_analyst": MagicMock(),
        "fraud_pattern_hunter": MagicMock(),
        "report_writer": MagicMock()
    }
    
    # Create tasks
    tasks = factory.create_tasks("red_blue_simulation", agents)
    
    # Verify tasks were created
    assert len(tasks) == 5
    
    # Verify Task constructor was called for each task
    assert mock_task.call_count == 5


def test_create_tasks_crypto_investigation(mock_task):
    """Test creating tasks for crypto investigation crew."""
    # Create factory
    factory = CrewFactory()
    
    # Create mock agents
    agents = {
        "crypto_data_collector": MagicMock(),
        "blockchain_detective": MagicMock(),
        "defi_analyst": MagicMock(),
        "whale_tracker": MagicMock(),
        "protocol_investigator": MagicMock(),
        "report_writer": MagicMock()
    }
    
    # Create tasks
    tasks = factory.create_tasks("crypto_investigation", agents)
    
    # Verify tasks were created
    assert len(tasks) == 6
    
    # Verify Task constructor was called for each task
    assert mock_task.call_count == 6


def test_create_tasks_unknown_crew(mock_task):
    """Test creating tasks for an unknown crew type."""
    # Create factory
    factory = CrewFactory()
    
    # Create mock agents
    agents = {"test_agent": MagicMock()}
    
    # Create tasks
    tasks = factory.create_tasks("unknown_crew", agents)
    
    # Verify no tasks were created
    assert len(tasks) == 0
    
    # Verify Task constructor was not called
    mock_task.assert_not_called()


# ---- Tests for crew creation ----

@pytest.mark.asyncio
async def test_create_crew_new(mock_crew, mock_load_crew_config, mock_agent, mock_task):
    """Test creating a new crew."""
    # Create factory
    factory = CrewFactory()
    
    # Mock create_agent to return a mock agent
    factory.create_agent = MagicMock(return_value=MagicMock())
    
    # Mock create_tasks to return a list of mock tasks
    factory.create_tasks = MagicMock(return_value=[MagicMock(), MagicMock()])
    
    # Create crew
    crew = await factory.create_crew("test_crew")
    
    # Verify crew was created
    assert crew is not None
    
    # Verify crew was cached
    assert "test_crew" in factory.crews_cache
    assert factory.crews_cache["test_crew"] == crew
    
    # Verify Crew constructor was called with correct parameters
    mock_crew.assert_called_once()
    args, kwargs = mock_crew.call_args
    assert len(kwargs["agents"]) == 2  # Two agents from config
    assert len(kwargs["tasks"]) == 2  # Two tasks from mock
    assert kwargs["process"] == Process.sequential
    assert kwargs["verbose"] is True
    assert kwargs["max_rpm"] == 10
    assert kwargs["memory"] is True
    assert kwargs["cache"] is True


@pytest.mark.asyncio
async def test_create_crew_cached(mock_crew, mock_load_crew_config):
    """Test creating a crew that's already cached."""
    # Create factory
    factory = CrewFactory()
    
    # Create a mock cached crew
    mock_cached_crew = MagicMock()
    factory.crews_cache["test_crew"] = mock_cached_crew
    
    # Create crew
    crew = await factory.create_crew("test_crew")
    
    # Verify cached crew was returned
    assert crew == mock_cached_crew
    
    # Verify Crew constructor was not called
    mock_crew.assert_not_called()


@pytest.mark.asyncio
async def test_create_crew_hierarchical(mock_crew, mock_load_crew_config, mock_agent, mock_task):
    """Test creating a hierarchical crew."""
    # Create factory
    factory = CrewFactory()
    
    # Mock create_agent to return a mock agent
    factory.create_agent = MagicMock(return_value=MagicMock())
    
    # Mock create_tasks to return a list of mock tasks
    factory.create_tasks = MagicMock(return_value=[MagicMock(), MagicMock()])
    
    # Modify crew config to use hierarchical process
    mock_load_crew_config.return_value.process_type = "hierarchical"
    
    # Create crew
    crew = await factory.create_crew("test_crew")
    
    # Verify crew was created
    assert crew is not None
    
    # Verify Crew constructor was called with hierarchical process
    mock_crew.assert_called_once()
    args, kwargs = mock_crew.call_args
    assert kwargs["process"] == Process.hierarchical


@pytest.mark.asyncio
async def test_load_crew(mock_crew, mock_load_crew_config):
    """Test loading a crew by name."""
    # Mock CrewFactory.connect
    with patch.object(CrewFactory, "connect", AsyncMock()) as mock_connect:
        # Mock CrewFactory.create_crew
        with patch.object(CrewFactory, "create_crew", AsyncMock()) as mock_create_crew:
            # Set up mock crew
            mock_test_crew = MagicMock()
            mock_create_crew.return_value = mock_test_crew
            
            # Load crew
            crew = await CrewFactory.load("test_crew")
            
            # Verify connect was called
            mock_connect.assert_called_once()
            
            # Verify create_crew was called
            mock_create_crew.assert_called_once_with("test_crew")
            
            # Verify crew was returned
            assert crew == mock_test_crew


# ---- Tests for running crews ----

@pytest.mark.asyncio
async def test_run_crew_success(mock_crew, mock_load_crew_config):
    """Test successfully running a crew."""
    # Create factory
    factory = CrewFactory()
    
    # Mock create_crew to return a mock crew
    mock_test_crew = MagicMock()
    mock_test_crew.kickoff.return_value = TaskOutput(
        raw_output="Test crew result",
        agent_id="test_agent_id",
        task_id="test_task_id"
    )
    factory.create_crew = AsyncMock(return_value=mock_test_crew)
    
    # Mock close method
    factory.close = AsyncMock()
    
    # Run crew
    result = await factory.run_crew("test_crew", {"input_key": "input_value"})
    
    # Verify crew was created
    factory.create_crew.assert_called_once_with("test_crew")
    
    # Verify crew was run
    mock_test_crew.kickoff.assert_called_once_with(inputs={"input_key": "input_value"})
    
    # Verify connections were closed
    factory.close.assert_called_once()
    
    # Verify result
    assert result["success"] is True
    assert result["result"] == "Test crew result"
    assert result["task_id"] == "test_task_id"
    assert result["agent_id"] == "test_agent_id"


@pytest.mark.asyncio
async def test_run_crew_string_result(mock_crew, mock_load_crew_config):
    """Test running a crew that returns a string result."""
    # Create factory
    factory = CrewFactory()
    
    # Mock create_crew to return a mock crew
    mock_test_crew = MagicMock()
    mock_test_crew.kickoff.return_value = "Simple string result"
    factory.create_crew = AsyncMock(return_value=mock_test_crew)
    
    # Mock close method
    factory.close = AsyncMock()
    
    # Run crew
    result = await factory.run_crew("test_crew")
    
    # Verify result
    assert result["success"] is True
    assert result["result"] == "Simple string result"
    assert "task_id" not in result
    assert "agent_id" not in result


@pytest.mark.asyncio
async def test_run_crew_error(mock_crew, mock_load_crew_config):
    """Test error handling when running a crew."""
    # Create factory
    factory = CrewFactory()
    
    # Mock create_crew to raise exception
    factory.create_crew = AsyncMock(side_effect=Exception("Crew creation error"))
    
    # Mock close method
    factory.close = AsyncMock()
    
    # Run crew
    result = await factory.run_crew("test_crew")
    
    # Verify error result
    assert result["success"] is False
    assert "error" in result
    assert "Crew creation error" in result["error"]
    
    # Verify connections were closed
    factory.close.assert_called_once()


@pytest.mark.asyncio
async def test_run_crew_execution_error(mock_crew, mock_load_crew_config):
    """Test error handling when crew execution fails."""
    # Create factory
    factory = CrewFactory()
    
    # Mock create_crew to return a mock crew
    mock_test_crew = MagicMock()
    mock_test_crew.kickoff.side_effect = Exception("Crew execution error")
    factory.create_crew = AsyncMock(return_value=mock_test_crew)
    
    # Mock close method
    factory.close = AsyncMock()
    
    # Run crew
    result = await factory.run_crew("test_crew")
    
    # Verify error result
    assert result["success"] is False
    assert "error" in result
    assert "Crew execution error" in result["error"]


# ---- Tests for utility methods ----

def test_get_available_crews():
    """Test getting available crew names."""
    # Mock DEFAULT_CREW_CONFIGS
    with patch("backend.agents.factory.DEFAULT_CREW_CONFIGS", {
        "fraud_investigation": {},
        "alert_enrichment": {},
        "crypto_investigation": {}
    }):
        # Get available crews
        crews = CrewFactory.get_available_crews()
        
        # Verify crews
        assert len(crews) == 3
        assert "fraud_investigation" in crews
        assert "alert_enrichment" in crews
        assert "crypto_investigation" in crews
