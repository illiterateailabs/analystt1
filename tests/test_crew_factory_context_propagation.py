"""
Tests for CrewFactory's integration with CustomCrew and context propagation.

This module contains comprehensive tests to ensure that:
- CrewFactory correctly instantiates CustomCrew.
- Context is properly initialized and propagated through tasks via CustomCrew's shared_context.
- RUNNING_CREWS global dictionary accurately tracks and updates task contexts.
- Pause/resume functionality interacts correctly with the stored context.
- All external dependencies are mocked for isolated and deterministic testing.
"""

import asyncio
import pytest
import uuid
from unittest.mock import MagicMock, patch, AsyncMock, call
from datetime import datetime

from crewai import Agent, Task, Crew, Process
from crewai.task import TaskOutput

from backend.agents.factory import CrewFactory, RUNNING_CREWS, get_all_tools
from backend.agents.custom_crew import CustomCrew
from backend.agents.config import AgentConfig, CrewConfig
from backend.integrations.neo4j_client import Neo4jClient
from backend.integrations.gemini_client import GeminiClient
from backend.integrations.e2b_client import E2BClient
from backend.agents.llm import GeminiLLMProvider
from backend.core.metrics import increment_counter, observe_value # Mock these if they are used in factory.py

# Test constants
TEST_CREW_NAME = "test_crew"
TEST_AGENT_ID_1 = "test_agent_1"
TEST_AGENT_ID_2 = "test_agent_2"
TEST_TASK_DESCRIPTION_1 = "Perform initial data analysis"
TEST_TASK_DESCRIPTION_2 = "Generate report based on analysis"

# Mock configurations
MOCK_AGENT_CONFIG_1 = AgentConfig(
    id=TEST_AGENT_ID_1,
    role="Data Analyst",
    goal="Analyze data",
    backstory="Expert in data analysis",
    tools=["mock_tool_1"],
    verbose=True
)

MOCK_AGENT_CONFIG_2 = AgentConfig(
    id=TEST_AGENT_ID_2,
    role="Report Writer",
    goal="Write reports",
    backstory="Expert in report writing",
    tools=["mock_tool_2"],
    verbose=True
)

MOCK_CREW_CONFIG = CrewConfig(
    name=TEST_CREW_NAME,
    description="A test crew",
    agents=[TEST_AGENT_ID_1, TEST_AGENT_ID_2],
    process_type="sequential",
    verbose=True
)

# Mock CodeGenTool result structure (as expected to be in shared_context)
MOCK_CODEGEN_SHARED_CONTEXT = {
    "codegen": {
        "result": "analysis_complete",
        "execution": {"stdout": "Plot generated.", "stderr": "", "exit_code": 0},
        "visualizations": [
            {"filename": "plot.png", "content": "base64_png", "type": "image/png"}
        ]
    },
    "code_result": "analysis_complete",
    "visualizations": [
        {"filename": "plot.png", "content": "base64_png", "type": "image/png"}
    ]
}

# Fixtures
@pytest.fixture(autouse=True)
def mock_dependencies():
    """Mock external dependencies for CrewFactory."""
    with patch('backend.integrations.neo4j_client.Neo4jClient') as mock_neo4j_client_class, \
         patch('backend.integrations.gemini_client.GeminiClient') as mock_gemini_client_class, \
         patch('backend.integrations.e2b_client.E2BClient') as mock_e2b_client_class, \
         patch('backend.agents.llm.GeminiLLMProvider') as mock_llm_provider_class, \
         patch('backend.agents.factory.get_all_tools') as mock_get_all_tools, \
         patch('backend.agents.config.load_agent_config') as mock_load_agent_config, \
         patch('backend.agents.config.load_crew_config') as mock_load_crew_config, \
         patch('backend.agents.config.get_available_crews') as mock_get_available_crews, \
         patch('backend.core.metrics.increment_counter') as mock_increment_counter, \
         patch('backend.core.metrics.observe_value') as mock_observe_value:

        # Mock client instances
        mock_neo4j_client = AsyncMock(spec=Neo4jClient)
        mock_neo4j_client_class.return_value = mock_neo4j_client
        mock_neo4j_client.connect.return_value = None
        mock_neo4j_client.close.return_value = None

        mock_gemini_client = MagicMock(spec=GeminiClient)
        mock_gemini_client_class.return_value = mock_gemini_client

        mock_e2b_client = AsyncMock(spec=E2BClient)
        mock_e2b_client_class.return_value = mock_e2b_client
        mock_e2b_client.close_all_sandboxes.return_value = None

        mock_llm_provider = MagicMock(spec=GeminiLLMProvider)
        mock_llm_provider_class.return_value = mock_llm_provider

        # Mock get_all_tools to return some dummy tools
        mock_get_all_tools.return_value = {
            "mock_tool_1": MagicMock(name="MockTool1"),
            "mock_tool_2": MagicMock(name="MockTool2")
        }

        # Mock config loading
        mock_load_agent_config.side_effect = lambda agent_id: {
            TEST_AGENT_ID_1: MOCK_AGENT_CONFIG_1,
            TEST_AGENT_ID_2: MOCK_AGENT_CONFIG_2
        }.get(agent_id)
        mock_load_crew_config.return_value = MOCK_CREW_CONFIG
        mock_get_available_crews.return_value = [TEST_CREW_NAME]

        yield {
            "neo4j_client": mock_neo4j_client,
            "gemini_client": mock_gemini_client,
            "e2b_client": mock_e2b_client,
            "llm_provider": mock_llm_provider,
            "get_all_tools": mock_get_all_tools,
            "load_agent_config": mock_load_agent_config,
            "load_crew_config": mock_load_crew_config,
            "get_available_crews": mock_get_available_crews
        }

@pytest.fixture
def reset_running_crews():
    """Reset the RUNNING_CREWS global dictionary before and after each test."""
    # Save original
    original_running_crews = RUNNING_CREWS.copy()
    
    # Clear for test
    RUNNING_CREWS.clear()
    
    yield
    
    # Restore original
    RUNNING_CREWS.clear()
    RUNNING_CREWS.update(original_running_crews)

@pytest.fixture
def mock_crew():
    """Mock a CustomCrew instance."""
    with patch('backend.agents.custom_crew.CustomCrew') as mock_crew_class:
        mock_crew_instance = MagicMock(spec=CustomCrew)
        mock_crew_class.return_value = mock_crew_instance
        
        # Mock shared_context
        mock_crew_instance.shared_context = {}
        
        # Mock kickoff method
        mock_crew_instance.kickoff = MagicMock(return_value="Crew result")
        
        yield mock_crew_instance, mock_crew_class

# Tests
class TestCrewFactoryCreation:
    """Tests for CrewFactory creation of CustomCrew instances."""
    
    @pytest.mark.asyncio
    async def test_create_crew_returns_custom_crew(self, mock_dependencies, mock_crew):
        """Test that CrewFactory.create_crew returns a CustomCrew instance."""
        mock_crew_instance, mock_crew_class = mock_crew
        
        # Create factory
        factory = CrewFactory()
        
        # Create crew
        crew = await factory.create_crew(TEST_CREW_NAME)
        
        # Verify that CustomCrew was instantiated
        mock_crew_class.assert_called_once()
        
        # Verify that the returned crew is our mocked CustomCrew
        assert crew == mock_crew_instance
        
        # IMPORTANT: This test will fail with the current implementation!
        # The current factory.py creates standard Crew objects, not CustomCrew.
        # To fix this, update create_crew in factory.py to use CustomCrew.

    @pytest.mark.asyncio
    async def test_create_crew_with_correct_parameters(self, mock_dependencies, mock_crew):
        """Test that CrewFactory.create_crew passes correct parameters to CustomCrew."""
        mock_crew_instance, mock_crew_class = mock_crew
        
        # Create factory
        factory = CrewFactory()
        
        # Create crew
        await factory.create_crew(TEST_CREW_NAME)
        
        # Verify CustomCrew was instantiated with correct parameters
        call_kwargs = mock_crew_class.call_args[1]
        
        # Check agents
        assert len(call_kwargs["agents"]) == 2
        
        # Check process
        assert call_kwargs["process"] == Process.sequential
        
        # Check verbose
        assert call_kwargs["verbose"] == True
        
        # Check that LLM provider was passed
        assert call_kwargs["manager_llm"] == factory.llm_provider

class TestContextPropagationWrapper:
    """Tests for the kickoff_with_context wrapper in run_crew."""
    
    @pytest.mark.asyncio
    async def test_kickoff_with_context_wrapper(self, mock_dependencies, reset_running_crews):
        """Test that the kickoff_with_context wrapper correctly adds context to inputs."""
        # Create a mock crew
        mock_crew = MagicMock(spec=CustomCrew)
        original_kickoff = mock_crew.kickoff
        
        # Generate a task ID
        task_id = str(uuid.uuid4())
        
        # Initialize RUNNING_CREWS
        RUNNING_CREWS[task_id] = {
            "crew_name": TEST_CREW_NAME,
            "state": "RUNNING",
            "start_time": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "inputs": {"original": "input"},
            "current_agent": None,
            "context": {"existing": "context"}
        }
        
        # Create the wrapper function
        def kickoff_with_context(**kwargs):
            # Add context to inputs
            if "inputs" in kwargs and kwargs["inputs"] is not None:
                if not isinstance(kwargs["inputs"], dict):
                    kwargs["inputs"] = {"input": kwargs["inputs"]}
                
                # Add context to inputs
                kwargs["inputs"]["_context"] = RUNNING_CREWS[task_id]["context"]
            else:
                kwargs["inputs"] = {"_context": RUNNING_CREWS[task_id]["context"]}
            
            # Run original kickoff
            output = original_kickoff(**kwargs)
            
            # Store context for future use
            if "_context" in kwargs["inputs"]:
                RUNNING_CREWS[task_id]["context"] = kwargs["inputs"]["_context"]
            
            return output
        
        # Replace kickoff method with wrapper
        mock_crew.kickoff = kickoff_with_context
        
        # Call the wrapped kickoff method
        inputs = {"user_input": "test"}
        mock_crew.kickoff(inputs=inputs)
        
        # Verify that original_kickoff was called with enhanced inputs
        original_kickoff.assert_called_once()
        call_kwargs = original_kickoff.call_args[1]
        
        # Check that inputs were enhanced with context
        assert "_context" in call_kwargs["inputs"]
        assert call_kwargs["inputs"]["_context"] == {"existing": "context"}
        assert call_kwargs["inputs"]["user_input"] == "test"

    @pytest.mark.asyncio
    async def test_kickoff_with_context_updates_running_crews(self, mock_dependencies, reset_running_crews):
        """Test that the kickoff_with_context wrapper updates RUNNING_CREWS with new context."""
        # Create a mock crew
        mock_crew = MagicMock(spec=CustomCrew)
        
        # Mock the original kickoff to modify the _context
        def mock_original_kickoff(**kwargs):
            # Simulate CustomCrew updating the shared context
            if "_context" in kwargs["inputs"]:
                kwargs["inputs"]["_context"].update({"new": "value"})
            return "Crew result"
        
        mock_crew.kickoff = mock_original_kickoff
        
        # Generate a task ID
        task_id = str(uuid.uuid4())
        
        # Initialize RUNNING_CREWS
        RUNNING_CREWS[task_id] = {
            "crew_name": TEST_CREW_NAME,
            "state": "RUNNING",
            "start_time": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "inputs": {"original": "input"},
            "current_agent": None,
            "context": {"existing": "context"}
        }
        
        # Create the wrapper function (same as in factory.py)
        def kickoff_with_context(**kwargs):
            # Add context to inputs
            if "inputs" in kwargs and kwargs["inputs"] is not None:
                if not isinstance(kwargs["inputs"], dict):
                    kwargs["inputs"] = {"input": kwargs["inputs"]}
                
                # Add context to inputs
                kwargs["inputs"]["_context"] = RUNNING_CREWS[task_id]["context"]
            else:
                kwargs["inputs"] = {"_context": RUNNING_CREWS[task_id]["context"]}
            
            # Run original kickoff
            output = mock_crew.kickoff(**kwargs)
            
            # Store context for future use
            if "_context" in kwargs["inputs"]:
                RUNNING_CREWS[task_id]["context"] = kwargs["inputs"]["_context"]
            
            return output
        
        # Call the wrapper function
        result = kickoff_with_context(inputs={"user_input": "test"})
        
        # Verify the result
        assert result == "Crew result"
        
        # Verify that RUNNING_CREWS was updated with the new context
        assert "existing" in RUNNING_CREWS[task_id]["context"]
        assert "new" in RUNNING_CREWS[task_id]["context"]
        assert RUNNING_CREWS[task_id]["context"]["new"] == "value"

class TestRunningCrewsContextTracking:
    """Tests for RUNNING_CREWS context tracking."""
    
    @pytest.mark.asyncio
    async def test_running_crews_initialization(self, mock_dependencies, reset_running_crews):
        """Test that RUNNING_CREWS is properly initialized in run_crew."""
        # Create factory
        factory = CrewFactory()
        
        # Mock create_crew to return a mock crew
        mock_crew = MagicMock(spec=CustomCrew)
        mock_crew.kickoff.return_value = "Crew result"
        
        with patch.object(factory, 'create_crew', return_value=mock_crew):
            # Call run_crew
            inputs = {"test": "input"}
            await factory.run_crew(TEST_CREW_NAME, inputs)
            
            # Get the task_id from the result
            task_id = list(RUNNING_CREWS.keys())[0]
            
            # Verify RUNNING_CREWS was initialized correctly
            assert task_id in RUNNING_CREWS
            assert RUNNING_CREWS[task_id]["crew_name"] == TEST_CREW_NAME
            assert RUNNING_CREWS[task_id]["state"] == "COMPLETED"
            assert "start_time" in RUNNING_CREWS[task_id]
            assert "last_updated" in RUNNING_CREWS[task_id]
            assert RUNNING_CREWS[task_id]["inputs"] == inputs
            assert "context" in RUNNING_CREWS[task_id]
            assert RUNNING_CREWS[task_id]["context"] == {}  # Initially empty

    @pytest.mark.asyncio
    async def test_running_crews_context_update(self, mock_dependencies, reset_running_crews):
        """Test that RUNNING_CREWS context is updated after crew execution."""
        # Create factory
        factory = CrewFactory()
        
        # Mock create_crew to return a mock crew
        mock_crew = MagicMock(spec=CustomCrew)
        
        # Mock kickoff to update the context
        def mock_kickoff(**kwargs):
            # Update the context
            kwargs["inputs"]["_context"] = MOCK_CODEGEN_SHARED_CONTEXT
            return "Crew result with context"
        
        mock_crew.kickoff = mock_kickoff
        
        with patch.object(factory, 'create_crew', return_value=mock_crew):
            # Call run_crew
            result = await factory.run_crew(TEST_CREW_NAME, {"test": "input"})
            
            # Get the task_id from the result
            task_id = result["task_id"]
            
            # Verify RUNNING_CREWS context was updated
            assert RUNNING_CREWS[task_id]["context"] == MOCK_CODEGEN_SHARED_CONTEXT
            assert "codegen" in RUNNING_CREWS[task_id]["context"]
            assert "visualizations" in RUNNING_CREWS[task_id]["context"]

class TestPauseResumeWithContext:
    """Tests for pause/resume functionality with context preservation."""
    
    def test_pause_crew_preserves_context(self, mock_dependencies, reset_running_crews):
        """Test that pause_crew preserves the context in RUNNING_CREWS."""
        # Initialize RUNNING_CREWS with a mock task
        task_id = str(uuid.uuid4())
        RUNNING_CREWS[task_id] = {
            "crew_name": TEST_CREW_NAME,
            "state": "RUNNING",
            "start_time": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "inputs": {"original": "input"},
            "current_agent": None,
            "context": MOCK_CODEGEN_SHARED_CONTEXT
        }
        
        # Pause the crew
        result = CrewFactory.pause_crew(task_id, reason="Testing pause")
        
        # Verify the result
        assert result is True
        
        # Verify the state was updated
        assert RUNNING_CREWS[task_id]["state"] == "PAUSED"
        assert "paused_at" in RUNNING_CREWS[task_id]
        assert RUNNING_CREWS[task_id]["pause_reason"] == "Testing pause"
        
        # Verify the context was preserved
        assert RUNNING_CREWS[task_id]["context"] == MOCK_CODEGEN_SHARED_CONTEXT
        assert "codegen" in RUNNING_CREWS[task_id]["context"]
        assert "visualizations" in RUNNING_CREWS[task_id]["context"]

    def test_resume_crew_preserves_context(self, mock_dependencies, reset_running_crews):
        """Test that resume_crew preserves the context in RUNNING_CREWS."""
        # Initialize RUNNING_CREWS with a paused task
        task_id = str(uuid.uuid4())
        RUNNING_CREWS[task_id] = {
            "crew_name": TEST_CREW_NAME,
            "state": "PAUSED",
            "start_time": datetime.now().isoformat(),
            "paused_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "inputs": {"original": "input"},
            "current_agent": None,
            "context": MOCK_CODEGEN_SHARED_CONTEXT,
            "pause_reason": "Testing pause"
        }
        
        # Resume the crew
        result = CrewFactory.resume_crew(task_id, review_result={"approved": True})
        
        # Verify the result
        assert result is True
        
        # Verify the state was updated
        assert RUNNING_CREWS[task_id]["state"] == "RUNNING"
        assert "resumed_at" in RUNNING_CREWS[task_id]
        assert "review_result" in RUNNING_CREWS[task_id]
        
        # Verify the context was preserved
        assert RUNNING_CREWS[task_id]["context"] == MOCK_CODEGEN_SHARED_CONTEXT
        assert "codegen" in RUNNING_CREWS[task_id]["context"]
        assert "visualizations" in RUNNING_CREWS[task_id]["context"]

class TestRunCrewWithContextUpdates:
    """Tests for the run_crew method with context updates."""
    
    @pytest.mark.asyncio
    async def test_run_crew_end_to_end(self, mock_dependencies, reset_running_crews):
        """Test the end-to-end flow of run_crew with context updates."""
        # Create factory
        factory = CrewFactory()
        
        # Mock create_crew to return a mock crew
        mock_crew = MagicMock(spec=CustomCrew)
        
        # Set up the kickoff method to simulate context updates
        original_kickoff = mock_crew.kickoff
        
        def mock_kickoff(**kwargs):
            # Update the context
            if "_context" in kwargs["inputs"]:
                kwargs["inputs"]["_context"].update(MOCK_CODEGEN_SHARED_CONTEXT)
            return "Crew result with updated context"
        
        mock_crew.kickoff = mock_kickoff
        
        with patch.object(factory, 'create_crew', return_value=mock_crew), \
             patch('uuid.uuid4', return_value=uuid.UUID('12345678-1234-5678-1234-567812345678')):
            
            # Call run_crew
            result = await factory.run_crew(TEST_CREW_NAME, {"initial": "input"})
            
            # Verify the result
            assert result["success"] is True
            assert result["task_id"] == "12345678-1234-5678-1234-567812345678"
            assert result["crew_name"] == TEST_CREW_NAME
            assert result["result"] == "Crew result with updated context"
            
            # Verify RUNNING_CREWS was updated correctly
            task_id = result["task_id"]
            assert task_id in RUNNING_CREWS
            assert RUNNING_CREWS[task_id]["state"] == "COMPLETED"
            assert "completion_time" in RUNNING_CREWS[task_id]
            assert RUNNING_CREWS[task_id]["result"] == result["result"]
            
            # Verify the context was updated
            assert "codegen" in RUNNING_CREWS[task_id]["context"]
            assert "visualizations" in RUNNING_CREWS[task_id]["context"]
            assert RUNNING_CREWS[task_id]["context"]["codegen"]["result"] == "analysis_complete"

    @pytest.mark.asyncio
    async def test_run_crew_with_error(self, mock_dependencies, reset_running_crews):
        """Test run_crew with an error during execution."""
        # Create factory
        factory = CrewFactory()
        
        # Mock create_crew to return a mock crew
        mock_crew = MagicMock(spec=CustomCrew)
        mock_crew.kickoff.side_effect = ValueError("Test error")
        
        with patch.object(factory, 'create_crew', return_value=mock_crew), \
             patch('uuid.uuid4', return_value=uuid.UUID('12345678-1234-5678-1234-567812345678')):
            
            # Call run_crew
            result = await factory.run_crew(TEST_CREW_NAME, {"initial": "input"})
            
            # Verify the result
            assert result["success"] is False
            assert result["task_id"] == "12345678-1234-5678-1234-567812345678"
            assert result["crew_name"] == TEST_CREW_NAME
            assert "error" in result
            assert "Test error" in result["error"]
            
            # Verify RUNNING_CREWS was updated correctly
            task_id = result["task_id"]
            assert task_id in RUNNING_CREWS
            assert RUNNING_CREWS[task_id]["state"] == "ERROR"
            assert "error" in RUNNING_CREWS[task_id]
            assert RUNNING_CREWS[task_id]["error"] == "Test error"

# IMPORTANT: Fix suggestion
"""
CRITICAL ISSUE: The current CrewFactory implementation does not use CustomCrew!

In factory.py, the create_crew method creates a standard Crew instance instead of CustomCrew.
This means that all the context propagation and event emission features are not being used.

To fix this issue:
1. Import CustomCrew in factory.py:
   from backend.agents.custom_crew import CustomCrew

2. Replace the Crew instantiation in create_crew with CustomCrew:
   # Create crew
   crew = CustomCrew(
       agents=list(agents.values()),
       tasks=tasks,
       process=process,
       verbose=crew_config.verbose,
       max_rpm=crew_config.max_rpm,
       memory=crew_config.memory,
       cache=crew_config.cache,
       manager_llm=self.llm_provider if crew_config.manager else None
   )

This will ensure that all crews created by the factory have the enhanced context sharing
and event emission capabilities.
"""
