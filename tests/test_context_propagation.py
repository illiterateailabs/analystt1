"""
Tests for CustomCrew context propagation and event emission functionality.

This module contains comprehensive tests for the CustomCrew class, ensuring:
- Shared context initialization and updates.
- Context propagation between tasks.
- CodeGenTool result extraction and sharing.
- Event emissions for various lifecycle stages (crew, agent, tool).
- Error handling and fallback behavior.
- Task execution with enhanced inputs.
- Visualization sharing between tasks.
- Mocking of all external dependencies to ensure isolated unit testing.
"""

import asyncio
import json
import pytest
import uuid
from unittest.mock import AsyncMock, MagicMock, patch, call

from crewai import Agent, Task
from crewai.task import TaskOutput

from backend.agents.custom_crew import CustomCrew
from backend.core.events import EventType, emit_event, initialize_events, shutdown_events

# Test constants
TEST_CREW_ID = str(uuid.uuid4())
TEST_TASK_ID_1 = str(uuid.uuid4())
TEST_TASK_ID_2 = str(uuid.uuid4())
TEST_AGENT_NAME_1 = "test_agent_1"
TEST_AGENT_NAME_2 = "test_agent_2"

# Mock CodeGenTool result
MOCK_CODEGEN_RESULT_JSON = """
```json
{
    "result": "analysis_complete",
    "execution": {
        "stdout": "Plot generated successfully.",
        "stderr": "",
        "exit_code": 0
    },
    "visualizations": [
        {"filename": "plot.png", "content": "base64_encoded_png", "type": "image/png"},
        {"filename": "report.html", "content": "<html>...</html>", "type": "text/html"}
    ]
}
```
"""

MOCK_CODEGEN_RESULT_TOOL_OUTPUT = {
    "result": "analysis_complete_from_tool",
    "execution": {
        "stdout": "Tool executed successfully.",
        "stderr": "",
        "exit_code": 0
    },
    "visualizations": [
        {"filename": "tool_plot.png", "content": "base64_encoded_tool_png", "type": "image/png"}
    ]
}

# Fixtures
@pytest.fixture
async def initialized_events():
    """Initialize and clean up the global event system."""
    await initialize_events()
    yield
    await shutdown_events()

@pytest.fixture
def mock_emit_event():
    """Mock the global emit_event function to capture calls."""
    with patch("backend.agents.custom_crew.emit_event", new_callable=AsyncMock) as mock_emit:
        yield mock_emit

@pytest.fixture
def mock_crewai_internals():
    """Mock CrewAI internal components for isolated testing of CustomCrew."""
    with patch("crewai.Crew.kickoff", new_callable=MagicMock) as mock_super_kickoff, \
         patch("crewai.Crew._process_task", new_callable=AsyncMock) as mock_super_process_task:
        yield mock_super_kickoff, mock_super_process_task

@pytest.fixture
def mock_agent_with_codegen():
    """Create a mock Agent with a CodeGenTool."""
    mock_codegen_tool = MagicMock()
    mock_codegen_tool.__class__.__name__ = "CodeGenTool"
    mock_codegen_tool._last_result = None # To simulate no direct last result initially

    mock_agent = MagicMock(spec=Agent)
    mock_agent.name = TEST_AGENT_NAME_1
    mock_agent.tools = [mock_codegen_tool]
    return mock_agent, mock_codegen_tool

@pytest.fixture
def mock_agent_without_codegen():
    """Create a mock Agent without a CodeGenTool."""
    mock_agent = MagicMock(spec=Agent)
    mock_agent.name = TEST_AGENT_NAME_2
    mock_agent.tools = []
    return mock_agent

@pytest.fixture
def mock_task_output():
    """Create a mock TaskOutput object."""
    mock_output = MagicMock(spec=TaskOutput)
    mock_output.raw_output = "Task completed successfully."
    mock_output.agent_id = "some_agent"
    return mock_output

# Helper function to create a CustomCrew instance with mock tasks
def create_mock_crew(tasks: list[Task], name: str = "test_crew") -> CustomCrew:
    """Helper to create a CustomCrew instance with given tasks."""
    crew = CustomCrew(
        agents=[MagicMock(spec=Agent)], # Agents are mocked within tasks
        tasks=tasks,
        process="sequential",
        verbose=True,
        manager_llm=MagicMock()
    )
    crew.name = name
    crew.crew_id = TEST_CREW_ID # Assign a fixed ID for testing
    return crew

# Test initialization
class TestCustomCrewInitialization:
    """Tests for CustomCrew initialization."""
    
    def test_init(self):
        """Test CustomCrew initialization."""
        crew = CustomCrew(
            agents=[MagicMock(spec=Agent)],
            tasks=[MagicMock(spec=Task)],
            process="sequential",
            verbose=True
        )
        
        assert isinstance(crew.shared_context, dict)
        assert len(crew.shared_context) == 0
        assert isinstance(crew.crew_id, str)
        assert isinstance(crew.task_ids, dict)
        assert crew.total_tasks == 1
        assert crew.completed_tasks == 0

    def test_init_no_tasks(self):
        """Test CustomCrew initialization with no tasks."""
        crew = CustomCrew(
            agents=[MagicMock(spec=Agent)],
            tasks=[],
            process="sequential",
            verbose=True
        )
        
        assert isinstance(crew.shared_context, dict)
        assert len(crew.shared_context) == 0
        assert crew.total_tasks == 0
        assert crew.completed_tasks == 0

# Test kickoff method
class TestCustomCrewKickoff:
    """Tests for CustomCrew kickoff method."""
    
    def test_kickoff_with_inputs(self, mock_crewai_internals, mock_emit_event):
        """Test kickoff method with inputs."""
        mock_super_kickoff, _ = mock_crewai_internals
        mock_super_kickoff.return_value = "Kickoff result"
        
        # Create crew
        crew = CustomCrew(
            agents=[MagicMock(spec=Agent)],
            tasks=[MagicMock(spec=Task)],
            process="sequential",
            verbose=True
        )
        
        # Call kickoff
        inputs = {"key1": "value1", "key2": "value2"}
        result = crew.kickoff(inputs)
        
        # Verify shared_context is updated
        assert crew.shared_context == inputs
        
        # Verify super().kickoff was called
        mock_super_kickoff.assert_called_once_with(inputs)
        
        # Verify result
        assert result == "Kickoff result"
        
        # Verify events were emitted
        assert mock_emit_event.call_count >= 2  # At least CREW_STARTED and CREW_COMPLETED
        
        # Check CREW_STARTED event
        crew_started_call = [call for call in mock_emit_event.call_args_list 
                            if call[0][0] == EventType.CREW_STARTED]
        assert len(crew_started_call) == 1
        assert crew_started_call[0][0][1]["crew_id"] == crew.crew_id
        assert crew_started_call[0][0][1]["progress"] == 0
        
        # Check CREW_COMPLETED event
        crew_completed_call = [call for call in mock_emit_event.call_args_list 
                               if call[0][0] == EventType.CREW_COMPLETED]
        assert len(crew_completed_call) == 1
        assert crew_completed_call[0][0][1]["crew_id"] == crew.crew_id
        assert crew_completed_call[0][0][1]["progress"] == 100
    
    def test_kickoff_without_inputs(self, mock_crewai_internals, mock_emit_event):
        """Test kickoff method without inputs."""
        mock_super_kickoff, _ = mock_crewai_internals
        mock_super_kickoff.return_value = "Kickoff result"
        
        # Create crew
        crew = CustomCrew(
            agents=[MagicMock(spec=Agent)],
            tasks=[MagicMock(spec=Task)],
            process="sequential",
            verbose=True
        )
        
        # Call kickoff
        result = crew.kickoff()
        
        # Verify shared_context is empty
        assert crew.shared_context == {}
        
        # Verify super().kickoff was called
        mock_super_kickoff.assert_called_once_with(None)
        
        # Verify result
        assert result == "Kickoff result"
        
        # Verify events were emitted
        assert mock_emit_event.call_count >= 2  # At least CREW_STARTED and CREW_COMPLETED
    
    def test_kickoff_with_exception(self, mock_crewai_internals, mock_emit_event):
        """Test kickoff method with exception."""
        mock_super_kickoff, _ = mock_crewai_internals
        mock_super_kickoff.side_effect = ValueError("Kickoff error")
        
        # Create crew
        crew = CustomCrew(
            agents=[MagicMock(spec=Agent)],
            tasks=[MagicMock(spec=Task)],
            process="sequential",
            verbose=True
        )
        
        # Call kickoff and expect exception
        with pytest.raises(ValueError, match="Kickoff error"):
            crew.kickoff({"key": "value"})
        
        # Verify shared_context is updated despite exception
        assert crew.shared_context == {"key": "value"}
        
        # Verify super().kickoff was called
        mock_super_kickoff.assert_called_once()
        
        # Verify events were emitted
        assert mock_emit_event.call_count >= 2  # At least CREW_STARTED and CREW_FAILED
        
        # Check CREW_STARTED event
        crew_started_call = [call for call in mock_emit_event.call_args_list 
                            if call[0][0] == EventType.CREW_STARTED]
        assert len(crew_started_call) == 1
        
        # Check CREW_FAILED event
        crew_failed_call = [call for call in mock_emit_event.call_args_list 
                           if call[0][0] == EventType.CREW_FAILED]
        assert len(crew_failed_call) == 1
        assert crew_failed_call[0][0][1]["crew_id"] == crew.crew_id
        assert "error" in crew_failed_call[0][0][1]["data"]
        assert "Kickoff error" in crew_failed_call[0][0][1]["message"]

# Test process_task method
class TestCustomCrewProcessTask:
    """Tests for CustomCrew _process_task method."""
    
    @pytest.mark.asyncio
    async def test_process_task_basic(self, mock_crewai_internals, mock_emit_event, mock_agent_without_codegen):
        """Test _process_task method with basic task."""
        _, mock_super_process_task = mock_crewai_internals
        mock_output = MagicMock(spec=TaskOutput)
        mock_output.raw_output = "Task result"
        mock_super_process_task.return_value = mock_output
        
        # Create task
        mock_task = MagicMock(spec=Task)
        mock_task.description = "Test task"
        mock_task.agent = mock_agent_without_codegen
        
        # Create crew
        crew = create_mock_crew([mock_task])
        
        # Set initial shared context
        crew.shared_context = {"initial": "context"}
        
        # Call _process_task
        inputs = {"input1": "value1"}
        result = await crew._process_task(mock_task, inputs)
        
        # Verify super()._process_task was called with enhanced inputs
        mock_super_process_task.assert_called_once()
        call_args = mock_super_process_task.call_args[0]
        assert call_args[0] == mock_task
        assert "input1" in call_args[1]
        assert "initial" in call_args[1]
        
        # Verify result
        assert result == mock_output
        
        # Verify task_id was generated
        assert mock_task.description in crew.task_ids
        
        # Verify completed_tasks was incremented
        assert crew.completed_tasks == 1
        
        # Verify events were emitted
        assert mock_emit_event.call_count >= 3  # AGENT_STARTED, AGENT_PROGRESS, AGENT_COMPLETED
        
        # Check AGENT_STARTED event
        agent_started_call = [call for call in mock_emit_event.call_args_list 
                             if call[0][0] == EventType.AGENT_STARTED]
        assert len(agent_started_call) == 1
        assert agent_started_call[0][0][1]["task_id"] == crew.task_ids[mock_task.description]
        assert agent_started_call[0][0][1]["crew_id"] == crew.crew_id
        assert agent_started_call[0][0][1]["agent_id"] == mock_agent_without_codegen.name
        
        # Check AGENT_COMPLETED event
        agent_completed_call = [call for call in mock_emit_event.call_args_list 
                               if call[0][0] == EventType.AGENT_COMPLETED]
        assert len(agent_completed_call) == 1
        assert agent_completed_call[0][0][1]["task_id"] == crew.task_ids[mock_task.description]
        assert agent_completed_call[0][0][1]["progress"] == 100  # Task is complete
    
    @pytest.mark.asyncio
    async def test_process_task_with_codegen_tool(self, mock_crewai_internals, mock_emit_event, mock_agent_with_codegen):
        """Test _process_task method with CodeGenTool."""
        _, mock_super_process_task = mock_crewai_internals
        mock_agent, mock_codegen_tool = mock_agent_with_codegen
        
        # Mock task output with CodeGenTool result
        mock_output = MagicMock(spec=TaskOutput)
        mock_output.raw_output = MOCK_CODEGEN_RESULT_JSON
        mock_super_process_task.return_value = mock_output
        
        # Create task
        mock_task = MagicMock(spec=Task)
        mock_task.description = "Test task with CodeGenTool"
        mock_task.agent = mock_agent
        
        # Create crew
        crew = create_mock_crew([mock_task])
        
        # Call _process_task
        result = await crew._process_task(mock_task, {})
        
        # Verify result
        assert result == mock_output
        
        # Verify shared_context was updated with CodeGenTool results
        assert "codegen" in crew.shared_context
        assert "result" in crew.shared_context["codegen"]
        assert crew.shared_context["codegen"]["result"] == "analysis_complete"
        
        # Verify code_result was extracted
        assert "code_result" in crew.shared_context
        assert crew.shared_context["code_result"] == "analysis_complete"
        
        # Verify visualizations were extracted
        assert "visualizations" in crew.shared_context
        assert len(crew.shared_context["visualizations"]) == 2
        assert crew.shared_context["visualizations"][0]["filename"] == "plot.png"
        
        # Verify TOOL_STARTED and TOOL_COMPLETED events were emitted
        tool_started_call = [call for call in mock_emit_event.call_args_list 
                            if call[0][0] == EventType.TOOL_STARTED]
        assert len(tool_started_call) == 1
        assert tool_started_call[0][0][1]["tool_id"] == "CodeGenTool"
        
        tool_completed_call = [call for call in mock_emit_event.call_args_list 
                              if call[0][0] == EventType.TOOL_COMPLETED]
        assert len(tool_completed_call) == 1
        assert tool_completed_call[0][0][1]["tool_id"] == "CodeGenTool"
        assert tool_completed_call[0][0][1]["data"]["has_visualizations"] == True
        assert tool_completed_call[0][0][1]["data"]["visualization_count"] == 2
    
    @pytest.mark.asyncio
    async def test_process_task_with_direct_tool_result(self, mock_crewai_internals, mock_emit_event, mock_agent_with_codegen):
        """Test _process_task method with direct tool result."""
        _, mock_super_process_task = mock_crewai_internals
        mock_agent, mock_codegen_tool = mock_agent_with_codegen
        
        # Set up direct tool result
        mock_codegen_tool._last_result = MOCK_CODEGEN_RESULT_TOOL_OUTPUT
        
        # Mock task output without CodeGenTool result in text
        mock_output = MagicMock(spec=TaskOutput)
        mock_output.raw_output = "Task completed but no JSON result in output."
        mock_super_process_task.return_value = mock_output
        
        # Create task
        mock_task = MagicMock(spec=Task)
        mock_task.description = "Test task with direct tool result"
        mock_task.agent = mock_agent
        
        # Create crew
        crew = create_mock_crew([mock_task])
        
        # Call _process_task
        result = await crew._process_task(mock_task, {})
        
        # Verify result
        assert result == mock_output
        
        # Verify shared_context was updated with direct tool result
        assert "codegen" in crew.shared_context
        assert "result" in crew.shared_context["codegen"]
        assert crew.shared_context["codegen"]["result"] == "analysis_complete_from_tool"
        
        # Verify visualizations were extracted
        assert "visualizations" in crew.shared_context
        assert len(crew.shared_context["visualizations"]) == 1
        assert crew.shared_context["visualizations"][0]["filename"] == "tool_plot.png"
    
    @pytest.mark.asyncio
    async def test_process_task_with_exception(self, mock_crewai_internals, mock_emit_event, mock_agent_without_codegen):
        """Test _process_task method with exception."""
        _, mock_super_process_task = mock_crewai_internals
        
        # First call raises exception, second call succeeds (fallback)
        mock_super_process_task.side_effect = [
            ValueError("Process task error"),
            MagicMock(spec=TaskOutput)
        ]
        
        # Create task
        mock_task = MagicMock(spec=Task)
        mock_task.description = "Test task with exception"
        mock_task.agent = mock_agent_without_codegen
        
        # Create crew
        crew = create_mock_crew([mock_task])
        
        # Call _process_task
        await crew._process_task(mock_task, {})
        
        # Verify super()._process_task was called twice (once for error, once for fallback)
        assert mock_super_process_task.call_count == 2
        
        # Verify AGENT_FAILED event was emitted
        agent_failed_call = [call for call in mock_emit_event.call_args_list 
                            if call[0][0] == EventType.AGENT_FAILED]
        assert len(agent_failed_call) == 1
        assert agent_failed_call[0][0][1]["task_id"] == crew.task_ids[mock_task.description]
        assert agent_failed_call[0][0][1]["crew_id"] == crew.crew_id
        assert agent_failed_call[0][0][1]["agent_id"] == mock_agent_without_codegen.name
        assert "error" in agent_failed_call[0][0][1]["data"]
        assert "Process task error" in agent_failed_call[0][0][1]["message"]
    
    @pytest.mark.asyncio
    async def test_process_task_with_existing_task_id(self, mock_crewai_internals, mock_emit_event, mock_agent_without_codegen):
        """Test _process_task method with existing task_id."""
        _, mock_super_process_task = mock_crewai_internals
        mock_output = MagicMock(spec=TaskOutput)
        mock_super_process_task.return_value = mock_output
        
        # Create task
        mock_task = MagicMock(spec=Task)
        mock_task.description = "Test task with existing ID"
        mock_task.agent = mock_agent_without_codegen
        
        # Create crew
        crew = create_mock_crew([mock_task])
        
        # Set existing task_id
        crew.task_ids[mock_task.description] = TEST_TASK_ID_1
        
        # Call _process_task
        await crew._process_task(mock_task, {})
        
        # Verify task_id was reused
        assert crew.task_ids[mock_task.description] == TEST_TASK_ID_1
        
        # Verify events used the correct task_id
        agent_started_call = [call for call in mock_emit_event.call_args_list 
                             if call[0][0] == EventType.AGENT_STARTED]
        assert agent_started_call[0][0][1]["task_id"] == TEST_TASK_ID_1

# Test extract_codegen_results method
class TestExtractCodegenResults:
    """Tests for CustomCrew _extract_codegen_results method."""
    
    def test_extract_codegen_results_json_format(self, mock_agent_with_codegen):
        """Test _extract_codegen_results with JSON format."""
        mock_agent, _ = mock_agent_with_codegen
        
        # Create task
        mock_task = MagicMock(spec=Task)
        mock_task.agent = mock_agent
        
        # Create result
        mock_output = MagicMock(spec=TaskOutput)
        mock_output.raw_output = MOCK_CODEGEN_RESULT_JSON
        
        # Create crew
        crew = create_mock_crew([mock_task])
        
        # Call _extract_codegen_results
        crew._extract_codegen_results(mock_task, mock_output)
        
        # Verify shared_context was updated
        assert "codegen" in crew.shared_context
        assert "result" in crew.shared_context["codegen"]
        assert crew.shared_context["codegen"]["result"] == "analysis_complete"
        
        # Verify code_result was extracted
        assert "code_result" in crew.shared_context
        assert crew.shared_context["code_result"] == "analysis_complete"
        
        # Verify visualizations were extracted
        assert "visualizations" in crew.shared_context
        assert len(crew.shared_context["visualizations"]) == 2
    
    def test_extract_codegen_results_tool_result_format(self, mock_agent_with_codegen):
        """Test _extract_codegen_results with Tool Result format."""
        mock_agent, _ = mock_agent_with_codegen
        
        # Create task
        mock_task = MagicMock(spec=Task)
        mock_task.agent = mock_agent
        
        # Create result with Tool Result format
        mock_output = MagicMock(spec=TaskOutput)
        mock_output.raw_output = """
        Analysis complete.
        
        Tool Result: {"result": "tool_analysis", "execution": {"stdout": "Success"}}
        
        Further analysis shows...
        """
        
        # Create crew
        crew = create_mock_crew([mock_task])
        
        # Call _extract_codegen_results
        crew._extract_codegen_results(mock_task, mock_output)
        
        # Verify shared_context was updated
        assert "codegen" in crew.shared_context
        assert "result" in crew.shared_context["codegen"]
        assert crew.shared_context["codegen"]["result"] == "tool_analysis"
    
    def test_extract_codegen_results_direct_tool_result(self, mock_agent_with_codegen):
        """Test _extract_codegen_results with direct tool result."""
        mock_agent, mock_codegen_tool = mock_agent_with_codegen
        
        # Set up direct tool result
        mock_codegen_tool._last_result = MOCK_CODEGEN_RESULT_TOOL_OUTPUT
        
        # Create task
        mock_task = MagicMock(spec=Task)
        mock_task.agent = mock_agent
        
        # Create result without CodeGenTool result in text
        mock_output = MagicMock(spec=TaskOutput)
        mock_output.raw_output = "Task completed but no JSON result in output."
        
        # Create crew
        crew = create_mock_crew([mock_task])
        
        # Call _extract_codegen_results
        crew._extract_codegen_results(mock_task, mock_output)
        
        # Verify shared_context was updated with direct tool result
        assert "codegen" in crew.shared_context
        assert "result" in crew.shared_context["codegen"]
        assert crew.shared_context["codegen"]["result"] == "analysis_complete_from_tool"
    
    def test_extract_codegen_results_no_tool(self):
        """Test _extract_codegen_results with no CodeGenTool."""
        # Create agent without CodeGenTool
        mock_agent = MagicMock(spec=Agent)
        mock_agent.tools = []
        
        # Create task
        mock_task = MagicMock(spec=Task)
        mock_task.agent = mock_agent
        
        # Create result
        mock_output = MagicMock(spec=TaskOutput)
        mock_output.raw_output = "Task completed."
        
        # Create crew
        crew = create_mock_crew([mock_task])
        
        # Call _extract_codegen_results
        crew._extract_codegen_results(mock_task, mock_output)
        
        # Verify shared_context was not updated
        assert len(crew.shared_context) == 0
    
    def test_extract_codegen_results_no_agent(self):
        """Test _extract_codegen_results with no agent."""
        # Create task without agent
        mock_task = MagicMock(spec=Task)
        mock_task.agent = None
        
        # Create result
        mock_output = MagicMock(spec=TaskOutput)
        mock_output.raw_output = "Task completed."
        
        # Create crew
        crew = create_mock_crew([mock_task])
        
        # Call _extract_codegen_results
        crew._extract_codegen_results(mock_task, mock_output)
        
        # Verify shared_context was not updated
        assert len(crew.shared_context) == 0
    
    def test_extract_codegen_results_no_result(self, mock_agent_with_codegen):
        """Test _extract_codegen_results with no result in output."""
        mock_agent, _ = mock_agent_with_codegen
        
        # Create task
        mock_task = MagicMock(spec=Task)
        mock_task.agent = mock_agent
        
        # Create result with no CodeGenTool result
        mock_output = MagicMock(spec=TaskOutput)
        mock_output.raw_output = "Task completed but no JSON or Tool Result."
        
        # Create crew
        crew = create_mock_crew([mock_task])
        
        # Call _extract_codegen_results
        crew._extract_codegen_results(mock_task, mock_output)
        
        # Verify shared_context was not updated
        assert len(crew.shared_context) == 0
    
    def test_extract_codegen_results_with_exception(self, mock_agent_with_codegen):
        """Test _extract_codegen_results with exception."""
        mock_agent, _ = mock_agent_with_codegen
        
        # Create task
        mock_task = MagicMock(spec=Task)
        mock_task.agent = mock_agent
        
        # Create result
        mock_output = MagicMock(spec=TaskOutput)
        mock_output.raw_output = MOCK_CODEGEN_RESULT_JSON
        
        # Create crew
        crew = create_mock_crew([mock_task])
        
        # Mock json.loads to raise exception
        with patch("json.loads", side_effect=ValueError("JSON error")):
            # Call _extract_codegen_results
            crew._extract_codegen_results(mock_task, mock_output)
            
            # Verify shared_context was not updated
            assert len(crew.shared_context) == 0

# Test context propagation between tasks
class TestContextPropagation:
    """Tests for context propagation between tasks."""
    
    @pytest.mark.asyncio
    async def test_context_propagation_between_tasks(self, mock_crewai_internals, mock_emit_event, mock_agent_with_codegen, mock_agent_without_codegen):
        """Test context propagation between tasks."""
        _, mock_super_process_task = mock_crewai_internals
        mock_agent1, mock_codegen_tool = mock_agent_with_codegen
        mock_agent2 = mock_agent_without_codegen
        
        # Create mock outputs
        mock_output1 = MagicMock(spec=TaskOutput)
        mock_output1.raw_output = MOCK_CODEGEN_RESULT_JSON
        
        mock_output2 = MagicMock(spec=TaskOutput)
        mock_output2.raw_output = "Task 2 completed with access to shared context."
        
        # Configure mock_super_process_task to return different outputs
        mock_super_process_task.side_effect = [mock_output1, mock_output2]
        
        # Create tasks
        mock_task1 = MagicMock(spec=Task)
        mock_task1.description = "Task 1 with CodeGenTool"
        mock_task1.agent = mock_agent1
        
        mock_task2 = MagicMock(spec=Task)
        mock_task2.description = "Task 2 uses context from Task 1"
        mock_task2.agent = mock_agent2
        
        # Create crew
        crew = create_mock_crew([mock_task1, mock_task2])
        
        # Process first task
        await crew._process_task(mock_task1, {"initial": "input"})
        
        # Verify shared_context was updated after task 1
        assert "codegen" in crew.shared_context
        assert "code_result" in crew.shared_context
        assert "visualizations" in crew.shared_context
        
        # Process second task
        await crew._process_task(mock_task2, {"task2_input": "value"})
        
        # Verify second task received enhanced inputs
        second_call_args = mock_super_process_task.call_args_list[1][0]
        assert second_call_args[0] == mock_task2
        
        # Check that task2 inputs include both direct inputs and shared context
        task2_inputs = second_call_args[1]
        assert "task2_input" in task2_inputs
        assert "initial" in task2_inputs
        assert "codegen" in task2_inputs
        assert "code_result" in task2_inputs
        assert "visualizations" in task2_inputs
    
    @pytest.mark.asyncio
    async def test_context_propagation_in_kickoff(self, mock_crewai_internals, mock_emit_event):
        """Test context propagation in kickoff method."""
        mock_super_kickoff, _ = mock_crewai_internals
        
        # Create crew
        crew = create_mock_crew([MagicMock(spec=Task)])
        
        # Call kickoff with inputs
        inputs = {"key1": "value1", "key2": {"nested": "value"}}
        crew.kickoff(inputs)
        
        # Verify shared_context contains all inputs
        assert crew.shared_context == inputs
        
        # Verify super().kickoff was called with the same inputs
        mock_super_kickoff.assert_called_once_with(inputs)

# Test event emissions
class TestEventEmissions:
    """Tests for event emissions."""
    
    def test_crew_started_event(self, mock_crewai_internals, mock_emit_event):
        """Test CREW_STARTED event emission."""
        mock_super_kickoff, _ = mock_crewai_internals
        
        # Create crew
        crew = create_mock_crew([MagicMock(spec=Task)])
        crew.name = "test_crew_name"
        
        # Call kickoff
        crew.kickoff()
        
        # Verify CREW_STARTED event was emitted
        crew_started_calls = [call for call in mock_emit_event.call_args_list 
                             if call[0][0] == EventType.CREW_STARTED]
        assert len(crew_started_calls) == 1
        
        # Check event data
        event_data = crew_started_calls[0][0][1]
        assert event_data["crew_id"] == crew.crew_id
        assert event_data["progress"] == 0
        assert event_data["data"]["agent_count"] == 1
        assert event_data["data"]["task_count"] == 1
        assert event_data["data"]["crew_name"] == "test_crew_name"
    
    def test_crew_completed_event(self, mock_crewai_internals, mock_emit_event):
        """Test CREW_COMPLETED event emission."""
        mock_super_kickoff, _ = mock_crewai_internals
        mock_super_kickoff.return_value = "Crew result"
        
        # Create crew
        crew = create_mock_crew([MagicMock(spec=Task)])
        
        # Call kickoff
        crew.kickoff()
        
        # Verify CREW_COMPLETED event was emitted
        crew_completed_calls = [call for call in mock_emit_event.call_args_list 
                               if call[0][0] == EventType.CREW_COMPLETED]
        assert len(crew_completed_calls) == 1
        
        # Check event data
        event_data = crew_completed_calls[0][0][1]
        assert event_data["crew_id"] == crew.crew_id
        assert event_data["progress"] == 100
        assert "result_summary" in event_data["data"]
        assert event_data["data"]["task_count"] == 1
        assert event_data["data"]["completed_tasks"] == 0  # No tasks were actually processed
    
    def test_crew_failed_event(self, mock_crewai_internals, mock_emit_event):
        """Test CREW_FAILED event emission."""
        mock_super_kickoff, _ = mock_crewai_internals
        mock_super_kickoff.side_effect = ValueError("Crew error")
        
        # Create crew
        crew = create_mock_crew([MagicMock(spec=Task)])
        
        # Call kickoff and expect exception
        with pytest.raises(ValueError):
            crew.kickoff()
        
        # Verify CREW_FAILED event was emitted
        crew_failed_calls = [call for call in mock_emit_event.call_args_list 
                            if call[0][0] == EventType.CREW_FAILED]
        assert len(crew_failed_calls) == 1
        
        # Check event data
        event_data = crew_failed_calls[0][0][1]
        assert event_data["crew_id"] == crew.crew_id
        assert event_data["progress"] == 100
        assert "error" in event_data["data"]
        assert "Crew error" in event_data["message"]
    
    @pytest.mark.asyncio
    async def test_agent_events(self, mock_crewai_internals, mock_emit_event, mock_agent_without_codegen):
        """Test agent event emissions."""
        _, mock_super_process_task = mock_crewai_internals
        mock_output = MagicMock(spec=TaskOutput)
        mock_super_process_task.return_value = mock_output
        
        # Create task
        mock_task = MagicMock(spec=Task)
        mock_task.description = "Test agent events"
        mock_task.agent = mock_agent_without_codegen
        
        # Create crew
        crew = create_mock_crew([mock_task])
        
        # Call _process_task
        await crew._process_task(mock_task, {})
        
        # Verify AGENT_STARTED event was emitted
        agent_started_calls = [call for call in mock_emit_event.call_args_list 
                              if call[0][0] == EventType.AGENT_STARTED]
        assert len(agent_started_calls) == 1
        
        # Check event data
        event_data = agent_started_calls[0][0][1]
        assert event_data["task_id"] == crew.task_ids[mock_task.description]
        assert event_data["crew_id"] == crew.crew_id
        assert event_data["agent_id"] == mock_agent_without_codegen.name
        assert "task_description" in event_data["data"]
        
        # Verify AGENT_PROGRESS event was emitted
        agent_progress_calls = [call for call in mock_emit_event.call_args_list 
                               if call[0][0] == EventType.AGENT_PROGRESS]
        assert len(agent_progress_calls) == 1
        
        # Verify AGENT_COMPLETED event was emitted
        agent_completed_calls = [call for call in mock_emit_event.call_args_list 
                                if call[0][0] == EventType.AGENT_COMPLETED]
        assert len(agent_completed_calls) == 1
        
        # Check event data
        event_data = agent_completed_calls[0][0][1]
        assert event_data["task_id"] == crew.task_ids[mock_task.description]
        assert event_data["crew_id"] == crew.crew_id
        assert event_data["agent_id"] == mock_agent_without_codegen.name
        assert event_data["progress"] == 100
        assert "result_summary" in event_data["data"]
    
    @pytest.mark.asyncio
    async def test_tool_events(self, mock_crewai_internals, mock_emit_event, mock_agent_with_codegen):
        """Test tool event emissions."""
        _, mock_super_process_task = mock_crewai_internals
        mock_agent, _ = mock_agent_with_codegen
        
        # Mock task output with CodeGenTool result
        mock_output = MagicMock(spec=TaskOutput)
        mock_output.raw_output = MOCK_CODEGEN_RESULT_JSON
        mock_super_process_task.return_value = mock_output
        
        # Create task
        mock_task = MagicMock(spec=Task)
        mock_task.description = "Test tool events"
        mock_task.agent = mock_agent
        
        # Create crew
        crew = create_mock_crew([mock_task])
        
        # Call _process_task
        await crew._process_task(mock_task, {})
        
        # Verify TOOL_STARTED event was emitted
        tool_started_calls = [call for call in mock_emit_event.call_args_list 
                             if call[0][0] == EventType.TOOL_STARTED]
        assert len(tool_started_calls) == 1
        
        # Check event data
        event_data = tool_started_calls[0][0][1]
        assert event_data["task_id"] == crew.task_ids[mock_task.description]
        assert event_data["crew_id"] == crew.crew_id
        assert event_data["agent_id"] == mock_agent.name
        assert event_data["tool_id"] == "CodeGenTool"
        
        # Verify TOOL_COMPLETED event was emitted
        tool_completed_calls = [call for call in mock_emit_event.call_args_list 
                               if call[0][0] == EventType.TOOL_COMPLETED]
        assert len(tool_completed_calls) == 1
        
        # Check event data
        event_data = tool_completed_calls[0][0][1]
        assert event_data["task_id"] == crew.task_ids[mock_task.description]
        assert event_data["crew_id"] == crew.crew_id
        assert event_data["agent_id"] == mock_agent.name
        assert event_data["tool_id"] == "CodeGenTool"
        assert event_data["data"]["has_visualizations"] == True
        assert event_data["data"]["visualization_count"] == 2

# Test error handling
class TestErrorHandling:
    """Tests for error handling."""
    
    @pytest.mark.asyncio
    async def test_process_task_error_handling(self, mock_crewai_internals, mock_emit_event, mock_agent_without_codegen):
        """Test _process_task error handling."""
        _, mock_super_process_task = mock_crewai_internals
        
        # First call raises exception, second call succeeds (fallback)
        mock_super_process_task.side_effect = [
            ValueError("Process task error"),
            MagicMock(spec=TaskOutput)
        ]
        
        # Create task
        mock_task = MagicMock(spec=Task)
        mock_task.description = "Test error handling"
        mock_task.agent = mock_agent_without_codegen
        
        # Create crew
        crew = create_mock_crew([mock_task])
        
        # Call _process_task
        await crew._process_task(mock_task, {})
        
        # Verify AGENT_FAILED event was emitted
        agent_failed_calls = [call for call in mock_emit_event.call_args_list 
                             if call[0][0] == EventType.AGENT_FAILED]
        assert len(agent_failed_calls) == 1
        
        # Check event data
        event_data = agent_failed_calls[0][0][1]
        assert event_data["task_id"] == crew.task_ids[mock_task.description]
        assert event_data["crew_id"] == crew.crew_id
        assert event_data["agent_id"] == mock_agent_without_codegen.name
        assert "error" in event_data["data"]
        assert "Process task error" in event_data["message"]
        
        # Verify super()._process_task was called twice (once for error, once for fallback)
        assert mock_super_process_task.call_count == 2
        
        # Verify both calls had the same arguments
        assert mock_super_process_task.call_args_list[0][0] == mock_super_process_task.call_args_list[1][0]
    
    def test_extract_codegen_results_error_handling(self, mock_agent_with_codegen):
        """Test _extract_codegen_results error handling."""
        mock_agent, _ = mock_agent_with_codegen
        
        # Create task
        mock_task = MagicMock(spec=Task)
        mock_task.agent = mock_agent
        
        # Create result
        mock_output = MagicMock(spec=TaskOutput)
        mock_output.raw_output = "Invalid JSON: {not valid json}"
        
        # Create crew
        crew = create_mock_crew([mock_task])
        
        # Call _extract_codegen_results
        crew._extract_codegen_results(mock_task, mock_output)
        
        # Verify shared_context was not updated (error was handled)
        assert len(crew.shared_context) == 0
    
    def test_kickoff_error_handling(self, mock_crewai_internals, mock_emit_event):
        """Test kickoff error handling."""
        mock_super_kickoff, _ = mock_crewai_internals
        mock_super_kickoff.side_effect = ValueError("Kickoff error")
        
        # Create crew
        crew = create_mock_crew([MagicMock(spec=Task)])
        
        # Call kickoff and expect exception
        with pytest.raises(ValueError, match="Kickoff error"):
            crew.kickoff()
        
        # Verify CREW_FAILED event was emitted
        crew_failed_calls = [call for call in mock_emit_event.call_args_list 
                            if call[0][0] == EventType.CREW_FAILED]
        assert len(crew_failed_calls) == 1
        
        # Check event data
        event_data = crew_failed_calls[0][0][1]
        assert event_data["crew_id"] == crew.crew_id
        assert "error" in event_data["data"]
        assert "Kickoff error" in event_data["message"]
    
    @pytest.mark.asyncio
    async def test_event_emission_error_handling(self, mock_crewai_internals, mock_agent_without_codegen):
        """Test error handling in event emissions."""
        _, mock_super_process_task = mock_crewai_internals
        mock_output = MagicMock(spec=TaskOutput)
        mock_super_process_task.return_value = mock_output
        
        # Create task
        mock_task = MagicMock(spec=Task)
        mock_task.description = "Test event emission error"
        mock_task.agent = mock_agent_without_codegen
        
        # Create crew
        crew = create_mock_crew([mock_task])
        
        # Mock emit_event to raise exception
        with patch("backend.agents.custom_crew.emit_event", side_effect=Exception("Event error")):
            # Call _process_task
            result = await crew._process_task(mock_task, {})
            
            # Verify task still completed despite event error
            assert result == mock_output
            
            # Verify shared_context was updated
            assert crew.completed_tasks == 1

# Test integration scenarios
class TestIntegrationScenarios:
    """Tests for integration scenarios."""
    
    @pytest.mark.asyncio
    async def test_multi_task_workflow(self, mock_crewai_internals, mock_emit_event, mock_agent_with_codegen, mock_agent_without_codegen):
        """Test multi-task workflow with context propagation."""
        _, mock_super_process_task = mock_crewai_internals
        mock_agent1, mock_codegen_tool = mock_agent_with_codegen
        mock_agent2 = mock_agent_without_codegen
        
        # Create mock outputs
        mock_output1 = MagicMock(spec=TaskOutput)
        mock_output1.raw_output = MOCK_CODEGEN_RESULT_JSON
        
        mock_output2 = MagicMock(spec=TaskOutput)
        mock_output2.raw_output = "Final report using visualizations from task 1."
        
        # Configure mock_super_process_task to return different outputs
        mock_super_process_task.side_effect = [mock_output1, mock_output2]
        
        # Create tasks
        mock_task1 = MagicMock(spec=Task)
        mock_task1.description = "Generate visualizations"
        mock_task1.agent = mock_agent1
        
        mock_task2 = MagicMock(spec=Task)
        mock_task2.description = "Create final report"
        mock_task2.agent = mock_agent2
        
        # Create crew
        crew = create_mock_crew([mock_task1, mock_task2])
        
        # Initial inputs
        initial_inputs = {
            "query": "Analyze transaction patterns",
            "data_source": "neo4j"
        }
        
        # Process tasks sequentially
        await crew._process_task(mock_task1, initial_inputs)
        await crew._process_task(mock_task2, {})
        
        # Verify task1 received initial inputs
        task1_inputs = mock_super_process_task.call_args_list[0][0][1]
        assert task1_inputs["query"] == "Analyze transaction patterns"
        assert task1_inputs["data_source"] == "neo4j"
        
        # Verify task2 received shared context from task1
        task2_inputs = mock_super_process_task.call_args_list[1][0][1]
        assert "query" in task2_inputs
        assert "data_source" in task2_inputs
        assert "codegen" in task2_inputs
        assert "code_result" in task2_inputs
        assert "visualizations" in task2_inputs
        assert len(task2_inputs["visualizations"]) == 2
        
        # Verify completed_tasks was updated correctly
        assert crew.completed_tasks == 2
        
        # Verify events were emitted for both tasks
        task1_events = [call for call in mock_emit_event.call_args_list 
                       if call[0][0] == EventType.AGENT_STARTED and 
                       call[0][1]["task_id"] == crew.task_ids[mock_task1.description]]
        assert len(task1_events) == 1
        
        task2_events = [call for call in mock_emit_event.call_args_list 
                       if call[0][0] == EventType.AGENT_STARTED and 
                       call[0][1]["task_id"] == crew.task_ids[mock_task2.description]]
        assert len(task2_events) == 1
