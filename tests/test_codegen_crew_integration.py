"""
Integration test for CodeGenTool results propagation to crew context.

This test verifies that results from CodeGenTool execution are properly
merged into the crew context and available to subsequent agents in the crew.
"""

import pytest
import json
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from crewai import Agent, Task, Crew
from backend.agents.factory import CrewFactory
from backend.agents.tools.code_gen_tool import CodeGenTool
from backend.agents.llm import GeminiLLMProvider
from backend.integrations.e2b_client import E2BClient


@pytest.fixture
def mock_e2b_client():
    """Mock E2B client for sandbox code execution."""
    mock_client = MagicMock(spec=E2BClient)
    mock_client.create_sandbox = AsyncMock(return_value="sandbox-123")
    mock_client.execute_code = AsyncMock(return_value={
        "success": True,
        "stdout": "Execution successful\n{\"mean\": 3, \"sum\": 15}",
        "stderr": "",
        "exit_code": 0,
        "execution_time": 0.5
    })
    mock_client.list_files = AsyncMock(return_value=[])
    mock_client.close_sandbox = AsyncMock(return_value=None)
    return mock_client


@pytest.mark.asyncio
async def test_codegen_execution_works(mock_e2b_client):
    """
    Test that CodeGenTool execution itself works correctly.
    
    This test should PASS to confirm the tool works, but the results
    aren't propagated to the crew context.
    """
    # Create CodeGenTool with mock E2B client
    with patch("backend.agents.tools.code_gen_tool.GeminiClient"):
        tool = CodeGenTool(e2b_client=mock_e2b_client)
        
        # Run the tool with execute_code=True
        result = await tool.run(
            question="Calculate statistics for [1, 2, 3, 4, 5]",
            execute_code=True
        )
        
        # Verify the result contains execution details
        assert result["success"] is True, "Tool execution should succeed"
        assert "execution" in result, "Result should contain execution details"
        assert result["execution"]["success"] is True, "Execution should succeed"
        assert "result" in result, "Result should be parsed from stdout"
        assert result["result"] == {"mean": 3, "sum": 15}, "Result should contain statistics"


@pytest.mark.xfail(reason="CodeGenTool results are not integrated into crew context")
@pytest.mark.asyncio
async def test_codegen_results_propagation():
    """
    Test that CodeGenTool results are properly integrated into crew context.
    
    This test is expected to FAIL with the current implementation because
    the CodeGenTool results are not being merged into the crew context.
    
    What needs to be fixed:
    1. In CrewFactory, add a post-task hook to merge CodeGenTool results into crew_context
    2. Create a CodeGenResult dataclass to standardize the result format
    3. Update the report_writer templates to include {{codegen.result}} references
    """
    # Mock the necessary components
    with patch("backend.agents.factory.CrewFactory.run_crew") as mock_run_crew:
        # Simulate crew execution with a sequence of tasks
        async def simulate_crew_run(crew_name, inputs=None):
            # Initial context
            context = inputs or {}
            context["data"] = [1, 2, 3, 4, 5]
            
            # Simulate code_analyst execution with CodeGenTool
            code_gen_result = {
                "success": True,
                "code": "print(json.dumps({'mean': 3, 'sum': 15}))",
                "execution": {
                    "success": True,
                    "stdout": "{\"mean\": 3, \"sum\": 15}",
                    "stderr": "",
                    "exit_code": 0
                },
                "result": {"mean": 3, "sum": 15}
            }
            
            # THIS IS THE MISSING STEP - The result should be merged into context
            # but currently isn't happening in the actual implementation
            # context["codegen"] = code_gen_result
            
            # Simulate report_writer execution
            # Without the codegen results in context, it can't include them in the report
            report = "# Analysis Report\n\n"
            if "codegen" in context and context["codegen"].get("result"):
                report += f"Mean: {context['codegen']['result']['mean']}\n"
                report += f"Sum: {context['codegen']['result']['sum']}\n"
            else:
                report += "No statistical results available\n"
            
            return {
                "success": True,
                "result": report,
                "task_id": "task-123"
            }
        
        mock_run_crew.side_effect = simulate_crew_run
        
        # Create a factory and run the crew
        factory = CrewFactory()
        result = await factory.run_crew("analysis_crew", {"data": [1, 2, 3, 4, 5]})
        
        # This assertion should fail because the report doesn't contain the statistics
        # since the CodeGenTool results aren't merged into the crew context
        assert "Mean: 3" in result["result"], "Report should contain mean from CodeGenTool"
        assert "Sum: 15" in result["result"], "Report should contain sum from CodeGenTool"
        assert "No statistical results available" not in result["result"], "Report should not indicate missing results"


@pytest.mark.asyncio
async def test_direct_crew_execution_with_codegen():
    """
    Test direct CrewAI execution with CodeGenTool to demonstrate the gap.
    
    This test uses actual CrewAI objects (Agent, Task, Crew) to show
    that without special handling, CodeGenTool results don't propagate.
    """
    # Create mock components
    mock_e2b = MagicMock(spec=E2BClient)
    mock_e2b.create_sandbox = AsyncMock(return_value="sandbox-123")
    mock_e2b.execute_code = AsyncMock(return_value={
        "success": True,
        "stdout": "{\"mean\": 3, \"sum\": 15}",
        "stderr": "",
        "exit_code": 0
    })
    
    # Create a CodeGenTool instance
    with patch("backend.agents.tools.code_gen_tool.GeminiClient"), \
         patch("backend.agents.tools.code_gen_tool.E2BClient", return_value=mock_e2b), \
         patch("backend.agents.llm.GeminiLLMProvider") as mock_llm_provider:
        
        # Setup mock LLM
        mock_llm = MagicMock()
        mock_llm_provider.return_value.get_llm.return_value = mock_llm
        
        # Create a code_gen_tool
        code_gen_tool = CodeGenTool(e2b_client=mock_e2b)
        
        # Create agents
        llm_provider = GeminiLLMProvider()
        
        code_analyst = Agent(
            role="Code Analyst",
            goal="Generate and execute code for data analysis",
            backstory="You are an expert data scientist who writes Python code to analyze data.",
            tools=[code_gen_tool],
            llm=llm_provider.get_llm()
        )
        
        report_writer = Agent(
            role="Report Writer",
            goal="Generate comprehensive reports based on analysis results",
            backstory="You are an expert report writer who creates clear, concise reports.",
            llm=llm_provider.get_llm()
        )
        
        # Create tasks
        analysis_task = Task(
            description="Analyze the data [1, 2, 3, 4, 5] and calculate statistics",
            expected_output="Statistical analysis with mean and sum",
            agent=code_analyst
        )
        
        report_task = Task(
            description="Generate a report with the analysis results",
            expected_output="Markdown report with statistics",
            agent=report_writer,
            context=[analysis_task]  # This should provide context from the previous task
        )
        
        # Create crew
        crew = Crew(
            agents=[code_analyst, report_writer],
            tasks=[analysis_task, report_task],
            process="sequential"
        )
        
        # Mock the crew.kickoff method to avoid actual execution
        # but simulate the missing context propagation issue
        original_process_task = crew._process_task
        
        async def mock_process_task(task, inputs):
            if task.agent.role == "Code Analyst":
                # Simulate code_analyst execution
                return "Analysis complete. Mean: 3, Sum: 15"
                
                # NOTE: The issue is that even though CodeGenTool returns a result
                # with the statistics, this isn't automatically added to the crew context
                # in a way that subsequent tasks can access it.
                
            elif task.agent.role == "Report Writer":
                # Simulate report_writer execution
                # It doesn't have access to the CodeGenTool results
                return "# Analysis Report\n\nNo statistical results available"
            
            return await original_process_task(task, inputs)
        
        # Apply the mock
        crew._process_task = mock_process_task
        
        # Run the crew
        with pytest.raises(AssertionError):
            result = crew.kickoff()
            
            # This should fail because the report doesn't contain the statistics
            assert "Mean: 3" in result, "Report should contain mean from CodeGenTool"
            assert "Sum: 15" in result, "Report should contain sum from CodeGenTool"
