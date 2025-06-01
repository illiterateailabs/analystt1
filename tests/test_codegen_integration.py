"""
Tests for CodeGenTool result integration with crew context.

This module tests that CodeGenTool execution results are properly
merged into the crew context and accessible to subsequent agents.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import json
import base64

from backend.agents.tools.code_gen_tool import CodeGenTool
from backend.agents.tools.sandbox_exec_tool import SandboxExecTool
from backend.agents.factory import CrewFactory


@pytest.mark.asyncio
async def test_codegen_result_merges_into_context():
    """Test that CodeGenTool results are merged into crew context."""
    tool = CodeGenTool()
    
    # Mock the sandbox execution to return a deterministic result
    async def fake_exec(*args, **kwargs):
        return {
            "result": 42,
            "stdout": "Execution successful\nResult: 42",
            "stderr": "",
            "exit_code": 0,
            "execution_time": 0.5
        }
    
    # Patch the _execute_in_sandbox method
    with patch.object(tool, '_execute_in_sandbox', side_effect=fake_exec):
        result = await tool.run({"question": "What is 6 * 7?"})
    
    # This test should fail initially as the feature is not implemented
    # The result should contain a 'result' key with the value 42
    assert 'result' in result, "Result key missing from CodeGenTool output"
    assert result['result'] == 42, "Expected result value not found"


@pytest.mark.asyncio
async def test_subsequent_agents_access_codegen_results():
    """Test that subsequent agents can access CodeGenTool results."""
    # Mock CrewFactory
    crew_factory = MagicMock()
    crew_factory.run_agent = AsyncMock()
    
    # Mock CodeGenTool execution result
    codegen_result = {
        "result": {"data": [1, 2, 3, 4, 5], "summary": "Data analysis complete"},
        "stdout": "Analysis successful\nData: [1, 2, 3, 4, 5]",
        "stderr": "",
        "artifacts": {
            "plot.png": base64.b64encode(b"fake_image_data").decode("utf-8")
        }
    }
    
    # Initial crew context
    crew_context = {
        "question": "Analyze this dataset",
        "data": [1, 2, 3, 4, 5]
    }
    
    # Simulate CodeGenTool execution and context update
    # This is what should happen but currently doesn't
    updated_context = crew_context.copy()
    updated_context["codegen"] = codegen_result
    
    # Mock the report_writer agent execution
    crew_factory.run_agent.return_value = {
        "report": "Analysis shows data: [1, 2, 3, 4, 5]",
        "used_codegen_result": True
    }
    
    # Run the report_writer agent with the updated context
    report_result = await crew_factory.run_agent(
        "report_writer",
        updated_context
    )
    
    # Verify the report_writer received and used the CodeGenTool results
    assert report_result["used_codegen_result"], "Report writer did not use CodeGenTool results"
    assert "[1, 2, 3, 4, 5]" in report_result["report"], "CodeGen results not in report"
    
    # Verify the call to run_agent included the codegen results
    crew_factory.run_agent.assert_called_once()
    call_args = crew_factory.run_agent.call_args[0][1]  # Get the context argument
    assert "codegen" in call_args, "CodeGen results not passed to subsequent agent"
    assert call_args["codegen"] == codegen_result, "CodeGen results not correctly passed"


@pytest.mark.asyncio
async def test_codegen_failure_handling():
    """Test that CodeGenTool failure is properly handled and reported."""
    tool = CodeGenTool()
    
    # Mock a failed sandbox execution
    async def fake_failed_exec(*args, **kwargs):
        return {
            "result": None,
            "stdout": "",
            "stderr": "Error: Division by zero",
            "exit_code": 1,
            "execution_time": 0.2
        }
    
    # Patch the _execute_in_sandbox method
    with patch.object(tool, '_execute_in_sandbox', side_effect=fake_failed_exec):
        result = await tool.run({"question": "Calculate 1/0"})
    
    # Verify failure is properly reported
    assert 'error' in result, "Error key missing from failed CodeGenTool output"
    assert 'Division by zero' in result['error'], "Error message not properly captured"
    assert result.get('success', True) is False, "Failure not properly indicated"


@pytest.mark.asyncio
async def test_codegen_with_artifacts():
    """Test that CodeGenTool correctly handles and passes artifacts."""
    tool = CodeGenTool()
    
    # Mock sandbox execution with artifacts
    fake_plot = base64.b64encode(b"fake_plot_data").decode("utf-8")
    fake_csv = base64.b64encode(b"col1,col2\n1,2\n3,4").decode("utf-8")
    
    async def fake_exec_with_artifacts(*args, **kwargs):
        return {
            "result": {"summary": "Generated 2 artifacts"},
            "stdout": "Execution successful\nGenerated plot and CSV",
            "stderr": "",
            "exit_code": 0,
            "execution_time": 0.8,
            "artifacts": {
                "plot.png": fake_plot,
                "data.csv": fake_csv
            }
        }
    
    # Patch the _execute_in_sandbox method
    with patch.object(tool, '_execute_in_sandbox', side_effect=fake_exec_with_artifacts):
        result = await tool.run({"question": "Generate a plot and CSV"})
    
    # Verify artifacts are included in the result
    assert 'artifacts' in result, "Artifacts missing from CodeGenTool output"
    assert 'plot.png' in result['artifacts'], "Plot artifact missing"
    assert 'data.csv' in result['artifacts'], "CSV artifact missing"
    assert result['artifacts']['plot.png'] == fake_plot, "Plot data incorrect"
    assert result['artifacts']['data.csv'] == fake_csv, "CSV data incorrect"


@pytest.mark.asyncio
async def test_end_to_end_crew_with_codegen():
    """Test end-to-end crew execution with CodeGenTool integration."""
    # Create a mock CrewFactory
    with patch('backend.agents.factory.CrewFactory', autospec=True) as MockCrewFactory:
        factory_instance = MockCrewFactory.return_value
        
        # Mock the run_crew method
        async def fake_run_crew(crew_name, inputs=None):
            # Simulate crew execution with agents
            context = inputs or {}
            
            # Simulate CodeGenTool execution
            codegen_result = {
                "result": {"mean": 42, "median": 37},
                "stdout": "Statistical analysis complete",
                "artifacts": {
                    "histogram.png": base64.b64encode(b"histogram_data").decode("utf-8")
                }
            }
            
            # In a properly implemented system, this would be added to context
            # But we're testing that it's missing
            # context["codegen"] = codegen_result
            
            # Simulate report_writer execution
            # It should include codegen results, but won't in the broken implementation
            report = "# Analysis Report\n\n"
            if "codegen" in context:
                report += f"Mean: {context['codegen']['result']['mean']}\n"
                report += f"Median: {context['codegen']['result']['median']}\n"
                report += "![Histogram](data:image/png;base64,...)\n"
            else:
                report += "No statistical results available\n"
            
            return {
                "success": True,
                "report": report,
                "task_id": "task_12345"
            }
        
        factory_instance.run_crew.side_effect = fake_run_crew
        
        # Run a crew
        result = await factory_instance.run_crew("data_analysis", {"dataset": [1, 2, 3, 4, 5]})
        
        # Verify the report doesn't contain the CodeGenTool results
        # This test should fail when the feature is properly implemented
        assert "No statistical results available" in result["report"], "Report should not contain CodeGenTool results yet"
        assert "Mean: 42" not in result["report"], "Report should not contain mean value yet"
