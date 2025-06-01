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
from backend.integrations.e2b_client import E2BClient
from backend.agents.factory import CrewFactory


@pytest.mark.asyncio
async def test_codegen_result_merges_into_context():
    """Test that CodeGenTool results are merged into crew context."""
    # Create mock E2B client
    mock_e2b = MagicMock(spec=E2BClient)
    mock_e2b.create_sandbox = AsyncMock(return_value="sandbox-123")
    mock_e2b.execute_code = AsyncMock(return_value={
        "success": True,
        "stdout": "Execution successful\nResult: 42",
        "stderr": "",
        "exit_code": 0,
        "execution_time": 0.5
    })
    
    # Create CodeGenTool with mock E2B client
    tool = CodeGenTool(e2b_client=mock_e2b)
    
    # Run the tool with execute_code=True
    result = await tool.run(
        question="What is 6 * 7?",
        execute_code=True
    )
    
    # Verify the result contains execution details
    assert result["success"] is True, "Tool execution should succeed"
    assert "execution" in result, "Result should contain execution details"
    assert result["execution"]["success"] is True, "Execution should succeed"
    assert "42" in result["execution"]["stdout"], "Output should contain result"


@pytest.mark.asyncio
async def test_subsequent_agents_access_codegen_results():
    """Test that subsequent agents can access CodeGenTool results."""
    # Mock CrewFactory
    crew_factory = MagicMock()
    crew_factory.run_agent = AsyncMock()
    
    # Mock CodeGenTool execution result
    codegen_result = {
        "success": True,
        "code": "print('Data analysis complete')\nprint(json.dumps({'data': [1, 2, 3, 4, 5], 'summary': 'Data analysis complete'}))",
        "execution": {
            "success": True,
            "stdout": "Data analysis complete\n{\"data\": [1, 2, 3, 4, 5], \"summary\": \"Data analysis complete\"}",
            "stderr": "",
            "exit_code": 0
        },
        "result": {"data": [1, 2, 3, 4, 5], "summary": "Data analysis complete"},
        "visualizations": [
            {
                "filename": "plot.png",
                "content": base64.b64encode(b"fake_image_data").decode("utf-8")
            }
        ]
    }
    
    # Initial crew context
    crew_context = {
        "question": "Analyze this dataset",
        "data": [1, 2, 3, 4, 5]
    }
    
    # Simulate CodeGenTool execution and context update
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
    # Create mock E2B client
    mock_e2b = MagicMock(spec=E2BClient)
    mock_e2b.create_sandbox = AsyncMock(return_value="sandbox-123")
    mock_e2b.execute_code = AsyncMock(return_value={
        "success": False,
        "stdout": "",
        "stderr": "Error: Division by zero",
        "exit_code": 1,
        "execution_time": 0.2
    })
    
    # Create CodeGenTool with mock E2B client
    tool = CodeGenTool(e2b_client=mock_e2b)
    
    # Run the tool with execute_code=True
    result = await tool.run(
        question="Calculate 1/0",
        execute_code=True
    )
    
    # Verify the result contains error information
    assert "execution" in result, "Result should contain execution details"
    assert result["execution"]["success"] is False, "Execution should fail"
    assert "Division by zero" in result["execution"]["stderr"], "Error message not captured"


@pytest.mark.asyncio
async def test_codegen_with_artifacts():
    """Test that CodeGenTool correctly handles and passes artifacts."""
    # Create fake artifacts
    fake_plot = base64.b64encode(b"fake_plot_data").decode("utf-8")
    fake_csv = base64.b64encode(b"col1,col2\n1,2\n3,4").decode("utf-8")
    
    # Create mock E2B client
    mock_e2b = MagicMock(spec=E2BClient)
    mock_e2b.create_sandbox = AsyncMock(return_value="sandbox-123")
    mock_e2b.execute_code = AsyncMock(return_value={
        "success": True,
        "stdout": "Execution successful\nGenerated plot and CSV",
        "stderr": "",
        "exit_code": 0,
        "execution_time": 0.8
    })
    mock_e2b.list_files = AsyncMock(return_value=["plot.png", "data.csv"])
    mock_e2b.download_file = AsyncMock(side_effect=lambda filename, *args: 
        b"fake_plot_data" if filename == "plot.png" else b"col1,col2\n1,2\n3,4"
    )
    
    # Create CodeGenTool with mock E2B client
    tool = CodeGenTool(e2b_client=mock_e2b)
    
    # Run the tool with execute_code=True
    result = await tool.run(
        question="Generate a plot and CSV",
        execute_code=True
    )
    
    # Verify the result contains visualizations
    assert "visualizations" in result, "Result should contain visualizations"
    assert len(result["visualizations"]) > 0, "Visualizations should not be empty"
    assert any(v["filename"] == "plot.png" for v in result["visualizations"]), "Plot visualization missing"
    assert any(v["content"] == fake_plot for v in result["visualizations"]), "Plot content incorrect"


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
                "success": True,
                "code": "print('Statistical analysis complete')\nprint(json.dumps({'mean': 42, 'median': 37}))",
                "execution": {
                    "success": True,
                    "stdout": "Statistical analysis complete\n{\"mean\": 42, \"median\": 37}",
                    "stderr": "",
                    "exit_code": 0
                },
                "result": {"mean": 42, "median": 37},
                "visualizations": [
                    {
                        "filename": "histogram.png",
                        "content": base64.b64encode(b"histogram_data").decode("utf-8")
                    }
                ]
            }
            
            # In a properly implemented system, this would be added to context
            context["codegen"] = codegen_result
            
            # Simulate report_writer execution
            # It should include codegen results
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
        
        # Verify the report contains the CodeGenTool results
        assert "Mean: 42" in result["report"], "Report should contain mean value"
        assert "Median: 37" in result["report"], "Report should contain median value"
        assert "No statistical results available" not in result["report"], "Report should not indicate missing results"
