"""
CodeGenTool for generating and executing Python code.

This module provides a tool for generating Python code based on natural language
queries and executing it in a sandboxed environment. The tool can generate
visualizations, perform data analysis, and return structured results.
"""

import json
import base64
import logging
import re
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field

from crewai_tools import BaseTool
from backend.integrations.gemini_client import GeminiClient
from backend.agents.tools.sandbox_exec_tool import SandboxExecTool

# Configure logging
logger = logging.getLogger(__name__)


class CodeGenInput(BaseModel):
    """Input model for CodeGenTool."""
    question: str = Field(
        ..., 
        description="The natural language question or task description"
    )
    context: Optional[Dict[str, Any]] = Field(
        None, 
        description="Additional context for code generation"
    )
    install_packages: Optional[List[str]] = Field(
        None, 
        description="Python packages to install before execution"
    )
    return_visualizations: bool = Field(
        True, 
        description="Whether to return generated visualizations"
    )


class CodeGenTool(BaseTool):
    """
    Tool for generating and executing Python code based on natural language queries.
    
    This tool uses Gemini to generate Python code that answers a given question,
    then executes the code in a sandboxed environment using the SandboxExecTool.
    It can generate visualizations, perform data analysis, and return structured
    results in JSON format.
    
    Example:
        ```python
        tool = CodeGenTool()
        result = await tool.execute(
            question="Analyze this transaction data and show me suspicious patterns",
            context={"transactions": [...]}
        )
        ```
    """
    
    name: str = "CodeGenTool"
    description: str = "Generates and executes Python code to answer questions, analyze data, and create visualizations."
    
    def __init__(self):
        """Initialize the CodeGenTool."""
        super().__init__()
        self.gemini_client = GeminiClient()
        self.sandbox_tool = SandboxExecTool()
    
    async def _execute(self, input_data: Union[str, Dict[str, Any]]) -> str:
        """
        Execute the tool with the given input.
        
        Args:
            input_data (Union[str, Dict[str, Any]]): Input data for the tool.
                If a string is provided, it's treated as the question.
                If a dictionary is provided, it should contain the fields defined in CodeGenInput.
        
        Returns:
            str: The result of code execution, including any visualizations.
        """
        try:
            # Parse input
            if isinstance(input_data, str):
                input_data = {"question": input_data}
            
            # Convert to CodeGenInput model
            input_model = CodeGenInput(**input_data)
            
            # Generate code
            prompt = self._prepare_prompt(input_model)
            code = await self.gemini_client.generate_python_code(prompt)
            
            if not code:
                return "Failed to generate code. Please provide more specific information."
            
            # Execute code
            sandbox_input = {
                "code": code,
                "install_packages": input_model.install_packages,
                "save_visualizations": input_model.return_visualizations
            }
            
            execution_result = await self.sandbox_tool.execute(sandbox_input)
            
            # Parse result
            result = self._parse_result(execution_result, input_model, code)
            
            # Store results in context if available
            if "_context" in input_data:
                self._store_in_context(input_data["_context"], result)
            
            return result
        except Exception as e:
            logger.error(f"Error in CodeGenTool: {e}")
            return f"Error generating or executing code: {str(e)}"
    
    def _prepare_prompt(self, input_model: CodeGenInput) -> str:
        """
        Prepare the prompt for code generation.
        
        Args:
            input_model (CodeGenInput): Input model.
            
        Returns:
            str: Prompt for code generation.
        """
        # Base prompt
        prompt = f"""Generate Python code to answer the following question:
{input_model.question}

Requirements:
1. Use pandas, numpy, matplotlib, seaborn, and other data science libraries as needed.
2. Save any visualizations as PNG files using plt.savefig().
3. Print the final results as a JSON object with the following structure:
   {{"result": "Your main findings here", "details": {...}}}
4. Include detailed comments explaining your approach.
5. Handle errors gracefully.
6. If creating visualizations, save them with descriptive filenames.

"""
        
        # Add context if provided
        if input_model.context:
            prompt += f"\nContext:\n{json.dumps(input_model.context, indent=2)}\n"
        
        # Add packages to import if specified
        if input_model.install_packages:
            prompt += f"\nMake sure to import these packages: {', '.join(input_model.install_packages)}\n"
        
        # Add visualization guidance if requested
        if input_model.return_visualizations:
            prompt += """
For visualizations:
- Use plt.figure(figsize=(10, 6)) for appropriate sizing
- Save visualizations with plt.savefig('visualization_name.png')
- Use meaningful titles, labels, and legends
- Consider using seaborn for statistical visualizations
"""
        
        return prompt
    
    def _parse_result(self, execution_result: str, input_model: CodeGenInput, code: str) -> str:
        """
        Parse the execution result.
        
        Args:
            execution_result (str): Result of code execution.
            input_model (CodeGenInput): Input model.
            code (str): Generated code.
            
        Returns:
            str: Parsed result.
        """
        # Extract JSON result from output if available
        json_result = self._parse_json_from_output(execution_result)
        
        # Get visualizations if requested
        visualizations = []
        if input_model.return_visualizations:
            visualizations = self._get_visualizations(execution_result)
        
        # Build response
        response = {
            "code": code,
            "execution_result": execution_result,
            "json_result": json_result,
            "visualizations": visualizations
        }
        
        # Return formatted response
        return json.dumps(response, indent=2)
    
    def _parse_json_from_output(self, output: str) -> Dict[str, Any]:
        """
        Extract JSON from output.
        
        Args:
            output (str): Output from code execution.
            
        Returns:
            Dict[str, Any]: Extracted JSON or empty dict if not found.
        """
        try:
            # Look for JSON in the output
            json_pattern = r'({[\s\S]*})'
            matches = re.findall(json_pattern, output)
            
            for match in matches:
                try:
                    # Try to parse as JSON
                    return json.loads(match)
                except:
                    continue
            
            return {}
        except Exception as e:
            logger.warning(f"Failed to parse JSON from output: {e}")
            return {}
    
    def _get_visualizations(self, output: str) -> List[Dict[str, Any]]:
        """
        Extract visualizations from output.
        
        Args:
            output (str): Output from code execution.
            
        Returns:
            List[Dict[str, Any]]: List of visualizations.
        """
        visualizations = []
        
        try:
            # Extract base64 encoded visualizations
            vis_pattern = r'VISUALIZATION:([a-zA-Z0-9_]+\.png):([^:]+)'
            matches = re.findall(vis_pattern, output)
            
            for filename, base64_data in matches:
                visualizations.append({
                    "filename": filename,
                    "type": "image/png",
                    "data": base64_data.strip()
                })
            
            return visualizations
        except Exception as e:
            logger.warning(f"Failed to extract visualizations: {e}")
            return []
    
    def _store_in_context(self, context: Dict[str, Any], result: str) -> None:
        """
        Store results in the shared context.
        
        Args:
            context (Dict[str, Any]): Shared context.
            result (str): Tool result.
        """
        try:
            # Parse result
            result_dict = json.loads(result)
            
            # Initialize context sections if they don't exist
            if "visualizations" not in context:
                context["visualizations"] = []
            
            if "code_results" not in context:
                context["code_results"] = []
            
            # Add visualizations to context
            if "visualizations" in result_dict and result_dict["visualizations"]:
                context["visualizations"].extend(result_dict["visualizations"])
            
            # Add JSON result to context
            if "json_result" in result_dict and result_dict["json_result"]:
                context["code_results"].append(result_dict["json_result"])
            
            # Add execution result to context
            if "execution_result" in result_dict:
                if "analysis_outputs" not in context:
                    context["analysis_outputs"] = []
                context["analysis_outputs"].append(result_dict["execution_result"])
            
            logger.info(f"Stored CodeGenTool results in context: {len(context['visualizations'])} visualizations, {len(context['code_results'])} results")
        except Exception as e:
            logger.error(f"Failed to store results in context: {e}")
