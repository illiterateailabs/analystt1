"""
CodeGenTool for generating and executing Python code.

This tool uses Gemini to generate Python code from natural language questions,
executes the code in a secure sandbox, and returns structured results that
can be used by subsequent agents in the crew.
"""

import json
import logging
import base64
import re
from typing import Any, Dict, List, Optional, Union

from crewai_tools import BaseTool
from pydantic import BaseModel, Field

from backend.integrations.gemini_client import GeminiClient
from backend.integrations.e2b_client import E2BClient
from backend.agents.tools.sandbox_exec_tool import SandboxExecTool

logger = logging.getLogger(__name__)


class CodeGenInput(BaseModel):
    """Input model for code generation and execution."""
    
    question: str = Field(
        ...,
        description="Natural language question or task description for code generation"
    )
    context: Optional[str] = Field(
        default=None,
        description="Additional context for code generation (e.g., data description, requirements)"
    )
    execute_code: bool = Field(
        default=True,
        description="Whether to execute the generated code"
    )
    install_packages: Optional[List[str]] = Field(
        default=None,
        description="List of Python packages to install before execution"
    )
    timeout_seconds: Optional[int] = Field(
        default=60,
        description="Maximum execution time in seconds"
    )
    return_visualizations: bool = Field(
        default=True,
        description="Whether to return generated visualizations"
    )


class CodeGenTool(BaseTool):
    """
    Tool for generating and executing Python code from natural language.
    
    This tool uses Gemini to generate Python code based on natural language
    questions, executes the code in a secure sandbox environment, and returns
    structured results that can be used by subsequent agents in the crew.
    
    Key features:
    - Generate Python code using Gemini API
    - Execute code in E2B sandbox
    - Collect and encode artifacts (images, CSVs, etc.)
    - Return structured result that agents can use
    - Parse JSON outputs from code execution
    - Handle visualization files and data exports
    """
    
    name: str = "code_gen_tool"
    description: str = """
    Generate and execute Python code based on natural language questions.
    
    Use this tool when you need to:
    - Analyze data or generate statistics
    - Create visualizations or charts
    - Process and transform data
    - Perform calculations or simulations
    - Generate reports with embedded visualizations
    
    The tool will:
    1. Generate appropriate Python code based on your question
    2. Execute the code in a secure sandbox
    3. Return the results, including any visualizations
    4. Parse JSON output for structured data
    
    Example usage:
    - "Generate a histogram of transaction amounts"
    - "Calculate the average transaction value per day"
    - "Find outliers in the transaction data using IQR"
    - "Create a network graph visualization of connected accounts"
    """
    args_schema: type[BaseModel] = CodeGenInput
    
    def __init__(
        self,
        gemini_client: Optional[GeminiClient] = None,
        e2b_client: Optional[E2BClient] = None
    ):
        """
        Initialize the CodeGenTool.
        
        Args:
            gemini_client: Optional GeminiClient instance for code generation
            e2b_client: Optional E2BClient instance for code execution
        """
        super().__init__()
        self.gemini_client = gemini_client or GeminiClient()
        self.e2b_client = e2b_client or E2BClient()
        self.sandbox_tool = SandboxExecTool(e2b_client=self.e2b_client)
    
    async def _arun(
        self,
        question: str,
        context: Optional[str] = None,
        execute_code: bool = True,
        install_packages: Optional[List[str]] = None,
        timeout_seconds: int = 60,
        return_visualizations: bool = True
    ) -> Dict[str, Any]:
        """
        Generate and execute Python code based on a natural language question.
        
        Args:
            question: Natural language question or task description
            context: Additional context for code generation
            execute_code: Whether to execute the generated code
            install_packages: Optional list of packages to install
            timeout_seconds: Maximum execution time in seconds
            return_visualizations: Whether to return generated visualizations
            
        Returns:
            Dictionary containing generated code, execution results, and visualizations
        """
        try:
            logger.info(f"Generating code for question: {question}")
            
            # Prepare prompt for code generation
            prompt = self._prepare_prompt(question, context)
            
            # Generate code using Gemini
            code = await self.gemini_client.generate_python_code(prompt)
            if not code:
                logger.warning("Failed to generate code")
                return {
                    "success": False,
                    "error": "Failed to generate code",
                    "code": None
                }
            
            logger.info("Successfully generated code")
            
            # Initialize result structure
            result = {
                "success": True,
                "code": code,
                "execution": None,
                "result": None,
                "visualizations": []
            }
            
            # Execute code if requested
            if execute_code:
                logger.info("Executing generated code")
                execution_result = await self._execute_code(
                    code,
                    install_packages,
                    timeout_seconds,
                    return_visualizations
                )
                
                # Add execution results to the result
                result["execution"] = execution_result
                
                # Try to parse JSON output from stdout
                parsed_result = self._parse_json_from_output(execution_result["stdout"])
                if parsed_result:
                    result["result"] = parsed_result
                
                # Add visualizations if available and requested
                if return_visualizations and execution_result["success"]:
                    visualizations = await self._get_visualizations()
                    if visualizations:
                        result["visualizations"] = visualizations
            
            return result
            
        except Exception as e:
            logger.error(f"Error in CodeGenTool: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "code": code if 'code' in locals() else None
            }
    
    def _run(
        self,
        question: str,
        context: Optional[str] = None,
        execute_code: bool = True,
        install_packages: Optional[List[str]] = None,
        timeout_seconds: int = 60,
        return_visualizations: bool = True
    ) -> Dict[str, Any]:
        """
        Synchronous wrapper for _arun.
        
        This method exists for compatibility with synchronous CrewAI operations.
        It should not be called directly in an async context.
        """
        import asyncio
        
        # Create a new event loop if needed
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(
            self._arun(
                question,
                context,
                execute_code,
                install_packages,
                timeout_seconds,
                return_visualizations
            )
        )
    
    def _prepare_prompt(self, question: str, context: Optional[str] = None) -> str:
        """
        Prepare a prompt for code generation.
        
        Args:
            question: Natural language question or task
            context: Additional context for code generation
            
        Returns:
            Formatted prompt for code generation
        """
        prompt = f"""
        Generate Python code to answer the following question or perform the task:
        
        QUESTION: {question}
        
        """
        
        if context:
            prompt += f"""
            ADDITIONAL CONTEXT:
            {context}
            
            """
        
        prompt += """
        REQUIREMENTS:
        1. Use standard data science libraries (pandas, numpy, matplotlib, seaborn, etc.)
        2. Include appropriate error handling
        3. Generate visualizations when appropriate
        4. Save visualizations as PNG files (e.g., "plot.png", "chart.png")
        5. For any significant results or data, print them as JSON to stdout
        6. Make the code self-contained and executable
        7. Include comments explaining key steps
        
        Return only the Python code without any additional text or explanations.
        """
        
        return prompt
    
    async def _execute_code(
        self,
        code: str,
        install_packages: Optional[List[str]] = None,
        timeout_seconds: int = 60,
        return_visualizations: bool = True
    ) -> Dict[str, Any]:
        """
        Execute generated code in a sandbox environment.
        
        Args:
            code: Python code to execute
            install_packages: Optional list of packages to install
            timeout_seconds: Maximum execution time in seconds
            return_visualizations: Whether to look for generated visualizations
            
        Returns:
            Dictionary containing execution results
        """
        # Prepare standard visualization packages if needed
        if return_visualizations and install_packages is None:
            install_packages = ["matplotlib", "seaborn", "plotly"]
        elif return_visualizations and install_packages is not None:
            for pkg in ["matplotlib", "seaborn", "plotly"]:
                if pkg not in install_packages:
                    install_packages.append(pkg)
        
        # Execute code using SandboxExecTool
        sandbox_result_str = await self.sandbox_tool._arun(
            code=code,
            install_packages=install_packages,
            timeout_seconds=timeout_seconds,
            return_files=["*.png", "*.jpg", "*.jpeg", "*.csv", "*.json"] if return_visualizations else None
        )
        
        # Parse the JSON result
        try:
            sandbox_result = json.loads(sandbox_result_str)
            return {
                "success": sandbox_result.get("success", False),
                "stdout": sandbox_result.get("stdout", ""),
                "stderr": sandbox_result.get("stderr", ""),
                "exit_code": sandbox_result.get("exit_code", 1),
                "output_files": sandbox_result.get("output_files", {})
            }
        except json.JSONDecodeError:
            logger.error(f"Failed to parse sandbox result: {sandbox_result_str}")
            return {
                "success": False,
                "stdout": "",
                "stderr": f"Failed to parse sandbox result: {sandbox_result_str}",
                "exit_code": 1,
                "output_files": {}
            }
    
    async def _get_visualizations(self) -> List[Dict[str, str]]:
        """
        Extract visualizations from sandbox output files.
        
        Returns:
            List of dictionaries containing filename and base64-encoded content
        """
        visualizations = []
        
        try:
            # List files in the sandbox
            sandbox = self.sandbox_tool._active_sandbox
            if not sandbox:
                return []
            
            # Get all files with image extensions
            files = await self.e2b_client.list_files(sandbox)
            image_files = [f for f in files if f.endswith((".png", ".jpg", ".jpeg"))]
            
            # Download and encode each file
            for filename in image_files:
                try:
                    file_content = await self.e2b_client.download_file(filename, sandbox)
                    encoded_content = base64.b64encode(file_content).decode("utf-8")
                    visualizations.append({
                        "filename": filename,
                        "content": encoded_content
                    })
                except Exception as e:
                    logger.warning(f"Failed to retrieve file {filename}: {e}")
            
            return visualizations
            
        except Exception as e:
            logger.error(f"Error getting visualizations: {e}")
            return []
    
    def _parse_json_from_output(self, stdout: str) -> Optional[Dict[str, Any]]:
        """
        Parse JSON data from stdout.
        
        Args:
            stdout: Standard output from code execution
            
        Returns:
            Parsed JSON object if found, None otherwise
        """
        if not stdout:
            return None
        
        try:
            # Look for JSON objects in the output
            json_pattern = r'(\{.*\}|\[.*\])'
            json_matches = re.findall(json_pattern, stdout, re.DOTALL)
            
            for match in json_matches:
                try:
                    return json.loads(match)
                except json.JSONDecodeError:
                    continue
            
            return None
            
        except Exception as e:
            logger.warning(f"Error parsing JSON from output: {e}")
            return None
    
    async def close(self):
        """Close the sandbox tool and release resources."""
        await self.sandbox_tool.close()
