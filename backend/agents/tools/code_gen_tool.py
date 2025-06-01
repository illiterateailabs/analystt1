"""
CodeGenTool for generating and executing Python code using Gemini AI.

This tool leverages the GeminiClient to generate Python code for data analysis,
machine learning, visualization, and other computational tasks. It produces
secure, well-commented code that can be executed in sandboxed environments.
"""

import json
import logging
import base64
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

from backend.integrations.gemini_client import GeminiClient
from backend.integrations.e2b_client import E2BClient

logger = logging.getLogger(__name__)


class CodeGenerationInput(BaseModel):
    """Input model for code generation."""
    
    task_description: str = Field(
        ...,
        description="Detailed description of the task the code should perform"
    )
    libraries: Optional[List[str]] = Field(
        default=None,
        description="List of Python libraries that can be used in the solution"
    )
    context: Optional[str] = Field(
        default=None,
        description="Additional context, such as data schema, sample data, or constraints"
    )
    code_style: Optional[str] = Field(
        default="standard",
        description="Preferred code style (e.g., 'standard', 'functional', 'object-oriented')"
    )
    security_level: Optional[str] = Field(
        default="high",
        description="Security level for generated code ('standard', 'high', 'paranoid')"
    )
    execute_code: Optional[bool] = Field(
        default=False,
        description="Whether to execute the generated code in a sandbox"
    )
    timeout_seconds: Optional[int] = Field(
        default=60,
        description="Maximum execution time in seconds (only used if execute_code is True)"
    )
    input_files: Optional[Dict[str, str]] = Field(
        default=None,
        description="Dictionary of filename:content pairs to create before execution"
    )
    return_files: Optional[List[str]] = Field(
        default=None,
        description="List of filenames to return after execution"
    )


class CodeGenTool:
    """
    Tool for generating and executing Python code using Gemini AI.
    
    This tool allows agents to request Python code generation for various
    tasks, including data analysis, machine learning, visualization, and
    data transformation. The generated code is designed to be secure,
    well-commented, and ready for execution in sandboxed environments.
    
    The tool can also optionally execute the generated code in a secure
    sandbox environment and return the execution results, which can be
    used by subsequent agents in a crew workflow.
    """
    
    def __init__(self, gemini_client: Optional[GeminiClient] = None, e2b_client: Optional[E2BClient] = None):
        """
        Initialize the CodeGenTool.
        
        Args:
            gemini_client: Optional GeminiClient instance. If not provided,
                          a new client will be created.
            e2b_client: Optional E2BClient instance for code execution.
                       If not provided, a new client will be created when needed.
        """
        self.gemini_client = gemini_client or GeminiClient()
        self.e2b_client = e2b_client
        self._active_sandbox = None
    
    async def _execute_in_sandbox(
        self,
        code: str,
        libraries: Optional[List[str]] = None,
        timeout_seconds: int = 60,
        input_files: Optional[Dict[str, str]] = None,
        return_files: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Execute the generated code in a sandbox environment.
        
        Args:
            code: Python code to execute
            libraries: Optional list of libraries to install
            timeout_seconds: Maximum execution time in seconds
            input_files: Dictionary of filename:content pairs to create
            return_files: List of filenames to return after execution
            
        Returns:
            Dictionary containing execution results and any output files
        """
        try:
            # Initialize e2b_client if not provided
            if not self.e2b_client:
                self.e2b_client = E2BClient()
            
            # Create or reuse sandbox
            if not self._active_sandbox:
                logger.info("Creating new e2b sandbox")
                self._active_sandbox = await self.e2b_client.create_sandbox()
            
            sandbox_id = self._active_sandbox
            
            # Install packages if requested
            if libraries and len(libraries) > 0:
                logger.info(f"Installing packages: {', '.join(libraries)}")
                for package in libraries:
                    # Install the package
                    await self.e2b_client.install_package(package, sandbox_id)
            
            # Create input files if provided
            if input_files:
                logger.info(f"Creating {len(input_files)} input files")
                for filename, content in input_files.items():
                    await self.e2b_client.upload_file(
                        content.encode('utf-8'), 
                        filename, 
                        sandbox_id
                    )
            
            # Execute the code with timeout
            logger.info(f"Executing code in sandbox (timeout: {timeout_seconds}s)")
            result = await self.e2b_client.execute_code(
                code, 
                sandbox=sandbox_id,
                timeout=timeout_seconds
            )
            
            # Parse the output to extract any structured results
            parsed_result = None
            try:
                if result["success"] and result["stdout"]:
                    # Try to parse JSON output if present
                    import re
                    json_match = re.search(r'```json\n(.*?)\n```', result["stdout"], re.DOTALL)
                    if json_match:
                        parsed_result = json.loads(json_match.group(1))
                    else:
                        # Look for JSON at the end of output
                        try:
                            parsed_result = json.loads(result["stdout"].strip())
                        except:
                            pass
            except Exception as e:
                logger.warning(f"Failed to parse JSON from output: {e}")
            
            # Collect output files if requested
            output_files = {}
            if return_files and result["success"]:
                logger.info(f"Retrieving {len(return_files)} output files")
                for filename in return_files:
                    try:
                        file_content = await self.e2b_client.download_file(filename, sandbox_id)
                        # Convert binary content to base64 for serialization
                        output_files[filename] = base64.b64encode(file_content).decode('utf-8')
                    except Exception as e:
                        logger.warning(f"Failed to retrieve file {filename}: {e}")
                        output_files[filename] = None
            
            # Check for generated visualizations
            visualizations = []
            if result["success"]:
                try:
                    files = await self.e2b_client.list_files(sandbox_id)
                    for file in files:
                        if any(ext in file for ext in ['.png', '.jpg', '.svg', '.html']):
                            try:
                                file_content = await self.e2b_client.download_file(file, sandbox_id)
                                visualizations.append({
                                    "filename": file,
                                    "content": base64.b64encode(file_content).decode('utf-8')
                                })
                            except Exception as e:
                                logger.warning(f"Failed to retrieve visualization {file}: {e}")
                except Exception as e:
                    logger.warning(f"Failed to list files: {e}")
            
            # Prepare response
            return {
                "success": result["success"],
                "stdout": result["stdout"],
                "stderr": result["stderr"],
                "exit_code": result["exit_code"],
                "execution_time": result.get("execution_time", 0),
                "result": parsed_result,
                "output_files": output_files,
                "visualizations": visualizations
            }
            
        except Exception as e:
            logger.error(f"Error executing code in sandbox: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "stdout": "",
                "stderr": f"Internal error: {str(e)}"
            }
    
    async def run(self, **kwargs) -> Dict[str, Any]:
        """
        Generate and optionally execute Python code based on the provided parameters.
        
        Args:
            **kwargs: Keyword arguments matching CodeGenerationInput fields
                task_description: Description of the task the code should perform
                libraries: Optional list of Python libraries to use
                context: Additional context for code generation
                code_style: Preferred code style
                security_level: Security level for generated code
                execute_code: Whether to execute the generated code
                timeout_seconds: Maximum execution time in seconds
                input_files: Dictionary of filename:content pairs to create
                return_files: List of filenames to return after execution
            
        Returns:
            Dictionary containing the generated code, execution results (if requested),
            and metadata that can be integrated into crew context
        """
        try:
            # Extract parameters
            task_description = kwargs.get("task_description") or kwargs.get("question")
            libraries = kwargs.get("libraries")
            context = kwargs.get("context")
            code_style = kwargs.get("code_style", "standard")
            security_level = kwargs.get("security_level", "high")
            execute_code = kwargs.get("execute_code", False)
            timeout_seconds = kwargs.get("timeout_seconds", 60)
            input_files = kwargs.get("input_files")
            return_files = kwargs.get("return_files")
            
            if not task_description:
                return {
                    "success": False,
                    "error": "Task description is required",
                    "code": "# Error: No task description provided"
                }
            
            # Prepare the full context for code generation
            full_context = f"""
Task: {task_description}

"""
            
            if context:
                full_context += f"Context: {context}\n\n"
            
            if libraries:
                full_context += f"Available libraries: {', '.join(libraries)}\n\n"
            
            # Add code style guidance
            if code_style == "functional":
                full_context += "Please use a functional programming style with pure functions and minimal state.\n\n"
            elif code_style == "object-oriented":
                full_context += "Please use an object-oriented programming style with appropriate classes and methods.\n\n"
            
            # Add security requirements based on security level
            security_guidance = ""
            if security_level == "high":
                security_guidance = """
- Validate all inputs before processing
- Use safe methods for file operations
- Avoid eval(), exec(), and other unsafe functions
- Handle exceptions properly with specific exception types
- Sanitize any data that might be used in queries or commands
"""
            elif security_level == "paranoid":
                security_guidance = """
- Implement strict input validation with explicit type checking
- Use allowlists instead of blocklists for validation
- Avoid all potentially unsafe functions (eval, exec, os.system, subprocess, etc.)
- Implement resource limits (memory, CPU time) where possible
- Use context managers for all resource handling
- Sanitize all external data with explicit encoding/escaping
- Add runtime assertions to verify invariants
"""
            
            # Build the final prompt
            system_instruction = f"""
You are an expert Python developer specializing in secure, efficient code for data analysis and machine learning.
Generate Python code that accomplishes the specified task while following these guidelines:

1. Write well-commented, readable code
2. Include proper error handling
3. Follow {code_style} programming style
4. Include necessary imports at the beginning
5. Use efficient algorithms and data structures
6. Return results in a structured format (JSON if possible)
7. Security requirements: {security_guidance}

The code should be ready to run in an isolated sandbox environment and should be self-contained.
If the code generates any visualizations, save them as files (e.g., 'plot.png', 'chart.html').
If possible, structure your final results as JSON and print them at the end of execution.
"""
            
            # Generate the code
            code = await self.gemini_client.generate_python_code(
                task_description=full_context,
                system_instruction=system_instruction,
                libraries=libraries
            )
            
            # Validate the generated code (basic security checks)
            if security_level in ["high", "paranoid"]:
                code = self._apply_security_checks(code, security_level)
            
            # Prepare the response with generated code
            response = {
                "success": True,
                "code": code,
                "language": "python",
                "libraries_used": libraries or []
            }
            
            # Execute the code if requested
            if execute_code:
                import asyncio
                execution_result = await self._execute_in_sandbox(
                    code=code,
                    libraries=libraries,
                    timeout_seconds=timeout_seconds,
                    input_files=input_files,
                    return_files=return_files
                )
                
                # Merge execution results into response
                response.update({
                    "execution": {
                        "success": execution_result["success"],
                        "stdout": execution_result["stdout"],
                        "stderr": execution_result["stderr"],
                        "exit_code": execution_result["exit_code"],
                        "execution_time": execution_result.get("execution_time", 0)
                    },
                    "result": execution_result.get("result"),
                    "output_files": execution_result.get("output_files", {}),
                    "visualizations": execution_result.get("visualizations", [])
                })
                
                # Update overall success based on execution
                if not execution_result["success"]:
                    response["error"] = execution_result.get("error") or "Execution failed"
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating/executing code: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "code": "# Error generating code"
            }
    
    def _run(self, **kwargs) -> Dict[str, Any]:
        """
        Synchronous wrapper for run.
        
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
        
        return loop.run_until_complete(self.run(**kwargs))
    
    def _apply_security_checks(self, code: str, security_level: str) -> str:
        """
        Apply security checks to the generated code.
        
        Args:
            code: The generated Python code
            security_level: The security level to apply
            
        Returns:
            Potentially modified code with security enhancements
        """
        # List of dangerous functions to check for
        dangerous_functions = [
            "eval(", "exec(", "os.system(", "subprocess.call(", 
            "subprocess.Popen(", "subprocess.run(", "subprocess.check_output(",
            "__import__(", "globals()", "locals()"
        ]
        
        # Check for dangerous functions
        for func in dangerous_functions:
            if func in code:
                # Add warning comment
                code = f"# WARNING: This code contains potentially unsafe function: {func}\n" + code
                
                if security_level == "paranoid":
                    # Replace the dangerous function with a safer alternative or comment it out
                    code = code.replace(func, f"# SECURITY RISK REMOVED: {func}")
        
        # Add security wrapper for paranoid level
        if security_level == "paranoid":
            # Add resource limits and other security measures
            code = f"""
# Security wrapper with resource limits
import resource
import sys
import signal

def limit_resources():
    # Set CPU time limit (seconds)
    resource.setrlimit(resource.RLIMIT_CPU, (30, 30))
    # Set memory limit (bytes)
    resource.setrlimit(resource.RLIMIT_AS, (1024 * 1024 * 1024, 1024 * 1024 * 1024))  # 1GB

def timeout_handler(signum, frame):
    raise TimeoutError("Execution timed out")

# Set timeout
signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(25)  # 25 seconds timeout

try:
    limit_resources()
    
    # Original code below
{code.replace('\n', '\n    ')}

except Exception as e:
    print(f"Error: {{e}}")
    sys.exit(1)
finally:
    # Reset alarm
    signal.alarm(0)
"""
        
        return code
    
    async def close(self):
        """Close the active sandbox if one exists."""
        if self._active_sandbox and self.e2b_client:
            try:
                await self.e2b_client.close_sandbox(self._active_sandbox)
                self._active_sandbox = None
                logger.info("Closed e2b sandbox")
            except Exception as e:
                logger.error(f"Error closing sandbox: {e}")
