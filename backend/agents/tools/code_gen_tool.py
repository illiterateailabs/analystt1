"""
CodeGenTool for generating Python code using Gemini AI.

This tool leverages the GeminiClient to generate Python code for data analysis,
machine learning, visualization, and other computational tasks. It produces
secure, well-commented code that can be executed in sandboxed environments.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

from backend.integrations.gemini_client import GeminiClient

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


class CodeGenerationTool:
    """
    Tool for generating Python code using Gemini AI.
    
    This tool allows agents to request Python code generation for various
    tasks, including data analysis, machine learning, visualization, and
    data transformation. The generated code is designed to be secure,
    well-commented, and ready for execution in sandboxed environments.
    """
    
    def __init__(self, gemini_client: Optional[GeminiClient] = None):
        """
        Initialize the CodeGenerationTool.
        
        Args:
            gemini_client: Optional GeminiClient instance. If not provided,
                          a new client will be created.
        """
        self.gemini_client = gemini_client or GeminiClient()
    
    def run(self, **kwargs) -> str:
        """
        Generate Python code based on the provided parameters.
        
        Args:
            **kwargs: Keyword arguments matching CodeGenerationInput fields
                task_description: Description of the task the code should perform
                libraries: Optional list of Python libraries to use
                context: Additional context for code generation
                code_style: Preferred code style
                security_level: Security level for generated code
            
        Returns:
            JSON string containing the generated code and metadata
        """
        try:
            # Extract parameters
            task_description = kwargs.get("task_description")
            libraries = kwargs.get("libraries")
            context = kwargs.get("context")
            code_style = kwargs.get("code_style", "standard")
            security_level = kwargs.get("security_level", "high")
            
            if not task_description:
                return json.dumps({
                    "success": False,
                    "error": "Task description is required",
                    "code": "# Error: No task description provided"
                })
            
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
6. Return results in a structured format
7. Security requirements: {security_guidance}

The code should be ready to run in an isolated sandbox environment and should be self-contained.
"""
            
            # Generate the code
            code = self.gemini_client.generate_python_code(
                task_description=full_context,
                system_instruction=system_instruction,
                libraries=libraries
            )
            
            # Validate the generated code (basic security checks)
            if security_level in ["high", "paranoid"]:
                code = self._apply_security_checks(code, security_level)
            
            return json.dumps({
                "success": True,
                "code": code,
                "language": "python",
                "libraries_used": libraries or []
            })
            
        except Exception as e:
            logger.error(f"Error generating code: {e}", exc_info=True)
            return json.dumps({
                "success": False,
                "error": str(e),
                "code": "# Error generating code"
            })
    
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
    # Limit CPU time to 30 seconds
    resource.setrlimit(resource.RLIMIT_CPU, (30, 30))
    # Limit memory to 1GB
    resource.setrlimit(resource.RLIMIT_AS, (1024 * 1024 * 1024, 1024 * 1024 * 1024))

def timeout_handler(signum, frame):
    raise TimeoutError("Execution timed out")

# Set timeout for 30 seconds
signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(30)

# Apply resource limits
limit_resources()

try:
    # Original code begins here
{code.replace('\n', '\n    ')}
    # Original code ends here
except Exception as e:
    print(f"Error during execution: {{e}}")
finally:
    # Reset alarm
    signal.alarm(0)
"""
        
        return code
