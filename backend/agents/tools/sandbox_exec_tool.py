"""
SandboxExecTool for executing Python code in isolated e2b sandboxes.

This tool wraps the E2BClient to allow CrewAI agents to execute
Python code in secure, isolated environments, install libraries,
manage files, and retrieve execution results.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional, Union, BinaryIO

from crewai_tools import BaseTool
from pydantic import BaseModel, Field

from backend.integrations.e2b_client import E2BClient

logger = logging.getLogger(__name__)


class SandboxCodeInput(BaseModel):
    """Input model for sandbox code execution."""
    
    code: str = Field(
        ...,
        description="Python code to execute in the sandbox"
    )
    install_packages: Optional[List[str]] = Field(
        default=None,
        description="List of Python packages to install before execution"
    )
    timeout_seconds: Optional[int] = Field(
        default=60,
        description="Maximum execution time in seconds"
    )
    input_files: Optional[Dict[str, str]] = Field(
        default=None,
        description="Dictionary of filename:content pairs to create before execution"
    )
    return_files: Optional[List[str]] = Field(
        default=None,
        description="List of filenames to return after execution"
    )


class SandboxExecTool(BaseTool):
    """
    Tool for executing Python code in isolated e2b sandboxes.
    
    This tool allows agents to run Python code in secure, isolated
    environments, install libraries on demand, manage files, and
    retrieve execution results safely.
    """
    
    name: str = "sandbox_exec_tool"
    description: str = """
    Execute Python code in a secure, isolated sandbox environment.
    
    Use this tool when you need to:
    - Run data analysis or processing code
    - Generate visualizations or reports
    - Execute machine learning algorithms
    - Process and transform data
    - Install and use Python libraries
    
    The sandbox provides a secure execution environment with:
    - Python 3.10+ and common data science libraries
    - Ability to install additional packages on-the-fly
    - File upload/download capabilities
    - Isolated execution from the main system
    
    Example usage:
    - Execute pandas code to analyze financial transactions
    - Run graph algorithms on network data
    - Generate matplotlib or plotly visualizations
    - Train and evaluate machine learning models
    """
    args_schema: type[BaseModel] = SandboxCodeInput
    
    def __init__(self, e2b_client: Optional[E2BClient] = None):
        """
        Initialize the SandboxExecTool.
        
        Args:
            e2b_client: Optional E2BClient instance. If not provided,
                       a new client will be created.
        """
        super().__init__()
        self.e2b_client = e2b_client or E2BClient()
        self._active_sandbox = None
    
    async def _arun(
        self,
        code: str,
        install_packages: Optional[List[str]] = None,
        timeout_seconds: int = 60,
        input_files: Optional[Dict[str, str]] = None,
        return_files: Optional[List[str]] = None
    ) -> str:
        """
        Execute Python code in an e2b sandbox asynchronously.
        
        Args:
            code: Python code to execute
            install_packages: Optional list of packages to install
            timeout_seconds: Maximum execution time in seconds
            input_files: Dictionary of filename:content pairs to create
            return_files: List of filenames to return after execution
            
        Returns:
            JSON string containing execution results and any output files
        """
        try:
            # Create or reuse sandbox
            if not self._active_sandbox:
                logger.info("Creating new e2b sandbox")
                self._active_sandbox = await self.e2b_client.create_sandbox()
            
            sandbox = self._active_sandbox
            
            # Install packages if requested
            if install_packages and len(install_packages) > 0:
                logger.info(f"Installing packages: {', '.join(install_packages)}")
                for package in install_packages:
                    # Security check - basic validation of package name
                    if not self._is_valid_package_name(package):
                        return json.dumps({
                            "success": False,
                            "error": f"Invalid package name: {package}",
                            "stdout": "",
                            "stderr": f"Security error: Invalid package name format: {package}"
                        })
                    
                    # Install the package
                    install_result = await self.e2b_client.install_package(package, sandbox)
                    if not install_result["success"]:
                        logger.warning(f"Failed to install package {package}: {install_result['stderr']}")
                        return json.dumps({
                            "success": False,
                            "error": f"Package installation failed: {package}",
                            "stdout": install_result["stdout"],
                            "stderr": install_result["stderr"]
                        })
            
            # Create input files if provided
            if input_files:
                logger.info(f"Creating {len(input_files)} input files")
                for filename, content in input_files.items():
                    # Security check - prevent path traversal
                    if not self._is_safe_filename(filename):
                        return json.dumps({
                            "success": False,
                            "error": f"Invalid filename: {filename}",
                            "stdout": "",
                            "stderr": f"Security error: Path traversal attempt detected: {filename}"
                        })
                    
                    # Write the file
                    await self.e2b_client.upload_file(
                        content.encode('utf-8'), 
                        filename, 
                        sandbox
                    )
            
            # Execute the code with timeout
            logger.info(f"Executing code in sandbox (timeout: {timeout_seconds}s)")
            result = await self.e2b_client.execute_code(
                code, 
                sandbox=sandbox,
                timeout=timeout_seconds
            )
            
            # Collect output files if requested
            output_files = {}
            if return_files and result["success"]:
                logger.info(f"Retrieving {len(return_files)} output files")
                for filename in return_files:
                    # Security check - prevent path traversal
                    if not self._is_safe_filename(filename):
                        logger.warning(f"Skipping unsafe filename: {filename}")
                        continue
                    
                    try:
                        file_content = await self.e2b_client.download_file(filename, sandbox)
                        # Convert binary content to base64 for JSON serialization
                        import base64
                        output_files[filename] = base64.b64encode(file_content).decode('utf-8')
                    except Exception as e:
                        logger.warning(f"Failed to retrieve file {filename}: {e}")
                        output_files[filename] = None
            
            # Prepare response
            response = {
                "success": result["success"],
                "stdout": result["stdout"],
                "stderr": result["stderr"],
                "exit_code": result["exit_code"],
                "output_files": output_files
            }
            
            return json.dumps(response)
            
        except Exception as e:
            logger.error(f"Error executing code in sandbox: {e}", exc_info=True)
            return json.dumps({
                "success": False,
                "error": str(e),
                "stdout": "",
                "stderr": f"Internal error: {str(e)}"
            })
    
    def _run(
        self,
        code: str,
        install_packages: Optional[List[str]] = None,
        timeout_seconds: int = 60,
        input_files: Optional[Dict[str, str]] = None,
        return_files: Optional[List[str]] = None
    ) -> str:
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
            self._arun(code, install_packages, timeout_seconds, input_files, return_files)
        )
    
    async def close(self):
        """Close the active sandbox if one exists."""
        if self._active_sandbox:
            try:
                await self.e2b_client.close_sandbox(self._active_sandbox)
                self._active_sandbox = None
                logger.info("Closed e2b sandbox")
            except Exception as e:
                logger.error(f"Error closing sandbox: {e}")
    
    def _is_valid_package_name(self, package_name: str) -> bool:
        """
        Validate a Python package name for security.
        
        Args:
            package_name: The package name to validate
            
        Returns:
            True if the package name is valid, False otherwise
        """
        # Basic validation - alphanumeric plus some special chars
        import re
        # PEP 508 compliant package name validation (simplified)
        pattern = r'^([A-Za-z0-9]|[A-Za-z0-9][A-Za-z0-9._-]*[A-Za-z0-9])([<>=!~]=?[A-Za-z0-9._-]+)?$'
        return bool(re.match(pattern, package_name))
    
    def _is_safe_filename(self, filename: str) -> bool:
        """
        Check if a filename is safe (no path traversal).
        
        Args:
            filename: The filename to check
            
        Returns:
            True if the filename is safe, False otherwise
        """
        # Prevent path traversal
        norm_path = os.path.normpath(filename)
        return not norm_path.startswith('..') and '/../' not in norm_path
