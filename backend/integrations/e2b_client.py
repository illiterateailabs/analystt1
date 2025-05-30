"""e2b.dev client for secure code execution."""

import logging
from typing import Dict, List, Optional, Any, Union
import asyncio
import json
import tempfile
import os

from e2b import Sandbox

from backend.config import E2BConfig


logger = logging.getLogger(__name__)


class E2BClient:
    """Client for e2b.dev secure code execution environment."""
    
    def __init__(self):
        """Initialize the e2b client."""
        self.config = E2BConfig()
        self._active_sandboxes: Dict[str, Sandbox] = {}
        
        logger.info("e2b client initialized")
    
    async def create_sandbox(
        self,
        template_id: Optional[str] = None,
        timeout: Optional[int] = None
    ) -> str:
        """Create a new sandbox and return its ID."""
        try:
            sandbox = Sandbox(
                template=template_id or self.config.TEMPLATE_ID,
                api_key=self.config.API_KEY,
                timeout=timeout or self.config.TIMEOUT
            )
            
            sandbox_id = sandbox.id
            self._active_sandboxes[sandbox_id] = sandbox
            
            logger.info(f"Created sandbox: {sandbox_id}")
            return sandbox_id
            
        except Exception as e:
            logger.error(f"Error creating sandbox: {e}")
            raise
    
    async def close_sandbox(self, sandbox_id: str) -> None:
        """Close and cleanup a sandbox."""
        try:
            if sandbox_id in self._active_sandboxes:
                sandbox = self._active_sandboxes[sandbox_id]
                sandbox.close()
                del self._active_sandboxes[sandbox_id]
                logger.info(f"Closed sandbox: {sandbox_id}")
            else:
                logger.warning(f"Sandbox {sandbox_id} not found in active sandboxes")
                
        except Exception as e:
            logger.error(f"Error closing sandbox {sandbox_id}: {e}")
    
    async def execute_python_code(
        self,
        code: str,
        sandbox_id: Optional[str] = None,
        timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """Execute Python code in a sandbox."""
        try:
            # Create sandbox if not provided
            if sandbox_id is None:
                sandbox_id = await self.create_sandbox(timeout=timeout)
                auto_cleanup = True
            else:
                auto_cleanup = False
            
            if sandbox_id not in self._active_sandboxes:
                raise ValueError(f"Sandbox {sandbox_id} not found")
            
            sandbox = self._active_sandboxes[sandbox_id]
            
            # Execute the code
            execution = sandbox.run_code(
                language="python",
                code=code
            )
            
            result = {
                "sandbox_id": sandbox_id,
                "stdout": execution.stdout,
                "stderr": execution.stderr,
                "exit_code": execution.exit_code,
                "execution_time": execution.execution_time,
                "success": execution.exit_code == 0
            }
            
            # Auto-cleanup if we created the sandbox
            if auto_cleanup:
                await self.close_sandbox(sandbox_id)
            
            logger.debug(f"Code execution completed in sandbox {sandbox_id}")
            return result
            
        except Exception as e:
            logger.error(f"Error executing Python code: {e}")
            if sandbox_id and auto_cleanup:
                await self.close_sandbox(sandbox_id)
            raise
    
    async def install_packages(
        self,
        packages: List[str],
        sandbox_id: str
    ) -> Dict[str, Any]:
        """Install Python packages in a sandbox."""
        try:
            if sandbox_id not in self._active_sandboxes:
                raise ValueError(f"Sandbox {sandbox_id} not found")
            
            # Create pip install command
            install_code = f"""
import subprocess
import sys

packages = {packages}
for package in packages:
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"Successfully installed {{package}}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to install {{package}}: {{e}}")
"""
            
            result = await self.execute_python_code(install_code, sandbox_id)
            logger.info(f"Package installation completed in sandbox {sandbox_id}")
            return result
            
        except Exception as e:
            logger.error(f"Error installing packages: {e}")
            raise
    
    async def upload_file(
        self,
        sandbox_id: str,
        file_content: Union[str, bytes],
        file_path: str
    ) -> bool:
        """Upload a file to the sandbox."""
        try:
            if sandbox_id not in self._active_sandboxes:
                raise ValueError(f"Sandbox {sandbox_id} not found")
            
            sandbox = self._active_sandboxes[sandbox_id]
            
            # Write file content
            if isinstance(file_content, str):
                sandbox.filesystem.write(file_path, file_content)
            else:
                # For bytes, write to a temporary file first
                with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                    tmp_file.write(file_content)
                    tmp_file.flush()
                    
                    # Read and upload
                    with open(tmp_file.name, 'rb') as f:
                        content = f.read()
                    sandbox.filesystem.write(file_path, content.decode('utf-8'))
                    
                    # Cleanup
                    os.unlink(tmp_file.name)
            
            logger.debug(f"Uploaded file {file_path} to sandbox {sandbox_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error uploading file: {e}")
            return False
    
    async def download_file(
        self,
        sandbox_id: str,
        file_path: str
    ) -> Optional[str]:
        """Download a file from the sandbox."""
        try:
            if sandbox_id not in self._active_sandboxes:
                raise ValueError(f"Sandbox {sandbox_id} not found")
            
            sandbox = self._active_sandboxes[sandbox_id]
            content = sandbox.filesystem.read(file_path)
            
            logger.debug(f"Downloaded file {file_path} from sandbox {sandbox_id}")
            return content
            
        except Exception as e:
            logger.error(f"Error downloading file: {e}")
            return None
    
    async def list_files(
        self,
        sandbox_id: str,
        directory: str = "."
    ) -> List[str]:
        """List files in a sandbox directory."""
        try:
            if sandbox_id not in self._active_sandboxes:
                raise ValueError(f"Sandbox {sandbox_id} not found")
            
            sandbox = self._active_sandboxes[sandbox_id]
            
            # Use ls command to list files
            execution = sandbox.run_code(
                language="bash",
                code=f"ls -la {directory}"
            )
            
            if execution.exit_code == 0:
                files = execution.stdout.strip().split('\n')
                return files
            else:
                logger.error(f"Error listing files: {execution.stderr}")
                return []
                
        except Exception as e:
            logger.error(f"Error listing files: {e}")
            return []
    
    async def execute_neo4j_script(
        self,
        script: str,
        neo4j_config: Dict[str, str],
        sandbox_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Execute a script that interacts with Neo4j."""
        try:
            # Prepare the script with Neo4j connection
            full_script = f"""
import os
from neo4j import GraphDatabase

# Neo4j configuration
NEO4J_URI = "{neo4j_config.get('uri', 'bolt://localhost:7687')}"
NEO4J_USERNAME = "{neo4j_config.get('username', 'neo4j')}"
NEO4J_PASSWORD = "{neo4j_config.get('password', '')}"

# Create driver
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

try:
    with driver.session() as session:
        # User script starts here
{script}
        
finally:
    driver.close()
"""
            
            result = await self.execute_python_code(full_script, sandbox_id)
            return result
            
        except Exception as e:
            logger.error(f"Error executing Neo4j script: {e}")
            raise
    
    async def cleanup_all_sandboxes(self) -> None:
        """Close all active sandboxes."""
        sandbox_ids = list(self._active_sandboxes.keys())
        for sandbox_id in sandbox_ids:
            await self.close_sandbox(sandbox_id)
        
        logger.info(f"Cleaned up {len(sandbox_ids)} sandboxes")
    
    def get_active_sandboxes(self) -> List[str]:
        """Get list of active sandbox IDs."""
        return list(self._active_sandboxes.keys())
