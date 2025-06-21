"""
Tools API Endpoints

This module provides API endpoints for tool discovery, metadata, and execution:
- Auto-discovery of tools in the agents/tools directory
- Tool metadata, schemas, and capabilities
- Tool health checks and status monitoring
- Tool filtering and categorization
- Tool execution endpoints
- MCP (Model Context Protocol) integration

Tools are discovered at startup and exposed via REST API endpoints,
with support for filtering, health checks, and direct execution.
"""

import importlib
import inspect
import logging
import os
import sys
import time
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Type, Union, cast

from fastapi import APIRouter, Body, Depends, HTTPException, Query, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError, validator

from backend.agents.tools.base_tool import AbstractApiTool
from backend.auth.dependencies import get_current_user, verify_permissions
from backend.core.events import publish_event
from backend.core.metrics import ApiMetrics
from backend.models.user import User

# Configure module logger
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Tool categories
class ToolCategory(str, Enum):
    """Categories for organizing tools."""
    BLOCKCHAIN = "blockchain"
    ANALYSIS = "analysis"
    REPORTING = "reporting"
    UTILITY = "utility"
    INTEGRATION = "integration"
    SIMULATION = "simulation"
    SECURITY = "security"
    UNKNOWN = "unknown"


# Tool status
class ToolStatus(str, Enum):
    """Status of a tool."""
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"


# Tool metadata model
class ToolMetadata(BaseModel):
    """Metadata for a tool."""
    id: str
    name: str
    description: str
    version: str = "1.0.0"
    category: ToolCategory = ToolCategory.UNKNOWN
    provider_id: Optional[str] = None
    requires_auth: bool = False
    supports_async: bool = False
    supports_streaming: bool = False
    status: ToolStatus = ToolStatus.UNKNOWN
    last_checked: Optional[str] = None
    capabilities: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    schema: Optional[Dict[str, Any]] = None
    examples: List[Dict[str, Any]] = Field(default_factory=list)
    
    class Config:
        schema_extra = {
            "example": {
                "id": "crypto_anomaly_tool",
                "name": "Crypto Anomaly Tool",
                "description": "Detects anomalies in cryptocurrency transactions",
                "version": "1.0.0",
                "category": "analysis",
                "provider_id": "sim-api",
                "requires_auth": True,
                "supports_async": True,
                "supports_streaming": False,
                "status": "available",
                "last_checked": "2025-06-21T12:34:56Z",
                "capabilities": ["anomaly_detection", "statistical_analysis"],
                "tags": ["crypto", "fraud", "detection"],
                "schema": {
                    "properties": {
                        "address": {"type": "string", "description": "Blockchain address to analyze"},
                        "chain": {"type": "string", "description": "Blockchain network"},
                        "time_range": {"type": "object", "description": "Time range for analysis"}
                    },
                    "required": ["address", "chain"]
                },
                "examples": [
                    {
                        "description": "Detect anomalies for an Ethereum address",
                        "request": {"address": "0x1234...", "chain": "ethereum"},
                        "response": {"anomalies": [{"type": "volume_spike", "confidence": 0.85}]}
                    }
                ]
            }
        }


# Tool execution request model
class ToolExecutionRequest(BaseModel):
    """Request for tool execution."""
    tool_id: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    async_execution: bool = False
    timeout_seconds: Optional[float] = None
    
    class Config:
        schema_extra = {
            "example": {
                "tool_id": "crypto_anomaly_tool",
                "parameters": {
                    "address": "0x1234567890abcdef1234567890abcdef12345678",
                    "chain": "ethereum",
                    "time_range": {
                        "start_date": "2025-01-01T00:00:00Z",
                        "end_date": "2025-06-21T00:00:00Z"
                    }
                },
                "async_execution": False,
                "timeout_seconds": 30.0
            }
        }


# Tool execution response model
class ToolExecutionResponse(BaseModel):
    """Response from tool execution."""
    tool_id: str
    status: str
    execution_time: float
    result: Optional[Any] = None
    error: Optional[str] = None
    task_id: Optional[str] = None  # For async execution
    
    class Config:
        schema_extra = {
            "example": {
                "tool_id": "crypto_anomaly_tool",
                "status": "success",
                "execution_time": 1.234,
                "result": {
                    "anomalies": [
                        {
                            "type": "volume_spike",
                            "confidence": 0.85,
                            "description": "Unusual transaction volume detected"
                        }
                    ]
                }
            }
        }


# Tool health check response model
class ToolHealthCheck(BaseModel):
    """Health check for a tool."""
    id: str
    status: ToolStatus
    latency_ms: float
    last_checked: str
    error: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


# MCP Tool Manifest model
class MCPToolManifest(BaseModel):
    """Tool manifest for MCP integration."""
    schema_version: str = "1.0"
    name: str
    description: str
    version: str
    tools: List[Dict[str, Any]]


# Global registry of discovered tools
_tool_registry: Dict[str, Type[AbstractApiTool]] = {}
_tool_instances: Dict[str, AbstractApiTool] = {}
_tool_metadata: Dict[str, ToolMetadata] = {}
_last_discovery_time: float = 0


def discover_tools() -> Dict[str, Type[AbstractApiTool]]:
    """
    Discover tools in the agents/tools directory.
    
    Returns:
        Dictionary of tool IDs to tool classes
    """
    global _last_discovery_time
    tools_dir = Path(__file__).parent.parent.parent / "agents" / "tools"
    
    if not tools_dir.exists() or not tools_dir.is_dir():
        logger.warning(f"Tools directory not found: {tools_dir}")
        return {}
    
    # Track discovery time
    start_time = time.time()
    _last_discovery_time = start_time
    
    # Dictionary to store discovered tools
    discovered_tools = {}
    
    # Get all Python files in the tools directory
    tool_files = list(tools_dir.glob("*.py"))
    tool_files.extend(list(tools_dir.glob("*/*.py")))  # Also check subdirectories
    
    # Exclude __init__.py and base_tool.py
    tool_files = [f for f in tool_files if f.name not in ("__init__.py", "base_tool.py")]
    
    # Import each tool module and find tool classes
    for tool_file in tool_files:
        try:
            # Convert file path to module path
            rel_path = tool_file.relative_to(Path(__file__).parent.parent.parent)
            module_path = str(rel_path.with_suffix("")).replace(os.sep, ".")
            
            # Import the module
            module = importlib.import_module(module_path)
            
            # Find tool classes in the module
            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and 
                    issubclass(obj, AbstractApiTool) and 
                    obj != AbstractApiTool):
                    
                    # Get tool ID from class name or name attribute
                    tool_id = getattr(obj, "name", name.lower())
                    if tool_id.endswith("Tool"):
                        tool_id = tool_id[:-4].lower()
                    
                    discovered_tools[tool_id] = obj
                    logger.debug(f"Discovered tool: {tool_id} ({obj.__name__})")
        
        except Exception as e:
            logger.error(f"Error discovering tool in {tool_file}: {e}")
    
    # Log discovery results
    discovery_time = time.time() - start_time
    logger.info(f"Discovered {len(discovered_tools)} tools in {discovery_time:.2f}s")
    
    return discovered_tools


def get_tool_metadata(tool_class: Type[AbstractApiTool], tool_id: str) -> ToolMetadata:
    """
    Get metadata for a tool class.
    
    Args:
        tool_class: Tool class
        tool_id: Tool ID
        
    Returns:
        Tool metadata
    """
    # Get basic metadata from class attributes
    name = getattr(tool_class, "name", tool_class.__name__)
    description = getattr(tool_class, "description", tool_class.__doc__ or "")
    version = getattr(tool_class, "version", "1.0.0")
    provider_id = getattr(tool_class, "provider_id", None)
    
    # Determine category based on name or module path
    category = ToolCategory.UNKNOWN
    module_path = tool_class.__module__
    
    if "crypto" in tool_id or "blockchain" in tool_id or "chain" in tool_id:
        category = ToolCategory.BLOCKCHAIN
    elif "analysis" in tool_id or "detect" in tool_id or "gnn" in tool_id:
        category = ToolCategory.ANALYSIS
    elif "report" in tool_id or "template" in tool_id:
        category = ToolCategory.REPORTING
    elif "sim" in tool_id:
        category = ToolCategory.SIMULATION
    elif "graph" in tool_id or "query" in tool_id:
        category = ToolCategory.INTEGRATION
    
    # Determine capabilities and tags
    capabilities = []
    tags = []
    
    if "crypto" in tool_id:
        tags.append("crypto")
    if "fraud" in tool_id or "anomaly" in tool_id:
        tags.append("fraud")
        capabilities.append("fraud_detection")
    if "gnn" in tool_id:
        tags.append("machine_learning")
        capabilities.append("graph_neural_network")
    if "graph" in tool_id:
        tags.append("graph")
        capabilities.append("graph_analysis")
    if "sim" in tool_id:
        tags.append("simulation")
        capabilities.append("simulation")
    
    # Get schema from request model if available
    schema = None
    if hasattr(tool_class, "request_model"):
        request_model = getattr(tool_class, "request_model")
        if hasattr(request_model, "schema"):
            try:
                schema = request_model.schema()
            except Exception as e:
                logger.warning(f"Error getting schema for {tool_id}: {e}")
    
    # Check if tool supports async execution
    supports_async = hasattr(tool_class, "async_execute") or hasattr(tool_class, "async_request")
    
    # Create metadata
    metadata = ToolMetadata(
        id=tool_id,
        name=name,
        description=description,
        version=version,
        category=category,
        provider_id=provider_id,
        requires_auth=True,  # Default to requiring auth for security
        supports_async=supports_async,
        supports_streaming=False,  # Default to not supporting streaming
        status=ToolStatus.UNKNOWN,  # Status will be updated by health check
        capabilities=capabilities,
        tags=tags,
        schema=schema,
    )
    
    return metadata


def get_tool_instance(tool_id: str) -> AbstractApiTool:
    """
    Get or create an instance of a tool.
    
    Args:
        tool_id: Tool ID
        
    Returns:
        Tool instance
        
    Raises:
        KeyError: If tool is not found
    """
    global _tool_instances
    
    # Return existing instance if available
    if tool_id in _tool_instances:
        return _tool_instances[tool_id]
    
    # Get tool class
    if tool_id not in _tool_registry:
        raise KeyError(f"Tool not found: {tool_id}")
    
    tool_class = _tool_registry[tool_id]
    
    # Create instance
    try:
        tool_instance = tool_class()
        _tool_instances[tool_id] = tool_instance
        return tool_instance
    except Exception as e:
        logger.error(f"Error creating instance of {tool_id}: {e}")
        raise


def check_tool_health(tool_id: str) -> ToolHealthCheck:
    """
    Check the health of a tool.
    
    Args:
        tool_id: Tool ID
        
    Returns:
        Tool health check
    """
    start_time = time.time()
    
    try:
        # Get tool instance
        tool = get_tool_instance(tool_id)
        
        # Try to access basic attributes
        name = tool.name
        description = tool.description
        
        # Update metadata status
        if tool_id in _tool_metadata:
            _tool_metadata[tool_id].status = ToolStatus.AVAILABLE
            _tool_metadata[tool_id].last_checked = time.strftime("%Y-%m-%dT%H:%M:%SZ")
        
        # Return health check
        return ToolHealthCheck(
            id=tool_id,
            status=ToolStatus.AVAILABLE,
            latency_ms=(time.time() - start_time) * 1000,
            last_checked=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        )
    
    except Exception as e:
        # Update metadata status
        if tool_id in _tool_metadata:
            _tool_metadata[tool_id].status = ToolStatus.UNAVAILABLE
            _tool_metadata[tool_id].last_checked = time.strftime("%Y-%m-%dT%H:%M:%SZ")
        
        # Return health check with error
        return ToolHealthCheck(
            id=tool_id,
            status=ToolStatus.UNAVAILABLE,
            latency_ms=(time.time() - start_time) * 1000,
            last_checked=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            error=str(e),
        )


def execute_tool(
    tool_id: str,
    parameters: Dict[str, Any],
    async_execution: bool = False,
    timeout_seconds: Optional[float] = None,
) -> ToolExecutionResponse:
    """
    Execute a tool with parameters.
    
    Args:
        tool_id: Tool ID
        parameters: Tool parameters
        async_execution: Whether to execute asynchronously
        timeout_seconds: Timeout in seconds
        
    Returns:
        Tool execution response
    """
    start_time = time.time()
    
    try:
        # Get tool instance
        tool = get_tool_instance(tool_id)
        
        # Execute tool
        if async_execution and hasattr(tool, "async_execute"):
            # For async execution, we would need to set up a task queue
            # This is a simplified implementation
            result = {"message": "Async execution not fully implemented yet"}
            task_id = f"{tool_id}_{int(time.time())}"
            
            return ToolExecutionResponse(
                tool_id=tool_id,
                status="pending",
                execution_time=time.time() - start_time,
                task_id=task_id,
            )
        else:
            # Synchronous execution
            result = tool(parameters)
            
            return ToolExecutionResponse(
                tool_id=tool_id,
                status="success",
                execution_time=time.time() - start_time,
                result=result,
            )
    
    except Exception as e:
        logger.error(f"Error executing tool {tool_id}: {e}")
        
        return ToolExecutionResponse(
            tool_id=tool_id,
            status="error",
            execution_time=time.time() - start_time,
            error=str(e),
        )


@lru_cache(maxsize=1)
def get_mcp_tool_manifest() -> MCPToolManifest:
    """
    Get MCP tool manifest for all tools.
    
    Returns:
        MCP tool manifest
    """
    tools = []
    
    for tool_id, metadata in _tool_metadata.items():
        # Skip tools without schemas
        if not metadata.schema:
            continue
        
        # Create tool definition
        tool_def = {
            "name": metadata.id,
            "description": metadata.description,
            "parameters": metadata.schema,
        }
        
        # Add examples if available
        if metadata.examples:
            tool_def["examples"] = metadata.examples
        
        tools.append(tool_def)
    
    return MCPToolManifest(
        name="Coding Analyst Droid Tools",
        description="Tools for blockchain fraud analysis and investigation",
        version="1.0.0",
        tools=tools,
    )


# Initialize tools on module import
_tool_registry = discover_tools()
for tool_id, tool_class in _tool_registry.items():
    try:
        _tool_metadata[tool_id] = get_tool_metadata(tool_class, tool_id)
    except Exception as e:
        logger.error(f"Error getting metadata for {tool_id}: {e}")


@router.get(
    "",
    response_model=List[ToolMetadata],
    summary="List available tools",
    description="Get a list of all available tools with metadata",
)
async def list_tools(
    category: Optional[ToolCategory] = Query(None, description="Filter by category"),
    tag: Optional[str] = Query(None, description="Filter by tag"),
    status: Optional[ToolStatus] = Query(None, description="Filter by status"),
    current_user: User = Depends(get_current_user),
) -> List[ToolMetadata]:
    """
    Get a list of all available tools with metadata.
    
    Args:
        category: Filter by category
        tag: Filter by tag
        status: Filter by status
        current_user: Current authenticated user
        
    Returns:
        List of tool metadata
    """
    # Verify permissions
    verify_permissions(current_user, "tools:list")
    
    # Check if tools need to be rediscovered
    global _last_discovery_time
    if time.time() - _last_discovery_time > 3600:  # Rediscover every hour
        _tool_registry.update(discover_tools())
        for tool_id, tool_class in _tool_registry.items():
            if tool_id not in _tool_metadata:
                try:
                    _tool_metadata[tool_id] = get_tool_metadata(tool_class, tool_id)
                except Exception as e:
                    logger.error(f"Error getting metadata for {tool_id}: {e}")
    
    # Apply filters
    filtered_tools = list(_tool_metadata.values())
    
    if category:
        filtered_tools = [t for t in filtered_tools if t.category == category]
    
    if tag:
        filtered_tools = [t for t in filtered_tools if tag in t.tags]
    
    if status:
        filtered_tools = [t for t in filtered_tools if t.status == status]
    
    # Track API usage
    ApiMetrics.track_call(
        provider="internal",
        endpoint="/api/v1/tools",
        func=lambda: None,
        environment=os.environ.get("ENVIRONMENT", "development"),
        version=os.environ.get("APP_VERSION", "1.8.0-beta"),
    )()
    
    return filtered_tools


@router.get(
    "/{tool_id}",
    response_model=ToolMetadata,
    summary="Get tool metadata",
    description="Get metadata for a specific tool",
)
async def get_tool(
    tool_id: str,
    current_user: User = Depends(get_current_user),
) -> ToolMetadata:
    """
    Get metadata for a specific tool.
    
    Args:
        tool_id: Tool ID
        current_user: Current authenticated user
        
    Returns:
        Tool metadata
        
    Raises:
        HTTPException: If tool is not found
    """
    # Verify permissions
    verify_permissions(current_user, "tools:read")
    
    # Check if tool exists
    if tool_id not in _tool_metadata:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tool not found: {tool_id}",
        )
    
    # Track API usage
    ApiMetrics.track_call(
        provider="internal",
        endpoint=f"/api/v1/tools/{tool_id}",
        func=lambda: None,
        environment=os.environ.get("ENVIRONMENT", "development"),
        version=os.environ.get("APP_VERSION", "1.8.0-beta"),
    )()
    
    return _tool_metadata[tool_id]


@router.get(
    "/{tool_id}/health",
    response_model=ToolHealthCheck,
    summary="Check tool health",
    description="Check the health of a specific tool",
)
async def check_health(
    tool_id: str,
    current_user: User = Depends(get_current_user),
) -> ToolHealthCheck:
    """
    Check the health of a specific tool.
    
    Args:
        tool_id: Tool ID
        current_user: Current authenticated user
        
    Returns:
        Tool health check
        
    Raises:
        HTTPException: If tool is not found
    """
    # Verify permissions
    verify_permissions(current_user, "tools:read")
    
    # Check if tool exists
    if tool_id not in _tool_registry:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tool not found: {tool_id}",
        )
    
    # Track API usage
    ApiMetrics.track_call(
        provider="internal",
        endpoint=f"/api/v1/tools/{tool_id}/health",
        func=lambda: None,
        environment=os.environ.get("ENVIRONMENT", "development"),
        version=os.environ.get("APP_VERSION", "1.8.0-beta"),
    )()
    
    # Check tool health
    return check_tool_health(tool_id)


@router.post(
    "/{tool_id}/execute",
    response_model=ToolExecutionResponse,
    summary="Execute tool",
    description="Execute a specific tool with parameters",
)
async def execute(
    tool_id: str,
    request: ToolExecutionRequest = Body(...),
    current_user: User = Depends(get_current_user),
) -> ToolExecutionResponse:
    """
    Execute a specific tool with parameters.
    
    Args:
        tool_id: Tool ID
        request: Tool execution request
        current_user: Current authenticated user
        
    Returns:
        Tool execution response
        
    Raises:
        HTTPException: If tool is not found or execution fails
    """
    # Verify permissions
    verify_permissions(current_user, "tools:execute")
    
    # Check if tool exists
    if tool_id not in _tool_registry:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tool not found: {tool_id}",
        )
    
    # Override tool_id in request
    request.tool_id = tool_id
    
    # Track API usage
    ApiMetrics.track_call(
        provider="internal",
        endpoint=f"/api/v1/tools/{tool_id}/execute",
        func=lambda: None,
        environment=os.environ.get("ENVIRONMENT", "development"),
        version=os.environ.get("APP_VERSION", "1.8.0-beta"),
    )()
    
    # Execute tool
    try:
        response = execute_tool(
            tool_id=tool_id,
            parameters=request.parameters,
            async_execution=request.async_execution,
            timeout_seconds=request.timeout_seconds,
        )
        
        # Publish event
        publish_event("tool_execution", {
            "tool_id": tool_id,
            "user_id": current_user.id,
            "status": response.status,
            "execution_time": response.execution_time,
            "async": request.async_execution,
        })
        
        return response
    
    except Exception as e:
        logger.error(f"Error executing tool {tool_id}: {e}")
        
        # Publish event
        publish_event("tool_execution_error", {
            "tool_id": tool_id,
            "user_id": current_user.id,
            "error": str(e),
        })
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Tool execution failed: {e}",
        )


@router.get(
    "/mcp/manifest",
    response_model=MCPToolManifest,
    summary="Get MCP tool manifest",
    description="Get tool manifest for MCP integration",
)
async def get_mcp_manifest(
    current_user: User = Depends(get_current_user),
) -> MCPToolManifest:
    """
    Get tool manifest for MCP integration.
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        MCP tool manifest
    """
    # Verify permissions
    verify_permissions(current_user, "tools:read")
    
    # Track API usage
    ApiMetrics.track_call(
        provider="internal",
        endpoint="/api/v1/tools/mcp/manifest",
        func=lambda: None,
        environment=os.environ.get("ENVIRONMENT", "development"),
        version=os.environ.get("APP_VERSION", "1.8.0-beta"),
    )()
    
    return get_mcp_tool_manifest()


@router.post(
    "/mcp/execute",
    summary="Execute tool via MCP",
    description="Execute a tool via MCP protocol",
)
async def execute_mcp(
    request: Request,
    current_user: User = Depends(get_current_user),
) -> JSONResponse:
    """
    Execute a tool via MCP protocol.
    
    Args:
        request: MCP request
        current_user: Current authenticated user
        
    Returns:
        MCP response
        
    Raises:
        HTTPException: If tool is not found or execution fails
    """
    # Verify permissions
    verify_permissions(current_user, "tools:execute")
    
    # Parse MCP request
    try:
        mcp_request = await request.json()
        
        # Extract tool name and parameters
        tool_name = mcp_request.get("name")
        parameters = mcp_request.get("parameters", {})
        
        if not tool_name:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"error": "Missing tool name in MCP request"},
            )
        
        # Map MCP tool name to internal tool ID
        tool_id = tool_name
        
        # Track API usage
        ApiMetrics.track_call(
            provider="internal",
            endpoint="/api/v1/tools/mcp/execute",
            func=lambda: None,
            environment=os.environ.get("ENVIRONMENT", "development"),
            version=os.environ.get("APP_VERSION", "1.8.0-beta"),
        )()
        
        # Execute tool
        response = execute_tool(
            tool_id=tool_id,
            parameters=parameters,
            async_execution=False,
        )
        
        # Convert to MCP response
        if response.status == "success":
            return JSONResponse(
                content={"result": response.result},
            )
        else:
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"error": response.error or "Tool execution failed"},
            )
    
    except Exception as e:
        logger.error(f"Error processing MCP request: {e}")
        
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": f"MCP request processing failed: {e}"},
        )


@router.get(
    "/categories",
    response_model=List[str],
    summary="List tool categories",
    description="Get a list of all tool categories",
)
async def list_categories(
    current_user: User = Depends(get_current_user),
) -> List[str]:
    """
    Get a list of all tool categories.
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        List of category names
    """
    # Verify permissions
    verify_permissions(current_user, "tools:list")
    
    # Track API usage
    ApiMetrics.track_call(
        provider="internal",
        endpoint="/api/v1/tools/categories",
        func=lambda: None,
        environment=os.environ.get("ENVIRONMENT", "development"),
        version=os.environ.get("APP_VERSION", "1.8.0-beta"),
    )()
    
    return [category.value for category in ToolCategory]


@router.get(
    "/tags",
    response_model=List[str],
    summary="List tool tags",
    description="Get a list of all tool tags",
)
async def list_tags(
    current_user: User = Depends(get_current_user),
) -> List[str]:
    """
    Get a list of all tool tags.
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        List of tag names
    """
    # Verify permissions
    verify_permissions(current_user, "tools:list")
    
    # Collect all tags
    all_tags = set()
    for metadata in _tool_metadata.values():
        all_tags.update(metadata.tags)
    
    # Track API usage
    ApiMetrics.track_call(
        provider="internal",
        endpoint="/api/v1/tools/tags",
        func=lambda: None,
        environment=os.environ.get("ENVIRONMENT", "development"),
        version=os.environ.get("APP_VERSION", "1.8.0-beta"),
    )()
    
    return sorted(all_tags)


@router.post(
    "/rediscover",
    summary="Rediscover tools",
    description="Force rediscovery of tools",
)
async def rediscover_tools(
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Force rediscovery of tools.
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        Discovery results
    """
    # Verify permissions
    verify_permissions(current_user, "tools:admin")
    
    # Rediscover tools
    global _tool_registry, _tool_metadata
    start_time = time.time()
    
    _tool_registry = discover_tools()
    _tool_metadata = {}
    
    for tool_id, tool_class in _tool_registry.items():
        try:
            _tool_metadata[tool_id] = get_tool_metadata(tool_class, tool_id)
        except Exception as e:
            logger.error(f"Error getting metadata for {tool_id}: {e}")
    
    # Clear tool instances to force recreation
    _tool_instances.clear()
    
    # Track API usage
    ApiMetrics.track_call(
        provider="internal",
        endpoint="/api/v1/tools/rediscover",
        func=lambda: None,
        environment=os.environ.get("ENVIRONMENT", "development"),
        version=os.environ.get("APP_VERSION", "1.8.0-beta"),
    )()
    
    # Return results
    return {
        "success": True,
        "tools_discovered": len(_tool_registry),
        "execution_time": time.time() - start_time,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


@router.get(
    "/health",
    response_model=List[ToolHealthCheck],
    summary="Check all tools health",
    description="Check the health of all tools",
)
async def check_all_health(
    current_user: User = Depends(get_current_user),
) -> List[ToolHealthCheck]:
    """
    Check the health of all tools.
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        List of tool health checks
    """
    # Verify permissions
    verify_permissions(current_user, "tools:read")
    
    # Check health of all tools
    health_checks = []
    
    for tool_id in _tool_registry:
        try:
            health_check = check_tool_health(tool_id)
            health_checks.append(health_check)
        except Exception as e:
            logger.error(f"Error checking health of {tool_id}: {e}")
            health_checks.append(ToolHealthCheck(
                id=tool_id,
                status=ToolStatus.UNKNOWN,
                latency_ms=0.0,
                last_checked=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                error=str(e),
            ))
    
    # Track API usage
    ApiMetrics.track_call(
        provider="internal",
        endpoint="/api/v1/tools/health",
        func=lambda: None,
        environment=os.environ.get("ENVIRONMENT", "development"),
        version=os.environ.get("APP_VERSION", "1.8.0-beta"),
    )()
    
    return health_checks
