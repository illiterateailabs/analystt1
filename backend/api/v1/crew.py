"""
CrewAI multi-agent system API endpoints.

This module provides FastAPI endpoints for managing and interacting with
CrewAI crews, including running crews, checking status, and listing available
crews and agents.
"""

import logging
from typing import Any, Dict, List, Optional, Union

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query, Path
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from backend.auth.dependencies import get_current_user, RateLimiter
from backend.agents.factory import CrewFactory
from backend.agents.config import load_agent_config, load_crew_config

logger = logging.getLogger(__name__)

# Create router
router = APIRouter()


# Models
class CrewRequest(BaseModel):
    """Model for crew execution requests."""
    
    crew_name: str = Field(..., description="Name of the crew to run")
    inputs: Dict[str, Any] = Field(default={}, description="Input data for the crew")
    async_execution: bool = Field(default=False, description="Whether to run the crew asynchronously")


class CrewResponse(BaseModel):
    """Model for crew execution responses."""
    
    success: bool = Field(..., description="Whether the request was successful")
    crew_name: str = Field(..., description="Name of the crew that was run")
    task_id: Optional[str] = Field(None, description="Task ID for async execution")
    result: Optional[Any] = Field(None, description="Result of the crew execution")
    error: Optional[str] = Field(None, description="Error message if execution failed")


class AgentInfo(BaseModel):
    """Model for agent information."""
    
    id: str = Field(..., description="Agent ID")
    role: str = Field(..., description="Agent role")
    goal: str = Field(..., description="Agent goal")
    tools: List[str] = Field(default=[], description="Tools used by the agent")


class CrewInfo(BaseModel):
    """Model for crew information."""
    
    name: str = Field(..., description="Crew name")
    process_type: str = Field(..., description="Process type (sequential or hierarchical)")
    manager: Optional[str] = Field(None, description="Manager agent ID")
    agents: List[str] = Field(..., description="List of agent IDs in the crew")
    description: Optional[str] = Field(None, description="Crew description")


# Background tasks storage
background_tasks = {}


# Rate limiters
crew_rate_limiter = RateLimiter(times=5, seconds=60)  # 5 requests per minute


@router.post(
    "/run",
    response_model=CrewResponse,
    summary="Run a CrewAI crew",
    dependencies=[Depends(get_current_user), Depends(crew_rate_limiter)]
)
async def run_crew(
    request: CrewRequest,
    background_tasks: BackgroundTasks
):
    """
    Run a CrewAI crew with the specified inputs.
    
    This endpoint allows running a crew either synchronously (waiting for completion)
    or asynchronously (returning immediately and processing in the background).
    
    Args:
        request: The crew execution request
        background_tasks: FastAPI background tasks manager
        
    Returns:
        Crew execution response
    """
    try:
        # Create crew factory
        factory = CrewFactory()
        
        # Connect to external services
        try:
            await factory.connect()
        except Exception as e:
            logger.error(f"Error connecting to external services: {e}")
            raise HTTPException(
                status_code=503,
                detail=f"Could not connect to required services: {str(e)}"
            )
        
        # Check if crew exists
        available_crews = factory.get_available_crews()
        if request.crew_name not in available_crews:
            raise HTTPException(
                status_code=404,
                detail=f"Crew '{request.crew_name}' not found. Available crews: {', '.join(available_crews)}"
            )
        
        # Run crew synchronously or asynchronously
        if request.async_execution:
            # Generate task ID
            import uuid
            task_id = str(uuid.uuid4())
            
            # Add task to background tasks
            background_tasks.add_task(
                _run_crew_in_background,
                task_id=task_id,
                crew_name=request.crew_name,
                inputs=request.inputs
            )
            
            # Return task ID
            return {
                "success": True,
                "crew_name": request.crew_name,
                "task_id": task_id,
                "result": None,
                "error": None
            }
        else:
            # Run crew synchronously
            result = await factory.run_crew(request.crew_name, request.inputs)
            
            # Check for errors
            if not result.get("success", False):
                return {
                    "success": False,
                    "crew_name": request.crew_name,
                    "task_id": None,
                    "result": None,
                    "error": result.get("error", "Unknown error")
                }
            
            # Return result
            return {
                "success": True,
                "crew_name": request.crew_name,
                "task_id": None,
                "result": result.get("result"),
                "error": None
            }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error running crew: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error running crew: {str(e)}"
        )


@router.get(
    "/status/{task_id}",
    response_model=CrewResponse,
    summary="Get status of an asynchronous crew execution",
    dependencies=[Depends(get_current_user)]
)
async def get_crew_status(
    task_id: str = Path(..., description="Task ID of the asynchronous execution")
):
    """
    Get the status of an asynchronous crew execution.
    
    Args:
        task_id: Task ID of the asynchronous execution
        
    Returns:
        Crew execution status
    """
    # Check if task exists
    if task_id not in background_tasks:
        raise HTTPException(
            status_code=404,
            detail=f"Task '{task_id}' not found"
        )
    
    # Get task status
    task_status = background_tasks[task_id]
    
    return task_status


@router.get(
    "/crews",
    summary="List available crews",
    dependencies=[Depends(get_current_user)]
)
async def list_crews():
    """
    List all available crews.
    
    Returns:
        List of available crews with their information
    """
    try:
        # Get available crews
        factory = CrewFactory()
        available_crews = factory.get_available_crews()
        
        # Get crew information
        crews_info = []
        for crew_name in available_crews:
            try:
                config = load_crew_config(crew_name)
                crews_info.append({
                    "name": crew_name,
                    "process_type": config.process_type,
                    "manager": config.manager,
                    "agents": config.agents,
                    "description": _get_crew_description(crew_name)
                })
            except Exception as e:
                logger.warning(f"Error loading crew config for '{crew_name}': {e}")
        
        return {"crews": crews_info}
    
    except Exception as e:
        logger.exception(f"Error listing crews: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error listing crews: {str(e)}"
        )


@router.get(
    "/agents",
    summary="List available agents",
    dependencies=[Depends(get_current_user)]
)
async def list_agents(
    crew_name: Optional[str] = Query(None, description="Filter agents by crew")
):
    """
    List all available agents, optionally filtered by crew.
    
    Args:
        crew_name: Optional crew name to filter agents
        
    Returns:
        List of available agents with their information
    """
    try:
        # Get agents
        if crew_name:
            try:
                config = load_crew_config(crew_name)
                agent_ids = config.agents
            except Exception as e:
                logger.warning(f"Error loading crew config for '{crew_name}': {e}")
                raise HTTPException(
                    status_code=404,
                    detail=f"Crew '{crew_name}' not found"
                )
        else:
            # Get all agents from default configs
            from backend.agents.config import DEFAULT_AGENT_CONFIGS
            agent_ids = list(DEFAULT_AGENT_CONFIGS.keys())
        
        # Get agent information
        agents_info = []
        for agent_id in agent_ids:
            try:
                config = load_agent_config(agent_id)
                agents_info.append({
                    "id": config.id,
                    "role": config.role,
                    "goal": config.goal,
                    "tools": config.tools
                })
            except Exception as e:
                logger.warning(f"Error loading agent config for '{agent_id}': {e}")
        
        return {"agents": agents_info}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error listing agents: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error listing agents: {str(e)}"
        )


@router.get(
    "/crews/{crew_name}",
    summary="Get crew details",
    dependencies=[Depends(get_current_user)]
)
async def get_crew_details(
    crew_name: str = Path(..., description="Name of the crew")
):
    """
    Get detailed information about a specific crew.
    
    Args:
        crew_name: Name of the crew
        
    Returns:
        Detailed crew information
    """
    try:
        # Check if crew exists
        factory = CrewFactory()
        available_crews = factory.get_available_crews()
        if crew_name not in available_crews:
            raise HTTPException(
                status_code=404,
                detail=f"Crew '{crew_name}' not found"
            )
        
        # Get crew configuration
        config = load_crew_config(crew_name)
        
        # Get agent details
        agents = []
        for agent_id in config.agents:
            try:
                agent_config = load_agent_config(agent_id)
                agents.append({
                    "id": agent_config.id,
                    "role": agent_config.role,
                    "goal": agent_config.goal,
                    "tools": agent_config.tools,
                    "backstory": agent_config.backstory
                })
            except Exception as e:
                logger.warning(f"Error loading agent config for '{agent_id}': {e}")
                agents.append({"id": agent_id, "error": str(e)})
        
        # Return crew details
        return {
            "name": crew_name,
            "process_type": config.process_type,
            "manager": config.manager,
            "agents": agents,
            "description": _get_crew_description(crew_name),
            "verbose": config.verbose,
            "memory": config.memory,
            "cache": config.cache
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error getting crew details: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting crew details: {str(e)}"
        )


@router.get(
    "/agents/{agent_id}",
    summary="Get agent details",
    dependencies=[Depends(get_current_user)]
)
async def get_agent_details(
    agent_id: str = Path(..., description="ID of the agent")
):
    """
    Get detailed information about a specific agent.
    
    Args:
        agent_id: ID of the agent
        
    Returns:
        Detailed agent information
    """
    try:
        # Get agent configuration
        try:
            config = load_agent_config(agent_id)
        except Exception as e:
            logger.warning(f"Error loading agent config for '{agent_id}': {e}")
            raise HTTPException(
                status_code=404,
                detail=f"Agent '{agent_id}' not found"
            )
        
        # Get available tools
        factory = CrewFactory()
        available_tools = list(factory.tools.keys())
        
        # Check which tools are available
        tools_info = []
        for tool_name in config.tools:
            tool = factory.get_tool(tool_name)
            tools_info.append({
                "name": tool_name,
                "available": tool is not None,
                "description": tool.description if tool else None
            })
        
        # Return agent details
        return {
            "id": config.id,
            "role": config.role,
            "goal": config.goal,
            "backstory": config.backstory,
            "tools": tools_info,
            "memory": config.memory,
            "max_iter": config.max_iter,
            "allow_delegation": config.allow_delegation,
            "verbose": config.verbose,
            "llm_model": config.llm_model
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error getting agent details: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting agent details: {str(e)}"
        )


# Helper functions
async def _run_crew_in_background(task_id: str, crew_name: str, inputs: Dict[str, Any]):
    """
    Run a crew in the background and store the result.
    
    Args:
        task_id: Task ID for tracking
        crew_name: Name of the crew to run
        inputs: Input data for the crew
    """
    try:
        # Update task status to running
        background_tasks[task_id] = {
            "success": True,
            "crew_name": crew_name,
            "task_id": task_id,
            "result": None,
            "error": None,
            "status": "running"
        }
        
        # Create crew factory
        factory = CrewFactory()
        
        # Connect to external services
        await factory.connect()
        
        # Run crew
        result = await factory.run_crew(crew_name, inputs)
        
        # Update task status with result
        if result.get("success", False):
            background_tasks[task_id] = {
                "success": True,
                "crew_name": crew_name,
                "task_id": task_id,
                "result": result.get("result"),
                "error": None,
                "status": "completed"
            }
        else:
            background_tasks[task_id] = {
                "success": False,
                "crew_name": crew_name,
                "task_id": task_id,
                "result": None,
                "error": result.get("error", "Unknown error"),
                "status": "failed"
            }
    
    except Exception as e:
        logger.exception(f"Error running crew in background: {e}")
        background_tasks[task_id] = {
            "success": False,
            "crew_name": crew_name,
            "task_id": task_id,
            "result": None,
            "error": str(e),
            "status": "failed"
        }
    finally:
        # Close connections
        if 'factory' in locals():
            await factory.close()


def _get_crew_description(crew_name: str) -> str:
    """
    Get a description for a crew based on its name.
    
    Args:
        crew_name: Name of the crew
        
    Returns:
        Description of the crew
    """
    descriptions = {
        "fraud_investigation": "Investigates complex fraud cases by analyzing graph data, detecting patterns, and generating comprehensive reports.",
        "alert_enrichment": "Enriches alerts with supporting evidence, risk scoring, and recommended actions in real-time.",
        "red_blue_simulation": "Simulates adversarial scenarios by generating synthetic fraud patterns and testing detection capabilities."
    }
    
    return descriptions.get(crew_name, "No description available")
