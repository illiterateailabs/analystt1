"""
Crew API endpoints for running and managing agent crews.

This module provides endpoints for listing available crews,
running specific crews with inputs, and managing crew execution
(pause, resume) for human-in-the-loop workflows.
"""

import logging
from typing import Dict, List, Optional, Any, Union
from fastapi import APIRouter, HTTPException, Depends, Request, status
from pydantic import BaseModel, Field

from backend.agents.factory import CrewFactory
from backend.auth.rbac import require_roles, Roles, RoleSets


logger = logging.getLogger(__name__)
router = APIRouter()


# Request/Response Models
class CrewRunRequest(BaseModel):
    """Request model for running a crew."""
    crew_name: str = Field(..., description="Name of the crew to run")
    inputs: Optional[Dict[str, Any]] = Field(default={}, description="Input parameters for the crew")


class CrewPauseRequest(BaseModel):
    """Request model for pausing a crew."""
    task_id: str = Field(..., description="Task ID of the running crew")
    reason: Optional[str] = Field(None, description="Reason for pausing")


class CrewResumeRequest(BaseModel):
    """Request model for resuming a paused crew."""
    task_id: str = Field(..., description="Task ID of the paused crew")
    approved: bool = Field(..., description="Whether the task is approved to continue")
    comment: Optional[str] = Field(None, description="Comment from the reviewer")


class CrewResponse(BaseModel):
    """Response model for crew operations."""
    success: bool = Field(..., description="Whether the operation was successful")
    task_id: Optional[str] = Field(None, description="Task ID for tracking")
    status: Optional[str] = Field(None, description="Current status of the crew")
    result: Optional[Any] = Field(None, description="Result of the crew execution")
    error: Optional[str] = Field(None, description="Error message if any")


# Dependency to get CrewFactory
async def get_crew_factory(request: Request) -> CrewFactory:
    """Get or create a CrewFactory instance."""
    if not hasattr(request.app.state, "crew_factory"):
        logger.info("Creating new CrewFactory instance")
        request.app.state.crew_factory = CrewFactory()
    return request.app.state.crew_factory


@router.get("")
@require_roles(RoleSets.ANALYSTS_AND_ADMIN)
async def list_crews(
    crew_factory: CrewFactory = Depends(get_crew_factory)
):
    """
    List all available crews.
    
    Returns:
        List of crew names that can be run
    """
    try:
        crews = crew_factory.get_available_crews()
        return {"success": True, "crews": crews}
    except Exception as e:
        logger.error(f"Failed to get crews: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get crews: {str(e)}"
        )


@router.post("/run")
@require_roles(RoleSets.ANALYSTS_AND_ADMIN)
async def run_crew(
    request: CrewRunRequest,
    crew_factory: CrewFactory = Depends(get_crew_factory)
):
    """
    Run a crew with the specified inputs.
    
    Args:
        request: Crew run request with crew name and inputs
        
    Returns:
        Crew execution result
    """
    try:
        logger.info(f"Running crew: {request.crew_name}")
        
        # Validate crew exists
        available_crews = crew_factory.get_available_crews()
        if request.crew_name not in available_crews:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Crew not found: {request.crew_name}"
            )
        
        # Run the crew
        result = await crew_factory.run_crew(request.crew_name, inputs=request.inputs)
        
        # Close the factory to release resources
        await crew_factory.close()
        
        # Return the result
        if isinstance(result, dict) and result.get("success") is False:
            # Crew execution failed but API call succeeded
            return CrewResponse(
                success=False,
                error=result.get("error", "Unknown error"),
                task_id=result.get("task_id"),
                status=result.get("status", "FAILED")
            )
        
        # Success case
        return CrewResponse(
            success=True,
            task_id=result.get("task_id") if isinstance(result, dict) else None,
            result=result,
            status=result.get("status", "COMPLETED") if isinstance(result, dict) else "COMPLETED"
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Failed to run crew: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to run crew: {str(e)}"
        )


@router.post("/{crew_name}")
@require_roles(RoleSets.ANALYSTS_AND_ADMIN)
async def run_crew_by_name(
    crew_name: str,
    inputs: Dict[str, Any] = {},
    crew_factory: CrewFactory = Depends(get_crew_factory)
):
    """
    Run a crew by name with the specified inputs.
    
    Args:
        crew_name: Name of the crew to run
        inputs: Input parameters for the crew
        
    Returns:
        Crew execution result
    """
    try:
        logger.info(f"Running crew by name: {crew_name}")
        
        # Validate crew exists
        available_crews = crew_factory.get_available_crews()
        if crew_name not in available_crews:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Crew not found: {crew_name}"
            )
        
        # Run the crew
        result = await crew_factory.run_crew(crew_name, inputs=inputs)
        
        # Close the factory to release resources
        await crew_factory.close()
        
        # Return the result
        if isinstance(result, dict) and result.get("success") is False:
            # Crew execution failed but API call succeeded
            return CrewResponse(
                success=False,
                error=result.get("error", "Unknown error"),
                task_id=result.get("task_id"),
                status=result.get("status", "FAILED")
            )
        
        # Success case
        return CrewResponse(
            success=True,
            task_id=result.get("task_id") if isinstance(result, dict) else None,
            result=result,
            status=result.get("status", "COMPLETED") if isinstance(result, dict) else "COMPLETED"
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Failed to run crew: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to run crew: {str(e)}"
        )


@router.patch("/pause")
@require_roles(RoleSets.COMPLIANCE_TEAM)
async def pause_crew(
    request: CrewPauseRequest,
    crew_factory: CrewFactory = Depends(get_crew_factory)
):
    """
    Pause a running crew task.
    
    Args:
        request: Crew pause request with task ID and reason
        
    Returns:
        Pause operation result
    """
    try:
        logger.info(f"Pausing crew task: {request.task_id}")
        
        # Pause the crew
        result = await crew_factory.pause_crew(
            task_id=request.task_id,
            reason=request.reason
        )
        
        return CrewResponse(
            success=result.get("success", False),
            task_id=request.task_id,
            status="PAUSED" if result.get("success", False) else "PAUSE_FAILED",
            error=result.get("error")
        )
        
    except Exception as e:
        logger.error(f"Failed to pause crew: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to pause crew: {str(e)}"
        )


@router.patch("/resume")
@require_roles(RoleSets.COMPLIANCE_TEAM)
async def resume_crew(
    request: CrewResumeRequest,
    crew_factory: CrewFactory = Depends(get_crew_factory)
):
    """
    Resume a paused crew task.
    
    Args:
        request: Crew resume request with task ID, approval status, and comment
        
    Returns:
        Resume operation result
    """
    try:
        logger.info(f"Resuming crew task: {request.task_id} (approved: {request.approved})")
        
        # Resume the crew
        result = await crew_factory.resume_crew(
            task_id=request.task_id,
            approved=request.approved,
            comment=request.comment
        )
        
        return CrewResponse(
            success=result.get("success", False),
            task_id=request.task_id,
            status="RESUMED" if result.get("success", False) else "RESUME_FAILED",
            error=result.get("error")
        )
        
    except Exception as e:
        logger.error(f"Failed to resume crew: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to resume crew: {str(e)}"
        )
