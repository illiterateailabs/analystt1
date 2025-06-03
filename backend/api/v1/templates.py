"""
Template API endpoints for managing CrewAI templates.

This module provides endpoints for creating, retrieving, updating, and deleting
agent crew templates, as well as requesting new templates and getting template
suggestions based on use cases.
"""

import logging
import os
import yaml
import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from fastapi import APIRouter, HTTPException, Depends, Request, status, BackgroundTasks
from pydantic import BaseModel, Field, validator

from backend.agents.factory import CrewFactory
from backend.auth.rbac import require_roles, Roles, RoleSets
from backend.auth.dependencies import get_current_user
from backend.models.user import User


logger = logging.getLogger(__name__)
router = APIRouter()

# Constants
TEMPLATES_DIR = Path("backend/agents/configs/crews")
TEMPLATE_EXTENSION = ".yaml"


# Request/Response Models
class AgentConfig(BaseModel):
    """Configuration for an agent in a template."""
    name: str = Field(..., description="Name of the agent")
    role: str = Field(..., description="Role description for the agent")
    goal: str = Field(..., description="Goal of the agent")
    backstory: Optional[str] = Field(None, description="Backstory for the agent")
    tools: Optional[List[str]] = Field(None, description="List of tool names for the agent")
    llm: Optional[Dict[str, Any]] = Field(None, description="LLM configuration for the agent")
    verbose: Optional[bool] = Field(None, description="Whether the agent is verbose")


class TaskConfig(BaseModel):
    """Configuration for a task in a template."""
    description: str = Field(..., description="Description of the task")
    agent: str = Field(..., description="Name of the agent assigned to the task")
    expected_output: Optional[str] = Field(None, description="Expected output of the task")
    tools: Optional[List[str]] = Field(None, description="List of tool names for the task")
    async_execution: Optional[bool] = Field(None, description="Whether the task is executed asynchronously")


class TemplateBase(BaseModel):
    """Base model for templates."""
    name: str = Field(..., description="Name of the template")
    description: Optional[str] = Field(None, description="Description of the template")
    agents: List[AgentConfig] = Field(..., description="List of agent configurations")
    tasks: List[TaskConfig] = Field(..., description="List of task configurations")
    workflow: Optional[Dict[str, Any]] = Field(None, description="Workflow configuration")
    verbose: Optional[bool] = Field(None, description="Whether the crew is verbose")
    memory: Optional[Dict[str, Any]] = Field(None, description="Memory configuration")
    max_rpm: Optional[int] = Field(None, description="Maximum requests per minute")
    sla_seconds: Optional[int] = Field(None, description="SLA in seconds")
    hitl_triggers: Optional[List[str]] = Field(None, description="List of HITL triggers")
    
    @validator('name')
    def validate_name(cls, v):
        """Validate template name."""
        if not v:
            raise ValueError("Template name cannot be empty")
        if " " in v:
            raise ValueError("Template name cannot contain spaces")
        if not v.islower() and not v.replace("_", "").islower():
            raise ValueError("Template name should be lowercase with optional underscores")
        return v


class TemplateCreate(TemplateBase):
    """Model for creating a template."""
    pass


class TemplateUpdate(BaseModel):
    """Model for updating a template."""
    description: Optional[str] = Field(None, description="Description of the template")
    agents: Optional[List[AgentConfig]] = Field(None, description="List of agent configurations")
    tasks: Optional[List[TaskConfig]] = Field(None, description="List of task configurations")
    workflow: Optional[Dict[str, Any]] = Field(None, description="Workflow configuration")
    verbose: Optional[bool] = Field(None, description="Whether the crew is verbose")
    memory: Optional[Dict[str, Any]] = Field(None, description="Memory configuration")
    max_rpm: Optional[int] = Field(None, description="Maximum requests per minute")
    sla_seconds: Optional[int] = Field(None, description="SLA in seconds")
    hitl_triggers: Optional[List[str]] = Field(None, description="List of HITL triggers")


class TemplateResponse(TemplateBase):
    """Response model for templates."""
    id: str = Field(..., description="ID of the template (same as name)")
    file_path: str = Field(..., description="Path to the template file")
    created_by: Optional[str] = Field(None, description="Username of the creator")
    created_at: Optional[str] = Field(None, description="Creation timestamp")
    updated_at: Optional[str] = Field(None, description="Last update timestamp")


class TemplateListResponse(BaseModel):
    """Response model for listing templates."""
    templates: List[TemplateResponse] = Field(..., description="List of templates")
    count: int = Field(..., description="Total number of templates")


class TemplateRequest(BaseModel):
    """Model for requesting a new template."""
    name: str = Field(..., description="Requested name for the template")
    description: str = Field(..., description="Description of the template")
    use_case: str = Field(..., description="Use case for the template")
    required_tools: Optional[List[str]] = Field(None, description="List of required tools")
    required_agents: Optional[List[str]] = Field(None, description="List of required agents")
    sla_requirement: Optional[str] = Field(None, description="SLA requirement (e.g., '15 minutes')")
    example_inputs: Optional[Dict[str, Any]] = Field(None, description="Example inputs for the template")
    additional_notes: Optional[str] = Field(None, description="Additional notes for the template")
    
    @validator('name')
    def validate_name(cls, v):
        """Validate template name."""
        if not v:
            raise ValueError("Template name cannot be empty")
        if " " in v:
            raise ValueError("Template name cannot contain spaces")
        if not v.islower() and not v.replace("_", "").islower():
            raise ValueError("Template name should be lowercase with optional underscores")
        return v


class TemplateSuggestion(BaseModel):
    """Model for template suggestions."""
    name: str = Field(..., description="Suggested name for the template")
    description: str = Field(..., description="Description of the template")
    suggested_agents: List[str] = Field(..., description="List of suggested agents")
    suggested_tools: List[str] = Field(..., description="List of suggested tools")
    estimated_creation_time: str = Field(..., description="Estimated time to create the template")
    confidence: float = Field(..., description="Confidence in the suggestion (0-1)")


class TemplateRequestResponse(BaseModel):
    """Response model for template requests."""
    request_id: str = Field(..., description="ID of the request")
    status: str = Field(..., description="Status of the request")
    suggestion: Optional[TemplateSuggestion] = Field(None, description="Template suggestion")
    message: Optional[str] = Field(None, description="Additional message")


# Helper functions
def get_template_file_path(template_name: str) -> Path:
    """Get the file path for a template."""
    return TEMPLATES_DIR / f"{template_name}{TEMPLATE_EXTENSION}"


def load_template(template_name: str) -> Dict[str, Any]:
    """Load a template from file."""
    file_path = get_template_file_path(template_name)
    
    if not file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Template not found: {template_name}"
        )
    
    try:
        with open(file_path, 'r') as f:
            template_data = yaml.safe_load(f)
            return template_data
    except Exception as e:
        logger.error(f"Failed to load template {template_name}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load template: {str(e)}"
        )


def save_template(template_name: str, template_data: Dict[str, Any]) -> Path:
    """Save a template to file."""
    file_path = get_template_file_path(template_name)
    
    try:
        # Create directory if it doesn't exist
        os.makedirs(TEMPLATES_DIR, exist_ok=True)
        
        # Save template to file
        with open(file_path, 'w') as f:
            yaml.dump(template_data, f, sort_keys=False)
        
        return file_path
    except Exception as e:
        logger.error(f"Failed to save template {template_name}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save template: {str(e)}"
        )


def template_to_response(template_name: str, template_data: Dict[str, Any]) -> TemplateResponse:
    """Convert template data to a response model."""
    file_path = get_template_file_path(template_name)
    
    # Extract metadata from template data
    created_by = template_data.get("metadata", {}).get("created_by")
    created_at = template_data.get("metadata", {}).get("created_at")
    updated_at = template_data.get("metadata", {}).get("updated_at")
    
    # Extract agents
    agents = []
    for agent_name, agent_data in template_data.get("agents", {}).items():
        agents.append(AgentConfig(
            name=agent_name,
            role=agent_data.get("role", ""),
            goal=agent_data.get("goal", ""),
            backstory=agent_data.get("backstory"),
            tools=agent_data.get("tools"),
            llm=agent_data.get("llm"),
            verbose=agent_data.get("verbose")
        ))
    
    # Extract tasks
    tasks = []
    for task_data in template_data.get("tasks", []):
        tasks.append(TaskConfig(
            description=task_data.get("description", ""),
            agent=task_data.get("agent", ""),
            expected_output=task_data.get("expected_output"),
            tools=task_data.get("tools"),
            async_execution=task_data.get("async_execution")
        ))
    
    # Create response
    return TemplateResponse(
        id=template_name,
        name=template_name,
        description=template_data.get("description"),
        agents=agents,
        tasks=tasks,
        workflow=template_data.get("workflow"),
        verbose=template_data.get("verbose"),
        memory=template_data.get("memory"),
        max_rpm=template_data.get("max_rpm"),
        sla_seconds=template_data.get("sla_seconds"),
        hitl_triggers=template_data.get("hitl_triggers"),
        file_path=str(file_path),
        created_by=created_by,
        created_at=created_at,
        updated_at=updated_at
    )


def template_model_to_dict(template: Union[TemplateCreate, TemplateUpdate]) -> Dict[str, Any]:
    """Convert a template model to a dictionary for saving."""
    template_dict = template.dict(exclude_unset=True)
    
    # Convert agents list to dict
    agents_dict = {}
    for agent in template_dict.get("agents", []):
        agent_name = agent.pop("name")
        agents_dict[agent_name] = agent
    
    template_dict["agents"] = agents_dict
    
    return template_dict


# Dependency to get CrewFactory
async def get_crew_factory(request: Request) -> CrewFactory:
    """Get or create a CrewFactory instance."""
    if not hasattr(request.app.state, "crew_factory"):
        logger.info("Creating new CrewFactory instance")
        request.app.state.crew_factory = CrewFactory()
    return request.app.state.crew_factory


# Endpoints
@router.get("", response_model=TemplateListResponse)
@require_roles(RoleSets.ANALYSTS_AND_ADMIN)
async def list_templates(
    crew_factory: CrewFactory = Depends(get_crew_factory)
):
    """
    List all available templates.
    
    Returns:
        List of templates
    """
    try:
        # Get available templates from CrewFactory
        template_names = crew_factory.get_available_crews()
        
        # Load template data
        templates = []
        for template_name in template_names:
            try:
                template_data = load_template(template_name)
                template_response = template_to_response(template_name, template_data)
                templates.append(template_response)
            except Exception as e:
                logger.error(f"Failed to load template {template_name}: {e}")
                # Skip templates that fail to load
                continue
        
        return TemplateListResponse(
            templates=templates,
            count=len(templates)
        )
    except Exception as e:
        logger.error(f"Failed to list templates: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list templates: {str(e)}"
        )


@router.get("/{template_name}", response_model=TemplateResponse)
@require_roles(RoleSets.ANALYSTS_AND_ADMIN)
async def get_template(
    template_name: str,
    crew_factory: CrewFactory = Depends(get_crew_factory)
):
    """
    Get a template by name.
    
    Args:
        template_name: Name of the template
        
    Returns:
        Template details
    """
    try:
        # Check if template exists
        available_templates = crew_factory.get_available_crews()
        if template_name not in available_templates:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Template not found: {template_name}"
            )
        
        # Load template data
        template_data = load_template(template_name)
        
        # Convert to response model
        return template_to_response(template_name, template_data)
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Failed to get template {template_name}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get template: {str(e)}"
        )


@router.post("", response_model=TemplateResponse, status_code=status.HTTP_201_CREATED)
@require_roles([Roles.ADMIN])
async def create_template(
    template: TemplateCreate,
    current_user: User = Depends(get_current_user),
    crew_factory: CrewFactory = Depends(get_crew_factory)
):
    """
    Create a new template.
    
    Args:
        template: Template to create
        
    Returns:
        Created template
    """
    try:
        template_name = template.name
        
        # Check if template already exists
        file_path = get_template_file_path(template_name)
        if file_path.exists():
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Template already exists: {template_name}"
            )
        
        # Convert template model to dict
        template_dict = template_model_to_dict(template)
        
        # Add metadata
        template_dict["metadata"] = {
            "created_by": current_user.username,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        
        # Save template
        save_template(template_name, template_dict)
        
        # Reload CrewFactory to pick up new template
        await crew_factory.reload()
        
        # Return template response
        return template_to_response(template_name, template_dict)
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Failed to create template {template.name}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create template: {str(e)}"
        )


@router.put("/{template_name}", response_model=TemplateResponse)
@require_roles([Roles.ADMIN])
async def update_template(
    template_name: str,
    template: TemplateUpdate,
    current_user: User = Depends(get_current_user),
    crew_factory: CrewFactory = Depends(get_crew_factory)
):
    """
    Update a template.
    
    Args:
        template_name: Name of the template to update
        template: Template updates
        
    Returns:
        Updated template
    """
    try:
        # Check if template exists
        available_templates = crew_factory.get_available_crews()
        if template_name not in available_templates:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Template not found: {template_name}"
            )
        
        # Load existing template
        existing_template = load_template(template_name)
        
        # Convert update model to dict
        update_dict = template_model_to_dict(template)
        
        # Update template
        for key, value in update_dict.items():
            if value is not None:
                existing_template[key] = value
        
        # Update metadata
        if "metadata" not in existing_template:
            existing_template["metadata"] = {}
        
        existing_template["metadata"]["updated_at"] = datetime.now().isoformat()
        existing_template["metadata"]["updated_by"] = current_user.username
        
        # Save updated template
        save_template(template_name, existing_template)
        
        # Reload CrewFactory to pick up changes
        await crew_factory.reload()
        
        # Return updated template
        return template_to_response(template_name, existing_template)
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Failed to update template {template_name}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update template: {str(e)}"
        )


@router.delete("/{template_name}", status_code=status.HTTP_204_NO_CONTENT)
@require_roles([Roles.ADMIN])
async def delete_template(
    template_name: str,
    crew_factory: CrewFactory = Depends(get_crew_factory)
):
    """
    Delete a template.
    
    Args:
        template_name: Name of the template to delete
    """
    try:
        # Check if template exists
        file_path = get_template_file_path(template_name)
        if not file_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Template not found: {template_name}"
            )
        
        # Delete template file
        os.remove(file_path)
        
        # Reload CrewFactory to pick up changes
        await crew_factory.reload()
        
        return None
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Failed to delete template {template_name}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete template: {str(e)}"
        )


@router.post("/request", response_model=TemplateRequestResponse)
@require_roles(RoleSets.ANALYSTS_AND_ADMIN)
async def request_template(
    request: TemplateRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user)
):
    """
    Request a new template.
    
    Args:
        request: Template request
        
    Returns:
        Template request response
    """
    try:
        # Generate request ID
        request_id = f"req_{uuid.uuid4().hex[:8]}"
        
        # Store request in database or file
        request_data = request.dict()
        request_data["request_id"] = request_id
        request_data["status"] = "PENDING"
        request_data["requested_by"] = current_user.username
        request_data["requested_at"] = datetime.now().isoformat()
        
        # Save request to file (in a real system, this would go to a database)
        requests_dir = Path("backend/data/template_requests")
        os.makedirs(requests_dir, exist_ok=True)
        
        with open(requests_dir / f"{request_id}.json", 'w') as f:
            json.dump(request_data, f, indent=2)
        
        # Generate suggestion in background
        background_tasks.add_task(
            generate_template_suggestion,
            request_id,
            request_data
        )
        
        # Return immediate response
        return TemplateRequestResponse(
            request_id=request_id,
            status="PENDING",
            message="Template request submitted and is being processed"
        )
    except Exception as e:
        logger.error(f"Failed to request template: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to request template: {str(e)}"
        )


@router.get("/request/{request_id}", response_model=TemplateRequestResponse)
@require_roles(RoleSets.ANALYSTS_AND_ADMIN)
async def get_template_request(
    request_id: str
):
    """
    Get a template request by ID.
    
    Args:
        request_id: ID of the request
        
    Returns:
        Template request response
    """
    try:
        # Load request from file (in a real system, this would come from a database)
        requests_dir = Path("backend/data/template_requests")
        request_file = requests_dir / f"{request_id}.json"
        
        if not request_file.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Template request not found: {request_id}"
            )
        
        with open(request_file, 'r') as f:
            request_data = json.load(f)
        
        # Check if suggestion exists
        suggestion = None
        suggestion_file = requests_dir / f"{request_id}_suggestion.json"
        
        if suggestion_file.exists():
            with open(suggestion_file, 'r') as f:
                suggestion_data = json.load(f)
                suggestion = TemplateSuggestion(**suggestion_data)
        
        # Return response
        return TemplateRequestResponse(
            request_id=request_id,
            status=request_data.get("status", "UNKNOWN"),
            suggestion=suggestion,
            message=request_data.get("message")
        )
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Failed to get template request {request_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get template request: {str(e)}"
        )


@router.get("/suggestions", response_model=List[TemplateSuggestion])
@require_roles(RoleSets.ANALYSTS_AND_ADMIN)
async def suggest_templates(
    use_case: str,
    crew_factory: CrewFactory = Depends(get_crew_factory)
):
    """
    Get template suggestions based on a use case.
    
    Args:
        use_case: Use case description
        
    Returns:
        List of template suggestions
    """
    try:
        # Get available tools
        available_tools = crew_factory.get_available_tools()
        tool_names = [tool.name for tool in available_tools]
        
        # Get available agents
        available_agents = crew_factory.get_available_agents()
        
        # Generate suggestions based on use case
        suggestions = []
        
        # This is a simplified version - in a real system, this would use AI to generate suggestions
        if "crypto" in use_case.lower() or "blockchain" in use_case.lower():
            suggestions.append(TemplateSuggestion(
                name="crypto_investigation",
                description="Template for investigating cryptocurrency transactions and wallets",
                suggested_agents=["blockchain_analyst", "fraud_pattern_hunter"],
                suggested_tools=[tool for tool in tool_names if "crypto" in tool.lower() or "blockchain" in tool.lower() or "graph" in tool.lower()],
                estimated_creation_time="2 hours",
                confidence=0.85
            ))
        
        if "defi" in use_case.lower() or "smart contract" in use_case.lower():
            suggestions.append(TemplateSuggestion(
                name="defi_exploit_investigation",
                description="Template for investigating DeFi protocol exploits and vulnerabilities",
                suggested_agents=["blockchain_analyst", "smart_contract_auditor", "defi_specialist"],
                suggested_tools=[tool for tool in tool_names if "contract" in tool.lower() or "code" in tool.lower() or "sandbox" in tool.lower()],
                estimated_creation_time="3 hours",
                confidence=0.75
            ))
        
        if "bank" in use_case.lower() or "wire" in use_case.lower() or "traditional" in use_case.lower():
            suggestions.append(TemplateSuggestion(
                name="banking_fraud_investigation",
                description="Template for investigating traditional banking fraud",
                suggested_agents=["banking_analyst", "fraud_pattern_hunter", "compliance_checker"],
                suggested_tools=[tool for tool in tool_names if "pattern" in tool.lower() or "graph" in tool.lower() or "policy" in tool.lower()],
                estimated_creation_time="2 hours",
                confidence=0.8
            ))
        
        # If no specific suggestions, provide a generic one
        if not suggestions:
            suggestions.append(TemplateSuggestion(
                name="custom_investigation",
                description="Custom investigation template based on your use case",
                suggested_agents=["graph_analyst", "fraud_pattern_hunter", "report_writer"],
                suggested_tools=tool_names[:5],  # First 5 tools
                estimated_creation_time="4 hours",
                confidence=0.6
            ))
        
        return suggestions
    except Exception as e:
        logger.error(f"Failed to suggest templates: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to suggest templates: {str(e)}"
        )


# Background tasks
async def generate_template_suggestion(request_id: str, request_data: Dict[str, Any]):
    """
    Generate a template suggestion based on a request.
    
    Args:
        request_id: ID of the request
        request_data: Request data
    """
    try:
        # In a real system, this would use AI to generate a suggestion
        # For now, we'll generate a simple suggestion based on the request
        
        use_case = request_data.get("use_case", "")
        name = request_data.get("name", "")
        required_tools = request_data.get("required_tools", [])
        
        # Generate suggestion
        if "crypto" in use_case.lower() or "blockchain" in use_case.lower():
            suggestion = {
                "name": name or "crypto_investigation",
                "description": "Template for investigating cryptocurrency transactions and wallets",
                "suggested_agents": ["blockchain_analyst", "fraud_pattern_hunter"],
                "suggested_tools": required_tools or ["CryptoCSVLoaderTool", "GraphQueryTool", "PatternLibraryTool"],
                "estimated_creation_time": "2 hours",
                "confidence": 0.85
            }
        elif "defi" in use_case.lower() or "smart contract" in use_case.lower():
            suggestion = {
                "name": name or "defi_exploit_investigation",
                "description": "Template for investigating DeFi protocol exploits and vulnerabilities",
                "suggested_agents": ["blockchain_analyst", "smart_contract_auditor", "defi_specialist"],
                "suggested_tools": required_tools or ["CodeGenTool", "SandboxExecTool", "GraphQueryTool"],
                "estimated_creation_time": "3 hours",
                "confidence": 0.75
            }
        else:
            suggestion = {
                "name": name or "custom_investigation",
                "description": "Custom investigation template based on your use case",
                "suggested_agents": ["graph_analyst", "fraud_pattern_hunter", "report_writer"],
                "suggested_tools": required_tools or ["GraphQueryTool", "PatternLibraryTool", "TemplateEngineTool"],
                "estimated_creation_time": "4 hours",
                "confidence": 0.6
            }
        
        # Save suggestion to file
        requests_dir = Path("backend/data/template_requests")
        os.makedirs(requests_dir, exist_ok=True)
        
        with open(requests_dir / f"{request_id}_suggestion.json", 'w') as f:
            json.dump(suggestion, f, indent=2)
        
        # Update request status
        with open(requests_dir / f"{request_id}.json", 'r') as f:
            request_data = json.load(f)
        
        request_data["status"] = "COMPLETED"
        request_data["completed_at"] = datetime.now().isoformat()
        
        with open(requests_dir / f"{request_id}.json", 'w') as f:
            json.dump(request_data, f, indent=2)
        
    except Exception as e:
        logger.error(f"Failed to generate template suggestion for request {request_id}: {e}")
        
        # Update request status to ERROR
        try:
            requests_dir = Path("backend/data/template_requests")
            
            with open(requests_dir / f"{request_id}.json", 'r') as f:
                request_data = json.load(f)
            
            request_data["status"] = "ERROR"
            request_data["error"] = str(e)
            
            with open(requests_dir / f"{request_id}.json", 'w') as f:
                json.dump(request_data, f, indent=2)
        except Exception as inner_e:
            logger.error(f"Failed to update request status: {inner_e}")
