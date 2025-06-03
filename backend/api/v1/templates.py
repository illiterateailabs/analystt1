"""
Templates API endpoints for managing investigation templates.

This module provides endpoints for creating, retrieving, updating, and deleting
CrewAI templates, as well as getting suggestions for template configurations
based on use cases.
"""

import os
import yaml
import logging
import json
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from fastapi import APIRouter, HTTPException, Depends, Request, status, BackgroundTasks
from pydantic import BaseModel, Field, validator

from backend.agents.config import AGENT_CONFIGS_CREWS_DIR, get_available_crews
from backend.agents.factory import CrewFactory
from backend.auth.rbac import require_roles, Roles, RoleSets
from backend.integrations.gemini_client import GeminiClient

# Configure logging
logger = logging.getLogger(__name__)
router = APIRouter()

# Request/Response Models
class AgentConfig(BaseModel):
    """Configuration for an agent in a template."""
    id: str
    role: str
    goal: str
    backstory: Optional[str] = None
    tools: List[str] = []
    allow_delegation: bool = False
    max_iter: int = 15
    verbose: bool = True

class TaskConfig(BaseModel):
    """Configuration for a task in a template."""
    description: str
    expected_output: str
    agent_id: str

class TemplateRequest(BaseModel):
    """Request model for creating a template request."""
    name: str = Field(..., description="Name of the template", min_length=3, max_length=50)
    use_case: str = Field(..., description="Description of the use case", min_length=10)
    sla_requirement: Optional[str] = Field(None, description="SLA requirement for the template")
    priority: Optional[str] = Field(None, description="Priority of the template request", pattern="^(low|medium|high|critical)$")
    
    @validator('name')
    def name_must_be_valid(cls, v):
        if not v.replace('_', '').replace('-', '').isalnum():
            raise ValueError('name must contain only alphanumeric characters, underscores, and hyphens')
        return v

class TemplateResponse(BaseModel):
    """Response model for a template."""
    id: str = Field(..., description="Template ID")
    name: str = Field(..., description="Template name")
    description: str = Field(..., description="Template description")
    agents: List[str] = Field(..., description="List of agent IDs")
    tasks: List[TaskConfig] = Field(..., description="List of task configurations")
    process_type: str = Field("sequential", description="Process type (sequential, hierarchical)")
    verbose: bool = Field(True, description="Verbose flag")
    created_at: str = Field(..., description="Creation timestamp")
    updated_at: str = Field(..., description="Last update timestamp")
    created_by: Optional[str] = Field(None, description="User who created the template")
    sla_seconds: Optional[int] = Field(None, description="SLA in seconds")
    hitl_triggers: Optional[List[str]] = Field(None, description="HITL trigger points")

class TemplateCreate(BaseModel):
    """Request model for creating a template."""
    name: str = Field(..., description="Template name", min_length=3, max_length=50)
    description: str = Field(..., description="Template description", min_length=10)
    agents: List[str] = Field(..., description="List of agent IDs")
    tasks: Optional[List[TaskConfig]] = Field(None, description="List of task configurations")
    process_type: str = Field("sequential", description="Process type (sequential, hierarchical)")
    verbose: bool = Field(True, description="Verbose flag")
    sla_seconds: Optional[int] = Field(None, description="SLA in seconds")
    hitl_triggers: Optional[List[str]] = Field(None, description="HITL trigger points")
    
    @validator('name')
    def name_must_be_valid(cls, v):
        if not v.replace('_', '').replace('-', '').isalnum():
            raise ValueError('name must contain only alphanumeric characters, underscores, and hyphens')
        return v
    
    @validator('process_type')
    def process_type_must_be_valid(cls, v):
        valid_types = ["sequential", "hierarchical"]
        if v not in valid_types:
            raise ValueError(f'process_type must be one of: {", ".join(valid_types)}')
        return v

class TemplateUpdate(BaseModel):
    """Request model for updating a template."""
    description: Optional[str] = Field(None, description="Template description", min_length=10)
    agents: Optional[List[str]] = Field(None, description="List of agent IDs")
    tasks: Optional[List[TaskConfig]] = Field(None, description="List of task configurations")
    process_type: Optional[str] = Field(None, description="Process type (sequential, hierarchical)")
    verbose: Optional[bool] = Field(None, description="Verbose flag")
    sla_seconds: Optional[int] = Field(None, description="SLA in seconds")
    hitl_triggers: Optional[List[str]] = Field(None, description="HITL trigger points")
    
    @validator('process_type')
    def process_type_must_be_valid(cls, v):
        if v is not None:
            valid_types = ["sequential", "hierarchical"]
            if v not in valid_types:
                raise ValueError(f'process_type must be one of: {", ".join(valid_types)}')
        return v

class TemplateRequestResponse(BaseModel):
    """Response model for a template request."""
    id: str = Field(..., description="Request ID")
    name: str = Field(..., description="Template name")
    use_case: str = Field(..., description="Description of the use case")
    sla_requirement: Optional[str] = Field(None, description="SLA requirement for the template")
    priority: str = Field(..., description="Priority of the template request")
    status: str = Field(..., description="Status of the request")
    created_at: str = Field(..., description="Creation timestamp")
    updated_at: str = Field(..., description="Last update timestamp")
    created_by: Optional[str] = Field(None, description="User who created the request")
    assigned_to: Optional[str] = Field(None, description="User assigned to the request")
    template_id: Optional[str] = Field(None, description="ID of the created template")

class TemplateRequestUpdate(BaseModel):
    """Request model for updating a template request."""
    status: Optional[str] = Field(None, description="Status of the request", pattern="^(pending|approved|rejected|completed)$")
    priority: Optional[str] = Field(None, description="Priority of the template request", pattern="^(low|medium|high|critical)$")
    assigned_to: Optional[str] = Field(None, description="User assigned to the request")
    template_id: Optional[str] = Field(None, description="ID of the created template")

class TemplateListResponse(BaseModel):
    """Response model for listing templates."""
    templates: List[TemplateResponse] = Field(..., description="List of templates")
    total: int = Field(..., description="Total number of templates")
    page: int = Field(1, description="Current page")
    page_size: int = Field(10, description="Page size")

class TemplateRequestListResponse(BaseModel):
    """Response model for listing template requests."""
    requests: List[TemplateRequestResponse] = Field(..., description="List of template requests")
    total: int = Field(..., description="Total number of template requests")
    page: int = Field(1, description="Current page")
    page_size: int = Field(10, description="Page size")

class TemplateSuggestion(BaseModel):
    """Model for a template suggestion."""
    name: str = Field(..., description="Suggested template name")
    description: str = Field(..., description="Suggested template description")
    agents: List[str] = Field(..., description="Suggested agent IDs")
    tools: List[str] = Field(..., description="Suggested tools")
    estimated_time: str = Field(..., description="Estimated time to run the template")
    confidence: float = Field(..., description="Confidence score for the suggestion")
    sla_seconds: Optional[int] = Field(None, description="Suggested SLA in seconds")

class TemplateSuggestionResponse(BaseModel):
    """Response model for template suggestions."""
    suggestions: List[TemplateSuggestion] = Field(..., description="List of template suggestions")

# Helper Functions
def get_template_path(template_id: str) -> Path:
    """Get the path to a template file."""
    return Path(AGENT_CONFIGS_CREWS_DIR) / f"{template_id}.yaml"

def template_exists(template_id: str) -> bool:
    """Check if a template exists."""
    return get_template_path(template_id).exists()

def load_template(template_id: str) -> Dict[str, Any]:
    """Load a template from file."""
    path = get_template_path(template_id)
    if not path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Template '{template_id}' not found"
        )
    
    try:
        with open(path, "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load template '{template_id}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load template: {str(e)}"
        )

def save_template(template_id: str, data: Dict[str, Any]) -> None:
    """Save a template to file."""
    path = get_template_path(template_id)
    
    try:
        # Ensure directory exists
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write template to file
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False)
    except Exception as e:
        logger.error(f"Failed to save template '{template_id}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save template: {str(e)}"
        )

def delete_template_file(template_id: str) -> None:
    """Delete a template file."""
    path = get_template_path(template_id)
    if not path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Template '{template_id}' not found"
        )
    
    try:
        path.unlink()
    except Exception as e:
        logger.error(f"Failed to delete template '{template_id}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete template: {str(e)}"
        )

def template_to_response(template_id: str, data: Dict[str, Any]) -> TemplateResponse:
    """Convert template data to a response model."""
    # Extract tasks
    tasks = []
    if "tasks" in data and data["tasks"]:
        for task in data["tasks"]:
            tasks.append(TaskConfig(
                description=task.get("description", ""),
                expected_output=task.get("expected_output", ""),
                agent_id=task.get("agent_id", "")
            ))
    
    # Create response
    return TemplateResponse(
        id=template_id,
        name=data.get("name", template_id),
        description=data.get("description", ""),
        agents=data.get("agents", []),
        tasks=tasks,
        process_type=data.get("process_type", "sequential"),
        verbose=data.get("verbose", True),
        created_at=data.get("created_at", ""),
        updated_at=data.get("updated_at", ""),
        created_by=data.get("created_by"),
        sla_seconds=data.get("sla_seconds"),
        hitl_triggers=data.get("hitl_triggers")
    )

async def generate_template_suggestions(use_case: str, template_name: str = None) -> List[TemplateSuggestion]:
    """
    Generate template suggestions based on a use case description.
    
    Args:
        use_case (str): Description of the use case.
        template_name (str, optional): Suggested template name. Defaults to None.
        
    Returns:
        List[TemplateSuggestion]: List of template suggestions.
    """
    # Create default suggestion
    default_suggestion = TemplateSuggestion(
        name=template_name or "default_investigation",
        description=f"Default investigation for: {use_case}",
        agents=["nlq_translator", "graph_analyst", "report_writer"],
        tools=["GraphQueryTool", "TemplateEngineTool"],
        estimated_time="5-10 minutes",
        confidence=0.7,
        sla_seconds=600  # 10 minutes
    )
    
    # Check for fraud keywords
    fraud_keywords = ["fraud", "scam", "money laundering", "suspicious", "criminal", "illegal", "theft", "stolen"]
    if any(keyword in use_case.lower() for keyword in fraud_keywords):
        fraud_suggestion = TemplateSuggestion(
            name=template_name or "fraud_investigation",
            description=f"Fraud investigation for: {use_case}",
            agents=["nlq_translator", "graph_analyst", "fraud_pattern_hunter", "compliance_checker", "report_writer"],
            tools=["GraphQueryTool", "PatternLibraryTool", "PolicyDocsTool", "TemplateEngineTool"],
            estimated_time="10-20 minutes",
            confidence=0.9,
            sla_seconds=1200  # 20 minutes
        )
        return [fraud_suggestion, default_suggestion]
    
    # Check for crypto keywords
    crypto_keywords = ["crypto", "bitcoin", "ethereum", "blockchain", "token", "wallet", "defi", "nft"]
    if any(keyword in use_case.lower() for keyword in crypto_keywords):
        crypto_suggestion = TemplateSuggestion(
            name=template_name or "crypto_investigation",
            description=f"Cryptocurrency investigation for: {use_case}",
            agents=["nlq_translator", "graph_analyst", "blockchain_detective", "report_writer"],
            tools=["GraphQueryTool", "CryptoAnomalyTool", "GraphQLQueryTool", "TemplateEngineTool"],
            estimated_time="15-25 minutes",
            confidence=0.85,
            sla_seconds=1500  # 25 minutes
        )
        return [crypto_suggestion, default_suggestion]
    
    # Check for analysis keywords
    analysis_keywords = ["analyze", "analysis", "investigate", "investigation", "examine", "review", "assess"]
    if any(keyword in use_case.lower() for keyword in analysis_keywords):
        analysis_suggestion = TemplateSuggestion(
            name=template_name or "detailed_analysis",
            description=f"Detailed analysis for: {use_case}",
            agents=["nlq_translator", "graph_analyst", "code_analyst", "report_writer"],
            tools=["GraphQueryTool", "CodeGenTool", "SandboxExecTool", "TemplateEngineTool"],
            estimated_time="15-25 minutes",
            confidence=0.75,
            sla_seconds=1500  # 25 minutes
        )
        return [analysis_suggestion, default_suggestion]
    
    # Check for compliance keywords
    compliance_keywords = ["compliance", "regulation", "policy", "aml", "kyc", "sanctions", "audit", "review"]
    if any(keyword in use_case.lower() for keyword in compliance_keywords):
        compliance_suggestion = TemplateSuggestion(
            name=template_name or "compliance_review",
            description=f"Compliance review for: {use_case}",
            agents=["nlq_translator", "graph_analyst", "compliance_checker", "report_writer"],
            tools=["GraphQueryTool", "PolicyDocsTool", "TemplateEngineTool"],
            estimated_time="10-20 minutes",
            confidence=0.85,
            sla_seconds=1200  # 20 minutes
        )
        return [compliance_suggestion, default_suggestion]
    
    # Return default suggestion if no specific keywords matched
    return [default_suggestion]

async def generate_template_from_use_case(use_case: str, template_name: str) -> Dict[str, Any]:
    """
    Generate a template configuration from a use case description.
    
    Args:
        use_case (str): Description of the use case.
        template_name (str): Name for the template.
        
    Returns:
        Dict[str, Any]: Template configuration.
    """
    # Get suggestions
    suggestions = await generate_template_suggestions(use_case, template_name)
    
    # Use the first suggestion
    suggestion = suggestions[0]
    
    # Create tasks
    tasks = []
    agent_map = {
        "nlq_translator": {
            "description": "Understand the user's query and translate it into a structured investigation plan.",
            "expected_output": "A structured investigation plan with entities, relationships, and potential patterns."
        },
        "graph_analyst": {
            "description": "Analyze the graph database to identify patterns and relationships between entities.",
            "expected_output": "A detailed analysis of patterns and relationships in the graph."
        },
        "fraud_pattern_hunter": {
            "description": "Identify specific fraud patterns in the data using known typologies.",
            "expected_output": "A list of detected fraud patterns with evidence and confidence scores."
        },
        "compliance_checker": {
            "description": "Evaluate the findings against compliance requirements and regulations.",
            "expected_output": "Compliance assessment and recommendations for regulatory considerations."
        },
        "report_writer": {
            "description": "Compile all findings into a comprehensive investigation report.",
            "expected_output": "A complete investigation report with all findings and recommendations."
        },
        "code_analyst": {
            "description": "Generate and execute Python code to perform advanced analysis on the data.",
            "expected_output": "Analysis results, visualizations, and statistical findings."
        },
        "blockchain_detective": {
            "description": "Analyze on-chain activity to identify suspicious patterns and transactions.",
            "expected_output": "Analysis of suspicious on-chain activities and patterns."
        }
    }
    
    for agent_id in suggestion.agents:
        if agent_id in agent_map:
            tasks.append({
                "description": agent_map[agent_id]["description"],
                "expected_output": agent_map[agent_id]["expected_output"],
                "agent_id": agent_id
            })
    
    # Create template
    from datetime import datetime
    timestamp = datetime.now().isoformat()
    
    template = {
        "name": template_name,
        "description": suggestion.description,
        "agents": suggestion.agents,
        "tasks": tasks,
        "process_type": "sequential",
        "verbose": True,
        "created_at": timestamp,
        "updated_at": timestamp,
        "sla_seconds": suggestion.sla_seconds,
        "hitl_triggers": ["compliance_issue", "high_risk"]
    }
    
    return template

# Endpoints
@router.get(
    "/suggestions",
    response_model=TemplateSuggestionResponse,
    summary="Get template suggestions",
    description="Get template suggestions based on a use case description."
)
async def get_template_suggestions(
    use_case: str,
    template_name: Optional[str] = None,
    roles: RoleSets = Depends(require_roles([Roles.ANALYST, Roles.ADMIN]))
):
    """
    Get template suggestions based on a use case description.
    
    Args:
        use_case (str): Description of the use case.
        template_name (str, optional): Suggested template name. Defaults to None.
        
    Returns:
        TemplateSuggestionResponse: Template suggestions.
    """
    try:
        suggestions = await generate_template_suggestions(use_case, template_name)
        return TemplateSuggestionResponse(suggestions=suggestions)
    except Exception as e:
        logger.error(f"Failed to generate template suggestions: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate template suggestions: {str(e)}"
        )

@router.get(
    "",
    response_model=TemplateListResponse,
    summary="List templates",
    description="List all templates."
)
async def list_templates(
    page: int = 1,
    page_size: int = 10,
    roles: RoleSets = Depends(require_roles([Roles.ANALYST, Roles.ADMIN]))
):
    """
    List all templates.
    
    Args:
        page (int, optional): Page number. Defaults to 1.
        page_size (int, optional): Page size. Defaults to 10.
        
    Returns:
        TemplateListResponse: List of templates.
    """
    try:
        # Get all template files
        template_files = list(Path(AGENT_CONFIGS_CREWS_DIR).glob("*.yaml"))
        
        # Calculate pagination
        total = len(template_files)
        start = (page - 1) * page_size
        end = start + page_size
        paginated_files = template_files[start:end]
        
        # Load templates
        templates = []
        for file in paginated_files:
            template_id = file.stem
            try:
                data = load_template(template_id)
                templates.append(template_to_response(template_id, data))
            except Exception as e:
                logger.warning(f"Failed to load template '{template_id}': {e}")
        
        return TemplateListResponse(
            templates=templates,
            total=total,
            page=page,
            page_size=page_size
        )
    except Exception as e:
        logger.error(f"Failed to list templates: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list templates: {str(e)}"
        )

@router.get(
    "/{template_id}",
    response_model=TemplateResponse,
    summary="Get template",
    description="Get a template by ID."
)
async def get_template(
    template_id: str,
    roles: RoleSets = Depends(require_roles([Roles.ANALYST, Roles.ADMIN]))
):
    """
    Get a template by ID.
    
    Args:
        template_id (str): Template ID.
        
    Returns:
        TemplateResponse: Template details.
    """
    try:
        data = load_template(template_id)
        return template_to_response(template_id, data)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get template '{template_id}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get template: {str(e)}"
        )

@router.post(
    "",
    response_model=TemplateResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create template",
    description="Create a new template."
)
async def create_template(
    template: TemplateCreate,
    request: Request,
    roles: RoleSets = Depends(require_roles([Roles.ADMIN]))
):
    """
    Create a new template.
    
    Args:
        template (TemplateCreate): Template data.
        
    Returns:
        TemplateResponse: Created template.
    """
    try:
        # Generate template ID from name
        template_id = template.name.lower().replace(" ", "_")
        
        # Check if template already exists
        if template_exists(template_id):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Template '{template_id}' already exists"
            )
        
        # Get user ID from request
        user_id = None
        if hasattr(request.state, "user") and hasattr(request.state.user, "id"):
            user_id = request.state.user.id
        
        # Create template data
        from datetime import datetime
        timestamp = datetime.now().isoformat()
        
        # Convert tasks
        tasks = []
        if template.tasks:
            for task in template.tasks:
                tasks.append({
                    "description": task.description,
                    "expected_output": task.expected_output,
                    "agent_id": task.agent_id
                })
        
        data = {
            "name": template.name,
            "description": template.description,
            "agents": template.agents,
            "tasks": tasks,
            "process_type": template.process_type,
            "verbose": template.verbose,
            "created_at": timestamp,
            "updated_at": timestamp,
            "created_by": user_id,
            "sla_seconds": template.sla_seconds,
            "hitl_triggers": template.hitl_triggers
        }
        
        # Save template
        save_template(template_id, data)
        
        # Reload CrewFactory to pick up new template
        factory = CrewFactory()
        factory.reload()
        
        return template_to_response(template_id, data)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create template: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create template: {str(e)}"
        )

@router.put(
    "/{template_id}",
    response_model=TemplateResponse,
    summary="Update template",
    description="Update an existing template."
)
async def update_template(
    template_id: str,
    template: TemplateUpdate,
    request: Request,
    roles: RoleSets = Depends(require_roles([Roles.ADMIN]))
):
    """
    Update an existing template.
    
    Args:
        template_id (str): Template ID.
        template (TemplateUpdate): Template data to update.
        
    Returns:
        TemplateResponse: Updated template.
    """
    try:
        # Check if template exists
        if not template_exists(template_id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Template '{template_id}' not found"
            )
        
        # Load existing template
        data = load_template(template_id)
        
        # Update fields
        if template.description is not None:
            data["description"] = template.description
        
        if template.agents is not None:
            data["agents"] = template.agents
        
        if template.tasks is not None:
            tasks = []
            for task in template.tasks:
                tasks.append({
                    "description": task.description,
                    "expected_output": task.expected_output,
                    "agent_id": task.agent_id
                })
            data["tasks"] = tasks
        
        if template.process_type is not None:
            data["process_type"] = template.process_type
        
        if template.verbose is not None:
            data["verbose"] = template.verbose
        
        if template.sla_seconds is not None:
            data["sla_seconds"] = template.sla_seconds
        
        if template.hitl_triggers is not None:
            data["hitl_triggers"] = template.hitl_triggers
        
        # Update timestamp
        from datetime import datetime
        data["updated_at"] = datetime.now().isoformat()
        
        # Save template
        save_template(template_id, data)
        
        # Reload CrewFactory to pick up updated template
        factory = CrewFactory()
        factory.reload()
        
        return template_to_response(template_id, data)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update template '{template_id}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update template: {str(e)}"
        )

@router.delete(
    "/{template_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete template",
    description="Delete a template."
)
async def delete_template(
    template_id: str,
    roles: RoleSets = Depends(require_roles([Roles.ADMIN]))
):
    """
    Delete a template.
    
    Args:
        template_id (str): Template ID.
    """
    try:
        # Delete template file
        delete_template_file(template_id)
        
        # Reload CrewFactory to remove deleted template
        factory = CrewFactory()
        factory.reload()
        
        return None
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete template '{template_id}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete template: {str(e)}"
        )

@router.post(
    "/request",
    response_model=TemplateRequestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create template request",
    description="Create a new template request."
)
async def create_template_request(
    template_request: TemplateRequest,
    request: Request,
    background_tasks: BackgroundTasks,
    roles: RoleSets = Depends(require_roles([Roles.ANALYST, Roles.ADMIN]))
):
    """
    Create a new template request.
    
    Args:
        template_request (TemplateRequest): Template request data.
        
    Returns:
        TemplateRequestResponse: Created template request.
    """
    try:
        # Generate request ID
        request_id = str(uuid.uuid4())
        
        # Get user ID from request
        user_id = None
        if hasattr(request.state, "user") and hasattr(request.state.user, "id"):
            user_id = request.state.user.id
        
        # Create request data
        from datetime import datetime
        timestamp = datetime.now().isoformat()
        
        # Set default priority if not provided
        priority = template_request.priority or "medium"
        
        data = {
            "id": request_id,
            "name": template_request.name,
            "use_case": template_request.use_case,
            "sla_requirement": template_request.sla_requirement,
            "priority": priority,
            "status": "pending",
            "created_at": timestamp,
            "updated_at": timestamp,
            "created_by": user_id,
            "assigned_to": None,
            "template_id": None
        }
        
        # Store request in database
        # Note: In a real implementation, this would be stored in a database
        # For now, we'll just return the data
        
        # Schedule background task to generate template
        if Roles.ADMIN in roles:
            # If user is admin, auto-approve and generate template
            background_tasks.add_task(
                auto_approve_and_generate_template,
                request_id,
                template_request.name,
                template_request.use_case,
                user_id
            )
        
        return TemplateRequestResponse(**data)
    except Exception as e:
        logger.error(f"Failed to create template request: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create template request: {str(e)}"
        )

async def auto_approve_and_generate_template(
    request_id: str,
    template_name: str,
    use_case: str,
    user_id: str
):
    """
    Auto-approve and generate a template from a request.
    
    Args:
        request_id (str): Request ID.
        template_name (str): Template name.
        use_case (str): Use case description.
        user_id (str): User ID.
    """
    try:
        # Generate template ID from name
        template_id = template_name.lower().replace(" ", "_")
        
        # Generate template
        template_data = await generate_template_from_use_case(use_case, template_name)
        
        # Add user ID
        template_data["created_by"] = user_id
        
        # Save template
        save_template(template_id, template_data)
        
        # Reload CrewFactory to pick up new template
        factory = CrewFactory()
        factory.reload()
        
        # Update request status
        # Note: In a real implementation, this would update the database
        logger.info(f"Auto-approved and generated template '{template_id}' for request '{request_id}'")
    except Exception as e:
        logger.error(f"Failed to auto-approve and generate template: {e}")
