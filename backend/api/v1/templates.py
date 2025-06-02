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
    agent: str
    expected_output: Optional[str] = None
    context: Optional[List[str]] = None
    async_execution: bool = False

class TemplateBase(BaseModel):
    """Base model for template data."""
    name: str
    description: str
    agents: List[AgentConfig]
    tasks: List[TaskConfig]
    process_type: str = "sequential"
    verbose: bool = True
    memory: bool = False
    cache: bool = True
    max_rpm: Optional[int] = None
    sla_seconds: Optional[int] = None
    hitl_triggers: Optional[List[str]] = None

    @validator("name")
    def validate_name(cls, v):
        """Validate template name (alphanumeric with underscores)."""
        if not v or not v.strip():
            raise ValueError("Template name cannot be empty")
        if not all(c.isalnum() or c == '_' for c in v):
            raise ValueError("Template name must contain only alphanumeric characters and underscores")
        return v
    
    @validator("process_type")
    def validate_process_type(cls, v):
        """Validate process type is one of the allowed values."""
        if v not in ["sequential", "hierarchical"]:
            raise ValueError(f"Process type must be 'sequential' or 'hierarchical', got '{v}'")
        return v

class TemplateCreate(TemplateBase):
    """Model for creating a new template."""
    pass

class TemplateUpdate(TemplateBase):
    """Model for updating an existing template."""
    pass

class TemplateResponse(TemplateBase):
    """Response model for template operations."""
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

class TemplateListResponse(BaseModel):
    """Response model for listing templates."""
    templates: List[TemplateResponse]

class TemplateRequestBase(BaseModel):
    """Base model for template request."""
    name: str
    use_case: str
    sla_requirement: Optional[str] = None

class TemplateRequestCreate(TemplateRequestBase):
    """Model for creating a new template request."""
    pass

class TemplateRequestResponse(TemplateRequestBase):
    """Response model for template request operations."""
    request_id: str
    status: str = "pending"
    created_at: str

class TemplateSuggestion(BaseModel):
    """Model for template suggestions."""
    name: str
    description: str
    agents: List[str]
    tools: List[str]
    estimated_time: str
    confidence: float
    sla_seconds: Optional[int] = None

class TemplateSuggestionResponse(BaseModel):
    """Response model for template suggestions."""
    suggestions: List[TemplateSuggestion]
    use_case: str

# Dependency to get CrewFactory
async def get_crew_factory(request: Request) -> CrewFactory:
    """Get or create a CrewFactory instance."""
    if not hasattr(request.app.state, "crew_factory"):
        logger.info("Creating new CrewFactory instance")
        request.app.state.crew_factory = CrewFactory()
    return request.app.state.crew_factory

# Dependency to get GeminiClient
async def get_gemini_client(request: Request) -> GeminiClient:
    """Get or create a GeminiClient instance."""
    if not hasattr(request.app.state, "gemini_client"):
        logger.info("Creating new GeminiClient instance")
        request.app.state.gemini_client = GeminiClient()
    return request.app.state.gemini_client

@router.get("")
@require_roles(RoleSets.ANALYSTS_AND_ADMIN)
async def list_templates():
    """
    List all available templates.
    
    Returns:
        List of template names and configurations
    """
    try:
        templates = []
        
        # Get all template files
        template_files = list(AGENT_CONFIGS_CREWS_DIR.glob("*.yaml"))
        
        # Load each template
        for template_file in template_files:
            try:
                with open(template_file, "r") as f:
                    template_data = yaml.safe_load(f)
                    
                    # Extract metadata
                    template_data["created_at"] = template_data.get("created_at", None)
                    template_data["updated_at"] = template_data.get("updated_at", None)
                    
                    # Add to list
                    templates.append(TemplateResponse(**template_data))
            except Exception as e:
                logger.error(f"Failed to load template {template_file}: {e}")
        
        return TemplateListResponse(templates=templates)
    except Exception as e:
        logger.error(f"Failed to list templates: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list templates: {str(e)}"
        )

@router.get("/{template_name}")
@require_roles(RoleSets.ANALYSTS_AND_ADMIN)
async def get_template(template_name: str):
    """
    Get a specific template by name.
    
    Args:
        template_name: Name of the template to get
        
    Returns:
        Template configuration
    """
    try:
        # Check if template exists
        template_path = AGENT_CONFIGS_CREWS_DIR / f"{template_name}.yaml"
        if not template_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Template not found: {template_name}"
            )
        
        # Load template
        with open(template_path, "r") as f:
            template_data = yaml.safe_load(f)
            
            # Extract metadata
            template_data["created_at"] = template_data.get("created_at", None)
            template_data["updated_at"] = template_data.get("updated_at", None)
            
            return TemplateResponse(**template_data)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get template {template_name}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get template {template_name}: {str(e)}"
        )

@router.post("")
@require_roles(RoleSets.ADMIN)
async def create_template(
    template: TemplateCreate,
    crew_factory: CrewFactory = Depends(get_crew_factory)
):
    """
    Create a new template.
    
    Args:
        template: Template configuration
        
    Returns:
        Created template
    """
    try:
        # Check if template already exists
        template_path = AGENT_CONFIGS_CREWS_DIR / f"{template.name}.yaml"
        if template_path.exists():
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Template already exists: {template.name}"
            )
        
        # Convert to dict for YAML serialization
        template_dict = template.dict()
        
        # Add metadata
        from datetime import datetime
        now = datetime.now().isoformat()
        template_dict["created_at"] = now
        template_dict["updated_at"] = now
        
        # Ensure crew_name is set
        template_dict["crew_name"] = template.name
        
        # Save template
        with open(template_path, "w") as f:
            yaml.dump(template_dict, f, sort_keys=False)
        
        logger.info(f"Created template: {template.name}")
        
        # Reload factory to pick up new template
        reload_result = crew_factory.reload()
        logger.info(f"Reloaded CrewFactory: {reload_result}")
        
        return TemplateResponse(**template_dict)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create template: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create template: {str(e)}"
        )

@router.put("/{template_name}")
@require_roles(RoleSets.ADMIN)
async def update_template(
    template_name: str,
    template: TemplateUpdate,
    crew_factory: CrewFactory = Depends(get_crew_factory)
):
    """
    Update an existing template.
    
    Args:
        template_name: Name of the template to update
        template: Updated template configuration
        
    Returns:
        Updated template
    """
    try:
        # Check if template exists
        template_path = AGENT_CONFIGS_CREWS_DIR / f"{template_name}.yaml"
        if not template_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Template not found: {template_name}"
            )
        
        # Load existing template to preserve metadata
        with open(template_path, "r") as f:
            existing_data = yaml.safe_load(f)
        
        # Convert to dict for YAML serialization
        template_dict = template.dict()
        
        # Preserve creation date
        template_dict["created_at"] = existing_data.get("created_at", None)
        
        # Update modification date
        from datetime import datetime
        template_dict["updated_at"] = datetime.now().isoformat()
        
        # Ensure crew_name is set
        template_dict["crew_name"] = template.name
        
        # Save template
        with open(template_path, "w") as f:
            yaml.dump(template_dict, f, sort_keys=False)
        
        logger.info(f"Updated template: {template_name}")
        
        # Reload factory to pick up updated template
        reload_result = crew_factory.reload()
        logger.info(f"Reloaded CrewFactory: {reload_result}")
        
        return TemplateResponse(**template_dict)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update template {template_name}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update template {template_name}: {str(e)}"
        )

@router.delete("/{template_name}")
@require_roles(RoleSets.ADMIN)
async def delete_template(
    template_name: str,
    crew_factory: CrewFactory = Depends(get_crew_factory)
):
    """
    Delete a template.
    
    Args:
        template_name: Name of the template to delete
        
    Returns:
        Success message
    """
    try:
        # Check if template exists
        template_path = AGENT_CONFIGS_CREWS_DIR / f"{template_name}.yaml"
        if not template_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Template not found: {template_name}"
            )
        
        # Delete template
        template_path.unlink()
        
        logger.info(f"Deleted template: {template_name}")
        
        # Reload factory to remove deleted template
        reload_result = crew_factory.reload()
        logger.info(f"Reloaded CrewFactory: {reload_result}")
        
        return {"success": True, "message": f"Template deleted: {template_name}"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete template {template_name}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete template {template_name}: {str(e)}"
        )

@router.post("/request")
@require_roles(RoleSets.ANALYSTS_AND_ADMIN)
async def request_template(
    request: TemplateRequestCreate,
    background_tasks: BackgroundTasks,
    gemini_client: GeminiClient = Depends(get_gemini_client)
):
    """
    Request a new template based on a use case.
    
    This endpoint initiates a background task to generate template suggestions
    based on the provided use case.
    
    Args:
        request: Template request with use case
        
    Returns:
        Template request details
    """
    try:
        # Generate a request ID
        import uuid
        request_id = str(uuid.uuid4())
        
        # Create response
        from datetime import datetime
        response = TemplateRequestResponse(
            request_id=request_id,
            name=request.name,
            use_case=request.use_case,
            sla_requirement=request.sla_requirement,
            status="processing",
            created_at=datetime.now().isoformat()
        )
        
        # Add background task to generate suggestions
        background_tasks.add_task(
            generate_template_suggestions,
            request.use_case,
            request.name,
            request.sla_requirement,
            gemini_client
        )
        
        logger.info(f"Template request created: {request_id}")
        
        return response
    except Exception as e:
        logger.error(f"Failed to create template request: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create template request: {str(e)}"
        )

@router.get("/suggestions")
@require_roles(RoleSets.ANALYSTS_AND_ADMIN)
async def get_suggestions(
    use_case: str,
    gemini_client: GeminiClient = Depends(get_gemini_client)
):
    """
    Get suggestions for template configuration based on a use case.
    
    Args:
        use_case: Description of the use case
        
    Returns:
        Suggested template configurations
    """
    try:
        # Generate suggestions
        suggestions = await generate_template_suggestions(use_case, None, None, gemini_client)
        
        return TemplateSuggestionResponse(
            suggestions=suggestions,
            use_case=use_case
        )
    except Exception as e:
        logger.error(f"Failed to get template suggestions: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get template suggestions: {str(e)}"
        )

async def generate_template_suggestions(
    use_case: str,
    template_name: Optional[str] = None,
    sla_requirement: Optional[str] = None,
    gemini_client: Optional[GeminiClient] = None
) -> List[TemplateSuggestion]:
    """
    Generate template suggestions based on a use case using Gemini.
    
    Args:
        use_case: Description of the use case
        template_name: Optional name for the template
        sla_requirement: Optional SLA requirement
        gemini_client: GeminiClient instance
        
    Returns:
        List of template suggestions
    """
    try:
        # Create GeminiClient if not provided
        if gemini_client is None:
            gemini_client = GeminiClient()
        
        # Prepare prompt for Gemini
        prompt = f"""
        You are an expert financial crime investigation template designer. 
        Create a template configuration for the following use case:
        
        USE CASE: {use_case}
        
        {f'TEMPLATE NAME: {template_name}' if template_name else ''}
        {f'SLA REQUIREMENT: {sla_requirement}' if sla_requirement else ''}
        
        Your task is to suggest an optimal template configuration including:
        1. A descriptive name (if not provided)
        2. A list of appropriate agent roles
        3. Recommended tools for each agent
        4. Estimated time to complete
        5. Confidence score (0.0-1.0)
        
        Available agents:
        - nlq_translator: Converts natural language to Cypher queries
        - graph_analyst: Analyzes graph data and runs algorithms
        - fraud_pattern_hunter: Identifies fraud patterns and anomalies
        - compliance_checker: Verifies compliance with regulations
        - report_writer: Generates investigation reports
        - crypto_data_collector: Collects data from blockchain sources
        - blockchain_detective: Analyzes blockchain transactions
        - defi_analyst: Specializes in DeFi protocol analysis
        - code_analyst: Writes and executes Python code
        
        Available tools:
        - GraphQueryTool: Executes Cypher queries against Neo4j
        - PatternLibraryTool: Matches known fraud patterns
        - PolicyDocsTool: Retrieves relevant policy documents
        - CodeGenTool: Generates and executes Python code
        - SandboxExecTool: Executes code in a sandbox
        - TemplateEngineTool: Generates reports from templates
        - Neo4jSchemaTool: Retrieves database schema
        - CryptoCSVLoaderTool: Loads crypto transaction data
        - CryptoAnomalyTool: Detects anomalies in crypto data
        - GNNFraudDetectionTool: Uses graph neural networks
        
        Return your response as a JSON object with the following structure:
        {
          "suggestions": [
            {
              "name": "template_name",
              "description": "Template description",
              "agents": ["agent1", "agent2"],
              "tools": ["tool1", "tool2"],
              "estimated_time": "X minutes/hours",
              "confidence": 0.85,
              "sla_seconds": 300
            }
          ]
        }
        
        Provide 1-3 suggestions with different approaches to the problem.
        """
        
        # Generate suggestions using Gemini
        response = await gemini_client.generate_text(prompt)
        
        # Parse JSON response
        try:
            # Extract JSON from response (it might be wrapped in markdown)
            import re
            json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = response
            
            # Clean up the string to ensure it's valid JSON
            json_str = json_str.strip()
            if json_str.startswith("```") and json_str.endswith("```"):
                json_str = json_str[3:-3].strip()
            
            # Parse JSON
            data = json.loads(json_str)
            
            # Extract suggestions
            suggestions = []
            for suggestion in data.get("suggestions", []):
                suggestions.append(TemplateSuggestion(**suggestion))
            
            return suggestions
        except Exception as e:
            logger.error(f"Failed to parse Gemini response: {e}")
            logger.debug(f"Raw response: {response}")
            
            # Fallback to heuristic suggestions if parsing fails
            return generate_heuristic_suggestions(use_case, template_name, sla_requirement)
    
    except Exception as e:
        logger.error(f"Failed to generate template suggestions: {e}")
        
        # Fallback to heuristic suggestions
        return generate_heuristic_suggestions(use_case, template_name, sla_requirement)

def generate_heuristic_suggestions(
    use_case: str,
    template_name: Optional[str] = None,
    sla_requirement: Optional[str] = None
) -> List[TemplateSuggestion]:
    """
    Generate template suggestions based on heuristics (fallback method).
    
    Args:
        use_case: Description of the use case
        template_name: Optional name for the template
        sla_requirement: Optional SLA requirement
        
    Returns:
        List of template suggestions
    """
    logger.info("Using heuristic suggestion generation (fallback)")
    
    # Default suggestion
    default_suggestion = TemplateSuggestion(
        name=template_name or "custom_investigation",
        description=f"Custom investigation for: {use_case}",
        agents=["nlq_translator", "graph_analyst", "fraud_pattern_hunter", "compliance_checker", "report_writer"],
        tools=["GraphQueryTool", "PatternLibraryTool", "PolicyDocsTool", "TemplateEngineTool"],
        estimated_time="15-30 minutes",
        confidence=0.7,
        sla_seconds=1800  # 30 minutes
    )
    
    # Check for crypto keywords
    crypto_keywords = ["crypto", "bitcoin", "ethereum", "blockchain", "wallet", "transaction", "mixer", "defi"]
    if any(keyword in use_case.lower() for keyword in crypto_keywords):
        crypto_suggestion = TemplateSuggestion(
            name=template_name or "crypto_investigation",
            description=f"Cryptocurrency investigation for: {use_case}",
            agents=["nlq_translator", "blockchain_detective", "defi_analyst", "compliance_checker", "report_writer"],
            tools=["GraphQueryTool", "CryptoCSVLoaderTool", "PatternLibraryTool", "PolicyDocsTool", "TemplateEngineTool"],
            estimated_time="20-40 minutes",
            confidence=0.8,
            sla_seconds=2400  # 40 minutes
        )
        return [crypto_suggestion, default_suggestion]
    
    # Check for code/data analysis keywords
    analysis_keywords = ["analyze", "data", "statistics", "visualization", "chart", "graph", "code", "python"]
    if any(keyword in use_case.lower() for keyword in analysis_keywords):
        analysis_suggestion = TemplateSuggestion(
            name=template_name or "data_analysis",
            description=f"Data analysis investigation for: {use_case}",
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
