from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import os
import yaml
import json
from pathlib import Path

from backend.auth.dependencies import require_admin
from backend.agents.factory import CrewFactory
from backend.core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(
    prefix="/prompts",
    tags=["prompts"],
    dependencies=[Depends(require_admin)],
)

# Path to agent configs directory
AGENT_CONFIGS_DIR = Path("backend/agents/configs")
AGENT_CONFIGS_CREWS_DIR = AGENT_CONFIGS_DIR / "crews"
DEFAULT_PROMPTS_DIR = AGENT_CONFIGS_DIR / "defaults"

# Ensure directories exist
AGENT_CONFIGS_DIR.mkdir(exist_ok=True)
AGENT_CONFIGS_CREWS_DIR.mkdir(exist_ok=True)
DEFAULT_PROMPTS_DIR.mkdir(exist_ok=True)

# In-memory cache of prompts
# This allows for runtime updates without file I/O for every agent instantiation
_prompt_cache: Dict[str, Dict[str, Any]] = {}


class PromptUpdate(BaseModel):
    """Model for updating an agent's prompt"""
    system_prompt: str = Field(..., description="The system prompt for the agent")
    description: Optional[str] = Field(None, description="Description of the prompt's purpose")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata for the prompt")


class PromptResponse(BaseModel):
    """Model for agent prompt response"""
    agent_id: str = Field(..., description="Unique identifier for the agent")
    system_prompt: str = Field(..., description="The system prompt for the agent")
    description: Optional[str] = Field(None, description="Description of the prompt's purpose")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata for the prompt")
    is_default: bool = Field(False, description="Whether this is the default prompt")


class AgentListItem(BaseModel):
    """Model for agent list item"""
    agent_id: str = Field(..., description="Unique identifier for the agent")
    description: Optional[str] = Field(None, description="Description of the agent")
    has_custom_prompt: bool = Field(False, description="Whether the agent has a custom prompt")


class AgentListResponse(BaseModel):
    """Model for agent list response"""
    agents: List[AgentListItem] = Field(..., description="List of available agents")


def _load_prompt_cache():
    """Load all prompts into memory cache"""
    global _prompt_cache
    
    # Clear existing cache
    _prompt_cache = {}
    
    # Load default prompts
    if DEFAULT_PROMPTS_DIR.exists():
        for prompt_file in DEFAULT_PROMPTS_DIR.glob("*.yaml"):
            try:
                agent_id = prompt_file.stem
                with open(prompt_file, "r") as f:
                    prompt_data = yaml.safe_load(f)
                    if not prompt_data:
                        prompt_data = {}
                    prompt_data["is_default"] = True
                    _prompt_cache[agent_id] = prompt_data
            except Exception as e:
                logger.error(f"Error loading default prompt for {agent_id}: {str(e)}")
    
    # Load custom prompts (overriding defaults)
    if AGENT_CONFIGS_DIR.exists():
        for prompt_file in AGENT_CONFIGS_DIR.glob("*.yaml"):
            try:
                agent_id = prompt_file.stem
                with open(prompt_file, "r") as f:
                    prompt_data = yaml.safe_load(f)
                    if not prompt_data:
                        prompt_data = {}
                    prompt_data["is_default"] = False
                    _prompt_cache[agent_id] = prompt_data
            except Exception as e:
                logger.error(f"Error loading custom prompt for {agent_id}: {str(e)}")
    
    # Also check crew configs for agent definitions
    if AGENT_CONFIGS_CREWS_DIR.exists():
        for crew_file in AGENT_CONFIGS_CREWS_DIR.glob("*.yaml"):
            try:
                with open(crew_file, "r") as f:
                    crew_data = yaml.safe_load(f)
                    if crew_data and "agents" in crew_data:
                        for agent in crew_data["agents"]:
                            if isinstance(agent, dict) and "id" in agent:
                                agent_id = agent["id"]
                                if agent_id not in _prompt_cache:
                                    # Just register the agent ID, actual prompt will be loaded on demand
                                    _prompt_cache[agent_id] = {
                                        "system_prompt": "",
                                        "description": f"Agent from crew {crew_file.stem}",
                                        "is_default": True
                                    }
            except Exception as e:
                logger.error(f"Error loading agents from crew file {crew_file}: {str(e)}")
    
    logger.info(f"Loaded {len(_prompt_cache)} agent prompts into cache")


# Initialize the prompt cache
_load_prompt_cache()


def _get_agent_prompt(agent_id: str) -> Dict[str, Any]:
    """Get an agent's prompt data, loading from CrewFactory if not in cache"""
    if agent_id not in _prompt_cache:
        # Try to get the prompt from CrewFactory
        try:
            # This assumes CrewFactory has a method to get agent configs
            agent_config = CrewFactory.get_agent_config(agent_id)
            if agent_config and "system_prompt" in agent_config:
                _prompt_cache[agent_id] = {
                    "system_prompt": agent_config["system_prompt"],
                    "description": agent_config.get("description", f"Agent {agent_id}"),
                    "is_default": True,
                }
            else:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Agent prompt not found: {agent_id}"
                )
        except Exception as e:
            logger.error(f"Error getting agent config from CrewFactory: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Agent prompt not found: {agent_id}"
            )
    
    return _prompt_cache[agent_id]


def _save_agent_prompt(agent_id: str, prompt_data: Dict[str, Any]):
    """Save an agent's prompt to file and update cache"""
    # Update cache
    prompt_data["is_default"] = False
    _prompt_cache[agent_id] = prompt_data
    
    # Save to file
    prompt_file = AGENT_CONFIGS_DIR / f"{agent_id}.yaml"
    try:
        with open(prompt_file, "w") as f:
            # Remove is_default before saving
            save_data = {k: v for k, v in prompt_data.items() if k != "is_default"}
            yaml.dump(save_data, f)
        logger.info(f"Saved custom prompt for agent {agent_id}")
        return True
    except Exception as e:
        logger.error(f"Error saving prompt for {agent_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save prompt: {str(e)}"
        )


def _reset_agent_prompt(agent_id: str):
    """Reset an agent's prompt to default"""
    # Check if there's a default prompt
    default_file = DEFAULT_PROMPTS_DIR / f"{agent_id}.yaml"
    if not default_file.exists():
        # No default file, try to get from CrewFactory
        try:
            agent_config = CrewFactory.get_agent_config(agent_id)
            if not agent_config or "system_prompt" not in agent_config:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"No default prompt found for agent: {agent_id}"
                )
            # Update cache with default from CrewFactory
            _prompt_cache[agent_id] = {
                "system_prompt": agent_config["system_prompt"],
                "description": agent_config.get("description", f"Agent {agent_id}"),
                "is_default": True,
            }
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error getting default config from CrewFactory: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to reset prompt: {str(e)}"
            )
    else:
        # Load default from file
        try:
            with open(default_file, "r") as f:
                default_data = yaml.safe_load(f)
                if not default_data:
                    default_data = {}
                default_data["is_default"] = True
                _prompt_cache[agent_id] = default_data
        except Exception as e:
            logger.error(f"Error loading default prompt for {agent_id}: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to load default prompt: {str(e)}"
            )
    
    # Remove custom prompt file if it exists
    custom_file = AGENT_CONFIGS_DIR / f"{agent_id}.yaml"
    if custom_file.exists():
        try:
            os.remove(custom_file)
            logger.info(f"Removed custom prompt for agent {agent_id}")
        except Exception as e:
            logger.error(f"Error removing custom prompt file for {agent_id}: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to remove custom prompt file: {str(e)}"
            )
    
    return _prompt_cache[agent_id]


@router.get("", response_model=AgentListResponse)
async def list_agents():
    """List all available agents and their prompt status"""
    # Refresh cache to ensure we have the latest data
    _load_prompt_cache()
    
    agents = []
    for agent_id, prompt_data in _prompt_cache.items():
        agents.append(AgentListItem(
            agent_id=agent_id,
            description=prompt_data.get("description", f"Agent {agent_id}"),
            has_custom_prompt=not prompt_data.get("is_default", True)
        ))
    
    # Sort alphabetically by agent_id
    agents.sort(key=lambda x: x.agent_id)
    
    return AgentListResponse(agents=agents)


@router.get("/{agent_id}", response_model=PromptResponse)
async def get_agent_prompt(agent_id: str):
    """Get a specific agent's prompt configuration"""
    try:
        prompt_data = _get_agent_prompt(agent_id)
        return PromptResponse(
            agent_id=agent_id,
            system_prompt=prompt_data.get("system_prompt", ""),
            description=prompt_data.get("description", f"Agent {agent_id}"),
            metadata=prompt_data.get("metadata", {}),
            is_default=prompt_data.get("is_default", True)
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving prompt for {agent_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve prompt: {str(e)}"
        )


@router.put("/{agent_id}", response_model=PromptResponse)
async def update_agent_prompt(agent_id: str, prompt_update: PromptUpdate):
    """Update an agent's prompt"""
    # First check if the agent exists
    try:
        existing_data = _get_agent_prompt(agent_id)
    except HTTPException:
        raise
    
    # Prepare updated data
    updated_data = {
        "system_prompt": prompt_update.system_prompt,
        "description": prompt_update.description or existing_data.get("description", f"Agent {agent_id}"),
        "metadata": prompt_update.metadata or existing_data.get("metadata", {})
    }
    
    # Save the updated prompt
    _save_agent_prompt(agent_id, updated_data)
    
    # Notify CrewFactory to update any cached agents
    try:
        CrewFactory.update_agent_prompt(agent_id, prompt_update.system_prompt)
    except Exception as e:
        logger.warning(f"Failed to update CrewFactory cache for {agent_id}: {str(e)}")
    
    # Return the updated prompt
    return PromptResponse(
        agent_id=agent_id,
        system_prompt=updated_data["system_prompt"],
        description=updated_data["description"],
        metadata=updated_data["metadata"],
        is_default=False
    )


@router.post("/{agent_id}/reset", response_model=PromptResponse)
async def reset_agent_prompt(agent_id: str):
    """Reset an agent's prompt to default"""
    try:
        prompt_data = _reset_agent_prompt(agent_id)
        
        # Notify CrewFactory to update any cached agents
        try:
            CrewFactory.reset_agent_prompt(agent_id)
        except Exception as e:
            logger.warning(f"Failed to reset CrewFactory cache for {agent_id}: {str(e)}")
        
        return PromptResponse(
            agent_id=agent_id,
            system_prompt=prompt_data.get("system_prompt", ""),
            description=prompt_data.get("description", f"Agent {agent_id}"),
            metadata=prompt_data.get("metadata", {}),
            is_default=True
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error resetting prompt for {agent_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to reset prompt: {str(e)}"
        )
