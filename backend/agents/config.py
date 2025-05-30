"""
Configuration module for agents and crews.

This module provides classes and functions for loading and managing
agent and crew configurations from YAML files.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Literal
import logging
from pydantic import BaseModel, Field, validator

# Configure logging
logger = logging.getLogger(__name__)

# Path to agent configs directory
AGENT_CONFIGS_DIR = Path("backend/agents/configs")
AGENT_CONFIGS_CREWS_DIR = AGENT_CONFIGS_DIR / "crews"
DEFAULT_PROMPTS_DIR = AGENT_CONFIGS_DIR / "defaults"

# Ensure directories exist
AGENT_CONFIGS_DIR.mkdir(exist_ok=True)
AGENT_CONFIGS_CREWS_DIR.mkdir(exist_ok=True)
DEFAULT_PROMPTS_DIR.mkdir(exist_ok=True)


class ToolConfig(BaseModel):
    """Configuration for a tool used by an agent."""
    
    type: str
    timeout_seconds: Optional[int] = None
    max_tokens: Optional[int] = None
    
    class Config:
        """Pydantic config."""
        
        extra = "allow"  # Allow extra fields for tool-specific configuration


class LLMConfig(BaseModel):
    """Configuration for an LLM provider."""
    
    model: str
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    
    class Config:
        """Pydantic config."""
        
        extra = "allow"  # Allow extra fields for provider-specific configuration


class AgentConfig(BaseModel):
    """Configuration for a CrewAI agent."""
    
    id: str
    role: str
    goal: str
    backstory: Optional[str] = None
    system_prompt: Optional[str] = None
    tools: List[Union[str, ToolConfig]] = []
    llm_model: Optional[str] = None
    llm: Optional[Union[str, LLMConfig]] = None
    allow_delegation: bool = False
    max_iter: int = 15
    max_rpm: Optional[int] = None
    verbose: bool = True
    memory: bool = False
    multimodal: bool = False
    
    class Config:
        """Pydantic config."""
        
        extra = "allow"  # Allow extra fields for future extensions


class TaskConfig(BaseModel):
    """Configuration for a task in a crew."""
    
    description: str
    agent: str
    expected_output: Optional[str] = None
    context: Optional[List[str]] = None
    async_execution: bool = False


class CrewConfig(BaseModel):
    """Configuration for a CrewAI crew."""
    
    crew_name: str
    process_type: Literal["sequential", "hierarchical"] = "sequential"
    manager: Optional[str] = None
    agents: List[Union[str, Dict[str, Any]]]
    tasks: Optional[List[TaskConfig]] = None
    memory: bool = False
    verbose: bool = True
    max_rpm: Optional[int] = None
    cache: bool = True
    
    @validator("process_type")
    def validate_process_type(cls, v: str) -> str:
        """Validate process_type is one of the allowed values."""
        if v not in ["sequential", "hierarchical"]:
            raise ValueError(f"process_type must be 'sequential' or 'hierarchical', got '{v}'")
        return v
    
    class Config:
        """Pydantic config."""
        
        extra = "allow"  # Allow extra fields for future extensions


# Default agent configurations
DEFAULT_AGENT_CONFIGS: Dict[str, Dict[str, Any]] = {}

# Load default agent configurations
for config_file in DEFAULT_PROMPTS_DIR.glob("*.yaml"):
    try:
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
            if config and "id" in config:
                DEFAULT_AGENT_CONFIGS[config["id"]] = config
                logger.debug(f"Loaded default agent config for {config['id']}")
    except Exception as e:
        logger.error(f"Error loading default agent config {config_file}: {str(e)}")


def load_agent_config(agent_id: str) -> AgentConfig:
    """
    Load an agent configuration by ID.
    
    Args:
        agent_id: The unique identifier for the agent
        
    Returns:
        The agent configuration
        
    Raises:
        ValueError: If the agent configuration is not found
    """
    # Check custom config first
    custom_config_path = AGENT_CONFIGS_DIR / f"{agent_id}.yaml"
    if custom_config_path.exists():
        try:
            with open(custom_config_path, "r") as f:
                config_data = yaml.safe_load(f)
                if config_data:
                    return AgentConfig(**config_data)
        except Exception as e:
            logger.error(f"Error loading custom config for {agent_id}: {str(e)}")
    
    # Check default config
    default_config_path = DEFAULT_PROMPTS_DIR / f"{agent_id}.yaml"
    if default_config_path.exists():
        try:
            with open(default_config_path, "r") as f:
                config_data = yaml.safe_load(f)
                if config_data:
                    return AgentConfig(**config_data)
        except Exception as e:
            logger.error(f"Error loading default config for {agent_id}: {str(e)}")
    
    # Check in crew configs
    for crew_file in AGENT_CONFIGS_CREWS_DIR.glob("*.yaml"):
        try:
            with open(crew_file, "r") as f:
                crew_data = yaml.safe_load(f)
                if crew_data and "agents" in crew_data:
                    for agent in crew_data["agents"]:
                        if isinstance(agent, dict) and agent.get("id") == agent_id:
                            return AgentConfig(**agent)
        except Exception as e:
            logger.error(f"Error loading agents from crew file {crew_file}: {str(e)}")
    
    # Check default configs dictionary
    if agent_id in DEFAULT_AGENT_CONFIGS:
        return AgentConfig(**DEFAULT_AGENT_CONFIGS[agent_id])
    
    # Not found
    raise ValueError(f"Agent configuration not found: {agent_id}")


def load_crew_config(crew_name: str) -> CrewConfig:
    """
    Load a crew configuration by name.
    
    Args:
        crew_name: The name of the crew
        
    Returns:
        The crew configuration
        
    Raises:
        ValueError: If the crew configuration is not found
    """
    # Check crew config
    crew_config_path = AGENT_CONFIGS_CREWS_DIR / f"{crew_name}.yaml"
    if crew_config_path.exists():
        try:
            with open(crew_config_path, "r") as f:
                config_data = yaml.safe_load(f)
                if config_data:
                    # Ensure crew_name is set
                    if "crew_name" not in config_data:
                        config_data["crew_name"] = crew_name
                    return CrewConfig(**config_data)
        except Exception as e:
            logger.error(f"Error loading crew config for {crew_name}: {str(e)}")
    
    # Not found
    raise ValueError(f"Crew configuration not found: {crew_name}")


def get_available_crews() -> List[str]:
    """
    Get a list of available crew names.
    
    Returns:
        List of crew names
    """
    crews = []
    for crew_file in AGENT_CONFIGS_CREWS_DIR.glob("*.yaml"):
        crews.append(crew_file.stem)
    return crews


def get_available_agents() -> List[str]:
    """
    Get a list of available agent IDs.
    
    Returns:
        List of agent IDs
    """
    agents = list(DEFAULT_AGENT_CONFIGS.keys())
    
    # Add agents from custom configs
    for agent_file in AGENT_CONFIGS_DIR.glob("*.yaml"):
        if agent_file.stem not in agents:
            agents.append(agent_file.stem)
    
    # Add agents from default configs
    for agent_file in DEFAULT_PROMPTS_DIR.glob("*.yaml"):
        if agent_file.stem not in agents:
            agents.append(agent_file.stem)
    
    return agents
