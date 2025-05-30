"""
Configuration management for CrewAI agents and crews.

This module provides configuration classes and utilities for defining
agent roles, goals, backstories, and tool access. It supports loading
configurations from YAML files and provides defaults for all agent types.
"""

import os
import yaml
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
from pydantic import BaseModel, Field, validator

# Base directory for agent configuration files
AGENT_CONFIG_DIR = Path("backend/agents/configs")


class ToolConfig(BaseModel):
    """Configuration for a tool that can be used by an agent."""
    
    name: str
    description: str
    parameters: Optional[Dict[str, Any]] = None
    required: bool = True


class AgentConfig(BaseModel):
    """Configuration for a CrewAI agent."""
    
    id: str
    role: str
    goal: str
    backstory: str
    verbose: bool = False
    allow_delegation: bool = False
    tools: List[str] = []
    max_iter: int = 5
    max_rpm: Optional[int] = None
    memory: bool = False
    llm_model: str = "gemini-1.5-pro"
    
    @validator('id')
    def validate_id(cls, v):
        """Ensure agent ID is valid."""
        if not v or not v.strip():
            raise ValueError("Agent ID cannot be empty")
        return v.strip()


class CrewConfig(BaseModel):
    """Configuration for a CrewAI crew."""
    
    crew_name: str
    process_type: str = "sequential"  # sequential or hierarchical
    manager: Optional[str] = None
    agents: List[str] = []
    tasks: List[str] = []
    verbose: bool = False
    max_rpm: Optional[int] = None
    memory: bool = False
    cache: bool = True


# Default agent configurations
DEFAULT_AGENT_CONFIGS = {
    "orchestrator_manager": {
        "id": "orchestrator_manager",
        "role": "Workflow Coordinator",
        "goal": "Coordinate the overall analysis workflow and ensure quality results",
        "backstory": """You are an experienced workflow coordinator with expertise in financial 
        crime investigation. Your job is to initialize the investigation process, ensure all 
        required inputs are validated, and coordinate the final outputs. You do not assign tasks 
        dynamically as they follow a predefined sequence, but you ensure the overall process 
        quality and completeness.""",
        "tools": [],
        "memory": True,
        "max_iter": 3,
    },
    "nlq_translator": {
        "id": "nlq_translator",
        "role": "Natural Language to Cypher Translator",
        "goal": "Convert natural language questions into optimized Cypher queries",
        "backstory": """You are a specialist in translating human language into precise database 
        queries. With deep knowledge of graph databases and the Neo4j Cypher query language, you 
        excel at understanding analyst questions and converting them into efficient queries that 
        extract exactly the information needed from complex financial data structures.""",
        "tools": ["neo4j_schema_tool", "graph_query_tool"],
        "memory": False,
        "max_iter": 3,
    },
    "graph_analyst": {
        "id": "graph_analyst",
        "role": "Graph Data Scientist",
        "goal": "Execute graph queries and analyze results using graph algorithms",
        "backstory": """You are an expert graph data scientist with extensive experience in 
        financial network analysis. You specialize in executing complex queries against graph 
        databases, running community detection, centrality, and path-finding algorithms, and 
        interpreting the results in the context of financial transactions and relationships.""",
        "tools": ["graph_query_tool"],
        "memory": True,
        "max_iter": 5,
    },
    "fraud_pattern_hunter": {
        "id": "fraud_pattern_hunter",
        "role": "Fraud Pattern Detection Specialist",
        "goal": "Identify known and unknown fraud patterns in financial data",
        "backstory": """You are a seasoned financial crime investigator with a talent for 
        spotting suspicious patterns. You've spent years studying money laundering techniques, 
        fraud schemes, and financial crime typologies. You use a combination of known pattern 
        templates and anomaly detection to identify potential fraud in complex financial data.""",
        "tools": ["graph_query_tool", "pattern_library_tool"],
        "memory": True,
        "max_iter": 5,
    },
    "sandbox_coder": {
        "id": "sandbox_coder",
        "role": "Secure Code Generation and Execution Specialist",
        "goal": "Generate and run Python code for data analysis in secure sandboxes",
        "backstory": """You are an expert Python developer specializing in data analysis and 
        machine learning. You can quickly generate efficient, secure code to analyze complex 
        datasets, visualize results, and apply machine learning techniques. You ensure all code 
        runs safely in isolated environments and produces reliable, reproducible results.""",
        "tools": ["code_gen_tool", "sandbox_exec_tool"],
        "memory": False,
        "max_iter": 5,
    },
    "compliance_checker": {
        "id": "compliance_checker",
        "role": "AML Compliance Officer",
        "goal": "Ensure outputs align with AML regulations and format SAR sections",
        "backstory": """You are a compliance officer with deep knowledge of Anti-Money Laundering 
        (AML) regulations and Suspicious Activity Report (SAR) filing requirements. You review 
        analysis results to ensure they meet regulatory standards, identify reportable activities, 
        and format findings appropriately for regulatory submissions.""",
        "tools": ["policy_docs_tool"],
        "memory": True,
        "max_iter": 3,
    },
    "report_writer": {
        "id": "report_writer",
        "role": "Financial Intelligence Report Writer",
        "goal": "Produce clear, actionable intelligence reports from analysis results",
        "backstory": """You are a skilled intelligence analyst and report writer. You excel at 
        synthesizing complex financial data and investigative findings into clear, concise, and 
        actionable reports. You know how to present technical information to both technical and 
        non-technical audiences, highlighting key insights and supporting evidence.""",
        "tools": ["template_engine_tool"],
        "memory": True,
        "max_iter": 3,
    },
    "red_team_adversary": {
        "id": "red_team_adversary",
        "role": "Financial Crime Simulator",
        "goal": "Simulate sophisticated financial crime scenarios to test detection systems",
        "backstory": """You are a red team specialist who understands how financial criminals 
        operate. Your job is to simulate realistic fraud scenarios, money laundering schemes, and 
        other financial crimes to test and improve detection systems. You think like an adversary 
        but work to strengthen defenses.""",
        "tools": ["sandbox_exec_tool", "random_tx_generator_tool"],
        "memory": True,
        "max_iter": 5,
    },
}


# Default crew configurations
DEFAULT_CREW_CONFIGS = {
    "fraud_investigation": {
        "crew_name": "fraud_investigation",
        "process_type": "sequential",
        "manager": "orchestrator_manager",
        "agents": [
            "nlq_translator",
            "graph_analyst",
            "fraud_pattern_hunter",
            "sandbox_coder",
            "compliance_checker",
            "report_writer"
        ],
        "verbose": True,
        "memory": True,
        "cache": True,
    },
    "alert_enrichment": {
        "crew_name": "alert_enrichment",
        "process_type": "sequential",
        "manager": "orchestrator_manager",
        "agents": [
            "nlq_translator",
            "graph_analyst",
            "fraud_pattern_hunter",
            "compliance_checker",
            "report_writer"
        ],
        "verbose": True,
        "memory": False,  # Faster response for real-time alerts
        "cache": True,
    },
    "red_blue_simulation": {
        "crew_name": "red_blue_simulation",
        "process_type": "hierarchical",  # Red team needs more autonomy
        "manager": "orchestrator_manager",
        "agents": [
            "red_team_adversary",
            "graph_analyst",
            "fraud_pattern_hunter",
            "report_writer"
        ],
        "verbose": True,
        "memory": True,
        "cache": False,  # Dynamic scenarios shouldn't be cached
    }
}


def load_agent_config(agent_id: str) -> AgentConfig:
    """
    Load agent configuration from YAML file or use default.
    
    Args:
        agent_id: The ID of the agent to load
        
    Returns:
        AgentConfig object with the agent's configuration
    """
    # Check if custom config file exists
    config_path = AGENT_CONFIG_DIR / f"{agent_id}.yaml"
    
    if config_path.exists():
        with open(config_path, "r") as f:
            config_data = yaml.safe_load(f)
        return AgentConfig(**config_data)
    
    # Fall back to default config
    if agent_id in DEFAULT_AGENT_CONFIGS:
        return AgentConfig(**DEFAULT_AGENT_CONFIGS[agent_id])
    
    raise ValueError(f"No configuration found for agent: {agent_id}")


def load_crew_config(crew_name: str) -> CrewConfig:
    """
    Load crew configuration from YAML file or use default.
    
    Args:
        crew_name: The name of the crew to load
        
    Returns:
        CrewConfig object with the crew's configuration
    """
    # Check if custom config file exists
    config_path = AGENT_CONFIG_DIR / "crews" / f"{crew_name}.yaml"
    
    if config_path.exists():
        with open(config_path, "r") as f:
            config_data = yaml.safe_load(f)
        return CrewConfig(**config_data)
    
    # Fall back to default config
    if crew_name in DEFAULT_CREW_CONFIGS:
        return CrewConfig(**DEFAULT_CREW_CONFIGS[crew_name])
    
    raise ValueError(f"No configuration found for crew: {crew_name}")


def save_agent_config(config: AgentConfig) -> None:
    """
    Save agent configuration to YAML file.
    
    Args:
        config: The AgentConfig object to save
    """
    # Ensure directory exists
    os.makedirs(AGENT_CONFIG_DIR, exist_ok=True)
    
    config_path = AGENT_CONFIG_DIR / f"{config.id}.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config.dict(), f)


def save_crew_config(config: CrewConfig) -> None:
    """
    Save crew configuration to YAML file.
    
    Args:
        config: The CrewConfig object to save
    """
    # Ensure directory exists
    crew_dir = AGENT_CONFIG_DIR / "crews"
    os.makedirs(crew_dir, exist_ok=True)
    
    config_path = crew_dir / f"{config.crew_name}.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config.dict(), f)
