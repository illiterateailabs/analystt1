"""
Crew Factory Module

This module provides the CrewFactory class, responsible for dynamically
creating and managing CrewAI agent crews based on configuration files.
It handles the loading of agents, tasks, and tools, and orchestrates
the execution of complex multi-agent workflows.
"""

import logging
import os
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional
import asyncio
from functools import lru_cache

from crewai import Agent, Task, Crew, Process
from crewai.utilities import IORunner
from crewai.tools import BaseTool

from backend.core.logging import get_logger
from backend.integrations.gemini_client import GeminiClient
from backend.agents.tools import get_all_tools
from backend.agents.config import get_available_agents as get_configured_agents # Avoid name collision


logger = get_logger(__name__)

# Constants
CREWS_CONFIG_DIR = Path("backend/agents/configs/crews")
DEFAULT_AGENT_CONFIG_DIR = Path("backend/agents/configs/defaults")
DEFAULT_CREW_CONFIG_FILE = CREWS_CONFIG_DIR / "fraud_investigation.yaml"

# In-memory storage for running crews (for HITL)
RUNNING_CREWS: Dict[str, Dict[str, Any]] = {}


class CrewFactory:
    """
    Dynamically creates and manages CrewAI agent crews.

    This factory reads YAML configuration files to define agents, tasks,
    and crews, and provides methods to run and manage them.
    """

    def __init__(self):
        self.llm_client = GeminiClient()
        self._available_crews: Dict[str, Dict[str, Any]] = {}
        self._available_agents: Dict[str, Dict[str, Any]] = {}
        self._available_tools: List[BaseTool] = []
        self._load_all_resources()
        logger.info("CrewFactory initialized.")

    async def reload(self):
        """
        Reloads all crew, agent, and tool configurations.
        This should be called after any changes to config files.
        """
        logger.info("Reloading CrewFactory configurations...")
        self._load_all_resources()
        logger.info("CrewFactory configurations reloaded.")

    @lru_cache(maxsize=1)
    def _load_all_resources(self):
        """
        Internal method to load all available crews, agents, and tools.
        Uses lru_cache to avoid redundant loading within a short period.
        """
        self._available_crews = self._load_crew_configs()
        self._available_agents = self._load_agent_configs()
        self._available_tools = self._load_tools()
        logger.info(f"Loaded {len(self._available_crews)} crews, {len(self._available_agents)} agents, {len(self._available_tools)} tools.")

    def _load_crew_configs(self) -> Dict[str, Dict[str, Any]]:
        """Loads all crew configurations from YAML files."""
        crews = {}
        if not CREWS_CONFIG_DIR.exists():
            logger.warning(f"Crew configurations directory not found: {CREWS_CONFIG_DIR}")
            return crews

        for config_file in CREWS_CONFIG_DIR.glob("*.yaml"):
            try:
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
                    crew_name = config.get("name", config_file.stem)
                    crews[crew_name] = config
                    logger.debug(f"Loaded crew config: {crew_name}")
            except Exception as e:
                logger.error(f"Error loading crew config {config_file}: {e}")
        return crews

    def _load_agent_configs(self) -> Dict[str, Dict[str, Any]]:
        """Loads all default agent configurations."""
        return get_configured_agents()

    def _load_tools(self) -> List[BaseTool]:
        """Loads all available tools."""
        return get_all_tools()

    def get_available_crews(self) -> List[str]:
        """Returns a list of names of all available crews."""
        return list(self._available_crews.keys())

    def get_available_agents(self) -> Dict[str, Dict[str, Any]]:
        """Returns a dictionary of all available agent configurations."""
        return self._available_agents

    def get_available_tools(self) -> List[BaseTool]:
        """Returns a list of all available tool instances."""
        return self._available_tools

    def _create_agent(self, agent_config: Dict[str, Any]) -> Agent:
        """Creates an Agent instance from a configuration dictionary."""
        agent_tools = []
        if "tools" in agent_config and agent_config["tools"]:
            for tool_name in agent_config["tools"]:
                found_tool = next((tool for tool in self._available_tools if tool.name == tool_name), None)
                if found_tool:
                    agent_tools.append(found_tool)
                else:
                    logger.warning(f"Tool '{tool_name}' not found for agent '{agent_config.get('name', 'unknown')}'")

        # Create the agent with the specified configuration
        agent = Agent(
            role=agent_config.get("role", ""),
            goal=agent_config.get("goal", ""),
            backstory=agent_config.get("backstory", ""),
            verbose=agent_config.get("verbose", True),
            allow_delegation=agent_config.get("allow_delegation", False),
            tools=agent_tools,
            llm=self.llm_client.get_llm(agent_config.get("llm", {}))
        )
        
        return agent

    def _create_task(self, task_config: Dict[str, Any], agents: Dict[str, Agent]) -> Task:
        """Creates a Task instance from a configuration dictionary."""
        agent_name = task_config.get("agent")
        if not agent_name or agent_name not in agents:
            raise ValueError(f"Task requires a valid agent name. Got: {agent_name}")
        
        # Get task-specific tools if specified
        task_tools = None
        if "tools" in task_config and task_config["tools"]:
            task_tools = []
            for tool_name in task_config["tools"]:
                found_tool = next((tool for tool in self._available_tools if tool.name == tool_name), None)
                if found_tool:
                    task_tools.append(found_tool)
                else:
                    logger.warning(f"Tool '{tool_name}' not found for task '{task_config.get('description', 'unknown')}'")
        
        # Create the task with the specified configuration
        task = Task(
            description=task_config.get("description", ""),
            expected_output=task_config.get("expected_output", ""),
            agent=agents[agent_name],
            tools=task_tools,  # Will use agent's tools if None
            async_execution=task_config.get("async_execution", False)
        )
        
        return task

    def _create_crew(self, crew_config: Dict[str, Any]) -> Crew:
        """Creates a Crew instance from a configuration dictionary."""
        # Create agents
        agents = {}
        for agent_name, agent_config in crew_config.get("agents", {}).items():
            # Ensure the agent has a name
            agent_config["name"] = agent_name
            # Create the agent
            agents[agent_name] = self._create_agent(agent_config)
        
        # Create tasks
        tasks = []
        for task_config in crew_config.get("tasks", []):
            tasks.append(self._create_task(task_config, agents))
        
        # Create the crew
        crew = Crew(
            agents=list(agents.values()),
            tasks=tasks,
            verbose=crew_config.get("verbose", True),
            process=Process.SEQUENTIAL,  # Default to sequential
            memory=crew_config.get("memory", None),
            max_rpm=crew_config.get("max_rpm", None)
        )
        
        return crew

    async def run_crew(self, crew_name: str, inputs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Runs a crew with the specified inputs.
        
        Args:
            crew_name: Name of the crew to run
            inputs: Input parameters for the crew
            
        Returns:
            Crew execution result
        """
        if crew_name not in self._available_crews:
            return {
                "success": False,
                "error": f"Crew not found: {crew_name}"
            }
        
        try:
            # Get crew configuration
            crew_config = self._available_crews[crew_name]
            
            # Create the crew
            crew = self._create_crew(crew_config)
            
            # Generate a unique task ID
            task_id = f"task_{len(RUNNING_CREWS) + 1}"
            
            # Store crew in running crews
            RUNNING_CREWS[task_id] = {
                "crew": crew,
                "config": crew_config,
                "inputs": inputs or {},
                "status": "RUNNING",
                "result": None
            }
            
            # Run the crew
            logger.info(f"Running crew '{crew_name}' with task ID '{task_id}'")
            result = crew.kickoff(inputs=inputs or {})
            
            # Update crew status
            RUNNING_CREWS[task_id]["status"] = "COMPLETED"
            RUNNING_CREWS[task_id]["result"] = result
            
            # Return the result with task ID
            return {
                "success": True,
                "task_id": task_id,
                "status": "COMPLETED",
                "result": result
            }
        except Exception as e:
            logger.error(f"Error running crew '{crew_name}': {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def pause_crew(self, task_id: str, reason: Optional[str] = None) -> Dict[str, Any]:
        """
        Pauses a running crew task.
        
        Args:
            task_id: Task ID of the running crew
            reason: Reason for pausing
            
        Returns:
            Pause operation result
        """
        if task_id not in RUNNING_CREWS:
            return {
                "success": False,
                "error": f"Task not found: {task_id}"
            }
        
        try:
            # Get crew from running crews
            crew_data = RUNNING_CREWS[task_id]
            
            # Check if crew is already paused or completed
            if crew_data["status"] == "PAUSED":
                return {
                    "success": False,
                    "error": f"Task already paused: {task_id}"
                }
            
            if crew_data["status"] == "COMPLETED":
                return {
                    "success": False,
                    "error": f"Task already completed: {task_id}"
                }
            
            # Pause the crew (in a real implementation, this would interact with the crew execution)
            # For now, we just update the status
            crew_data["status"] = "PAUSED"
            crew_data["pause_reason"] = reason
            
            logger.info(f"Paused crew task '{task_id}' with reason: {reason}")
            
            return {
                "success": True,
                "task_id": task_id,
                "status": "PAUSED"
            }
        except Exception as e:
            logger.error(f"Error pausing crew task '{task_id}': {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def resume_crew(self, task_id: str, approved: bool, comment: Optional[str] = None) -> Dict[str, Any]:
        """
        Resumes a paused crew task.
        
        Args:
            task_id: Task ID of the paused crew
            approved: Whether the task is approved to continue
            comment: Comment from the reviewer
            
        Returns:
            Resume operation result
        """
        if task_id not in RUNNING_CREWS:
            return {
                "success": False,
                "error": f"Task not found: {task_id}"
            }
        
        try:
            # Get crew from running crews
            crew_data = RUNNING_CREWS[task_id]
            
            # Check if crew is paused
            if crew_data["status"] != "PAUSED":
                return {
                    "success": False,
                    "error": f"Task is not paused: {task_id}"
                }
            
            if approved:
                # Resume the crew (in a real implementation, this would interact with the crew execution)
                # For now, we just update the status
                crew_data["status"] = "RESUMED"
                crew_data["review_comment"] = comment
                
                logger.info(f"Resumed crew task '{task_id}' with approval")
                
                # In a real implementation, we would continue the crew execution
                # For now, we'll just mark it as completed
                crew_data["status"] = "COMPLETED"
                
                return {
                    "success": True,
                    "task_id": task_id,
                    "status": "RESUMED"
                }
            else:
                # Mark the crew as rejected
                crew_data["status"] = "REJECTED"
                crew_data["review_comment"] = comment
                
                logger.info(f"Rejected crew task '{task_id}' with comment: {comment}")
                
                return {
                    "success": True,
                    "task_id": task_id,
                    "status": "REJECTED"
                }
        except Exception as e:
            logger.error(f"Error resuming crew task '{task_id}': {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def close(self):
        """
        Closes the factory and releases resources.
        """
        # In a real implementation, this would clean up resources
        # For now, we'll just log the action
        logger.info("Closing CrewFactory")
