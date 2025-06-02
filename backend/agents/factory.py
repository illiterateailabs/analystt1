"""
CrewFactory for building and running CrewAI crews.

This module provides a factory class for creating and managing CrewAI crews,
including agent creation, tool assignment, task definition, and crew orchestration.
"""

import logging
import uuid
import asyncio
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

from crewai import Agent, Task, Crew, Process
from crewai.agent import TaskOutput

from backend.agents.config import load_agent_config, load_crew_config, get_available_crews
from backend.agents.llm import GeminiLLMProvider
from backend.integrations.neo4j_client import Neo4jClient
from backend.integrations.gemini_client import GeminiClient
from backend.integrations.e2b_client import E2BClient
from backend.agents.tools import get_all_tools

# Configure logging
logger = logging.getLogger(__name__)

# In-memory storage for running crews (task_id → state)
# This will be replaced with a database implementation in the future
RUNNING_CREWS = {}


class CrewFactory:
    """
    Factory for creating and running CrewAI crews.
    
    This class provides methods for creating agents, tasks, and crews,
    as well as running crews with specific inputs and managing their
    execution state for human-in-the-loop workflows.
    """
    
    def __init__(self):
        """Initialize the CrewFactory with required clients and caches."""
        # Initialize clients
        self.neo4j_client = Neo4jClient()
        self.gemini_client = GeminiClient()
        self.e2b_client = E2BClient()
        self.llm_provider = GeminiLLMProvider()
        
        # Initialize tool registry
        self.tools = {tool.name: tool for tool in get_all_tools()}
        logger.info(f"Initialized {len(self.tools)} tools: {', '.join(self.tools.keys())}")
        
        # Initialize caches
        self.agents_cache = {}
        self.crews_cache = {}
    
    async def connect(self):
        """Connect to external services."""
        try:
            # Connect to Neo4j
            await self.neo4j_client.connect()
            logger.info("Connected to Neo4j")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise
    
    async def close(self):
        """Close connections to external services."""
        try:
            # Close Neo4j connection if it exists
            if self.neo4j_client.driver:
                await self.neo4j_client.close()
                logger.info("Closed Neo4j connection")
            
            # Close all e2b sandboxes
            await self.e2b_client.close_all_sandboxes()
            logger.info("Closed all e2b sandboxes")
        except Exception as e:
            logger.error(f"Error closing connections: {e}")
    
    def get_tool(self, tool_name: str) -> Any:
        """
        Get a tool by name.
        
        Args:
            tool_name: Name of the tool to get
            
        Returns:
            The tool instance, or None if not found
        """
        return self.tools.get(tool_name)
    
    def create_agent(self, agent_id: str) -> Agent:
        """
        Create a CrewAI agent with the specified configuration.
        
        Args:
            agent_id: ID of the agent to create
            
        Returns:
            CrewAI Agent instance
        """
        # Check if agent is already cached
        if agent_id in self.agents_cache:
            return self.agents_cache[agent_id]
        
        # Load agent configuration
        try:
            config = load_agent_config(agent_id)
        except ValueError as e:
            logger.error(f"Failed to load agent config: {e}")
            raise
        
        # Get tools for the agent
        agent_tools = []
        for tool_config in config.tools:
            if isinstance(tool_config, str):
                # Simple tool name
                tool = self.get_tool(tool_config)
                if tool:
                    agent_tools.append(tool)
                else:
                    logger.warning(f"Tool not found: {tool_config}")
            else:
                # Tool with configuration
                tool_name = tool_config.type
                tool = self.get_tool(tool_name)
                if tool:
                    # TODO: Configure tool with tool_config
                    agent_tools.append(tool)
                else:
                    logger.warning(f"Tool not found: {tool_name}")
        
        # Create the agent
        agent = Agent(
            id=config.id,
            role=config.role,
            goal=config.goal,
            backstory=config.backstory,
            verbose=config.verbose,
            allow_delegation=config.allow_delegation,
            tools=agent_tools,
            llm=self.llm_provider,
            max_iter=config.max_iter,
            max_rpm=config.max_rpm,
            memory=config.memory,
        )
        
        # Cache the agent
        self.agents_cache[agent_id] = agent
        
        return agent
    
    def create_tasks(self, crew_name: str, agents: Dict[str, Agent]) -> List[Task]:
        """
        Create tasks for a specific crew.
        
        Args:
            crew_name: Name of the crew
            agents: Dictionary of agent instances by ID
            
        Returns:
            List of CrewAI Task instances
        """
        try:
            # Load crew configuration
            crew_config = load_crew_config(crew_name)
            
            # Create tasks based on configuration
            tasks = []
            
            # If tasks are defined in the crew config, use them
            if crew_config.tasks:
                for i, task_config in enumerate(crew_config.tasks):
                    # Get the agent for this task
                    agent = agents.get(task_config.agent)
                    if not agent:
                        logger.warning(f"Agent not found for task: {task_config.agent}")
                        continue
                    
                    # Create the task
                    task = Task(
                        description=task_config.description,
                        expected_output=task_config.expected_output,
                        agent=agent,
                        async_execution=task_config.async_execution,
                        context=task_config.context,
                    )
                    tasks.append(task)
            
            return tasks
            
        except Exception as e:
            logger.error(f"Failed to create tasks for crew {crew_name}: {e}")
            return []
    
    async def create_crew(self, crew_name: str) -> Crew:
        """
        Create a CrewAI crew with the specified configuration.
        
        Args:
            crew_name: Name of the crew to create
            
        Returns:
            CrewAI Crew instance
        """
        # Check if crew is already cached
        if crew_name in self.crews_cache:
            return self.crews_cache[crew_name]
        
        try:
            # Load crew configuration
            crew_config = load_crew_config(crew_name)
            
            # Create agents
            agents = {}
            for agent_config in crew_config.agents:
                if isinstance(agent_config, str):
                    # Simple agent ID
                    agent_id = agent_config
                    agent = self.create_agent(agent_id)
                    agents[agent_id] = agent
                else:
                    # Agent with configuration
                    agent_id = agent_config["id"]
                    agent = self.create_agent(agent_id)
                    agents[agent_id] = agent
            
            # Create tasks
            tasks = self.create_tasks(crew_name, agents)
            
            # Determine process type
            process = Process.sequential
            if crew_config.process_type == "hierarchical":
                process = Process.hierarchical
            
            # Create the crew
            crew = Crew(
                agents=list(agents.values()),
                tasks=tasks,
                process=process,
                verbose=crew_config.verbose,
                max_rpm=crew_config.max_rpm,
                memory=crew_config.memory,
                cache=crew_config.cache,
                manager_llm=self.llm_provider,
            )
            
            # Cache the crew
            self.crews_cache[crew_name] = crew
            
            return crew
            
        except Exception as e:
            logger.error(f"Failed to create crew {crew_name}: {e}")
            raise
    
    async def run_crew(self, crew_name: str, inputs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run a crew with the specified inputs.
        
        Args:
            crew_name: Name of the crew to run
            inputs: Input parameters for the crew
            
        Returns:
            Result of the crew execution
        """
        try:
            # Create the crew
            crew = await self.create_crew(crew_name)
            
            # Generate a task ID
            task_id = str(uuid.uuid4())
            
            # Initialize crew state
            RUNNING_CREWS[task_id] = {
                "crew_name": crew_name,
                "state": "RUNNING",
                "start_time": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
                "current_agent": None,
                "current_task": None,
                "paused_at": None,
                "context": inputs or {},
                "crew": crew
            }
            
            # Run the crew with inputs
            logger.info(f"Running crew {crew_name} with task ID {task_id}")
            
            # Initialize context for result propagation
            context = inputs or {}
            
            # Override the kickoff method to track context between tasks
            original_kickoff = crew.kickoff
            
            async def kickoff_with_context(inputs=None):
                nonlocal context
                
                # Merge inputs with context
                if inputs:
                    context.update(inputs)
                
                # Run the crew with the context
                result = original_kickoff(inputs=context)
                
                # Update task state
                RUNNING_CREWS[task_id]["state"] = "COMPLETED"
                RUNNING_CREWS[task_id]["last_updated"] = datetime.now().isoformat()
                
                # Add task_id to result if it's a dict
                if isinstance(result, dict):
                    result["task_id"] = task_id
                
                return result
            
            # Replace the kickoff method
            crew.kickoff = kickoff_with_context
            
            # Run the crew
            result = crew.kickoff(inputs=context)
            
            # Close connections
            await self.close()
            
            # Return the result
            if isinstance(result, TaskOutput):
                return {
                    "success": True,
                    "result": result.raw_output,
                    "task_id": task_id,
                    "agent_id": result.agent_id,
                    "task_id": result.task_id
                }
            else:
                return {
                    "success": True,
                    "result": result,
                    "task_id": task_id
                }
            
        except Exception as e:
            logger.error(f"Failed to run crew {crew_name}: {e}")
            
            # Close connections
            await self.close()
            
            return {
                "success": False,
                "error": str(e)
            }
    
    async def pause_crew(self, task_id: str, reason: Optional[str] = None) -> Dict[str, Any]:
        """
        Pause a running crew task.
        
        Args:
            task_id: ID of the task to pause
            reason: Reason for pausing
            
        Returns:
            Result of the pause operation
        """
        try:
            # Check if task exists
            if task_id not in RUNNING_CREWS:
                return {
                    "success": False,
                    "error": f"Task not found: {task_id}"
                }
            
            # Get crew state
            crew_state = RUNNING_CREWS[task_id]
            
            # Check if task is already paused
            if crew_state["state"] == "PAUSED":
                return {
                    "success": False,
                    "error": f"Task is already paused: {task_id}"
                }
            
            # Pause the task
            crew_state["state"] = "PAUSED"
            crew_state["paused_at"] = datetime.now().isoformat()
            crew_state["pause_reason"] = reason
            crew_state["last_updated"] = datetime.now().isoformat()
            
            logger.info(f"Paused crew task {task_id}")
            
            return {
                "success": True,
                "task_id": task_id,
                "state": "PAUSED",
                "reason": reason
            }
            
        except Exception as e:
            logger.error(f"Failed to pause crew task {task_id}: {e}")
            
            return {
                "success": False,
                "error": str(e)
            }
    
    async def resume_crew(self, task_id: str, approved: bool, comment: Optional[str] = None) -> Dict[str, Any]:
        """
        Resume a paused crew task.
        
        Args:
            task_id: ID of the task to resume
            approved: Whether the task is approved to continue
            comment: Comment from the reviewer
            
        Returns:
            Result of the resume operation
        """
        try:
            # Check if task exists
            if task_id not in RUNNING_CREWS:
                return {
                    "success": False,
                    "error": f"Task not found: {task_id}"
                }
            
            # Get crew state
            crew_state = RUNNING_CREWS[task_id]
            
            # Check if task is paused
            if crew_state["state"] != "PAUSED":
                return {
                    "success": False,
                    "error": f"Task is not paused: {task_id}"
                }
            
            # Update the task state
            crew_state["state"] = "RUNNING" if approved else "REJECTED"
            crew_state["last_updated"] = datetime.now().isoformat()
            crew_state["review_comment"] = comment
            crew_state["approved"] = approved
            
            logger.info(f"Resumed crew task {task_id} (approved: {approved})")
            
            # TODO: Implement actual resumption of the crew execution
            # This would require modifying the CrewAI library to support
            # pausing and resuming execution mid-flow
            
            return {
                "success": True,
                "task_id": task_id,
                "state": crew_state["state"],
                "approved": approved
            }
            
        except Exception as e:
            logger.error(f"Failed to resume crew task {task_id}: {e}")
            
            return {
                "success": False,
                "error": str(e)
            }
    
    def reload(self):
        """
        Reload all configurations and clear caches.
        
        This method is used to hot-reload configurations when they change,
        such as when templates are created or modified.
        """
        logger.info("Reloading CrewFactory configurations")
        
        # Clear caches
        self.agents_cache = {}
        self.crews_cache = {}
        
        # Re-initialize tools
        self.tools = {tool.name: tool for tool in get_all_tools()}
        logger.info(f"Reloaded {len(self.tools)} tools: {', '.join(self.tools.keys())}")
        
        return {
            "success": True,
            "message": "CrewFactory configurations reloaded",
            "tools_count": len(self.tools),
            "tools": list(self.tools.keys())
        }
    
    @staticmethod
    def get_available_crews() -> List[str]:
        """
        Get a list of available crew names.
        
        Returns:
            List of crew names
        """
        return get_available_crews()
    
    @classmethod
    async def load(cls, crew_name: str) -> Crew:
        """
        Load a crew by name.
        
        This is a convenience method for creating a factory,
        connecting to services, and creating a crew.
        
        Args:
            crew_name: Name of the crew to load
            
        Returns:
            CrewAI Crew instance
        """
        # Create factory
        factory = cls()
        
        # Connect to services
        await factory.connect()
        
        # Create crew
        crew = await factory.create_crew(crew_name)
        
        return crew
