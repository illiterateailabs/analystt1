"""
CrewAI Factory for creating and managing agent crews.

This module provides a factory for creating and managing CrewAI agents and crews.
It handles the initialization of tools, creation of agents with specific roles,
and orchestration of crews for different tasks.
"""

import asyncio
import logging
import os
from typing import Any, Dict, List, Optional, Set, Type, Union, cast

from crewai import Agent, Crew, Task
from crewai.agent import AgentConfig
import yaml

from backend.agents.config import AgentConfig as AppAgentConfig
from backend.agents.llm import GeminiLLMProvider
from backend.agents.custom_crew import CustomCrew
from backend.core.metrics import increment_counter, observe_value
from backend.integrations.neo4j_client import Neo4jClient
from backend.integrations.e2b_client import E2BClient
from backend.agents.tools.graph_query_tool import GraphQueryTool
from backend.agents.tools.neo4j_schema_tool import Neo4jSchemaTool
from backend.agents.tools.pattern_library_tool import PatternLibraryTool
from backend.agents.tools.code_gen_tool import CodeGenTool
from backend.agents.tools.sandbox_exec_tool import SandboxExecTool
from backend.agents.tools.template_engine_tool import TemplateEngineTool
from backend.agents.tools.policy_docs_tool import PolicyDocsTool
from backend.agents.tools.random_tx_generator_tool import RandomTxGeneratorTool
from backend.agents.tools.fraud_ml_tool import FraudMLTool
from backend.agents.tools.graphql_query_tool import GraphQLQueryTool

# Optional crypto tools
try:
    from backend.agents.tools.crypto_anomaly_tool import CryptoAnomalyTool
    from backend.agents.tools.crypto_csv_loader_tool import CryptoCSVLoaderTool
    CRYPTO_TOOLS_AVAILABLE = True
except ImportError:
    CRYPTO_TOOLS_AVAILABLE = False


logger = logging.getLogger(__name__)


class CrewFactory:
    """
    Factory for creating and managing CrewAI agents and crews.
    
    This class handles the initialization of tools, creation of agents with
    specific roles, and orchestration of crews for different tasks.
    
    Attributes:
        config_dir: Directory containing agent and crew configurations
        llm_provider: LLM provider for agents
        tools: Dictionary of available tools
        agent_cache: Cache of created agents
        crew_cache: Cache of created crews
        task_state: Dictionary of task states for HITL workflows
    """
    
    DEFAULT_CONFIG_DIR = os.path.join(
        os.path.dirname(__file__), "configs"
    )
    
    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize the CrewFactory.
        
        Args:
            config_dir: Directory containing agent and crew configurations
        """
        self.config_dir = config_dir or self.DEFAULT_CONFIG_DIR
        
        # Initialize LLM provider
        self.llm_provider = GeminiLLMProvider()
        
        # Initialize tools
        self._tools: Dict[str, Any] = {}
        self._init_tools()
        
        # Cache for created agents and crews
        self._agent_cache: Dict[str, Agent] = {}
        self._crew_cache: Dict[str, Crew] = {}
        
        # Task state for HITL workflows
        self._task_state: Dict[str, Dict[str, Any]] = {}
    
    def _init_tools(self):
        """Initialize all available tools."""
        try:
            # Initialize Neo4j client
            neo4j_client = Neo4jClient()
            
            # Initialize E2B client
            e2b_client = E2BClient()
            
            # Initialize core tools
            self._tools = {
                "graph_query": GraphQueryTool(neo4j_client),
                "neo4j_schema": Neo4jSchemaTool(neo4j_client),
                "pattern_library": PatternLibraryTool(neo4j_client),
                "code_gen": CodeGenTool(e2b_client=e2b_client),
                "sandbox_exec": SandboxExecTool(e2b_client),
                "template_engine": TemplateEngineTool(),
                "policy_docs": PolicyDocsTool(),
                "random_tx_generator": RandomTxGeneratorTool(neo4j_client),
                "fraud_ml": FraudMLTool(),
                "graphql_query": GraphQLQueryTool(),
            }
            
            # Initialize crypto tools if available
            if CRYPTO_TOOLS_AVAILABLE:
                self._tools.update({
                    "crypto_anomaly": CryptoAnomalyTool(),
                    "crypto_csv_loader": CryptoCSVLoaderTool(neo4j_client),
                })
            
            logger.info(f"Initialized {len(self._tools)} tools")
            
        except Exception as e:
            logger.error(f"Error initializing tools: {e}")
            raise
    
    def get_available_crews(self) -> List[str]:
        """
        Get a list of available crew configurations.
        
        Returns:
            List of crew names that can be created
        """
        crews_dir = os.path.join(self.config_dir, "crews")
        if not os.path.exists(crews_dir):
            logger.warning(f"Crews directory not found: {crews_dir}")
            return []
        
        crew_files = [
            f for f in os.listdir(crews_dir)
            if f.endswith(".yaml") and not f.startswith("_")
        ]
        
        return [os.path.splitext(f)[0] for f in crew_files]
    
    def get_available_agents(self) -> List[str]:
        """
        Get a list of available agent configurations.
        
        Returns:
            List of agent names that can be created
        """
        defaults_dir = os.path.join(self.config_dir, "defaults")
        if not os.path.exists(defaults_dir):
            logger.warning(f"Defaults directory not found: {defaults_dir}")
            return []
        
        agent_files = [
            f for f in os.listdir(defaults_dir)
            if f.endswith(".yaml") and not f.startswith("_")
        ]
        
        return [os.path.splitext(f)[0] for f in agent_files]
    
    def get_agent_config(self, agent_name: str) -> Dict[str, Any]:
        """
        Get the configuration for an agent.
        
        Args:
            agent_name: Name of the agent
            
        Returns:
            Agent configuration dictionary
            
        Raises:
            ValueError: If agent configuration is not found
        """
        config_path = os.path.join(self.config_dir, "defaults", f"{agent_name}.yaml")
        
        if not os.path.exists(config_path):
            raise ValueError(f"Agent configuration not found: {agent_name}")
        
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        
        return config
    
    def get_crew_config(self, crew_name: str) -> Dict[str, Any]:
        """
        Get the configuration for a crew.
        
        Args:
            crew_name: Name of the crew
            
        Returns:
            Crew configuration dictionary
            
        Raises:
            ValueError: If crew configuration is not found
        """
        config_path = os.path.join(self.config_dir, "crews", f"{crew_name}.yaml")
        
        if not os.path.exists(config_path):
            raise ValueError(f"Crew configuration not found: {crew_name}")
        
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        
        return config
    
    def create_agent(self, agent_name: str, **kwargs) -> Agent:
        """
        Create an agent with the specified configuration.
        
        Args:
            agent_name: Name of the agent
            **kwargs: Additional parameters to override configuration
            
        Returns:
            CrewAI Agent instance
            
        Raises:
            ValueError: If agent configuration is not found
        """
        # Check if agent is already cached
        cache_key = f"{agent_name}:{hash(frozenset(kwargs.items()))}"
        if cache_key in self._agent_cache:
            logger.debug(f"Using cached agent: {agent_name}")
            return self._agent_cache[cache_key]
        
        # Get agent configuration
        config = self.get_agent_config(agent_name)
        
        # Override with kwargs
        for key, value in kwargs.items():
            if key in config:
                if isinstance(config[key], dict) and isinstance(value, dict):
                    config[key].update(value)
                else:
                    config[key] = value
        
        # Create agent config
        agent_config = AgentConfig(
            allow_delegation=config.get("allow_delegation", False),
            max_iter=config.get("max_iterations", 15),
            verbose=config.get("verbose", False),
            allow_model_params_override=config.get("allow_model_params_override", True),
        )
        
        # Get tools for agent
        tools = self._get_tools_for_agent(config.get("tools", []))
        
        # Create agent
        agent = Agent(
            role=config.get("role", agent_name),
            goal=config.get("goal", ""),
            backstory=config.get("backstory", ""),
            llm=self.llm_provider.get_llm(
                model=config.get("model", "gemini-1.5-pro"),
                temperature=config.get("temperature", 0.7),
                max_tokens=config.get("max_tokens", 1024),
            ),
            tools=tools,
            agent_config=agent_config,
            verbose=config.get("verbose", False),
        )
        
        # Cache agent
        self._agent_cache[cache_key] = agent
        
        logger.info(f"Created agent: {agent_name}")
        return agent
    
    def create_crew(self, crew_name: str, **kwargs) -> Crew:
        """
        Create a crew with the specified configuration.
        
        Args:
            crew_name: Name of the crew
            **kwargs: Additional parameters to override configuration
            
        Returns:
            CrewAI Crew instance
            
        Raises:
            ValueError: If crew configuration is not found
        """
        # Check if crew is already cached
        cache_key = f"{crew_name}:{hash(frozenset(kwargs.items()))}"
        if cache_key in self._crew_cache:
            logger.debug(f"Using cached crew: {crew_name}")
            return self._crew_cache[cache_key]
        
        # Get crew configuration
        config = self.get_crew_config(crew_name)
        
        # Override with kwargs
        for key, value in kwargs.items():
            if key in config:
                if isinstance(config[key], dict) and isinstance(value, dict):
                    config[key].update(value)
                else:
                    config[key] = value
        
        # Create agents
        agents = []
        for agent_config in config.get("agents", []):
            agent_name = agent_config.get("name")
            agent_kwargs = agent_config.get("config", {})
            agent = self.create_agent(agent_name, **agent_kwargs)
            agents.append(agent)
        
        # Create tasks
        tasks = []
        for task_config in config.get("tasks", []):
            task = Task(
                description=task_config.get("description", ""),
                expected_output=task_config.get("expected_output", ""),
                agent=next(
                    (a for a in agents if a.role == task_config.get("agent")),
                    agents[0] if agents else None,
                ),
                async_execution=task_config.get("async", False),
                context=task_config.get("context", []),
            )
            tasks.append(task)
        
        # Create crew using CustomCrew instead of Crew
        crew = CustomCrew(
            agents=agents,
            tasks=tasks,
            verbose=config.get("verbose", False),
            process=config.get("process", "sequential"),
            memory=config.get("memory", False),
            cache=config.get("cache", False),
            manager_llm=self.llm_provider.get_llm(
                model=config.get("manager_model", "gemini-1.5-pro"),
                temperature=config.get("manager_temperature", 0.7),
                max_tokens=config.get("manager_max_tokens", 1024),
            ),
        )
        
        # Cache crew
        self._crew_cache[cache_key] = crew
        
        logger.info(f"Created crew: {crew_name}")
        return crew
    
    async def run_crew(self, crew_name: str, inputs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run a crew with the specified inputs.
        
        Args:
            crew_name: Name of the crew
            inputs: Input parameters for the crew
            
        Returns:
            Result of the crew execution
            
        Raises:
            ValueError: If crew configuration is not found
        """
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Create crew
            crew = self.create_crew(crew_name)
            
            # Create task ID
            import uuid
            task_id = str(uuid.uuid4())
            
            # Initialize inputs
            crew_inputs = inputs or {}
            
            # Add task ID to inputs
            crew_inputs["task_id"] = task_id
            
            # Run crew
            logger.info(f"Running crew: {crew_name} (task_id: {task_id})")
            result = crew.kickoff(inputs=crew_inputs)
            
            # Record metrics
            duration = asyncio.get_event_loop().time() - start_time
            observe_value("crew_task_duration_seconds", duration, {"crew": crew_name})
            
            # Process result
            if isinstance(result, dict):
                result["task_id"] = task_id
                return result
            else:
                return {
                    "success": True,
                    "result": result,
                    "task_id": task_id,
                }
            
        except Exception as e:
            logger.error(f"Error running crew: {e}")
            
            # Record error metric
            increment_counter("crew_errors_total", {"crew": crew_name, "type": type(e).__name__})
            
            return {
                "success": False,
                "error": str(e),
                "task_id": locals().get("task_id", "unknown"),
            }
    
    async def run_agent(self, agent_name: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a single agent with the specified context.
        
        Args:
            agent_name: Name of the agent
            context: Context for the agent
            
        Returns:
            Result of the agent execution
            
        Raises:
            ValueError: If agent configuration is not found
        """
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Create agent
            agent = self.create_agent(agent_name)
            
            # Run agent
            logger.info(f"Running agent: {agent_name}")
            result = agent.run(context)
            
            # Record metrics
            duration = asyncio.get_event_loop().time() - start_time
            observe_value("agent_task_duration_seconds", duration, {"agent": agent_name})
            
            return {
                "success": True,
                "result": result,
                "agent": agent_name,
            }
            
        except Exception as e:
            logger.error(f"Error running agent: {e}")
            
            # Record error metric
            increment_counter("agent_errors_total", {"agent": agent_name, "type": type(e).__name__})
            
            return {
                "success": False,
                "error": str(e),
                "agent": agent_name,
            }
    
    async def pause_crew(self, task_id: str, reason: Optional[str] = None) -> Dict[str, Any]:
        """
        Pause a running crew task for HITL review.
        
        Args:
            task_id: ID of the task to pause
            reason: Reason for pausing
            
        Returns:
            Status of the pause operation
        """
        logger.info(f"Pausing crew task: {task_id}")
        
        # Store task state
        self._task_state[task_id] = {
            "status": "PAUSED",
            "reason": reason,
            "paused_at": asyncio.get_event_loop().time(),
        }
        
        return {
            "success": True,
            "task_id": task_id,
            "status": "PAUSED",
        }
    
    async def resume_crew(
        self,
        task_id: str,
        approved: bool,
        comment: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Resume a paused crew task after HITL review.
        
        Args:
            task_id: ID of the task to resume
            approved: Whether the task is approved to continue
            comment: Comment from the reviewer
            
        Returns:
            Status of the resume operation
        """
        logger.info(f"Resuming crew task: {task_id} (approved: {approved})")
        
        # Check if task exists
        if task_id not in self._task_state:
            return {
                "success": False,
                "error": f"Task not found: {task_id}",
            }
        
        # Check if task is paused
        if self._task_state[task_id].get("status") != "PAUSED":
            return {
                "success": False,
                "error": f"Task is not paused: {task_id}",
            }
        
        # Update task state
        self._task_state[task_id].update({
            "status": "RESUMED" if approved else "REJECTED",
            "approved": approved,
            "comment": comment,
            "resumed_at": asyncio.get_event_loop().time(),
        })
        
        # Calculate pause duration
        paused_at = self._task_state[task_id].get("paused_at", 0)
        resumed_at = self._task_state[task_id].get("resumed_at", 0)
        pause_duration = resumed_at - paused_at
        
        # Record metrics
        observe_value(
            "hitl_review_duration_seconds",
            pause_duration,
            {"approved": str(approved).lower()}
        )
        
        return {
            "success": True,
            "task_id": task_id,
            "status": "RESUMED" if approved else "REJECTED",
            "approved": approved,
        }
    
    def get_task_state(self, task_id: str) -> Dict[str, Any]:
        """
        Get the state of a task.
        
        Args:
            task_id: ID of the task
            
        Returns:
            Task state dictionary
        """
        return self._task_state.get(task_id, {"status": "UNKNOWN"})
    
    def _get_tools_for_agent(self, tool_names: List[str]) -> List[Any]:
        """
        Get tools for an agent based on tool names.
        
        Args:
            tool_names: List of tool names
            
        Returns:
            List of tool instances
        """
        tools = []
        
        for name in tool_names:
            if name in self._tools:
                tools.append(self._tools[name])
            else:
                logger.warning(f"Tool not found: {name}")
        
        return tools
    
    def get_tool(self, tool_name: str) -> Optional[Any]:
        """
        Get a tool by name.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Tool instance or None if not found
        """
        return self._tools.get(tool_name)
    
    def reload_agent_config(self, agent_name: str) -> bool:
        """
        Reload an agent configuration from disk.
        
        Args:
            agent_name: Name of the agent
            
        Returns:
            True if configuration was reloaded, False otherwise
        """
        # Remove agent from cache
        to_remove = [k for k in self._agent_cache.keys() if k.startswith(f"{agent_name}:")]
        for key in to_remove:
            del self._agent_cache[key]
        
        # Remove crews that use this agent
        crew_configs = self.get_available_crews()
        for crew_name in crew_configs:
            try:
                config = self.get_crew_config(crew_name)
                agent_names = [a.get("name") for a in config.get("agents", [])]
                if agent_name in agent_names:
                    # Remove crew from cache
                    to_remove = [k for k in self._crew_cache.keys() if k.startswith(f"{crew_name}:")]
                    for key in to_remove:
                        del self._crew_cache[key]
            except Exception as e:
                logger.error(f"Error checking crew config: {e}")
        
        logger.info(f"Reloaded agent configuration: {agent_name}")
        return True
    
    def reload_crew_config(self, crew_name: str) -> bool:
        """
        Reload a crew configuration from disk.
        
        Args:
            crew_name: Name of the crew
            
        Returns:
            True if configuration was reloaded, False otherwise
        """
        # Remove crew from cache
        to_remove = [k for k in self._crew_cache.keys() if k.startswith(f"{crew_name}:")]
        for key in to_remove:
            del self._crew_cache[key]
        
        logger.info(f"Reloaded crew configuration: {crew_name}")
        return True
    
    async def close(self):
        """Close all resources."""
        # Close tools
        for tool_name, tool in self._tools.items():
            if hasattr(tool, "close") and callable(getattr(tool, "close")):
                try:
                    if asyncio.iscoroutinefunction(tool.close):
                        await tool.close()
                    else:
                        tool.close()
                except Exception as e:
                    logger.error(f"Error closing tool {tool_name}: {e}")
        
        # Clear caches
        self._agent_cache.clear()
        self._crew_cache.clear()
        
        logger.info("CrewFactory closed")

    def update_agent_prompt(self, agent_name: str, prompt_type: str, prompt: str) -> bool:
        """
        Update an agent prompt in the configuration.
        
        Args:
            agent_name: Name of the agent
            prompt_type: Type of prompt (role, goal, backstory)
            prompt: New prompt text
            
        Returns:
            True if prompt was updated, False otherwise
        """
        try:
            # Get agent configuration
            config_path = os.path.join(self.config_dir, "defaults", f"{agent_name}.yaml")
            
            if not os.path.exists(config_path):
                logger.error(f"Agent configuration not found: {agent_name}")
                return False
            
            # Load configuration
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
            
            # Update prompt
            if prompt_type in ["role", "goal", "backstory"]:
                config[prompt_type] = prompt
            else:
                logger.error(f"Invalid prompt type: {prompt_type}")
                return False
            
            # Save configuration
            with open(config_path, "w") as f:
                yaml.dump(config, f, default_flow_style=False)
            
            # Reload agent
            self.reload_agent_config(agent_name)
            
            logger.info(f"Updated {prompt_type} prompt for agent: {agent_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating agent prompt: {e}")
            return False
