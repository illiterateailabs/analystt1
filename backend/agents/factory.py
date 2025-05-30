from typing import Dict, List, Optional, Any, Type, Union
import os
import yaml
from pathlib import Path
import logging
from functools import lru_cache
import asyncio

from crewai import Agent, Crew, Task, Process
from crewai.agent import AgentConfig
from crewai.llm import LLM

# Import tools with graceful error handling
# Initialize all tools to None by default
GraphQueryTool = None
SandboxExecTool = None
CodeGenTool = None
PatternLibraryTool = None
PolicyDocsTool = None
TemplateEngineTool = None
Neo4jSchemaTool = None
RandomTxGeneratorTool = None

# Try to import each tool individually
try:
    from backend.agents.tools.graph_query_tool import GraphQueryTool
except ImportError:
    pass

try:
    from backend.agents.tools.sandbox_exec_tool import SandboxExecTool
except ImportError:
    pass

try:
    from backend.agents.tools.code_gen_tool import CodeGenTool
except ImportError:
    pass

try:
    from backend.agents.tools.pattern_library_tool import PatternLibraryTool
except ImportError:
    pass

try:
    from backend.agents.tools.policy_docs_tool import PolicyDocsTool
except ImportError:
    pass

try:
    from backend.agents.tools.template_engine_tool import TemplateEngineTool
except ImportError:
    pass

try:
    from backend.agents.tools.neo4j_schema_tool import Neo4jSchemaTool
except ImportError:
    pass

try:
    from backend.agents.tools.random_tx_generator_tool import RandomTxGeneratorTool
except ImportError:
    pass

from backend.agents.llm import get_llm
from backend.agents.config import load_agent_config, load_crew_config, get_available_crews
from backend.integrations.neo4j_client import Neo4jClient
from backend.integrations.e2b_client import E2BClient
from backend.core.logging import get_logger
from backend.core.metrics import track_crew_task_duration, increment_active_crews, decrement_active_crews

logger = get_logger(__name__)

# Path to agent configs directory
AGENT_CONFIGS_DIR = Path("backend/agents/configs")
AGENT_CONFIGS_CREWS_DIR = AGENT_CONFIGS_DIR / "crews"
DEFAULT_PROMPTS_DIR = AGENT_CONFIGS_DIR / "defaults"

# Ensure directories exist
AGENT_CONFIGS_DIR.mkdir(exist_ok=True)
AGENT_CONFIGS_CREWS_DIR.mkdir(exist_ok=True)
DEFAULT_PROMPTS_DIR.mkdir(exist_ok=True)


class CrewFactory:
    """
    Factory for creating and managing CrewAI agents and crews.
    
    This class handles the creation of agents with their tools,
    loading configurations from YAML files, and building crews.
    """
    
    # Cache for agent configurations
    _agent_config_cache: Dict[str, Dict[str, Any]] = {}
    
    # Cache for created agents
    _agent_cache: Dict[str, Agent] = {}
    
    # Cache for created tools
    _tool_cache: Dict[str, Any] = {}
    
    @classmethod
    def get_agent_config(cls, agent_id: str) -> Optional[Dict[str, Any]]:
        """
        Get an agent's configuration from cache or load from file.
        
        Args:
            agent_id: The unique identifier for the agent
            
        Returns:
            The agent configuration dictionary or None if not found
        """
        # Check cache first
        if agent_id in cls._agent_config_cache:
            return cls._agent_config_cache[agent_id]
        
        # Try to load from custom config
        custom_config_path = AGENT_CONFIGS_DIR / f"{agent_id}.yaml"
        if custom_config_path.exists():
            try:
                with open(custom_config_path, "r") as f:
                    config = yaml.safe_load(f)
                    if config:
                        cls._agent_config_cache[agent_id] = config
                        return config
            except Exception as e:
                logger.error(f"Error loading custom config for {agent_id}: {str(e)}")
        
        # Try to load from default config
        default_config_path = DEFAULT_PROMPTS_DIR / f"{agent_id}.yaml"
        if default_config_path.exists():
            try:
                with open(default_config_path, "r") as f:
                    config = yaml.safe_load(f)
                    if config:
                        cls._agent_config_cache[agent_id] = config
                        return config
            except Exception as e:
                logger.error(f"Error loading default config for {agent_id}: {str(e)}")
        
        # Try to find in crew configs
        for crew_file in AGENT_CONFIGS_CREWS_DIR.glob("*.yaml"):
            try:
                with open(crew_file, "r") as f:
                    crew_data = yaml.safe_load(f)
                    if crew_data and "agents" in crew_data:
                        for agent in crew_data["agents"]:
                            if isinstance(agent, dict) and agent.get("id") == agent_id:
                                cls._agent_config_cache[agent_id] = agent
                                return agent
            except Exception as e:
                logger.error(f"Error loading agents from crew file {crew_file}: {str(e)}")
        
        return None
    
    @classmethod
    def update_agent_prompt(cls, agent_id: str, system_prompt: str) -> None:
        """
        Update an agent's system prompt in the cache.
        
        Args:
            agent_id: The unique identifier for the agent
            system_prompt: The new system prompt
        """
        # Get existing config or create new one
        config = cls.get_agent_config(agent_id) or {}
        
        # Update system prompt
        config["system_prompt"] = system_prompt
        
        # Update cache
        cls._agent_config_cache[agent_id] = config
        
        # Remove from agent cache to force recreation with new prompt
        if agent_id in cls._agent_cache:
            del cls._agent_cache[agent_id]
            logger.info(f"Removed agent {agent_id} from cache due to prompt update")
    
    @classmethod
    def reset_agent_prompt(cls, agent_id: str) -> None:
        """
        Reset an agent's prompt to default.
        
        Args:
            agent_id: The unique identifier for the agent
        """
        # Remove from config cache
        if agent_id in cls._agent_config_cache:
            del cls._agent_config_cache[agent_id]
        
        # Remove from agent cache
        if agent_id in cls._agent_cache:
            del cls._agent_cache[agent_id]
            logger.info(f"Removed agent {agent_id} from cache due to prompt reset")
    
    @classmethod
    def get_available_crews(cls) -> List[str]:
        """
        Get a list of available crew names.
        
        Returns:
            List of crew names
        """
        return get_available_crews()
    
    @classmethod
    def create_tool(cls, tool_type: str, **kwargs) -> Any:
        """
        Create a tool instance based on the tool type.
        
        Args:
            tool_type: The type of tool to create
            **kwargs: Additional arguments for the tool
            
        Returns:
            The created tool instance
        """
        # Create a cache key based on tool type and kwargs
        cache_key = f"{tool_type}:{hash(frozenset(kwargs.items()))}"
        
        # Check cache
        if cache_key in cls._tool_cache:
            return cls._tool_cache[cache_key]
        
        # Create the tool based on type
        tool = None
        try:
            if tool_type == "GraphQueryTool":
                if GraphQueryTool is None:
                    logger.warning(f"GraphQueryTool is not available")
                    return None
                neo4j_client = Neo4jClient()
                tool = GraphQueryTool(neo4j_client=neo4j_client, **kwargs)
            elif tool_type == "SandboxExecTool":
                if SandboxExecTool is None:
                    logger.warning(f"SandboxExecTool is not available")
                    return None
                e2b_client = E2BClient()
                tool = SandboxExecTool(e2b_client=e2b_client, **kwargs)
            elif tool_type == "CodeGenTool":
                if CodeGenTool is None:
                    logger.warning(f"CodeGenTool is not available")
                    return None
                tool = CodeGenTool(**kwargs)
            elif tool_type == "PatternLibraryTool":
                if PatternLibraryTool is None:
                    logger.warning(f"PatternLibraryTool is not available")
                    return None
                tool = PatternLibraryTool(**kwargs)
            elif tool_type == "PolicyDocsTool":
                if PolicyDocsTool is None:
                    logger.warning(f"PolicyDocsTool is not available")
                    return None
                tool = PolicyDocsTool(**kwargs)
            elif tool_type == "TemplateEngineTool":
                if TemplateEngineTool is None:
                    logger.warning(f"TemplateEngineTool is not available")
                    return None
                tool = TemplateEngineTool(**kwargs)
            elif tool_type == "Neo4jSchemaTool":
                if Neo4jSchemaTool is None:
                    logger.warning(f"Neo4jSchemaTool is not available")
                    return None
                tool = Neo4jSchemaTool(**kwargs)
            elif tool_type == "RandomTxGeneratorTool":
                if RandomTxGeneratorTool is None:
                    logger.warning(f"RandomTxGeneratorTool is not available")
                    return None
                tool = RandomTxGeneratorTool(**kwargs)
            else:
                logger.warning(f"Unknown tool type: {tool_type}")
                return None
        except ImportError as e:
            logger.warning(f"Could not import tool {tool_type}: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Error creating tool {tool_type}: {str(e)}")
            return None
        
        # Cache the tool
        cls._tool_cache[cache_key] = tool
        
        return tool
    
    @classmethod
    def _create_agent(cls, agent_id: str, **override_kwargs) -> Agent:
        """
        Create an agent with the given ID and optional overrides.
        
        Args:
            agent_id: The unique identifier for the agent
            **override_kwargs: Optional overrides for agent configuration
            
        Returns:
            The created Agent instance
        """
        # Check if agent is already in cache
        if agent_id in cls._agent_cache:
            return cls._agent_cache[agent_id]
        
        # Get agent configuration
        agent_config = cls.get_agent_config(agent_id) or {}
        
        # Apply overrides
        agent_config.update(override_kwargs)
        
        # Get required parameters
        role = agent_config.get("role", f"Agent {agent_id}")
        goal = agent_config.get("goal", "Assist the user with their tasks")
        backstory = agent_config.get("backstory", f"You are {role}, an AI assistant.")
        system_prompt = agent_config.get("system_prompt", "")
        
        # Create tools if specified
        tools = []
        if "tools" in agent_config:
            for tool_config in agent_config["tools"]:
                if isinstance(tool_config, str):
                    # Simple tool reference
                    tool = cls.create_tool(tool_config)
                    if tool:
                        tools.append(tool)
                elif isinstance(tool_config, dict) and "type" in tool_config:
                    # Tool with parameters
                    tool_type = tool_config.pop("type")
                    tool = cls.create_tool(tool_type, **tool_config)
                    if tool:
                        tools.append(tool)
        
        # Get LLM
        llm_config = agent_config.get("llm", {})
        if isinstance(llm_config, str):
            # Simple model name
            llm = get_llm(model=llm_config)
        else:
            # LLM with parameters
            llm = get_llm(**llm_config)
        
        # Check for multimodal flag
        multimodal = agent_config.get("multimodal", False)
        
        # Create the agent
        agent = Agent(
            role=role,
            goal=goal,
            backstory=backstory,
            llm=llm,
            tools=tools,
            verbose=True,
            allow_delegation=agent_config.get("allow_delegation", False),
            max_iter=agent_config.get("max_iter", 15),
            max_rpm=agent_config.get("max_rpm", None),
            system_prompt=system_prompt,
            config=AgentConfig(
                multimodal=multimodal
            )
        )
        
        # Cache the agent
        cls._agent_cache[agent_id] = agent
        
        return agent
    
    @classmethod
    def create_crew(cls, crew_name: str, **override_kwargs) -> Optional[Crew]:
        """
        Create a crew with the given name and optional overrides.
        
        Args:
            crew_name: The name of the crew to create
            **override_kwargs: Optional overrides for crew configuration
            
        Returns:
            The created Crew instance or None if creation fails
        """
        # Load crew configuration
        crew_config_path = AGENT_CONFIGS_CREWS_DIR / f"{crew_name}.yaml"
        if not crew_config_path.exists():
            logger.error(f"Crew configuration not found: {crew_name}")
            return None
        
        try:
            with open(crew_config_path, "r") as f:
                crew_config = yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading crew configuration {crew_name}: {str(e)}")
            return None
        
        # Apply overrides
        crew_config.update(override_kwargs)
        
        # Get process type
        process_type_str = crew_config.get("process_type", "sequential").lower()
        if process_type_str == "sequential":
            process_type = Process.sequential
        elif process_type_str == "hierarchical":
            process_type = Process.hierarchical
        else:
            logger.warning(f"Unknown process type: {process_type_str}, defaulting to sequential")
            process_type = Process.sequential
        
        # Create agents
        agents = []
        if "agents" in crew_config:
            for agent_config in crew_config["agents"]:
                if isinstance(agent_config, str):
                    # Simple agent reference
                    agent = cls._create_agent(agent_config)
                    agents.append(agent)
                elif isinstance(agent_config, dict) and "id" in agent_config:
                    # Agent with overrides
                    agent_id = agent_config.pop("id")
                    agent = cls._create_agent(agent_id, **agent_config)
                    agents.append(agent)
        
        # Create tasks if specified
        tasks = []
        if "tasks" in crew_config:
            for task_config in crew_config["tasks"]:
                if not isinstance(task_config, dict):
                    continue
                
                # Get required parameters
                description = task_config.get("description", "")
                agent_id = task_config.get("agent", None)
                
                # Skip if no description or agent
                if not description or not agent_id:
                    continue
                
                # Find the agent
                agent = next((a for a in agents if a.role.lower() == agent_id.lower()), None)
                if not agent:
                    # Try to create the agent
                    agent = cls._create_agent(agent_id)
                    if agent:
                        agents.append(agent)
                    else:
                        logger.warning(f"Agent not found for task: {agent_id}")
                        continue
                
                # Create the task
                task = Task(
                    description=description,
                    agent=agent,
                    expected_output=task_config.get("expected_output", None),
                    async_execution=task_config.get("async_execution", False),
                )
                
                # Add context if specified
                if "context" in task_config:
                    context_tasks = []
                    for context_task_id in task_config["context"]:
                        context_task = next((t for t in tasks if t.description == context_task_id), None)
                        if context_task:
                            context_tasks.append(context_task)
                    
                    if context_tasks:
                        task.context = context_tasks
                
                tasks.append(task)
        
        # Create the crew
        try:
            crew = Crew(
                agents=agents,
                tasks=tasks,
                process=process_type,
                verbose=True,
                memory=crew_config.get("memory", False),
                max_rpm=crew_config.get("max_rpm", None),
                manager_llm=get_llm() if process_type == Process.hierarchical else None,
            )
            
            return crew
        except Exception as e:
            logger.error(f"Error creating crew {crew_name}: {str(e)}")
            return None
    
    @classmethod
    async def run_crew(cls, crew_name: str, inputs: Dict[str, Any] = None, task_id: str = None, resume: bool = False, **override_kwargs) -> Dict[str, Any]:
        """
        Run a crew with the given name and inputs.
        
        Args:
            crew_name: The name of the crew to run
            inputs: Optional inputs for the crew
            task_id: Optional task ID for tracking
            resume: Whether this is resuming a paused execution
            **override_kwargs: Optional overrides for crew configuration
            
        Returns:
            The crew execution results
        """
        # Create the crew
        crew = cls.create_crew(crew_name, **override_kwargs)
        if not crew:
            return {
                "success": False,
                "error": f"Failed to create crew: {crew_name}"
            }
        
        # Track active crew
        increment_active_crews(crew_name)
        
        # Run the crew
        try:
            # Get tasks
            tasks = crew.tasks
            
            # Initialize results
            all_results = {}
            
            # Execute tasks
            for i, task in enumerate(tasks):
                agent_id = task.agent.role
                task_description = task.description
                
                # Use context manager to track task duration and metrics
                with track_crew_task_duration(crew_name, agent_id, task_description):
                    # If resuming and this is the compliance_checker task that was paused
                    if resume and agent_id.lower() == "compliance_checker" and i > 0:
                        # Add review result to task context
                        compliance_result = inputs.get("compliance_review_result", {})
                        logger.info(f"Resuming with compliance review result: {compliance_result}")
                        
                        # Execute task with review result
                        task_result = await task.execute(
                            context=[all_results[prev_task.description] for prev_task in tasks[:i] if prev_task.description in all_results],
                            inputs={**inputs, "review_result": compliance_result}
                        )
                    else:
                        # Normal task execution
                        task_result = await task.execute(
                            context=[all_results[prev_task.description] for prev_task in tasks[:i] if prev_task.description in all_results],
                            inputs=inputs
                        )
                    
                    # Store result
                    all_results[task.description] = task_result
            
            # Final result is the last task's result
            result = all_results[tasks[-1].description] if tasks else None
            
            # Decrement active crew count
            decrement_active_crews(crew_name)
            
            return {
                "success": True,
                "result": result
            }
        except Exception as e:
            # Decrement active crew count
            decrement_active_crews(crew_name)
            
            logger.error(f"Error running crew {crew_name}: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    @classmethod
    def clear_caches(cls) -> None:
        """Clear all caches."""
        cls._agent_config_cache.clear()
        cls._agent_cache.clear()
        cls._tool_cache.clear()
        logger.info("All caches cleared")
    
    # Connection management for external services
    @classmethod
    async def connect(cls) -> None:
        """Connect to external services."""
        # Connect to Neo4j
        try:
            neo4j_client = Neo4jClient()
            await neo4j_client.connect()
            logger.info("Connected to Neo4j")
        except Exception as e:
            logger.warning(f"Failed to connect to Neo4j: {str(e)}")
        
        # Connect to e2b
        try:
            e2b_client = E2BClient()
            # e2b connection is lazy, no explicit connect needed
            logger.info("e2b client initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize e2b client: {str(e)}")
    
    @classmethod
    async def close(cls) -> None:
        """Close connections to external services."""
        # Close Neo4j connection
        try:
            neo4j_client = Neo4jClient()
            await neo4j_client.close()
            logger.info("Closed Neo4j connection")
        except Exception as e:
            logger.warning(f"Error closing Neo4j connection: {str(e)}")
