"""
Custom Crew Implementation with Multiple Execution Modes

This module provides a flexible implementation of CrewAI crews with support for
different execution strategies:
- Sequential: Tasks are executed in a predefined sequence
- Hierarchical: Tasks are delegated by a manager agent to specialized agents
- Planning: A planner agent creates and executes a dynamic plan

Configuration is loaded from YAML files following the crew config convention,
with support for Human-in-the-Loop (HITL) integration and robust error handling.
"""

import enum
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Type, Union, cast

import yaml
from crewai import Agent, Crew, Process, Task
from crewai.agents import CrewAgentExecutor
from crewai.tasks import TaskOutput
from pydantic import BaseModel, Field, validator

from backend.agents.llm import get_llm_provider
from backend.agents.tools.base_tool import AbstractApiTool
from backend.core.events import (
    AgentExecutionEvent,
    EventPriority,
    publish_event,
)
from backend.core.metrics import AgentMetrics

# Configure module logger
logger = logging.getLogger(__name__)


class CrewMode(str, enum.Enum):
    """Execution modes for crews."""
    SEQUENTIAL = "sequential"
    HIERARCHICAL = "hierarchical"
    PLANNING = "planning"


class CrewConfig(BaseModel):
    """Configuration for a crew loaded from YAML."""
    metadata: Dict[str, Any] = Field(default_factory=dict)
    environment: Dict[str, Any] = Field(default_factory=dict)
    agents: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    tasks: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    workflows: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    tools: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    hitl: Optional[Dict[str, Any]] = None
    dependencies: Optional[Dict[str, Any]] = None
    monitoring: Optional[Dict[str, Any]] = None
    validation: Optional[Dict[str, Any]] = None


class TaskConfig(BaseModel):
    """Configuration for a task loaded from YAML."""
    name: str
    description: str
    assigned_to: str
    inputs: List[Dict[str, Any]] = Field(default_factory=list)
    outputs: List[Dict[str, Any]] = Field(default_factory=list)
    tools: List[str] = Field(default_factory=list)
    success_criteria: List[str] = Field(default_factory=list)
    template: Optional[str] = None
    timeout_minutes: Optional[int] = None
    hitl: Optional[Dict[str, Any]] = None
    retry: Optional[Dict[str, Any]] = None


class AgentConfig(BaseModel):
    """Configuration for an agent loaded from YAML."""
    name: str
    role: str
    description: str
    backstory: str
    goals: List[str] = Field(default_factory=list)
    constraints: List[str] = Field(default_factory=list)
    tools: List[str] = Field(default_factory=list)
    allowed_task_delegations: List[str] = Field(default_factory=list)
    llm: Optional[Dict[str, Any]] = None


class WorkflowConfig(BaseModel):
    """Configuration for a workflow loaded from YAML."""
    name: str
    description: str
    trigger: Dict[str, Any] = Field(default_factory=dict)
    tasks: List[Dict[str, Any]] = Field(default_factory=list)
    outputs: List[Dict[str, Any]] = Field(default_factory=list)
    success_criteria: List[str] = Field(default_factory=list)
    timeout_minutes: Optional[int] = None
    retry: Optional[Dict[str, Any]] = None


class HITLConfig(BaseModel):
    """Configuration for Human-in-the-Loop integration."""
    enabled: bool = True
    review_points: List[Dict[str, Any]] = Field(default_factory=list)
    notification: Dict[str, Any] = Field(default_factory=dict)
    ui: Optional[Dict[str, Any]] = None


class HITLReview(BaseModel):
    """Model for a HITL review request."""
    crew_id: str
    task_id: str
    agent_id: str
    timestamp: str
    task_name: str
    task_output: Dict[str, Any]
    confidence: float
    review_type: str = "approval"  # approval, feedback, or edit
    status: str = "pending"  # pending, approved, rejected, or feedback_provided
    feedback: Optional[str] = None
    edited_output: Optional[Dict[str, Any]] = None
    timeout_at: Optional[str] = None


class CustomCrew:
    """
    Custom implementation of CrewAI crews with support for different execution modes,
    configuration from YAML files, and HITL integration.
    """
    
    def __init__(
        self,
        config_path: Union[str, Path],
        workflow_name: Optional[str] = None,
        mode: Optional[CrewMode] = None,
        tools: Optional[Dict[str, AbstractApiTool]] = None,
        hitl_callback: Optional[callable] = None,
        environment_vars: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize a custom crew from a YAML configuration file.
        
        Args:
            config_path: Path to the crew configuration YAML file
            workflow_name: Name of the workflow to execute (default: first workflow in config)
            mode: Execution mode (default: from CREW_MODE env var or config)
            tools: Dictionary of tools to use (default: auto-discover)
            hitl_callback: Callback function for HITL integration
            environment_vars: Additional environment variables to override config
        """
        self.config_path = Path(config_path)
        self.tools = tools or {}
        self.hitl_callback = hitl_callback
        self.environment_vars = environment_vars or {}
        
        # Load and process configuration
        self.config = self._load_config(self.config_path)
        self.domain = self.config.metadata.get("domain", "unknown")
        
        # Determine execution mode
        self.mode = mode or self._get_execution_mode()
        logger.info(f"Initializing crew in {self.mode} mode")
        
        # Select workflow
        self.workflow_name = workflow_name or self._get_default_workflow_name()
        if self.workflow_name not in self.config.workflows:
            raise ValueError(f"Workflow '{self.workflow_name}' not found in configuration")
        self.workflow = WorkflowConfig(**self.config.workflows[self.workflow_name])
        
        # Initialize components
        self.agents = {}
        self.tasks = {}
        self.task_dependencies = {}
        self._initialize_components()
        
        # Create CrewAI crew based on mode
        self.crew = self._create_crew()
    
    def _load_config(self, config_path: Path) -> CrewConfig:
        """
        Load configuration from a YAML file with environment variable substitution.
        
        Args:
            config_path: Path to the configuration file
            
        Returns:
            Parsed configuration
            
        Raises:
            FileNotFoundError: If the configuration file doesn't exist
            ValueError: If the configuration is invalid
        """
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        try:
            # Load YAML with environment variable substitution
            with open(config_path, "r") as f:
                config_str = f.read()
            
            # Substitute environment variables
            import re
            import os
            
            def replace_env_var(match):
                env_var = match.group(1)
                default = match.group(2) if match.group(2) is not None else ""
                
                # Check custom environment vars first, then system environment
                if env_var in self.environment_vars:
                    return str(self.environment_vars[env_var])
                return os.environ.get(env_var, default)
            
            # Replace ${VAR:-default} patterns
            pattern = r'\${([A-Za-z0-9_]+)(?::-([^}]*))?}'
            config_str = re.sub(pattern, replace_env_var, config_str)
            
            # Parse YAML
            config_dict = yaml.safe_load(config_str)
            return CrewConfig(**config_dict)
        
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in configuration: {e}")
        except Exception as e:
            raise ValueError(f"Error loading configuration: {e}")
    
    def _get_execution_mode(self) -> CrewMode:
        """
        Determine the execution mode from environment variables or configuration.
        
        Returns:
            Execution mode enum value
        """
        # Check custom environment vars first
        if "CREW_MODE" in self.environment_vars:
            mode_str = self.environment_vars["CREW_MODE"]
        else:
            # Then check system environment
            mode_str = os.environ.get("CREW_MODE")
        
        # Then check configuration
        if not mode_str and "variables" in self.config.environment:
            mode_str = self.config.environment["variables"].get("CREW_MODE")
        
        # Default to sequential if not specified
        if not mode_str:
            return CrewMode.SEQUENTIAL
        
        try:
            return CrewMode(mode_str.lower())
        except ValueError:
            logger.warning(f"Invalid CREW_MODE '{mode_str}', defaulting to sequential")
            return CrewMode.SEQUENTIAL
    
    def _get_default_workflow_name(self) -> str:
        """
        Get the default workflow name from configuration.
        
        Returns:
            Name of the default workflow
        """
        if not self.config.workflows:
            raise ValueError("No workflows defined in configuration")
        
        # Return the first workflow
        return next(iter(self.config.workflows.keys()))
    
    def _initialize_components(self) -> None:
        """Initialize agents, tasks, and task dependencies from configuration."""
        # Initialize tools first if not provided
        if not self.tools:
            self._initialize_tools()
        
        # Initialize agents
        self._initialize_agents()
        
        # Initialize tasks based on workflow
        self._initialize_tasks()
        
        # Build task dependencies
        self._build_task_dependencies()
    
    def _initialize_tools(self) -> None:
        """
        Initialize tools based on configuration.
        
        This method auto-discovers tools based on the configuration and
        instantiates them with the appropriate parameters.
        """
        from importlib import import_module
        
        # Get tool configurations
        tool_configs = self.config.tools or {}
        
        for tool_id, config in tool_configs.items():
            try:
                # Try to import the tool class
                module_path = f"backend.agents.tools.{tool_id}"
                module = import_module(module_path)
                
                # Find the tool class (usually named *Tool)
                tool_class = None
                for attr_name in dir(module):
                    if attr_name.endswith("Tool") and attr_name != "AbstractApiTool":
                        tool_class = getattr(module, attr_name)
                        break
                
                if not tool_class:
                    logger.warning(f"Could not find tool class in {module_path}")
                    continue
                
                # Initialize the tool with configuration
                tool_instance = tool_class(**config.get("config", {}))
                self.tools[tool_id] = tool_instance
                logger.debug(f"Initialized tool: {tool_id}")
            
            except ImportError:
                logger.warning(f"Could not import tool: {tool_id}")
            except Exception as e:
                logger.error(f"Error initializing tool {tool_id}: {e}")
    
    def _initialize_agents(self) -> None:
        """
        Initialize agents based on configuration.
        
        Creates CrewAI Agent objects for each agent defined in the configuration.
        """
        # Get LLM provider
        llm_config = self.config.environment.get("llm", {})
        default_llm = get_llm_provider(**llm_config)
        
        # Initialize agents
        for agent_id, agent_config_dict in self.config.agents.items():
            try:
                # Parse agent configuration
                agent_config = AgentConfig(**agent_config_dict)
                
                # Get agent-specific LLM if configured
                agent_llm = default_llm
                if agent_config.llm:
                    # Merge with default LLM config
                    merged_config = {**llm_config, **agent_config.llm}
                    agent_llm = get_llm_provider(**merged_config)
                
                # Get tools for this agent
                agent_tools = []
                for tool_id in agent_config.tools:
                    if tool_id in self.tools:
                        agent_tools.append(self.tools[tool_id])
                    else:
                        logger.warning(f"Tool {tool_id} not found for agent {agent_id}")
                
                # Create CrewAI Agent
                agent = Agent(
                    role=agent_config.role,
                    goal="\n".join(agent_config.goals),
                    backstory=agent_config.backstory,
                    verbose=True,
                    llm=agent_llm,
                    tools=agent_tools,
                    allow_delegation=bool(agent_config.allowed_task_delegations),
                )
                
                # Store agent with metadata
                self.agents[agent_id] = {
                    "agent": agent,
                    "config": agent_config,
                }
                
                logger.debug(f"Initialized agent: {agent_id}")
            
            except Exception as e:
                logger.error(f"Error initializing agent {agent_id}: {e}")
                raise ValueError(f"Failed to initialize agent {agent_id}: {e}")
    
    def _initialize_tasks(self) -> None:
        """
        Initialize tasks based on workflow configuration.
        
        Creates CrewAI Task objects for each task in the workflow.
        """
        # Get tasks from workflow
        workflow_tasks = self.workflow.tasks
        
        for task_entry in workflow_tasks:
            task_id = task_entry["id"]
            task_name = task_entry["task"]
            
            # Get task configuration
            if task_name not in self.config.tasks:
                raise ValueError(f"Task '{task_name}' not found in configuration")
            
            task_config_dict = self.config.tasks[task_name]
            task_config = TaskConfig(**task_config_dict)
            
            # Get assigned agent
            agent_id = task_config.assigned_to
            if agent_id not in self.agents:
                raise ValueError(f"Agent '{agent_id}' not found for task '{task_name}'")
            
            agent = self.agents[agent_id]["agent"]
            
            # Load task template if specified
            description = task_config.description
            if task_config.template:
                template_path = self.config_path.parent / task_config.template
                if template_path.exists():
                    try:
                        task_template = self._load_config(template_path)
                        # Enhance description with template instructions
                        if "instructions" in task_template.dict():
                            instructions = task_template.dict()["instructions"]
                            if "overview" in instructions:
                                description = f"{description}\n\n{instructions['overview']}"
                    except Exception as e:
                        logger.warning(f"Error loading task template {template_path}: {e}")
            
            # Create CrewAI Task
            task = Task(
                description=description,
                agent=agent,
                expected_output="\n".join(f"- {output}" for output in task_config.success_criteria),
                context=self._get_task_context(task_config),
            )
            
            # Store task with metadata
            self.tasks[task_id] = {
                "task": task,
                "config": task_config,
                "name": task_name,
            }
            
            logger.debug(f"Initialized task: {task_id} ({task_name})")
    
    def _get_task_context(self, task_config: TaskConfig) -> str:
        """
        Generate context for a task based on its configuration.
        
        Args:
            task_config: Task configuration
            
        Returns:
            Context string for the task
        """
        context_parts = []
        
        # Add inputs description
        if task_config.inputs:
            context_parts.append("## Inputs")
            for input_def in task_config.inputs:
                required = input_def.get("required", False)
                req_str = " (Required)" if required else " (Optional)"
                context_parts.append(f"- {input_def['name']}{req_str}: {input_def['description']}")
        
        # Add outputs description
        if task_config.outputs:
            context_parts.append("\n## Expected Outputs")
            for output_def in task_config.outputs:
                context_parts.append(f"- {output_def['name']}: {output_def['description']}")
        
        # Add success criteria
        if task_config.success_criteria:
            context_parts.append("\n## Success Criteria")
            for criterion in task_config.success_criteria:
                context_parts.append(f"- {criterion}")
        
        # Add tools information
        if task_config.tools:
            context_parts.append("\n## Available Tools")
            for tool_id in task_config.tools:
                if tool_id in self.tools:
                    tool = self.tools[tool_id]
                    context_parts.append(f"- {tool_id}: {tool.description}")
        
        return "\n".join(context_parts)
    
    def _build_task_dependencies(self) -> None:
        """Build task dependencies based on workflow configuration."""
        self.task_dependencies = {}
        
        # Process workflow tasks
        for task_entry in self.workflow.tasks:
            task_id = task_entry["id"]
            depends_on = task_entry.get("depends_on", [])
            next_tasks = task_entry.get("next", [])
            
            # Store dependencies
            self.task_dependencies[task_id] = {
                "depends_on": depends_on,
                "next": next_tasks,
            }
        
        # Validate dependencies
        self._validate_task_dependencies()
    
    def _validate_task_dependencies(self) -> None:
        """
        Validate task dependencies to ensure there are no cycles or missing tasks.
        
        Raises:
            ValueError: If dependencies are invalid
        """
        # Check for missing tasks
        all_tasks = set(self.tasks.keys())
        referenced_tasks = set()
        
        for task_id, deps in self.task_dependencies.items():
            referenced_tasks.update(deps["depends_on"])
            referenced_tasks.update(deps["next"])
        
        missing_tasks = referenced_tasks - all_tasks
        if missing_tasks:
            raise ValueError(f"Referenced tasks not found: {missing_tasks}")
        
        # Check for cycles using depth-first search
        visited = set()
        temp_visited = set()
        
        def has_cycle(task_id):
            if task_id in temp_visited:
                return True
            if task_id in visited:
                return False
            
            temp_visited.add(task_id)
            
            for next_task in self.task_dependencies.get(task_id, {}).get("next", []):
                if has_cycle(next_task):
                    return True
            
            temp_visited.remove(task_id)
            visited.add(task_id)
            return False
        
        for task_id in self.tasks:
            if task_id not in visited:
                if has_cycle(task_id):
                    raise ValueError(f"Cycle detected in task dependencies starting from {task_id}")
    
    def _create_crew(self) -> Crew:
        """
        Create a CrewAI Crew based on the execution mode.
        
        Returns:
            CrewAI Crew object
        """
        # Determine process based on mode
        if self.mode == CrewMode.SEQUENTIAL:
            process = Process.sequential
        elif self.mode == CrewMode.HIERARCHICAL:
            process = Process.hierarchical
        elif self.mode == CrewMode.PLANNING:
            process = Process.planning
        else:
            logger.warning(f"Unknown mode {self.mode}, defaulting to sequential")
            process = Process.sequential
        
        # Get tasks in the right order based on dependencies
        ordered_tasks = self._get_ordered_tasks()
        
        # Create crew
        crew = Crew(
            agents=[self.agents[agent_id]["agent"] for agent_id in self.agents],
            tasks=[self.tasks[task_id]["task"] for task_id in ordered_tasks],
            verbose=True,
            process=process,
            manager_llm=get_llm_provider(**self.config.environment.get("llm", {})),
        )
        
        return crew
    
    def _get_ordered_tasks(self) -> List[str]:
        """
        Get tasks in topological order based on dependencies.
        
        Returns:
            List of task IDs in execution order
        """
        # For sequential mode, we need to sort tasks by dependencies
        if self.mode == CrewMode.SEQUENTIAL:
            return self._topological_sort()
        
        # For other modes, return tasks in the order they appear in the workflow
        return [task_entry["id"] for task_entry in self.workflow.tasks]
    
    def _topological_sort(self) -> List[str]:
        """
        Perform topological sort on tasks based on dependencies.
        
        Returns:
            List of task IDs in topological order
            
        Raises:
            ValueError: If there's a cycle in the dependencies
        """
        # Build adjacency list
        graph = {task_id: set() for task_id in self.tasks}
        for task_id, deps in self.task_dependencies.items():
            for dep in deps["depends_on"]:
                graph[dep].add(task_id)
        
        # Find tasks with no dependencies
        no_deps = [task_id for task_id in self.tasks if not self.task_dependencies.get(task_id, {}).get("depends_on")]
        
        # Perform topological sort
        result = []
        while no_deps:
            current = no_deps.pop(0)
            result.append(current)
            
            # Remove edges from current to its dependents
            dependents = list(graph[current])
            for dependent in dependents:
                graph[current].remove(dependent)
                
                # If dependent has no more dependencies, add it to no_deps
                if all(dependent not in graph[dep] for dep in self.task_dependencies.get(dependent, {}).get("depends_on", [])):
                    no_deps.append(dependent)
        
        # Check if all tasks are included
        if len(result) != len(self.tasks):
            raise ValueError("Cycle detected in task dependencies")
        
        return result
    
    def run(self, inputs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run the crew with the specified inputs.
        
        Args:
            inputs: Input values for the crew
            
        Returns:
            Dictionary of outputs from the crew
        """
        start_time = time.time()
        inputs = inputs or {}
        
        try:
            # Track crew execution
            logger.info(f"Running crew with workflow '{self.workflow_name}' in {self.mode} mode")
            
            # Set up HITL if enabled
            self._setup_hitl()
            
            # Run crew based on mode
            if self.mode == CrewMode.SEQUENTIAL:
                result = self._run_sequential(inputs)
            elif self.mode == CrewMode.HIERARCHICAL:
                result = self._run_hierarchical(inputs)
            elif self.mode == CrewMode.PLANNING:
                result = self._run_planning(inputs)
            else:
                raise ValueError(f"Unknown mode: {self.mode}")
            
            # Process and validate results
            outputs = self._process_results(result)
            
            # Track execution metrics
            duration_ms = (time.time() - start_time) * 1000
            self._track_execution_metrics(duration_ms, True)
            
            return outputs
        
        except Exception as e:
            # Track execution failure
            duration_ms = (time.time() - start_time) * 1000
            self._track_execution_metrics(duration_ms, False, str(e))
            
            # Re-raise with context
            logger.error(f"Crew execution failed: {e}")
            raise RuntimeError(f"Crew execution failed: {e}")
    
    def _run_sequential(self, inputs: Dict[str, Any]) -> Dict[str, TaskOutput]:
        """
        Run crew in sequential mode.
        
        Args:
            inputs: Input values for the crew
            
        Returns:
            Dictionary of task outputs
        """
        # In sequential mode, we need to manually pass outputs between tasks
        task_outputs = {}
        
        # Get tasks in topological order
        ordered_tasks = self._get_ordered_tasks()
        
        for task_id in ordered_tasks:
            task_info = self.tasks[task_id]
            task = task_info["task"]
            
            # Prepare context with outputs from dependencies
            task_inputs = self._prepare_task_inputs(task_id, inputs, task_outputs)
            
            # Update task context with inputs
            task.context = self._update_task_context(task_info["config"], task.context, task_inputs)
            
            # Execute task with HITL if configured
            result = self._execute_task_with_hitl(task_id, task)
            
            # Store result
            task_outputs[task_id] = result
        
        return task_outputs
    
    def _run_hierarchical(self, inputs: Dict[str, Any]) -> Dict[str, TaskOutput]:
        """
        Run crew in hierarchical mode.
        
        Args:
            inputs: Input values for the crew
            
        Returns:
            Dictionary of task outputs
        """
        # In hierarchical mode, CrewAI handles task delegation
        # We just need to set up the initial context
        
        # Find the manager agent (first agent in the workflow)
        manager_task_id = self.workflow.tasks[0]["id"]
        manager_task = self.tasks[manager_task_id]["task"]
        
        # Update task context with inputs
        manager_task.context = self._update_task_context(
            self.tasks[manager_task_id]["config"],
            manager_task.context,
            inputs
        )
        
        # Run the crew
        result = self.crew.kickoff()
        
        # Process results into task outputs
        task_outputs = {}
        for task_id, task_info in self.tasks.items():
            task = task_info["task"]
            if task.output:
                task_outputs[task_id] = task.output
        
        return task_outputs
    
    def _run_planning(self, inputs: Dict[str, Any]) -> Dict[str, TaskOutput]:
        """
        Run crew in planning mode.
        
        Args:
            inputs: Input values for the crew
            
        Returns:
            Dictionary of task outputs
        """
        # In planning mode, the planner agent creates a dynamic plan
        # We need to provide the initial objective and context
        
        # Create an objective from the workflow description
        objective = self.workflow.description
        
        # Add workflow inputs to the context
        context = "## Inputs\n"
        for key, value in inputs.items():
            context += f"- {key}: {value}\n"
        
        # Add workflow outputs to the context
        context += "\n## Expected Outputs\n"
        for output in self.workflow.outputs:
            task_id = output["task_id"]
            output_name = output["output"]
            task_name = self.tasks[task_id]["name"]
            context += f"- {output_name} (from {task_name})\n"
        
        # Run the crew with planning
        result = self.crew.kickoff(
            inputs={"objective": objective, "context": context}
        )
        
        # Process results into task outputs
        task_outputs = {}
        for task_id, task_info in self.tasks.items():
            task = task_info["task"]
            if task.output:
                task_outputs[task_id] = task.output
        
        return task_outputs
    
    def _prepare_task_inputs(
        self,
        task_id: str,
        global_inputs: Dict[str, Any],
        task_outputs: Dict[str, TaskOutput]
    ) -> Dict[str, Any]:
        """
        Prepare inputs for a task based on global inputs and outputs from dependencies.
        
        Args:
            task_id: ID of the task
            global_inputs: Global inputs for the crew
            task_outputs: Outputs from previous tasks
            
        Returns:
            Dictionary of inputs for the task
        """
        task_inputs = {}
        
        # Add global inputs
        task_inputs.update(global_inputs)
        
        # Add outputs from dependencies
        dependencies = self.task_dependencies.get(task_id, {}).get("depends_on", [])
        for dep_id in dependencies:
            if dep_id in task_outputs:
                # Add output as input with task name as prefix
                dep_name = self.tasks[dep_id]["name"]
                dep_output = task_outputs[dep_id]
                
                # If output is a dictionary, add each key
                if isinstance(dep_output.raw_output, dict):
                    for key, value in dep_output.raw_output.items():
                        task_inputs[f"{dep_name}_{key}"] = value
                else:
                    # Otherwise add the whole output
                    task_inputs[dep_name] = dep_output.raw_output
        
        return task_inputs
    
    def _update_task_context(
        self,
        task_config: TaskConfig,
        current_context: str,
        inputs: Dict[str, Any]
    ) -> str:
        """
        Update task context with input values.
        
        Args:
            task_config: Task configuration
            current_context: Current task context
            inputs: Input values
            
        Returns:
            Updated task context
        """
        # Start with the current context
        context = current_context
        
        # Add input values section
        if inputs:
            context += "\n\n## Input Values\n"
            for key, value in inputs.items():
                # Format value based on type
                if isinstance(value, dict):
                    value_str = "\n```json\n" + str(value) + "\n```"
                elif isinstance(value, list):
                    value_str = "\n" + "\n".join(f"- {item}" for item in value)
                else:
                    value_str = str(value)
                
                context += f"- {key}: {value_str}\n"
        
        return context
    
    def _execute_task_with_hitl(self, task_id: str, task: Task) -> TaskOutput:
        """
        Execute a task with Human-in-the-Loop integration if configured.
        
        Args:
            task_id: ID of the task
            task: CrewAI Task object
            
        Returns:
            Task output
        """
        # Get task and agent info
        task_info = self.tasks[task_id]
        task_config = task_info["config"]
        agent_id = task_config.assigned_to
        agent_info = self.agents[agent_id]
        
        # Check if HITL is enabled for this task
        hitl_enabled = False
        hitl_config = task_config.hitl
        
        if hitl_config and self.hitl_callback:
            hitl_enabled = hitl_config.get("enabled", False)
        
        # Execute task
        logger.info(f"Executing task: {task_id} ({task_info['name']})")
        result = task.execute()
        
        # Handle HITL if enabled
        if hitl_enabled and self.hitl_callback:
            review_threshold = hitl_config.get("review_threshold", 0.0)
            review_type = hitl_config.get("review_type", "approval")
            
            # Determine confidence (default to 1.0 if not specified)
            confidence = 1.0
            if isinstance(result.raw_output, dict) and "confidence" in result.raw_output:
                confidence = float(result.raw_output["confidence"])
            
            # Check if review is needed
            if confidence >= review_threshold:
                logger.info(f"Task {task_id} requires HITL review (confidence: {confidence})")
                
                # Create HITL review request
                review = HITLReview(
                    crew_id=self.domain,
                    task_id=task_id,
                    agent_id=agent_id,
                    timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    task_name=task_info["name"],
                    task_output=result.raw_output,
                    confidence=confidence,
                    review_type=review_type,
                )
                
                # Call HITL callback and wait for response
                try:
                    review_result = self.hitl_callback(review)
                    
                    # Process review result
                    if review_result and review_result.get("status") == "approved":
                        logger.info(f"Task {task_id} approved by human reviewer")
                        
                        # Use edited output if provided
                        if review_result.get("edited_output"):
                            result = TaskOutput(
                                raw_output=review_result["edited_output"],
                                output=str(review_result["edited_output"]),
                            )
                            logger.info(f"Task {task_id} output edited by human reviewer")
                    
                    elif review_result and review_result.get("status") == "rejected":
                        logger.warning(f"Task {task_id} rejected by human reviewer")
                        raise ValueError(f"Task {task_id} rejected by human reviewer: {review_result.get('feedback')}")
                
                except Exception as e:
                    logger.error(f"Error in HITL review for task {task_id}: {e}")
                    # Continue with original result if HITL fails
        
        return result
    
    def _process_results(self, task_outputs: Dict[str, TaskOutput]) -> Dict[str, Any]:
        """
        Process task outputs into final crew outputs based on workflow configuration.
        
        Args:
            task_outputs: Dictionary of task outputs
            
        Returns:
            Dictionary of crew outputs
        """
        outputs = {}
        
        # Extract outputs based on workflow configuration
        for output_config in self.workflow.outputs:
            task_id = output_config["task_id"]
            output_name = output_config["output"]
            
            if task_id in task_outputs:
                task_output = task_outputs[task_id]
                
                # Extract specific output field if raw_output is a dictionary
                if isinstance(task_output.raw_output, dict) and output_name in task_output.raw_output:
                    outputs[output_name] = task_output.raw_output[output_name]
                else:
                    # Otherwise use the whole output
                    outputs[output_name] = task_output.raw_output
            else:
                logger.warning(f"Task {task_id} output not found for {output_name}")
                outputs[output_name] = None
        
        return outputs
    
    def _setup_hitl(self) -> None:
        """Set up Human-in-the-Loop integration if enabled in configuration."""
        if not self.config.hitl or not self.config.hitl.get("enabled", False):
            return
        
        # Check if HITL callback is provided
        if not self.hitl_callback:
            logger.warning("HITL is enabled in configuration but no callback provided")
            return
        
        logger.info("HITL integration enabled")
        
        # TODO: Set up HITL notification channels if needed
    
    def _track_execution_metrics(
        self,
        duration_ms: float,
        success: bool,
        error: Optional[str] = None
    ) -> None:
        """
        Track crew execution metrics.
        
        Args:
            duration_ms: Execution duration in milliseconds
            success: Whether execution was successful
            error: Error message if execution failed
        """
        # Track with metrics
        AgentMetrics.track_execution(
            agent_type="crew",
            task=self.workflow_name,
            func=lambda: None,
            environment=os.environ.get("ENVIRONMENT", "development"),
            version=os.environ.get("APP_VERSION", "1.8.0-beta"),
        )()
        
        # Track execution duration
        from backend.core.metrics import agent_execution_duration_seconds
        agent_execution_duration_seconds.labels(
            agent_type="crew",
            task=self.workflow_name,
            environment=os.environ.get("ENVIRONMENT", "development"),
            version=os.environ.get("APP_VERSION", "1.8.0-beta"),
            status="success" if success else "error",
        ).observe(duration_ms / 1000)  # Convert to seconds
        
        # Publish event
        publish_event(
            event_type="analytics.agent_execution",
            data={
                "agent_type": "crew",
                "task": self.workflow_name,
                "duration_ms": duration_ms,
                "success": success,
                "error": error,
                "priority": EventPriority.HIGH.value if not success else EventPriority.NORMAL.value,
            },
        )


def load_crew_config(config_path: Union[str, Path]) -> CrewConfig:
    """
    Load a crew configuration from a YAML file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Parsed configuration
        
    Raises:
        FileNotFoundError: If the configuration file doesn't exist
        ValueError: If the configuration is invalid
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)
        
        return CrewConfig(**config_dict)
    
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in configuration: {e}")
    except Exception as e:
        raise ValueError(f"Error loading configuration: {e}")


def discover_crew_configs(base_dir: Union[str, Path] = "backend/agents/crews") -> List[Path]:
    """
    Discover crew configuration files in the specified directory.
    
    Args:
        base_dir: Base directory to search in
        
    Returns:
        List of paths to crew configuration files
    """
    base_dir = Path(base_dir)
    if not base_dir.exists():
        logger.warning(f"Crew configuration directory not found: {base_dir}")
        return []
    
    configs = []
    
    # Search for crew.yaml files in subdirectories
    for domain_dir in base_dir.iterdir():
        if domain_dir.is_dir():
            crew_config = domain_dir / "crew.yaml"
            if crew_config.exists():
                configs.append(crew_config)
    
    return configs


def create_crew(
    domain: str,
    workflow: Optional[str] = None,
    mode: Optional[CrewMode] = None,
    tools: Optional[Dict[str, AbstractApiTool]] = None,
    hitl_callback: Optional[callable] = None,
    environment_vars: Optional[Dict[str, Any]] = None,
) -> CustomCrew:
    """
    Create a crew for the specified domain.
    
    Args:
        domain: Domain name (subdirectory in crews)
        workflow: Name of the workflow to execute (default: first workflow in config)
        mode: Execution mode (default: from CREW_MODE env var or config)
        tools: Dictionary of tools to use (default: auto-discover)
        hitl_callback: Callback function for HITL integration
        environment_vars: Additional environment variables to override config
        
    Returns:
        CustomCrew instance
        
    Raises:
        ValueError: If the domain or workflow is not found
    """
    # Find crew configuration
    config_path = Path(f"backend/agents/crews/{domain}/crew.yaml")
    if not config_path.exists():
        raise ValueError(f"Crew configuration not found for domain: {domain}")
    
    # Create crew
    crew = CustomCrew(
        config_path=config_path,
        workflow_name=workflow,
        mode=mode,
        tools=tools,
        hitl_callback=hitl_callback,
        environment_vars=environment_vars,
    )
    
    return crew
