"""
Custom Crew implementation with enhanced context sharing for CodeGenTool results.

This module extends the CrewAI Crew class to provide improved context sharing
between tasks, particularly for CodeGenTool execution results. It ensures that
outputs from code execution are properly merged into the crew context and
made available to subsequent tasks.

The implementation also emits events for real-time progress tracking via
WebSockets and Server-Sent Events (SSE).
"""

import logging
import json
import uuid
import asyncio
from typing import Any, Dict, List, Optional, Union, cast

from crewai import Crew, Agent, Task
from crewai.task import TaskOutput

from backend.core.events import EventType, emit_event

logger = logging.getLogger(__name__)


class CustomCrew(Crew):
    """
    Extended Crew implementation with enhanced context sharing and event emissions.
    
    This class extends the CrewAI Crew to intercept task execution results
    and ensure that CodeGenTool outputs are properly merged into the shared
    context for subsequent tasks. It also emits events for real-time progress
    tracking.
    
    Attributes:
        shared_context: Dictionary containing shared context across tasks
        crew_id: Unique identifier for this crew instance
        task_ids: Dictionary mapping task descriptions to task IDs
        total_tasks: Total number of tasks in this crew
        completed_tasks: Number of completed tasks
    """
    
    def __init__(self, *args, **kwargs):
        """
        Initialize the CustomCrew.
        
        Args:
            *args: Arguments to pass to the parent Crew constructor
            **kwargs: Keyword arguments to pass to the parent Crew constructor
        """
        super().__init__(*args, **kwargs)
        self.shared_context: Dict[str, Any] = {}
        self.crew_id = str(uuid.uuid4())
        self.task_ids = {}
        self.total_tasks = len(self.tasks) if hasattr(self, "tasks") else 0
        self.completed_tasks = 0
        logger.info(f"CustomCrew initialized with ID {self.crew_id} and enhanced context sharing")
    
    async def _process_task(self, task: Task, inputs: Dict[str, Any]) -> TaskOutput:
        """
        Process a task with enhanced context sharing and event emissions.
        
        This method overrides the parent _process_task to intercept task execution,
        merge CodeGenTool results into the shared context, and emit progress events.
        
        Args:
            task: Task to process
            inputs: Input context for the task
            
        Returns:
            TaskOutput from task execution
        """
        # Generate or retrieve task ID
        task_id = self.task_ids.get(task.description)
        if not task_id:
            task_id = str(uuid.uuid4())
            self.task_ids[task.description] = task_id
        
        # Calculate progress
        progress = (self.completed_tasks / self.total_tasks) * 100 if self.total_tasks > 0 else 0
        
        try:
            # Emit AGENT_STARTED event
            try:
                asyncio.create_task(emit_event(
                    EventType.AGENT_STARTED,
                    task_id=task_id,
                    crew_id=self.crew_id,
                    agent_id=task.agent.name if task.agent else "unknown",
                    message=f"Starting task: {task.description[:100]}...",
                    progress=progress,
                    data={"task_description": task.description}
                ))
            except Exception as e:
                logger.error(f"Error emitting AGENT_STARTED event: {e}")
            
            # Merge shared context into inputs
            enhanced_inputs = {**inputs, **self.shared_context}
            logger.debug(f"Enhanced inputs for task '{task.description[:50]}...' with shared context")
            
            # Check if task uses CodeGenTool
            has_codegen_tool = False
            if task.agent and hasattr(task.agent, "tools") and task.agent.tools:
                for tool in task.agent.tools:
                    if hasattr(tool, "__class__") and tool.__class__.__name__ == "CodeGenTool":
                        has_codegen_tool = True
                        try:
                            # Emit TOOL_STARTED event for CodeGenTool
                            asyncio.create_task(emit_event(
                                EventType.TOOL_STARTED,
                                task_id=task_id,
                                crew_id=self.crew_id,
                                agent_id=task.agent.name,
                                tool_id="CodeGenTool",
                                message=f"Starting CodeGenTool for task: {task.description[:50]}...",
                                progress=progress
                            ))
                        except Exception as e:
                            logger.error(f"Error emitting TOOL_STARTED event: {e}")
            
            # Emit AGENT_PROGRESS event at 50%
            try:
                asyncio.create_task(emit_event(
                    EventType.AGENT_PROGRESS,
                    task_id=task_id,
                    crew_id=self.crew_id,
                    agent_id=task.agent.name if task.agent else "unknown",
                    message=f"Processing task: {task.description[:100]}...",
                    progress=progress + (100 / self.total_tasks / 2) if self.total_tasks > 0 else 50
                ))
            except Exception as e:
                logger.error(f"Error emitting AGENT_PROGRESS event: {e}")
            
            # Execute the task with enhanced inputs
            result = await super()._process_task(task, enhanced_inputs)
            
            # Extract CodeGenTool results if present
            self._extract_codegen_results(task, result)
            
            # Increment completed tasks
            self.completed_tasks += 1
            
            # Calculate new progress
            progress = (self.completed_tasks / self.total_tasks) * 100 if self.total_tasks > 0 else 100
            
            # Emit AGENT_COMPLETED event
            try:
                asyncio.create_task(emit_event(
                    EventType.AGENT_COMPLETED,
                    task_id=task_id,
                    crew_id=self.crew_id,
                    agent_id=task.agent.name if task.agent else "unknown",
                    message=f"Completed task: {task.description[:100]}...",
                    progress=progress,
                    data={"result_summary": str(result)[:500] if result else "No result"}
                ))
            except Exception as e:
                logger.error(f"Error emitting AGENT_COMPLETED event: {e}")
            
            # Emit TOOL_COMPLETED event for CodeGenTool if used
            if has_codegen_tool and "codegen" in self.shared_context:
                try:
                    asyncio.create_task(emit_event(
                        EventType.TOOL_COMPLETED,
                        task_id=task_id,
                        crew_id=self.crew_id,
                        agent_id=task.agent.name if task.agent else "unknown",
                        tool_id="CodeGenTool",
                        message="CodeGenTool execution completed",
                        progress=progress,
                        data={
                            "has_visualizations": "visualizations" in self.shared_context,
                            "visualization_count": len(self.shared_context.get("visualizations", []))
                        }
                    ))
                except Exception as e:
                    logger.error(f"Error emitting TOOL_COMPLETED event: {e}")
            
            return result
            
        except Exception as e:
            # Emit AGENT_FAILED event
            try:
                asyncio.create_task(emit_event(
                    EventType.AGENT_FAILED,
                    task_id=task_id,
                    crew_id=self.crew_id,
                    agent_id=task.agent.name if task.agent else "unknown",
                    message=f"Task failed: {str(e)}",
                    progress=progress,
                    data={"error": str(e)}
                ))
            except Exception as emit_error:
                logger.error(f"Error emitting AGENT_FAILED event: {emit_error}")
            
            logger.error(f"Error processing task with enhanced context: {e}", exc_info=True)
            # Fall back to parent implementation
            return await super()._process_task(task, inputs)
    
    def _extract_codegen_results(self, task: Task, result: TaskOutput) -> None:
        """
        Extract CodeGenTool results from task output and merge into shared context.
        
        Args:
            task: Task that was executed
            result: Output from task execution
        """
        try:
            # Check if task has tools
            if not task.agent or not hasattr(task.agent, "tools") or not task.agent.tools:
                return
            
            # Check if any of the tools is CodeGenTool
            code_gen_tools = [
                tool for tool in task.agent.tools 
                if hasattr(tool, "__class__") and 
                tool.__class__.__name__ == "CodeGenTool"
            ]
            
            if not code_gen_tools:
                return
            
            # Try to extract CodeGenTool results from the output
            result_str = str(result)
            
            # Look for code execution results in various formats
            codegen_result = None
            
            # 1. Look for JSON-formatted results
            try:
                # Try to find JSON in the result
                import re
                json_match = re.search(r'```json\n(.*?)\n```', result_str, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group(1))
                    if isinstance(data, dict) and ("result" in data or "execution" in data):
                        codegen_result = data
                
                # Look for tool execution results in the text
                tool_result_match = re.search(r'Tool Result:(.*?)(?:\n\n|$)', result_str, re.DOTALL)
                if tool_result_match and not codegen_result:
                    try:
                        data = json.loads(tool_result_match.group(1).strip())
                        if isinstance(data, dict):
                            codegen_result = data
                    except:
                        pass
            except Exception as e:
                logger.debug(f"Failed to extract JSON from result: {e}")
            
            # 2. If agent output doesn't contain structured results, 
            # check if the tool was executed directly and we can access its last result
            if not codegen_result:
                for tool in code_gen_tools:
                    if hasattr(tool, "_last_result") and tool._last_result:
                        codegen_result = tool._last_result
                        break
            
            # If we found results, merge them into shared context
            if codegen_result:
                logger.info(f"Found CodeGenTool results in task '{task.description[:50]}...'")
                self.shared_context["codegen"] = codegen_result
                
                # If there's a parsed result, make it directly accessible
                if "result" in codegen_result and codegen_result["result"]:
                    self.shared_context["code_result"] = codegen_result["result"]
                    
                # If there are visualizations, make them accessible
                if "visualizations" in codegen_result and codegen_result["visualizations"]:
                    self.shared_context["visualizations"] = codegen_result["visualizations"]
                    
                logger.debug(f"Updated shared context with CodeGenTool results")
            
        except Exception as e:
            logger.error(f"Error extracting CodeGenTool results: {e}", exc_info=True)
    
    def kickoff(self, inputs: Optional[Dict[str, Any]] = None) -> Any:
        """
        Start the crew execution with the given inputs.
        
        This method overrides the parent kickoff to initialize the shared context
        with the inputs and emit CREW_STARTED and CREW_COMPLETED events.
        
        Args:
            inputs: Initial inputs for the crew
            
        Returns:
            Result of crew execution
        """
        # Initialize shared context with inputs
        if inputs:
            self.shared_context.update(inputs)
            logger.info("Initialized shared context with input values")
        
        # Emit CREW_STARTED event
        try:
            # We can't use await in a synchronous method, so we create a task
            asyncio.create_task(emit_event(
                EventType.CREW_STARTED,
                crew_id=self.crew_id,
                message=f"Starting crew execution with {self.total_tasks} tasks",
                progress=0,
                data={
                    "agent_count": len(self.agents) if hasattr(self, "agents") else 0,
                    "task_count": self.total_tasks,
                    "crew_name": self.name if hasattr(self, "name") else "Unknown"
                }
            ))
        except Exception as e:
            logger.error(f"Error emitting CREW_STARTED event: {e}")
        
        try:
            # Call parent kickoff
            result = super().kickoff(inputs)
            
            # Emit CREW_COMPLETED event
            try:
                asyncio.create_task(emit_event(
                    EventType.CREW_COMPLETED,
                    crew_id=self.crew_id,
                    message="Crew execution completed successfully",
                    progress=100,
                    data={
                        "result_summary": str(result)[:500] if result else "No result",
                        "task_count": self.total_tasks,
                        "completed_tasks": self.completed_tasks
                    }
                ))
            except Exception as e:
                logger.error(f"Error emitting CREW_COMPLETED event: {e}")
            
            return result
            
        except Exception as e:
            # Emit CREW_FAILED event
            try:
                asyncio.create_task(emit_event(
                    EventType.CREW_FAILED,
                    crew_id=self.crew_id,
                    message=f"Crew execution failed: {str(e)}",
                    progress=100,
                    data={"error": str(e)}
                ))
            except Exception as emit_error:
                logger.error(f"Error emitting CREW_FAILED event: {emit_error}")
            
            # Re-raise the original exception
            raise
