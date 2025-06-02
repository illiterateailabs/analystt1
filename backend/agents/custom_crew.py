"""
Custom Crew implementation with enhanced context sharing for CodeGenTool results.

This module extends the CrewAI Crew class to provide improved context sharing
between tasks, particularly for CodeGenTool execution results. It ensures that
outputs from code execution are properly merged into the crew context and
made available to subsequent tasks.
"""

import logging
import json
from typing import Any, Dict, List, Optional, Union, cast

from crewai import Crew, Agent, Task
from crewai.task import TaskOutput

logger = logging.getLogger(__name__)


class CustomCrew(Crew):
    """
    Extended Crew implementation with enhanced context sharing.
    
    This class extends the CrewAI Crew to intercept task execution results
    and ensure that CodeGenTool outputs are properly merged into the shared
    context for subsequent tasks.
    
    Attributes:
        shared_context: Dictionary containing shared context across tasks
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
        logger.info("CustomCrew initialized with enhanced context sharing")
    
    async def _process_task(self, task: Task, inputs: Dict[str, Any]) -> TaskOutput:
        """
        Process a task with enhanced context sharing.
        
        This method overrides the parent _process_task to intercept task execution
        and merge CodeGenTool results into the shared context.
        
        Args:
            task: Task to process
            inputs: Input context for the task
            
        Returns:
            TaskOutput from task execution
        """
        try:
            # Merge shared context into inputs
            enhanced_inputs = {**inputs, **self.shared_context}
            logger.debug(f"Enhanced inputs for task '{task.description[:50]}...' with shared context")
            
            # Execute the task with enhanced inputs
            result = await super()._process_task(task, enhanced_inputs)
            
            # Extract CodeGenTool results if present
            self._extract_codegen_results(task, result)
            
            return result
            
        except Exception as e:
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
        with the inputs.
        
        Args:
            inputs: Initial inputs for the crew
            
        Returns:
            Result of crew execution
        """
        # Initialize shared context with inputs
        if inputs:
            self.shared_context.update(inputs)
            logger.info("Initialized shared context with input values")
        
        # Call parent kickoff
        return super().kickoff(inputs)
