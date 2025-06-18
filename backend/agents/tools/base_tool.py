"""
BaseTool - A foundational class for all custom CrewAI tools.

This module defines an abstract BaseTool class that provides common functionality
for CrewAI tools, including initialization, metrics recording, and an abstract
method for tool execution. All custom tools should inherit from this class
to ensure consistency and proper integration with the system's observability.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from crewai_tools import BaseTool as CrewAIBaseTool

# Import metrics recording functions
from backend.core.metrics import record_tool_usage, record_tool_error

logger = logging.getLogger(__name__)


class BaseTool(CrewAIBaseTool, ABC):
    """
    Abstract base class for all custom CrewAI tools in the Analyst Augmentation Agent.

    This class extends CrewAI's BaseTool and provides common functionalities such as:
    - Initialization with a name and description.
    - Integration with application-wide metrics for tool usage and errors.
    - An abstract `_execute` method that subclasses must implement for their specific logic.
      This method is designed to be asynchronous.
    - Basic error handling and logging.
    """

    name: str
    description: str

    def __init__(self, **kwargs):
        """
        Initializes the BaseTool.

        Args:
            **kwargs: Arbitrary keyword arguments passed to the CrewAI BaseTool.
        """
        super().__init__(**kwargs)
        # Ensure name and description are set, either from kwargs or class attributes
        if not hasattr(self, 'name') or not self.name:
            raise ValueError("Tool must have a 'name' attribute.")
        if not hasattr(self, 'description') or not self.description:
            raise ValueError("Tool must have a 'description' attribute.")

    async def _run(self, **kwargs) -> Any:
        """
        The main execution method for the tool.

        This method wraps the abstract `_execute` method, adding metrics recording
        and error handling. It is designed to be called by the CrewAI framework.

        Args:
            **kwargs: Arguments specific to the tool's operation.

        Returns:
            Any: The result of the tool's execution.

        Raises:
            Exception: Any exception raised by the `_execute` method.
        """
        try:
            logger.info(f"Executing tool: {self.name} with args: {kwargs}")
            record_tool_usage(self.name)
            result = await self._execute(**kwargs)
            logger.info(f"Tool {self.name} executed successfully.")
            return result
        except Exception as e:
            error_message = f"Error executing tool {self.name}: {e}"
            logger.error(error_message, exc_info=True)
            record_tool_error(self.name, str(e))
            raise  # Re-raise the exception after logging and recording

    @abstractmethod
    async def _execute(self, **kwargs) -> Any:
        """
        Abstract method for the tool's core logic.

        Subclasses must implement this method to define the specific functionality
        of their tool. This method is expected to be asynchronous.

        Args:
            **kwargs: Arguments specific to the tool's operation.

        Returns:
            Any: The result of the tool's operation.
        """
        pass
