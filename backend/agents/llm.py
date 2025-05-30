"""
Custom LLM provider for CrewAI that integrates with Google's Gemini API.

This module implements a GeminiLLMProvider that bridges the existing GeminiClient
with CrewAI's BaseLLM interface, enabling Gemini models to be used as the
reasoning engine for CrewAI agents.
"""

import json
import logging
import asyncio
from typing import Any, Dict, List, Optional, Union, Tuple, Callable
import time
import backoff

from crewai.llm import BaseLLM
from crewai.utilities.function_calling import FunctionCall, FunctionConfig
from google.generativeai.types import FunctionDeclaration, Tool

from backend.integrations.gemini_client import GeminiClient
from backend.core.metrics import track_llm_usage

logger = logging.getLogger(__name__)

# Gemini pricing constants (per 1M tokens)
GEMINI_PRICING = {
    "gemini-1.5-flash": {
        "input": 0.35 / 1_000_000,  # $0.35 per 1M input tokens
        "output": 0.70 / 1_000_000,  # $0.70 per 1M output tokens
    },
    "gemini-1.5-pro": {
        "input": 3.50 / 1_000_000,  # $3.50 per 1M input tokens
        "output": 10.50 / 1_000_000,  # $10.50 per 1M output tokens
    },
    # Default pricing for other models
    "default": {
        "input": 3.50 / 1_000_000,  # Default to Pro pricing
        "output": 10.50 / 1_000_000,
    }
}


class GeminiLLMProvider(BaseLLM):
    """
    CrewAI-compatible LLM provider for Google's Gemini API.
    
    This provider integrates the existing GeminiClient with CrewAI's
    BaseLLM interface, enabling Gemini models to be used as the reasoning
    engine for CrewAI agents.
    """
    
    def __init__(
        self,
        model: str = "gemini-1.5-pro",
        temperature: float = 0.1,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        max_tokens: Optional[int] = None,
        client: Optional[GeminiClient] = None,
    ):
        """
        Initialize the GeminiLLMProvider.
        
        Args:
            model: The Gemini model to use
            temperature: The sampling temperature (0.0 to 1.0)
            max_retries: Maximum number of retries for failed API calls
            retry_delay: Delay between retries (with exponential backoff)
            max_tokens: Maximum number of tokens to generate
            client: Optional pre-configured GeminiClient instance
        """
        super().__init__()
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.max_tokens = max_tokens
        
        # Use provided client or create a new one
        self.client = client or GeminiClient()
        
        # Track rate limiting
        self.last_call_time = 0
        self.min_call_interval = 0.1  # 100ms minimum between calls
        
        # Get pricing for this model
        model_base = model.split("-")[0] + "-" + model.split("-")[1]
        self.pricing = GEMINI_PRICING.get(model, GEMINI_PRICING["default"])
        
        logger.info(f"Initialized GeminiLLMProvider with model: {model}")
    
    def supports_function_calling(self) -> bool:
        """
        Indicate whether this LLM supports function calling.
        
        Returns:
            True if the LLM supports function calling, False otherwise
        """
        # Gemini 1.5 Pro supports function calling
        return self.model.startswith("gemini-1.5")
    
    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """
        Calculate the cost of a Gemini API call.
        
        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            
        Returns:
            Cost in USD
        """
        input_cost = input_tokens * self.pricing["input"]
        output_cost = output_tokens * self.pricing["output"]
        return input_cost + output_cost
    
    @backoff.on_exception(
        backoff.expo,
        (Exception),
        max_tries=3,
        giveup=lambda e: "rate limit" not in str(e).lower(),
    )
    async def call(
        self,
        messages: List[Dict[str, Any]],
        functions: Optional[List[FunctionConfig]] = None,
        **kwargs: Any,
    ) -> Union[str, Tuple[str, Optional[FunctionCall]]]:
        """
        Call the Gemini API with the provided messages and functions.
        
        Args:
            messages: List of messages in the CrewAI format
            functions: Optional list of function configurations
            **kwargs: Additional keyword arguments
            
        Returns:
            Either a string response or a tuple of (response, function_call)
        """
        # Rate limiting - ensure minimum interval between calls
        current_time = time.time()
        time_since_last_call = current_time - self.last_call_time
        if time_since_last_call < self.min_call_interval:
            await asyncio.sleep(self.min_call_interval - time_since_last_call)
        
        # Convert CrewAI messages to Gemini format
        prompt = self._format_messages(messages)
        
        # Convert CrewAI functions to Gemini tools if provided
        tools = self._convert_functions_to_tools(functions) if functions else None
        
        try:
            # If tools are provided and function calling is supported
            if tools and self.supports_function_calling():
                # Call Gemini with tools
                response = await self.client.generate_text_with_tools(
                    prompt=prompt,
                    tools=tools,
                    temperature=self.temperature,
                    max_output_tokens=self.max_tokens,
                )
                
                # Track token usage and cost
                input_tokens = response.usage.prompt_tokens if hasattr(response, "usage") else len(prompt) // 4
                output_tokens = response.usage.completion_tokens if hasattr(response, "usage") else len(response.text) // 4
                cost_usd = self._calculate_cost(input_tokens, output_tokens)
                
                # Record metrics
                track_llm_usage(
                    model=self.model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cost_usd=cost_usd,
                    success=True
                )
                
                # Parse tool calls if present
                if hasattr(response, "tool_calls") and response.tool_calls:
                    function_call = self._parse_tool_call(response.tool_calls[0], functions)
                    return response.text, function_call
                
                return response.text, None
            else:
                # Standard text generation without tools
                response = await self.client.generate_text(
                    prompt=prompt,
                    temperature=self.temperature,
                    max_output_tokens=self.max_tokens,
                )
                
                # Track token usage and cost
                input_tokens = response.usage.prompt_tokens if hasattr(response, "usage") else len(prompt) // 4
                output_tokens = response.usage.completion_tokens if hasattr(response, "usage") else len(response.text) // 4
                cost_usd = self._calculate_cost(input_tokens, output_tokens)
                
                # Record metrics
                track_llm_usage(
                    model=self.model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cost_usd=cost_usd,
                    success=True
                )
                
                return response.text
        
        except Exception as e:
            # Estimate token usage for failed requests
            estimated_input_tokens = len(prompt) // 4
            
            # Record metrics for failed request
            track_llm_usage(
                model=self.model,
                input_tokens=estimated_input_tokens,
                output_tokens=0,
                cost_usd=self._calculate_cost(estimated_input_tokens, 0),
                success=False
            )
            
            # Handle specific error types
            if "rate limit" in str(e).lower():
                logger.warning(f"Rate limit exceeded, retrying after delay: {e}")
                # This will be caught by the backoff decorator
                raise
            elif "invalid request" in str(e).lower():
                logger.error(f"Invalid request to Gemini API: {e}")
                return "I encountered an error processing your request. Please try again with a clearer instruction."
            else:
                logger.exception(f"Error calling Gemini API: {e}")
                return "I apologize, but I encountered an unexpected error. Please try again."
        finally:
            # Update last call time
            self.last_call_time = time.time()
    
    def _format_messages(self, messages: List[Dict[str, Any]]) -> str:
        """
        Format CrewAI messages into a prompt string for Gemini.
        
        Args:
            messages: List of messages in the CrewAI format
            
        Returns:
            Formatted prompt string
        """
        formatted_prompt = ""
        
        for message in messages:
            role = message.get("role", "").lower()
            content = message.get("content", "")
            
            if role == "system":
                formatted_prompt += f"System: {content}\n\n"
            elif role == "user":
                formatted_prompt += f"Human: {content}\n\n"
            elif role == "assistant":
                formatted_prompt += f"Assistant: {content}\n\n"
            elif role == "function":
                # Handle function responses
                name = message.get("name", "unknown_function")
                formatted_prompt += f"Function ({name}): {content}\n\n"
            else:
                # Default handling for unknown roles
                formatted_prompt += f"{role.capitalize()}: {content}\n\n"
        
        return formatted_prompt.strip()
    
    def _convert_functions_to_tools(
        self, functions: List[FunctionConfig]
    ) -> List[Tool]:
        """
        Convert CrewAI function configurations to Gemini tools.
        
        Args:
            functions: List of CrewAI function configurations
            
        Returns:
            List of Gemini Tool objects
        """
        tools = []
        
        for func in functions:
            # Create parameter schema
            parameters = {
                "type": "object",
                "properties": {},
                "required": [],
            }
            
            # Add parameters from function config
            if func.parameters:
                for param_name, param_info in func.parameters.items():
                    parameters["properties"][param_name] = {
                        "type": param_info.get("type", "string"),
                        "description": param_info.get("description", ""),
                    }
                    
                    if param_info.get("required", False):
                        parameters["required"].append(param_name)
            
            # Create function declaration
            function_declaration = FunctionDeclaration(
                name=func.name,
                description=func.description,
                parameters=parameters,
            )
            
            # Create tool with the function
            tool = Tool(function_declarations=[function_declaration])
            tools.append(tool)
        
        return tools
    
    def _parse_tool_call(
        self, tool_call: Any, functions: List[FunctionConfig]
    ) -> FunctionCall:
        """
        Parse a Gemini tool call into a CrewAI FunctionCall.
        
        Args:
            tool_call: The tool call from Gemini
            functions: List of available functions
            
        Returns:
            CrewAI FunctionCall object
        """
        # Extract function name and arguments
        function_name = tool_call.function_name
        function_args = json.loads(tool_call.function_args)
        
        # Find matching function config
        function_config = next(
            (f for f in functions if f.name == function_name), None
        )
        
        if not function_config:
            logger.warning(f"Function call to unknown function: {function_name}")
        
        # Create and return FunctionCall
        return FunctionCall(
            name=function_name,
            arguments=function_args,
        )
    
    # Add a method to handle function call responses
    async def parse_tool_call_responses(
        self, 
        messages: List[Dict[str, Any]], 
        function_results: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Parse function call results and add them to the message history.
        
        Args:
            messages: The current message history
            function_results: Results from function calls
            
        Returns:
            Updated message history with function results
        """
        # Create a new message for the function response
        function_message = {
            "role": "function",
            "name": function_results.get("name", "unknown_function"),
            "content": json.dumps(function_results.get("content", {}))
        }
        
        # Add to messages
        return messages + [function_message]
