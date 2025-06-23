"""
Google Gemini LLM Client

This module provides a client for interacting with Google's Gemini API for
text generation, embeddings, and image analysis. It handles authentication,
rate limiting, retries, and cost tracking for all Gemini endpoints.

The client is configured via the central provider registry and automatically tracks
token usage and costs for budget monitoring and back-pressure control.
"""

import asyncio
import base64
import logging
import os
import time
from typing import Any, Dict, List, Optional, Union, Tuple

import aiohttp
from aiohttp import ClientResponseError, ClientSession

from backend.core.metrics import ApiMetrics, LlmMetrics
from backend.providers import get_provider

# Configure module logger
logger = logging.getLogger(__name__)

class GeminiApiError(Exception):
    """Exception raised for Gemini API errors."""
    def __init__(self, message: str, status_code: Optional[int] = None, response_text: Optional[str] = None):
        self.status_code = status_code
        self.response_text = response_text
        super().__init__(message)

class GeminiClient:
    """
    Client for the Google Gemini API.
    
    This client handles authentication, rate limiting, retries, and cost tracking
    for all Gemini endpoints. It is configured via the central provider registry.
    """
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """
        Initialize the Gemini API client.
        
        Args:
            api_key: Optional API key. If not provided, it will be loaded from the provider registry.
            model: Optional default model to use. If not provided, it will use the default from provider config.
        """
        self._provider_config = get_provider("gemini")
        if not self._provider_config:
            raise ValueError("Gemini provider configuration not found in registry")
        
        # Get API key from parameter, provider config, or environment variable
        self._api_key = api_key
        if not self._api_key:
            api_key_env_var = self._provider_config.get("auth", {}).get("api_key_env_var")
            if api_key_env_var:
                self._api_key = os.getenv(api_key_env_var)
        
        if not self._api_key:
            raise ValueError("Gemini API key not provided and not found in environment")
        
        # Get base URL from provider config or use default
        self._base_url = self._provider_config.get("connection_uri", "https://generativelanguage.googleapis.com/v1beta")
        
        # Get default model from parameter or provider config
        self._default_model = model or self._provider_config.get("default_model", "gemini-1.5-pro-latest")
        self._default_embedding_model = self._provider_config.get("default_embedding_model", "text-embedding-004")
        
        # Get retry configuration
        retry_config = self._provider_config.get("retry_policy", {})
        self._max_retries = retry_config.get("attempts", 3)
        self._backoff_factor = retry_config.get("backoff_factor", 0.5)
        
        # Get rate limits
        self._rate_limits = self._provider_config.get("rate_limits", {})
        self._requests_per_minute = self._rate_limits.get("requests_per_minute", 60)
        
        # Get cost rules
        self._cost_rules = self._provider_config.get("cost_rules", {})
        self._default_cost = self._cost_rules.get("default_cost_per_request", 0.0001)
        self._param_multipliers = self._cost_rules.get("param_multipliers", {})
        
        # Initialize session
        self._session = None
        
        logger.info(f"Gemini client initialized with base URL: {self._base_url}")
    
    async def _ensure_session(self) -> ClientSession:
        """
        Ensure that an aiohttp ClientSession exists.
        
        Returns:
            An aiohttp ClientSession
        """
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "User-Agent": "AnalystDroidOne/1.0",
                }
            )
        return self._session
    
    async def close(self) -> None:
        """Close the aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
    
    async def _make_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        retry_count: int = 0
    ) -> Dict[str, Any]:
        """
        Make an HTTP request to the Gemini API with retry logic.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path
            params: Optional query parameters
            data: Optional request body data
            retry_count: Current retry attempt (used internally)
            
        Returns:
            API response as a dictionary
            
        Raises:
            GeminiApiError: If the API request fails after all retries
        """
        session = await self._ensure_session()
        
        # Add API key to query parameters
        if params is None:
            params = {}
        params["key"] = self._api_key
        
        url = f"{self._base_url}/{endpoint}"
        
        try:
            # Make the request
            async with session.request(method, url, params=params, json=data) as response:
                # Check for rate limiting
                if response.status == 429:
                    retry_after = int(response.headers.get("Retry-After", "5"))
                    logger.warning(f"Rate limited by Gemini API. Retrying after {retry_after} seconds.")
                    await asyncio.sleep(retry_after)
                    return await self._make_request(method, endpoint, params, data, retry_count)
                
                # Check for other errors
                if response.status >= 400:
                    error_text = await response.text()
                    logger.error(f"Gemini API error: {response.status} - {error_text}")
                    
                    # Retry on server errors (5xx) or specific client errors
                    if (response.status >= 500 or response.status in [408, 429]) and retry_count < self._max_retries:
                        retry_delay = self._backoff_factor * (2 ** retry_count)
                        logger.info(f"Retrying Gemini API request in {retry_delay:.2f} seconds (attempt {retry_count + 1}/{self._max_retries})")
                        await asyncio.sleep(retry_delay)
                        return await self._make_request(method, endpoint, params, data, retry_count + 1)
                    
                    raise GeminiApiError(
                        f"Gemini API request failed: {response.status}",
                        status_code=response.status,
                        response_text=error_text
                    )
                
                # Parse successful response
                result = await response.json()
                return result
                
        except aiohttp.ClientError as e:
            # Handle network errors
            logger.error(f"Network error in Gemini API request: {str(e)}")
            
            if retry_count < self._max_retries:
                retry_delay = self._backoff_factor * (2 ** retry_count)
                logger.info(f"Retrying Gemini API request in {retry_delay:.2f} seconds (attempt {retry_count + 1}/{self._max_retries})")
                await asyncio.sleep(retry_delay)
                return await self._make_request(method, endpoint, params, data, retry_count + 1)
            
            raise GeminiApiError(f"Gemini API network error after {self._max_retries} retries: {str(e)}")
    
    def _calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """
        Calculate the cost of a Gemini API call based on token usage.
        
        Args:
            model: Model name
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            
        Returns:
            Cost in USD
        """
        # Get model-specific cost multipliers
        model_multipliers = self._param_multipliers.get(model, {})
        
        # Calculate input token cost
        input_cost_per_token = model_multipliers.get("input_tokens", 0.0)
        input_cost = input_tokens * input_cost_per_token
        
        # Calculate output token cost
        output_cost_per_token = model_multipliers.get("output_tokens", 0.0)
        output_cost = output_tokens * output_cost_per_token
        
        # Calculate total cost
        total_cost = input_cost + output_cost
        
        # Apply minimum cost if needed
        if total_cost < self._default_cost:
            total_cost = self._default_cost
        
        return total_cost
    
    def _extract_token_counts(self, response: Dict[str, Any]) -> Tuple[int, int]:
        """
        Extract input and output token counts from a Gemini API response.
        
        Args:
            response: Gemini API response
            
        Returns:
            Tuple of (input_tokens, output_tokens)
        """
        usage = response.get("usage", {})
        input_tokens = usage.get("promptTokenCount", 0)
        output_tokens = usage.get("candidatesTokenCount", 0)
        return input_tokens, output_tokens
    
    @LlmMetrics.track_llm_request("gemini", "text-generation")
    async def generate_text(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_output_tokens: int = 1024,
        top_p: float = 0.95,
        top_k: int = 40,
        safety_settings: Optional[List[Dict[str, Any]]] = None,
        system_instruction: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate text using the Gemini model.
        
        Args:
            prompt: Text prompt
            model: Model name (defaults to client's default model)
            temperature: Sampling temperature (0.0-1.0)
            max_output_tokens: Maximum number of tokens to generate
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            safety_settings: Optional safety settings
            system_instruction: Optional system instruction
            
        Returns:
            Generated text response
        """
        model_name = model or self._default_model
        endpoint = f"models/{model_name}:generateContent"
        
        # Build request payload
        content_parts = [{"text": prompt}]
        
        data = {
            "contents": [
                {
                    "parts": content_parts
                }
            ],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_output_tokens,
                "topP": top_p,
                "topK": top_k
            }
        }
        
        # Add system instruction if provided
        if system_instruction:
            data["systemInstruction"] = {"parts": [{"text": system_instruction}]}
        
        # Add safety settings if provided
        if safety_settings:
            data["safetySettings"] = safety_settings
        
        # Make the request
        response = await self._make_request("POST", endpoint, data=data)
        
        # Extract token counts and calculate cost
        input_tokens, output_tokens = self._extract_token_counts(response)
        cost = self._calculate_cost(model_name, input_tokens, output_tokens)
        
        # Record token usage and cost
        LlmMetrics.record_token_usage("gemini", model_name, input_tokens, output_tokens, cost)
        
        # Also record as external API cost for consistency
        ApiMetrics.record_api_cost("gemini", "text-generation", cost)
        
        return response
    
    @LlmMetrics.track_llm_request("gemini", "embedding")
    async def get_embeddings(
        self,
        text: Union[str, List[str]],
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get embeddings for text using the Gemini embedding model.
        
        Args:
            text: Text to embed (string or list of strings)
            model: Model name (defaults to client's default embedding model)
            
        Returns:
            Embedding response
        """
        model_name = model or self._default_embedding_model
        endpoint = f"models/{model_name}:embedContent"
        
        # Convert single string to list
        if isinstance(text, str):
            text = [text]
        
        # Build request payload
        data = {
            "model": model_name,
            "content": {
                "parts": [{"text": t} for t in text]
            }
        }
        
        # Make the request
        response = await self._make_request("POST", endpoint, data=data)
        
        # Calculate token count (approximate)
        total_chars = sum(len(t) for t in text)
        approximate_tokens = total_chars // 4  # Rough approximation
        
        # Calculate cost
        cost = self._calculate_cost(model_name, approximate_tokens, 0)
        
        # Record token usage and cost
        LlmMetrics.record_token_usage("gemini", model_name, approximate_tokens, 0, cost)
        
        # Also record as external API cost for consistency
        ApiMetrics.record_api_cost("gemini", "embedding", cost)
        
        return response
    
    @LlmMetrics.track_llm_request("gemini", "vision")
    async def analyze_image(
        self,
        image_data: Union[bytes, str],
        prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_output_tokens: int = 1024
    ) -> str:
        """
        Analyze an image using the Gemini Vision model.
        
        Args:
            image_data: Image data as bytes or base64-encoded string
            prompt: Text prompt to guide the image analysis
            model: Model name (defaults to client's default model)
            temperature: Sampling temperature (0.0-1.0)
            max_output_tokens: Maximum number of tokens to generate
            
        Returns:
            Analysis text
        """
        model_name = model or self._default_model
        endpoint = f"models/{model_name}:generateContent"
        
        # Convert bytes to base64 if needed
        if isinstance(image_data, bytes):
            image_base64 = base64.b64encode(image_data).decode("utf-8")
        else:
            image_base64 = image_data
        
        # Build request payload
        data = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt},
                        {
                            "inline_data": {
                                "mime_type": "image/jpeg",
                                "data": image_base64
                            }
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_output_tokens
            }
        }
        
        # Make the request
        response = await self._make_request("POST", endpoint, data=data)
        
        # Extract token counts and calculate cost
        input_tokens, output_tokens = self._extract_token_counts(response)
        
        # Add image token estimate (approximate)
        image_token_estimate = 1000  # Rough approximation for a typical image
        input_tokens += image_token_estimate
        
        # Calculate cost
        cost = self._calculate_cost(model_name, input_tokens, output_tokens)
        
        # Record token usage and cost
        LlmMetrics.record_token_usage("gemini", model_name, input_tokens, output_tokens, cost)
        
        # Also record as external API cost for consistency
        ApiMetrics.record_api_cost("gemini", "vision", cost)
        
        # Extract the text from the response
        try:
            text = response["candidates"][0]["content"]["parts"][0]["text"]
            return text
        except (KeyError, IndexError):
            logger.error("Failed to extract text from Gemini Vision response")
            return "Image analysis failed"
    
    @LlmMetrics.track_llm_request("gemini", "chat")
    async def chat(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_output_tokens: int = 1024,
        system_instruction: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Chat with the Gemini model.
        
        Args:
            messages: List of message objects (role and content)
            model: Model name (defaults to client's default model)
            temperature: Sampling temperature (0.0-1.0)
            max_output_tokens: Maximum number of tokens to generate
            system_instruction: Optional system instruction
            
        Returns:
            Chat response
        """
        model_name = model or self._default_model
        endpoint = f"models/{model_name}:generateContent"
        
        # Convert messages to Gemini format
        contents = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                # System messages are handled differently in Gemini
                system_instruction = content
                continue
            
            gemini_role = "user" if role == "user" else "model"
            contents.append({
                "role": gemini_role,
                "parts": [{"text": content}]
            })
        
        # Build request payload
        data = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_output_tokens
            }
        }
        
        # Add system instruction if provided
        if system_instruction:
            data["systemInstruction"] = {"parts": [{"text": system_instruction}]}
        
        # Make the request
        response = await self._make_request("POST", endpoint, data=data)
        
        # Extract token counts and calculate cost
        input_tokens, output_tokens = self._extract_token_counts(response)
        cost = self._calculate_cost(model_name, input_tokens, output_tokens)
        
        # Record token usage and cost
        LlmMetrics.record_token_usage("gemini", model_name, input_tokens, output_tokens, cost)
        
        # Also record as external API cost for consistency
        ApiMetrics.record_api_cost("gemini", "chat", cost)
        
        return response
