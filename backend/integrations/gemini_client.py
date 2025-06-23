"""
Gemini API Client Integration

This module provides a comprehensive client for interacting with Google's Gemini API.
It handles text generation, embeddings, vision capabilities, and specialized tasks
like Cypher query generation. It is fully integrated with the application's
back-pressure, cost tracking, and observability systems.
"""

import logging
import os
from typing import Any, Dict, List, Optional

import json
import google.generativeai as genai
from google.generativeai.types import GenerationConfig, HarmCategory, HarmBlockThreshold
from PIL import Image
import io

from backend.core.backpressure import with_backpressure
from backend.core.metrics import ApiMetrics, LlmMetrics
from backend.providers import get_provider

logger = logging.getLogger(__name__)


class GeminiClient:
    """A client for Google's Gemini API, with integrated cost and back-pressure controls."""

    def __init__(self):
        """
        Initializes the Gemini client by loading configuration from the provider
        registry and setting up the API models.
        """
        provider_config = get_provider("gemini")
        if not provider_config:
            raise ValueError("Gemini provider configuration not found in registry.")

        api_key = provider_config.get("auth", {}).get("api_key_env_var")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set.")

        genai.configure(api_key=api_key)

        self.text_model_name = provider_config.get("text_model", "gemini-1.5-pro-latest")
        self.vision_model_name = provider_config.get("vision_model", "gemini-1.5-pro-latest") # Vision model is the same for 1.5
        self.embedding_model_name = provider_config.get("embedding_model", "models/text-embedding-004")

        self.cost_rules = provider_config.get("cost_rules", {})

        # Configure safety settings to be less restrictive
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }

        self.generation_config = GenerationConfig(
            temperature=0.2,
            top_p=0.9,
            top_k=32,
            max_output_tokens=4096,
        )

        try:
            self.model = genai.GenerativeModel(
                self.text_model_name,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings,
            )
            self.vision_model = genai.GenerativeModel(
                self.vision_model_name,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings,
            )
            logger.info(f"Gemini models '{self.text_model_name}' and '{self.vision_model_name}' initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini models: {e}")
            raise

    def _calculate_and_track_cost(
        self,
        model_name: str,
        operation: str,
        input_tokens: int,
        output_tokens: int = 0,
    ):
        """Calculates cost based on token usage and emits metrics."""
        cost = 0.0
        model_cost_rules = self.cost_rules.get("param_multipliers", {}).get(model_name, {})
        
        input_cost_per_token = model_cost_rules.get("input_tokens", 0.0)
        output_cost_per_token = model_cost_rules.get("output_tokens", 0.0)

        cost += input_tokens * input_cost_per_token
        cost += output_tokens * output_cost_per_token
        
        if cost > 0:
            logger.debug(f"Gemini API call cost: ${cost:.6f} ({operation}, {model_name})")
            # Emit detailed LLM cost metric
            LlmMetrics.track_cost(
                model=model_name,
                operation=operation,
                cost=cost,
            )
            # Emit generic provider credit metric
            ApiMetrics.track_credits(
                provider="gemini",
                endpoint=operation,
                credit_type="usd",
                amount=cost
            )
        
        # Track token usage
        LlmMetrics.track_tokens(
            model=model_name,
            operation=operation,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    @with_backpressure(provider_id="gemini", endpoint="generate_content")
    async def generate_text(self, prompt: str, context: Optional[str] = None) -> str:
        """
        Generates text using the Gemini model.

        Args:
            prompt: The main prompt for the model.
            context: Optional additional context to provide.

        Returns:
            The generated text response.
        """
        full_prompt = f"{context}\n\n{prompt}" if context else prompt
        try:
            response = await self.model.generate_content_async(full_prompt)
            
            # Track cost and tokens
            if response.usage_metadata:
                self._calculate_and_track_cost(
                    model_name=self.text_model_name,
                    operation="generate_content",
                    input_tokens=response.usage_metadata.prompt_token_count,
                    output_tokens=response.usage_metadata.candidates_token_count,
                )
            
            return response.text
        except Exception as e:
            logger.error(f"Error generating text with Gemini: {e}")
            raise

    @with_backpressure(provider_id="gemini", endpoint="embed_content")
    async def get_embeddings(self, text: str) -> List[float]:
        """
        Generates vector embeddings for a given text.

        Args:
            text: The text to embed.

        Returns:
            A list of floats representing the vector embedding.
        """
        try:
            result = await genai.embed_content_async(
                model=self.embedding_model_name,
                content=text,
                task_type="retrieval_document"
            )
            
            # Estimate token count for cost tracking (embedding API doesn't return it)
            token_count = await self.model.count_tokens_async(text)
            self._calculate_and_track_cost(
                model_name=self.embedding_model_name,
                operation="embed_content",
                input_tokens=token_count.total_tokens,
            )

            return result['embedding']
        except Exception as e:
            logger.error(f"Error getting embeddings from Gemini: {e}")
            raise

    @with_backpressure(provider_id="gemini", endpoint="generate_content_vision")
    async def analyze_image(self, image_data: bytes, prompt: str) -> str:
        """
        Analyzes an image using the Gemini vision model.

        Args:
            image_data: The raw bytes of the image.
            prompt: The prompt to guide the analysis.

        Returns:
            A text description of the image analysis.
        """
        try:
            image = Image.open(io.BytesIO(image_data))
            response = await self.vision_model.generate_content_async([prompt, image])

            # Track cost and tokens
            if response.usage_metadata:
                self._calculate_and_track_cost(
                    model_name=self.vision_model_name,
                    operation="generate_content_vision",
                    input_tokens=response.usage_metadata.prompt_token_count,
                    output_tokens=response.usage_metadata.candidates_token_count,
                )

            return response.text
        except Exception as e:
            logger.error(f"Error analyzing image with Gemini: {e}")
            raise

    @with_backpressure(provider_id="gemini", endpoint="generate_cypher")
    async def generate_cypher_query(self, user_prompt: str, schema_context: str) -> str:
        """
        Generates a Cypher query from a natural language prompt and schema context.

        Args:
            user_prompt: The user's natural language request.
            schema_context: The Neo4j graph schema description.

        Returns:
            A Cypher query string.
        """
        system_prompt = f"""
You are an expert Neo4j developer. Your task is to write a Cypher query based on the user's request and the provided graph schema.
- Only return the Cypher query. Do not include any explanations, comments, or markdown formatting like ```cypher.
- The query must be compatible with Neo4j 5.
- Use the provided schema to ensure correct node labels, relationship types, and property keys.

Graph Schema:
{schema_context}
"""
        return await self.generate_text(prompt=user_prompt, context=system_prompt)

    @with_backpressure(provider_id="gemini", endpoint="explain_results")
    async def explain_results(self, user_prompt: str, results: List[Dict[str, Any]], context: Optional[str] = None) -> str:
        """
        Generates a natural language explanation of graph query results.

        Args:
            user_prompt: The original user prompt that led to the results.
            results: The data returned from the Cypher query execution.
            context: Optional additional context.

        Returns:
            A natural language explanation.
        """
        system_prompt = f"""
You are a helpful data analyst. The user asked: "{user_prompt}".
A graph query was executed, and it returned the following data (in JSON format):
{json.dumps(results, indent=2)}

Your task is to provide a concise, natural language explanation of these results.
- Do not just restate the JSON. Synthesize the findings into a clear answer.
- If the results are empty, state that no data was found.
- If there are many results, summarize them.
"""
        return await self.generate_text(prompt="Explain these results to me.", context=system_prompt)

