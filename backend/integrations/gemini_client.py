"""Google Gemini API client for multimodal AI capabilities."""

import logging
from typing import Dict, List, Optional, Any, Union
import base64
import io
from PIL import Image

from google import genai
from google.genai.types import HarmCategory, HarmBlockThreshold, GenerateContentResponse

from backend.config import GeminiConfig
from backend.core.metrics import track_llm_usage


logger = logging.getLogger(__name__)

# Define pricing per million tokens for different Gemini models
GEMINI_PRICING = {
    "models/gemini-1.5-flash-latest": {"input": 0.35/1_000_000, "output": 0.70/1_000_000},
    "models/gemini-1.5-pro-latest": {"input": 3.50/1_000_000, "output": 10.50/1_000_000},
    # Default pricing if model not found
    "default": {"input": 1.0/1_000_000, "output": 2.0/1_000_000},
}


class GeminiClient:
    """Client for interacting with Google's Gemini API."""
    
    def __init__(self):
        """Initialize the Gemini client."""
        self.config = GeminiConfig()
        
        # Configure the API
        genai.configure(api_key=self.config.API_KEY)
        
        # Initialize the model
        self.model = genai.GenerativeModel(
            model_name=self.config.MODEL,
            generation_config=genai.types.GenerationConfig(
                temperature=self.config.TEMPERATURE,
                top_p=self.config.TOP_P,
                top_k=self.config.TOP_K,
                max_output_tokens=self.config.MAX_OUTPUT_TOKENS,
            ),
            safety_settings={
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            }
        )
        
        logger.info(f"Gemini client initialized with model: {self.config.MODEL}")
    
    def _update_metrics(self, model: str, response: GenerateContentResponse, success: bool):
        """
        Update metrics for LLM usage.
        
        Args:
            model: The model name
            response: The response from Gemini
            success: Whether the request was successful
        """
        # Extract token usage from response metadata
        prompt_tokens = response.usage_metadata.get("prompt_token_count", 0) if response.usage_metadata else 0
        output_tokens = response.usage_metadata.get("candidates_token_count", 0) if response.usage_metadata else 0
        
        # Get pricing for the model
        pricing = GEMINI_PRICING.get(model, GEMINI_PRICING["default"])
        
        # Calculate cost
        cost = (prompt_tokens * pricing["input"]) + (output_tokens * pricing["output"])
        
        # Track metrics
        track_llm_usage(model, prompt_tokens, output_tokens, cost, success)
    
    async def generate_text(
        self,
        prompt: str,
        context: Optional[str] = None,
        system_instruction: Optional[str] = None,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
    ) -> str:
        """Generate text response from Gemini."""
        try:
            # Prepare the full prompt
            full_prompt = prompt
            if context:
                full_prompt = f"Context: {context}\n\nQuery: {prompt}"
            if system_instruction:
                full_prompt = f"System: {system_instruction}\n\n{full_prompt}"
            
            # Use custom parameters if provided, otherwise use defaults
            generation_config = None
            if temperature is not None or max_output_tokens is not None:
                generation_config = genai.types.GenerationConfig(
                    temperature=temperature if temperature is not None else self.config.TEMPERATURE,
                    max_output_tokens=max_output_tokens if max_output_tokens is not None else self.config.MAX_OUTPUT_TOKENS,
                )
            
            # Generate content
            response = await self.model.generate_content_async(
                full_prompt,
                generation_config=generation_config
            )
            
            if response.text:
                logger.debug(f"Generated text response: {response.text[:100]}...")
                # Update metrics
                self._update_metrics(self.config.MODEL, response, True)
                return response.text
            else:
                logger.warning("Empty response from Gemini")
                # Update metrics with empty response
                self._update_metrics(self.config.MODEL, response, False)
                return "I apologize, but I couldn't generate a response for that query."
                
        except Exception as e:
            logger.error(f"Error generating text with Gemini: {e}")
            # Track failed request
            track_llm_usage(self.config.MODEL, 0, 0, 0, False)
            raise
    
    async def generate_text_with_tools(
        self,
        prompt: str,
        tools: List[Any],
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
    ) -> Any:
        """Generate text with tool calling capabilities."""
        try:
            # Use custom parameters if provided, otherwise use defaults
            generation_config = None
            if temperature is not None or max_output_tokens is not None:
                generation_config = genai.types.GenerationConfig(
                    temperature=temperature if temperature is not None else self.config.TEMPERATURE,
                    max_output_tokens=max_output_tokens if max_output_tokens is not None else self.config.MAX_OUTPUT_TOKENS,
                )
            
            # Generate content with tools
            response = await self.model.generate_content_async(
                prompt,
                tools=tools,
                generation_config=generation_config
            )
            
            # Update metrics
            self._update_metrics(self.config.MODEL, response, True)
            return response
                
        except Exception as e:
            logger.error(f"Error generating text with tools: {e}")
            # Track failed request
            track_llm_usage(self.config.MODEL, 0, 0, 0, False)
            raise
    
    async def analyze_image(
        self,
        image_data: Union[bytes, str, Image.Image],
        prompt: str = "Analyze this image and describe what you see."
    ) -> str:
        """Analyze an image using Gemini's vision capabilities."""
        try:
            # Convert image data to PIL Image if needed
            if isinstance(image_data, bytes):
                image = Image.open(io.BytesIO(image_data))
            elif isinstance(image_data, str):
                # Assume base64 encoded image
                image_bytes = base64.b64decode(image_data)
                image = Image.open(io.BytesIO(image_bytes))
            elif isinstance(image_data, Image.Image):
                image = image_data
            else:
                raise ValueError("Unsupported image data format")
            
            # Generate content with image and prompt
            response = await self.model.generate_content_async([prompt, image])
            
            if response.text:
                logger.debug(f"Image analysis response: {response.text[:100]}...")
                # Update metrics
                self._update_metrics(self.config.MODEL, response, True)
                return response.text
            else:
                logger.warning("Empty response from Gemini image analysis")
                # Update metrics with empty response
                self._update_metrics(self.config.MODEL, response, False)
                return "I couldn't analyze this image."
                
        except Exception as e:
            logger.error(f"Error analyzing image with Gemini: {e}")
            # Track failed request
            track_llm_usage(self.config.MODEL, 0, 0, 0, False)
            raise
    
    async def generate_cypher_query(
        self,
        natural_language_query: str,
        schema_context: str,
        examples: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """Convert natural language to Cypher query."""
        try:
            # Build the prompt for Cypher generation
            prompt = f"""
You are an expert in converting natural language queries to Neo4j Cypher queries.

Graph Schema Context:
{schema_context}

Natural Language Query: {natural_language_query}

Please generate a Cypher query that answers the natural language question.
Follow these guidelines:
1. Use proper Cypher syntax
2. Consider the provided schema context
3. Return only the Cypher query without explanations
4. Use appropriate MATCH, WHERE, RETURN clauses
5. Handle case-insensitive matching where appropriate

"""
            
            if examples:
                prompt += "\nExamples:\n"
                for example in examples:
                    prompt += f"Q: {example['question']}\nCypher: {example['cypher']}\n\n"
            
            prompt += "Cypher Query:"
            
            response = await self.model.generate_content_async(prompt)
            
            if response.text:
                # Extract just the Cypher query
                cypher_query = response.text.strip()
                # Remove any markdown formatting
                if cypher_query.startswith("```"):
                    lines = cypher_query.split("\n")
                    cypher_query = "\n".join(lines[1:-1])
                
                logger.debug(f"Generated Cypher query: {cypher_query}")
                # Update metrics
                self._update_metrics(self.config.MODEL, response, True)
                return cypher_query
            else:
                logger.warning("Empty Cypher query response from Gemini")
                # Update metrics with empty response
                self._update_metrics(self.config.MODEL, response, False)
                return "MATCH (n) RETURN n LIMIT 10"  # Fallback query
                
        except Exception as e:
            logger.error(f"Error generating Cypher query: {e}")
            # Track failed request
            track_llm_usage(self.config.MODEL, 0, 0, 0, False)
            raise
    
    async def generate_python_code(
        self,
        task_description: str,
        context: Optional[str] = None,
        libraries: Optional[List[str]] = None
    ) -> str:
        """Generate Python code for data analysis tasks."""
        try:
            prompt = f"""
You are an expert Python developer specializing in data analysis and graph analytics.

Task: {task_description}

"""
            
            if context:
                prompt += f"Context: {context}\n\n"
            
            if libraries:
                prompt += f"Available libraries: {', '.join(libraries)}\n\n"
            
            prompt += """
Please generate Python code that:
1. Is well-commented and readable
2. Handles errors appropriately
3. Uses best practices
4. Returns results in a structured format
5. Includes necessary imports

Return only the Python code without explanations.
"""
            
            response = await self.model.generate_content_async(prompt)
            
            if response.text:
                # Extract Python code
                code = response.text.strip()
                # Remove markdown formatting if present
                if code.startswith("```python"):
                    lines = code.split("\n")
                    code = "\n".join(lines[1:-1])
                elif code.startswith("```"):
                    lines = code.split("\n")
                    code = "\n".join(lines[1:-1])
                
                logger.debug(f"Generated Python code: {code[:200]}...")
                # Update metrics
                self._update_metrics(self.config.MODEL, response, True)
                return code
            else:
                logger.warning("Empty Python code response from Gemini")
                # Update metrics with empty response
                self._update_metrics(self.config.MODEL, response, False)
                return "# No code generated"
                
        except Exception as e:
            logger.error(f"Error generating Python code: {e}")
            # Track failed request
            track_llm_usage(self.config.MODEL, 0, 0, 0, False)
            raise
    
    async def explain_results(
        self,
        query: str,
        results: Any,
        context: Optional[str] = None
    ) -> str:
        """Generate natural language explanation of query results."""
        try:
            prompt = f"""
You are an expert data analyst. Please provide a clear, concise explanation of the following query results.

Original Query: {query}

Results: {str(results)[:2000]}  # Limit results size

"""
            
            if context:
                prompt += f"Context: {context}\n\n"
            
            prompt += """
Please provide:
1. A summary of what the results show
2. Key insights or patterns
3. Any notable findings or anomalies
4. Recommendations for further analysis if applicable

Keep the explanation clear and accessible to business users.
"""
            
            response = await self.model.generate_content_async(prompt)
            
            if response.text:
                logger.debug(f"Generated explanation: {response.text[:100]}...")
                # Update metrics
                self._update_metrics(self.config.MODEL, response, True)
                return response.text
            else:
                logger.warning("Empty explanation response from Gemini")
                # Update metrics with empty response
                self._update_metrics(self.config.MODEL, response, False)
                return "Unable to generate explanation for the results."
                
        except Exception as e:
            logger.error(f"Error generating explanation: {e}")
            # Track failed request
            track_llm_usage(self.config.MODEL, 0, 0, 0, False)
            raise
