"""Google Gemini API client for multimodal AI capabilities."""

import logging
from typing import Dict, List, Optional, Any, Union
import base64
import io
from PIL import Image

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

from backend.config import GeminiConfig


logger = logging.getLogger(__name__)


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
    
    async def generate_text(
        self,
        prompt: str,
        context: Optional[str] = None,
        system_instruction: Optional[str] = None
    ) -> str:
        """Generate text response from Gemini."""
        try:
            # Prepare the full prompt
            full_prompt = prompt
            if context:
                full_prompt = f"Context: {context}\n\nQuery: {prompt}"
            if system_instruction:
                full_prompt = f"System: {system_instruction}\n\n{full_prompt}"
            
            response = await self.model.generate_content_async(full_prompt)
            
            if response.text:
                logger.debug(f"Generated text response: {response.text[:100]}...")
                return response.text
            else:
                logger.warning("Empty response from Gemini")
                return "I apologize, but I couldn't generate a response for that query."
                
        except Exception as e:
            logger.error(f"Error generating text with Gemini: {e}")
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
                return response.text
            else:
                logger.warning("Empty response from Gemini image analysis")
                return "I couldn't analyze this image."
                
        except Exception as e:
            logger.error(f"Error analyzing image with Gemini: {e}")
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
                return cypher_query
            else:
                logger.warning("Empty Cypher query response from Gemini")
                return "MATCH (n) RETURN n LIMIT 10"  # Fallback query
                
        except Exception as e:
            logger.error(f"Error generating Cypher query: {e}")
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
                return code
            else:
                logger.warning("Empty Python code response from Gemini")
                return "# No code generated"
                
        except Exception as e:
            logger.error(f"Error generating Python code: {e}")
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
                return response.text
            else:
                logger.warning("Empty explanation response from Gemini")
                return "Unable to generate explanation for the results."
                
        except Exception as e:
            logger.error(f"Error generating explanation: {e}")
            raise
