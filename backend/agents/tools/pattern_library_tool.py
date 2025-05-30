"""
PatternLibraryTool for managing and converting fraud pattern definitions to Cypher queries.

This tool manages a library of fraud pattern templates and provides methods to
convert these patterns into executable Cypher queries. It supports both rule-based
(deterministic) conversion and LLM-assisted conversion for more complex patterns.
"""

import json
import logging
import os
import yaml
from typing import Any, Dict, List, Optional, Union, Tuple
from pathlib import Path

from crewai_tools import BaseTool
from pydantic import BaseModel, Field

from backend.integrations.gemini_client import GeminiClient
from backend.integrations.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)

# Default directory for pattern definitions
PATTERN_LIBRARY_DIR = Path("backend/agents/patterns")


class PatternInput(BaseModel):
    """Input model for pattern library operations."""
    
    operation: str = Field(
        ...,
        description="Operation to perform: 'search', 'convert', 'score', or 'list'"
    )
    pattern_id: Optional[str] = Field(
        default=None,
        description="ID of the specific pattern to use (for 'convert' or 'score')"
    )
    pattern_type: Optional[str] = Field(
        default=None,
        description="Type of pattern to search for (e.g., 'money_laundering', 'fraud', 'tax_evasion')"
    )
    parameters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Parameters to apply to the pattern template"
    )
    conversion_method: Optional[str] = Field(
        default="rule_based",
        description="Method to use for converting patterns to Cypher: 'rule_based', 'llm', or 'hybrid'"
    )
    custom_pattern: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Custom pattern definition (for one-time conversion)"
    )


class PatternLibraryTool(BaseTool):
    """
    Tool for managing and converting fraud pattern definitions to Cypher queries.
    
    This tool provides access to a library of fraud pattern templates and
    converts them into executable Cypher queries. It supports both rule-based
    (deterministic) conversion for known patterns and LLM-assisted conversion
    for more complex or novel patterns.
    """
    
    name: str = "pattern_library_tool"
    description: str = """
    Access and convert fraud pattern templates to executable Cypher queries.
    
    Use this tool when you need to:
    - Search for known fraud patterns in the library
    - Convert pattern templates to executable Cypher queries
    - Score results against pattern criteria
    - List available patterns by type or category
    
    The tool supports both rule-based (deterministic) conversion for well-defined
    patterns and LLM-assisted conversion for more complex or novel patterns.
    
    Example usage:
    - Convert a "circular_transaction" pattern to a Cypher query
    - Search for money laundering patterns in the library
    - Score transaction results against known fraud indicators
    - List all available tax evasion patterns
    """
    args_schema: type[BaseModel] = PatternInput
    
    def __init__(
        self,
        gemini_client: Optional[GeminiClient] = None,
        neo4j_client: Optional[Neo4jClient] = None,
        pattern_dir: Optional[Path] = None
    ):
        """
        Initialize the PatternLibraryTool.
        
        Args:
            gemini_client: Optional GeminiClient for LLM-assisted conversion
            neo4j_client: Optional Neo4jClient for executing queries
            pattern_dir: Optional custom directory for pattern definitions
        """
        super().__init__()
        self.gemini_client = gemini_client or GeminiClient()
        self.neo4j_client = neo4j_client or Neo4jClient()
        self.pattern_dir = pattern_dir or PATTERN_LIBRARY_DIR
        self.patterns = {}
        self.load_patterns()
    
    def load_patterns(self):
        """Load all pattern definitions from the pattern directory."""
        try:
            # Create pattern directory if it doesn't exist
            os.makedirs(self.pattern_dir, exist_ok=True)
            
            # Load patterns from files
            for file_path in self.pattern_dir.glob("**/*.yaml"):
                try:
                    with open(file_path, "r") as f:
                        pattern = yaml.safe_load(f)
                        if self._validate_pattern(pattern):
                            self.patterns[pattern["id"]] = pattern
                except Exception as e:
                    logger.error(f"Error loading pattern from {file_path}: {e}")
            
            for file_path in self.pattern_dir.glob("**/*.json"):
                try:
                    with open(file_path, "r") as f:
                        pattern = json.load(f)
                        if self._validate_pattern(pattern):
                            self.patterns[pattern["id"]] = pattern
                except Exception as e:
                    logger.error(f"Error loading pattern from {file_path}: {e}")
            
            # If no patterns were loaded, use default patterns
            if not self.patterns:
                logger.info("No patterns found, loading default patterns")
                self._load_default_patterns()
                
            logger.info(f"Loaded {len(self.patterns)} patterns")
            
        except Exception as e:
            logger.error(f"Error loading patterns: {e}")
            # Load default patterns as fallback
            self._load_default_patterns()
    
    def _load_default_patterns(self):
        """Load default pattern definitions."""
        self.patterns = {
            "circular_transaction": {
                "id": "circular_transaction",
                "name": "Circular Transaction Pattern",
                "description": "Detects funds that flow in a circular pattern through multiple accounts and return to the origin",
                "type": "money_laundering",
                "severity": "high",
                "template": {
                    "nodes": [
                        {"id": "origin", "labels": ["Account"], "properties": {}},
                        {"id": "intermediate", "labels": ["Account"], "properties": {}, "repeat": {"min": 1, "max": 5}}
                    ],
                    "relationships": [
                        {"start": "origin", "end": "intermediate.0", "type": "TRANSFER", "properties": {"amount": {"param": "min_amount", "default": 10000}}},
                        {"start": "intermediate.i", "end": "intermediate.i+1", "type": "TRANSFER", "properties": {}, "repeat": {"min": 0, "max": 4}},
                        {"start": "intermediate.last", "end": "origin", "type": "TRANSFER", "properties": {}}
                    ]
                },
                "cypher_template": """
                MATCH path = (origin:Account)-[:TRANSFER*1..{max_hops}]->(origin)
                WHERE {min_amount_condition}
                RETURN path, 
                       nodes(path) as entities, 
                       relationships(path) as transfers,
                       length(path) as path_length
                ORDER BY path_length DESC
                LIMIT {limit}
                """,
                "parameters": {
                    "max_hops": {"type": "int", "default": 6, "description": "Maximum number of hops in the circular path"},
                    "min_amount": {"type": "float", "default": 10000, "description": "Minimum transaction amount to consider"},
                    "limit": {"type": "int", "default": 10, "description": "Maximum number of results to return"}
                }
            },
            "rapid_succession": {
                "id": "rapid_succession",
                "name": "Rapid Succession Transactions",
                "description": "Detects multiple transactions in rapid succession that may indicate structuring",
                "type": "structuring",
                "severity": "medium",
                "template": {
                    "nodes": [
                        {"id": "source", "labels": ["Account"], "properties": {}},
                        {"id": "target", "labels": ["Account"], "properties": {}, "repeat": {"min": 1, "max": 10}}
                    ],
                    "relationships": [
                        {"start": "source", "end": "target.i", "type": "TRANSFER", "properties": {
                            "amount": {"param": "max_amount", "default": 9999},
                            "timestamp": {"param": "time_window", "default": "24h"}
                        }, "repeat": {"min": 3, "max": 10}}
                    ]
                },
                "cypher_template": """
                MATCH (source:Account)-[t:TRANSFER]->(target:Account)
                WHERE t.amount <= {max_amount}
                WITH source, collect(t) as transfers
                WHERE size(transfers) >= {min_transfers}
                AND (max(transfers.timestamp) - min(transfers.timestamp)) <= duration({time_window})
                RETURN source, transfers, size(transfers) as transfer_count
                ORDER BY transfer_count DESC
                LIMIT {limit}
                """,
                "parameters": {
                    "max_amount": {"type": "float", "default": 9999, "description": "Maximum amount per transaction (just below reporting threshold)"},
                    "min_transfers": {"type": "int", "default": 3, "description": "Minimum number of transfers to consider suspicious"},
                    "time_window": {"type": "string", "default": "24h", "description": "Time window for the transactions (e.g., '24h', '7d')"},
                    "limit": {"type": "int", "default": 10, "description": "Maximum number of results to return"}
                }
            },
            "shell_company_network": {
                "id": "shell_company_network",
                "name": "Shell Company Network",
                "description": "Identifies networks of shell companies with shared directors or addresses",
                "type": "tax_evasion",
                "severity": "high",
                "template": {
                    "nodes": [
                        {"id": "company", "labels": ["Company"], "properties": {"incorporation_date": {"param": "max_age", "default": "2y"}}},
                        {"id": "director", "labels": ["Person"], "properties": {}},
                        {"id": "address", "labels": ["Address"], "properties": {}}
                    ],
                    "relationships": [
                        {"start": "company", "end": "director", "type": "HAS_DIRECTOR", "properties": {}},
                        {"start": "company", "end": "address", "type": "REGISTERED_AT", "properties": {}},
                        {"start": "director", "end": "company", "type": "DIRECTS", "properties": {}, "repeat": {"min": 3, "max": 20}}
                    ]
                },
                "cypher_template": """
                MATCH (c:Company)-[:REGISTERED_AT]->(a:Address)<-[:REGISTERED_AT]-(other:Company)
                WHERE c.incorporation_date >= datetime() - duration({max_age})
                WITH a, collect(c) as companies
                WHERE size(companies) >= {min_companies}
                
                MATCH (director:Person)-[:DIRECTS]->(company:Company)
                WHERE company IN companies
                WITH director, collect(company) as directed_companies
                WHERE size(directed_companies) >= {min_directed}
                
                RETURN director, directed_companies, size(directed_companies) as company_count
                ORDER BY company_count DESC
                LIMIT {limit}
                """,
                "parameters": {
                    "max_age": {"type": "string", "default": "2y", "description": "Maximum age of companies to consider (e.g., '2y', '18m')"},
                    "min_companies": {"type": "int", "default": 3, "description": "Minimum number of companies at same address"},
                    "min_directed": {"type": "int", "default": 2, "description": "Minimum number of companies directed by same person"},
                    "limit": {"type": "int", "default": 10, "description": "Maximum number of results to return"}
                }
            }
        }
    
    def _validate_pattern(self, pattern: Dict[str, Any]) -> bool:
        """
        Validate a pattern definition.
        
        Args:
            pattern: Pattern definition to validate
            
        Returns:
            True if valid, False otherwise
        """
        required_fields = ["id", "name", "description", "type"]
        for field in required_fields:
            if field not in pattern:
                logger.warning(f"Pattern missing required field: {field}")
                return False
        
        # Either template or cypher_template must be present
        if "template" not in pattern and "cypher_template" not in pattern:
            logger.warning("Pattern missing both template and cypher_template")
            return False
        
        return True
    
    async def _arun(
        self,
        operation: str,
        pattern_id: Optional[str] = None,
        pattern_type: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        conversion_method: str = "rule_based",
        custom_pattern: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Execute the requested pattern library operation asynchronously.
        
        Args:
            operation: Operation to perform
            pattern_id: ID of the specific pattern to use
            pattern_type: Type of pattern to search for
            parameters: Parameters to apply to the pattern template
            conversion_method: Method to use for converting patterns to Cypher
            custom_pattern: Custom pattern definition
            
        Returns:
            JSON string containing operation results
        """
        try:
            parameters = parameters or {}
            
            if operation == "list":
                return await self._list_patterns(pattern_type)
            
            elif operation == "search":
                return await self._search_patterns(pattern_type)
            
            elif operation == "convert":
                if custom_pattern:
                    return await self._convert_custom_pattern(custom_pattern, parameters, conversion_method)
                elif pattern_id:
                    if pattern_id not in self.patterns:
                        return json.dumps({
                            "success": False,
                            "error": f"Pattern not found: {pattern_id}"
                        })
                    return await self._convert_pattern(self.patterns[pattern_id], parameters, conversion_method)
                else:
                    return json.dumps({
                        "success": False,
                        "error": "Either pattern_id or custom_pattern must be provided for 'convert' operation"
                    })
            
            elif operation == "score":
                if not pattern_id or pattern_id not in self.patterns:
                    return json.dumps({
                        "success": False,
                        "error": f"Valid pattern_id must be provided for 'score' operation"
                    })
                return await self._score_pattern(self.patterns[pattern_id], parameters)
            
            else:
                return json.dumps({
                    "success": False,
                    "error": f"Unknown operation: {operation}"
                })
                
        except Exception as e:
            logger.error(f"Error in PatternLibraryTool: {e}", exc_info=True)
            return json.dumps({
                "success": False,
                "error": str(e)
            })
    
    def _run(
        self,
        operation: str,
        pattern_id: Optional[str] = None,
        pattern_type: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        conversion_method: str = "rule_based",
        custom_pattern: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Synchronous wrapper for _arun.
        
        This method exists for compatibility with synchronous CrewAI operations.
        It should not be called directly in an async context.
        """
        import asyncio
        
        # Create a new event loop if needed
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(
            self._arun(operation, pattern_id, pattern_type, parameters, conversion_method, custom_pattern)
        )
    
    async def _list_patterns(self, pattern_type: Optional[str] = None) -> str:
        """
        List available patterns, optionally filtered by type.
        
        Args:
            pattern_type: Optional type to filter patterns by
            
        Returns:
            JSON string containing matching patterns
        """
        matching_patterns = {}
        
        for pattern_id, pattern in self.patterns.items():
            if pattern_type is None or pattern.get("type") == pattern_type:
                # Include only basic pattern information
                matching_patterns[pattern_id] = {
                    "id": pattern["id"],
                    "name": pattern["name"],
                    "description": pattern["description"],
                    "type": pattern["type"],
                    "severity": pattern.get("severity", "medium")
                }
        
        return json.dumps({
            "success": True,
            "patterns": matching_patterns,
            "count": len(matching_patterns)
        })
    
    async def _search_patterns(self, pattern_type: Optional[str] = None) -> str:
        """
        Search for patterns by type or keywords.
        
        Args:
            pattern_type: Optional type to filter patterns by
            
        Returns:
            JSON string containing matching patterns
        """
        # For now, this is the same as list_patterns
        # In a more advanced implementation, this could use semantic search
        return await self._list_patterns(pattern_type)
    
    async def _convert_pattern(
        self,
        pattern: Dict[str, Any],
        parameters: Dict[str, Any],
        conversion_method: str
    ) -> str:
        """
        Convert a pattern to a Cypher query.
        
        Args:
            pattern: Pattern definition
            parameters: Parameters to apply to the pattern
            conversion_method: Conversion method to use
            
        Returns:
            JSON string containing the Cypher query
        """
        # Merge parameters with defaults
        merged_params = {}
        if "parameters" in pattern:
            for param_name, param_info in pattern["parameters"].items():
                merged_params[param_name] = parameters.get(param_name, param_info.get("default"))
        
        # Apply parameters to any other parameters
        merged_params.update(parameters)
        
        # Choose conversion method
        if conversion_method == "rule_based" and "cypher_template" in pattern:
            cypher_query = await self._rule_based_conversion(pattern, merged_params)
        elif conversion_method == "llm" or (conversion_method == "hybrid" and "cypher_template" not in pattern):
            cypher_query = await self._llm_based_conversion(pattern, merged_params)
        elif conversion_method == "hybrid":
            # Try rule-based first, fall back to LLM if it fails
            try:
                cypher_query = await self._rule_based_conversion(pattern, merged_params)
            except Exception as e:
                logger.warning(f"Rule-based conversion failed, falling back to LLM: {e}")
                cypher_query = await self._llm_based_conversion(pattern, merged_params)
        else:
            return json.dumps({
                "success": False,
                "error": f"Unsupported conversion method: {conversion_method}"
            })
        
        return json.dumps({
            "success": True,
            "pattern_id": pattern["id"],
            "pattern_name": pattern["name"],
            "cypher_query": cypher_query,
            "parameters": merged_params
        })
    
    async def _convert_custom_pattern(
        self,
        custom_pattern: Dict[str, Any],
        parameters: Dict[str, Any],
        conversion_method: str
    ) -> str:
        """
        Convert a custom pattern to a Cypher query.
        
        Args:
            custom_pattern: Custom pattern definition
            parameters: Parameters to apply to the pattern
            conversion_method: Conversion method to use
            
        Returns:
            JSON string containing the Cypher query
        """
        # Validate the custom pattern
        if not self._validate_pattern(custom_pattern):
            return json.dumps({
                "success": False,
                "error": "Invalid custom pattern"
            })
        
        # Convert using the same method as regular patterns
        return await self._convert_pattern(custom_pattern, parameters, conversion_method)
    
    async def _score_pattern(
        self,
        pattern: Dict[str, Any],
        parameters: Dict[str, Any]
    ) -> str:
        """
        Score results against a pattern.
        
        Args:
            pattern: Pattern definition
            parameters: Parameters including results to score
            
        Returns:
            JSON string containing the score
        """
        # This would typically execute the pattern query and then score the results
        # For now, we'll just return a placeholder
        return json.dumps({
            "success": True,
            "pattern_id": pattern["id"],
            "pattern_name": pattern["name"],
            "score": 0.85,  # Placeholder score
            "matches": 3,   # Placeholder match count
            "details": "Pattern scoring is a placeholder in this version"
        })
    
    async def _rule_based_conversion(
        self,
        pattern: Dict[str, Any],
        parameters: Dict[str, Any]
    ) -> str:
        """
        Convert a pattern to Cypher using rule-based conversion.
        
        Args:
            pattern: Pattern definition
            parameters: Parameters to apply to the pattern
            
        Returns:
            Cypher query string
        """
        if "cypher_template" not in pattern:
            raise ValueError("Pattern does not contain a cypher_template for rule-based conversion")
        
        cypher_template = pattern["cypher_template"]
        
        # Replace parameter placeholders in the template
        cypher_query = cypher_template
        
        # Simple string replacement for basic parameters
        for param_name, param_value in parameters.items():
            placeholder = f"{{{param_name}}}"
            if placeholder in cypher_query:
                cypher_query = cypher_query.replace(placeholder, str(param_value))
        
        # Handle special parameter conditions
        if "min_amount_condition" in cypher_query and "min_amount" in parameters:
            min_amount = parameters["min_amount"]
            min_amount_condition = f"ALL(r IN relationships(path) WHERE r.amount >= {min_amount})"
            cypher_query = cypher_query.replace("{min_amount_condition}", min_amount_condition)
        
        # Clean up the query (remove extra whitespace, etc.)
        cypher_query = "\n".join(line.strip() for line in cypher_query.split("\n") if line.strip())
        
        return cypher_query
    
    async def _llm_based_conversion(
        self,
        pattern: Dict[str, Any],
        parameters: Dict[str, Any]
    ) -> str:
        """
        Convert a pattern to Cypher using LLM-based conversion.
        
        Args:
            pattern: Pattern definition
            parameters: Parameters to apply to the pattern
            
        Returns:
            Cypher query string
        """
        # Create a prompt for the LLM to generate a Cypher query
        prompt = f"""
You are an expert in Neo4j and Cypher query language. Convert the following fraud pattern definition into an optimized Cypher query.

Pattern Name: {pattern["name"]}
Pattern Description: {pattern["description"]}
Pattern Type: {pattern["type"]}

"""
        
        # Add template information if available
        if "template" in pattern:
            prompt += "Pattern Template:\n"
            prompt += json.dumps(pattern["template"], indent=2)
            prompt += "\n\n"
        
        # Add parameters
        prompt += "Parameters:\n"
        for param_name, param_value in parameters.items():
            prompt += f"- {param_name}: {param_value}\n"
        
        prompt += """
Generate a Cypher query that:
1. Accurately represents the pattern described above
2. Uses the provided parameters
3. Is optimized for performance
4. Includes appropriate LIMIT clauses for safety
5. Returns meaningful results for analysis

Return ONLY the Cypher query without any explanation or markdown formatting.
"""
        
        # Get schema information to help the LLM
        try:
            if hasattr(self.neo4j_client, 'driver') and self.neo4j_client.driver is not None:
                schema_query = """
                CALL apoc.meta.schema() YIELD value
                RETURN value
                """
                schema_results = await self.neo4j_client.run_query(schema_query)
                if schema_results and len(schema_results) > 0:
                    prompt += "\nDatabase Schema:\n"
                    prompt += json.dumps(schema_results[0]["value"], indent=2)
            else:
                # Provide some generic schema information
                prompt += "\nTypical Financial Crime Schema:\n"
                prompt += """
                {
                  "Account": {"properties": ["id", "owner", "balance", "created_at", "status"]},
                  "Person": {"properties": ["id", "name", "dob", "nationality", "risk_score"]},
                  "Company": {"properties": ["id", "name", "incorporation_date", "jurisdiction", "status"]},
                  "Transaction": {"properties": ["id", "amount", "timestamp", "currency", "status"]},
                  "Address": {"properties": ["id", "street", "city", "country", "postal_code"]}
                }
                """
        except Exception as e:
            logger.warning(f"Error getting schema information: {e}")
        
        # Generate the Cypher query using the LLM
        cypher_query = await self.gemini_client.generate_cypher_query(
            prompt,
            schema_context=""  # Already included in the prompt
        )
        
        # Clean up the query
        cypher_query = cypher_query.strip()
        
        # Remove any markdown formatting
        if cypher_query.startswith("```cypher"):
            lines = cypher_query.split("\n")
            cypher_query = "\n".join(lines[1:-1])
        elif cypher_query.startswith("```"):
            lines = cypher_query.split("\n")
            cypher_query = "\n".join(lines[1:-1])
        
        return cypher_query
