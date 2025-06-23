"""
GraphQueryTool for executing Cypher queries against Neo4j.

This tool wraps the Neo4jClient to allow CrewAI agents to execute
Cypher queries against the Neo4j graph database and receive structured
results for analysis.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Union

from crewai_tools import BaseTool
from pydantic import BaseModel, Field

from backend.integrations.neo4j_client import Neo4jClient
from backend.core.explain_cypher import (
    CypherExplanationService,
    QuerySource,
)

logger = logging.getLogger(__name__)


class CypherQueryInput(BaseModel):
    """Input model for Cypher queries."""
    
    query: str = Field(
        ...,
        description="The Cypher query to execute against Neo4j"
    )
    parameters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional parameters for the Cypher query"
    )
    limit_results: Optional[int] = Field(
        default=1000,
        description="Maximum number of results to return (for safety)"
    )


class GraphQueryTool(BaseTool):
    """
    Tool for executing Cypher queries against Neo4j.
    
    This tool allows agents to run Cypher queries against the Neo4j database
    and receive structured results for analysis. It handles query execution,
    error handling, and result formatting.
    """
    
    name: str = "graph_query_tool"
    description: str = """
    Execute Cypher queries against the Neo4j graph database.
    
    Use this tool when you need to:
    - Retrieve data from the graph database
    - Run graph algorithms and analytics
    - Check for specific patterns or relationships
    - Count or aggregate graph data
    
    The query should be valid Cypher syntax. You can use parameters for 
    better security and performance.
    
    Examples:
    - MATCH (n:Person) RETURN n.name, n.age LIMIT 10
    - MATCH (a:Account)-[t:TRANSACTION]->(b:Account) WHERE t.amount > 10000 RETURN a, t, b
    - MATCH p=shortestPath((a:Entity {id: $id})-[*1..5]-(b:Entity)) RETURN p
    """
    args_schema: type[BaseModel] = CypherQueryInput
    
    def __init__(self, neo4j_client: Optional[Neo4jClient] = None):
        """
        Initialize the GraphQueryTool.
        
        Args:
            neo4j_client: Optional Neo4jClient instance. If not provided,
                         a new client will be created.
        """
        super().__init__()
        self.neo4j_client = neo4j_client or Neo4jClient()
        # CypherExplanationService handles provenance, caching & evidence
        self.explanation_service: CypherExplanationService = CypherExplanationService()
    
    async def _arun(self, query: str, parameters: Optional[Dict[str, Any]] = None, 
                   limit_results: int = 1000) -> str:
        """
        Execute a Cypher query asynchronously and return the results.
        
        Args:
            query: The Cypher query to execute
            parameters: Optional parameters for the query
            limit_results: Maximum number of results to return
            
        Returns:
            JSON string containing the query results
        """
        try:
            # Ensure the client is connected
            if not hasattr(self.neo4j_client, 'driver') or self.neo4j_client.driver is None:
                await self.neo4j_client.connect()
                logger.info("Connected to Neo4j database")
            
            # Add safety LIMIT if not present in the query
            if "LIMIT" not in query.upper() and limit_results > 0:
                query = f"{query} LIMIT {limit_results}"
            
            # Execute & track via CypherExplanationService
            logger.info(f"Executing Cypher query via ExplanationService: {query}")
            parameters = parameters or {}
            results, exec_record = await self.explanation_service.execute_and_track_query(
                query_text=query,
                parameters=parameters,
                source=QuerySource.TOOL_GENERATED,
                generated_by="GraphQueryTool",
            )
            
            # Format and return results
            if not results:
                return json.dumps(
                    {
                        "status": "success",
                        "results": [],
                        "count": 0,
                        "query_id": exec_record.query_id,
                        "execution_id": exec_record.execution_id,
                    }
                )
            
            return json.dumps({
                "status": "success",
                "results": results,
                "count": len(results),
                "query_id": exec_record.query_id,
                "execution_id": exec_record.execution_id,
            }, default=self._json_serializer)
            
        except Exception as e:
            logger.error(f"Error executing Cypher query: {e}", exc_info=True)
            error_message = str(e)
            
            # Provide helpful error messages for common issues
            if "SyntaxError" in error_message:
                error_message = f"Cypher syntax error: {error_message}"
            elif "ConstraintValidationFailed" in error_message:
                error_message = f"Constraint violation: {error_message}"
            elif "EntityNotFound" in error_message:
                error_message = f"Entity not found: {error_message}"
            
            return json.dumps({
                "status": "error",
                "error": error_message,
                "query": query
            })
    
    def _run(self, query: str, parameters: Optional[Dict[str, Any]] = None,
            limit_results: int = 1000) -> str:
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
            self._arun(query, parameters, limit_results)
        )
    
    def _json_serializer(self, obj: Any) -> Any:
        """
        Custom JSON serializer to handle Neo4j-specific types.
        
        Args:
            obj: The object to serialize
            
        Returns:
            JSON-serializable representation of the object
        """
        # Handle Neo4j spatial types
        if hasattr(obj, 'x') and hasattr(obj, 'y') and hasattr(obj, 'z'):
            return {"x": obj.x, "y": obj.y, "z": obj.z}
        
        # Handle Neo4j temporal types
        if hasattr(obj, 'year') and hasattr(obj, 'month') and hasattr(obj, 'day'):
            if hasattr(obj, 'hour') and hasattr(obj, 'minute'):
                return f"{obj.year}-{obj.month:02d}-{obj.day:02d}T{obj.hour:02d}:{obj.minute:02d}:{obj.second:02d}"
            return f"{obj.year}-{obj.month:02d}-{obj.day:02d}"
        
        # Handle Neo4j Node objects
        if hasattr(obj, 'id') and hasattr(obj, 'labels') and hasattr(obj, 'items'):
            return {
                "_id": obj.id,
                "_labels": list(obj.labels),
                **dict(obj.items())
            }
        
        # Handle Neo4j Relationship objects
        if hasattr(obj, 'id') and hasattr(obj, 'type') and hasattr(obj, 'start_node') and hasattr(obj, 'end_node'):
            return {
                "_id": obj.id,
                "_type": obj.type,
                "_start_node": obj.start_node.id,
                "_end_node": obj.end_node.id,
                **dict(obj.items())
            }
        
        # Default handling for other types
        if hasattr(obj, '__dict__'):
            return obj.__dict__
        
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
