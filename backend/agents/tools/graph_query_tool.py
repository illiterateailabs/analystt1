"""
GraphQueryTool for executing Cypher queries against Neo4j.

This tool wraps the Neo4jClient to allow CrewAI agents to execute
Cypher queries against the Neo4j graph database and receive structured
results for analysis.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Union, Tuple

from crewai_tools import BaseTool
from pydantic import BaseModel, Field

from backend.integrations.neo4j_client import Neo4jClient
from backend.core.explain_cypher import (
    CypherExplanationService,
    QuerySource,
)
from backend.core.evidence import (
    EvidenceBundle, GraphElementEvidence, EvidenceSource, 
    create_evidence_bundle
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
    create_evidence: Optional[bool] = Field(
        default=True,
        description="Whether to create evidence from query results"
    )
    evidence_description: Optional[str] = Field(
        default=None,
        description="Custom description for the evidence"
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
    
    async def _create_evidence_from_query(
        self,
        query_id: str,
        query_text: str,
        results: List[Dict[str, Any]],
        description: Optional[str] = None
    ) -> Tuple[EvidenceBundle, str]:
        """
        Creates evidence from query results.
        
        Args:
            query_id: ID of the executed query
            query_text: The Cypher query text
            results: Query results
            description: Optional description for the evidence
            
        Returns:
            Tuple of (evidence_bundle, evidence_id)
        """
        # Create description if not provided
        if not description:
            result_count = len(results)
            description = f"Evidence from Cypher query: {result_count} results found"
            if result_count > 0:
                # Add a bit more context about what was found
                sample_keys = list(results[0].keys())
                if sample_keys:
                    description += f" with fields: {', '.join(sample_keys[:5])}"
                if len(sample_keys) > 5:
                    description += f" and {len(sample_keys) - 5} more"
        
        # Create evidence bundle
        bundle = create_evidence_bundle(
            narrative=f"Results from Cypher query: {query_text}",
            metadata={
                "query_id": query_id,
                "result_count": len(results)
            }
        )
        
        # Create evidence item from query
        evidence_id = await self.explanation_service.create_evidence_from_query(
            query_id=query_id,
            description=description,
            confidence=0.9,  # High confidence for direct database results
            source=EvidenceSource.GRAPH_ANALYSIS,
            bundle=bundle
        )
        
        # Add sample results as raw data
        if results:
            # Limit to first 10 results to avoid huge bundles
            sample_results = results[:10]
            bundle.add_raw_data(
                data={"sample_results": sample_results},
                description=f"Sample of {len(sample_results)} results from query"
            )
        
        # Generate citation
        citation = await self.explanation_service.cite_query_in_response(
            query_id=query_id,
            response_text=""  # We'll just get the citation part
        )
        
        # Add citation to narrative
        bundle.narrative += f"\n\n## Query Citation\n{citation}"
        
        return bundle, evidence_id
    
    async def _arun(
        self, 
        query: str, 
        parameters: Optional[Dict[str, Any]] = None,
        limit_results: int = 1000,
        create_evidence: bool = True,
        evidence_description: Optional[str] = None
    ) -> str:
        """
        Execute a Cypher query asynchronously and return the results.
        
        Args:
            query: The Cypher query to execute
            parameters: Optional parameters for the query
            limit_results: Maximum number of results to return
            create_evidence: Whether to create evidence from query results
            evidence_description: Optional description for the evidence
            
        Returns:
            JSON string containing the query results and evidence information
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
            
            # Format response
            response = {
                "status": "success",
                "results": results,
                "count": len(results),
                "query_id": exec_record.query_id,
                "execution_id": exec_record.execution_id,
            }
            
            # Create evidence if enabled
            if create_evidence and results:
                try:
                    bundle, evidence_id = await self._create_evidence_from_query(
                        query_id=exec_record.query_id,
                        query_text=query,
                        results=results,
                        description=evidence_description
                    )
                    
                    # Add evidence information to response
                    response["evidence"] = {
                        "evidence_id": evidence_id,
                        "bundle_id": bundle.investigation_id,
                        "narrative": bundle.narrative,
                        "quality_score": bundle.calculate_overall_confidence(),
                    }
                    
                    logger.info(f"Created evidence {evidence_id} from query {exec_record.query_id}")
                except Exception as e:
                    logger.error(f"Error creating evidence from query: {e}", exc_info=True)
                    response["evidence"] = {
                        "error": f"Failed to create evidence: {str(e)}"
                    }
            
            # Return JSON response
            if not results:
                response["results"] = []
                return json.dumps(response)
            
            return json.dumps(response, default=self._json_serializer)
            
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
    
    def _run(
        self, 
        query: str, 
        parameters: Optional[Dict[str, Any]] = None,
        limit_results: int = 1000,
        create_evidence: bool = True,
        evidence_description: Optional[str] = None
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
            self._arun(query, parameters, limit_results, create_evidence, evidence_description)
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
