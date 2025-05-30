"""
Neo4jSchemaTool for retrieving database schema information from Neo4j.

This tool provides CrewAI agents with information about the Neo4j database schema,
including node labels, relationship types, properties, and constraints. This
information is essential for generating accurate Cypher queries and understanding
the data model.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Union

from crewai_tools import BaseTool
from pydantic import BaseModel, Field

from backend.integrations.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)


class SchemaQueryInput(BaseModel):
    """Input model for schema queries."""
    
    detail_level: str = Field(
        default="basic",
        description="Level of schema detail to retrieve: 'basic', 'detailed', or 'full'"
    )
    include_constraints: bool = Field(
        default=True,
        description="Whether to include constraints and indexes in the schema"
    )
    include_stats: bool = Field(
        default=False,
        description="Whether to include node and relationship counts"
    )
    specific_labels: Optional[List[str]] = Field(
        default=None,
        description="Optional list of specific node labels to get schema for"
    )


class Neo4jSchemaTool(BaseTool):
    """
    Tool for retrieving Neo4j database schema information.
    
    This tool allows agents to query the Neo4j database for schema information,
    including node labels, relationship types, properties, and constraints. This
    information is essential for generating accurate Cypher queries and
    understanding the data model.
    """
    
    name: str = "neo4j_schema_tool"
    description: str = """
    Retrieve Neo4j database schema information.
    
    Use this tool when you need to:
    - Understand the data model before generating Cypher queries
    - Get information about available node labels and their properties
    - Get information about relationship types and their properties
    - Check for constraints and indexes in the database
    - Get statistics about the data (counts of nodes and relationships)
    
    The tool provides different levels of detail to suit your needs.
    """
    args_schema: type[BaseModel] = SchemaQueryInput
    
    def __init__(self, neo4j_client: Optional[Neo4jClient] = None):
        """
        Initialize the Neo4jSchemaTool.
        
        Args:
            neo4j_client: Optional Neo4jClient instance. If not provided,
                         a new client will be created.
        """
        super().__init__()
        self.neo4j_client = neo4j_client or Neo4jClient()
    
    async def _arun(
        self,
        detail_level: str = "basic",
        include_constraints: bool = True,
        include_stats: bool = False,
        specific_labels: Optional[List[str]] = None
    ) -> str:
        """
        Retrieve Neo4j schema information asynchronously.
        
        Args:
            detail_level: Level of schema detail to retrieve
            include_constraints: Whether to include constraints and indexes
            include_stats: Whether to include node and relationship counts
            specific_labels: Optional list of specific node labels to get schema for
            
        Returns:
            JSON string containing the schema information
        """
        try:
            # Ensure the client is connected
            if not hasattr(self.neo4j_client, 'driver') or self.neo4j_client.driver is None:
                await self.neo4j_client.connect()
                logger.info("Connected to Neo4j database")
            
            # Get basic schema information
            schema = {}
            
            # Get node labels and their properties
            if detail_level in ["basic", "detailed", "full"]:
                labels_query = """
                CALL db.labels() YIELD label
                RETURN collect(label) AS labels
                """
                labels_result = await self.neo4j_client.run_query(labels_query)
                labels = labels_result[0]["labels"] if labels_result else []
                
                # Filter labels if specific ones are requested
                if specific_labels:
                    labels = [label for label in labels if label in specific_labels]
                
                schema["node_labels"] = labels
                
                # Get properties for each label
                if detail_level in ["detailed", "full"]:
                    schema["node_properties"] = {}
                    for label in labels:
                        properties_query = f"""
                        MATCH (n:{label})
                        UNWIND keys(n) AS property
                        RETURN collect(DISTINCT property) AS properties
                        """
                        properties_result = await self.neo4j_client.run_query(properties_query)
                        properties = properties_result[0]["properties"] if properties_result else []
                        schema["node_properties"][label] = properties
            
            # Get relationship types and their properties
            if detail_level in ["basic", "detailed", "full"]:
                rel_types_query = """
                CALL db.relationshipTypes() YIELD relationshipType
                RETURN collect(relationshipType) AS relationshipTypes
                """
                rel_types_result = await self.neo4j_client.run_query(rel_types_query)
                rel_types = rel_types_result[0]["relationshipTypes"] if rel_types_result else []
                
                schema["relationship_types"] = rel_types
                
                # Get properties for each relationship type
                if detail_level in ["detailed", "full"]:
                    schema["relationship_properties"] = {}
                    for rel_type in rel_types:
                        properties_query = f"""
                        MATCH ()-[r:{rel_type}]->()
                        UNWIND keys(r) AS property
                        RETURN collect(DISTINCT property) AS properties
                        """
                        properties_result = await self.neo4j_client.run_query(properties_query)
                        properties = properties_result[0]["properties"] if properties_result else []
                        schema["relationship_properties"][rel_type] = properties
            
            # Get detailed schema information
            if detail_level == "full":
                # Get node label combinations
                combinations_query = """
                MATCH (n)
                WITH labels(n) AS labels
                RETURN collect(DISTINCT labels) AS label_combinations
                """
                combinations_result = await self.neo4j_client.run_query(combinations_query)
                combinations = combinations_result[0]["label_combinations"] if combinations_result else []
                schema["label_combinations"] = combinations
                
                # Get relationship patterns
                patterns_query = """
                MATCH (a)-[r]->(b)
                RETURN 
                    labels(a) AS source_labels,
                    type(r) AS relationship_type,
                    labels(b) AS target_labels,
                    count(*) AS frequency
                ORDER BY frequency DESC
                LIMIT 100
                """
                patterns_result = await self.neo4j_client.run_query(patterns_query)
                schema["relationship_patterns"] = patterns_result
            
            # Get constraints and indexes
            if include_constraints:
                if detail_level in ["detailed", "full"]:
                    # Get constraints
                    constraints_query = """
                    SHOW CONSTRAINTS
                    """
                    try:
                        constraints_result = await self.neo4j_client.run_query(constraints_query)
                        schema["constraints"] = constraints_result
                    except Exception as e:
                        logger.warning(f"Error getting constraints: {e}")
                        schema["constraints"] = []
                    
                    # Get indexes
                    indexes_query = """
                    SHOW INDEXES
                    """
                    try:
                        indexes_result = await self.neo4j_client.run_query(indexes_query)
                        schema["indexes"] = indexes_result
                    except Exception as e:
                        logger.warning(f"Error getting indexes: {e}")
                        schema["indexes"] = []
            
            # Get statistics
            if include_stats:
                # Get node counts
                node_counts_query = """
                MATCH (n)
                RETURN labels(n) AS labels, count(*) AS count
                """
                node_counts_result = await self.neo4j_client.run_query(node_counts_query)
                schema["node_counts"] = node_counts_result
                
                # Get relationship counts
                rel_counts_query = """
                MATCH ()-[r]->()
                RETURN type(r) AS type, count(*) AS count
                """
                rel_counts_result = await self.neo4j_client.run_query(rel_counts_query)
                schema["relationship_counts"] = rel_counts_result
            
            return json.dumps({
                "success": True,
                "schema": schema,
                "detail_level": detail_level
            }, default=str)
            
        except Exception as e:
            logger.error(f"Error retrieving Neo4j schema: {e}", exc_info=True)
            return json.dumps({
                "success": False,
                "error": str(e)
            })
    
    def _run(
        self,
        detail_level: str = "basic",
        include_constraints: bool = True,
        include_stats: bool = False,
        specific_labels: Optional[List[str]] = None
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
            self._arun(detail_level, include_constraints, include_stats, specific_labels)
        )
