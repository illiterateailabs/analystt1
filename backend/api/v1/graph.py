"""Graph database API endpoints for Neo4j operations."""

import logging
from typing import Dict, List, Optional, Any, Union
from fastapi import APIRouter, HTTPException, Depends, Request
from pydantic import BaseModel, Field

from backend.integrations.neo4j_client import Neo4jClient
from backend.integrations.gemini_client import GeminiClient


logger = logging.getLogger(__name__)
router = APIRouter()


# Request/Response Models
class CypherQueryRequest(BaseModel):
    query: str = Field(..., description="Cypher query to execute")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Query parameters")


class NaturalLanguageQueryRequest(BaseModel):
    question: str = Field(..., description="Natural language question")
    context: Optional[str] = Field(None, description="Additional context")


class NodeCreationRequest(BaseModel):
    labels: Union[str, List[str]] = Field(..., description="Node labels")
    properties: Dict[str, Any] = Field(..., description="Node properties")


class RelationshipCreationRequest(BaseModel):
    from_node_id: int = Field(..., description="Source node ID")
    to_node_id: int = Field(..., description="Target node ID")
    relationship_type: str = Field(..., description="Relationship type")
    properties: Optional[Dict[str, Any]] = Field(None, description="Relationship properties")


class GraphSearchRequest(BaseModel):
    labels: Optional[Union[str, List[str]]] = Field(None, description="Node labels to search")
    properties: Optional[Dict[str, Any]] = Field(None, description="Properties to match")
    limit: int = Field(100, description="Maximum number of results")


class GraphResponse(BaseModel):
    success: bool = Field(..., description="Operation success status")
    data: Optional[List[Dict[str, Any]]] = Field(None, description="Query results")
    cypher_query: Optional[str] = Field(None, description="Generated Cypher query")
    explanation: Optional[str] = Field(None, description="AI explanation of results")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


# Dependency functions
async def get_neo4j_client(request: Request) -> Neo4jClient:
    return request.app.state.neo4j


async def get_gemini_client(request: Request) -> GeminiClient:
    return request.app.state.gemini


@router.get("/schema", response_model=Dict[str, Any])
async def get_graph_schema(neo4j: Neo4jClient = Depends(get_neo4j_client)):
    """Get the current graph database schema."""
    try:
        schema_info = await neo4j.get_schema_info()
        return {
            "success": True,
            "schema": schema_info
        }
    except Exception as e:
        logger.error(f"Error getting graph schema: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query/cypher", response_model=GraphResponse)
async def execute_cypher_query(
    request: CypherQueryRequest,
    neo4j: Neo4jClient = Depends(get_neo4j_client)
):
    """Execute a raw Cypher query."""
    try:
        logger.info(f"Executing Cypher query: {request.query[:100]}...")
        
        results = await neo4j.execute_query(request.query, request.parameters)
        
        return GraphResponse(
            success=True,
            data=results,
            cypher_query=request.query,
            metadata={"result_count": len(results)}
        )
        
    except Exception as e:
        logger.error(f"Error executing Cypher query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query/natural", response_model=GraphResponse)
async def natural_language_query(
    request: NaturalLanguageQueryRequest,
    neo4j: Neo4jClient = Depends(get_neo4j_client),
    gemini: GeminiClient = Depends(get_gemini_client)
):
    """Convert natural language to Cypher and execute."""
    try:
        logger.info(f"Processing natural language query: {request.question[:100]}...")
        
        # Get schema context
        schema_info = await neo4j.get_schema_info()
        schema_context = f"""
Graph Database Schema:
- Node Labels: {', '.join(schema_info['labels'])}
- Relationship Types: {', '.join(schema_info['relationship_types'])}
- Property Keys: {', '.join(schema_info['property_keys'])}
- Total Nodes: {schema_info['node_count']}
- Total Relationships: {schema_info['relationship_count']}
"""
        
        # Add examples for better Cypher generation
        examples = [
            {
                "question": "Find all people",
                "cypher": "MATCH (p:Person) RETURN p LIMIT 100"
            },
            {
                "question": "Show transactions over $10,000",
                "cypher": "MATCH (t:Transaction) WHERE t.amount > 10000 RETURN t"
            },
            {
                "question": "Find connections between two people",
                "cypher": "MATCH (p1:Person)-[r]-(p2:Person) RETURN p1, r, p2 LIMIT 50"
            }
        ]
        
        # Generate Cypher query
        cypher_query = await gemini.generate_cypher_query(
            request.question,
            schema_context,
            examples=examples
        )
        
        # Execute the generated query
        results = await neo4j.execute_query(cypher_query)
        
        # Generate explanation
        explanation = await gemini.explain_results(
            request.question,
            results,
            context=request.context
        )
        
        return GraphResponse(
            success=True,
            data=results,
            cypher_query=cypher_query,
            explanation=explanation,
            metadata={"result_count": len(results)}
        )
        
    except Exception as e:
        logger.error(f"Error processing natural language query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/nodes", response_model=GraphResponse)
async def create_node(
    request: NodeCreationRequest,
    neo4j: Neo4jClient = Depends(get_neo4j_client)
):
    """Create a new node in the graph."""
    try:
        logger.info(f"Creating node with labels: {request.labels}")
        
        node = await neo4j.create_node(request.labels, request.properties)
        
        return GraphResponse(
            success=True,
            data=[node],
            metadata={"operation": "create_node", "labels": request.labels}
        )
        
    except Exception as e:
        logger.error(f"Error creating node: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/relationships", response_model=GraphResponse)
async def create_relationship(
    request: RelationshipCreationRequest,
    neo4j: Neo4jClient = Depends(get_neo4j_client)
):
    """Create a new relationship between nodes."""
    try:
        logger.info(f"Creating relationship: {request.from_node_id} -> {request.to_node_id}")
        
        relationship = await neo4j.create_relationship(
            request.from_node_id,
            request.to_node_id,
            request.relationship_type,
            request.properties
        )
        
        return GraphResponse(
            success=True,
            data=[relationship],
            metadata={
                "operation": "create_relationship",
                "type": request.relationship_type
            }
        )
        
    except Exception as e:
        logger.error(f"Error creating relationship: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search", response_model=GraphResponse)
async def search_nodes(
    request: GraphSearchRequest,
    neo4j: Neo4jClient = Depends(get_neo4j_client)
):
    """Search for nodes matching criteria."""
    try:
        logger.info(f"Searching nodes with labels: {request.labels}")
        
        nodes = await neo4j.find_nodes(
            request.labels,
            request.properties,
            request.limit
        )
        
        return GraphResponse(
            success=True,
            data=nodes,
            metadata={
                "operation": "search_nodes",
                "result_count": len(nodes),
                "limit": request.limit
            }
        )
        
    except Exception as e:
        logger.error(f"Error searching nodes: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analytics/centrality")
async def calculate_centrality(
    algorithm: str = "pagerank",
    limit: int = 20,
    neo4j: Neo4jClient = Depends(get_neo4j_client)
):
    """Calculate centrality metrics using Neo4j GDS."""
    try:
        logger.info(f"Calculating {algorithm} centrality")
        
        # Map algorithm names to GDS procedures
        gds_algorithms = {
            "pagerank": "gds.pageRank.stream",
            "betweenness": "gds.betweenness.stream",
            "degree": "gds.degree.stream",
            "closeness": "gds.closeness.stream"
        }
        
        if algorithm not in gds_algorithms:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported algorithm: {algorithm}"
            )
        
        # Create a graph projection (simplified)
        projection_query = """
        CALL gds.graph.project.cypher(
            'centrality-graph',
            'MATCH (n) RETURN id(n) AS id, labels(n) AS labels',
            'MATCH (a)-[r]->(b) RETURN id(a) AS source, id(b) AS target, type(r) AS type'
        )
        """
        
        try:
            await neo4j.execute_query(projection_query)
        except:
            # Graph might already exist
            pass
        
        # Run centrality algorithm
        centrality_query = f"""
        CALL {gds_algorithms[algorithm]}('centrality-graph')
        YIELD nodeId, score
        MATCH (n) WHERE id(n) = nodeId
        RETURN n, score
        ORDER BY score DESC
        LIMIT {limit}
        """
        
        results = await neo4j.execute_query(centrality_query)
        
        # Cleanup graph projection
        cleanup_query = "CALL gds.graph.drop('centrality-graph')"
        try:
            await neo4j.execute_query(cleanup_query)
        except:
            pass
        
        return {
            "success": True,
            "algorithm": algorithm,
            "results": results,
            "metadata": {"result_count": len(results)}
        }
        
    except Exception as e:
        logger.error(f"Error calculating centrality: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analytics/communities")
async def detect_communities(
    algorithm: str = "louvain",
    neo4j: Neo4jClient = Depends(get_neo4j_client)
):
    """Detect communities in the graph using GDS algorithms."""
    try:
        logger.info(f"Detecting communities using {algorithm}")
        
        # Community detection algorithms
        community_algorithms = {
            "louvain": "gds.louvain.stream",
            "wcc": "gds.wcc.stream",
            "lpa": "gds.labelPropagation.stream"
        }
        
        if algorithm not in community_algorithms:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported algorithm: {algorithm}"
            )
        
        # Create graph projection
        projection_query = """
        CALL gds.graph.project.cypher(
            'community-graph',
            'MATCH (n) RETURN id(n) AS id',
            'MATCH (a)-[r]-(b) RETURN id(a) AS source, id(b) AS target'
        )
        """
        
        try:
            await neo4j.execute_query(projection_query)
        except:
            pass
        
        # Run community detection
        community_query = f"""
        CALL {community_algorithms[algorithm]}('community-graph')
        YIELD nodeId, communityId
        MATCH (n) WHERE id(n) = nodeId
        RETURN communityId, collect(n) as members, count(n) as size
        ORDER BY size DESC
        """
        
        results = await neo4j.execute_query(community_query)
        
        # Cleanup
        try:
            await neo4j.execute_query("CALL gds.graph.drop('community-graph')")
        except:
            pass
        
        return {
            "success": True,
            "algorithm": algorithm,
            "communities": results,
            "metadata": {"community_count": len(results)}
        }
        
    except Exception as e:
        logger.error(f"Error detecting communities: {e}")
        raise HTTPException(status_code=500, detail=str(e))
