"""
Advanced Graph Algorithms API - Endpoints for advanced graph analysis

This module provides FastAPI endpoints for advanced graph algorithms including:
- Community detection using Neo4j GDS
- Risk propagation through transaction networks
- Enhanced Graph Attention Network (GAT) analysis
- Centrality and structural analysis

These endpoints expose the functionality of the advanced_graph_tool to the API,
enabling sophisticated graph-based fraud detection and network analysis.
"""

import logging
from typing import List, Dict, Any, Optional, Union
from enum import Enum

from fastapi import APIRouter, Depends, HTTPException, Query, Body, Path
from pydantic import BaseModel, Field, validator

from backend.auth.dependencies import get_current_user, RoleChecker
from backend.auth.rbac import Role
from backend.integrations.neo4j_client import Neo4jClient, get_neo4j_client
from backend.agents.tools.advanced_graph_tool import (
    AdvancedGraphTool, 
    CommunityAlgorithm, 
    get_advanced_graph_tool
)
from backend.core.logging import get_logger
from backend.core.metrics import REGISTRY, Counter
from backend.core.evidence import EvidenceBundle

# Configure logger
logger = get_logger(__name__)

# Create router
router = APIRouter(prefix="/advanced-graph", tags=["Advanced Graph"])

# Metrics
ADVANCED_GRAPH_API_REQUESTS = Counter(
    "advanced_graph_api_requests_total",
    "Total number of advanced graph API requests",
    ["endpoint", "status"]
)

# Role checker for analyst access
analyst_role = RoleChecker([Role.ANALYST, Role.ADMIN])


# Request and Response Models
class CommunityDetectionRequest(BaseModel):
    """Request model for community detection"""
    node_type: str = Field("Wallet", description="Type of nodes to analyze")
    relationship_types: Optional[List[str]] = Field(
        None, 
        description="Types of relationships to include (defaults to all)"
    )
    algorithm: CommunityAlgorithm = Field(
        CommunityAlgorithm.LOUVAIN, 
        description="Community detection algorithm to use"
    )
    store_results: bool = Field(
        True, 
        description="Whether to store community IDs in the graph"
    )
    community_property: str = Field(
        "community_id", 
        description="Property name for storing community ID"
    )
    weight_property: Optional[str] = Field(
        "amount_usd", 
        description="Edge property to use as weight"
    )
    min_community_size: int = Field(
        3, 
        description="Minimum nodes for a valid community",
        ge=2
    )
    max_communities: int = Field(
        10, 
        description="Maximum communities to return",
        ge=1, 
        le=100
    )
    tenant_id: Optional[str] = Field(
        None, 
        description="Tenant ID for multi-tenant deployments"
    )


class CommunityAnalysisRequest(BaseModel):
    """Request model for community analysis"""
    community_id: Union[int, str] = Field(..., description="ID of the community to analyze")
    community_property: str = Field(
        "community_id", 
        description="Property name storing community ID"
    )
    node_type: str = Field("Wallet", description="Type of nodes in the community")
    include_centrality: bool = Field(
        True, 
        description="Whether to compute centrality metrics"
    )
    tenant_id: Optional[str] = Field(
        None, 
        description="Tenant ID for multi-tenant deployments"
    )


class RiskPropagationRequest(BaseModel):
    """Request model for risk propagation"""
    seed_nodes: List[str] = Field(..., description="List of node IDs to start propagation from")
    node_type: str = Field("Wallet", description="Type of nodes to propagate risk to")
    risk_property: str = Field(
        "risk_score", 
        description="Node property containing risk score"
    )
    weight_property: Optional[str] = Field(
        "amount_usd", 
        description="Edge property for weighted propagation"
    )
    direction: str = Field(
        "both", 
        description="Direction of propagation ('both', 'in', 'out')"
    )
    decay_factor: float = Field(
        0.5, 
        description="Risk decay per hop (0-1)",
        ge=0.0, 
        le=1.0
    )
    max_hops: int = Field(
        3, 
        description="Maximum propagation distance",
        ge=1, 
        le=10
    )
    risk_threshold: float = Field(
        0.7, 
        description="Threshold for high-risk nodes",
        ge=0.0, 
        le=1.0
    )
    update_scores: bool = Field(
        False, 
        description="Whether to update risk scores in the graph"
    )
    update_property: Optional[str] = Field(
        "propagated_risk_score", 
        description="Property to store propagated risk"
    )
    update_threshold: Optional[float] = Field(
        0.5, 
        description="Minimum risk delta to update",
        ge=0.0
    )
    tenant_id: Optional[str] = Field(
        None, 
        description="Tenant ID for multi-tenant deployments"
    )
    
    @validator('direction')
    def validate_direction(cls, v):
        valid_directions = ['both', 'in', 'out']
        if v not in valid_directions:
            raise ValueError(f"Direction must be one of: {', '.join(valid_directions)}")
        return v


class EnhancedGNNRequest(BaseModel):
    """Request model for enhanced GNN analysis"""
    node_ids: List[str] = Field(..., description="List of node IDs to analyze")
    node_type: str = Field("Wallet", description="Type of nodes to analyze")
    n_hops: int = Field(
        2, 
        description="Number of hops to include in subgraph",
        ge=1, 
        le=5
    )
    model_version: Optional[str] = Field(
        "latest", 
        description="Model version to use"
    )
    include_explanation: bool = Field(
        True, 
        description="Whether to include feature importance"
    )
    tenant_id: Optional[str] = Field(
        None, 
        description="Tenant ID for multi-tenant deployments"
    )


class GenerateEvidenceRequest(BaseModel):
    """Request model for evidence generation"""
    operation_results: Dict[str, Any] = Field(..., description="Results from graph operation")
    operation_type: str = Field(..., description="Type of operation performed")
    title: str = Field(..., description="Title for the evidence bundle")
    description: str = Field(..., description="Description of the evidence")
    cypher_query: Optional[str] = Field(None, description="Cypher query for reproducibility")
    visualization_data: Optional[Dict[str, Any]] = Field(
        None, 
        description="Data for visualization"
    )


class AdvancedGraphResponse(BaseModel):
    """Base response model for advanced graph operations"""
    status: str = Field("success", description="Status of the operation")
    operation: str = Field(..., description="Operation performed")
    execution_time_seconds: Optional[float] = Field(None, description="Execution time in seconds")
    result: Dict[str, Any] = Field(..., description="Operation result")


@router.post(
    "/community/detect",
    response_model=AdvancedGraphResponse,
    summary="Detect communities in the graph",
    description="Detect communities of related entities using algorithms like Louvain or Label Propagation"
)
async def detect_communities(
    request: CommunityDetectionRequest,
    current_user: Dict = Depends(get_current_user),
    role_check: bool = Depends(analyst_role),
    neo4j_client: Neo4jClient = Depends(get_neo4j_client)
):
    """
    Detect communities in the graph using Neo4j GDS algorithms
    
    This endpoint runs community detection algorithms to find clusters of related entities
    in the graph. It can use different algorithms like Louvain or Label Propagation and
    supports storing the results back to the graph.
    """
    try:
        # Create tool
        tool = AdvancedGraphTool(neo4j_client)
        
        # Run community detection
        result = tool._detect_communities(
            node_type=request.node_type,
            relationship_types=request.relationship_types,
            algorithm=request.algorithm.value,
            store_results=request.store_results,
            community_property=request.community_property,
            weight_property=request.weight_property,
            min_community_size=request.min_community_size
        )
        
        # Record metrics
        ADVANCED_GRAPH_API_REQUESTS.labels(
            endpoint="detect_communities",
            status="success"
        ).inc()
        
        return AdvancedGraphResponse(
            status="success",
            operation="community_detection",
            execution_time_seconds=result.get("execution_time_seconds"),
            result=result
        )
        
    except Exception as e:
        logger.error(f"Error in community detection: {str(e)}")
        ADVANCED_GRAPH_API_REQUESTS.labels(
            endpoint="detect_communities",
            status="error"
        ).inc()
        raise HTTPException(status_code=500, detail=f"Error in community detection: {str(e)}")


@router.post(
    "/community/analyze",
    response_model=AdvancedGraphResponse,
    summary="Analyze a specific community",
    description="Get detailed analysis of a community including members, relationships, and centrality metrics"
)
async def analyze_community(
    request: CommunityAnalysisRequest,
    current_user: Dict = Depends(get_current_user),
    role_check: bool = Depends(analyst_role),
    neo4j_client: Neo4jClient = Depends(get_neo4j_client)
):
    """
    Analyze a specific community in detail
    
    This endpoint provides detailed analysis of a community, including its members,
    internal relationships, and centrality metrics. It helps understand the structure
    and key entities within a community.
    """
    try:
        # Create tool
        tool = AdvancedGraphTool(neo4j_client)
        
        # Run community analysis
        result = tool._analyze_community(
            community_id=request.community_id,
            community_property=request.community_property,
            node_type=request.node_type,
            include_centrality=request.include_centrality
        )
        
        # Record metrics
        ADVANCED_GRAPH_API_REQUESTS.labels(
            endpoint="analyze_community",
            status="success"
        ).inc()
        
        return AdvancedGraphResponse(
            status="success",
            operation="community_analysis",
            execution_time_seconds=result.get("execution_time_seconds"),
            result=result
        )
        
    except Exception as e:
        logger.error(f"Error in community analysis: {str(e)}")
        ADVANCED_GRAPH_API_REQUESTS.labels(
            endpoint="analyze_community",
            status="error"
        ).inc()
        raise HTTPException(status_code=500, detail=f"Error in community analysis: {str(e)}")


@router.post(
    "/risk/propagate",
    response_model=AdvancedGraphResponse,
    summary="Propagate risk through the graph",
    description="Spread risk scores from seed nodes through the network to find hidden high-risk entities"
)
async def propagate_risk(
    request: RiskPropagationRequest,
    current_user: Dict = Depends(get_current_user),
    role_check: bool = Depends(analyst_role),
    neo4j_client: Neo4jClient = Depends(get_neo4j_client)
):
    """
    Propagate risk from seed nodes through the graph
    
    This endpoint runs a risk propagation algorithm that spreads risk scores from
    seed nodes through the network. It helps identify hidden high-risk entities
    that are connected to known risky nodes.
    """
    try:
        # Create tool
        tool = AdvancedGraphTool(neo4j_client)
        
        # Run risk propagation
        result = tool._propagate_risk(
            seed_nodes=request.seed_nodes,
            node_type=request.node_type,
            risk_property=request.risk_property,
            weight_property=request.weight_property,
            direction=request.direction,
            decay_factor=request.decay_factor,
            max_hops=request.max_hops,
            risk_threshold=request.risk_threshold,
            update_scores=request.update_scores,
            update_property=request.update_property,
            update_threshold=request.update_threshold
        )
        
        # Record metrics
        ADVANCED_GRAPH_API_REQUESTS.labels(
            endpoint="propagate_risk",
            status="success"
        ).inc()
        
        return AdvancedGraphResponse(
            status="success",
            operation="risk_propagation",
            execution_time_seconds=result.get("execution_time_seconds"),
            result=result
        )
        
    except Exception as e:
        logger.error(f"Error in risk propagation: {str(e)}")
        ADVANCED_GRAPH_API_REQUESTS.labels(
            endpoint="propagate_risk",
            status="error"
        ).inc()
        raise HTTPException(status_code=500, detail=f"Error in risk propagation: {str(e)}")


@router.post(
    "/gnn/analyze",
    response_model=AdvancedGraphResponse,
    summary="Analyze entities using enhanced GNN",
    description="Use Graph Attention Networks to analyze entities and predict fraud risk"
)
async def enhanced_gnn_analysis(
    request: EnhancedGNNRequest,
    current_user: Dict = Depends(get_current_user),
    role_check: bool = Depends(analyst_role),
    neo4j_client: Neo4jClient = Depends(get_neo4j_client)
):
    """
    Analyze entities using enhanced Graph Neural Networks
    
    This endpoint uses advanced Graph Attention Networks (GAT) to analyze entities
    and predict fraud risk. It extracts features from the graph and uses a trained
    GAT model to make predictions.
    """
    try:
        # Create tool
        tool = AdvancedGraphTool(neo4j_client)
        
        # For now, this is a placeholder as the enhanced GNN prediction
        # functionality is still being implemented
        result = {
            "message": "Enhanced GNN analysis is being implemented",
            "node_ids": request.node_ids,
            "node_type": request.node_type,
            "n_hops": request.n_hops,
            "model_version": request.model_version,
            "status": "pending"
        }
        
        # Record metrics
        ADVANCED_GRAPH_API_REQUESTS.labels(
            endpoint="enhanced_gnn_analysis",
            status="success"
        ).inc()
        
        return AdvancedGraphResponse(
            status="success",
            operation="enhanced_gnn_analysis",
            result=result
        )
        
    except Exception as e:
        logger.error(f"Error in enhanced GNN analysis: {str(e)}")
        ADVANCED_GRAPH_API_REQUESTS.labels(
            endpoint="enhanced_gnn_analysis",
            status="error"
        ).inc()
        raise HTTPException(status_code=500, detail=f"Error in enhanced GNN analysis: {str(e)}")


@router.post(
    "/evidence/generate",
    response_model=Dict[str, Any],
    summary="Generate evidence bundle from graph analysis",
    description="Create a structured evidence bundle from graph analysis results"
)
async def generate_evidence(
    request: GenerateEvidenceRequest,
    current_user: Dict = Depends(get_current_user),
    role_check: bool = Depends(analyst_role),
    neo4j_client: Neo4jClient = Depends(get_neo4j_client)
):
    """
    Generate evidence bundle from graph analysis results
    
    This endpoint creates a structured evidence bundle from graph analysis results.
    It includes the analysis results, any Cypher queries used, and visualization data.
    The evidence bundle can be used for reporting and audit purposes.
    """
    try:
        # Create tool
        tool = AdvancedGraphTool(neo4j_client)
        
        # Generate evidence
        result = tool._generate_evidence(
            operation_results=request.operation_results,
            operation_type=request.operation_type,
            title=request.title,
            description=request.description,
            cypher_query=request.cypher_query,
            visualization_data=request.visualization_data
        )
        
        # Record metrics
        ADVANCED_GRAPH_API_REQUESTS.labels(
            endpoint="generate_evidence",
            status="success"
        ).inc()
        
        return result
        
    except Exception as e:
        logger.error(f"Error generating evidence: {str(e)}")
        ADVANCED_GRAPH_API_REQUESTS.labels(
            endpoint="generate_evidence",
            status="error"
        ).inc()
        raise HTTPException(status_code=500, detail=f"Error generating evidence: {str(e)}")


@router.get(
    "/algorithms",
    response_model=Dict[str, List[str]],
    summary="List available graph algorithms",
    description="Get a list of available advanced graph algorithms"
)
async def list_algorithms(
    current_user: Dict = Depends(get_current_user)
):
    """
    List available advanced graph algorithms
    
    This endpoint returns a list of available advanced graph algorithms,
    including community detection algorithms, centrality algorithms,
    and other graph analysis capabilities.
    """
    try:
        # List available algorithms
        algorithms = {
            "community_detection": [algo.value for algo in CommunityAlgorithm],
            "risk_propagation": ["pagerank_based", "heat_diffusion"],
            "centrality": ["pagerank", "betweenness", "closeness", "degree"],
            "gnn_models": ["gat", "gatv2", "gcn", "graphsage"]
        }
        
        # Record metrics
        ADVANCED_GRAPH_API_REQUESTS.labels(
            endpoint="list_algorithms",
            status="success"
        ).inc()
        
        return algorithms
        
    except Exception as e:
        logger.error(f"Error listing algorithms: {str(e)}")
        ADVANCED_GRAPH_API_REQUESTS.labels(
            endpoint="list_algorithms",
            status="error"
        ).inc()
        raise HTTPException(status_code=500, detail=f"Error listing algorithms: {str(e)}")
