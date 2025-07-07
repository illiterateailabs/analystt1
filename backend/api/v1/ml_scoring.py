"""
ML Risk Scoring API - Endpoints for automated risk scoring of transactions and entities

This module provides FastAPI endpoints for ML-based risk scoring, including:
- Transaction risk scoring
- Entity (wallet, account) risk scoring
- Subgraph risk analysis
- Batch scoring operations
- Model information and metrics

These endpoints expose the functionality of the RiskScoringService to the API,
enabling real-time risk assessment with confidence intervals and explanations.
"""

import logging
from typing import List, Dict, Any, Optional, Union
from enum import Enum

from fastapi import APIRouter, Depends, HTTPException, Query, Body, Path
from pydantic import BaseModel, Field, validator

from backend.auth.dependencies import get_current_user, RoleChecker
from backend.auth.rbac import Role
from backend.integrations.neo4j_client import Neo4jClient, get_neo4j_client
from backend.ml.scoring import RiskScoringService, ConfidenceMethod, ScoringMode
from backend.ml import get_risk_scoring_service, ModelType
from backend.core.logging import get_logger
from backend.core.metrics import REGISTRY, Counter

# Configure logger
logger = get_logger(__name__)

# Create router
router = APIRouter(prefix="/ml-scoring", tags=["ML Scoring"])

# Metrics
ML_SCORING_API_REQUESTS = Counter(
    "ml_scoring_api_requests_total",
    "Total number of ML scoring API requests",
    ["endpoint", "status"]
)

# Role checker for analyst access
analyst_role = RoleChecker([Role.ANALYST, Role.ADMIN])


# Request and Response Models
class TransactionScoringRequest(BaseModel):
    """Request model for transaction scoring"""
    transaction_data: Dict[str, Any] = Field(..., description="Transaction data to score")
    include_confidence: bool = Field(True, description="Include confidence intervals")
    include_explanation: bool = Field(True, description="Include feature importance")
    confidence_method: ConfidenceMethod = Field(
        ConfidenceMethod.BOOTSTRAP, 
        description="Method for confidence intervals"
    )
    use_cache: bool = Field(True, description="Use cached results if available")
    tenant_id: Optional[str] = Field(None, description="Tenant ID for multi-tenant deployments")


class EntityScoringRequest(BaseModel):
    """Request model for entity scoring"""
    entity_data: Dict[str, Any] = Field(..., description="Entity data to score")
    entity_type: str = Field("Wallet", description="Type of entity")
    include_confidence: bool = Field(True, description="Include confidence intervals")
    include_explanation: bool = Field(True, description="Include feature importance")
    include_graph_features: bool = Field(True, description="Include graph-based features")
    confidence_method: ConfidenceMethod = Field(
        ConfidenceMethod.BOOTSTRAP, 
        description="Method for confidence intervals"
    )
    use_cache: bool = Field(True, description="Use cached results if available")
    tenant_id: Optional[str] = Field(None, description="Tenant ID for multi-tenant deployments")


class SubgraphScoringRequest(BaseModel):
    """Request model for subgraph scoring"""
    node_ids: List[str] = Field(..., description="List of node IDs to include in the subgraph")
    node_type: str = Field("Wallet", description="Type of nodes")
    n_hops: int = Field(1, description="Number of hops to include in the subgraph", ge=1, le=3)
    include_confidence: bool = Field(True, description="Include confidence intervals")
    include_explanation: bool = Field(True, description="Include feature importance")
    confidence_method: ConfidenceMethod = Field(
        ConfidenceMethod.BOOTSTRAP, 
        description="Method for confidence intervals"
    )
    use_cache: bool = Field(True, description="Use cached results if available")
    tenant_id: Optional[str] = Field(None, description="Tenant ID for multi-tenant deployments")


class BatchTransactionScoringRequest(BaseModel):
    """Request model for batch transaction scoring"""
    transactions: List[Dict[str, Any]] = Field(..., description="List of transactions to score")
    include_confidence: bool = Field(False, description="Include confidence intervals")
    include_explanation: bool = Field(False, description="Include feature importance")
    tenant_id: Optional[str] = Field(None, description="Tenant ID for multi-tenant deployments")
    
    @validator('transactions')
    def validate_batch_size(cls, v):
        if len(v) > 100:
            raise ValueError("Batch size cannot exceed 100 transactions")
        return v


class BatchEntityScoringRequest(BaseModel):
    """Request model for batch entity scoring"""
    entities: List[Dict[str, Any]] = Field(..., description="List of entities to score")
    entity_type: str = Field("Wallet", description="Type of entities")
    include_confidence: bool = Field(False, description="Include confidence intervals")
    include_explanation: bool = Field(False, description="Include feature importance")
    include_graph_features: bool = Field(True, description="Include graph-based features")
    tenant_id: Optional[str] = Field(None, description="Tenant ID for multi-tenant deployments")
    
    @validator('entities')
    def validate_batch_size(cls, v):
        if len(v) > 100:
            raise ValueError("Batch size cannot exceed 100 entities")
        return v


class RiskScoringResponse(BaseModel):
    """Base response model for risk scoring operations"""
    status: str = Field("success", description="Status of the operation")
    model_version: str = Field(..., description="Model version used for scoring")
    result: Dict[str, Any] = Field(..., description="Scoring result")


@router.post(
    "/transaction",
    response_model=RiskScoringResponse,
    summary="Score a transaction for fraud risk",
    description="Calculate fraud risk score for a financial transaction with confidence intervals and explanations"
)
async def score_transaction(
    request: TransactionScoringRequest,
    current_user: Dict = Depends(get_current_user),
    role_check: bool = Depends(analyst_role),
    neo4j_client: Neo4jClient = Depends(get_neo4j_client)
):
    """
    Score a transaction for fraud risk
    
    This endpoint calculates a risk score for a financial transaction using
    machine learning models. It can include confidence intervals and feature
    importance explanations to help understand the risk factors.
    """
    try:
        # Get risk scoring service
        risk_service = get_risk_scoring_service()
        
        # Set Neo4j client if needed for graph features
        if not hasattr(risk_service, 'neo4j_client') or risk_service.neo4j_client is None:
            risk_service.neo4j_client = neo4j_client
        
        # Score the transaction
        result = await risk_service.score_transaction(
            transaction_data=request.transaction_data,
            include_confidence=request.include_confidence,
            include_explanation=request.include_explanation,
            confidence_method=request.confidence_method,
            use_cache=request.use_cache
        )
        
        # Record metrics
        ML_SCORING_API_REQUESTS.labels(
            endpoint="score_transaction",
            status="success"
        ).inc()
        
        return RiskScoringResponse(
            status="success",
            model_version=result.get("model_version", "unknown"),
            result=result
        )
        
    except Exception as e:
        logger.error(f"Error scoring transaction: {str(e)}")
        ML_SCORING_API_REQUESTS.labels(
            endpoint="score_transaction",
            status="error"
        ).inc()
        raise HTTPException(status_code=500, detail=f"Error scoring transaction: {str(e)}")


@router.post(
    "/entity",
    response_model=RiskScoringResponse,
    summary="Score an entity for fraud risk",
    description="Calculate fraud risk score for an entity (wallet, account) with confidence intervals and explanations"
)
async def score_entity(
    request: EntityScoringRequest,
    current_user: Dict = Depends(get_current_user),
    role_check: bool = Depends(analyst_role),
    neo4j_client: Neo4jClient = Depends(get_neo4j_client)
):
    """
    Score an entity for fraud risk
    
    This endpoint calculates a risk score for an entity (wallet, account, etc.)
    using machine learning models. It can include confidence intervals, feature
    importance explanations, and graph-based features.
    """
    try:
        # Get risk scoring service
        risk_service = get_risk_scoring_service()
        
        # Set Neo4j client if needed for graph features
        if not hasattr(risk_service, 'neo4j_client') or risk_service.neo4j_client is None:
            risk_service.neo4j_client = neo4j_client
        
        # Score the entity
        result = await risk_service.score_entity(
            entity_data=request.entity_data,
            entity_type=request.entity_type,
            include_confidence=request.include_confidence,
            include_explanation=request.include_explanation,
            include_graph_features=request.include_graph_features,
            confidence_method=request.confidence_method,
            use_cache=request.use_cache
        )
        
        # Record metrics
        ML_SCORING_API_REQUESTS.labels(
            endpoint="score_entity",
            status="success"
        ).inc()
        
        return RiskScoringResponse(
            status="success",
            model_version=result.get("model_version", "unknown"),
            result=result
        )
        
    except Exception as e:
        logger.error(f"Error scoring entity: {str(e)}")
        ML_SCORING_API_REQUESTS.labels(
            endpoint="score_entity",
            status="error"
        ).inc()
        raise HTTPException(status_code=500, detail=f"Error scoring entity: {str(e)}")


@router.post(
    "/subgraph",
    response_model=RiskScoringResponse,
    summary="Score a subgraph for fraud risk",
    description="Calculate fraud risk scores for a subgraph of related entities"
)
async def score_subgraph(
    request: SubgraphScoringRequest,
    current_user: Dict = Depends(get_current_user),
    role_check: bool = Depends(analyst_role),
    neo4j_client: Neo4jClient = Depends(get_neo4j_client)
):
    """
    Score a subgraph for fraud risk
    
    This endpoint calculates risk scores for a subgraph of related entities.
    It extracts a subgraph from Neo4j centered around the specified nodes,
    scores each node, and provides an overall subgraph risk assessment.
    """
    try:
        # Get risk scoring service
        risk_service = get_risk_scoring_service()
        
        # Set Neo4j client if needed for graph features
        if not hasattr(risk_service, 'neo4j_client') or risk_service.neo4j_client is None:
            risk_service.neo4j_client = neo4j_client
        
        # Score the subgraph
        result = await risk_service.score_subgraph(
            node_ids=request.node_ids,
            node_type=request.node_type,
            n_hops=request.n_hops,
            include_confidence=request.include_confidence,
            include_explanation=request.include_explanation,
            confidence_method=request.confidence_method,
            use_cache=request.use_cache
        )
        
        # Record metrics
        ML_SCORING_API_REQUESTS.labels(
            endpoint="score_subgraph",
            status="success"
        ).inc()
        
        return RiskScoringResponse(
            status="success",
            model_version=result.get("model_version", "unknown"),
            result=result
        )
        
    except Exception as e:
        logger.error(f"Error scoring subgraph: {str(e)}")
        ML_SCORING_API_REQUESTS.labels(
            endpoint="score_subgraph",
            status="error"
        ).inc()
        raise HTTPException(status_code=500, detail=f"Error scoring subgraph: {str(e)}")


@router.post(
    "/batch/transactions",
    response_model=List[Dict[str, Any]],
    summary="Score a batch of transactions",
    description="Calculate risk scores for multiple transactions in a single request"
)
async def batch_score_transactions(
    request: BatchTransactionScoringRequest,
    current_user: Dict = Depends(get_current_user),
    role_check: bool = Depends(analyst_role),
    neo4j_client: Neo4jClient = Depends(get_neo4j_client)
):
    """
    Score a batch of transactions
    
    This endpoint calculates risk scores for multiple transactions in a single request.
    It's more efficient than making separate requests for each transaction.
    """
    try:
        # Get risk scoring service
        risk_service = get_risk_scoring_service()
        
        # Set Neo4j client if needed for graph features
        if not hasattr(risk_service, 'neo4j_client') or risk_service.neo4j_client is None:
            risk_service.neo4j_client = neo4j_client
        
        # Score the batch of transactions
        results = await risk_service.batch_score_transactions(
            transactions=request.transactions,
            include_confidence=request.include_confidence,
            include_explanation=request.include_explanation
        )
        
        # Record metrics
        ML_SCORING_API_REQUESTS.labels(
            endpoint="batch_score_transactions",
            status="success"
        ).inc()
        
        return results
        
    except Exception as e:
        logger.error(f"Error batch scoring transactions: {str(e)}")
        ML_SCORING_API_REQUESTS.labels(
            endpoint="batch_score_transactions",
            status="error"
        ).inc()
        raise HTTPException(status_code=500, detail=f"Error batch scoring transactions: {str(e)}")


@router.post(
    "/batch/entities",
    response_model=List[Dict[str, Any]],
    summary="Score a batch of entities",
    description="Calculate risk scores for multiple entities in a single request"
)
async def batch_score_entities(
    request: BatchEntityScoringRequest,
    current_user: Dict = Depends(get_current_user),
    role_check: bool = Depends(analyst_role),
    neo4j_client: Neo4jClient = Depends(get_neo4j_client)
):
    """
    Score a batch of entities
    
    This endpoint calculates risk scores for multiple entities in a single request.
    It's more efficient than making separate requests for each entity.
    """
    try:
        # Get risk scoring service
        risk_service = get_risk_scoring_service()
        
        # Set Neo4j client if needed for graph features
        if not hasattr(risk_service, 'neo4j_client') or risk_service.neo4j_client is None:
            risk_service.neo4j_client = neo4j_client
        
        # Score the batch of entities
        results = await risk_service.batch_score_entities(
            entities=request.entities,
            entity_type=request.entity_type,
            include_confidence=request.include_confidence,
            include_explanation=request.include_explanation,
            include_graph_features=request.include_graph_features
        )
        
        # Record metrics
        ML_SCORING_API_REQUESTS.labels(
            endpoint="batch_score_entities",
            status="success"
        ).inc()
        
        return results
        
    except Exception as e:
        logger.error(f"Error batch scoring entities: {str(e)}")
        ML_SCORING_API_REQUESTS.labels(
            endpoint="batch_score_entities",
            status="error"
        ).inc()
        raise HTTPException(status_code=500, detail=f"Error batch scoring entities: {str(e)}")


@router.get(
    "/model/info",
    response_model=Dict[str, Any],
    summary="Get model information",
    description="Get information about the loaded ML models"
)
async def get_model_info(
    current_user: Dict = Depends(get_current_user)
):
    """
    Get information about the loaded ML models
    
    This endpoint returns information about the ML models currently loaded in the
    risk scoring service, including versions, types, and metrics.
    """
    try:
        # Get risk scoring service
        risk_service = get_risk_scoring_service()
        
        # Get model info
        model_info = await risk_service.get_model_info()
        
        # Record metrics
        ML_SCORING_API_REQUESTS.labels(
            endpoint="get_model_info",
            status="success"
        ).inc()
        
        return model_info
        
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        ML_SCORING_API_REQUESTS.labels(
            endpoint="get_model_info",
            status="error"
        ).inc()
        raise HTTPException(status_code=500, detail=f"Error getting model info: {str(e)}")


@router.get(
    "/threshold",
    response_model=Dict[str, float],
    summary="Get risk threshold",
    description="Get the current risk threshold for high-risk classification"
)
async def get_risk_threshold(
    current_user: Dict = Depends(get_current_user)
):
    """
    Get the current risk threshold
    
    This endpoint returns the current threshold used to classify entities as high-risk.
    """
    try:
        # Get risk scoring service
        risk_service = get_risk_scoring_service()
        
        # Get threshold
        threshold = risk_service.threshold
        
        # Record metrics
        ML_SCORING_API_REQUESTS.labels(
            endpoint="get_risk_threshold",
            status="success"
        ).inc()
        
        return {"threshold": threshold}
        
    except Exception as e:
        logger.error(f"Error getting risk threshold: {str(e)}")
        ML_SCORING_API_REQUESTS.labels(
            endpoint="get_risk_threshold",
            status="error"
        ).inc()
        raise HTTPException(status_code=500, detail=f"Error getting risk threshold: {str(e)}")


@router.put(
    "/threshold",
    response_model=Dict[str, float],
    summary="Update risk threshold",
    description="Update the risk threshold for high-risk classification"
)
async def update_risk_threshold(
    threshold: float = Body(..., ge=0.0, le=1.0, embed=True),
    current_user: Dict = Depends(get_current_user),
    role_check: bool = Depends(analyst_role)
):
    """
    Update the risk threshold
    
    This endpoint updates the threshold used to classify entities as high-risk.
    Only analysts and admins can update the threshold.
    """
    try:
        # Get risk scoring service
        risk_service = get_risk_scoring_service()
        
        # Update threshold
        old_threshold = risk_service.threshold
        risk_service.threshold = threshold
        
        # Record metrics
        ML_SCORING_API_REQUESTS.labels(
            endpoint="update_risk_threshold",
            status="success"
        ).inc()
        
        return {
            "threshold": threshold,
            "previous_threshold": old_threshold
        }
        
    except Exception as e:
        logger.error(f"Error updating risk threshold: {str(e)}")
        ML_SCORING_API_REQUESTS.labels(
            endpoint="update_risk_threshold",
            status="error"
        ).inc()
        raise HTTPException(status_code=500, detail=f"Error updating risk threshold: {str(e)}")
