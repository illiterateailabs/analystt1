"""
ML Risk Scoring API - Endpoints for automated risk scoring of transactions and entities

This module provides FastAPI endpoints for ML-based risk scoring, including:
- Transaction risk scoring
- Entity (wallet, account) risk scoring
- Subgraph risk analysis
- Batch scoring operations
- Model information and metrics

These endpoints expose risk assessment functionality with confidence intervals and explanations.
"""

import logging
import random
import json
from typing import List, Dict, Any, Optional, Union
from enum import Enum
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query, Body, Path
from pydantic import BaseModel, Field, validator

from backend.auth.dependencies import get_current_user, RoleChecker
from backend.auth.rbac import Role
from backend.integrations.neo4j_client import Neo4jClient, get_neo4j_client
from backend.core.logging import get_logger
from backend.core.metrics import REGISTRY, Counter

# Configure logger
logger = get_logger(__name__)

# Create router
router = APIRouter(tags=["ML Scoring"])

# Metrics
ML_SCORING_API_REQUESTS = Counter(
    "ml_scoring_api_requests_total",
    "Total number of ML scoring API requests",
    ["endpoint", "status"]
)

# Role checker for analyst access
analyst_role = RoleChecker([Role.ANALYST, Role.ADMIN])

# Constants
DEFAULT_MODEL_VERSION = "mock_v1"
DEFAULT_THRESHOLD = 0.7


# Simplified enums to avoid circular imports
class ConfidenceMethod(str, Enum):
    """Methods for calculating confidence intervals"""
    BOOTSTRAP = "bootstrap"
    QUANTILE = "quantile"
    VARIANCE = "variance"


class ScoringMode(str, Enum):
    """Scoring modes for risk prediction"""
    TRANSACTION = "transaction"
    ENTITY = "entity"
    SUBGRAPH = "subgraph"


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


# Mock risk scoring service (to avoid circular imports)
class MockRiskScoringService:
    """
    Mock implementation of risk scoring service
    
    This is a simplified implementation that returns mock risk scores
    for development and testing purposes.
    """
    
    def __init__(self, neo4j_client=None):
        """Initialize mock risk scoring service"""
        self.neo4j_client = neo4j_client
        self.threshold = DEFAULT_THRESHOLD
        self.model_version = DEFAULT_MODEL_VERSION
    
    async def score_transaction(
        self,
        transaction_data: Dict[str, Any],
        include_confidence: bool = True,
        include_explanation: bool = True,
        confidence_method: ConfidenceMethod = ConfidenceMethod.BOOTSTRAP,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """Generate mock transaction risk score"""
        # Extract transaction amount for more realistic scoring
        amount = transaction_data.get('amount_usd', 0)
        if amount > 100000:
            # Higher risk for large transactions
            base_risk = random.uniform(0.6, 0.9)
        else:
            base_risk = random.uniform(0.1, 0.7)
        
        # Create result
        result = {
            "risk_score": base_risk,
            "is_high_risk": base_risk >= self.threshold,
            "model_version": self.model_version,
            "timestamp": datetime.now().isoformat(),
            "transaction_id": transaction_data.get("id", "unknown")
        }
        
        # Add confidence interval if requested
        if include_confidence:
            # Simple mock confidence interval
            lower = max(0, base_risk - 0.1)
            upper = min(1, base_risk + 0.1)
            result["confidence_interval"] = {
                "lower": lower,
                "upper": upper,
                "level": 0.95,
                "method": confidence_method
            }
        
        # Add explanation if requested
        if include_explanation:
            # Mock explanation with random feature importance
            result["explanation"] = {
                "feature_importance": {
                    "amount": 0.4,
                    "velocity": 0.3,
                    "account_age": 0.2,
                    "destination_risk": 0.1
                },
                "top_features": [
                    {"feature": "amount", "importance": 0.4},
                    {"feature": "velocity", "importance": 0.3},
                    {"feature": "account_age", "importance": 0.2},
                    {"feature": "destination_risk", "importance": 0.1}
                ],
                "risk_factors": [
                    {
                        "feature": "amount",
                        "display_name": "Transaction Amount",
                        "value": transaction_data.get("amount_usd", 0),
                        "importance": 0.4
                    }
                ]
            }
        
        return result
    
    async def score_entity(
        self,
        entity_data: Dict[str, Any],
        entity_type: str = "Wallet",
        include_confidence: bool = True,
        include_explanation: bool = True,
        include_graph_features: bool = True,
        confidence_method: ConfidenceMethod = ConfidenceMethod.BOOTSTRAP,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """Generate mock entity risk score"""
        # Generate base risk score
        base_risk = random.uniform(0.2, 0.8)
        
        # Create result
        result = {
            "risk_score": base_risk,
            "is_high_risk": base_risk >= self.threshold,
            "model_version": self.model_version,
            "timestamp": datetime.now().isoformat(),
            "entity_id": entity_data.get("id", "unknown"),
            "entity_type": entity_type
        }
        
        # Add confidence interval if requested
        if include_confidence:
            # Simple mock confidence interval
            lower = max(0, base_risk - 0.1)
            upper = min(1, base_risk + 0.1)
            result["confidence_interval"] = {
                "lower": lower,
                "upper": upper,
                "level": 0.95,
                "method": confidence_method
            }
        
        # Add explanation if requested
        if include_explanation:
            # Mock explanation with random feature importance
            result["explanation"] = {
                "feature_importance": {
                    "transaction_count": 0.3,
                    "average_amount": 0.3,
                    "account_age": 0.2,
                    "network_centrality": 0.2
                },
                "top_features": [
                    {"feature": "transaction_count", "importance": 0.3},
                    {"feature": "average_amount", "importance": 0.3},
                    {"feature": "account_age", "importance": 0.2},
                    {"feature": "network_centrality", "importance": 0.2}
                ],
                "risk_factors": [
                    {
                        "feature": "transaction_count",
                        "display_name": "Transaction Count",
                        "value": random.randint(10, 100),
                        "importance": 0.3
                    }
                ]
            }
        
        # Add graph features if requested and Neo4j client is available
        if include_graph_features and self.neo4j_client:
            # Mock graph features
            result["graph_features"] = {
                "degree_centrality": random.uniform(0, 1),
                "betweenness_centrality": random.uniform(0, 1),
                "community_id": random.randint(1, 5)
            }
        
        return result
    
    async def score_subgraph(
        self,
        node_ids: List[str],
        node_type: str = "Wallet",
        n_hops: int = 1,
        include_confidence: bool = True,
        include_explanation: bool = True,
        confidence_method: ConfidenceMethod = ConfidenceMethod.BOOTSTRAP,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """Generate mock subgraph risk score"""
        # Generate individual node scores
        node_scores = []
        high_risk_count = 0
        
        for node_id in node_ids:
            risk = random.uniform(0.2, 0.8)
            is_high_risk = risk >= self.threshold
            
            if is_high_risk:
                high_risk_count += 1
            
            node_scores.append({
                "entity_id": node_id,
                "entity_type": node_type,
                "risk_score": risk,
                "is_high_risk": is_high_risk
            })
        
        # Calculate subgraph score (average of node scores)
        subgraph_score = sum(node["risk_score"] for node in node_scores) / len(node_scores) if node_scores else 0
        
        # Create result
        result = {
            "subgraph_risk_score": subgraph_score,
            "is_high_risk": subgraph_score >= self.threshold,
            "high_risk_node_count": high_risk_count,
            "total_node_count": len(node_scores),
            "model_version": self.model_version,
            "timestamp": datetime.now().isoformat(),
            "node_scores": node_scores
        }
        
        # Add confidence interval if requested
        if include_confidence:
            # Simple mock confidence interval
            lower = max(0, subgraph_score - 0.1)
            upper = min(1, subgraph_score + 0.1)
            result["confidence_interval"] = {
                "lower": lower,
                "upper": upper,
                "level": 0.95,
                "method": confidence_method
            }
        
        # Add explanation if requested
        if include_explanation:
            # Mock explanation
            result["explanation"] = {
                "high_risk_nodes": [
                    {
                        "entity_id": node["entity_id"],
                        "entity_type": node["entity_type"],
                        "risk_score": node["risk_score"],
                        "risk_factors": [
                            {
                                "feature": "centrality",
                                "display_name": "Network Centrality",
                                "value": random.uniform(0, 1),
                                "importance": 0.3
                            }
                        ]
                    }
                    for node in node_scores if node["is_high_risk"]
                ],
                "risk_distribution": {
                    "0.0-0.2": sum(1 for node in node_scores if node["risk_score"] < 0.2),
                    "0.2-0.4": sum(1 for node in node_scores if 0.2 <= node["risk_score"] < 0.4),
                    "0.4-0.6": sum(1 for node in node_scores if 0.4 <= node["risk_score"] < 0.6),
                    "0.6-0.8": sum(1 for node in node_scores if 0.6 <= node["risk_score"] < 0.8),
                    "0.8-1.0": sum(1 for node in node_scores if node["risk_score"] >= 0.8)
                }
            }
        
        return result
    
    async def batch_score_transactions(
        self,
        transactions: List[Dict[str, Any]],
        include_confidence: bool = False,
        include_explanation: bool = False
    ) -> List[Dict[str, Any]]:
        """Generate mock batch transaction scores"""
        results = []
        
        for tx in transactions:
            result = await self.score_transaction(
                transaction_data=tx,
                include_confidence=include_confidence,
                include_explanation=include_explanation
            )
            results.append(result)
        
        return results
    
    async def batch_score_entities(
        self,
        entities: List[Dict[str, Any]],
        entity_type: str = "Wallet",
        include_confidence: bool = False,
        include_explanation: bool = False,
        include_graph_features: bool = True
    ) -> List[Dict[str, Any]]:
        """Generate mock batch entity scores"""
        results = []
        
        for entity in entities:
            result = await self.score_entity(
                entity_data=entity,
                entity_type=entity_type,
                include_confidence=include_confidence,
                include_explanation=include_explanation,
                include_graph_features=include_graph_features
            )
            results.append(result)
        
        return results
    
    async def get_model_info(self) -> Dict[str, Any]:
        """Get mock model information"""
        return {
            "active_version": self.model_version,
            "loaded_models": 1,
            "models": [
                {
                    "version": self.model_version,
                    "model_type": "ensemble",
                    "created_at": datetime.now().isoformat(),
                    "feature_count": 10,
                    "metrics": {
                        "accuracy": 0.85,
                        "auc": 0.82,
                        "precision": 0.78,
                        "recall": 0.76
                    },
                    "is_active": True
                }
            ]
        }


# Global service instance
_risk_scoring_service = None


def get_risk_scoring_service(neo4j_client=None) -> MockRiskScoringService:
    """
    Get or create the risk scoring service
    
    This is a simplified implementation that returns a mock service
    for development and testing purposes.
    
    Args:
        neo4j_client: Optional Neo4j client for graph features
        
    Returns:
        MockRiskScoringService instance
    """
    global _risk_scoring_service
    
    if _risk_scoring_service is None:
        _risk_scoring_service = MockRiskScoringService(neo4j_client)
    
    # Update Neo4j client if provided
    if neo4j_client and _risk_scoring_service.neo4j_client is None:
        _risk_scoring_service.neo4j_client = neo4j_client
    
    return _risk_scoring_service


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
        risk_service = get_risk_scoring_service(neo4j_client)
        
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
        risk_service = get_risk_scoring_service(neo4j_client)
        
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
        risk_service = get_risk_scoring_service(neo4j_client)
        
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
        risk_service = get_risk_scoring_service(neo4j_client)
        
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
        risk_service = get_risk_scoring_service(neo4j_client)
        
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
