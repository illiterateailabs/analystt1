"""
Risk Scoring Service - Automated ML-based risk scoring for transactions and entities

This module provides a comprehensive risk scoring service that uses ensemble machine learning
models to predict fraud risk for transactions and entities. It supports real-time scoring,
confidence intervals, and explainable predictions to help analysts understand risk factors.

Key features:
- Ensemble model combination (XGBoost, LightGBM, CatBoost)
- Transaction-level and entity-level risk scoring
- Confidence intervals for prediction uncertainty
- Feature importance and explanation generation
- Integration with graph-based features
- Caching for high-performance scoring
"""

import os
import json
import time
import pickle
import logging
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple, Set
from enum import Enum
from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score
from sklearn.base import BaseEstimator

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

from backend.core.logging import get_logger
from backend.core.metrics import REGISTRY, Counter, Histogram, Gauge
from backend.core.redis_client import get_redis_client
from backend.ml.registry import ModelRegistry, ModelMetadata
from backend.integrations.neo4j_client import Neo4jClient
from backend.ml import ModelType, FeatureType, DEFAULT_THRESHOLD, DEFAULT_ENSEMBLE_SIZE

# Configure logger
logger = get_logger(__name__)

# Prometheus metrics
RISK_SCORE_REQUESTS = Counter(
    "risk_score_requests_total",
    "Total number of risk score requests",
    ["type", "model_version", "status"]
)

RISK_SCORE_LATENCY = Histogram(
    "risk_score_latency_seconds",
    "Latency of risk score calculations",
    ["type", "model_version"]
)

HIGH_RISK_ENTITIES = Counter(
    "high_risk_entities_total",
    "Total number of high-risk entities detected",
    ["type", "model_version", "threshold"]
)

MODEL_PREDICTION_DISTRIBUTION = Histogram(
    "model_prediction_distribution",
    "Distribution of model predictions",
    ["type", "model_version"],
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

FEATURE_IMPORTANCE = Gauge(
    "feature_importance",
    "Importance of features in the model",
    ["feature", "model_version"]
)

# Constants
CACHE_TTL = 3600  # 1 hour
DEFAULT_CONFIDENCE_LEVEL = 0.95
DEFAULT_CACHE_KEY_PREFIX = "risk_score:"
DEFAULT_NUM_FEATURES = 20


class ScoringMode(str, Enum):
    """Scoring modes for risk prediction"""
    TRANSACTION = "transaction"  # Score individual transactions
    ENTITY = "entity"            # Score entities (wallets, accounts)
    SUBGRAPH = "subgraph"        # Score a subgraph of related entities


class ConfidenceMethod(str, Enum):
    """Methods for calculating confidence intervals"""
    BOOTSTRAP = "bootstrap"      # Bootstrap sampling of ensemble models
    QUANTILE = "quantile"        # Quantile regression (for models that support it)
    VARIANCE = "variance"        # Based on prediction variance across models


class RiskScoringService:
    """
    Service for ML-based risk scoring of transactions and entities
    
    Provides methods for scoring financial transactions and entities using
    ensemble machine learning models. Supports confidence intervals, feature
    importance, and explanation generation.
    """
    
    def __init__(
        self,
        model_registry: ModelRegistry,
        model_version: str = "latest",
        neo4j_client: Optional[Neo4jClient] = None,
        threshold: float = DEFAULT_THRESHOLD,
        ensemble_size: int = DEFAULT_ENSEMBLE_SIZE,
        cache_ttl: int = CACHE_TTL,
        confidence_level: float = DEFAULT_CONFIDENCE_LEVEL
    ):
        """
        Initialize the risk scoring service
        
        Args:
            model_registry: Model registry for loading models
            model_version: Model version to use (or "latest")
            neo4j_client: Neo4j client for graph features
            threshold: Risk threshold for high-risk classification
            ensemble_size: Number of models to use in ensemble
            cache_ttl: Cache time-to-live in seconds
            confidence_level: Confidence level for intervals (0-1)
        """
        self.model_registry = model_registry
        self.model_version = model_version
        self.neo4j_client = neo4j_client
        self.threshold = threshold
        self.ensemble_size = ensemble_size
        self.cache_ttl = cache_ttl
        self.confidence_level = confidence_level
        
        # Initialize Redis client for caching
        self.redis = get_redis_client()
        
        # Lazy-loaded models and metadata
        self._models = {}
        self._metadata = {}
        self._feature_columns = {}
        self._feature_importance = {}
        self._shap_explainers = {}
        
        # Initialize service
        self._initialize_service()
        
        logger.info(f"Initialized RiskScoringService with model version: {model_version}")
    
    def _initialize_service(self):
        """Initialize the service by loading models and metadata"""
        # Load models asynchronously
        asyncio.create_task(self._load_models())
        
        # Register feature importance metrics
        asyncio.create_task(self._register_feature_importance_metrics())
    
    async def _load_models(self):
        """Load models from the model registry"""
        try:
            # Get model metadata
            if self.model_version == "latest":
                metadata_list = await self.model_registry.get_latest_models(
                    model_type=ModelType.ENSEMBLE,
                    limit=1
                )
                if not metadata_list:
                    # Try individual models if ensemble not found
                    metadata_list = await self.model_registry.get_latest_models(
                        limit=self.ensemble_size
                    )
            else:
                # Get specific version
                metadata = await self.model_registry.get_model_metadata(self.model_version)
                metadata_list = [metadata] if metadata else []
                
                # If it's an ensemble, also get its components
                if metadata and metadata.model_type == ModelType.ENSEMBLE:
                    ensemble_metadata = json.loads(metadata.metadata_json)
                    component_versions = ensemble_metadata.get("component_versions", [])
                    for version in component_versions:
                        component_metadata = await self.model_registry.get_model_metadata(version)
                        if component_metadata:
                            metadata_list.append(component_metadata)
            
            if not metadata_list:
                logger.error(f"No models found for version: {self.model_version}")
                return
            
            # Load each model
            for metadata in metadata_list:
                model_path = await self.model_registry.get_model_path(metadata.version)
                if not model_path:
                    logger.warning(f"Model path not found for version: {metadata.version}")
                    continue
                
                try:
                    # Load the model
                    model = self._load_model_from_path(model_path)
                    if model:
                        self._models[metadata.version] = model
                        self._metadata[metadata.version] = metadata
                        
                        # Extract feature columns
                        model_metadata = json.loads(metadata.metadata_json)
                        self._feature_columns[metadata.version] = model_metadata.get("feature_columns", [])
                        
                        # Extract feature importance if available
                        if "feature_importance" in model_metadata:
                            self._feature_importance[metadata.version] = model_metadata["feature_importance"]
                        
                        # Create SHAP explainer if available
                        if SHAP_AVAILABLE and hasattr(model, "predict"):
                            try:
                                # For tree-based models
                                if hasattr(model, "feature_importances_") or hasattr(model, "feature_importance_"):
                                    self._shap_explainers[metadata.version] = shap.TreeExplainer(model)
                                # For other models
                                else:
                                    self._shap_explainers[metadata.version] = shap.Explainer(model)
                            except Exception as e:
                                logger.warning(f"Failed to create SHAP explainer for {metadata.version}: {str(e)}")
                        
                        logger.info(f"Loaded model: {metadata.version} ({metadata.model_type})")
                    else:
                        logger.warning(f"Failed to load model: {metadata.version}")
                
                except Exception as e:
                    logger.error(f"Error loading model {metadata.version}: {str(e)}")
            
            # Set the active model version
            if self._models:
                if self.model_version == "latest":
                    # Use the most recent model
                    self.model_version = max(
                        self._metadata.keys(),
                        key=lambda v: self._metadata[v].created_at
                    )
                
                logger.info(f"Active model version set to: {self.model_version}")
                logger.info(f"Loaded {len(self._models)} models for ensemble")
            else:
                logger.error("No models could be loaded")
        
        except Exception as e:
            logger.error(f"Error initializing risk scoring service: {str(e)}")
    
    def _load_model_from_path(self, model_path: str) -> Optional[BaseEstimator]:
        """
        Load a model from the given path
        
        Args:
            model_path: Path to the model file
            
        Returns:
            Loaded model or None if loading fails
        """
        try:
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            return model
        except Exception as e:
            logger.error(f"Error loading model from {model_path}: {str(e)}")
            return None
    
    async def _register_feature_importance_metrics(self):
        """Register feature importance metrics with Prometheus"""
        # Wait for models to load
        for _ in range(10):  # Try for up to 10 seconds
            if self._feature_importance:
                break
            await asyncio.sleep(1)
        
        # Register metrics for each feature
        for version, importance in self._feature_importance.items():
            for feature, value in importance.items():
                FEATURE_IMPORTANCE.labels(
                    feature=feature,
                    model_version=version
                ).set(value)
    
    async def score_transaction(
        self,
        transaction_data: Dict[str, Any],
        include_confidence: bool = True,
        include_explanation: bool = True,
        confidence_method: ConfidenceMethod = ConfidenceMethod.BOOTSTRAP,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Score a transaction for fraud risk
        
        Args:
            transaction_data: Transaction data as a dictionary
            include_confidence: Whether to include confidence intervals
            include_explanation: Whether to include feature importance
            confidence_method: Method for calculating confidence intervals
            use_cache: Whether to use cached results
            
        Returns:
            Dictionary with risk score and optional confidence/explanation
        """
        start_time = time.time()
        
        # Generate cache key if caching is enabled
        cache_key = None
        if use_cache:
            cache_key = self._generate_cache_key(
                "transaction",
                transaction_data,
                include_confidence,
                include_explanation
            )
            # Check cache
            cached_result = await self._get_cached_result(cache_key)
            if cached_result:
                RISK_SCORE_REQUESTS.labels(
                    type="transaction",
                    model_version=self.model_version,
                    status="cache_hit"
                ).inc()
                return cached_result
        
        try:
            # Ensure models are loaded
            if not self._models:
                await self._load_models()
                if not self._models:
                    raise ValueError("No models available for scoring")
            
            # Extract features
            features_df = self._extract_transaction_features(transaction_data)
            
            # Make predictions with ensemble
            predictions, model_versions = self._predict_with_ensemble(
                features_df,
                mode=ScoringMode.TRANSACTION
            )
            
            # Calculate ensemble score
            ensemble_score = np.mean(predictions)
            
            # Record prediction distribution
            MODEL_PREDICTION_DISTRIBUTION.labels(
                type="transaction",
                model_version=self.model_version
            ).observe(ensemble_score)
            
            # Prepare result
            result = {
                "risk_score": float(ensemble_score),
                "is_high_risk": ensemble_score >= self.threshold,
                "model_version": self.model_version,
                "timestamp": datetime.now().isoformat(),
                "transaction_id": transaction_data.get("id", "unknown")
            }
            
            # Add confidence intervals if requested
            if include_confidence:
                confidence_interval = self._calculate_confidence_interval(
                    predictions,
                    method=confidence_method
                )
                result["confidence_interval"] = {
                    "lower": float(confidence_interval[0]),
                    "upper": float(confidence_interval[1]),
                    "level": self.confidence_level,
                    "method": confidence_method
                }
            
            # Add explanation if requested
            if include_explanation:
                explanation = await self._generate_explanation(
                    features_df,
                    model_versions[0] if model_versions else self.model_version,
                    mode=ScoringMode.TRANSACTION
                )
                result["explanation"] = explanation
            
            # Cache result if caching is enabled
            if use_cache and cache_key:
                await self._cache_result(cache_key, result)
            
            # Record metrics
            duration = time.time() - start_time
            RISK_SCORE_LATENCY.labels(
                type="transaction",
                model_version=self.model_version
            ).observe(duration)
            RISK_SCORE_REQUESTS.labels(
                type="transaction",
                model_version=self.model_version,
                status="success"
            ).inc()
            
            if result["is_high_risk"]:
                HIGH_RISK_ENTITIES.labels(
                    type="transaction",
                    model_version=self.model_version,
                    threshold=str(self.threshold)
                ).inc()
            
            return result
            
        except Exception as e:
            logger.error(f"Error scoring transaction: {str(e)}")
            RISK_SCORE_REQUESTS.labels(
                type="transaction",
                model_version=self.model_version,
                status="error"
            ).inc()
            
            return {
                "error": str(e),
                "risk_score": None,
                "is_high_risk": None,
                "model_version": self.model_version,
                "timestamp": datetime.now().isoformat()
            }
    
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
        """
        Score an entity (wallet, account) for fraud risk
        
        Args:
            entity_data: Entity data as a dictionary
            entity_type: Type of entity (Wallet, Account, etc.)
            include_confidence: Whether to include confidence intervals
            include_explanation: Whether to include feature importance
            include_graph_features: Whether to include graph-based features
            confidence_method: Method for calculating confidence intervals
            use_cache: Whether to use cached results
            
        Returns:
            Dictionary with risk score and optional confidence/explanation
        """
        start_time = time.time()
        
        # Generate cache key if caching is enabled
        cache_key = None
        if use_cache:
            cache_key = self._generate_cache_key(
                f"entity_{entity_type}",
                entity_data,
                include_confidence,
                include_explanation,
                include_graph_features
            )
            # Check cache
            cached_result = await self._get_cached_result(cache_key)
            if cached_result:
                RISK_SCORE_REQUESTS.labels(
                    type=f"entity_{entity_type}",
                    model_version=self.model_version,
                    status="cache_hit"
                ).inc()
                return cached_result
        
        try:
            # Ensure models are loaded
            if not self._models:
                await self._load_models()
                if not self._models:
                    raise ValueError("No models available for scoring")
            
            # Extract features
            features_df = await self._extract_entity_features(
                entity_data,
                entity_type,
                include_graph_features
            )
            
            # Make predictions with ensemble
            predictions, model_versions = self._predict_with_ensemble(
                features_df,
                mode=ScoringMode.ENTITY
            )
            
            # Calculate ensemble score
            ensemble_score = np.mean(predictions)
            
            # Record prediction distribution
            MODEL_PREDICTION_DISTRIBUTION.labels(
                type=f"entity_{entity_type}",
                model_version=self.model_version
            ).observe(ensemble_score)
            
            # Prepare result
            result = {
                "risk_score": float(ensemble_score),
                "is_high_risk": ensemble_score >= self.threshold,
                "model_version": self.model_version,
                "timestamp": datetime.now().isoformat(),
                "entity_id": entity_data.get("id", "unknown"),
                "entity_type": entity_type
            }
            
            # Add confidence intervals if requested
            if include_confidence:
                confidence_interval = self._calculate_confidence_interval(
                    predictions,
                    method=confidence_method
                )
                result["confidence_interval"] = {
                    "lower": float(confidence_interval[0]),
                    "upper": float(confidence_interval[1]),
                    "level": self.confidence_level,
                    "method": confidence_method
                }
            
            # Add explanation if requested
            if include_explanation:
                explanation = await self._generate_explanation(
                    features_df,
                    model_versions[0] if model_versions else self.model_version,
                    mode=ScoringMode.ENTITY
                )
                result["explanation"] = explanation
            
            # Cache result if caching is enabled
            if use_cache and cache_key:
                await self._cache_result(cache_key, result)
            
            # Record metrics
            duration = time.time() - start_time
            RISK_SCORE_LATENCY.labels(
                type=f"entity_{entity_type}",
                model_version=self.model_version
            ).observe(duration)
            RISK_SCORE_REQUESTS.labels(
                type=f"entity_{entity_type}",
                model_version=self.model_version,
                status="success"
            ).inc()
            
            if result["is_high_risk"]:
                HIGH_RISK_ENTITIES.labels(
                    type=f"entity_{entity_type}",
                    model_version=self.model_version,
                    threshold=str(self.threshold)
                ).inc()
            
            return result
            
        except Exception as e:
            logger.error(f"Error scoring entity: {str(e)}")
            RISK_SCORE_REQUESTS.labels(
                type=f"entity_{entity_type}",
                model_version=self.model_version,
                status="error"
            ).inc()
            
            return {
                "error": str(e),
                "risk_score": None,
                "is_high_risk": None,
                "model_version": self.model_version,
                "timestamp": datetime.now().isoformat(),
                "entity_id": entity_data.get("id", "unknown"),
                "entity_type": entity_type
            }
    
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
        """
        Score a subgraph of related entities for fraud risk
        
        Args:
            node_ids: List of node IDs to include in the subgraph
            node_type: Type of nodes (Wallet, Account, etc.)
            n_hops: Number of hops to include in the subgraph
            include_confidence: Whether to include confidence intervals
            include_explanation: Whether to include feature importance
            confidence_method: Method for calculating confidence intervals
            use_cache: Whether to use cached results
            
        Returns:
            Dictionary with risk scores for the subgraph and individual nodes
        """
        start_time = time.time()
        
        # Generate cache key if caching is enabled
        cache_key = None
        if use_cache:
            cache_key = self._generate_cache_key(
                f"subgraph_{node_type}",
                {"node_ids": node_ids, "n_hops": n_hops},
                include_confidence,
                include_explanation
            )
            # Check cache
            cached_result = await self._get_cached_result(cache_key)
            if cached_result:
                RISK_SCORE_REQUESTS.labels(
                    type=f"subgraph_{node_type}",
                    model_version=self.model_version,
                    status="cache_hit"
                ).inc()
                return cached_result
        
        try:
            # Ensure Neo4j client is available
            if not self.neo4j_client:
                raise ValueError("Neo4j client is required for subgraph scoring")
            
            # Ensure models are loaded
            if not self._models:
                await self._load_models()
                if not self._models:
                    raise ValueError("No models available for scoring")
            
            # Extract subgraph from Neo4j
            subgraph_query = f"""
            MATCH (n:{node_type})
            WHERE n.id IN $node_ids
            CALL apoc.path.subgraphAll(n, {{maxLevel: $n_hops}})
            YIELD nodes, relationships
            RETURN nodes, relationships
            """
            
            subgraph_result = await self.neo4j_client.aquery(
                subgraph_query,
                {"node_ids": node_ids, "n_hops": n_hops}
            )
            
            if not subgraph_result:
                raise ValueError(f"No subgraph found for node IDs: {node_ids}")
            
            # Extract nodes and relationships
            all_nodes = []
            for record in subgraph_result:
                all_nodes.extend(record["nodes"])
            
            # Score each node in the subgraph
            node_scores = []
            for node in all_nodes:
                # Convert Neo4j node to dictionary
                node_data = {
                    "id": node.get("id"),
                    "labels": list(node.labels),
                    "properties": dict(node)
                }
                
                # Score the node
                node_score = await self.score_entity(
                    node_data,
                    entity_type=list(node.labels)[0] if node.labels else node_type,
                    include_confidence=include_confidence,
                    include_explanation=False,  # Skip explanation for individual nodes
                    include_graph_features=True,
                    confidence_method=confidence_method,
                    use_cache=use_cache
                )
                
                node_scores.append(node_score)
            
            # Calculate subgraph-level risk score
            valid_scores = [
                score["risk_score"] for score in node_scores
                if score["risk_score"] is not None
            ]
            
            if valid_scores:
                subgraph_score = np.mean(valid_scores)
                subgraph_is_high_risk = subgraph_score >= self.threshold
                
                # Count high-risk nodes
                high_risk_count = sum(
                    1 for score in node_scores
                    if score.get("is_high_risk", False)
                )
                
                # Calculate confidence interval for subgraph score
                if include_confidence:
                    confidence_interval = self._calculate_confidence_interval(
                        valid_scores,
                        method=confidence_method
                    )
                else:
                    confidence_interval = None
            else:
                subgraph_score = None
                subgraph_is_high_risk = None
                high_risk_count = 0
                confidence_interval = None
            
            # Prepare result
            result = {
                "subgraph_risk_score": float(subgraph_score) if subgraph_score is not None else None,
                "is_high_risk": subgraph_is_high_risk,
                "high_risk_node_count": high_risk_count,
                "total_node_count": len(node_scores),
                "model_version": self.model_version,
                "timestamp": datetime.now().isoformat(),
                "node_scores": node_scores
            }
            
            # Add confidence intervals if requested
            if include_confidence and confidence_interval:
                result["confidence_interval"] = {
                    "lower": float(confidence_interval[0]),
                    "upper": float(confidence_interval[1]),
                    "level": self.confidence_level,
                    "method": confidence_method
                }
            
            # Add explanation if requested
            if include_explanation:
                # Generate explanation based on node risk factors
                explanation = self._generate_subgraph_explanation(node_scores)
                result["explanation"] = explanation
            
            # Cache result if caching is enabled
            if use_cache and cache_key:
                await self._cache_result(cache_key, result)
            
            # Record metrics
            duration = time.time() - start_time
            RISK_SCORE_LATENCY.labels(
                type=f"subgraph_{node_type}",
                model_version=self.model_version
            ).observe(duration)
            RISK_SCORE_REQUESTS.labels(
                type=f"subgraph_{node_type}",
                model_version=self.model_version,
                status="success"
            ).inc()
            
            if result["is_high_risk"]:
                HIGH_RISK_ENTITIES.labels(
                    type=f"subgraph_{node_type}",
                    model_version=self.model_version,
                    threshold=str(self.threshold)
                ).inc()
            
            return result
            
        except Exception as e:
            logger.error(f"Error scoring subgraph: {str(e)}")
            RISK_SCORE_REQUESTS.labels(
                type=f"subgraph_{node_type}",
                model_version=self.model_version,
                status="error"
            ).inc()
            
            return {
                "error": str(e),
                "subgraph_risk_score": None,
                "is_high_risk": None,
                "model_version": self.model_version,
                "timestamp": datetime.now().isoformat(),
                "node_ids": node_ids
            }
    
    async def batch_score_transactions(
        self,
        transactions: List[Dict[str, Any]],
        include_confidence: bool = False,
        include_explanation: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Score a batch of transactions for fraud risk
        
        Args:
            transactions: List of transaction data dictionaries
            include_confidence: Whether to include confidence intervals
            include_explanation: Whether to include feature importance
            
        Returns:
            List of dictionaries with risk scores
        """
        # Process transactions in parallel
        tasks = [
            self.score_transaction(
                tx,
                include_confidence=include_confidence,
                include_explanation=include_explanation,
                use_cache=True
            )
            for tx in transactions
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error in batch scoring transaction {i}: {str(result)}")
                processed_results.append({
                    "error": str(result),
                    "risk_score": None,
                    "is_high_risk": None,
                    "transaction_id": transactions[i].get("id", f"unknown_{i}"),
                    "model_version": self.model_version,
                    "timestamp": datetime.now().isoformat()
                })
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def batch_score_entities(
        self,
        entities: List[Dict[str, Any]],
        entity_type: str = "Wallet",
        include_confidence: bool = False,
        include_explanation: bool = False,
        include_graph_features: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Score a batch of entities for fraud risk
        
        Args:
            entities: List of entity data dictionaries
            entity_type: Type of entities
            include_confidence: Whether to include confidence intervals
            include_explanation: Whether to include feature importance
            include_graph_features: Whether to include graph-based features
            
        Returns:
            List of dictionaries with risk scores
        """
        # Process entities in parallel
        tasks = [
            self.score_entity(
                entity,
                entity_type=entity_type,
                include_confidence=include_confidence,
                include_explanation=include_explanation,
                include_graph_features=include_graph_features,
                use_cache=True
            )
            for entity in entities
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error in batch scoring entity {i}: {str(result)}")
                processed_results.append({
                    "error": str(result),
                    "risk_score": None,
                    "is_high_risk": None,
                    "entity_id": entities[i].get("id", f"unknown_{i}"),
                    "entity_type": entity_type,
                    "model_version": self.model_version,
                    "timestamp": datetime.now().isoformat()
                })
            else:
                processed_results.append(result)
        
        return processed_results
    
    def _extract_transaction_features(self, transaction_data: Dict[str, Any]) -> pd.DataFrame:
        """
        Extract features from transaction data
        
        Args:
            transaction_data: Transaction data as a dictionary
            
        Returns:
            DataFrame with extracted features
        """
        # Get feature columns for the active model
        feature_cols = self._get_feature_columns(ScoringMode.TRANSACTION)
        
        # Extract basic features
        features = {}
        
        # Handle numeric features
        for col in feature_cols:
            if col in transaction_data:
                features[col] = transaction_data[col]
            else:
                # Try to extract from nested dictionaries
                parts = col.split('.')
                value = transaction_data
                for part in parts:
                    if isinstance(value, dict) and part in value:
                        value = value[part]
                    else:
                        value = None
                        break
                
                features[col] = value
        
        # Fill missing values
        for col in feature_cols:
            if col not in features or features[col] is None:
                features[col] = 0.0
        
        # Create DataFrame
        df = pd.DataFrame([features])
        
        # Ensure all columns are present and in the right order
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0.0
        
        return df[feature_cols]
    
    async def _extract_entity_features(
        self,
        entity_data: Dict[str, Any],
        entity_type: str,
        include_graph_features: bool
    ) -> pd.DataFrame:
        """
        Extract features from entity data
        
        Args:
            entity_data: Entity data as a dictionary
            entity_type: Type of entity
            include_graph_features: Whether to include graph-based features
            
        Returns:
            DataFrame with extracted features
        """
        # Get feature columns for the active model
        feature_cols = self._get_feature_columns(ScoringMode.ENTITY)
        
        # Extract basic features
        features = {}
        
        # Handle properties directly in entity_data
        if "properties" in entity_data:
            properties = entity_data["properties"]
            for col in feature_cols:
                if col in properties:
                    features[col] = properties[col]
        else:
            # Assume entity_data contains the properties directly
            for col in feature_cols:
                if col in entity_data:
                    features[col] = entity_data[col]
        
        # Extract graph features if requested and Neo4j client is available
        if include_graph_features and self.neo4j_client:
            entity_id = entity_data.get("id") or entity_data.get("properties", {}).get("id")
            if entity_id:
                graph_features = await self._extract_graph_features(entity_id, entity_type)
                features.update(graph_features)
        
        # Fill missing values
        for col in feature_cols:
            if col not in features or features[col] is None:
                features[col] = 0.0
        
        # Create DataFrame
        df = pd.DataFrame([features])
        
        # Ensure all columns are present and in the right order
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0.0
        
        return df[feature_cols]
    
    async def _extract_graph_features(
        self,
        entity_id: str,
        entity_type: str
    ) -> Dict[str, float]:
        """
        Extract graph-based features for an entity
        
        Args:
            entity_id: Entity ID
            entity_type: Type of entity
            
        Returns:
            Dictionary of graph features
        """
        if not self.neo4j_client:
            return {}
        
        # Query for graph features
        query = f"""
        MATCH (n:{entity_type} {{id: $entity_id}})
        
        // Degree centrality
        OPTIONAL MATCH (n)-[r]-()
        WITH n, count(r) AS degree
        
        // Transaction volume (last 30 days)
        OPTIONAL MATCH (n)-[tx:SENDS_TO|RECEIVES_FROM]-()
        WHERE tx.timestamp >= datetime() - duration('P30D')
        WITH n, degree, sum(coalesce(tx.amount_usd, 0)) AS tx_volume_30d, count(tx) AS tx_count_30d
        
        // Clustering coefficient
        OPTIONAL MATCH (n)-[]->(neighbor)-[]->(neighbor2)-[]->(n)
        WITH n, degree, tx_volume_30d, tx_count_30d, count(neighbor2) AS triangles
        
        // PageRank (pre-computed)
        WITH n, degree, tx_volume_30d, tx_count_30d, triangles,
             coalesce(n.pagerank, 0.0) AS pagerank,
             coalesce(n.betweenness, 0.0) AS betweenness
        
        // Community features
        OPTIONAL MATCH (n)-[]->(m)
        WHERE n.community_id IS NOT NULL AND n.community_id = m.community_id
        WITH n, degree, tx_volume_30d, tx_count_30d, triangles, pagerank, betweenness,
             count(m) AS same_community_neighbors
        
        // Risk propagation features
        WITH n, degree, tx_volume_30d, tx_count_30d, triangles, pagerank, betweenness,
             same_community_neighbors,
             coalesce(n.propagated_risk_score, 0.0) AS propagated_risk
        
        RETURN
            degree,
            tx_volume_30d,
            tx_count_30d,
            CASE WHEN degree <= 1 THEN 0 ELSE triangles * 1.0 / (degree * (degree - 1)) END AS clustering_coef,
            pagerank,
            betweenness,
            same_community_neighbors,
            propagated_risk
        """
        
        try:
            result = await self.neo4j_client.aquery(query, {"entity_id": entity_id})
            
            if not result:
                logger.warning(f"No graph features found for entity {entity_id}")
                return {}
            
            # Extract features from the result
            record = result[0]
            features = {
                "graph_degree": record["degree"],
                "graph_tx_volume_30d": record["tx_volume_30d"],
                "graph_tx_count_30d": record["tx_count_30d"],
                "graph_clustering_coef": record["clustering_coef"],
                "graph_pagerank": record["pagerank"],
                "graph_betweenness": record["betweenness"],
                "graph_same_community_neighbors": record["same_community_neighbors"],
                "graph_propagated_risk": record["propagated_risk"]
            }
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting graph features for entity {entity_id}: {str(e)}")
            return {}
    
    def _predict_with_ensemble(
        self,
        features_df: pd.DataFrame,
        mode: ScoringMode
    ) -> Tuple[List[float], List[str]]:
        """
        Make predictions using ensemble of models
        
        Args:
            features_df: DataFrame with features
            mode: Scoring mode (transaction, entity, subgraph)
            
        Returns:
            Tuple of (predictions, model_versions)
        """
        if not self._models:
            raise ValueError("No models available for prediction")
        
        # Get appropriate models for this mode
        model_versions = self._get_model_versions_for_mode(mode)
        
        if not model_versions:
            raise ValueError(f"No models available for mode: {mode}")
        
        # Make predictions with each model
        predictions = []
        used_versions = []
        
        for version in model_versions:
            model = self._models.get(version)
            if not model:
                continue
            
            try:
                # Ensure features match what the model expects
                model_features = self._feature_columns.get(version, [])
                if model_features:
                    # Fill missing columns with zeros
                    for col in model_features:
                        if col not in features_df.columns:
                            features_df[col] = 0.0
                    
                    # Select only the columns the model expects
                    model_df = features_df[model_features]
                else:
                    model_df = features_df
                
                # Make prediction
                if hasattr(model, "predict_proba"):
                    # Classification model with probability output
                    pred = model.predict_proba(model_df)
                    # Take the probability of the positive class
                    if pred.shape[1] >= 2:
                        pred = pred[:, 1]
                    else:
                        pred = pred[:, 0]
                elif hasattr(model, "predict"):
                    # Regression model or classification without probabilities
                    pred = model.predict(model_df)
                else:
                    logger.warning(f"Model {version} does not have predict method")
                    continue
                
                # Add to ensemble
                predictions.append(float(pred[0]))
                used_versions.append(version)
                
            except Exception as e:
                logger.error(f"Error making prediction with model {version}: {str(e)}")
        
        if not predictions:
            raise ValueError("All models failed to make predictions")
        
        return predictions, used_versions
    
    def _get_model_versions_for_mode(self, mode: ScoringMode) -> List[str]:
        """
        Get model versions appropriate for the given scoring mode
        
        Args:
            mode: Scoring mode (transaction, entity, subgraph)
            
        Returns:
            List of model versions
        """
        if not self._metadata:
            return []
        
        # Check if we have an ensemble model
        if self.model_version in self._metadata:
            metadata = self._metadata[self.model_version]
            if metadata.model_type == ModelType.ENSEMBLE:
                # Use the ensemble components
                ensemble_metadata = json.loads(metadata.metadata_json)
                component_versions = ensemble_metadata.get("component_versions", [])
                return component_versions
        
        # Otherwise, find models that match the mode
        matching_versions = []
        for version, metadata in self._metadata.items():
            model_metadata = json.loads(metadata.metadata_json)
            model_mode = model_metadata.get("scoring_mode", "").lower()
            
            # If mode matches or no specific mode is set
            if not model_mode or model_mode == mode.value:
                matching_versions.append(version)
        
        # Limit to ensemble size
        if len(matching_versions) > self.ensemble_size:
            # Sort by creation date (newest first)
            matching_versions.sort(
                key=lambda v: self._metadata[v].created_at,
                reverse=True
            )
            matching_versions = matching_versions[:self.ensemble_size]
        
        return matching_versions
    
    def _get_feature_columns(self, mode: ScoringMode) -> List[str]:
        """
        Get feature columns for the active model and mode
        
        Args:
            mode: Scoring mode (transaction, entity, subgraph)
            
        Returns:
            List of feature column names
        """
        # Try to get columns from the active model
        if self.model_version in self._feature_columns:
            return self._feature_columns[self.model_version]
        
        # Otherwise, find columns from any model matching the mode
        for version, metadata in self._metadata.items():
            model_metadata = json.loads(metadata.metadata_json)
            model_mode = model_metadata.get("scoring_mode", "").lower()
            
            if model_mode == mode.value and version in self._feature_columns:
                return self._feature_columns[version]
        
        # Fallback to empty list
        logger.warning(f"No feature columns found for mode: {mode}")
        return []
    
    def _calculate_confidence_interval(
        self,
        predictions: List[float],
        method: ConfidenceMethod = ConfidenceMethod.BOOTSTRAP,
        n_bootstrap: int = 1000
    ) -> Tuple[float, float]:
        """
        Calculate confidence interval for ensemble predictions
        
        Args:
            predictions: List of predictions from ensemble models
            method: Method for calculating confidence intervals
            n_bootstrap: Number of bootstrap samples (for bootstrap method)
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        if not predictions:
            return (0.0, 0.0)
        
        if len(predictions) == 1:
            # Can't calculate interval with just one prediction
            score = predictions[0]
            # Return a default interval
            return (max(0.0, score - 0.1), min(1.0, score + 0.1))
        
        alpha = 1.0 - self.confidence_level
        
        if method == ConfidenceMethod.BOOTSTRAP:
            # Bootstrap method
            bootstrap_means = []
            rng = np.random.RandomState(42)  # For reproducibility
            
            for _ in range(n_bootstrap):
                # Sample with replacement
                sample_idx = rng.randint(0, len(predictions), len(predictions))
                sample = [predictions[i] for i in sample_idx]
                bootstrap_means.append(np.mean(sample))
            
            # Calculate percentiles
            lower = np.percentile(bootstrap_means, alpha * 100 / 2)
            upper = np.percentile(bootstrap_means, 100 - alpha * 100 / 2)
            
        elif method == ConfidenceMethod.VARIANCE:
            # Variance-based method
            mean = np.mean(predictions)
            std = np.std(predictions, ddof=1)
            
            # t-distribution for small samples
            from scipy import stats
            t_val = stats.t.ppf(1 - alpha / 2, len(predictions) - 1)
            
            # Standard error of the mean
            se = std / np.sqrt(len(predictions))
            
            lower = mean - t_val * se
            upper = mean + t_val * se
            
        else:  # ConfidenceMethod.QUANTILE
            # Simple quantile method
            lower = np.percentile(predictions, alpha * 100 / 2)
            upper = np.percentile(predictions, 100 - alpha * 100 / 2)
        
        # Ensure bounds are within [0, 1]
        lower = max(0.0, lower)
        upper = min(1.0, upper)
        
        return (lower, upper)
    
    async def _generate_explanation(
        self,
        features_df: pd.DataFrame,
        model_version: str,
        mode: ScoringMode
    ) -> Dict[str, Any]:
        """
        Generate explanation for a prediction
        
        Args:
            features_df: DataFrame with features
            model_version: Model version to use for explanation
            mode: Scoring mode (transaction, entity, subgraph)
            
        Returns:
            Dictionary with explanation details
        """
        explanation = {
            "feature_importance": {},
            "top_features": [],
            "risk_factors": []
        }
        
        try:
            # Get model and feature importance
            model = self._models.get(model_version)
            if not model:
                return explanation
            
            # Try different methods to get feature importance
            importance_dict = {}
            
            # 1. Try SHAP values if available
            if SHAP_AVAILABLE and model_version in self._shap_explainers:
                try:
                    explainer = self._shap_explainers[model_version]
                    shap_values = explainer(features_df)
                    
                    # Extract feature importance from SHAP values
                    if hasattr(shap_values, "values"):
                        # For newer SHAP versions
                        if len(shap_values.values.shape) > 2:
                            # Multi-class, take positive class
                            importance_values = shap_values.values[:, 1]
                        else:
                            importance_values = shap_values.values
                    else:
                        # For older SHAP versions
                        importance_values = shap_values
                    
                    # Create importance dictionary
                    for i, col in enumerate(features_df.columns):
                        importance_dict[col] = abs(float(importance_values[0][i]))
                
                except Exception as e:
                    logger.warning(f"Error calculating SHAP values: {str(e)}")
            
            # 2. Try model's feature_importances_ attribute
            if not importance_dict and hasattr(model, "feature_importances_"):
                for i, col in enumerate(features_df.columns):
                    importance_dict[col] = float(model.feature_importances_[i])
            
            # 3. Try model's coef_ attribute (for linear models)
            elif not importance_dict and hasattr(model, "coef_"):
                coef = model.coef_
                if len(coef.shape) > 1:
                    # Multi-class, take positive class or average
                    if coef.shape[0] > 1:
                        coef = coef[1] if coef.shape[0] > 1 else coef[0]
                
                for i, col in enumerate(features_df.columns):
                    importance_dict[col] = abs(float(coef[i]))
            
            # 4. Fall back to stored feature importance
            elif not importance_dict and model_version in self._feature_importance:
                importance_dict = self._feature_importance[model_version]
            
            # If we have importance values, create the explanation
            if importance_dict:
                # Normalize importance values
                total_importance = sum(importance_dict.values())
                if total_importance > 0:
                    normalized_importance = {
                        k: v / total_importance
                        for k, v in importance_dict.items()
                    }
                else:
                    normalized_importance = importance_dict
                
                # Sort by importance
                sorted_features = sorted(
                    normalized_importance.items(),
                    key=lambda x: x[1],
                    reverse=True
                )
                
                # Take top features
                top_n = min(DEFAULT_NUM_FEATURES, len(sorted_features))
                top_features = sorted_features[:top_n]
                
                # Add to explanation
                explanation["feature_importance"] = {
                    k: float(v) for k, v in normalized_importance.items()
                }
                
                explanation["top_features"] = [
                    {"feature": k, "importance": float(v)}
                    for k, v in top_features
                ]
                
                # Generate risk factors
                risk_factors = []
                for feature, importance in top_features:
                    if importance > 0.01:  # Only include significant features
                        # Get feature value
                        if feature in features_df.columns:
                            value = float(features_df[feature].iloc[0])
                            
                            # Determine if this is a risk factor
                            is_risk_factor = False
                            
                            # Check if this is a positive coefficient (for linear models)
                            if hasattr(model, "coef_"):
                                coef = model.coef_
                                if len(coef.shape) > 1:
                                    coef = coef[1] if coef.shape[0] > 1 else coef[0]
                                
                                feature_idx = list(features_df.columns).index(feature)
                                is_risk_factor = coef[feature_idx] > 0 and value > 0
                            else:
                                # For other models, assume high values are risky
                                is_risk_factor = value > 0
                            
                            if is_risk_factor:
                                # Format the feature name for display
                                display_name = feature.replace("_", " ").title()
                                
                                risk_factors.append({
                                    "feature": feature,
                                    "display_name": display_name,
                                    "value": value,
                                    "importance": float(importance)
                                })
                
                explanation["risk_factors"] = risk_factors
            
            return explanation
            
        except Exception as e:
            logger.error(f"Error generating explanation: {str(e)}")
            return explanation
    
    def _generate_subgraph_explanation(
        self,
        node_scores: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate explanation for a subgraph score
        
        Args:
            node_scores: List of node risk scores
            
        Returns:
            Dictionary with explanation details
        """
        explanation = {
            "high_risk_nodes": [],
            "risk_distribution": {},
            "community_risk": {}
        }
        
        try:
            # Count high-risk nodes
            high_risk_nodes = [
                score for score in node_scores
                if score.get("is_high_risk", False)
            ]
            
            # Add high-risk node details
            for node in high_risk_nodes:
                explanation["high_risk_nodes"].append({
                    "entity_id": node.get("entity_id", "unknown"),
                    "entity_type": node.get("entity_type", "unknown"),
                    "risk_score": node.get("risk_score", 0.0),
                    "risk_factors": node.get("explanation", {}).get("risk_factors", [])
                })
            
            # Calculate risk distribution
            risk_buckets = {
                "0.0-0.2": 0,
                "0.2-0.4": 0,
                "0.4-0.6": 0,
                "0.6-0.8": 0,
                "0.8-1.0": 0
            }
            
            for score in node_scores:
                risk = score.get("risk_score")
                if risk is not None:
                    if risk < 0.2:
                        risk_buckets["0.0-0.2"] += 1
                    elif risk < 0.4:
                        risk_buckets["0.2-0.4"] += 1
                    elif risk < 0.6:
                        risk_buckets["0.4-0.6"] += 1
                    elif risk < 0.8:
                        risk_buckets["0.6-0.8"] += 1
                    else:
                        risk_buckets["0.8-1.0"] += 1
            
            explanation["risk_distribution"] = risk_buckets
            
            # Calculate community risk if available
            communities = {}
            for score in node_scores:
                entity_data = score.get("entity_data", {})
                community_id = entity_data.get("community_id")
                if community_id:
                    if community_id not in communities:
                        communities[community_id] = {
                            "count": 0,
                            "high_risk_count": 0,
                            "total_risk": 0.0
                        }
                    
                    communities[community_id]["count"] += 1
                    if score.get("is_high_risk", False):
                        communities[community_id]["high_risk_count"] += 1
                    
                    risk = score.get("risk_score")
                    if risk is not None:
                        communities[community_id]["total_risk"] += risk
            
            # Calculate average risk per community
            for community_id, data in communities.items():
                if data["count"] > 0:
                    data["average_risk"] = data["total_risk"] / data["count"]
                else:
                    data["average_risk"] = 0.0
            
            explanation["community_risk"] = communities
            
            return explanation
            
        except Exception as e:
            logger.error(f"Error generating subgraph explanation: {str(e)}")
            return explanation
    
    def _generate_cache_key(self, prefix: str, data: Dict[str, Any], *args) -> str:
        """
        Generate a cache key for the given data
        
        Args:
            prefix: Prefix for the cache key
            data: Data to include in the cache key
            *args: Additional arguments to include in the cache key
            
        Returns:
            Cache key string
        """
        # Create a stable representation of the data
        data_str = json.dumps(data, sort_keys=True)
        
        # Hash the data
        import hashlib
        data_hash = hashlib.md5(data_str.encode()).hexdigest()
        
        # Create key with prefix and hash
        key = f"{DEFAULT_CACHE_KEY_PREFIX}{prefix}:{data_hash}"
        
        # Add args to the key
        if args:
            args_str = "_".join(str(arg) for arg in args)
            key += f":{args_str}"
        
        return key
    
    async def _get_cached_result(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """
        Get a cached result
        
        Args:
            cache_key: Cache key
            
        Returns:
            Cached result or None if not found
        """
        if not self.redis:
            return None
        
        try:
            cached = await self.redis.get(cache_key)
            if cached:
                return json.loads(cached)
        except Exception as e:
            logger.debug(f"Error getting cached result: {str(e)}")
        
        return None
    
    async def _cache_result(self, cache_key: str, result: Dict[str, Any]) -> bool:
        """
        Cache a result
        
        Args:
            cache_key: Cache key
            result: Result to cache
            
        Returns:
            True if successful, False otherwise
        """
        if not self.redis:
            return False
        
        try:
            await self.redis.setex(
                cache_key,
                self.cache_ttl,
                json.dumps(result)
            )
            return True
        except Exception as e:
            logger.debug(f"Error caching result: {str(e)}")
            return False
    
    async def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded models
        
        Returns:
            Dictionary with model information
        """
        # Ensure models are loaded
        if not self._models:
            await self._load_models()
        
        model_info = {
            "active_version": self.model_version,
            "loaded_models": len(self._models),
            "models": []
        }
        
        # Add information for each model
        for version, metadata in self._metadata.items():
            model_metadata = json.loads(metadata.metadata_json)
            
            model_info["models"].append({
                "version": version,
                "model_type": metadata.model_type,
                "created_at": metadata.created_at.isoformat() if metadata.created_at else None,
                "feature_count": len(self._feature_columns.get(version, [])),
                "metrics": model_metadata.get("metrics", {}),
                "is_active": version == self.model_version
            })
        
        return model_info
