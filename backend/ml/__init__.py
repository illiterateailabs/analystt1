"""
Machine Learning Module - Automated risk scoring and model management

This module provides infrastructure for ML-based risk scoring of financial transactions
and entities. It supports ensemble models, automated retraining, and model versioning
to enable continuous improvement of fraud detection capabilities.

Key components:
- ModelRegistry: Manages model versions, metadata, and deployment
- RiskScoringService: Provides prediction endpoints for real-time scoring
- ModelTrainer: Handles periodic retraining of models with new data
- FeatureStore: Manages feature extraction and transformation pipelines
- EvaluationService: Monitors model performance and drift

Usage:
    from backend.ml import get_risk_scoring_service, setup_model_trainer
    
    # Get the risk scoring service for predictions
    risk_service = get_risk_scoring_service()
    
    # Score a transaction or entity
    risk_score = await risk_service.score_transaction(transaction_data)
    
    # Setup automated model retraining
    trainer = setup_model_trainer()
    await trainer.schedule_retraining(cron_schedule="0 0 * * *")  # Daily at midnight
"""

import os
import logging
from enum import Enum
from typing import Optional, Dict, Any, Union, List, Tuple

from backend.core.logging import get_logger

# Configure logger
logger = get_logger(__name__)

# Constants
DEFAULT_MODEL_DIR = "models"
DEFAULT_MODEL_VERSION = "latest"
DEFAULT_THRESHOLD = 0.7
DEFAULT_ENSEMBLE_SIZE = 3


class ModelType(str, Enum):
    """Supported model types"""
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    CATBOOST = "catboost"
    SKLEARN_RF = "random_forest"
    ENSEMBLE = "ensemble"
    NEURAL_NET = "neural_network"


class ModelFramework(str, Enum):
    """Supported ML frameworks"""
    SKLEARN = "sklearn"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    CATBOOST = "catboost"
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"


class FeatureType(str, Enum):
    """Types of features used in models"""
    TRANSACTION = "transaction"
    ENTITY = "entity"
    GRAPH = "graph"
    TEMPORAL = "temporal"
    TEXT = "text"


# Determine model storage location
MODEL_STORAGE_TYPE = os.environ.get("ML_MODEL_STORAGE", "local")
MODEL_REGISTRY_URL = os.environ.get("ML_MODEL_REGISTRY_URL", "")
MODEL_DIR = os.environ.get("ML_MODEL_DIR", DEFAULT_MODEL_DIR)

# Lazy-loaded services
_risk_scoring_service = None
_model_registry = None
_model_trainer = None
_feature_store = None


def get_risk_scoring_service(model_version: str = DEFAULT_MODEL_VERSION):
    """
    Get or create the risk scoring service
    
    Args:
        model_version: Model version to use for scoring
        
    Returns:
        RiskScoringService instance
    """
    global _risk_scoring_service
    
    if _risk_scoring_service is not None:
        return _risk_scoring_service
    
    # Import here to avoid circular imports
    from backend.ml.scoring import RiskScoringService
    from backend.ml.registry import get_model_registry
    
    # Get model registry
    model_registry = get_model_registry()
    
    # Create risk scoring service
    _risk_scoring_service = RiskScoringService(
        model_registry=model_registry,
        model_version=model_version
    )
    
    logger.info(f"Initialized RiskScoringService with model version: {model_version}")
    return _risk_scoring_service


def get_model_registry():
    """
    Get or create the model registry
    
    Returns:
        ModelRegistry instance
    """
    global _model_registry
    
    if _model_registry is not None:
        return _model_registry
    
    # Import here to avoid circular imports
    from backend.ml.registry import ModelRegistry
    
    # Create model registry based on storage type
    if MODEL_STORAGE_TYPE == "s3":
        from backend.ml.registry import S3ModelRegistry
        _model_registry = S3ModelRegistry(registry_url=MODEL_REGISTRY_URL)
    elif MODEL_STORAGE_TYPE == "mlflow":
        from backend.ml.registry import MLflowModelRegistry
        _model_registry = MLflowModelRegistry(tracking_uri=MODEL_REGISTRY_URL)
    else:
        from backend.ml.registry import LocalModelRegistry
        _model_registry = LocalModelRegistry(model_dir=MODEL_DIR)
    
    logger.info(f"Initialized ModelRegistry with storage type: {MODEL_STORAGE_TYPE}")
    return _model_registry


def get_feature_store():
    """
    Get or create the feature store
    
    Returns:
        FeatureStore instance
    """
    global _feature_store
    
    if _feature_store is not None:
        return _feature_store
    
    # Import here to avoid circular imports
    from backend.ml.features import FeatureStore
    
    # Create feature store
    _feature_store = FeatureStore()
    
    logger.info("Initialized FeatureStore")
    return _feature_store


def setup_model_trainer(schedule_retraining: bool = False):
    """
    Set up the model trainer for automated retraining
    
    Args:
        schedule_retraining: Whether to schedule automatic retraining
        
    Returns:
        ModelTrainer instance
    """
    global _model_trainer
    
    if _model_trainer is not None:
        return _model_trainer
    
    # Import here to avoid circular imports
    from backend.ml.training import ModelTrainer
    from backend.ml.registry import get_model_registry
    from backend.ml.features import get_feature_store
    
    # Get dependencies
    model_registry = get_model_registry()
    feature_store = get_feature_store()
    
    # Create model trainer
    _model_trainer = ModelTrainer(
        model_registry=model_registry,
        feature_store=feature_store
    )
    
    # Schedule retraining if requested
    if schedule_retraining:
        import asyncio
        asyncio.create_task(_model_trainer.schedule_retraining())
    
    logger.info("Initialized ModelTrainer")
    return _model_trainer


def get_model_evaluation_service():
    """
    Get the model evaluation service
    
    Returns:
        EvaluationService instance
    """
    # Import here to avoid circular imports
    from backend.ml.evaluation import EvaluationService
    from backend.ml.registry import get_model_registry
    
    # Get model registry
    model_registry = get_model_registry()
    
    # Create evaluation service
    evaluation_service = EvaluationService(model_registry=model_registry)
    
    logger.info("Initialized EvaluationService")
    return evaluation_service


# Initialize on module import
logger.info("ML module initialized")
