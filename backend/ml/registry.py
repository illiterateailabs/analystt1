"""
Model Registry - Storage and versioning for ML models

This module provides a registry for storing, retrieving, and versioning ML models.
It supports different storage backends (local filesystem, S3, MLflow) with a
consistent interface for model management.

Key components:
- ModelMetadata: Stores information about models (version, type, metrics)
- ModelRegistry: Abstract base class for model registries
- LocalModelRegistry: Filesystem-based implementation
- S3ModelRegistry: Amazon S3 storage implementation (stub)
- MLflowModelRegistry: MLflow integration (stub)

The registry enables model versioning, A/B testing, and rollback capabilities
for the risk scoring service.
"""

import os
import json
import pickle
import logging
import shutil
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from uuid import uuid4

from pydantic import BaseModel, Field

from backend.core.logging import get_logger
from backend.ml import ModelType, ModelFramework, DEFAULT_MODEL_DIR

# Configure logger
logger = get_logger(__name__)

# Constants
DEFAULT_MODEL_EXTENSION = ".pkl"
DEFAULT_METADATA_EXTENSION = ".json"


class ModelMetadata(BaseModel):
    """
    Metadata for a machine learning model
    
    Stores information about a model, including its version, type, creation date,
    and performance metrics.
    """
    version: str = Field(..., description="Unique model version identifier")
    model_type: ModelType = Field(..., description="Type of model")
    framework: ModelFramework = Field(..., description="ML framework used")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    created_by: Optional[str] = Field(None, description="User who created the model")
    description: Optional[str] = Field(None, description="Model description")
    metrics: Dict[str, float] = Field(default_factory=dict, description="Performance metrics")
    metadata_json: str = Field(default="{}", description="Additional metadata as JSON string")
    
    class Config:
        orm_mode = True


class ModelRegistry(ABC):
    """
    Abstract base class for model registries
    
    Defines the interface for storing and retrieving ML models and their metadata.
    Concrete implementations handle different storage backends.
    """
    
    @abstractmethod
    async def save_model(
        self,
        model: Any,
        metadata: ModelMetadata,
        model_path: Optional[str] = None
    ) -> str:
        """
        Save a model and its metadata to the registry
        
        Args:
            model: Model object to save
            metadata: Model metadata
            model_path: Optional path to save the model
            
        Returns:
            Model version string
        """
        pass
    
    @abstractmethod
    async def get_model(self, version: str) -> Optional[Any]:
        """
        Get a model from the registry
        
        Args:
            version: Model version
            
        Returns:
            Model object or None if not found
        """
        pass
    
    @abstractmethod
    async def get_model_path(self, version: str) -> Optional[str]:
        """
        Get the path to a model file
        
        Args:
            version: Model version
            
        Returns:
            Path to the model file or None if not found
        """
        pass
    
    @abstractmethod
    async def get_model_metadata(self, version: str) -> Optional[ModelMetadata]:
        """
        Get model metadata
        
        Args:
            version: Model version
            
        Returns:
            Model metadata or None if not found
        """
        pass
    
    @abstractmethod
    async def list_models(
        self,
        model_type: Optional[ModelType] = None,
        framework: Optional[ModelFramework] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[ModelMetadata]:
        """
        List models in the registry
        
        Args:
            model_type: Filter by model type
            framework: Filter by ML framework
            limit: Maximum number of models to return
            offset: Offset for pagination
            
        Returns:
            List of model metadata
        """
        pass
    
    @abstractmethod
    async def get_latest_models(
        self,
        model_type: Optional[ModelType] = None,
        framework: Optional[ModelFramework] = None,
        limit: int = 1
    ) -> List[ModelMetadata]:
        """
        Get the latest models from the registry
        
        Args:
            model_type: Filter by model type
            framework: Filter by ML framework
            limit: Maximum number of models to return
            
        Returns:
            List of model metadata for the latest models
        """
        pass
    
    @abstractmethod
    async def delete_model(self, version: str) -> bool:
        """
        Delete a model from the registry
        
        Args:
            version: Model version
            
        Returns:
            True if successful, False otherwise
        """
        pass


class LocalModelRegistry(ModelRegistry):
    """
    Local filesystem implementation of model registry
    
    Stores models and metadata as files in a local directory.
    """
    
    def __init__(self, model_dir: str = DEFAULT_MODEL_DIR):
        """
        Initialize local model registry
        
        Args:
            model_dir: Directory to store models
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir = self.model_dir / "metadata"
        self.metadata_dir.mkdir(exist_ok=True)
        
        logger.info(f"Initialized LocalModelRegistry in {self.model_dir}")
    
    async def save_model(
        self,
        model: Any,
        metadata: ModelMetadata,
        model_path: Optional[str] = None
    ) -> str:
        """
        Save a model and its metadata to the local filesystem
        
        Args:
            model: Model object to save
            metadata: Model metadata
            model_path: Optional path to save the model
            
        Returns:
            Model version string
        """
        # Generate version if not provided
        if not metadata.version:
            metadata.version = f"{metadata.model_type.value}_{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid4().hex[:8]}"
        
        # Determine model path
        if not model_path:
            model_path = self.model_dir / f"{metadata.version}{DEFAULT_MODEL_EXTENSION}"
        else:
            model_path = Path(model_path)
        
        # Save model
        try:
            with open(model_path, "wb") as f:
                pickle.dump(model, f)
            
            # Save metadata
            metadata_path = self.metadata_dir / f"{metadata.version}{DEFAULT_METADATA_EXTENSION}"
            with open(metadata_path, "w") as f:
                f.write(metadata.json())
            
            logger.info(f"Saved model {metadata.version} to {model_path}")
            return metadata.version
            
        except Exception as e:
            logger.error(f"Error saving model {metadata.version}: {str(e)}")
            raise
    
    async def get_model(self, version: str) -> Optional[Any]:
        """
        Get a model from the local filesystem
        
        Args:
            version: Model version
            
        Returns:
            Model object or None if not found
        """
        model_path = self.model_dir / f"{version}{DEFAULT_MODEL_EXTENSION}"
        
        if not model_path.exists():
            logger.warning(f"Model {version} not found at {model_path}")
            return None
        
        try:
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            
            logger.debug(f"Loaded model {version} from {model_path}")
            return model
            
        except Exception as e:
            logger.error(f"Error loading model {version}: {str(e)}")
            return None
    
    async def get_model_path(self, version: str) -> Optional[str]:
        """
        Get the path to a model file
        
        Args:
            version: Model version
            
        Returns:
            Path to the model file or None if not found
        """
        model_path = self.model_dir / f"{version}{DEFAULT_MODEL_EXTENSION}"
        
        if not model_path.exists():
            logger.warning(f"Model {version} not found at {model_path}")
            return None
        
        return str(model_path)
    
    async def get_model_metadata(self, version: str) -> Optional[ModelMetadata]:
        """
        Get model metadata from the local filesystem
        
        Args:
            version: Model version
            
        Returns:
            Model metadata or None if not found
        """
        metadata_path = self.metadata_dir / f"{version}{DEFAULT_METADATA_EXTENSION}"
        
        if not metadata_path.exists():
            logger.warning(f"Metadata for model {version} not found at {metadata_path}")
            return None
        
        try:
            with open(metadata_path, "r") as f:
                metadata_json = f.read()
            
            metadata = ModelMetadata.parse_raw(metadata_json)
            logger.debug(f"Loaded metadata for model {version}")
            return metadata
            
        except Exception as e:
            logger.error(f"Error loading metadata for model {version}: {str(e)}")
            return None
    
    async def list_models(
        self,
        model_type: Optional[ModelType] = None,
        framework: Optional[ModelFramework] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[ModelMetadata]:
        """
        List models in the local filesystem
        
        Args:
            model_type: Filter by model type
            framework: Filter by ML framework
            limit: Maximum number of models to return
            offset: Offset for pagination
            
        Returns:
            List of model metadata
        """
        metadata_files = list(self.metadata_dir.glob(f"*{DEFAULT_METADATA_EXTENSION}"))
        
        # Load all metadata
        all_metadata = []
        for metadata_path in metadata_files:
            try:
                with open(metadata_path, "r") as f:
                    metadata_json = f.read()
                
                metadata = ModelMetadata.parse_raw(metadata_json)
                all_metadata.append(metadata)
                
            except Exception as e:
                logger.error(f"Error loading metadata from {metadata_path}: {str(e)}")
        
        # Apply filters
        filtered_metadata = all_metadata
        
        if model_type:
            filtered_metadata = [m for m in filtered_metadata if m.model_type == model_type]
        
        if framework:
            filtered_metadata = [m for m in filtered_metadata if m.framework == framework]
        
        # Sort by creation date (newest first)
        filtered_metadata.sort(key=lambda m: m.created_at, reverse=True)
        
        # Apply pagination
        paginated_metadata = filtered_metadata[offset:offset + limit]
        
        return paginated_metadata
    
    async def get_latest_models(
        self,
        model_type: Optional[ModelType] = None,
        framework: Optional[ModelFramework] = None,
        limit: int = 1
    ) -> List[ModelMetadata]:
        """
        Get the latest models from the local filesystem
        
        Args:
            model_type: Filter by model type
            framework: Filter by ML framework
            limit: Maximum number of models to return
            
        Returns:
            List of model metadata for the latest models
        """
        # List all models
        all_metadata = await self.list_models(
            model_type=model_type,
            framework=framework,
            limit=1000  # High limit to get all models
        )
        
        # Sort by creation date (newest first)
        all_metadata.sort(key=lambda m: m.created_at, reverse=True)
        
        # Take the latest N models
        latest_metadata = all_metadata[:limit]
        
        return latest_metadata
    
    async def delete_model(self, version: str) -> bool:
        """
        Delete a model from the local filesystem
        
        Args:
            version: Model version
            
        Returns:
            True if successful, False otherwise
        """
        model_path = self.model_dir / f"{version}{DEFAULT_MODEL_EXTENSION}"
        metadata_path = self.metadata_dir / f"{version}{DEFAULT_METADATA_EXTENSION}"
        
        try:
            # Delete model file
            if model_path.exists():
                model_path.unlink()
            
            # Delete metadata file
            if metadata_path.exists():
                metadata_path.unlink()
            
            logger.info(f"Deleted model {version}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting model {version}: {str(e)}")
            return False


class S3ModelRegistry(ModelRegistry):
    """
    Amazon S3 implementation of model registry (stub)
    
    Stores models and metadata in an S3 bucket.
    """
    
    def __init__(self, registry_url: str):
        """
        Initialize S3 model registry
        
        Args:
            registry_url: S3 bucket URL
        """
        self.registry_url = registry_url
        logger.info(f"Initialized S3ModelRegistry with URL: {registry_url}")
        
        # This is a stub implementation
        logger.warning("S3ModelRegistry is a stub implementation")
    
    async def save_model(
        self,
        model: Any,
        metadata: ModelMetadata,
        model_path: Optional[str] = None
    ) -> str:
        """Stub implementation"""
        logger.warning("S3ModelRegistry.save_model is not implemented")
        return metadata.version
    
    async def get_model(self, version: str) -> Optional[Any]:
        """Stub implementation"""
        logger.warning("S3ModelRegistry.get_model is not implemented")
        return None
    
    async def get_model_path(self, version: str) -> Optional[str]:
        """Stub implementation"""
        logger.warning("S3ModelRegistry.get_model_path is not implemented")
        return None
    
    async def get_model_metadata(self, version: str) -> Optional[ModelMetadata]:
        """Stub implementation"""
        logger.warning("S3ModelRegistry.get_model_metadata is not implemented")
        return None
    
    async def list_models(
        self,
        model_type: Optional[ModelType] = None,
        framework: Optional[ModelFramework] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[ModelMetadata]:
        """Stub implementation"""
        logger.warning("S3ModelRegistry.list_models is not implemented")
        return []
    
    async def get_latest_models(
        self,
        model_type: Optional[ModelType] = None,
        framework: Optional[ModelFramework] = None,
        limit: int = 1
    ) -> List[ModelMetadata]:
        """Stub implementation"""
        logger.warning("S3ModelRegistry.get_latest_models is not implemented")
        return []
    
    async def delete_model(self, version: str) -> bool:
        """Stub implementation"""
        logger.warning("S3ModelRegistry.delete_model is not implemented")
        return False


class MLflowModelRegistry(ModelRegistry):
    """
    MLflow implementation of model registry (stub)
    
    Uses MLflow for model tracking and versioning.
    """
    
    def __init__(self, tracking_uri: str):
        """
        Initialize MLflow model registry
        
        Args:
            tracking_uri: MLflow tracking URI
        """
        self.tracking_uri = tracking_uri
        logger.info(f"Initialized MLflowModelRegistry with tracking URI: {tracking_uri}")
        
        # This is a stub implementation
        logger.warning("MLflowModelRegistry is a stub implementation")
    
    async def save_model(
        self,
        model: Any,
        metadata: ModelMetadata,
        model_path: Optional[str] = None
    ) -> str:
        """Stub implementation"""
        logger.warning("MLflowModelRegistry.save_model is not implemented")
        return metadata.version
    
    async def get_model(self, version: str) -> Optional[Any]:
        """Stub implementation"""
        logger.warning("MLflowModelRegistry.get_model is not implemented")
        return None
    
    async def get_model_path(self, version: str) -> Optional[str]:
        """Stub implementation"""
        logger.warning("MLflowModelRegistry.get_model_path is not implemented")
        return None
    
    async def get_model_metadata(self, version: str) -> Optional[ModelMetadata]:
        """Stub implementation"""
        logger.warning("MLflowModelRegistry.get_model_metadata is not implemented")
        return None
    
    async def list_models(
        self,
        model_type: Optional[ModelType] = None,
        framework: Optional[ModelFramework] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[ModelMetadata]:
        """Stub implementation"""
        logger.warning("MLflowModelRegistry.list_models is not implemented")
        return []
    
    async def get_latest_models(
        self,
        model_type: Optional[ModelType] = None,
        framework: Optional[ModelFramework] = None,
        limit: int = 1
    ) -> List[ModelMetadata]:
        """Stub implementation"""
        logger.warning("MLflowModelRegistry.get_latest_models is not implemented")
        return []
    
    async def delete_model(self, version: str) -> bool:
        """Stub implementation"""
        logger.warning("MLflowModelRegistry.delete_model is not implemented")
        return False


# Global registry instance
_model_registry = None


def get_model_registry() -> ModelRegistry:
    """
    Get or create the model registry
    
    Returns:
        ModelRegistry instance
    """
    global _model_registry
    
    if _model_registry is not None:
        return _model_registry
    
    # Determine registry type from environment
    registry_type = os.environ.get("ML_MODEL_STORAGE", "local")
    
    if registry_type == "s3":
        registry_url = os.environ.get("ML_MODEL_REGISTRY_URL", "")
        _model_registry = S3ModelRegistry(registry_url=registry_url)
    elif registry_type == "mlflow":
        tracking_uri = os.environ.get("ML_MODEL_REGISTRY_URL", "")
        _model_registry = MLflowModelRegistry(tracking_uri=tracking_uri)
    else:
        model_dir = os.environ.get("ML_MODEL_DIR", DEFAULT_MODEL_DIR)
        _model_registry = LocalModelRegistry(model_dir=model_dir)
    
    return _model_registry


async def create_dummy_model():
    """
    Create a dummy model for testing
    
    This function creates a simple sklearn model and registers it in the model registry.
    It's useful for development and testing when no real models are available.
    """
    try:
        from sklearn.ensemble import RandomForestClassifier
        import numpy as np
        
        # Create a simple model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        X = np.random.rand(100, 10)
        y = np.random.randint(0, 2, 100)
        model.fit(X, y)
        
        # Create metadata
        metadata = ModelMetadata(
            version="dummy_v1",
            model_type=ModelType.SKLEARN_RF,
            framework=ModelFramework.SKLEARN,
            created_by="system",
            description="Dummy model for testing",
            metrics={"accuracy": 0.85, "auc": 0.82},
            metadata_json=json.dumps({
                "feature_columns": [f"feature_{i}" for i in range(10)],
                "scoring_mode": "transaction",
                "feature_importance": {f"feature_{i}": 0.1 for i in range(10)}
            })
        )
        
        # Save to registry
        registry = get_model_registry()
        await registry.save_model(model, metadata)
        
        logger.info("Created dummy model for testing")
        
    except ImportError:
        logger.warning("Could not create dummy model: sklearn not installed")
    except Exception as e:
        logger.error(f"Error creating dummy model: {str(e)}")
