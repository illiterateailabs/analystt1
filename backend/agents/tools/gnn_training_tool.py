"""
GNN Training Tool - Specialized tool for training Graph Neural Network models

This tool focuses on data preparation, model training, and evaluation for Graph Neural Networks.
It supports different training strategies, hyperparameter tuning, and comprehensive evaluation
metrics, with proper model versioning and experiment tracking.

Features:
- Data extraction and preprocessing from Neo4j
- Multiple training strategies (supervised, semi-supervised, unsupervised)
- Proper train/validation/test splits with cross-validation
- Comprehensive evaluation metrics and reporting
- Hyperparameter tuning with grid/random search
- Model versioning and experiment tracking
- Integration with existing graph databases and visualization tools
"""

import os
import json
import logging
import pickle
import uuid
import time
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
from datetime import datetime
from enum import Enum
from pathlib import Path
import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, HeteroData, Dataset
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch_geometric.loader import NeighborLoader, DataLoader
from torch_geometric.transforms import NormalizeFeatures, RandomNodeSplit
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, StratifiedKFold
from tqdm import tqdm
import optuna

from backend.core.logging import get_logger
from backend.integrations.neo4j_client import Neo4jClient

# Configure logger
logger = get_logger(__name__)

# Constants
MODEL_DIR = Path("models/gnn")
EXPERIMENT_DIR = Path("experiments/gnn")
DEFAULT_HIDDEN_CHANNELS = 64
DEFAULT_NUM_LAYERS = 2
DEFAULT_DROPOUT = 0.2
DEFAULT_LEARNING_RATE = 0.01
DEFAULT_EPOCHS = 100
DEFAULT_PATIENCE = 10
DEFAULT_BATCH_SIZE = 32
DEFAULT_NUM_NEIGHBORS = [10, 10]  # For neighbor sampling


class TrainingStrategy(str, Enum):
    """Supported training strategies"""
    SUPERVISED = "supervised"
    SEMI_SUPERVISED = "semi_supervised"
    UNSUPERVISED = "unsupervised"
    SELF_SUPERVISED = "self_supervised"


class GNNArchitecture(str, Enum):
    """Supported GNN architectures"""
    GCN = "gcn"  # Graph Convolutional Network
    GAT = "gat"  # Graph Attention Network
    SAGE = "sage"  # GraphSAGE


class GNNModel(nn.Module):
    """Base GNN model implementation supporting multiple architectures"""
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        dropout: float,
        architecture: GNNArchitecture,
    ):
        """
        Initialize GNN model
        
        Args:
            in_channels: Number of input features
            hidden_channels: Number of hidden features
            out_channels: Number of output features
            num_layers: Number of GNN layers
            dropout: Dropout probability
            architecture: GNN architecture to use (GCN, GAT, SAGE)
        """
        super().__init__()
        
        self.architecture = architecture
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Select the appropriate convolution based on architecture
        if architecture == GNNArchitecture.GCN:
            conv_layer = GCNConv
        elif architecture == GNNArchitecture.GAT:
            conv_layer = GATConv
        elif architecture == GNNArchitecture.SAGE:
            conv_layer = SAGEConv
        else:
            raise ValueError(f"Unsupported GNN architecture: {architecture}")
        
        # Input layer
        self.convs = nn.ModuleList()
        self.convs.append(conv_layer(in_channels, hidden_channels))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(conv_layer(hidden_channels, hidden_channels))
        
        # Output layer
        if num_layers > 1:
            self.convs.append(conv_layer(hidden_channels, hidden_channels))
        
        # Final MLP for prediction
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels)
        )
    
    def forward(self, x, edge_index):
        """Forward pass through the GNN model"""
        # Initial feature transformation
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Final GNN layer
        if self.num_layers > 0:
            x = self.convs[-1](x, edge_index)
        
        # MLP for final prediction
        x = self.mlp(x)
        
        return x


class GraphDataProcessor:
    """Processes Neo4j graph data into PyTorch Geometric format"""
    
    def __init__(self, neo4j_client: Neo4jClient):
        """
        Initialize the graph data processor
        
        Args:
            neo4j_client: Neo4j client for database access
        """
        self.neo4j_client = neo4j_client
    
    def extract_data(
        self,
        query: str,
        label_field: str = "is_fraud",
        feature_fields: Optional[List[str]] = None,
        node_type_field: str = "labels",
        limit: int = 10000
    ) -> Dict:
        """
        Extract data from Neo4j using a custom query
        
        Args:
            query: Cypher query to extract data
            label_field: Field containing the target labels
            feature_fields: Fields to use as node features
            node_type_field: Field containing node types/labels
            limit: Maximum number of nodes to extract
            
        Returns:
            Dictionary containing extracted data
        """
        try:
            result = self.neo4j_client.execute_query(query)
            if not result:
                logger.warning("No data found with the provided query")
                return {"nodes": [], "edges": [], "features": {}, "labels": []}
            
            # Process query results
            nodes = []
            edges = []
            features = {}
            labels = []
            node_id_map = {}  # Map Neo4j IDs to consecutive indices
            
            # Process each record
            for record in result:
                # Extract nodes and edges from the record
                # This assumes the query returns nodes and relationships
                # Adjust based on actual query structure
                if "nodes" in record and "relationships" in record:
                    # Process nodes
                    for node in record["nodes"]:
                        node_id = node.id if hasattr(node, "id") else str(node)
                        
                        # Skip if already processed
                        if node_id in node_id_map:
                            continue
                        
                        # Map node ID to index
                        node_id_map[node_id] = len(node_id_map)
                        
                        # Extract node type
                        if hasattr(node, "labels"):
                            node_type = list(node.labels)[0]  # Get primary label
                        else:
                            node_type = "Unknown"
                        
                        # Extract node features
                        node_features = {}
                        if feature_fields:
                            for field in feature_fields:
                                if hasattr(node, field):
                                    try:
                                        node_features[field] = float(getattr(node, field))
                                    except (ValueError, TypeError):
                                        # Skip non-numeric features
                                        pass
                        
                        # Extract label if available
                        label = None
                        if hasattr(node, label_field):
                            try:
                                label = float(getattr(node, label_field))
                                labels.append((node_id_map[node_id], label))
                            except (ValueError, TypeError):
                                pass
                        
                        nodes.append({
                            "id": node_id,
                            "index": node_id_map[node_id],
                            "type": node_type,
                            "features": node_features
                        })
                        
                        # Store features
                        features[node_id_map[node_id]] = node_features
                    
                    # Process relationships
                    for rel in record["relationships"]:
                        source_id = rel.start_node.id if hasattr(rel.start_node, "id") else str(rel.start_node)
                        target_id = rel.end_node.id if hasattr(rel.end_node, "id") else str(rel.end_node)
                        
                        # Skip if nodes not in our map
                        if source_id not in node_id_map or target_id not in node_id_map:
                            continue
                        
                        source_idx = node_id_map[source_id]
                        target_idx = node_id_map[target_id]
                        
                        edges.append({
                            "source": source_id,
                            "source_idx": source_idx,
                            "target": target_id,
                            "target_idx": target_idx,
                            "type": rel.type if hasattr(rel, "type") else "Unknown"
                        })
            
            # Create edge index
            edge_index = [[], []]  # [source_indices, target_indices]
            for edge in edges:
                edge_index[0].append(edge["source_idx"])
                edge_index[1].append(edge["target_idx"])
            
            # Create feature matrix
            feature_keys = set()
            for node_features in features.values():
                feature_keys.update(node_features.keys())
            
            feature_keys = sorted(feature_keys)
            feature_matrix = np.zeros((len(node_id_map), len(feature_keys)))
            
            # Fill feature matrix
            for node_idx, node_features in features.items():
                for i, key in enumerate(feature_keys):
                    feature_matrix[node_idx, i] = node_features.get(key, 0.0)
            
            # Create label array
            label_array = np.zeros(len(node_id_map))
            for node_idx, label in labels:
                label_array[node_idx] = label
            
            return {
                "nodes": nodes,
                "edges": edges,
                "edge_index": edge_index,
                "features": feature_matrix,
                "feature_names": feature_keys,
                "labels": label_array,
                "node_id_map": node_id_map
            }
            
        except Exception as e:
            logger.error(f"Error extracting data from Neo4j: {str(e)}")
            raise
    
    def create_pyg_data(
        self, 
        graph_data: Dict,
        mask_ratio: float = 0.2
    ) -> Data:
        """
        Convert extracted graph data to PyTorch Geometric Data object
        
        Args:
            graph_data: Graph data extracted from Neo4j
            mask_ratio: Ratio of nodes to mask for semi-supervised learning
            
        Returns:
            PyTorch Geometric Data object
        """
        # Extract components
        features = torch.tensor(graph_data["features"], dtype=torch.float)
        edge_index = torch.tensor(graph_data["edge_index"], dtype=torch.long)
        
        # Create PyG Data object
        data = Data(x=features, edge_index=edge_index)
        
        # Add labels if available
        if "labels" in graph_data and len(graph_data["labels"]) > 0:
            labels = torch.tensor(graph_data["labels"], dtype=torch.float)
            data.y = labels
            
            # Create train/val/test masks for semi-supervised learning
            num_nodes = features.size(0)
            indices = torch.randperm(num_nodes)
            
            # Create masks
            train_size = int((1 - 2 * mask_ratio) * num_nodes)
            val_size = int(mask_ratio * num_nodes)
            
            train_mask = torch.zeros(num_nodes, dtype=torch.bool)
            val_mask = torch.zeros(num_nodes, dtype=torch.bool)
            test_mask = torch.zeros(num_nodes, dtype=torch.bool)
            
            train_mask[indices[:train_size]] = True
            val_mask[indices[train_size:train_size + val_size]] = True
            test_mask[indices[train_size + val_size:]] = True
            
            data.train_mask = train_mask
            data.val_mask = val_mask
            data.test_mask = test_mask
        
        # Add metadata
        data.node_id_map = graph_data["node_id_map"]
        data.feature_names = graph_data["feature_names"]
        
        return data
    
    def create_dataset(
        self,
        data_list: List[Dict],
        transform: Optional[Callable] = None
    ) -> List[Data]:
        """
        Create a dataset from a list of graph data dictionaries
        
        Args:
            data_list: List of graph data dictionaries
            transform: Optional transform to apply to each data object
            
        Returns:
            List of PyTorch Geometric Data objects
        """
        dataset = []
        for graph_data in data_list:
            data = self.create_pyg_data(graph_data)
            if transform:
                data = transform(data)
            dataset.append(data)
        
        return dataset


class HyperparameterTuner:
    """Handles hyperparameter tuning for GNN models"""
    
    def __init__(
        self,
        data: Data,
        architecture: GNNArchitecture,
        strategy: TrainingStrategy,
        n_trials: int = 20,
        timeout: Optional[int] = None,
        metric: str = "auc"
    ):
        """
        Initialize the hyperparameter tuner
        
        Args:
            data: PyTorch Geometric Data object
            architecture: GNN architecture to use
            strategy: Training strategy to use
            n_trials: Number of trials for hyperparameter search
            timeout: Maximum time for hyperparameter search (seconds)
            metric: Metric to optimize ('auc', 'f1', 'loss')
        """
        self.data = data
        self.architecture = architecture
        self.strategy = strategy
        self.n_trials = n_trials
        self.timeout = timeout
        self.metric = metric
        
        # Setup device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def objective(self, trial):
        """Objective function for Optuna"""
        # Sample hyperparameters
        hidden_channels = trial.suggest_int("hidden_channels", 16, 128, log=True)
        num_layers = trial.suggest_int("num_layers", 2, 4)
        dropout = trial.suggest_float("dropout", 0.1, 0.5)
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
        
        # Create model
        model = GNNModel(
            in_channels=self.data.num_features,
            hidden_channels=hidden_channels,
            out_channels=1,  # Binary classification
            num_layers=num_layers,
            dropout=dropout,
            architecture=self.architecture
        ).to(self.device)
        
        # Setup training
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = torch.nn.BCEWithLogitsLoss()
        
        # Move data to device
        data = self.data.to(self.device)
        
        # Training loop
        patience = 10
        best_val_metric = 0 if self.metric != "loss" else float("inf")
        epochs_no_improve = 0
        
        for epoch in range(50):  # Reduced epochs for hyperparameter search
            # Training
            model.train()
            optimizer.zero_grad()
            
            # Forward pass
            out = model(data.x, data.edge_index)
            
            # Loss computation
            if self.strategy == TrainingStrategy.SUPERVISED:
                # Full supervision - use all labels
                loss = criterion(out.squeeze(), data.y)
            else:
                # Semi-supervised - use only training mask
                loss = criterion(out[data.train_mask].squeeze(), data.y[data.train_mask])
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Validation
            model.eval()
            with torch.no_grad():
                out = model(data.x, data.edge_index)
                
                # Compute validation metric
                if self.strategy == TrainingStrategy.SUPERVISED:
                    # Use test split
                    val_mask = data.test_mask
                else:
                    # Use validation mask
                    val_mask = data.val_mask
                
                val_out = out[val_mask].squeeze().cpu().numpy()
                val_y = data.y[val_mask].cpu().numpy()
                
                if self.metric == "auc":
                    val_metric = roc_auc_score(val_y, val_out)
                elif self.metric == "f1":
                    val_preds = (val_out > 0).astype(int)
                    val_metric = f1_score(val_y, val_preds)
                else:  # loss
                    val_loss = criterion(out[val_mask].squeeze(), data.y[val_mask])
                    val_metric = val_loss.item()
            
            # Early stopping
            if (self.metric != "loss" and val_metric > best_val_metric) or \
               (self.metric == "loss" and val_metric < best_val_metric):
                best_val_metric = val_metric
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            
            if epochs_no_improve >= patience:
                break
        
        # Return best validation metric
        return best_val_metric if self.metric != "loss" else -best_val_metric
    
    def tune(self) -> Dict:
        """
        Run hyperparameter tuning
        
        Returns:
            Dictionary with best hyperparameters and metrics
        """
        study = optuna.create_study(
            direction="maximize" if self.metric != "loss" else "minimize"
        )
        
        study.optimize(
            self.objective,
            n_trials=self.n_trials,
            timeout=self.timeout
        )
        
        # Get best hyperparameters
        best_params = study.best_params
        best_value = study.best_value
        
        # Add fixed parameters
        best_params["architecture"] = self.architecture
        best_params["strategy"] = self.strategy
        
        return {
            "best_params": best_params,
            "best_value": best_value,
            "metric": self.metric,
            "n_trials": self.n_trials,
            "study_summary": str(study.trials_dataframe())
        }


class ExperimentTracker:
    """Tracks and manages GNN training experiments"""
    
    def __init__(self, base_dir: Optional[str] = None):
        """
        Initialize the experiment tracker
        
        Args:
            base_dir: Base directory for experiment tracking
        """
        if base_dir:
            self.base_dir = Path(base_dir)
        else:
            self.base_dir = EXPERIMENT_DIR
        
        # Create directory if it doesn't exist
        os.makedirs(self.base_dir, exist_ok=True)
    
    def create_experiment(self, name: Optional[str] = None) -> str:
        """
        Create a new experiment
        
        Args:
            name: Optional name for the experiment
            
        Returns:
            Experiment ID
        """
        # Generate experiment ID
        experiment_id = str(uuid.uuid4())[:8]
        if name:
            experiment_id = f"{name}_{experiment_id}"
        
        # Create experiment directory
        experiment_dir = self.base_dir / experiment_id
        os.makedirs(experiment_dir, exist_ok=True)
        
        return experiment_id
    
    def log_config(self, experiment_id: str, config: Dict):
        """
        Log experiment configuration
        
        Args:
            experiment_id: Experiment ID
            config: Configuration dictionary
        """
        experiment_dir = self.base_dir / experiment_id
        config_path = experiment_dir / "config.json"
        
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
    
    def log_metrics(self, experiment_id: str, metrics: Dict, step: Optional[int] = None):
        """
        Log experiment metrics
        
        Args:
            experiment_id: Experiment ID
            metrics: Metrics dictionary
            step: Optional step number
        """
        experiment_dir = self.base_dir / experiment_id
        metrics_path = experiment_dir / "metrics.json"
        
        # Load existing metrics if available
        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                existing_metrics = json.load(f)
        else:
            existing_metrics = []
        
        # Add step information
        metrics_entry = copy.deepcopy(metrics)
        if step is not None:
            metrics_entry["step"] = step
        metrics_entry["timestamp"] = datetime.now().isoformat()
        
        # Append new metrics
        existing_metrics.append(metrics_entry)
        
        # Save metrics
        with open(metrics_path, "w") as f:
            json.dump(existing_metrics, f, indent=2)
    
    def save_model(self, experiment_id: str, model: nn.Module, model_info: Dict):
        """
        Save model for an experiment
        
        Args:
            experiment_id: Experiment ID
            model: PyTorch model
            model_info: Model metadata
        """
        experiment_dir = self.base_dir / experiment_id
        model_path = experiment_dir / "model.pt"
        
        # Save model state and metadata
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_info': model_info
        }, model_path)
    
    def load_model(self, experiment_id: str, device: Optional[str] = None):
        """
        Load model from an experiment
        
        Args:
            experiment_id: Experiment ID
            device: Device to load the model on
            
        Returns:
            Tuple of (model, model_info)
        """
        experiment_dir = self.base_dir / experiment_id
        model_path = experiment_dir / "model.pt"
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found for experiment {experiment_id}")
        
        # Set device
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load model
        checkpoint = torch.load(model_path, map_location=device)
        model_info = checkpoint['model_info']
        
        # Create model with the same architecture
        model = GNNModel(
            in_channels=model_info['in_channels'],
            hidden_channels=model_info['hidden_channels'],
            out_channels=model_info['out_channels'],
            num_layers=model_info['num_layers'],
            dropout=model_info['dropout'],
            architecture=model_info['architecture']
        ).to(device)
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return model, model_info
    
    def get_experiments(self) -> List[Dict]:
        """
        Get list of experiments
        
        Returns:
            List of experiment dictionaries
        """
        experiments = []
        
        for experiment_dir in self.base_dir.iterdir():
            if experiment_dir.is_dir():
                experiment_id = experiment_dir.name
                
                # Load config if available
                config_path = experiment_dir / "config.json"
                config = None
                if config_path.exists():
                    with open(config_path, "r") as f:
                        config = json.load(f)
                
                # Check if model exists
                model_exists = (experiment_dir / "model.pt").exists()
                
                # Get metrics if available
                metrics_path = experiment_dir / "metrics.json"
                latest_metrics = None
                if metrics_path.exists():
                    with open(metrics_path, "r") as f:
                        metrics_list = json.load(f)
                        if metrics_list:
                            latest_metrics = metrics_list[-1]
                
                experiments.append({
                    "id": experiment_id,
                    "config": config,
                    "model_exists": model_exists,
                    "latest_metrics": latest_metrics,
                    "created_at": datetime.fromtimestamp(experiment_dir.stat().st_ctime).isoformat()
                })
        
        # Sort by creation time (newest first)
        experiments.sort(key=lambda x: x["created_at"], reverse=True)
        
        return experiments


class GNNTrainingTool:
    """
    Tool for training Graph Neural Network models
    
    This tool provides capabilities to:
    1. Extract and preprocess data from Neo4j
    2. Train GNN models with different strategies
    3. Tune hyperparameters
    4. Evaluate models with comprehensive metrics
    5. Track experiments and model versions
    """
    
    name = "gnn_training_tool"
    description = "Trains Graph Neural Network models for fraud detection"
    
    def __init__(
        self,
        neo4j_client: Optional[Neo4jClient] = None,
        experiment_tracker: Optional[ExperimentTracker] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize the GNN training tool
        
        Args:
            neo4j_client: Neo4j client for database access
            experiment_tracker: Experiment tracker for model versioning
            device: Device to run the model on ('cpu' or 'cuda')
        """
        # Initialize Neo4j client if not provided
        if neo4j_client is None:
            self.neo4j_client = Neo4jClient()
        else:
            self.neo4j_client = neo4j_client
        
        # Initialize data processor
        self.data_processor = GraphDataProcessor(self.neo4j_client)
        
        # Initialize experiment tracker
        if experiment_tracker is None:
            self.experiment_tracker = ExperimentTracker()
        else:
            self.experiment_tracker = experiment_tracker
        
        # Set device (use CUDA if available)
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        logger.info(f"GNN Training Tool initialized (device: {self.device})")
    
    def run(self, **kwargs):
        """
        Run the GNN training tool
        
        Args:
            mode: Operation mode ('extract_data', 'train', 'tune', 'evaluate', 'list_experiments')
            **kwargs: Additional arguments specific to each mode
            
        Returns:
            Results based on the operation mode
        """
        mode = kwargs.get('mode', 'train')
        
        try:
            if mode == 'extract_data':
                return self._extract_data_mode(**kwargs)
            elif mode == 'train':
                return self._train_mode(**kwargs)
            elif mode == 'tune':
                return self._tune_mode(**kwargs)
            elif mode == 'evaluate':
                return self._evaluate_mode(**kwargs)
            elif mode == 'list_experiments':
                return self._list_experiments_mode(**kwargs)
            else:
                raise ValueError(f"Unsupported mode: {mode}")
        except Exception as e:
            logger.error(f"Error in GNN training tool: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "mode": mode
            }
    
    def _extract_data_mode(self, **kwargs):
        """
        Extract data from Neo4j
        
        Args:
            query: Cypher query to extract data
            label_field: Field containing the target labels
            feature_fields: Fields to use as node features
            save_path: Path to save the extracted data
            **kwargs: Additional extraction parameters
            
        Returns:
            Extraction results
        """
        # Extract parameters
        query = kwargs.get('query')
        label_field = kwargs.get('label_field', 'is_fraud')
        feature_fields = kwargs.get('feature_fields')
        save_path = kwargs.get('save_path')
        
        if not query:
            raise ValueError("Query is required for data extraction")
        
        # Extract data
        graph_data = self.data_processor.extract_data(
            query=query,
            label_field=label_field,
            feature_fields=feature_fields
        )
        
        # Save data if path provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'wb') as f:
                pickle.dump(graph_data, f)
        
        return {
            "success": True,
            "mode": "extract_data",
            "data_summary": {
                "nodes": len(graph_data["nodes"]),
                "edges": len(graph_data["edges"]),
                "features": len(graph_data["feature_names"]),
                "labels": np.sum(graph_data["labels"] > 0) if "labels" in graph_data else 0
            },
            "feature_names": graph_data["feature_names"],
            "save_path": save_path
        }
    
    def _train_mode(self, **kwargs):
        """
        Train a GNN model
        
        Args:
            data_path: Path to the extracted data
            architecture: GNN architecture to use
            strategy: Training strategy
            experiment_name: Name for the experiment
            hidden_channels: Number of hidden channels
            num_layers: Number of GNN layers
            dropout: Dropout probability
            learning_rate: Learning rate for optimizer
            epochs: Number of training epochs
            patience: Early stopping patience
            **kwargs: Additional training parameters
            
        Returns:
            Training results and metrics
        """
        # Extract parameters
        data_path = kwargs.get('data_path')
        query = kwargs.get('query')
        architecture = kwargs.get('architecture', GNNArchitecture.GCN)
        strategy = kwargs.get('strategy', TrainingStrategy.SUPERVISED)
        experiment_name = kwargs.get('experiment_name', f"gnn_{architecture}_{strategy}")
        hidden_channels = kwargs.get('hidden_channels', DEFAULT_HIDDEN_CHANNELS)
        num_layers = kwargs.get('num_layers', DEFAULT_NUM_LAYERS)
        dropout = kwargs.get('dropout', DEFAULT_DROPOUT)
        learning_rate = kwargs.get('learning_rate', DEFAULT_LEARNING_RATE)
        epochs = kwargs.get('epochs', DEFAULT_EPOCHS)
        patience = kwargs.get('patience', DEFAULT_PATIENCE)
        
        # Load data or extract from Neo4j
        if data_path and os.path.exists(data_path):
            with open(data_path, 'rb') as f:
                graph_data = pickle.load(f)
        elif query:
            graph_data = self.data_processor.extract_data(query=query)
        else:
            raise ValueError("Either data_path or query must be provided")
        
        # Create PyG data
        data = self.data_processor.create_pyg_data(graph_data)
        data = data.to(self.device)
        
        # Create experiment
        experiment_id = self.experiment_tracker.create_experiment(experiment_name)
        
        # Log configuration
        config = {
            "architecture": architecture,
            "strategy": strategy,
            "hidden_channels": hidden_channels,
            "num_layers": num_layers,
            "dropout": dropout,
            "learning_rate": learning_rate,
            "epochs": epochs,
            "patience": patience,
            "device": self.device,
            "data_summary": {
                "nodes": data.num_nodes,
                "edges": data.num_edges,
                "features": data.num_features,
                "positive_labels": int(data.y.sum().item()) if hasattr(data, 'y') else 0
            }
        }
        self.experiment_tracker.log_config(experiment_id, config)
        
        # Create model
        model = GNNModel(
            in_channels=data.num_features,
            hidden_channels=hidden_channels,
            out_channels=1,  # Binary classification
            num_layers=num_layers,
            dropout=dropout,
            architecture=architecture
        ).to(self.device)
        
        # Setup training
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = torch.nn.BCEWithLogitsLoss()
        
        # Training loop
        best_val_auc = 0
        best_epoch = 0
        epochs_no_improve = 0
        train_losses = []
        val_metrics = []
        
        logger.info(f"Starting training for experiment {experiment_id}")
        for epoch in range(epochs):
            # Training
            model.train()
            optimizer.zero_grad()
            
            # Forward pass
            out = model(data.x, data.edge_index)
            
            # Loss computation based on strategy
            if strategy == TrainingStrategy.SUPERVISED:
                # Full supervision - use all labels
                loss = criterion(out.squeeze(), data.y)
            else:
                # Semi-supervised - use only training mask
                loss = criterion(out[data.train_mask].squeeze(), data.y[data.train_mask])
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Record training loss
            train_loss = loss.item()
            train_losses.append(train_loss)
            
            # Validation
            model.eval()
            with torch.no_grad():
                out = model(data.x, data.edge_index)
                
                # Compute validation metrics
                if strategy == TrainingStrategy.SUPERVISED:
                    # Use test split for validation in supervised setting
                    val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
                    indices = torch.randperm(data.num_nodes)
                    val_size = int(0.2 * data.num_nodes)
                    val_mask[indices[:val_size]] = True
                else:
                    # Use validation mask for semi-supervised setting
                    val_mask = data.val_mask
                
                val_out = out[val_mask].squeeze().cpu().numpy()
                val_y = data.y[val_mask].cpu().numpy()
                
                # Compute metrics
                val_auc = roc_auc_score(val_y, val_out)
                val_preds = (val_out > 0).astype(int)
                val_f1 = f1_score(val_y, val_preds)
                val_loss = criterion(out[val_mask].squeeze(), data.y[val_mask]).item()
                
                # Record validation metrics
                val_metrics.append({
                    "epoch": epoch,
                    "loss": val_loss,
                    "auc": val_auc,
                    "f1": val_f1
                })
                
                # Log metrics
                self.experiment_tracker.log_metrics(
                    experiment_id,
                    {
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "val_auc": val_auc,
                        "val_f1": val_f1
                    },
                    epoch
                )
                
                logger.info(f"Epoch {epoch}: Loss: {train_loss:.4f}, Val AUC: {val_auc:.4f}, Val F1: {val_f1:.4f}")
            
            # Early stopping
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_epoch = epoch
                epochs_no_improve = 0
                
                # Save best model
                model_info = {
                    'architecture': architecture,
                    'in_channels': data.num_features,
                    'hidden_channels': hidden_channels,
                    'out_channels': 1,
                    'num_layers': num_layers,
                    'dropout': dropout,
                    'best_epoch': best_epoch,
                    'best_val_auc': best_val_auc,
                    'feature_names': data.feature_names if hasattr(data, 'feature_names') else None,
                    'training_date': datetime.now().isoformat(),
                    'experiment_id': experiment_id
                }
                
                self.experiment_tracker.save_model(experiment_id, model, model_info)
            else:
                epochs_no_improve += 1
            
            # Early stopping check
            if epochs_no_improve >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
        
        # Final evaluation
        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            
            # Test metrics
            if strategy == TrainingStrategy.SUPERVISED:
                # Use a different test split
                test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
                indices = torch.randperm(data.num_nodes)
                test_size = int(0.2 * data.num_nodes)
                test_mask[indices[-test_size:]] = True
            else:
                # Use test mask
                test_mask = data.test_mask
            
            test_out = out[test_mask].squeeze().cpu().numpy()
            test_y = data.y[test_mask].cpu().numpy()
            
            # Compute metrics
            test_auc = roc_auc_score(test_y, test_out)
            test_preds = (test_out > 0).astype(int)
            test_f1 = f1_score(test_y, test_preds)
            
            # Compute confusion matrix
            tn, fp, fn, tp = confusion_matrix(test_y, test_preds).ravel()
            
            # Compute precision-recall curve
            precision, recall, thresholds = precision_recall_curve(test_y, test_out)
            
            # Final metrics
            final_metrics = {
                "test_auc": test_auc,
                "test_f1": test_f1,
                "test_precision": tp / (tp + fp) if (tp + fp) > 0 else 0,
                "test_recall": tp / (tp + fn) if (tp + fn) > 0 else 0,
                "test_accuracy": (tp + tn) / (tp + tn + fp + fn),
                "confusion_matrix": {
                    "true_negatives": int(tn),
                    "false_positives": int(fp),
                    "false_negatives": int(fn),
                    "true_positives": int(tp)
                }
            }
            
            # Log final metrics
            self.experiment_tracker.log_metrics(experiment_id, final_metrics)
        
        return {
            "success": True,
            "mode": "train",
            "experiment_id": experiment_id,
            "metrics": final_metrics,
            "training_history": {
                "best_epoch": best_epoch,
                "best_val_auc": best_val_auc,
                "epochs_trained": epoch + 1,
                "early_stopped": epochs_no_improve >= patience
            },
            "precision_recall": {
                "precision": precision.tolist(),
                "recall": recall.tolist(),
                "thresholds": thresholds.tolist() if len(thresholds) > 0 else []
            },
            "model_info": model_info
        }
    
    def _tune_mode(self, **kwargs):
        """
        Tune hyperparameters for a GNN model
        
        Args:
            data_path: Path to the extracted data
            query: Cypher query to extract data
            architecture: GNN architecture to use
            strategy: Training strategy
            n_trials: Number of trials for hyperparameter search
            timeout: Maximum time for hyperparameter search (seconds)
            metric: Metric to optimize ('auc', 'f1', 'loss')
            **kwargs: Additional tuning parameters
            
        Returns:
            Tuning results with best hyperparameters
        """
        # Extract parameters
        data_path = kwargs.get('data_path')
        query = kwargs.get('query')
        architecture = kwargs.get('architecture', GNNArchitecture.GCN)
        strategy = kwargs.get('strategy', TrainingStrategy.SUPERVISED)
        n_trials = kwargs.get('n_trials', 20)
        timeout = kwargs.get('timeout')
        metric = kwargs.get('metric', 'auc')
        experiment_name = kwargs.get('experiment_name', f"tune_{architecture}_{strategy}")
        
        # Load data or extract from Neo4j
        if data_path and os.path.exists(data_path):
            with open(data_path, 'rb') as f:
                graph_data = pickle.load(f)
        elif query:
            graph_data = self.data_processor.extract_data(query=query)
        else:
            raise ValueError("Either data_path or query must be provided")
        
        # Create PyG data
        data = self.data_processor.create_pyg_data(graph_data)
        data = data.to(self.device)
        
        # Create experiment
        experiment_id = self.experiment_tracker.create_experiment(experiment_name)
        
        # Log configuration
        config = {
            "mode": "tune",
            "architecture": architecture,
            "strategy": strategy,
            "n_trials": n_trials,
            "timeout": timeout,
            "metric": metric,
            "device": self.device,
            "data_summary": {
                "nodes": data.num_nodes,
                "edges": data.num_edges,
                "features": data.num_features,
                "positive_labels": int(data.y.sum().item()) if hasattr(data, 'y') else 0
            }
        }
        self.experiment_tracker.log_config(experiment_id, config)
        
        # Create hyperparameter tuner
        tuner = HyperparameterTuner(
            data=data,
            architecture=architecture,
            strategy=strategy,
            n_trials=n_trials,
            timeout=timeout,
            metric=metric
        )
        
        # Run hyperparameter tuning
        logger.info(f"Starting hyperparameter tuning for experiment {experiment_id}")
        start_time = time.time()
        tuning_results = tuner.tune()
        elapsed_time = time.time() - start_time
        
        # Log tuning results
        self.experiment_tracker.log_metrics(experiment_id, {
            "best_params": tuning_results["best_params"],
            "best_value": tuning_results["best_value"],
            "elapsed_time": elapsed_time
        })
        
        # Train model with best hyperparameters
        best_params = tuning_results["best_params"]
        
        # Return tuning results
        return {
            "success": True,
            "mode": "tune",
            "experiment_id": experiment_id,
            "best_params": best_params,
            "best_value": tuning_results["best_value"],
            "metric": metric,
            "n_trials": n_trials,
            "elapsed_time": elapsed_time,
            "next_steps": f"Train a model with these parameters using mode='train' and the best hyperparameters."
        }
    
    def _evaluate_mode(self, **kwargs):
        """
        Evaluate a trained GNN model
        
        Args:
            experiment_id: ID of the experiment with the trained model
            data_path: Path to the evaluation data
            query: Cypher query to extract evaluation data
            **kwargs: Additional evaluation parameters
            
        Returns:
            Evaluation results and metrics
        """
        # Extract parameters
        experiment_id = kwargs.get('experiment_id')
        data_path = kwargs.get('data_path')
        query = kwargs.get('query')
        
        if not experiment_id:
            raise ValueError("experiment_id is required for evaluation")
        
        # Load model
        try:
            model, model_info = self.experiment_tracker.load_model(experiment_id, self.device)
        except FileNotFoundError:
            raise ValueError(f"No model found for experiment {experiment_id}")
        
        # Load data or extract from Neo4j
        if data_path and os.path.exists(data_path):
            with open(data_path, 'rb') as f:
                graph_data = pickle.load(f)
        elif query:
            graph_data = self.data_processor.extract_data(query=query)
        else:
            raise ValueError("Either data_path or query must be provided for evaluation")
        
        # Create PyG data
        data = self.data_processor.create_pyg_data(graph_data)
        data = data.to(self.device)
        
        # Evaluate model
        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            probabilities = torch.sigmoid(out).cpu().numpy()
        
        # Compute metrics
        y_true = data.y.cpu().numpy()
        y_pred = (probabilities > 0.5).astype(int)
        
        # Basic metrics
        accuracy = (y_true == y_pred).mean()
        auc = roc_auc_score(y_true, probabilities)
        
        # Precision, recall, F1
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Detailed classification report
        class_report = classification_report(y_true, y_pred, output_dict=True)
        
        # Precision-recall curve
        precision_curve, recall_curve, thresholds = precision_recall_curve(y_true, probabilities)
        
        # Log evaluation results
        self.experiment_tracker.log_metrics(experiment_id, {
            "evaluation": {
                "accuracy": accuracy,
                "auc": auc,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "confusion_matrix": {
                    "true_negatives": int(tn),
                    "false_positives": int(fp),
                    "false_negatives": int(fn),
                    "true_positives": int(tp)
                }
            }
        })
        
        return {
            "success": True,
            "mode": "evaluate",
            "experiment_id": experiment_id,
            "metrics": {
                "accuracy": accuracy,
                "auc": auc,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "confusion_matrix": {
                    "true_negatives": int(tn),
                    "false_positives": int(fp),
                    "false_negatives": int(fn),
                    "true_positives": int(tp)
                },
                "classification_report": class_report
            },
            "precision_recall_curve": {
                "precision": precision_curve.tolist(),
                "recall": recall_curve.tolist(),
                "thresholds": thresholds.tolist() if len(thresholds) > 0 else []
            },
            "model_info": model_info
        }
    
    def _list_experiments_mode(self, **kwargs):
        """
        List available experiments
        
        Args:
            limit: Maximum number of experiments to return
            **kwargs: Additional parameters
            
        Returns:
            List of experiments
        """
        # Extract parameters
        limit = kwargs.get('limit', 10)
        
        # Get experiments
        experiments = self.experiment_tracker.get_experiments()
        
        # Apply limit
        if limit:
            experiments = experiments[:limit]
        
        return {
            "success": True,
            "mode": "list_experiments",
            "experiments": experiments,
            "count": len(experiments),
            "total": len(self.experiment_tracker.get_experiments())
        }
