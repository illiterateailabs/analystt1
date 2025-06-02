"""
GNN Fraud Detection Tool - Graph Neural Network based fraud detection for financial transactions

This tool implements Graph Neural Networks (GNNs) for fraud detection in financial transaction
networks. It leverages PyTorch Geometric for GNN implementations and interfaces with Neo4j
to extract graph data for training and inference.

The tool supports multiple GNN architectures:
- Graph Convolutional Networks (GCN)
- Graph Attention Networks (GAT)
- GraphSAGE

Usage:
    - Training: Train a GNN model on historical transaction data
    - Inference: Predict fraud probability for transactions or entities
    - Subgraph Analysis: Extract and analyze suspicious subgraphs
"""

import os
import json
import logging
import pickle
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
from datetime import datetime
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, HeteroData
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch_geometric.loader import NeighborLoader
from torch_geometric.transforms import NormalizeFeatures
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from backend.core.logging import get_logger
from backend.integrations.neo4j_client import Neo4jClient

# Configure logger
logger = get_logger(__name__)

# Constants
MODEL_DIR = Path("models/gnn")
DEFAULT_HIDDEN_CHANNELS = 64
DEFAULT_NUM_LAYERS = 2
DEFAULT_DROPOUT = 0.2
DEFAULT_LEARNING_RATE = 0.01
DEFAULT_EPOCHS = 100
DEFAULT_PATIENCE = 10
DEFAULT_BATCH_SIZE = 32
DEFAULT_NUM_NEIGHBORS = [10, 10]  # For neighbor sampling


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
            out_channels: Number of output features (typically 1 for binary classification)
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
    
    def extract_subgraph(
        self, 
        entity_ids: Optional[List[str]] = None,
        transaction_ids: Optional[List[str]] = None,
        n_hops: int = 2,
        limit: int = 1000,
        include_properties: List[str] = None
    ) -> Dict:
        """
        Extract a subgraph from Neo4j centered around specified entities or transactions
        
        Args:
            entity_ids: List of entity IDs to extract subgraph for
            transaction_ids: List of transaction IDs to extract subgraph for
            n_hops: Number of hops to traverse from seed nodes
            limit: Maximum number of nodes to extract
            include_properties: List of node properties to include as features
            
        Returns:
            Dictionary containing nodes, edges, and features
        """
        if not entity_ids and not transaction_ids:
            raise ValueError("Must provide either entity_ids or transaction_ids")
        
        # Default properties to extract if none specified
        if include_properties is None:
            include_properties = [
                "amount", "timestamp", "risk_score", "account_age_days", 
                "transaction_count", "average_transaction_amount"
            ]
        
        # Build Cypher query based on seed nodes
        if entity_ids:
            match_clause = f"MATCH (n) WHERE n.id IN {json.dumps(entity_ids)}"
        else:
            match_clause = f"MATCH (n:Transaction) WHERE n.id IN {json.dumps(transaction_ids)}"
        
        # Extract subgraph with n_hops neighborhood
        query = f"""
        {match_clause}
        CALL apoc.path.subgraphAll(n, {{maxLevel: {n_hops}, limit: {limit}}})
        YIELD nodes, relationships
        RETURN nodes, relationships
        """
        
        try:
            result = self.neo4j_client.execute_query(query)
            if not result:
                logger.warning(f"No subgraph found for the provided seed nodes")
                return {"nodes": [], "edges": [], "features": {}}
            
            # Process nodes
            nodes = []
            node_features = {}
            node_id_map = {}  # Map Neo4j IDs to consecutive indices
            
            for record in result:
                for node in record["nodes"]:
                    node_id = node["id"]
                    node_type = list(node.labels)[0]  # Get primary label
                    
                    # Map node ID to index
                    if node_id not in node_id_map:
                        node_id_map[node_id] = len(node_id_map)
                    
                    # Extract node properties for features
                    features = {}
                    for prop in include_properties:
                        if prop in node:
                            # Convert to numeric if possible
                            try:
                                features[prop] = float(node[prop])
                            except (ValueError, TypeError):
                                # For non-numeric features, we would need embedding or one-hot encoding
                                # For simplicity, we'll skip non-numeric features for now
                                pass
                    
                    nodes.append({
                        "id": node_id,
                        "index": node_id_map[node_id],
                        "type": node_type,
                        "properties": node.properties,
                        "features": features
                    })
                    
                    # Store features
                    node_features[node_id_map[node_id]] = features
            
            # Process edges
            edges = []
            edge_index = [[], []]  # [source_indices, target_indices]
            
            for record in result:
                for rel in record["relationships"]:
                    source_id = rel.start_node["id"]
                    target_id = rel.end_node["id"]
                    
                    # Skip if nodes not in our map (should not happen with subgraphAll)
                    if source_id not in node_id_map or target_id not in node_id_map:
                        continue
                    
                    source_idx = node_id_map[source_id]
                    target_idx = node_id_map[target_id]
                    
                    edge_index[0].append(source_idx)
                    edge_index[1].append(target_idx)
                    
                    edges.append({
                        "source": source_id,
                        "source_idx": source_idx,
                        "target": target_id,
                        "target_idx": target_idx,
                        "type": rel.type,
                        "properties": rel.properties
                    })
            
            # Create feature matrix
            feature_keys = set()
            for features in node_features.values():
                feature_keys.update(features.keys())
            
            feature_keys = sorted(feature_keys)
            feature_matrix = np.zeros((len(node_id_map), len(feature_keys)))
            
            # Fill feature matrix
            for node_idx, features in node_features.items():
                for i, key in enumerate(feature_keys):
                    feature_matrix[node_idx, i] = features.get(key, 0.0)
            
            return {
                "nodes": nodes,
                "edges": edges,
                "edge_index": edge_index,
                "features": feature_matrix,
                "feature_names": feature_keys,
                "node_id_map": node_id_map
            }
            
        except Exception as e:
            logger.error(f"Error extracting subgraph from Neo4j: {str(e)}")
            raise
    
    def create_pyg_data(self, graph_data: Dict) -> Data:
        """
        Convert extracted graph data to PyTorch Geometric Data object
        
        Args:
            graph_data: Graph data extracted from Neo4j
            
        Returns:
            PyTorch Geometric Data object
        """
        # Extract components
        features = torch.tensor(graph_data["features"], dtype=torch.float)
        edge_index = torch.tensor(graph_data["edge_index"], dtype=torch.long)
        
        # Create PyG Data object
        data = Data(x=features, edge_index=edge_index)
        
        # Add node mapping for interpretation
        data.node_id_map = graph_data["node_id_map"]
        data.feature_names = graph_data["feature_names"]
        
        return data


class GNNFraudDetectionTool:
    """
    Tool for detecting fraud using Graph Neural Networks
    
    This tool provides capabilities to:
    1. Train GNN models on historical transaction data
    2. Predict fraud probability for new transactions or entities
    3. Extract and analyze suspicious subgraphs
    4. Explain model predictions
    """
    
    name = "gnn_fraud_detection_tool"
    description = "Detects potential fraud using Graph Neural Networks"
    
    def __init__(
        self,
        neo4j_client: Optional[Neo4jClient] = None,
        model_path: Optional[str] = None,
        architecture: GNNArchitecture = GNNArchitecture.GCN,
        device: Optional[str] = None,
    ):
        """
        Initialize the GNN fraud detection tool
        
        Args:
            neo4j_client: Neo4j client for database access
            model_path: Path to a pretrained model file
            architecture: GNN architecture to use
            device: Device to run the model on ('cpu' or 'cuda')
        """
        # Initialize Neo4j client if not provided
        if neo4j_client is None:
            self.neo4j_client = Neo4jClient()
        else:
            self.neo4j_client = neo4j_client
        
        # Initialize graph data processor
        self.data_processor = GraphDataProcessor(self.neo4j_client)
        
        # Set device (use CUDA if available)
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # Set architecture
        self.architecture = architecture
        
        # Initialize model
        self.model = None
        self.model_info = {}
        
        # Load pretrained model if provided
        if model_path:
            self.load_model(model_path)
    
    def run(self, **kwargs):
        """
        Run the GNN fraud detection tool
        
        Args:
            mode: Operation mode ('train', 'predict', 'analyze')
            entity_ids: List of entity IDs to analyze
            transaction_ids: List of transaction IDs to analyze
            model_path: Path to save/load model
            **kwargs: Additional arguments specific to each mode
            
        Returns:
            Results based on the operation mode
        """
        mode = kwargs.get('mode', 'predict')
        
        try:
            if mode == 'train':
                return self._train_mode(**kwargs)
            elif mode == 'predict':
                return self._predict_mode(**kwargs)
            elif mode == 'analyze':
                return self._analyze_mode(**kwargs)
            else:
                raise ValueError(f"Unsupported mode: {mode}")
        except Exception as e:
            logger.error(f"Error in GNN fraud detection tool: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "mode": mode
            }
    
    def _train_mode(self, **kwargs):
        """
        Train a GNN model on historical transaction data
        
        Args:
            query: Cypher query to extract training data
            label_field: Node property containing fraud labels
            test_size: Fraction of data to use for testing
            hidden_channels: Number of hidden channels
            num_layers: Number of GNN layers
            dropout: Dropout probability
            learning_rate: Learning rate for optimizer
            epochs: Number of training epochs
            model_path: Path to save the trained model
            **kwargs: Additional training parameters
            
        Returns:
            Training results and metrics
        """
        # Extract parameters
        query = kwargs.get('query')
        label_field = kwargs.get('label_field', 'is_fraud')
        test_size = kwargs.get('test_size', 0.2)
        hidden_channels = kwargs.get('hidden_channels', DEFAULT_HIDDEN_CHANNELS)
        num_layers = kwargs.get('num_layers', DEFAULT_NUM_LAYERS)
        dropout = kwargs.get('dropout', DEFAULT_DROPOUT)
        learning_rate = kwargs.get('learning_rate', DEFAULT_LEARNING_RATE)
        epochs = kwargs.get('epochs', DEFAULT_EPOCHS)
        patience = kwargs.get('patience', DEFAULT_PATIENCE)
        model_path = kwargs.get('model_path')
        
        # Extract graph data for training
        if query:
            logger.info(f"Extracting training data with custom query")
            result = self.neo4j_client.execute_query(query)
            # Process custom query result
            # This would need custom processing based on the query structure
            raise NotImplementedError("Custom query training not yet implemented")
        else:
            # Use default query to extract labeled transaction data
            logger.info("Extracting training data with default query")
            query = f"""
            MATCH (t:Transaction)
            WHERE t.{label_field} IS NOT NULL
            WITH t LIMIT 10000
            CALL apoc.path.subgraphAll(t, {{maxLevel: 2, limit: 100}})
            YIELD nodes, relationships
            RETURN t.id AS transaction_id, t.{label_field} AS label, nodes, relationships
            """
            result = self.neo4j_client.execute_query(query)
            
            # Process results into a dataset
            transaction_data = []
            for record in result:
                # Extract subgraph for each transaction
                subgraph = self.data_processor.extract_subgraph(
                    transaction_ids=[record['transaction_id']]
                )
                
                # Add label
                label = 1 if record['label'] else 0
                
                transaction_data.append({
                    'transaction_id': record['transaction_id'],
                    'subgraph': subgraph,
                    'label': label
                })
        
        # Split into train/test sets
        train_data, test_data = train_test_split(
            transaction_data, test_size=test_size, stratify=[d['label'] for d in transaction_data]
        )
        
        logger.info(f"Training set: {len(train_data)} samples, Test set: {len(test_data)} samples")
        
        # Create PyG data objects
        train_dataset = [
            self.data_processor.create_pyg_data(d['subgraph']) for d in train_data
        ]
        test_dataset = [
            self.data_processor.create_pyg_data(d['subgraph']) for d in test_data
        ]
        
        # Determine input dimension from features
        in_channels = train_dataset[0].num_features if train_dataset else 0
        if in_channels == 0:
            raise ValueError("No features found in training data")
        
        # Create model
        model = GNNModel(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=1,  # Binary classification
            num_layers=num_layers,
            dropout=dropout,
            architecture=self.architecture
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
        
        logger.info(f"Starting training for {epochs} epochs")
        for epoch in range(epochs):
            # Training
            model.train()
            total_loss = 0
            
            for data in train_dataset:
                data = data.to(self.device)
                optimizer.zero_grad()
                
                # Forward pass
                out = model(data.x, data.edge_index)
                
                # Loss computation (assuming node 0 is the transaction node)
                loss = criterion(out[0], torch.tensor([[train_data[0]['label']]], dtype=torch.float).to(self.device))
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_dataset)
            train_losses.append(avg_loss)
            
            # Validation
            model.eval()
            y_true = []
            y_pred = []
            
            with torch.no_grad():
                for i, data in enumerate(test_dataset):
                    data = data.to(self.device)
                    out = model(data.x, data.edge_index)
                    y_true.append(test_data[i]['label'])
                    y_pred.append(torch.sigmoid(out[0]).cpu().numpy()[0][0])
            
            # Compute metrics
            auc = roc_auc_score(y_true, y_pred)
            ap = average_precision_score(y_true, y_pred)
            
            val_metrics.append({
                'epoch': epoch,
                'loss': avg_loss,
                'auc': auc,
                'ap': ap
            })
            
            logger.info(f"Epoch {epoch}: Loss: {avg_loss:.4f}, AUC: {auc:.4f}, AP: {ap:.4f}")
            
            # Early stopping
            if auc > best_val_auc:
                best_val_auc = auc
                best_epoch = epoch
                epochs_no_improve = 0
                
                # Save best model
                self.model = model
                self.model_info = {
                    'architecture': self.architecture,
                    'in_channels': in_channels,
                    'hidden_channels': hidden_channels,
                    'num_layers': num_layers,
                    'dropout': dropout,
                    'best_epoch': best_epoch,
                    'best_val_auc': best_val_auc,
                    'feature_names': train_dataset[0].feature_names if hasattr(train_dataset[0], 'feature_names') else None,
                    'training_date': datetime.now().isoformat()
                }
                
                if model_path:
                    self.save_model(model_path)
            else:
                epochs_no_improve += 1
                
            # Early stopping check
            if epochs_no_improve >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
        
        # Final evaluation on test set
        model.eval()
        y_true = []
        y_pred = []
        
        with torch.no_grad():
            for i, data in enumerate(test_dataset):
                data = data.to(self.device)
                out = model(data.x, data.edge_index)
                y_true.append(test_data[i]['label'])
                y_pred.append(torch.sigmoid(out[0]).cpu().numpy()[0][0])
        
        # Compute final metrics
        final_auc = roc_auc_score(y_true, y_pred)
        final_ap = average_precision_score(y_true, y_pred)
        
        # Compute precision-recall curve
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
        
        # Return training results
        return {
            "success": True,
            "mode": "train",
            "metrics": {
                "auc": final_auc,
                "average_precision": final_ap,
                "best_epoch": best_epoch,
                "best_val_auc": best_val_auc
            },
            "model_info": self.model_info,
            "precision_recall": {
                "precision": precision.tolist(),
                "recall": recall.tolist(),
                "thresholds": thresholds.tolist() if len(thresholds) > 0 else []
            },
            "training_history": {
                "losses": train_losses,
                "val_metrics": val_metrics
            },
            "model_path": model_path if model_path else None
        }
    
    def _predict_mode(self, **kwargs):
        """
        Predict fraud probability for transactions or entities
        
        Args:
            entity_ids: List of entity IDs to analyze
            transaction_ids: List of transaction IDs to analyze
            threshold: Probability threshold for fraud classification
            explain: Whether to include explanation of predictions
            **kwargs: Additional prediction parameters
            
        Returns:
            Prediction results
        """
        # Check if model is loaded
        if self.model is None:
            raise ValueError("No model loaded. Please train or load a model first.")
        
        # Extract parameters
        entity_ids = kwargs.get('entity_ids')
        transaction_ids = kwargs.get('transaction_ids')
        threshold = kwargs.get('threshold', 0.5)
        explain = kwargs.get('explain', False)
        
        # Extract subgraph for prediction
        subgraph = self.data_processor.extract_subgraph(
            entity_ids=entity_ids,
            transaction_ids=transaction_ids
        )
        
        # Convert to PyG data
        data = self.data_processor.create_pyg_data(subgraph)
        data = data.to(self.device)
        
        # Run prediction
        self.model.eval()
        with torch.no_grad():
            out = self.model(data.x, data.edge_index)
            probabilities = torch.sigmoid(out).cpu().numpy()
        
        # Map predictions back to nodes
        predictions = []
        for i, node in enumerate(subgraph["nodes"]):
            node_id = node["id"]
            node_type = node["type"]
            
            predictions.append({
                "id": node_id,
                "type": node_type,
                "fraud_probability": float(probabilities[i][0]),
                "is_fraud": float(probabilities[i][0]) >= threshold,
                "properties": node["properties"]
            })
        
        # Sort by fraud probability (descending)
        predictions.sort(key=lambda x: x["fraud_probability"], reverse=True)
        
        # Generate explanations if requested
        explanations = None
        if explain:
            # Implement GNNExplainer or similar for explanations
            # This is a simplified placeholder
            explanations = {
                "method": "Feature importance",
                "note": "Explanation functionality is simplified in this version"
            }
        
        return {
            "success": True,
            "mode": "predict",
            "predictions": predictions,
            "threshold": threshold,
            "subgraph_size": {
                "nodes": len(subgraph["nodes"]),
                "edges": len(subgraph["edges"])
            },
            "explanations": explanations if explain else None
        }
    
    def _analyze_mode(self, **kwargs):
        """
        Analyze a subgraph for fraud patterns
        
        Args:
            entity_ids: List of entity IDs to analyze
            transaction_ids: List of transaction IDs to analyze
            n_hops: Number of hops to traverse from seed nodes
            **kwargs: Additional analysis parameters
            
        Returns:
            Analysis results
        """
        # Extract parameters
        entity_ids = kwargs.get('entity_ids')
        transaction_ids = kwargs.get('transaction_ids')
        n_hops = kwargs.get('n_hops', 2)
        
        # Extract subgraph for analysis
        subgraph = self.data_processor.extract_subgraph(
            entity_ids=entity_ids,
            transaction_ids=transaction_ids,
            n_hops=n_hops
        )
        
        # Run prediction if model is available
        if self.model is not None:
            # Convert to PyG data
            data = self.data_processor.create_pyg_data(subgraph)
            data = data.to(self.device)
            
            # Run prediction
            self.model.eval()
            with torch.no_grad():
                out = self.model(data.x, data.edge_index)
                probabilities = torch.sigmoid(out).cpu().numpy()
            
            # Add predictions to nodes
            for i, node in enumerate(subgraph["nodes"]):
                node["fraud_probability"] = float(probabilities[i][0])
        
        # Compute graph metrics
        metrics = self._compute_graph_metrics(subgraph)
        
        # Identify suspicious patterns
        patterns = self._identify_suspicious_patterns(subgraph)
        
        return {
            "success": True,
            "mode": "analyze",
            "subgraph": {
                "nodes": len(subgraph["nodes"]),
                "edges": len(subgraph["edges"]),
                "node_types": self._count_node_types(subgraph["nodes"])
            },
            "metrics": metrics,
            "suspicious_patterns": patterns,
            "visualization_data": self._prepare_visualization_data(subgraph)
        }
    
    def _compute_graph_metrics(self, subgraph: Dict) -> Dict:
        """Compute metrics for the subgraph"""
        # This is a simplified implementation
        # In a full version, we would compute centrality, clustering, etc.
        
        nodes = subgraph["nodes"]
        edges = subgraph["edges"]
        
        # Basic metrics
        metrics = {
            "node_count": len(nodes),
            "edge_count": len(edges),
            "density": len(edges) / (len(nodes) * (len(nodes) - 1)) if len(nodes) > 1 else 0,
            "average_degree": 2 * len(edges) / len(nodes) if len(nodes) > 0 else 0
        }
        
        return metrics
    
    def _count_node_types(self, nodes: List[Dict]) -> Dict:
        """Count nodes by type"""
        counts = {}
        for node in nodes:
            node_type = node["type"]
            counts[node_type] = counts.get(node_type, 0) + 1
        return counts
    
    def _identify_suspicious_patterns(self, subgraph: Dict) -> List[Dict]:
        """Identify suspicious patterns in the subgraph"""
        # This is a simplified implementation
        # In a full version, we would implement pattern recognition algorithms
        
        patterns = []
        
        # Example pattern: Circular transactions
        # In a real implementation, this would be much more sophisticated
        if len(subgraph["nodes"]) > 3 and len(subgraph["edges"]) > 3:
            patterns.append({
                "pattern": "potential_circular_flow",
                "description": "Potential circular flow of funds detected",
                "confidence": 0.7,
                "affected_nodes": [node["id"] for node in subgraph["nodes"][:3]]
            })
        
        return patterns
    
    def _prepare_visualization_data(self, subgraph: Dict) -> Dict:
        """Prepare data for visualization"""
        # Format data for visualization libraries like vis.js
        vis_nodes = []
        vis_edges = []
        
        # Process nodes
        for node in subgraph["nodes"]:
            node_data = {
                "id": node["id"],
                "label": f"{node['type']}",
                "group": node["type"]
            }
            
            # Add fraud probability if available
            if "fraud_probability" in node:
                node_data["fraud_probability"] = node["fraud_probability"]
                # Color based on fraud probability
                if node["fraud_probability"] > 0.7:
                    node_data["color"] = "#ff0000"  # Red
                elif node["fraud_probability"] > 0.4:
                    node_data["color"] = "#ff9900"  # Orange
                else:
                    node_data["color"] = "#00cc00"  # Green
            
            vis_nodes.append(node_data)
        
        # Process edges
        for edge in subgraph["edges"]:
            vis_edges.append({
                "from": edge["source"],
                "to": edge["target"],
                "label": edge["type"]
            })
        
        return {
            "nodes": vis_nodes,
            "edges": vis_edges
        }
    
    def save_model(self, path: str):
        """Save the trained model and metadata"""
        if self.model is None:
            raise ValueError("No model to save")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model state and metadata
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_info': self.model_info
        }, path)
        
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load a trained model"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        
        # Load model state and metadata
        checkpoint = torch.load(path, map_location=self.device)
        
        # Extract model info
        model_info = checkpoint['model_info']
        
        # Create model with the same architecture
        model = GNNModel(
            in_channels=model_info['in_channels'],
            hidden_channels=model_info['hidden_channels'],
            out_channels=1,  # Binary classification
            num_layers=model_info['num_layers'],
            dropout=model_info['dropout'],
            architecture=model_info['architecture']
        ).to(self.device)
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Set model and info
        self.model = model
        self.model_info = model_info
        
        logger.info(f"Model loaded from {path}")
        return True
