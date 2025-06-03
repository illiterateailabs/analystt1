"""
Tests for GNN Fraud Detection Tool

This module contains comprehensive tests for the GNN Fraud Detection Tool, including:
- GNN model architectures (GCN, GAT, SAGE)
- Graph data processing from Neo4j
- Training, prediction, and analysis modes
- Model saving and loading
- Edge cases and error handling
"""

import os
import json
import pickle
import tempfile
from unittest.mock import MagicMock, patch, Mock, ANY
import pytest
import numpy as np
import torch
from torch_geometric.data import Data

from backend.agents.tools.gnn_fraud_detection_tool import (
    GNNModel, 
    GraphDataProcessor, 
    GNNFraudDetectionTool, 
    GNNArchitecture
)
from backend.integrations.neo4j_client import Neo4jClient


# Fixtures and helper functions
@pytest.fixture
def mock_neo4j_client():
    """Create a mock Neo4j client for testing"""
    client = MagicMock(spec=Neo4jClient)
    return client


@pytest.fixture
def mock_torch_device():
    """Mock torch device selection to always use CPU"""
    with patch('torch.cuda.is_available', return_value=False):
        yield 'cpu'


@pytest.fixture
def sample_subgraph_data():
    """Create sample subgraph data for testing"""
    return {
        "nodes": [
            {
                "id": "n1",
                "index": 0,
                "type": "Person",
                "properties": {"name": "Alice", "risk_score": 0.2},
                "features": {"risk_score": 0.2, "account_age_days": 365}
            },
            {
                "id": "n2",
                "index": 1,
                "type": "Person",
                "properties": {"name": "Bob", "risk_score": 0.7},
                "features": {"risk_score": 0.7, "account_age_days": 30}
            },
            {
                "id": "t1",
                "index": 2,
                "type": "Transaction",
                "properties": {"amount": 1000, "timestamp": "2023-01-01"},
                "features": {"amount": 1000, "timestamp": 1672531200}
            }
        ],
        "edges": [
            {
                "source": "n1",
                "source_idx": 0,
                "target": "t1",
                "target_idx": 2,
                "type": "PERFORMED",
                "properties": {}
            },
            {
                "source": "t1",
                "source_idx": 2,
                "target": "n2",
                "target_idx": 1,
                "type": "TO",
                "properties": {}
            }
        ],
        "edge_index": [[0, 2], [2, 1]],
        "features": np.array([
            [0.2, 365],
            [0.7, 30],
            [1000, 1672531200]
        ]),
        "feature_names": ["risk_score", "account_age_days"],
        "node_id_map": {"n1": 0, "n2": 1, "t1": 2}
    }


@pytest.fixture
def mock_pyg_data():
    """Create a mock PyTorch Geometric Data object"""
    data = MagicMock(spec=Data)
    data.x = torch.tensor([[0.2, 365], [0.7, 30], [1000, 1672531200]], dtype=torch.float)
    data.edge_index = torch.tensor([[0, 2], [2, 1]], dtype=torch.long)
    data.num_features = 2
    data.node_id_map = {"n1": 0, "n2": 1, "t1": 2}
    data.feature_names = ["risk_score", "account_age_days"]
    return data


@pytest.fixture
def mock_neo4j_response():
    """Create a mock Neo4j response for testing"""
    return [
        {
            "nodes": [
                {
                    "id": "n1",
                    "labels": ["Person"],
                    "properties": {"name": "Alice", "risk_score": 0.2, "account_age_days": 365}
                },
                {
                    "id": "n2",
                    "labels": ["Person"],
                    "properties": {"name": "Bob", "risk_score": 0.7, "account_age_days": 30}
                },
                {
                    "id": "t1",
                    "labels": ["Transaction"],
                    "properties": {"amount": 1000, "timestamp": "2023-01-01"}
                }
            ],
            "relationships": [
                {
                    "id": "r1",
                    "type": "PERFORMED",
                    "start_node": {"id": "n1"},
                    "end_node": {"id": "t1"},
                    "properties": {}
                },
                {
                    "id": "r2",
                    "type": "TO",
                    "start_node": {"id": "t1"},
                    "end_node": {"id": "n2"},
                    "properties": {}
                }
            ]
        }
    ]


@pytest.fixture
def mock_training_data():
    """Create mock training data for GNN model"""
    return [
        {
            'transaction_id': 't1',
            'subgraph': {
                "nodes": [
                    {"id": "n1", "index": 0, "type": "Person", "properties": {}, "features": {"risk_score": 0.2}},
                    {"id": "t1", "index": 1, "type": "Transaction", "properties": {}, "features": {"amount": 1000}}
                ],
                "edges": [
                    {"source": "n1", "source_idx": 0, "target": "t1", "target_idx": 1, "type": "PERFORMED", "properties": {}}
                ],
                "edge_index": [[0], [1]],
                "features": np.array([[0.2], [1000]]),
                "feature_names": ["value"],
                "node_id_map": {"n1": 0, "t1": 1}
            },
            'label': 0
        },
        {
            'transaction_id': 't2',
            'subgraph': {
                "nodes": [
                    {"id": "n2", "index": 0, "type": "Person", "properties": {}, "features": {"risk_score": 0.8}},
                    {"id": "t2", "index": 1, "type": "Transaction", "properties": {}, "features": {"amount": 5000}}
                ],
                "edges": [
                    {"source": "n2", "source_idx": 0, "target": "t2", "target_idx": 1, "type": "PERFORMED", "properties": {}}
                ],
                "edge_index": [[0], [1]],
                "features": np.array([[0.8], [5000]]),
                "feature_names": ["value"],
                "node_id_map": {"n2": 0, "t2": 1}
            },
            'label': 1
        }
    ]


# Test GNN Model
class TestGNNModel:
    """Tests for the GNN Model class"""
    
    def test_init_gcn(self):
        """Test GNN model initialization with GCN architecture"""
        model = GNNModel(
            in_channels=2,
            hidden_channels=16,
            out_channels=1,
            num_layers=2,
            dropout=0.1,
            architecture=GNNArchitecture.GCN
        )
        
        assert model.architecture == GNNArchitecture.GCN
        assert model.num_layers == 2
        assert model.dropout == 0.1
        assert len(model.convs) == 2
        assert isinstance(model.mlp, torch.nn.Sequential)
    
    def test_init_gat(self):
        """Test GNN model initialization with GAT architecture"""
        model = GNNModel(
            in_channels=2,
            hidden_channels=16,
            out_channels=1,
            num_layers=2,
            dropout=0.1,
            architecture=GNNArchitecture.GAT
        )
        
        assert model.architecture == GNNArchitecture.GAT
        assert len(model.convs) == 2
    
    def test_init_sage(self):
        """Test GNN model initialization with GraphSAGE architecture"""
        model = GNNModel(
            in_channels=2,
            hidden_channels=16,
            out_channels=1,
            num_layers=2,
            dropout=0.1,
            architecture=GNNArchitecture.SAGE
        )
        
        assert model.architecture == GNNArchitecture.SAGE
        assert len(model.convs) == 2
    
    def test_init_invalid_architecture(self):
        """Test GNN model initialization with invalid architecture"""
        with pytest.raises(ValueError, match="Unsupported GNN architecture"):
            GNNModel(
                in_channels=2,
                hidden_channels=16,
                out_channels=1,
                num_layers=2,
                dropout=0.1,
                architecture="invalid"
            )
    
    def test_forward(self, mock_torch_device):
        """Test forward pass of GNN model"""
        model = GNNModel(
            in_channels=2,
            hidden_channels=16,
            out_channels=1,
            num_layers=2,
            dropout=0.1,
            architecture=GNNArchitecture.GCN
        )
        
        # Create dummy input data
        x = torch.randn(3, 2)  # 3 nodes, 2 features
        edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)  # 2 edges
        
        # Forward pass
        with patch.object(torch.nn.functional, 'dropout', return_value=x):
            output = model(x, edge_index)
        
        assert output.shape == (3, 1)  # 3 nodes, 1 output feature


# Test GraphDataProcessor
class TestGraphDataProcessor:
    """Tests for the GraphDataProcessor class"""
    
    def test_init(self, mock_neo4j_client):
        """Test initialization of GraphDataProcessor"""
        processor = GraphDataProcessor(mock_neo4j_client)
        assert processor.neo4j_client == mock_neo4j_client
    
    def test_extract_subgraph_with_entity_ids(self, mock_neo4j_client, mock_neo4j_response):
        """Test extracting subgraph with entity IDs"""
        # Setup mock
        mock_neo4j_client.execute_query.return_value = mock_neo4j_response
        
        processor = GraphDataProcessor(mock_neo4j_client)
        entity_ids = ["n1", "n2"]
        
        # Call method
        result = processor.extract_subgraph(entity_ids=entity_ids)
        
        # Verify
        mock_neo4j_client.execute_query.assert_called_once()
        assert "nodes" in result
        assert "edges" in result
        assert "edge_index" in result
        assert "features" in result
        assert "feature_names" in result
        assert "node_id_map" in result
        assert len(result["nodes"]) > 0
        assert len(result["edges"]) > 0
    
    def test_extract_subgraph_with_transaction_ids(self, mock_neo4j_client, mock_neo4j_response):
        """Test extracting subgraph with transaction IDs"""
        # Setup mock
        mock_neo4j_client.execute_query.return_value = mock_neo4j_response
        
        processor = GraphDataProcessor(mock_neo4j_client)
        transaction_ids = ["t1"]
        
        # Call method
        result = processor.extract_subgraph(transaction_ids=transaction_ids)
        
        # Verify
        mock_neo4j_client.execute_query.assert_called_once()
        assert "nodes" in result
        assert "edges" in result
        assert len(result["nodes"]) > 0
    
    def test_extract_subgraph_no_ids(self, mock_neo4j_client):
        """Test extracting subgraph with no IDs raises error"""
        processor = GraphDataProcessor(mock_neo4j_client)
        
        with pytest.raises(ValueError, match="Must provide either entity_ids or transaction_ids"):
            processor.extract_subgraph()
    
    def test_extract_subgraph_empty_result(self, mock_neo4j_client):
        """Test extracting subgraph with empty result"""
        # Setup mock
        mock_neo4j_client.execute_query.return_value = []
        
        processor = GraphDataProcessor(mock_neo4j_client)
        entity_ids = ["non_existent"]
        
        # Call method
        result = processor.extract_subgraph(entity_ids=entity_ids)
        
        # Verify
        assert result["nodes"] == []
        assert result["edges"] == []
        assert result["features"] == {}
    
    def test_extract_subgraph_database_error(self, mock_neo4j_client):
        """Test extracting subgraph with database error"""
        # Setup mock
        mock_neo4j_client.execute_query.side_effect = Exception("Database error")
        
        processor = GraphDataProcessor(mock_neo4j_client)
        entity_ids = ["n1"]
        
        # Call method
        with pytest.raises(Exception, match="Database error"):
            processor.extract_subgraph(entity_ids=entity_ids)
    
    def test_create_pyg_data(self, sample_subgraph_data, mock_neo4j_client):
        """Test creating PyTorch Geometric Data object"""
        processor = GraphDataProcessor(mock_neo4j_client)
        
        # Call method
        data = processor.create_pyg_data(sample_subgraph_data)
        
        # Verify
        assert isinstance(data, Data)
        assert data.x.shape == (3, 2)  # 3 nodes, 2 features
        assert data.edge_index.shape == (2, 2)  # 2 edges
        assert hasattr(data, "node_id_map")
        assert hasattr(data, "feature_names")


# Test GNNFraudDetectionTool
class TestGNNFraudDetectionTool:
    """Tests for the GNNFraudDetectionTool class"""
    
    def test_init_default(self, mock_torch_device):
        """Test initialization with default parameters"""
        with patch('backend.agents.tools.gnn_fraud_detection_tool.Neo4jClient') as mock_neo4j:
            tool = GNNFraudDetectionTool()
            
            assert tool.neo4j_client is not None
            assert tool.data_processor is not None
            assert tool.device == 'cpu'
            assert tool.architecture == GNNArchitecture.GCN
            assert tool.model is None
    
    def test_init_with_params(self, mock_neo4j_client, mock_torch_device):
        """Test initialization with custom parameters"""
        tool = GNNFraudDetectionTool(
            neo4j_client=mock_neo4j_client,
            architecture=GNNArchitecture.GAT,
            device='cpu'
        )
        
        assert tool.neo4j_client == mock_neo4j_client
        assert tool.architecture == GNNArchitecture.GAT
        assert tool.device == 'cpu'
    
    def test_run_invalid_mode(self, mock_neo4j_client, mock_torch_device):
        """Test run method with invalid mode"""
        tool = GNNFraudDetectionTool(neo4j_client=mock_neo4j_client)
        
        result = tool.run(mode='invalid')
        
        assert result['success'] is False
        assert 'error' in result
        assert 'Unsupported mode' in result['error']
    
    def test_run_exception(self, mock_neo4j_client, mock_torch_device):
        """Test run method with exception"""
        tool = GNNFraudDetectionTool(neo4j_client=mock_neo4j_client)
        
        # Mock _train_mode to raise exception
        with patch.object(tool, '_train_mode', side_effect=Exception("Test error")):
            result = tool.run(mode='train')
        
        assert result['success'] is False
        assert result['error'] == "Test error"
        assert result['mode'] == 'train'
    
    @patch('torch.save')
    def test_save_model(self, mock_save, mock_neo4j_client, mock_torch_device):
        """Test save_model method"""
        tool = GNNFraudDetectionTool(neo4j_client=mock_neo4j_client)
        
        # Create a mock model
        tool.model = MagicMock()
        tool.model_info = {"architecture": GNNArchitecture.GCN}
        
        # Mock os.makedirs
        with patch('os.makedirs') as mock_makedirs:
            tool.save_model("models/test_model.pt")
            
            mock_makedirs.assert_called_once()
            mock_save.assert_called_once()
    
    def test_save_model_no_model(self, mock_neo4j_client, mock_torch_device):
        """Test save_model method with no model"""
        tool = GNNFraudDetectionTool(neo4j_client=mock_neo4j_client)
        
        with pytest.raises(ValueError, match="No model to save"):
            tool.save_model("models/test_model.pt")
    
    @patch('torch.load')
    def test_load_model(self, mock_load, mock_neo4j_client, mock_torch_device):
        """Test load_model method"""
        # Setup mock
        mock_load.return_value = {
            'model_state_dict': {},
            'model_info': {
                'architecture': GNNArchitecture.GCN,
                'in_channels': 2,
                'hidden_channels': 16,
                'num_layers': 2,
                'dropout': 0.1
            }
        }
        
        # Mock os.path.exists
        with patch('os.path.exists', return_value=True):
            tool = GNNFraudDetectionTool(neo4j_client=mock_neo4j_client)
            result = tool.load_model("models/test_model.pt")
            
            assert result is True
            assert tool.model is not None
            assert tool.model_info is not None
    
    def test_load_model_file_not_found(self, mock_neo4j_client, mock_torch_device):
        """Test load_model method with file not found"""
        # Mock os.path.exists
        with patch('os.path.exists', return_value=False):
            tool = GNNFraudDetectionTool(neo4j_client=mock_neo4j_client)
            
            with pytest.raises(FileNotFoundError):
                tool.load_model("models/non_existent_model.pt")
    
    @patch('backend.agents.tools.gnn_fraud_detection_tool.train_test_split')
    def test_train_mode(self, mock_split, mock_neo4j_client, mock_torch_device, mock_training_data, mock_pyg_data):
        """Test _train_mode method"""
        # Setup mocks
        mock_split.return_value = (mock_training_data[:1], mock_training_data[1:])
        mock_neo4j_client.execute_query.return_value = [
            {"transaction_id": "t1", "label": 0, "nodes": [], "relationships": []},
            {"transaction_id": "t2", "label": 1, "nodes": [], "relationships": []}
        ]
        
        tool = GNNFraudDetectionTool(neo4j_client=mock_neo4j_client)
        
        # Mock data_processor.extract_subgraph and create_pyg_data
        with patch.object(tool.data_processor, 'extract_subgraph', return_value=mock_training_data[0]['subgraph']), \
             patch.object(tool.data_processor, 'create_pyg_data', return_value=mock_pyg_data), \
             patch('torch.sigmoid', return_value=torch.tensor([[0.2]])), \
             patch('sklearn.metrics.roc_auc_score', return_value=0.8), \
             patch('sklearn.metrics.average_precision_score', return_value=0.7), \
             patch('sklearn.metrics.precision_recall_curve', return_value=([0.7, 0.8], [0.6, 0.5], [0.3, 0.4])), \
             patch('torch.save'):
            
            result = tool._train_mode(
                hidden_channels=16,
                num_layers=2,
                dropout=0.1,
                learning_rate=0.01,
                epochs=5,
                patience=2,
                model_path="models/test_model.pt"
            )
            
            assert result['success'] is True
            assert result['mode'] == 'train'
            assert 'metrics' in result
            assert 'model_info' in result
            assert 'precision_recall' in result
            assert 'training_history' in result
            assert result['model_path'] == "models/test_model.pt"
    
    def test_predict_mode_no_model(self, mock_neo4j_client, mock_torch_device):
        """Test _predict_mode method with no model"""
        tool = GNNFraudDetectionTool(neo4j_client=mock_neo4j_client)
        
        with pytest.raises(ValueError, match="No model loaded"):
            tool._predict_mode(entity_ids=["n1"])
    
    def test_predict_mode(self, mock_neo4j_client, mock_torch_device, sample_subgraph_data, mock_pyg_data):
        """Test _predict_mode method"""
        tool = GNNFraudDetectionTool(neo4j_client=mock_neo4j_client)
        
        # Create a mock model
        tool.model = MagicMock()
        
        # Mock data_processor methods
        with patch.object(tool.data_processor, 'extract_subgraph', return_value=sample_subgraph_data), \
             patch.object(tool.data_processor, 'create_pyg_data', return_value=mock_pyg_data), \
             patch('torch.sigmoid', return_value=torch.tensor([[0.2], [0.8], [0.4]])):
            
            result = tool._predict_mode(
                entity_ids=["n1", "n2"],
                threshold=0.5,
                explain=True
            )
            
            assert result['success'] is True
            assert result['mode'] == 'predict'
            assert 'predictions' in result
            assert len(result['predictions']) == 3
            assert result['threshold'] == 0.5
            assert 'explanations' in result
    
    def test_analyze_mode(self, mock_neo4j_client, mock_torch_device, sample_subgraph_data):
        """Test _analyze_mode method"""
        tool = GNNFraudDetectionTool(neo4j_client=mock_neo4j_client)
        
        # Mock data_processor methods
        with patch.object(tool.data_processor, 'extract_subgraph', return_value=sample_subgraph_data), \
             patch.object(tool, '_compute_graph_metrics', return_value={"density": 0.3}), \
             patch.object(tool, '_identify_suspicious_patterns', return_value=[{"pattern": "circular"}]), \
             patch.object(tool, '_prepare_visualization_data', return_value={"nodes": [], "edges": []}):
            
            result = tool._analyze_mode(
                entity_ids=["n1", "n2"],
                n_hops=2
            )
            
            assert result['success'] is True
            assert result['mode'] == 'analyze'
            assert 'subgraph' in result
            assert 'metrics' in result
            assert 'suspicious_patterns' in result
            assert 'visualization_data' in result
    
    def test_analyze_mode_with_model(self, mock_neo4j_client, mock_torch_device, sample_subgraph_data, mock_pyg_data):
        """Test _analyze_mode method with model"""
        tool = GNNFraudDetectionTool(neo4j_client=mock_neo4j_client)
        
        # Create a mock model
        tool.model = MagicMock()
        
        # Mock data_processor methods
        with patch.object(tool.data_processor, 'extract_subgraph', return_value=sample_subgraph_data), \
             patch.object(tool.data_processor, 'create_pyg_data', return_value=mock_pyg_data), \
             patch('torch.sigmoid', return_value=torch.tensor([[0.2], [0.8], [0.4]])), \
             patch.object(tool, '_compute_graph_metrics', return_value={"density": 0.3}), \
             patch.object(tool, '_identify_suspicious_patterns', return_value=[{"pattern": "circular"}]), \
             patch.object(tool, '_prepare_visualization_data', return_value={"nodes": [], "edges": []}):
            
            result = tool._analyze_mode(
                entity_ids=["n1", "n2"],
                n_hops=2
            )
            
            assert result['success'] is True
            assert result['mode'] == 'analyze'
            assert 'subgraph' in result
            assert 'metrics' in result
            assert 'suspicious_patterns' in result
            assert 'visualization_data' in result
    
    def test_compute_graph_metrics(self, mock_neo4j_client, mock_torch_device, sample_subgraph_data):
        """Test _compute_graph_metrics method"""
        tool = GNNFraudDetectionTool(neo4j_client=mock_neo4j_client)
        
        metrics = tool._compute_graph_metrics(sample_subgraph_data)
        
        assert 'node_count' in metrics
        assert 'edge_count' in metrics
        assert 'density' in metrics
        assert 'average_degree' in metrics
    
    def test_count_node_types(self, mock_neo4j_client, mock_torch_device):
        """Test _count_node_types method"""
        tool = GNNFraudDetectionTool(neo4j_client=mock_neo4j_client)
        
        nodes = [
            {"type": "Person"},
            {"type": "Person"},
            {"type": "Transaction"}
        ]
        
        counts = tool._count_node_types(nodes)
        
        assert counts["Person"] == 2
        assert counts["Transaction"] == 1
    
    def test_identify_suspicious_patterns(self, mock_neo4j_client, mock_torch_device, sample_subgraph_data):
        """Test _identify_suspicious_patterns method"""
        tool = GNNFraudDetectionTool(neo4j_client=mock_neo4j_client)
        
        patterns = tool._identify_suspicious_patterns(sample_subgraph_data)
        
        assert isinstance(patterns, list)
        assert len(patterns) > 0
    
    def test_prepare_visualization_data(self, mock_neo4j_client, mock_torch_device, sample_subgraph_data):
        """Test _prepare_visualization_data method"""
        tool = GNNFraudDetectionTool(neo4j_client=mock_neo4j_client)
        
        # Add fraud probability to nodes
        for node in sample_subgraph_data["nodes"]:
            node["fraud_probability"] = 0.5
        
        vis_data = tool._prepare_visualization_data(sample_subgraph_data)
        
        assert 'nodes' in vis_data
        assert 'edges' in vis_data
        assert len(vis_data['nodes']) == len(sample_subgraph_data['nodes'])
        assert len(vis_data['edges']) == len(sample_subgraph_data['edges'])
        
        # Check node color based on fraud probability
        for node in vis_data['nodes']:
            assert 'color' in node
