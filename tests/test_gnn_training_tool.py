"""
Tests for GNN Training Tool

This module contains comprehensive tests for the GNN Training Tool, including:
- GraphDataProcessor data extraction and PyG data creation
- HyperparameterTuner with Optuna integration
- ExperimentTracker functionality
- Training, tuning, and evaluation modes
- Experiment listing and management
- Edge cases and error handling
"""

import os
import json
import pickle
import tempfile
import shutil
from unittest.mock import MagicMock, patch, Mock, ANY
import pytest
import numpy as np
import torch
import optuna
from torch_geometric.data import Data
from datetime import datetime

from backend.agents.tools.gnn_training_tool import (
    GNNModel, 
    GraphDataProcessor, 
    HyperparameterTuner,
    ExperimentTracker,
    GNNTrainingTool, 
    GNNArchitecture,
    TrainingStrategy
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
def sample_graph_data():
    """Create sample graph data for testing"""
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
        "node_id_map": {"n1": 0, "n2": 1, "t1": 2},
        "labels": np.array([0, 1, 0])  # Binary labels for nodes
    }


@pytest.fixture
def mock_pyg_data():
    """Create a mock PyTorch Geometric Data object"""
    data = MagicMock(spec=Data)
    data.x = torch.tensor([[0.2, 365], [0.7, 30], [1000, 1672531200]], dtype=torch.float)
    data.edge_index = torch.tensor([[0, 2], [2, 1]], dtype=torch.long)
    data.y = torch.tensor([0, 1, 0], dtype=torch.float)
    data.num_nodes = 3
    data.num_edges = 2
    data.num_features = 2
    data.train_mask = torch.tensor([True, False, True], dtype=torch.bool)
    data.val_mask = torch.tensor([False, True, False], dtype=torch.bool)
    data.test_mask = torch.tensor([False, False, True], dtype=torch.bool)
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
                    "properties": {"name": "Alice", "risk_score": 0.2, "account_age_days": 365, "is_fraud": 0}
                },
                {
                    "id": "n2",
                    "labels": ["Person"],
                    "properties": {"name": "Bob", "risk_score": 0.7, "account_age_days": 30, "is_fraud": 1}
                },
                {
                    "id": "t1",
                    "labels": ["Transaction"],
                    "properties": {"amount": 1000, "timestamp": "2023-01-01", "is_fraud": 0}
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
def temp_experiment_dir():
    """Create a temporary directory for experiment tracking"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


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
    
    def test_extract_data(self, mock_neo4j_client, mock_neo4j_response):
        """Test extracting data from Neo4j"""
        # Setup mock
        mock_neo4j_client.execute_query.return_value = mock_neo4j_response
        
        processor = GraphDataProcessor(mock_neo4j_client)
        
        # Call method
        result = processor.extract_data(
            query="MATCH (n) RETURN n",
            label_field="is_fraud",
            feature_fields=["risk_score", "account_age_days"]
        )
        
        # Verify
        mock_neo4j_client.execute_query.assert_called_once()
        assert "nodes" in result
        assert "edges" in result
        assert "edge_index" in result
        assert "features" in result
        assert "feature_names" in result
        assert "labels" in result
        assert "node_id_map" in result
        assert len(result["nodes"]) > 0
        assert len(result["edges"]) > 0
    
    def test_extract_data_empty_result(self, mock_neo4j_client):
        """Test extracting data with empty result"""
        # Setup mock
        mock_neo4j_client.execute_query.return_value = []
        
        processor = GraphDataProcessor(mock_neo4j_client)
        
        # Call method
        result = processor.extract_data(query="MATCH (n) RETURN n")
        
        # Verify
        assert result["nodes"] == []
        assert result["edges"] == []
        assert result["features"] == {}
        assert result["labels"] == []
    
    def test_extract_data_database_error(self, mock_neo4j_client):
        """Test extracting data with database error"""
        # Setup mock
        mock_neo4j_client.execute_query.side_effect = Exception("Database error")
        
        processor = GraphDataProcessor(mock_neo4j_client)
        
        # Call method
        with pytest.raises(Exception, match="Database error"):
            processor.extract_data(query="MATCH (n) RETURN n")
    
    def test_create_pyg_data(self, sample_graph_data, mock_neo4j_client):
        """Test creating PyTorch Geometric Data object"""
        processor = GraphDataProcessor(mock_neo4j_client)
        
        # Call method
        data = processor.create_pyg_data(sample_graph_data)
        
        # Verify
        assert isinstance(data, Data)
        assert data.x.shape == (3, 2)  # 3 nodes, 2 features
        assert data.edge_index.shape == (2, 2)  # 2 edges
        assert hasattr(data, "y")  # Labels
        assert hasattr(data, "train_mask")  # Train mask
        assert hasattr(data, "val_mask")  # Validation mask
        assert hasattr(data, "test_mask")  # Test mask
        assert hasattr(data, "node_id_map")
        assert hasattr(data, "feature_names")
    
    def test_create_dataset(self, sample_graph_data, mock_neo4j_client):
        """Test creating a dataset from graph data"""
        processor = GraphDataProcessor(mock_neo4j_client)
        
        # Create a list of graph data
        data_list = [sample_graph_data, sample_graph_data]
        
        # Mock create_pyg_data
        with patch.object(processor, 'create_pyg_data') as mock_create:
            mock_create.return_value = MagicMock(spec=Data)
            
            # Call method
            dataset = processor.create_dataset(data_list)
            
            # Verify
            assert len(dataset) == 2
            assert mock_create.call_count == 2


# Test HyperparameterTuner
class TestHyperparameterTuner:
    """Tests for the HyperparameterTuner class"""
    
    def test_init(self, mock_pyg_data, mock_torch_device):
        """Test initialization of HyperparameterTuner"""
        tuner = HyperparameterTuner(
            data=mock_pyg_data,
            architecture=GNNArchitecture.GCN,
            strategy=TrainingStrategy.SUPERVISED,
            n_trials=5,
            timeout=60,
            metric="auc"
        )
        
        assert tuner.data == mock_pyg_data
        assert tuner.architecture == GNNArchitecture.GCN
        assert tuner.strategy == TrainingStrategy.SUPERVISED
        assert tuner.n_trials == 5
        assert tuner.timeout == 60
        assert tuner.metric == "auc"
        assert tuner.device == "cpu"
    
    def test_objective(self, mock_pyg_data, mock_torch_device):
        """Test objective function for Optuna"""
        tuner = HyperparameterTuner(
            data=mock_pyg_data,
            architecture=GNNArchitecture.GCN,
            strategy=TrainingStrategy.SUPERVISED,
            n_trials=5,
            metric="auc"
        )
        
        # Create mock trial
        trial = MagicMock()
        trial.suggest_int.side_effect = lambda name, low, high, **kwargs: {
            "hidden_channels": 32,
            "num_layers": 2
        }[name]
        trial.suggest_float.side_effect = lambda name, low, high, **kwargs: {
            "dropout": 0.2,
            "learning_rate": 0.01
        }[name]
        
        # Mock model and optimizer
        with patch('backend.agents.tools.gnn_training_tool.GNNModel') as mock_model_class, \
             patch('torch.optim.Adam') as mock_adam, \
             patch('sklearn.metrics.roc_auc_score', return_value=0.8):
            
            # Setup mock model
            mock_model = MagicMock()
            mock_model_class.return_value = mock_model
            
            # Call objective function
            result = tuner.objective(trial)
            
            # Verify
            assert isinstance(result, float)
            mock_model_class.assert_called_once()
            mock_adam.assert_called_once()
    
    def test_tune(self, mock_pyg_data, mock_torch_device):
        """Test tune method"""
        tuner = HyperparameterTuner(
            data=mock_pyg_data,
            architecture=GNNArchitecture.GCN,
            strategy=TrainingStrategy.SUPERVISED,
            n_trials=5,
            metric="auc"
        )
        
        # Mock Optuna study
        mock_study = MagicMock()
        mock_study.best_params = {"hidden_channels": 32, "num_layers": 2, "dropout": 0.2, "learning_rate": 0.01}
        mock_study.best_value = 0.8
        mock_study.trials_dataframe.return_value = "DataFrame"
        
        # Mock create_study and optimize
        with patch('optuna.create_study', return_value=mock_study):
            # Call tune method
            result = tuner.tune()
            
            # Verify
            assert result["best_params"]["architecture"] == GNNArchitecture.GCN
            assert result["best_params"]["strategy"] == TrainingStrategy.SUPERVISED
            assert result["best_value"] == 0.8
            assert result["metric"] == "auc"
            assert result["n_trials"] == 5
            assert "study_summary" in result


# Test ExperimentTracker
class TestExperimentTracker:
    """Tests for the ExperimentTracker class"""
    
    def test_init(self, temp_experiment_dir):
        """Test initialization of ExperimentTracker"""
        tracker = ExperimentTracker(base_dir=temp_experiment_dir)
        assert tracker.base_dir == temp_experiment_dir
        assert os.path.exists(temp_experiment_dir)
    
    def test_create_experiment(self, temp_experiment_dir):
        """Test creating a new experiment"""
        tracker = ExperimentTracker(base_dir=temp_experiment_dir)
        
        # Call method
        experiment_id = tracker.create_experiment(name="test_experiment")
        
        # Verify
        assert "test_experiment" in experiment_id
        assert os.path.exists(os.path.join(temp_experiment_dir, experiment_id))
    
    def test_log_config(self, temp_experiment_dir):
        """Test logging experiment configuration"""
        tracker = ExperimentTracker(base_dir=temp_experiment_dir)
        experiment_id = tracker.create_experiment()
        
        # Call method
        config = {"architecture": "gcn", "hidden_channels": 32}
        tracker.log_config(experiment_id, config)
        
        # Verify
        config_path = os.path.join(temp_experiment_dir, experiment_id, "config.json")
        assert os.path.exists(config_path)
        
        # Check content
        with open(config_path, "r") as f:
            loaded_config = json.load(f)
            assert loaded_config == config
    
    def test_log_metrics(self, temp_experiment_dir):
        """Test logging experiment metrics"""
        tracker = ExperimentTracker(base_dir=temp_experiment_dir)
        experiment_id = tracker.create_experiment()
        
        # Call method
        metrics = {"accuracy": 0.8, "loss": 0.2}
        tracker.log_metrics(experiment_id, metrics)
        
        # Call again to test appending
        metrics2 = {"accuracy": 0.85, "loss": 0.15}
        tracker.log_metrics(experiment_id, metrics2, step=1)
        
        # Verify
        metrics_path = os.path.join(temp_experiment_dir, experiment_id, "metrics.json")
        assert os.path.exists(metrics_path)
        
        # Check content
        with open(metrics_path, "r") as f:
            loaded_metrics = json.load(f)
            assert len(loaded_metrics) == 2
            assert loaded_metrics[0]["accuracy"] == 0.8
            assert loaded_metrics[1]["accuracy"] == 0.85
            assert loaded_metrics[1]["step"] == 1
    
    def test_save_model(self, temp_experiment_dir, mock_torch_device):
        """Test saving model for an experiment"""
        tracker = ExperimentTracker(base_dir=temp_experiment_dir)
        experiment_id = tracker.create_experiment()
        
        # Create a mock model
        model = MagicMock()
        model_info = {"architecture": "gcn", "in_channels": 2}
        
        # Call method with mock torch.save
        with patch('torch.save') as mock_save:
            tracker.save_model(experiment_id, model, model_info)
            
            # Verify
            mock_save.assert_called_once()
            args = mock_save.call_args[0]
            assert 'model_state_dict' in args[0]
            assert 'model_info' in args[0]
            assert args[0]['model_info'] == model_info
    
    def test_load_model(self, temp_experiment_dir, mock_torch_device):
        """Test loading model from an experiment"""
        tracker = ExperimentTracker(base_dir=temp_experiment_dir)
        experiment_id = tracker.create_experiment()
        
        # Create experiment directory and model file
        experiment_dir = os.path.join(temp_experiment_dir, experiment_id)
        model_path = os.path.join(experiment_dir, "model.pt")
        
        # Mock torch.load
        mock_checkpoint = {
            'model_state_dict': {},
            'model_info': {
                'architecture': GNNArchitecture.GCN,
                'in_channels': 2,
                'hidden_channels': 16,
                'out_channels': 1,
                'num_layers': 2,
                'dropout': 0.1
            }
        }
        
        with patch('os.path.exists', return_value=True), \
             patch('torch.load', return_value=mock_checkpoint), \
             patch('backend.agents.tools.gnn_training_tool.GNNModel') as mock_model_class:
            
            # Setup mock model
            mock_model = MagicMock()
            mock_model_class.return_value = mock_model
            
            # Call method
            model, model_info = tracker.load_model(experiment_id)
            
            # Verify
            assert model == mock_model
            assert model_info == mock_checkpoint['model_info']
    
    def test_load_model_not_found(self, temp_experiment_dir):
        """Test loading model that doesn't exist"""
        tracker = ExperimentTracker(base_dir=temp_experiment_dir)
        experiment_id = tracker.create_experiment()
        
        # Mock os.path.exists
        with patch('os.path.exists', return_value=False):
            with pytest.raises(FileNotFoundError):
                tracker.load_model(experiment_id)
    
    def test_get_experiments(self, temp_experiment_dir):
        """Test getting list of experiments"""
        tracker = ExperimentTracker(base_dir=temp_experiment_dir)
        
        # Create experiments
        exp1 = tracker.create_experiment(name="exp1")
        exp2 = tracker.create_experiment(name="exp2")
        
        # Add config and metrics
        tracker.log_config(exp1, {"architecture": "gcn"})
        tracker.log_metrics(exp1, {"accuracy": 0.8})
        
        # Call method
        experiments = tracker.get_experiments()
        
        # Verify
        assert len(experiments) == 2
        assert experiments[0]["id"] in [exp1, exp2]
        assert experiments[1]["id"] in [exp1, exp2]
        
        # Check experiment with config and metrics
        for exp in experiments:
            if exp["id"] == exp1:
                assert exp["config"] is not None
                assert exp["latest_metrics"] is not None


# Test GNNTrainingTool
class TestGNNTrainingTool:
    """Tests for the GNNTrainingTool class"""
    
    def test_init_default(self, mock_torch_device):
        """Test initialization with default parameters"""
        with patch('backend.agents.tools.gnn_training_tool.Neo4jClient') as mock_neo4j, \
             patch('backend.agents.tools.gnn_training_tool.ExperimentTracker') as mock_tracker:
            
            tool = GNNTrainingTool()
            
            assert tool.neo4j_client is not None
            assert tool.data_processor is not None
            assert tool.experiment_tracker is not None
            assert tool.device == 'cpu'
    
    def test_init_with_params(self, mock_neo4j_client, mock_torch_device):
        """Test initialization with custom parameters"""
        mock_tracker = MagicMock(spec=ExperimentTracker)
        
        tool = GNNTrainingTool(
            neo4j_client=mock_neo4j_client,
            experiment_tracker=mock_tracker,
            device='cpu'
        )
        
        assert tool.neo4j_client == mock_neo4j_client
        assert tool.experiment_tracker == mock_tracker
        assert tool.device == 'cpu'
    
    def test_run_invalid_mode(self, mock_neo4j_client, mock_torch_device):
        """Test run method with invalid mode"""
        mock_tracker = MagicMock(spec=ExperimentTracker)
        tool = GNNTrainingTool(neo4j_client=mock_neo4j_client, experiment_tracker=mock_tracker)
        
        result = tool.run(mode='invalid')
        
        assert result['success'] is False
        assert 'error' in result
        assert 'Unsupported mode' in result['error']
    
    def test_run_exception(self, mock_neo4j_client, mock_torch_device):
        """Test run method with exception"""
        mock_tracker = MagicMock(spec=ExperimentTracker)
        tool = GNNTrainingTool(neo4j_client=mock_neo4j_client, experiment_tracker=mock_tracker)
        
        # Mock _extract_data_mode to raise exception
        with patch.object(tool, '_extract_data_mode', side_effect=Exception("Test error")):
            result = tool.run(mode='extract_data')
        
        assert result['success'] is False
        assert result['error'] == "Test error"
        assert result['mode'] == 'extract_data'
    
    def test_extract_data_mode(self, mock_neo4j_client, sample_graph_data, mock_torch_device):
        """Test _extract_data_mode method"""
        mock_tracker = MagicMock(spec=ExperimentTracker)
        tool = GNNTrainingTool(neo4j_client=mock_neo4j_client, experiment_tracker=mock_tracker)
        
        # Mock data_processor.extract_data
        with patch.object(tool.data_processor, 'extract_data', return_value=sample_graph_data), \
             patch('pickle.dump'):
            
            result = tool._extract_data_mode(
                query="MATCH (n) RETURN n",
                label_field="is_fraud",
                feature_fields=["risk_score", "account_age_days"],
                save_path="data/test.pkl"
            )
            
            assert result['success'] is True
            assert result['mode'] == 'extract_data'
            assert 'data_summary' in result
            assert 'feature_names' in result
            assert result['save_path'] == "data/test.pkl"
    
    def test_extract_data_mode_no_query(self, mock_neo4j_client, mock_torch_device):
        """Test _extract_data_mode method with no query"""
        mock_tracker = MagicMock(spec=ExperimentTracker)
        tool = GNNTrainingTool(neo4j_client=mock_neo4j_client, experiment_tracker=mock_tracker)
        
        with pytest.raises(ValueError, match="Query is required"):
            tool._extract_data_mode()
    
    def test_train_mode_with_data_path(self, mock_neo4j_client, sample_graph_data, mock_pyg_data, mock_torch_device):
        """Test _train_mode method with data path"""
        mock_tracker = MagicMock(spec=ExperimentTracker)
        tool = GNNTrainingTool(neo4j_client=mock_neo4j_client, experiment_tracker=mock_tracker)
        
        # Mock experiment tracker methods
        mock_tracker.create_experiment.return_value = "test_experiment"
        
        # Mock data loading and processing
        with patch('os.path.exists', return_value=True), \
             patch('pickle.load', return_value=sample_graph_data), \
             patch.object(tool.data_processor, 'create_pyg_data', return_value=mock_pyg_data), \
             patch('backend.agents.tools.gnn_training_tool.GNNModel') as mock_model_class, \
             patch('torch.optim.Adam'), \
             patch('torch.sigmoid', return_value=torch.tensor([[0.2], [0.8], [0.4]])), \
             patch('sklearn.metrics.roc_auc_score', return_value=0.8), \
             patch('sklearn.metrics.f1_score', return_value=0.7), \
             patch('sklearn.metrics.confusion_matrix', return_value=np.array([[1, 1], [0, 1]])), \
             patch('sklearn.metrics.precision_recall_curve', return_value=([0.7, 0.8], [0.6, 0.5], [0.3, 0.4])):
            
            # Setup mock model
            mock_model = MagicMock()
            mock_model_class.return_value = mock_model
            
            result = tool._train_mode(
                data_path="data/test.pkl",
                architecture=GNNArchitecture.GCN,
                strategy=TrainingStrategy.SUPERVISED,
                experiment_name="test_experiment",
                hidden_channels=32,
                num_layers=2,
                dropout=0.2,
                learning_rate=0.01,
                epochs=5,
                patience=2
            )
            
            assert result['success'] is True
            assert result['mode'] == 'train'
            assert result['experiment_id'] == "test_experiment"
            assert 'metrics' in result
            assert 'training_history' in result
            assert 'precision_recall' in result
            assert 'model_info' in result
    
    def test_train_mode_with_query(self, mock_neo4j_client, sample_graph_data, mock_pyg_data, mock_torch_device):
        """Test _train_mode method with query"""
        mock_tracker = MagicMock(spec=ExperimentTracker)
        tool = GNNTrainingTool(neo4j_client=mock_neo4j_client, experiment_tracker=mock_tracker)
        
        # Mock experiment tracker methods
        mock_tracker.create_experiment.return_value = "test_experiment"
        
        # Mock data extraction and processing
        with patch.object(tool.data_processor, 'extract_data', return_value=sample_graph_data), \
             patch.object(tool.data_processor, 'create_pyg_data', return_value=mock_pyg_data), \
             patch('backend.agents.tools.gnn_training_tool.GNNModel') as mock_model_class, \
             patch('torch.optim.Adam'), \
             patch('torch.sigmoid', return_value=torch.tensor([[0.2], [0.8], [0.4]])), \
             patch('sklearn.metrics.roc_auc_score', return_value=0.8), \
             patch('sklearn.metrics.f1_score', return_value=0.7), \
             patch('sklearn.metrics.confusion_matrix', return_value=np.array([[1, 1], [0, 1]])), \
             patch('sklearn.metrics.precision_recall_curve', return_value=([0.7, 0.8], [0.6, 0.5], [0.3, 0.4])):
            
            # Setup mock model
            mock_model = MagicMock()
            mock_model_class.return_value = mock_model
            
            result = tool._train_mode(
                query="MATCH (n) RETURN n",
                architecture=GNNArchitecture.GCN,
                strategy=TrainingStrategy.SUPERVISED,
                experiment_name="test_experiment",
                hidden_channels=32,
                num_layers=2,
                dropout=0.2,
                learning_rate=0.01,
                epochs=5,
                patience=2
            )
            
            assert result['success'] is True
            assert result['mode'] == 'train'
            assert result['experiment_id'] == "test_experiment"
            assert 'metrics' in result
            assert 'training_history' in result
            assert 'precision_recall' in result
            assert 'model_info' in result
    
    def test_train_mode_no_data(self, mock_neo4j_client, mock_torch_device):
        """Test _train_mode method with no data"""
        mock_tracker = MagicMock(spec=ExperimentTracker)
        tool = GNNTrainingTool(neo4j_client=mock_neo4j_client, experiment_tracker=mock_tracker)
        
        with pytest.raises(ValueError, match="Either data_path or query must be provided"):
            tool._train_mode()
    
    def test_train_mode_semi_supervised(self, mock_neo4j_client, sample_graph_data, mock_pyg_data, mock_torch_device):
        """Test _train_mode method with semi-supervised strategy"""
        mock_tracker = MagicMock(spec=ExperimentTracker)
        tool = GNNTrainingTool(neo4j_client=mock_neo4j_client, experiment_tracker=mock_tracker)
        
        # Mock experiment tracker methods
        mock_tracker.create_experiment.return_value = "test_experiment"
        
        # Mock data loading and processing
        with patch('os.path.exists', return_value=True), \
             patch('pickle.load', return_value=sample_graph_data), \
             patch.object(tool.data_processor, 'create_pyg_data', return_value=mock_pyg_data), \
             patch('backend.agents.tools.gnn_training_tool.GNNModel') as mock_model_class, \
             patch('torch.optim.Adam'), \
             patch('torch.sigmoid', return_value=torch.tensor([[0.2], [0.8], [0.4]])), \
             patch('sklearn.metrics.roc_auc_score', return_value=0.8), \
             patch('sklearn.metrics.f1_score', return_value=0.7), \
             patch('sklearn.metrics.confusion_matrix', return_value=np.array([[1, 1], [0, 1]])), \
             patch('sklearn.metrics.precision_recall_curve', return_value=([0.7, 0.8], [0.6, 0.5], [0.3, 0.4])):
            
            # Setup mock model
            mock_model = MagicMock()
            mock_model_class.return_value = mock_model
            
            result = tool._train_mode(
                data_path="data/test.pkl",
                architecture=GNNArchitecture.GCN,
                strategy=TrainingStrategy.SEMI_SUPERVISED,
                experiment_name="test_experiment",
                hidden_channels=32,
                num_layers=2,
                dropout=0.2,
                learning_rate=0.01,
                epochs=5,
                patience=2
            )
            
            assert result['success'] is True
            assert result['mode'] == 'train'
            assert result['experiment_id'] == "test_experiment"
            assert 'metrics' in result
            assert 'training_history' in result
            assert 'precision_recall' in result
            assert 'model_info' in result
    
    def test_tune_mode(self, mock_neo4j_client, sample_graph_data, mock_pyg_data, mock_torch_device):
        """Test _tune_mode method"""
        mock_tracker = MagicMock(spec=ExperimentTracker)
        tool = GNNTrainingTool(neo4j_client=mock_neo4j_client, experiment_tracker=mock_tracker)
        
        # Mock experiment tracker methods
        mock_tracker.create_experiment.return_value = "test_experiment"
        
        # Mock data loading and processing
        with patch('os.path.exists', return_value=True), \
             patch('pickle.load', return_value=sample_graph_data), \
             patch.object(tool.data_processor, 'create_pyg_data', return_value=mock_pyg_data), \
             patch('backend.agents.tools.gnn_training_tool.HyperparameterTuner') as mock_tuner_class:
            
            # Setup mock tuner
            mock_tuner = MagicMock()
            mock_tuner.tune.return_value = {
                "best_params": {
                    "hidden_channels": 32,
                    "num_layers": 2,
                    "dropout": 0.2,
                    "learning_rate": 0.01
                },
                "best_value": 0.8,
                "metric": "auc",
                "n_trials": 5,
                "study_summary": "Summary"
            }
            mock_tuner_class.return_value = mock_tuner
            
            result = tool._tune_mode(
                data_path="data/test.pkl",
                architecture=GNNArchitecture.GCN,
                strategy=TrainingStrategy.SUPERVISED,
                n_trials=5,
                timeout=60,
                metric="auc",
                experiment_name="test_experiment"
            )
            
            assert result['success'] is True
            assert result['mode'] == 'tune'
            assert result['experiment_id'] == "test_experiment"
            assert 'best_params' in result
            assert 'best_value' in result
            assert 'metric' in result
            assert 'elapsed_time' in result
            assert 'next_steps' in result
    
    def test_tune_mode_no_data(self, mock_neo4j_client, mock_torch_device):
        """Test _tune_mode method with no data"""
        mock_tracker = MagicMock(spec=ExperimentTracker)
        tool = GNNTrainingTool(neo4j_client=mock_neo4j_client, experiment_tracker=mock_tracker)
        
        with pytest.raises(ValueError, match="Either data_path or query must be provided"):
            tool._tune_mode()
    
    def test_evaluate_mode(self, mock_neo4j_client, sample_graph_data, mock_pyg_data, mock_torch_device):
        """Test _evaluate_mode method"""
        mock_tracker = MagicMock(spec=ExperimentTracker)
        tool = GNNTrainingTool(neo4j_client=mock_neo4j_client, experiment_tracker=mock_tracker)
        
        # Mock experiment tracker methods
        mock_model = MagicMock()
        mock_model_info = {
            "architecture": GNNArchitecture.GCN,
            "in_channels": 2,
            "hidden_channels": 32,
            "out_channels": 1,
            "num_layers": 2,
            "dropout": 0.2
        }
        mock_tracker.load_model.return_value = (mock_model, mock_model_info)
        
        # Mock data loading and processing
        with patch('os.path.exists', return_value=True), \
             patch('pickle.load', return_value=sample_graph_data), \
             patch.object(tool.data_processor, 'create_pyg_data', return_value=mock_pyg_data), \
             patch('torch.sigmoid', return_value=torch.tensor([[0.2], [0.8], [0.4]])), \
             patch('sklearn.metrics.roc_auc_score', return_value=0.8), \
             patch('sklearn.metrics.f1_score', return_value=0.7), \
             patch('sklearn.metrics.confusion_matrix', return_value=np.array([[1, 1], [0, 1]])), \
             patch('sklearn.metrics.classification_report', return_value={"accuracy": 0.8}), \
             patch('sklearn.metrics.precision_recall_curve', return_value=([0.7, 0.8], [0.6, 0.5], [0.3, 0.4])):
            
            result = tool._evaluate_mode(
                experiment_id="test_experiment",
                data_path="data/test.pkl"
            )
            
            assert result['success'] is True
            assert result['mode'] == 'evaluate'
            assert result['experiment_id'] == "test_experiment"
            assert 'metrics' in result
            assert 'precision_recall_curve' in result
            assert 'model_info' in result
    
    def test_evaluate_mode_no_experiment_id(self, mock_neo4j_client, mock_torch_device):
        """Test _evaluate_mode method with no experiment ID"""
        mock_tracker = MagicMock(spec=ExperimentTracker)
        tool = GNNTrainingTool(neo4j_client=mock_neo4j_client, experiment_tracker=mock_tracker)
        
        with pytest.raises(ValueError, match="experiment_id is required"):
            tool._evaluate_mode(data_path="data/test.pkl")
    
    def test_evaluate_mode_no_model(self, mock_neo4j_client, mock_torch_device):
        """Test _evaluate_mode method with no model"""
        mock_tracker = MagicMock(spec=ExperimentTracker)
        tool = GNNTrainingTool(neo4j_client=mock_neo4j_client, experiment_tracker=mock_tracker)
        
        # Mock experiment tracker methods
        mock_tracker.load_model.side_effect = FileNotFoundError("No model found")
        
        with pytest.raises(ValueError, match="No model found"):
            tool._evaluate_mode(experiment_id="test_experiment", data_path="data/test.pkl")
    
    def test_evaluate_mode_no_data(self, mock_neo4j_client, mock_torch_device):
        """Test _evaluate_mode method with no data"""
        mock_tracker = MagicMock(spec=ExperimentTracker)
        tool = GNNTrainingTool(neo4j_client=mock_neo4j_client, experiment_tracker=mock_tracker)
        
        # Mock experiment tracker methods
        mock_model = MagicMock()
        mock_model_info = {
            "architecture": GNNArchitecture.GCN,
            "in_channels": 2,
            "hidden_channels": 32,
            "out_channels": 1,
            "num_layers": 2,
            "dropout": 0.2
        }
        mock_tracker.load_model.return_value = (mock_model, mock_model_info)
        
        with pytest.raises(ValueError, match="Either data_path or query must be provided"):
            tool._evaluate_mode(experiment_id="test_experiment")
    
    def test_list_experiments_mode(self, mock_neo4j_client, mock_torch_device):
        """Test _list_experiments_mode method"""
        mock_tracker = MagicMock(spec=ExperimentTracker)
        tool = GNNTrainingTool(neo4j_client=mock_neo4j_client, experiment_tracker=mock_tracker)
        
        # Mock experiment tracker methods
        mock_experiments = [
            {"id": "exp1", "config": {}, "latest_metrics": {}},
            {"id": "exp2", "config": {}, "latest_metrics": {}}
        ]
        mock_tracker.get_experiments.return_value = mock_experiments
        
        result = tool._list_experiments_mode(limit=5)
        
        assert result['success'] is True
        assert result['mode'] == 'list_experiments'
        assert result['experiments'] == mock_experiments
        assert result['count'] == 2
        assert result['total'] == 2
