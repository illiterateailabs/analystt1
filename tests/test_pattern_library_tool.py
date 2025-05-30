"""
Tests for the PatternLibraryTool

This module contains tests for the PatternLibraryTool, which is responsible for
managing and converting fraud pattern definitions to Cypher queries.
"""

import os
import yaml
import json
import pytest
from unittest.mock import patch, mock_open, MagicMock
from pathlib import Path
from datetime import datetime, timedelta

from backend.agents.tools.pattern_library_tool import PatternLibraryTool, PatternSearchParams


# Sample pattern data for testing
SAMPLE_PATTERNS = {
    "STRUCT_001": {
        "metadata": {
            "id": "STRUCT_001",
            "name": "Basic Structuring",
            "description": "Multiple transactions just below reporting threshold",
            "category": "STRUCTURING",
            "risk_level": "HIGH",
            "regulatory_implications": ["SAR filing required", "BSA/AML violation"],
            "tags": ["money_laundering", "tax_evasion"]
        },
        "detection": {
            "graph_pattern": {
                "nodes": [
                    {
                        "id": "source",
                        "labels": ["Person"],
                        "properties": {}
                    },
                    {
                        "id": "account",
                        "labels": ["Account"],
                        "properties": {}
                    },
                    {
                        "id": "transactions",
                        "labels": ["Transaction"],
                        "properties": {
                            "amount": {"$gte": 8000, "$lt": 10000}
                        }
                    }
                ],
                "relationships": [
                    {
                        "source": "source",
                        "target": "account",
                        "type": "OWNS",
                        "direction": "OUTGOING"
                    },
                    {
                        "source": "account",
                        "target": "transactions",
                        "type": "SENT",
                        "direction": "OUTGOING"
                    }
                ]
            },
            "temporal_constraints": [
                {
                    "type": "TIME_WINDOW",
                    "node_id": "transactions",
                    "property": "timestamp",
                    "parameters": {
                        "window": "P7D"
                    }
                }
            ],
            "aggregation_rules": [
                {
                    "type": "COUNT",
                    "group_by": ["source.id"],
                    "having": {
                        "count": {"$gte": 3}
                    },
                    "window": {
                        "duration": "P7D"
                    }
                }
            ]
        },
        "cypher_template": "MATCH (source)-[:OWNS]->(account:Account)-[:SENT]->(tx:Transaction)\nWHERE tx.amount >= $min_amount AND tx.amount < $threshold\nAND tx.timestamp > datetime() - duration($time_window)\nWITH source, count(tx) as txCount, sum(tx.amount) as total\nWHERE txCount >= $min_transactions\nRETURN source, txCount, total\nORDER BY txCount DESC",
        "response_actions": [
            {
                "action": "ALERT",
                "priority": "HIGH",
                "details": "Multiple transactions below reporting threshold detected"
            }
        ]
    },
    "LAYER_001": {
        "metadata": {
            "id": "LAYER_001",
            "name": "Complex Layering",
            "description": "Funds moved through multiple accounts",
            "category": "LAYERING",
            "risk_level": "HIGH",
            "regulatory_implications": ["Money laundering red flag"],
            "tags": ["money_laundering", "cross_border"]
        },
        "detection": {
            "graph_pattern": {
                "path_patterns": [
                    {
                        "start_node": "source_account",
                        "end_node": "destination_account",
                        "relationship_types": ["SENT", "RECEIVED_BY"],
                        "min_length": 3,
                        "max_length": 10,
                        "direction": "OUTGOING"
                    }
                ]
            },
            "value_constraints": [
                {
                    "type": "RATIO",
                    "node_id": "transactions",
                    "property": "amount",
                    "parameters": {
                        "min_ratio": 0.9
                    }
                }
            ]
        },
        "cypher_template": "MATCH path = (source:Account)-[:SENT]->(:Transaction)-[:RECEIVED_BY]->\n(i1:Account)-[:SENT]->(:Transaction)-[:RECEIVED_BY]->\n(i2:Account)-[:SENT]->(:Transaction)-[:RECEIVED_BY]->\n(dest:Account)\nWHERE source <> dest\nAND source.jurisdiction <> dest.jurisdiction\nWITH path, source, dest,\n[n IN nodes(path) WHERE n:Transaction] AS txs\nWITH path, source, dest, txs,\ntxs[0].amount AS start_amount,\ntxs[size(txs)-1].amount AS end_amount,\ntxs[0].timestamp AS start_time,\ntxs[size(txs)-1].timestamp AS end_time\nWHERE end_amount >= start_amount * $min_ratio\nAND duration.between(start_time, end_time).days <= $max_days\nRETURN path, source, dest, start_amount, end_amount,\nduration.between(start_time, end_time).days AS days"
    },
    "MIXER_001": {
        "metadata": {
            "id": "MIXER_001",
            "name": "Cryptocurrency Mixing",
            "description": "Use of mixing services to obscure transaction trail",
            "category": "MIXER_USAGE",
            "risk_level": "HIGH",
            "regulatory_implications": ["Virtual asset service provider regulations"],
            "tags": ["cryptocurrency", "darknet"]
        },
        "cypher_template": "MATCH (source:Wallet)-[:TRANSFERRED]->(tx1:Transaction)-[:RECEIVED_BY]->(mixer:Wallet)\nWHERE mixer.is_known_mixer = true\nWITH source, mixer\nMATCH (mixer)-[:TRANSFERRED]->(tx2:Transaction)-[:RECEIVED_BY]->(dest:Wallet)\nWHERE tx2.timestamp > tx1.timestamp\nAND duration.between(tx1.timestamp, tx2.timestamp).days <= $max_days\nRETURN source, mixer, dest, tx1, tx2,\nduration.between(tx1.timestamp, tx2.timestamp).days AS days"
    }
}

# Sample schema data
SAMPLE_SCHEMA = {
    "schema_version": "1.0.0",
    "pattern_schema": {
        "metadata": {
            "id": {
                "type": "string",
                "required": True
            }
        }
    },
    "example_patterns": [
        SAMPLE_PATTERNS["STRUCT_001"],
        SAMPLE_PATTERNS["LAYER_001"]
    ]
}


@pytest.fixture
def mock_patterns_dir():
    """Mock the patterns directory"""
    with patch('backend.agents.tools.pattern_library_tool.PATTERNS_DIR', Path('/mock/patterns')):
        yield


@pytest.fixture
def mock_file_system(mock_patterns_dir):
    """Mock file system operations"""
    # Mock directory existence check
    with patch('pathlib.Path.mkdir') as mock_mkdir, \
         patch('pathlib.Path.exists') as mock_exists, \
         patch('pathlib.Path.glob') as mock_glob, \
         patch('builtins.open', new_callable=mock_open) as mock_file:
        
        # Setup mock behavior
        mock_exists.return_value = True
        mock_glob.side_effect = lambda pattern: [
            Path('/mock/patterns/fraud_motifs_schema.yaml'),
            Path('/mock/patterns/structuring_patterns.yaml'),
            Path('/mock/patterns/layering_patterns.yaml')
        ] if pattern in ('*.yaml', '*.yml') else []
        
        # Mock file content based on filename
        def mock_yaml_load(file_obj):
            path = file_obj.name
            if 'schema' in path:
                return SAMPLE_SCHEMA
            elif 'structuring' in path:
                return {"patterns": [SAMPLE_PATTERNS["STRUCT_001"]]}
            elif 'layering' in path:
                return {"patterns": [SAMPLE_PATTERNS["LAYER_001"]]}
            elif 'mixer' in path:
                return {"patterns": [SAMPLE_PATTERNS["MIXER_001"]]}
            return {}
        
        # Mock yaml.safe_load
        with patch('yaml.safe_load', side_effect=mock_yaml_load):
            yield


@pytest.fixture
def pattern_library_tool(mock_file_system):
    """Create a PatternLibraryTool instance with mocked file system"""
    tool = PatternLibraryTool()
    # Force cache to be populated with our sample data
    tool._patterns_cache = SAMPLE_PATTERNS.copy()
    tool._schema = SAMPLE_SCHEMA.copy()
    tool._last_load_time = datetime.now()
    return tool


def test_init(mock_file_system):
    """Test initialization of PatternLibraryTool"""
    tool = PatternLibraryTool()
    assert tool.name == "PatternLibraryTool"
    assert tool.description == "Access and convert fraud pattern definitions to Cypher queries"
    assert hasattr(tool, '_patterns_cache')
    assert hasattr(tool, '_schema')
    assert hasattr(tool, '_last_load_time')


def test_ensure_patterns_dir(mock_patterns_dir):
    """Test that patterns directory is created if it doesn't exist"""
    with patch('pathlib.Path.mkdir') as mock_mkdir, \
         patch('pathlib.Path.exists') as mock_exists, \
         patch('builtins.open', new_callable=mock_open) as mock_file:
        
        mock_exists.return_value = False
        
        tool = PatternLibraryTool()
        
        # Check that mkdir was called with parents=True and exist_ok=True
        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
        
        # Check that README.md was created
        mock_file.assert_called_once()
        assert 'README.md' in mock_file.call_args[0][0]
        assert 'w' in mock_file.call_args[0][1]


def test_load_patterns(mock_file_system):
    """Test loading patterns from files"""
    tool = PatternLibraryTool()
    
    # Clear cache and force reload
    tool._patterns_cache = {}
    tool._last_load_time = datetime.min
    tool._load_patterns(force=True)
    
    # Check that patterns were loaded
    assert len(tool._patterns_cache) >= 2
    assert "STRUCT_001" in tool._patterns_cache
    assert "LAYER_001" in tool._patterns_cache
    
    # Check that schema was loaded
    assert tool._schema is not None
    assert "schema_version" in tool._schema


def test_load_patterns_cache_recent(mock_file_system, pattern_library_tool):
    """Test that patterns are not reloaded if cache is recent"""
    # Set last load time to recent
    original_cache = pattern_library_tool._patterns_cache.copy()
    pattern_library_tool._last_load_time = datetime.now() - timedelta(minutes=1)
    
    # Mock yaml.safe_load to ensure it's not called
    with patch('yaml.safe_load') as mock_yaml_load:
        pattern_library_tool._load_patterns()
        
        # Check that yaml.safe_load was not called
        mock_yaml_load.assert_not_called()
    
    # Check that cache is unchanged
    assert pattern_library_tool._patterns_cache == original_cache


def test_load_patterns_force_reload(mock_file_system, pattern_library_tool):
    """Test forcing reload of patterns"""
    # Set last load time to recent
    pattern_library_tool._last_load_time = datetime.now() - timedelta(minutes=1)
    
    # Mock yaml.safe_load to ensure it's called
    with patch('yaml.safe_load') as mock_yaml_load:
        mock_yaml_load.side_effect = lambda f: SAMPLE_SCHEMA if 'schema' in f.name else {"patterns": [SAMPLE_PATTERNS["STRUCT_001"]]}
        
        pattern_library_tool._load_patterns(force=True)
        
        # Check that yaml.safe_load was called
        assert mock_yaml_load.call_count > 0


def test_load_patterns_handles_errors(mock_patterns_dir):
    """Test that load_patterns handles errors gracefully"""
    with patch('pathlib.Path.exists') as mock_exists, \
         patch('pathlib.Path.glob') as mock_glob, \
         patch('builtins.open', mock_open()) as mock_file, \
         patch('yaml.safe_load') as mock_yaml_load, \
         patch('logging.Logger.error') as mock_error:
        
        # Setup mocks
        mock_exists.return_value = True
        mock_glob.return_value = [Path('/mock/patterns/error_pattern.yaml')]
        mock_yaml_load.side_effect = yaml.YAMLError("Invalid YAML")
        
        tool = PatternLibraryTool()
        tool._load_patterns(force=True)
        
        # Check that error was logged
        mock_error.assert_called()
        assert "Error loading pattern file" in mock_error.call_args[0][0]
        
        # Check that tool still works
        assert tool._patterns_cache == {}


def test_get_pattern_summary(pattern_library_tool):
    """Test getting a summary of a pattern"""
    pattern = SAMPLE_PATTERNS["STRUCT_001"]
    summary = pattern_library_tool._get_pattern_summary(pattern)
    
    # Check summary fields
    assert summary["id"] == "STRUCT_001"
    assert summary["name"] == "Basic Structuring"
    assert "description" in summary
    assert summary["category"] == "STRUCTURING"
    assert summary["risk_level"] == "HIGH"
    assert "money_laundering" in summary["tags"]
    assert "SAR filing required" in summary["regulatory_implications"]


def test_search_patterns_by_id(pattern_library_tool):
    """Test searching patterns by ID"""
    params = PatternSearchParams(pattern_id="STRUCT_001")
    results = pattern_library_tool._search_patterns(params)
    
    assert len(results) == 1
    assert results[0]["metadata"]["id"] == "STRUCT_001"


def test_search_patterns_by_category(pattern_library_tool):
    """Test searching patterns by category"""
    params = PatternSearchParams(category="LAYERING")
    results = pattern_library_tool._search_patterns(params)
    
    assert len(results) == 1
    assert results[0]["metadata"]["id"] == "LAYER_001"
    assert results[0]["metadata"]["category"] == "LAYERING"


def test_search_patterns_by_risk_level(pattern_library_tool):
    """Test searching patterns by risk level"""
    params = PatternSearchParams(risk_level="HIGH")
    results = pattern_library_tool._search_patterns(params)
    
    assert len(results) == 3  # All sample patterns are HIGH risk
    assert all(p["metadata"]["risk_level"] == "HIGH" for p in results)


def test_search_patterns_by_tags(pattern_library_tool):
    """Test searching patterns by tags"""
    params = PatternSearchParams(tags=["cryptocurrency"])
    results = pattern_library_tool._search_patterns(params)
    
    assert len(results) == 1
    assert results[0]["metadata"]["id"] == "MIXER_001"
    assert "cryptocurrency" in results[0]["metadata"]["tags"]


def test_search_patterns_by_regulatory_implications(pattern_library_tool):
    """Test searching patterns by regulatory implications"""
    params = PatternSearchParams(regulatory_implications=["SAR filing required"])
    results = pattern_library_tool._search_patterns(params)
    
    assert len(results) == 1
    assert results[0]["metadata"]["id"] == "STRUCT_001"
    assert "SAR filing required" in results[0]["metadata"]["regulatory_implications"]


def test_search_patterns_multiple_criteria(pattern_library_tool):
    """Test searching patterns with multiple criteria"""
    params = PatternSearchParams(
        category="STRUCTURING",
        risk_level="HIGH",
        tags=["money_laundering"]
    )
    results = pattern_library_tool._search_patterns(params)
    
    assert len(results) == 1
    assert results[0]["metadata"]["id"] == "STRUCT_001"
    assert results[0]["metadata"]["category"] == "STRUCTURING"
    assert results[0]["metadata"]["risk_level"] == "HIGH"
    assert "money_laundering" in results[0]["metadata"]["tags"]


def test_search_patterns_no_results(pattern_library_tool):
    """Test searching patterns with no matching results"""
    params = PatternSearchParams(category="NONEXISTENT")
    results = pattern_library_tool._search_patterns(params)
    
    assert len(results) == 0


def test_convert_graph_pattern_to_cypher(pattern_library_tool):
    """Test converting a graph pattern to Cypher MATCH and WHERE clauses"""
    graph_pattern = SAMPLE_PATTERNS["STRUCT_001"]["detection"]["graph_pattern"]
    match_clause, where_clause, parameters = pattern_library_tool._convert_graph_pattern_to_cypher(graph_pattern)
    
    # Check MATCH clause
    assert match_clause.startswith("MATCH ")
    assert "(source:Person)" in match_clause
    assert "(account:Account)" in match_clause
    assert "(transactions:Transaction)" in match_clause
    assert "OWNS" in match_clause
    assert "SENT" in match_clause
    
    # Check WHERE clause
    assert where_clause.startswith("WHERE ")
    assert "transactions.amount >=" in where_clause
    assert "transactions.amount <" in where_clause
    
    # Check parameters
    assert "transactions_amount_gte" in parameters
    assert "transactions_amount_lt" in parameters
    assert parameters["transactions_amount_gte"] == 8000
    assert parameters["transactions_amount_lt"] == 10000


def test_process_temporal_constraints(pattern_library_tool):
    """Test processing temporal constraints"""
    constraints = SAMPLE_PATTERNS["STRUCT_001"]["detection"]["temporal_constraints"]
    where_parts, parameters = pattern_library_tool._process_temporal_constraints(constraints)
    
    # Check WHERE parts
    assert len(where_parts) == 1
    assert "transactions.timestamp > datetime() - duration($transactions_timestamp_window)" in where_parts[0]
    
    # Check parameters
    assert "transactions_timestamp_window" in parameters
    assert parameters["transactions_timestamp_window"] == "P7D"


def test_process_value_constraints(pattern_library_tool):
    """Test processing value constraints"""
    constraints = [
        {
            "type": "THRESHOLD",
            "node_id": "tx",
            "property": "amount",
            "parameters": {
                "min": 5000,
                "max": 10000
            }
        },
        {
            "type": "STRUCTURING",
            "node_id": "tx",
            "property": "amount",
            "parameters": {
                "threshold": 10000,
                "margin": 0.2
            }
        }
    ]
    
    where_parts, parameters = pattern_library_tool._process_value_constraints(constraints)
    
    # Check WHERE parts
    assert len(where_parts) == 3
    assert "tx.amount >= $tx_amount_min" in where_parts[0]
    assert "tx.amount <= $tx_amount_max" in where_parts[1]
    assert "tx.amount >= $tx_amount_min AND tx.amount < $tx_amount_threshold" in where_parts[2]
    
    # Check parameters
    assert "tx_amount_min" in parameters
    assert "tx_amount_max" in parameters
    assert "tx_amount_threshold" in parameters
    assert parameters["tx_amount_min"] == 5000
    assert parameters["tx_amount_max"] == 10000
    assert parameters["tx_amount_threshold"] == 10000


def test_process_aggregation_rules(pattern_library_tool):
    """Test processing aggregation rules"""
    rules = SAMPLE_PATTERNS["STRUCT_001"]["detection"]["aggregation_rules"]
    with_clause, having_clause, parameters = pattern_library_tool._process_aggregation_rules(rules)
    
    # Check WITH clause
    assert with_clause.startswith("WITH ")
    assert "source.id" in with_clause
    assert "count(*) as count" in with_clause
    
    # Check HAVING clause
    assert having_clause.startswith("HAVING ")
    assert "count >= $count_gte" in having_clause
    
    # Check parameters
    assert "count_gte" in parameters
    assert parameters["count_gte"] == 3


def test_convert_pattern_to_cypher_template(pattern_library_tool):
    """Test converting a pattern to Cypher using template method"""
    pattern = SAMPLE_PATTERNS["STRUCT_001"]
    user_params = {
        "min_amount": 8000,
        "threshold": 10000,
        "time_window": "P7D",
        "min_transactions": 3
    }
    
    cypher_query, parameters = pattern_library_tool._convert_pattern_to_cypher_template(pattern, user_params)
    
    # Check query
    assert "MATCH (source)-[:OWNS]->(account:Account)-[:SENT]->(tx:Transaction)" in cypher_query
    assert "WHERE tx.amount >= $min_amount AND tx.amount < $threshold" in cypher_query
    assert "WITH source, count(tx) as txCount, sum(tx.amount) as total" in cypher_query
    
    # Check parameters
    assert "min_amount" in parameters
    assert "threshold" in parameters
    assert "time_window" in parameters
    assert "min_transactions" in parameters
    assert parameters["min_amount"] == 8000
    assert parameters["threshold"] == 10000
    assert parameters["time_window"] == "P7D"
    assert parameters["min_transactions"] == 3


def test_convert_pattern_to_cypher_dynamic(pattern_library_tool):
    """Test converting a pattern to Cypher using dynamic method"""
    pattern = SAMPLE_PATTERNS["STRUCT_001"]
    user_params = {
        "transactions_amount_gte": 8000,
        "transactions_amount_lt": 10000,
        "transactions_timestamp_window": "P7D",
        "count_gte": 3
    }
    
    cypher_query, parameters = pattern_library_tool._convert_pattern_to_cypher_dynamic(pattern, user_params)
    
    # Check query
    assert cypher_query.startswith("MATCH ")
    assert "WHERE" in cypher_query
    assert "RETURN" in cypher_query
    
    # Check parameters
    assert "transactions_amount_gte" in parameters
    assert "transactions_amount_lt" in parameters
    assert "transactions_timestamp_window" in parameters
    assert "count_gte" in parameters


def test_convert_pattern_to_cypher_missing_template(pattern_library_tool):
    """Test error handling when template is missing"""
    # Create a pattern without cypher_template
    pattern = {
        "metadata": {
            "id": "TEST_001",
            "name": "Test Pattern"
        }
    }
    
    # Should raise ValueError
    with pytest.raises(ValueError) as excinfo:
        pattern_library_tool._convert_pattern_to_cypher_template(pattern)
    
    assert "has no cypher_template" in str(excinfo.value)


def test_convert_pattern_to_cypher_wrapper(pattern_library_tool):
    """Test the convert_pattern_to_cypher wrapper method"""
    # Test with valid pattern ID
    result = pattern_library_tool._convert_pattern_to_cypher("STRUCT_001", {"min_amount": 8000}, True)
    
    assert result["success"] is True
    assert result["pattern_id"] == "STRUCT_001"
    assert "cypher_query" in result
    assert "parameters" in result
    assert result["generation_method"] == "template"
    
    # Test with invalid pattern ID
    result = pattern_library_tool._convert_pattern_to_cypher("NONEXISTENT", {}, True)
    
    assert result["success"] is False
    assert "error" in result
    assert "Pattern not found" in result["error"]


def test_run_list_action(pattern_library_tool):
    """Test the _run method with 'list' action"""
    query = json.dumps({"action": "list"})
    result = pattern_library_tool._run(query)
    result_json = json.loads(result)
    
    assert result_json["success"] is True
    assert "count" in result_json
    assert "patterns" in result_json
    assert len(result_json["patterns"]) >= 3
    assert any(p["id"] == "STRUCT_001" for p in result_json["patterns"])


def test_run_get_action(pattern_library_tool):
    """Test the _run method with 'get' action"""
    query = json.dumps({"action": "get", "pattern_id": "STRUCT_001"})
    result = pattern_library_tool._run(query)
    result_json = json.loads(result)
    
    assert result_json["success"] is True
    assert "pattern" in result_json
    assert result_json["pattern"]["metadata"]["id"] == "STRUCT_001"
    
    # Test with invalid pattern ID
    query = json.dumps({"action": "get", "pattern_id": "NONEXISTENT"})
    result = pattern_library_tool._run(query)
    result_json = json.loads(result)
    
    assert result_json["success"] is False
    assert "error" in result_json
    assert "Pattern not found" in result_json["error"]
    
    # Test without pattern_id
    query = json.dumps({"action": "get"})
    result = pattern_library_tool._run(query)
    result_json = json.loads(result)
    
    assert result_json["success"] is False
    assert "error" in result_json
    assert "Missing pattern_id" in result_json["error"]


def test_run_search_action(pattern_library_tool):
    """Test the _run method with 'search' action"""
    query = json.dumps({
        "action": "search",
        "params": {
            "category": "STRUCTURING",
            "risk_level": "HIGH"
        }
    })
    result = pattern_library_tool._run(query)
    result_json = json.loads(result)
    
    assert result_json["success"] is True
    assert "count" in result_json
    assert "patterns" in result_json
    assert len(result_json["patterns"]) == 1
    assert result_json["patterns"][0]["id"] == "STRUCT_001"


def test_run_convert_action(pattern_library_tool):
    """Test the _run method with 'convert' action"""
    query = json.dumps({
        "action": "convert",
        "pattern_id": "STRUCT_001",
        "parameters": {
            "min_amount": 8000,
            "threshold": 10000,
            "time_window": "P7D",
            "min_transactions": 3
        },
        "use_template": True
    })
    result = pattern_library_tool._run(query)
    result_json = json.loads(result)
    
    assert result_json["success"] is True
    assert "pattern_id" in result_json
    assert "cypher_query" in result_json
    assert "parameters" in result_json
    assert result_json["pattern_id"] == "STRUCT_001"
    
    # Test without pattern_id
    query = json.dumps({"action": "convert"})
    result = pattern_library_tool._run(query)
    result_json = json.loads(result)
    
    assert result_json["success"] is False
    assert "error" in result_json
    assert "Missing pattern_id" in result_json["error"]


def test_run_unknown_action(pattern_library_tool):
    """Test the _run method with unknown action"""
    query = json.dumps({"action": "unknown"})
    result = pattern_library_tool._run(query)
    result_json = json.loads(result)
    
    assert result_json["success"] is False
    assert "error" in result_json
    assert "Unknown action" in result_json["error"]


def test_run_exception_handling(pattern_library_tool):
    """Test exception handling in _run method"""
    # Invalid JSON
    with patch('json.loads') as mock_loads:
        mock_loads.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
        
        result = pattern_library_tool._run("invalid json")
        result_json = json.loads(result)
        
        assert result_json["success"] is False
        assert "error" in result_json


def test_public_methods(pattern_library_tool):
    """Test the public methods of PatternLibraryTool"""
    # get_pattern
    pattern = pattern_library_tool.get_pattern("STRUCT_001")
    assert pattern is not None
    assert pattern["metadata"]["id"] == "STRUCT_001"
    
    # list_patterns
    patterns = pattern_library_tool.list_patterns()
    assert len(patterns) >= 3
    assert any(p["id"] == "STRUCT_001" for p in patterns)
    
    # search_patterns
    patterns = pattern_library_tool.search_patterns(category="LAYERING")
    assert len(patterns) == 1
    assert patterns[0]["id"] == "LAYER_001"
    
    # convert_pattern
    result = pattern_library_tool.convert_pattern("MIXER_001", {"max_days": 7})
    assert result["success"] is True
    assert result["pattern_id"] == "MIXER_001"
    assert "cypher_query" in result
    assert "parameters" in result
