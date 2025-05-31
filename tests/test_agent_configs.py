"""
Tests for agent configuration files.

This module contains tests for the YAML configuration files for agents,
verifying that they can be loaded correctly, contain all required fields,
and integrate properly with the CrewFactory.
"""

import os
import pytest
import yaml
from unittest.mock import AsyncMock, MagicMock, patch

from backend.agents.config import load_agent_config, AgentConfig
from backend.agents.factory import CrewFactory


@pytest.fixture
def mock_gemini_client():
    """Fixture for mocked GeminiClient."""
    with patch("backend.agents.factory.GeminiClient", autospec=True) as mock_client:
        mock_instance = mock_client.return_value
        mock_instance.generate_text = AsyncMock(return_value="Test response")
        yield mock_instance


@pytest.fixture
def mock_neo4j_client():
    """Fixture for mocked Neo4jClient."""
    with patch("backend.agents.factory.Neo4jClient", autospec=True) as mock_client:
        mock_instance = mock_client.return_value
        mock_instance.connect = AsyncMock()
        mock_instance.run_query = AsyncMock(return_value=[{"result": "test"}])
        yield mock_instance


@pytest.fixture
def mock_e2b_client():
    """Fixture for mocked E2BClient."""
    with patch("backend.agents.factory.E2BClient", autospec=True) as mock_client:
        mock_instance = mock_client.return_value
        mock_instance.create_sandbox = AsyncMock()
        mock_instance.execute_code = AsyncMock(return_value={
            "success": True,
            "stdout": "Test output",
            "stderr": "",
            "exit_code": 0
        })
        yield mock_instance


@pytest.fixture
def mock_llm_provider():
    """Fixture for mocked GeminiLLMProvider."""
    with patch("backend.agents.factory.GeminiLLMProvider", autospec=True) as mock_provider:
        mock_instance = mock_provider.return_value
        yield mock_instance


def test_agent_config_files_exist():
    """Test that the agent configuration files exist."""
    config_dir = os.path.join("backend", "agents", "configs", "defaults")
    
    # Check for required agent config files
    assert os.path.exists(os.path.join(config_dir, "graph_analyst.yaml")), "graph_analyst.yaml not found"
    assert os.path.exists(os.path.join(config_dir, "compliance_checker.yaml")), "compliance_checker.yaml not found"
    assert os.path.exists(os.path.join(config_dir, "report_writer.yaml")), "report_writer.yaml not found"
    assert os.path.exists(os.path.join(config_dir, "nlq_translator.yaml")), "nlq_translator.yaml not found"


def test_load_graph_analyst_config():
    """Test loading the graph_analyst configuration."""
    config = load_agent_config("graph_analyst")
    
    # Verify it's a valid AgentConfig
    assert isinstance(config, AgentConfig), "Config should be an AgentConfig instance"
    
    # Check required fields
    assert config.id == "graph_analyst"
    assert "Graph Data Analyst" in config.role
    assert "execute" in config.goal.lower() and "cypher" in config.goal.lower()
    assert config.system_prompt is not None and len(config.system_prompt) > 100
    
    # Check that prompt contains necessary context
    assert "Graph Data Science" in config.system_prompt
    assert "Neo4j" in config.system_prompt
    assert "PageRank" in config.system_prompt or "centrality" in config.system_prompt.lower()
    
    # Check metadata
    assert hasattr(config, "metadata")
    assert "capabilities" in config.metadata
    assert len(config.metadata["capabilities"]) >= 4  # Should have multiple capabilities


def test_load_compliance_checker_config():
    """Test loading the compliance_checker configuration."""
    config = load_agent_config("compliance_checker")
    
    # Verify it's a valid AgentConfig
    assert isinstance(config, AgentConfig), "Config should be an AgentConfig instance"
    
    # Check required fields
    assert config.id == "compliance_checker"
    assert "Compliance" in config.role or "Regulatory" in config.role
    assert "compliance" in config.goal.lower() or "regulation" in config.goal.lower()
    assert config.system_prompt is not None and len(config.system_prompt) > 100
    
    # Check that prompt contains necessary context
    assert "HITL" in config.system_prompt or "Human-in-the-Loop" in config.system_prompt
    assert "SAR" in config.system_prompt or "Suspicious Activity Report" in config.system_prompt
    assert "AML" in config.system_prompt or "Anti-Money Laundering" in config.system_prompt
    
    # Check metadata
    assert hasattr(config, "metadata")
    assert "capabilities" in config.metadata
    assert "decision_thresholds" in config.metadata
    assert "hitl_required" in config.metadata["decision_thresholds"]
    assert "sar_filing" in config.metadata["decision_thresholds"]


def test_load_report_writer_config():
    """Test loading the report_writer configuration."""
    config = load_agent_config("report_writer")
    
    # Verify it's a valid AgentConfig
    assert isinstance(config, AgentConfig), "Config should be an AgentConfig instance"
    
    # Check required fields
    assert config.id == "report_writer"
    assert "Writer" in config.role or "Report" in config.role
    assert "report" in config.goal.lower() or "narrative" in config.goal.lower()
    assert config.system_prompt is not None and len(config.system_prompt) > 100
    
    # Check that prompt contains necessary context
    assert "markdown" in config.system_prompt.lower()
    assert "executive summary" in config.system_prompt.lower()
    assert "graph" in config.system_prompt.lower() and "visualization" in config.system_prompt.lower()
    assert "JSON" in config.system_prompt
    
    # Check metadata
    assert hasattr(config, "metadata")
    assert "capabilities" in config.metadata
    assert "report_types" in config.metadata
    assert "graph_visualization" in config.metadata


def test_yaml_structure_validity():
    """Test the YAML structure of agent configuration files."""
    config_dir = os.path.join("backend", "agents", "configs", "defaults")
    
    for agent_file in ["graph_analyst.yaml", "compliance_checker.yaml", "report_writer.yaml"]:
        file_path = os.path.join(config_dir, agent_file)
        
        # Load YAML directly to check structure
        with open(file_path, "r") as f:
            yaml_content = yaml.safe_load(f)
        
        # Check required top-level keys
        assert "system_prompt" in yaml_content, f"{agent_file} missing system_prompt"
        assert "description" in yaml_content, f"{agent_file} missing description"
        assert "metadata" in yaml_content, f"{agent_file} missing metadata"
        
        # Check metadata structure
        assert "capabilities" in yaml_content["metadata"], f"{agent_file} missing capabilities in metadata"
        assert isinstance(yaml_content["metadata"]["capabilities"], list), f"{agent_file} capabilities should be a list"
        assert len(yaml_content["metadata"]["capabilities"]) > 0, f"{agent_file} should have at least one capability"
        
        # Check system_prompt
        assert isinstance(yaml_content["system_prompt"], str), f"{agent_file} system_prompt should be a string"
        assert len(yaml_content["system_prompt"]) > 100, f"{agent_file} system_prompt is too short"


@pytest.mark.asyncio
async def test_crew_factory_integration(mock_neo4j_client, mock_gemini_client, mock_e2b_client, mock_llm_provider):
    """Test integration of agent configs with CrewFactory."""
    # Create factory
    factory = CrewFactory()
    
    # Test creating each agent
    graph_analyst = factory.create_agent("graph_analyst")
    assert graph_analyst is not None
    assert graph_analyst.id == "graph_analyst"
    assert "Graph Data Analyst" in graph_analyst.role
    
    compliance_checker = factory.create_agent("compliance_checker")
    assert compliance_checker is not None
    assert compliance_checker.id == "compliance_checker"
    assert "Compliance" in compliance_checker.role or "Regulatory" in compliance_checker.role
    
    report_writer = factory.create_agent("report_writer")
    assert report_writer is not None
    assert report_writer.id == "report_writer"
    assert "Writer" in report_writer.role or "Report" in report_writer.role
    
    # Test that agents are cached
    assert "graph_analyst" in factory.agents_cache
    assert "compliance_checker" in factory.agents_cache
    assert "report_writer" in factory.agents_cache
    
    # Test that creating the same agent returns the cached instance
    graph_analyst_2 = factory.create_agent("graph_analyst")
    assert graph_analyst_2 is graph_analyst
    

@pytest.mark.asyncio
async def test_fraud_investigation_crew_with_new_agents(mock_neo4j_client, mock_gemini_client, mock_e2b_client, mock_llm_provider):
    """Test creating a fraud_investigation crew with the new agent configs."""
    # Mock load_crew_config to return a simplified crew config
    with patch("backend.agents.factory.load_crew_config") as mock_load:
        # Create a mock crew config
        mock_config = MagicMock()
        mock_config.name = "fraud_investigation"
        mock_config.description = "A test crew for fraud investigation"
        mock_config.agents = ["nlq_translator", "graph_analyst", "fraud_pattern_hunter", "compliance_checker", "report_writer"]
        mock_config.process_type = "sequential"
        mock_config.verbose = True
        mock_config.max_rpm = 10
        mock_config.memory = True
        mock_config.cache = True
        
        mock_load.return_value = mock_config
        
        # Create factory
        factory = CrewFactory()
        
        # Create the crew
        crew = await factory.create_crew("fraud_investigation")
        
        # Verify crew was created
        assert crew is not None
        
        # Verify agents were created
        assert len(factory.agents_cache) >= 5
        assert "nlq_translator" in factory.agents_cache
        assert "graph_analyst" in factory.agents_cache
        assert "fraud_pattern_hunter" in factory.agents_cache
        assert "compliance_checker" in factory.agents_cache
        assert "report_writer" in factory.agents_cache
