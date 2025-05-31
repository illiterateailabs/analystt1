"""
CrewFactory for creating and managing AI crews with different agent configurations.

This module provides a factory class for creating and managing CrewAI crews
with different agent configurations, tools, and tasks. It handles caching,
configuration loading, and crew execution.
"""

import os
import json
import logging
import yaml
from typing import Dict, List, Optional, Any, Union, Tuple
from pathlib import Path
import time
from datetime import datetime

from crewai import Agent, Task, Crew, Process
from crewai.tools import Tool

from backend.agents.config import AgentConfig, TaskConfig
from backend.agents.tools.graph_query_tool import GraphQueryTool
from backend.agents.tools.neo4j_schema_tool import Neo4jSchemaTool
from backend.agents.tools.pattern_library_tool import PatternLibraryTool
from backend.agents.tools.code_gen_tool import CodeGenerationTool
from backend.agents.tools.sandbox_exec_tool import SandboxExecutionTool
from backend.agents.tools.template_engine_tool import TemplateEngineTool
from backend.agents.tools.policy_docs_tool import PolicyDocsTool
from backend.agents.tools.random_tx_generator_tool import RandomTransactionGeneratorTool
from backend.agents.tools.fraud_ml_tool import FraudMLTool
from backend.agents.tools.crypto_anomaly_tool import CryptoAnomalyTool
from backend.agents.tools.crypto_csv_loader_tool import CryptoCSVLoaderTool

from backend.core.metrics import increment_counter, observe_value
from backend.integrations.neo4j_client import Neo4jClient
from backend.integrations.e2b_client import E2BClient
from backend.integrations.gemini_client import GeminiClient


class CrewFactory:
    """
    Factory for creating and managing AI crews with different agent configurations.
    
    This class handles the creation and management of CrewAI crews, including
    loading agent configurations, initializing tools, and executing crews.
    It also manages caching of agents and crews for better performance.
    """
    
    # Default agent configuration directories
    DEFAULT_CONFIG_DIR = Path(__file__).parent / "configs"
    DEFAULT_AGENT_CONFIG_DIR = DEFAULT_CONFIG_DIR / "defaults"
    DEFAULT_CREW_CONFIG_DIR = DEFAULT_CONFIG_DIR / "crews"
    DEFAULT_PATTERN_DIR = Path(__file__).parent / "patterns"
    
    # Default agent types
    AGENT_TYPE_NLQ_TRANSLATOR = "nlq_translator"
    AGENT_TYPE_GRAPH_ANALYST = "graph_analyst"
    AGENT_TYPE_FRAUD_PATTERN_HUNTER = "fraud_pattern_hunter"
    AGENT_TYPE_COMPLIANCE_CHECKER = "compliance_checker"
    AGENT_TYPE_REPORT_WRITER = "report_writer"
    
    def __init__(
        self,
        config_dir: Optional[str] = None,
        agent_config_dir: Optional[str] = None,
        crew_config_dir: Optional[str] = None,
        pattern_dir: Optional[str] = None,
        neo4j_client: Optional[Neo4jClient] = None,
        e2b_client: Optional[E2BClient] = None,
        gemini_client: Optional[GeminiClient] = None,
        cache_agents: bool = True,
        cache_crews: bool = True,
        verbose: bool = False
    ):
        """
        Initialize the CrewFactory.
        
        Args:
            config_dir: Directory containing all configuration files
            agent_config_dir: Directory containing agent configuration files
            crew_config_dir: Directory containing crew configuration files
            pattern_dir: Directory containing pattern library files
            neo4j_client: Neo4j client for database access
            e2b_client: E2B client for sandbox execution
            gemini_client: Gemini client for LLM access
            cache_agents: Whether to cache agent instances
            cache_crews: Whether to cache crew instances
            verbose: Whether to enable verbose logging
        """
        # Set configuration directories
        self.config_dir = Path(config_dir) if config_dir else self.DEFAULT_CONFIG_DIR
        self.agent_config_dir = Path(agent_config_dir) if agent_config_dir else (
            self.config_dir / "defaults" if config_dir else self.DEFAULT_AGENT_CONFIG_DIR
        )
        self.crew_config_dir = Path(crew_config_dir) if crew_config_dir else (
            self.config_dir / "crews" if config_dir else self.DEFAULT_CREW_CONFIG_DIR
        )
        self.pattern_dir = Path(pattern_dir) if pattern_dir else self.DEFAULT_PATTERN_DIR
        
        # Set clients
        self.neo4j_client = neo4j_client or Neo4jClient()
        self.e2b_client = e2b_client or E2BClient()
        self.gemini_client = gemini_client or GeminiClient()
        
        # Set caching and logging options
        self.cache_agents = cache_agents
        self.cache_crews = cache_crews
        self.verbose = verbose
        self.logger = logging.getLogger(__name__)
        
        # Initialize caches
        self._agent_cache = {}
        self._crew_cache = {}
        self._config_cache = {}
        
        # Initialize tools
        self.graph_query_tool = Tool(
            name="graph_query",
            description="Execute Cypher queries against the Neo4j graph database",
            func=lambda query: GraphQueryTool(self.neo4j_client).run(query)
        )
        
        self.neo4j_schema_tool = Tool(
            name="neo4j_schema",
            description="Get information about the Neo4j database schema",
            func=lambda query: Neo4jSchemaTool(self.neo4j_client).run(**json.loads(query))
        )
        
        self.pattern_library_tool = Tool(
            name="pattern_library",
            description="Access and use fraud detection patterns from the pattern library",
            func=lambda query: PatternLibraryTool(self.neo4j_client, self.pattern_dir).run(**json.loads(query))
        )
        
        self.code_generation_tool = Tool(
            name="code_generation",
            description="Generate Python code for data analysis and visualization",
            func=lambda query: CodeGenerationTool(self.gemini_client).run(**json.loads(query))
        )
        
        self.sandbox_execution_tool = Tool(
            name="sandbox_execution",
            description="Execute Python code in a secure sandbox environment",
            func=lambda query: SandboxExecutionTool(self.e2b_client).run(**json.loads(query))
        )
        
        self.template_engine_tool = Tool(
            name="template_engine",
            description="Render templates using Jinja2 for report generation",
            func=lambda query: TemplateEngineTool().run(**json.loads(query))
        )
        
        self.policy_docs_tool = Tool(
            name="policy_docs",
            description="Access and search compliance policy documents",
            func=lambda query: PolicyDocsTool().run(**json.loads(query))
        )
        
        self.random_tx_generator_tool = Tool(
            name="random_tx_generator",
            description="Generate random transaction data for testing",
            func=lambda query: RandomTransactionGeneratorTool().run(**json.loads(query))
        )
        
        self.fraud_ml_tool = Tool(
            name="fraud_ml",
            description="Use machine learning models for fraud detection",
            func=lambda query: FraudMLTool(self.neo4j_client).run(**json.loads(query))
        )
        
        self.crypto_anomaly_tool = Tool(
            name="crypto_anomaly_detection",
            description="Detect various anomalies in cryptocurrency transactions including wash trading, pump-and-dump, and time-series anomalies",
            func=lambda query: CryptoAnomalyTool(self.neo4j_client).run(**json.loads(query))
        )

        self.crypto_csv_loader = Tool(
            name="crypto_csv_loader",
            description="Load cryptocurrency transaction data from CSV files into Neo4j graph database",
            func=lambda query: CryptoCSVLoaderTool(self.neo4j_client).run(**json.loads(query))
        )
    
    def get_agent_config(self, agent_type: str) -> AgentConfig:
        """
        Get the configuration for a specific agent type.
        
        Args:
            agent_type: Type of agent to get configuration for
            
        Returns:
            Agent configuration
            
        Raises:
            ValueError: If agent type is not supported or configuration file is not found
        """
        # Check cache first
        if agent_type in self._config_cache:
            return self._config_cache[agent_type]
        
        # Look for configuration file
        config_path = self.agent_config_dir / f"{agent_type}.yaml"
        if not config_path.exists():
            config_path = self.agent_config_dir / f"{agent_type}.yml"
        
        if not config_path.exists():
            raise ValueError(f"Configuration file for agent type '{agent_type}' not found")
        
        # Load configuration
        try:
            with open(config_path, "r") as f:
                config_data = yaml.safe_load(f)
            
            config = AgentConfig(**config_data)
            
            # Cache configuration
            self._config_cache[agent_type] = config
            
            return config
        except Exception as e:
            self.logger.error(f"Error loading agent configuration: {str(e)}")
            raise ValueError(f"Error loading configuration for agent type '{agent_type}': {str(e)}")
    
    def create_agent(self, agent_type: str, override_config: Optional[Dict[str, Any]] = None) -> Agent:
        """
        Create an agent of the specified type.
        
        Args:
            agent_type: Type of agent to create
            override_config: Optional configuration overrides
            
        Returns:
            CrewAI Agent instance
            
        Raises:
            ValueError: If agent type is not supported
        """
        # Check cache first if enabled
        cache_key = f"{agent_type}_{json.dumps(override_config or {})}"
        if self.cache_agents and cache_key in self._agent_cache:
            return self._agent_cache[cache_key]
        
        # Get agent configuration
        config = self.get_agent_config(agent_type)
        
        # Apply overrides if provided
        if override_config:
            # Create a copy of the configuration
            config_dict = config.dict()
            # Apply overrides
            config_dict.update(override_config)
            # Create new configuration
            config = AgentConfig(**config_dict)
        
        # Get tools for this agent
        tools = self._get_tools_for_agent(agent_type)
        
        # Create agent
        agent = Agent(
            role=config.role,
            goal=config.goal,
            backstory=config.backstory,
            verbose=self.verbose,
            llm=self.gemini_client.get_llm(model=config.llm_model),
            tools=tools,
            allow_delegation=config.allow_delegation
        )
        
        # Cache agent if enabled
        if self.cache_agents:
            self._agent_cache[cache_key] = agent
        
        return agent
    
    def _get_tools_for_agent(self, agent_type: str) -> List[Tool]:
        """
        Get the tools for a specific agent type.
        
        Args:
            agent_type: Type of agent to get tools for
            
        Returns:
            List of tools for the agent
        """
        # Common tools for all agents
        common_tools = []
        
        # Agent-specific tools
        if agent_type == self.AGENT_TYPE_NLQ_TRANSLATOR:
            return common_tools + [
                self.neo4j_schema_tool
            ]
        
        elif agent_type == self.AGENT_TYPE_GRAPH_ANALYST:
            return common_tools + [
                self.graph_query_tool,
                self.neo4j_schema_tool,
                self.code_generation_tool,
                self.sandbox_execution_tool
            ]
        
        elif agent_type == self.AGENT_TYPE_FRAUD_PATTERN_HUNTER:
            return common_tools + [
                self.graph_query_tool,
                self.neo4j_schema_tool,
                self.pattern_library_tool,
                self.code_generation_tool,
                self.sandbox_execution_tool,
                self.fraud_ml_tool,
                self.crypto_anomaly_tool,
                self.crypto_csv_loader
            ]
        
        elif agent_type == self.AGENT_TYPE_COMPLIANCE_CHECKER:
            return common_tools + [
                self.graph_query_tool,
                self.policy_docs_tool
            ]
        
        elif agent_type == self.AGENT_TYPE_REPORT_WRITER:
            return common_tools + [
                self.graph_query_tool,
                self.template_engine_tool,
                self.code_generation_tool,
                self.sandbox_execution_tool
            ]
        
        # Default: return common tools only
        return common_tools
    
    def create_crew(
        self,
        crew_name: str,
        agents: Optional[List[Agent]] = None,
        tasks: Optional[List[Task]] = None,
        process: Optional[Process] = None,
        verbose: Optional[bool] = None
    ) -> Crew:
        """
        Create a crew with the specified agents and tasks.
        
        Args:
            crew_name: Name of the crew
            agents: List of agents for the crew
            tasks: List of tasks for the crew
            process: Process for the crew
            verbose: Whether to enable verbose logging
            
        Returns:
            CrewAI Crew instance
        """
        # Check cache first if enabled
        if self.cache_crews and crew_name in self._crew_cache:
            return self._crew_cache[crew_name]
        
        # Use provided values or defaults
        agents = agents or []
        tasks = tasks or []
        process = process or Process.sequential
        verbose = verbose if verbose is not None else self.verbose
        
        # Create crew
        crew = Crew(
            agents=agents,
            tasks=tasks,
            process=process,
            verbose=verbose
        )
        
        # Cache crew if enabled
        if self.cache_crews:
            self._crew_cache[crew_name] = crew
        
        return crew
    
    def create_default_crew(self) -> Crew:
        """
        Create the default crew with all standard agents and tasks.
        
        Returns:
            CrewAI Crew instance
        """
        # Create agents
        nlq_translator = self.create_agent(self.AGENT_TYPE_NLQ_TRANSLATOR)
        graph_analyst = self.create_agent(self.AGENT_TYPE_GRAPH_ANALYST)
        fraud_pattern_hunter = self.create_agent(self.AGENT_TYPE_FRAUD_PATTERN_HUNTER)
        compliance_checker = self.create_agent(self.AGENT_TYPE_COMPLIANCE_CHECKER)
        report_writer = self.create_agent(self.AGENT_TYPE_REPORT_WRITER)
        
        # Create tasks
        tasks = [
            Task(
                description="Translate natural language query to Cypher",
                expected_output="Cypher query",
                agent=nlq_translator
            ),
            Task(
                description="Analyze graph data and identify patterns",
                expected_output="Graph analysis results",
                agent=graph_analyst
            ),
            Task(
                description="Detect fraud patterns in the data",
                expected_output="Fraud detection results",
                agent=fraud_pattern_hunter
            ),
            Task(
                description="Check compliance with regulations",
                expected_output="Compliance report",
                agent=compliance_checker
            ),
            Task(
                description="Generate final report",
                expected_output="Final report",
                agent=report_writer
            )
        ]
        
        # Create crew
        return self.create_crew(
            crew_name="default_crew",
            agents=[nlq_translator, graph_analyst, fraud_pattern_hunter, compliance_checker, report_writer],
            tasks=tasks,
            process=Process.sequential
        )
    
    def run_crew(
        self,
        crew: Crew,
        inputs: Optional[Dict[str, Any]] = None,
        callbacks: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Run a crew with the specified inputs and callbacks.
        
        Args:
            crew: CrewAI Crew instance
            inputs: Inputs for the crew
            callbacks: Callbacks for the crew
            
        Returns:
            Results from the crew execution
        """
        # Initialize metrics
        start_time = time.time()
        token_count_before = self.gemini_client.get_token_count()
        cost_before = self.gemini_client.get_cost()
        
        # Run crew
        try:
            result = crew.kickoff(inputs=inputs or {})
            
            # Calculate metrics
            duration = time.time() - start_time
            token_count_after = self.gemini_client.get_token_count()
            cost_after = self.gemini_client.get_cost()
            
            tokens_used = token_count_after - token_count_before
            cost_incurred = cost_after - cost_before
            
            # Record metrics
            observe_value("crew_task_duration_seconds", duration)
            increment_counter("llm_tokens_used_total", tokens_used)
            increment_counter("llm_cost_usd_total", cost_incurred)
            
            self.logger.info(f"Crew execution completed in {duration:.2f} seconds")
            self.logger.info(f"Tokens used: {tokens_used}, Cost: ${cost_incurred:.4f}")
            
            return {
                "result": result,
                "metrics": {
                    "duration_seconds": duration,
                    "tokens_used": tokens_used,
                    "cost_usd": cost_incurred
                }
            }
        
        except Exception as e:
            self.logger.error(f"Error running crew: {str(e)}")
            
            # Calculate metrics even on error
            duration = time.time() - start_time
            token_count_after = self.gemini_client.get_token_count()
            cost_after = self.gemini_client.get_cost()
            
            tokens_used = token_count_after - token_count_before
            cost_incurred = cost_after - cost_before
            
            # Record metrics
            observe_value("crew_task_duration_seconds", duration)
            increment_counter("llm_tokens_used_total", tokens_used)
            increment_counter("llm_cost_usd_total", cost_incurred)
            
            return {
                "error": str(e),
                "metrics": {
                    "duration_seconds": duration,
                    "tokens_used": tokens_used,
                    "cost_usd": cost_incurred
                }
            }
    
    def clear_cache(self, clear_agents: bool = True, clear_crews: bool = True, clear_configs: bool = True) -> None:
        """
        Clear the agent and crew caches.
        
        Args:
            clear_agents: Whether to clear the agent cache
            clear_crews: Whether to clear the crew cache
            clear_configs: Whether to clear the config cache
        """
        if clear_agents:
            self._agent_cache.clear()
            self.logger.info("Agent cache cleared")
        
        if clear_crews:
            self._crew_cache.clear()
            self.logger.info("Crew cache cleared")
        
        if clear_configs:
            self._config_cache.clear()
            self.logger.info("Config cache cleared")
