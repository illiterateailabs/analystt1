"""
CrewFactory for creating and managing CrewAI crews.

This module provides the CrewFactory class, which is responsible for
building and managing CrewAI crews, including agent creation, tool
assignment, task definition, and crew orchestration.
"""

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union, cast

from crewai import Agent, Task, Crew, Process
from crewai.agent import TaskOutput

from backend.agents.config import (
    load_agent_config, 
    load_crew_config,
    AgentConfig,
    CrewConfig,
    DEFAULT_CREW_CONFIGS
)
from backend.agents.llm import GeminiLLMProvider
from backend.agents.tools import (
    GraphQueryTool,
    SandboxExecTool,
    CodeGenTool,
    PatternLibraryTool,
    Neo4jSchemaTool,
    TemplateEngineTool,
    PolicyDocsTool,
    FraudMLTool
)
from backend.integrations.neo4j_client import Neo4jClient
from backend.integrations.gemini_client import GeminiClient
from backend.integrations.e2b_client import E2BClient
from backend.core.metrics import (
    crew_task_duration_seconds,
    llm_tokens_used_total,
    llm_cost_usd_total
)

logger = logging.getLogger(__name__)


def create_crypto_tools() -> Dict[str, Any]:
    """
    Create cryptocurrency-specific tools.
    
    This function is separated to handle optional dependencies gracefully.
    If crypto-related packages are not installed, it returns empty tools.
    
    Returns:
        Dictionary of crypto tools
    """
    crypto_tools = {}
    
    try:
        from backend.agents.tools.crypto.dune_analytics_tool import DuneAnalyticsTool
        crypto_tools["dune_analytics_tool"] = DuneAnalyticsTool()
    except (ImportError, ModuleNotFoundError):
        logger.warning("DuneAnalyticsTool not available - skipping initialization")
    
    try:
        from backend.agents.tools.crypto.defillama_tool import DefiLlamaTool
        crypto_tools["defillama_tool"] = DefiLlamaTool()
    except (ImportError, ModuleNotFoundError):
        logger.warning("DefiLlamaTool not available - skipping initialization")
    
    try:
        from backend.agents.tools.crypto.etherscan_tool import EtherscanTool
        crypto_tools["etherscan_tool"] = EtherscanTool()
    except (ImportError, ModuleNotFoundError):
        logger.warning("EtherscanTool not available - skipping initialization")
    
    return crypto_tools


class CrewFactory:
    """
    Factory for creating and managing CrewAI crews.
    
    This class is responsible for building and managing CrewAI crews,
    including agent creation, tool assignment, task definition, and
    crew orchestration. It also handles caching of agents and crews
    to avoid recreating them unnecessarily.
    """
    
    def __init__(self):
        """Initialize the CrewFactory with necessary clients and tools."""
        # Initialize clients
        self.neo4j_client = Neo4jClient()
        self.gemini_client = GeminiClient()
        self.e2b_client = E2BClient()
        self.llm_provider = GeminiLLMProvider()
        
        # Initialize tools
        self.tools = self._initialize_tools()
        
        # Initialize caches
        self.agents_cache = {}
        self.crews_cache = {}
    
    def _initialize_tools(self) -> Dict[str, Any]:
        """
        Initialize all available tools.
        
        Returns:
            Dictionary of tools by name
        """
        tools = {}
        
        # Core tools
        tools["graph_query_tool"] = GraphQueryTool(neo4j_client=self.neo4j_client)
        tools["sandbox_exec_tool"] = SandboxExecTool(e2b_client=self.e2b_client)
        tools["code_gen_tool"] = CodeGenTool(gemini_client=self.gemini_client)
        tools["pattern_library_tool"] = PatternLibraryTool(neo4j_client=self.neo4j_client)
        tools["neo4j_schema_tool"] = Neo4jSchemaTool(neo4j_client=self.neo4j_client)
        tools["template_engine_tool"] = TemplateEngineTool()
        tools["policy_docs_tool"] = PolicyDocsTool(gemini_client=self.gemini_client)
        tools["fraud_ml_tool"] = FraudMLTool(neo4j_client=self.neo4j_client)
        
        # Add crypto tools if available
        crypto_tools = create_crypto_tools()
        tools.update(crypto_tools)
        
        return tools
    
    async def connect(self):
        """
        Connect to external services.
        
        This method establishes connections to Neo4j and other services
        that require explicit connection.
        
        Raises:
            Exception: If connection to any service fails
        """
        try:
            # Connect to Neo4j
            await self.neo4j_client.connect()
            
            # Other connections can be added here
            
            logger.info("Connected to external services")
        except Exception as e:
            logger.error(f"Error connecting to external services: {e}")
            raise
    
    async def close(self):
        """
        Close connections to external services.
        
        This method should be called when the factory is no longer needed
        to ensure proper cleanup of resources.
        """
        try:
            # Close Neo4j connection
            if hasattr(self.neo4j_client, "driver") and self.neo4j_client.driver:
                await self.neo4j_client.close()
            
            # Close E2B sandboxes
            await self.e2b_client.close_all_sandboxes()
            
            logger.info("Closed connections to external services")
        except Exception as e:
            logger.error(f"Error closing connections: {e}")
    
    def get_tool(self, tool_name: str) -> Optional[Any]:
        """
        Get a tool by name.
        
        Args:
            tool_name: Name of the tool to get
            
        Returns:
            Tool instance or None if not found
        """
        return self.tools.get(tool_name)
    
    def create_agent(self, agent_id: str) -> Agent:
        """
        Create a CrewAI agent from configuration.
        
        This method loads the agent configuration from YAML files and
        creates a CrewAI agent with the specified tools.
        
        Args:
            agent_id: ID of the agent to create
            
        Returns:
            CrewAI Agent instance
        """
        # Check if agent is already cached
        if agent_id in self.agents_cache:
            return self.agents_cache[agent_id]
        
        # Load agent configuration
        config = load_agent_config(agent_id)
        
        # Get tools for agent
        agent_tools = []
        for tool_name in config.tools:
            tool = self.get_tool(tool_name)
            if tool:
                agent_tools.append(tool)
            else:
                logger.warning(f"Tool '{tool_name}' not found for agent '{agent_id}'")
        
        # Create agent
        agent = Agent(
            id=config.id,
            role=config.role,
            goal=config.goal,
            backstory=config.backstory,
            verbose=config.verbose,
            allow_delegation=config.allow_delegation,
            tools=agent_tools,
            llm=self.llm_provider,
            max_iter=config.max_iter,
            max_rpm=config.max_rpm if hasattr(config, "max_rpm") else None,
            memory=config.memory if hasattr(config, "memory") else False
        )
        
        # Cache agent
        self.agents_cache[agent_id] = agent
        
        logger.info(f"Created agent '{agent_id}' with {len(agent_tools)} tools")
        return agent
    
    def create_tasks(self, crew_name: str, agents: Dict[str, Agent]) -> List[Task]:
        """
        Create tasks for a crew.
        
        This method creates CrewAI tasks for the specified crew based on
        predefined task templates.
        
        Args:
            crew_name: Name of the crew
            agents: Dictionary of agents by ID
            
        Returns:
            List of CrewAI Task instances
        """
        tasks = []
        
        # Create tasks based on crew type
        if crew_name == "fraud_investigation":
            # NLQ translator task
            if "nlq_translator" in agents:
                tasks.append(Task(
                    description="Translate user query into Cypher",
                    agent=agents["nlq_translator"],
                    expected_output="Executable Cypher query and explanation"
                ))
            
            # Graph analyst task
            if "graph_analyst" in agents:
                tasks.append(Task(
                    description="Execute graph query and analyze results",
                    agent=agents["graph_analyst"],
                    expected_output="Structured graph data with initial analysis",
                    context=["Translate user query into Cypher"]
                ))
            
            # Fraud pattern hunter task
            if "fraud_pattern_hunter" in agents:
                tasks.append(Task(
                    description="Detect fraud patterns and anomalies",
                    agent=agents["fraud_pattern_hunter"],
                    expected_output="Identified patterns, anomalies, and risk assessment",
                    context=["Execute graph query and analyze results"]
                ))
            
            # Sandbox coder task (optional)
            if "sandbox_coder" in agents:
                tasks.append(Task(
                    description="Generate and execute code for advanced analysis",
                    agent=agents["sandbox_coder"],
                    expected_output="Code execution results and insights",
                    context=["Execute graph query and analyze results", "Detect fraud patterns and anomalies"]
                ))
            
            # Compliance checker task
            if "compliance_checker" in agents:
                tasks.append(Task(
                    description="Verify compliance with regulations",
                    agent=agents["compliance_checker"],
                    expected_output="Compliance assessment and SAR recommendations",
                    context=["Detect fraud patterns and anomalies"]
                ))
            
            # Report writer task
            if "report_writer" in agents:
                tasks.append(Task(
                    description="Generate comprehensive investigation report",
                    agent=agents["report_writer"],
                    expected_output="Markdown report with executive summary, findings, and visualizations",
                    context=[
                        "Execute graph query and analyze results",
                        "Detect fraud patterns and anomalies",
                        "Verify compliance with regulations"
                    ]
                ))
        
        elif crew_name == "alert_enrichment":
            # NLQ translator task
            if "nlq_translator" in agents:
                tasks.append(Task(
                    description="Translate alert data into Cypher",
                    agent=agents["nlq_translator"],
                    expected_output="Executable Cypher query for alert context"
                ))
            
            # Graph analyst task
            if "graph_analyst" in agents:
                tasks.append(Task(
                    description="Enrich alert with graph context",
                    agent=agents["graph_analyst"],
                    expected_output="Alert context with related entities and transactions",
                    context=["Translate alert data into Cypher"]
                ))
            
            # Fraud pattern hunter task
            if "fraud_pattern_hunter" in agents:
                tasks.append(Task(
                    description="Apply pattern detection to alert",
                    agent=agents["fraud_pattern_hunter"],
                    expected_output="Pattern matches and anomaly scores",
                    context=["Enrich alert with graph context"]
                ))
            
            # Compliance checker task
            if "compliance_checker" in agents:
                tasks.append(Task(
                    description="Assess regulatory implications",
                    agent=agents["compliance_checker"],
                    expected_output="Regulatory assessment and recommendations",
                    context=["Apply pattern detection to alert"]
                ))
            
            # Report writer task
            if "report_writer" in agents:
                tasks.append(Task(
                    description="Generate alert enrichment report",
                    agent=agents["report_writer"],
                    expected_output="Structured alert report with findings and recommendations",
                    context=[
                        "Enrich alert with graph context",
                        "Apply pattern detection to alert",
                        "Assess regulatory implications"
                    ]
                ))
        
        elif crew_name == "red_blue_simulation":
            # Red team adversary task
            if "red_team_adversary" in agents:
                tasks.append(Task(
                    description="Generate synthetic fraud scenario",
                    agent=agents["red_team_adversary"],
                    expected_output="Synthetic transaction data and ground truth"
                ))
                
                # Additional red team task
                tasks.append(Task(
                    description="Execute synthetic transactions",
                    agent=agents["red_team_adversary"],
                    expected_output="Executed transaction graph with obfuscation",
                    context=["Generate synthetic fraud scenario"]
                ))
            
            # Graph analyst task
            if "graph_analyst" in agents:
                tasks.append(Task(
                    description="Analyze transaction graph",
                    agent=agents["graph_analyst"],
                    expected_output="Graph analysis and initial findings",
                    context=["Execute synthetic transactions"]
                ))
            
            # Fraud pattern hunter task
            if "fraud_pattern_hunter" in agents:
                tasks.append(Task(
                    description="Detect patterns in synthetic data",
                    agent=agents["fraud_pattern_hunter"],
                    expected_output="Detected patterns and anomalies",
                    context=["Analyze transaction graph"]
                ))
            
            # Report writer task
            if "report_writer" in agents:
                tasks.append(Task(
                    description="Generate simulation report",
                    agent=agents["report_writer"],
                    expected_output="Simulation report with detection effectiveness",
                    context=[
                        "Generate synthetic fraud scenario",
                        "Analyze transaction graph",
                        "Detect patterns in synthetic data"
                    ]
                ))
        
        elif crew_name == "crypto_investigation":
            # Crypto data collector task
            if "crypto_data_collector" in agents:
                tasks.append(Task(
                    description="Collect on-chain and off-chain data",
                    agent=agents["crypto_data_collector"],
                    expected_output="Structured crypto data from multiple sources"
                ))
            
            # Blockchain detective task
            if "blockchain_detective" in agents:
                tasks.append(Task(
                    description="Trace on-chain transactions",
                    agent=agents["blockchain_detective"],
                    expected_output="Transaction flow and entity identification",
                    context=["Collect on-chain and off-chain data"]
                ))
            
            # DeFi analyst task
            if "defi_analyst" in agents:
                tasks.append(Task(
                    description="Analyze DeFi protocol interactions",
                    agent=agents["defi_analyst"],
                    expected_output="DeFi protocol analysis and risk assessment",
                    context=["Collect on-chain and off-chain data", "Trace on-chain transactions"]
                ))
            
            # Whale tracker task
            if "whale_tracker" in agents:
                tasks.append(Task(
                    description="Identify and analyze whale activity",
                    agent=agents["whale_tracker"],
                    expected_output="Whale activity patterns and impact analysis",
                    context=["Trace on-chain transactions"]
                ))
            
            # Protocol investigator task
            if "protocol_investigator" in agents:
                tasks.append(Task(
                    description="Investigate protocol-specific risks",
                    agent=agents["protocol_investigator"],
                    expected_output="Protocol risk assessment and security analysis",
                    context=["Analyze DeFi protocol interactions"]
                ))
            
            # Report writer task
            if "report_writer" in agents:
                tasks.append(Task(
                    description="Generate crypto investigation report",
                    agent=agents["report_writer"],
                    expected_output="Comprehensive crypto investigation report",
                    context=[
                        "Trace on-chain transactions",
                        "Analyze DeFi protocol interactions",
                        "Identify and analyze whale activity",
                        "Investigate protocol-specific risks"
                    ]
                ))
        
        # If no tasks were created, log a warning
        if not tasks:
            logger.warning(f"No tasks created for crew '{crew_name}'")
        
        return tasks
    
    async def create_crew(self, crew_name: str) -> Crew:
        """
        Create a CrewAI crew from configuration.
        
        This method loads the crew configuration from YAML files and
        creates a CrewAI crew with the specified agents and tasks.
        
        Args:
            crew_name: Name of the crew to create
            
        Returns:
            CrewAI Crew instance
        """
        # Check if crew is already cached
        if crew_name in self.crews_cache:
            return self.crews_cache[crew_name]
        
        # Load crew configuration
        config = load_crew_config(crew_name)
        
        # Create agents
        agents = {}
        for agent_id in config.agents:
            agents[agent_id] = self.create_agent(agent_id)
        
        # Create tasks
        tasks = self.create_tasks(crew_name, agents)
        
        # Determine process type
        process = Process.sequential
        if config.process_type == "hierarchical":
            process = Process.hierarchical
        
        # Create crew
        crew = Crew(
            agents=list(agents.values()),
            tasks=tasks,
            process=process,
            verbose=config.verbose,
            max_rpm=config.max_rpm if hasattr(config, "max_rpm") else None,
            memory=config.memory if hasattr(config, "memory") else False,
            cache=config.cache if hasattr(config, "cache") else False,
            manager_llm=self.llm_provider if process == Process.hierarchical else None,
            manager=agents.get(config.manager) if hasattr(config, "manager") and config.manager else None
        )
        
        # Cache crew
        self.crews_cache[crew_name] = crew
        
        logger.info(f"Created crew '{crew_name}' with {len(agents)} agents and {len(tasks)} tasks")
        return crew
    
    async def run_crew(
        self, 
        crew_name: str, 
        inputs: Optional[Dict[str, Any]] = None,
        task_id: Optional[str] = None,
        resume: bool = False
    ) -> Dict[str, Any]:
        """
        Run a CrewAI crew.
        
        This method creates and runs a CrewAI crew with the specified inputs.
        
        Args:
            crew_name: Name of the crew to run
            inputs: Input data for the crew
            task_id: Optional task ID for tracking
            resume: Whether this is resuming a paused execution
            
        Returns:
            Dictionary with execution results
        """
        try:
            # Create crew
            crew = await self.create_crew(crew_name)
            
            # Prepare inputs
            crew_inputs = inputs or {}
            
            # Add task_id to inputs if provided
            if task_id:
                crew_inputs["task_id"] = task_id
            
            # Add resume flag to inputs if provided
            if resume:
                crew_inputs["resume"] = True
            
            # Start timer for metrics
            start_time = asyncio.get_event_loop().time()
            
            # Run crew
            result = crew.kickoff(inputs=crew_inputs)
            
            # Record metrics
            duration = asyncio.get_event_loop().time() - start_time
            crew_task_duration_seconds.labels(crew=crew_name).observe(duration)
            
            # Process result
            if isinstance(result, TaskOutput):
                return {
                    "success": True,
                    "result": result.raw_output,
                    "task_id": result.task_id,
                    "agent_id": result.agent_id
                }
            else:
                # Handle string or other result types
                return {
                    "success": True,
                    "result": result
                }
        
        except Exception as e:
            logger.exception(f"Error running crew '{crew_name}': {e}")
            return {
                "success": False,
                "error": str(e)
            }
        finally:
            # Close connections
            await self.close()
    
    @staticmethod
    async def load(crew_name: str) -> Crew:
        """
        Load a crew by name.
        
        This static method creates a CrewFactory, connects to external services,
        and creates a crew with the specified name.
        
        Args:
            crew_name: Name of the crew to load
            
        Returns:
            CrewAI Crew instance
        """
        factory = CrewFactory()
        await factory.connect()
        return await factory.create_crew(crew_name)
    
    @staticmethod
    def get_available_crews() -> List[str]:
        """
        Get a list of available crew names.
        
        Returns:
            List of crew names
        """
        return list(DEFAULT_CREW_CONFIGS.keys())
