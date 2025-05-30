"""
CrewFactory for building and managing CrewAI crews.

This module provides a factory class for creating and managing CrewAI crews
based on configuration. It handles agent creation, tool assignment, task definition,
and crew orchestration for different use cases like fraud investigation and
alert enrichment.
"""

import logging
import asyncio
from typing import Any, Dict, List, Optional, Union, Tuple
from pathlib import Path

from crewai import Agent, Task, Crew, Process
from crewai.agent import TaskOutput

from backend.agents.config import AgentConfig, CrewConfig, load_agent_config, load_crew_config
from backend.agents.tools import (
    GraphQueryTool,
    SandboxExecTool,
    CodeGenTool,
    PatternLibraryTool,
    PolicyDocsTool,
    TemplateEngineTool,
    Neo4jSchemaTool,
    RandomTxGeneratorTool,
)
# Import crypto tools module
from backend.agents.tools.crypto import (
    DuneAnalyticsTool,
    DefiLlamaTool, 
    EtherscanTool,
    create_crypto_tools
)
from backend.agents.llm import GeminiLLMProvider
from backend.integrations.neo4j_client import Neo4jClient
from backend.integrations.gemini_client import GeminiClient
from backend.integrations.e2b_client import E2BClient
from backend.config import settings

logger = logging.getLogger(__name__)


class CrewFactory:
    """
    Factory for building and managing CrewAI crews.
    
    This class handles the creation of agents, tools, tasks, and crews
    based on configuration. It supports different use cases like fraud
    investigation, alert enrichment, and red-blue team simulations.
    """
    
    def __init__(self):
        """Initialize the CrewFactory."""
        self.neo4j_client = Neo4jClient()
        self.gemini_client = GeminiClient()
        self.e2b_client = E2BClient()
        self.llm_provider = GeminiLLMProvider()
        
        # Initialize tools
        self.tools = self._initialize_tools()
        
        # Cache for created agents and crews
        self.agents_cache = {}
        self.crews_cache = {}
        
        logger.info("CrewFactory initialized")
    
    def _initialize_tools(self) -> Dict[str, Any]:
        """
        Initialize all available tools.
        
        Returns:
            Dictionary of tool instances by name
        """
        tools = {
            "graph_query_tool": GraphQueryTool(neo4j_client=self.neo4j_client),
            "sandbox_exec_tool": SandboxExecTool(e2b_client=self.e2b_client),
            "code_gen_tool": CodeGenTool(gemini_client=self.gemini_client),
            "pattern_library_tool": PatternLibraryTool(
                gemini_client=self.gemini_client,
                neo4j_client=self.neo4j_client
            ),
            "neo4j_schema_tool": Neo4jSchemaTool(neo4j_client=self.neo4j_client),
        }
        
        # Add additional tools if available
        try:
            tools["policy_docs_tool"] = PolicyDocsTool()
        except ImportError:
            logger.warning("PolicyDocsTool not available, skipping")
        
        try:
            tools["template_engine_tool"] = TemplateEngineTool()
        except ImportError:
            logger.warning("TemplateEngineTool not available, skipping")
        
        try:
            tools["random_tx_generator_tool"] = RandomTxGeneratorTool()
        except ImportError:
            logger.warning("RandomTxGeneratorTool not available, skipping")
            
        # Initialize crypto tools
        try:
            # Prepare API keys for crypto tools
            crypto_api_keys = {
                "dune_api_key": getattr(settings, "dune_api_key", None),
                "etherscan_api_key": getattr(settings, "etherscan_api_key", None),
                "bscscan_api_key": getattr(settings, "bscscan_api_key", None),
                "polygonscan_api_key": getattr(settings, "polygonscan_api_key", None),
            }
            
            # Create crypto tools
            crypto_tools = create_crypto_tools(
                neo4j_client=self.neo4j_client,
                api_keys=crypto_api_keys
            )
            
            # Add crypto tools to the tools dictionary
            tools.update(crypto_tools)
            
            logger.info(f"Initialized {len(crypto_tools)} crypto tools")
        except Exception as e:
            logger.warning(f"Error initializing crypto tools: {e}")
        
        logger.info(f"Initialized {len(tools)} tools")
        return tools
    
    async def connect(self):
        """Connect to external services."""
        try:
            await self.neo4j_client.connect()
            logger.info("Connected to Neo4j database")
        except Exception as e:
            logger.error(f"Error connecting to Neo4j: {e}")
            raise
    
    async def close(self):
        """Close connections to external services."""
        try:
            if hasattr(self.neo4j_client, 'driver') and self.neo4j_client.driver is not None:
                await self.neo4j_client.close()
                logger.info("Closed Neo4j connection")
            
            # Close any active sandbox
            for tool_name, tool in self.tools.items():
                if tool_name == "sandbox_exec_tool" and hasattr(tool, "close"):
                    await tool.close()
                    logger.info("Closed e2b sandbox")
                    
                # Close crypto tool connections if they have close methods
                if tool_name in ["dune_analytics_tool", "defillama_tool", "etherscan_tool"] and hasattr(tool, "close"):
                    await tool.close()
                    logger.info(f"Closed {tool_name} connection")
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
        Create a CrewAI agent based on configuration.
        
        Args:
            agent_id: ID of the agent to create
            
        Returns:
            CrewAI Agent instance
        """
        # Check cache first
        if agent_id in self.agents_cache:
            return self.agents_cache[agent_id]
        
        # Load agent configuration
        config = load_agent_config(agent_id)
        
        # Get tools for the agent
        agent_tools = []
        for tool_name in config.tools:
            tool = self.get_tool(tool_name)
            if tool:
                agent_tools.append(tool)
            else:
                logger.warning(f"Tool not found for agent {agent_id}: {tool_name}")
        
        # Create the agent
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
            max_rpm=config.max_rpm,
            memory=config.memory,
        )
        
        # Cache the agent
        self.agents_cache[agent_id] = agent
        
        logger.info(f"Created agent: {agent_id}")
        return agent
    
    def create_tasks(self, crew_name: str, agents: Dict[str, Agent]) -> List[Task]:
        """
        Create tasks for a crew based on the use case.
        
        Args:
            crew_name: Name of the crew
            agents: Dictionary of agents by ID
            
        Returns:
            List of CrewAI Task instances
        """
        tasks = []
        
        if crew_name == "fraud_investigation":
            # Create tasks for fraud investigation crew
            tasks = [
                Task(
                    description="Convert the user's question into a Cypher query",
                    expected_output="A valid Cypher query that can be executed against Neo4j",
                    agent=agents["nlq_translator"],
                ),
                Task(
                    description="Execute the Cypher query and analyze the results",
                    expected_output="Structured analysis of the graph query results",
                    agent=agents["graph_analyst"],
                    context=[tasks[0]] if tasks else None,  # Use output from previous task
                ),
                Task(
                    description="Identify potential fraud patterns in the data",
                    expected_output="List of suspicious patterns with risk scores",
                    agent=agents["fraud_pattern_hunter"],
                    context=[tasks[1]] if len(tasks) > 1 else None,
                ),
                Task(
                    description="Generate Python code for detailed analysis if needed",
                    expected_output="Python code for advanced analysis",
                    agent=agents["sandbox_coder"],
                    context=[tasks[2]] if len(tasks) > 2 else None,
                ),
                Task(
                    description="Check compliance with AML regulations",
                    expected_output="Compliance assessment and recommendations",
                    agent=agents["compliance_checker"],
                    context=[tasks[2], tasks[3]] if len(tasks) > 3 else None,
                ),
                Task(
                    description="Produce a comprehensive investigation report",
                    expected_output="Markdown report with findings and visualizations",
                    agent=agents["report_writer"],
                    context=[tasks[1], tasks[2], tasks[3], tasks[4]] if len(tasks) > 4 else None,
                ),
            ]
        
        elif crew_name == "alert_enrichment":
            # Create tasks for alert enrichment crew
            tasks = [
                Task(
                    description="Convert the alert details into Cypher queries",
                    expected_output="Cypher queries to gather context about the alert",
                    agent=agents["nlq_translator"],
                ),
                Task(
                    description="Execute queries to gather supporting evidence",
                    expected_output="Structured evidence related to the alert",
                    agent=agents["graph_analyst"],
                    context=[tasks[0]] if tasks else None,
                ),
                Task(
                    description="Identify relevant fraud patterns and calculate risk score",
                    expected_output="Risk assessment with supporting patterns",
                    agent=agents["fraud_pattern_hunter"],
                    context=[tasks[1]] if len(tasks) > 1 else None,
                ),
                Task(
                    description="Check compliance requirements for this alert type",
                    expected_output="Compliance assessment and required actions",
                    agent=agents["compliance_checker"],
                    context=[tasks[2]] if len(tasks) > 2 else None,
                ),
                Task(
                    description="Generate alert summary with recommended actions",
                    expected_output="Concise alert summary with action items",
                    agent=agents["report_writer"],
                    context=[tasks[1], tasks[2], tasks[3]] if len(tasks) > 3 else None,
                ),
            ]
        
        elif crew_name == "red_blue_simulation":
            # Create tasks for red-blue team simulation crew
            tasks = [
                Task(
                    description="Generate synthetic fraud scenario",
                    expected_output="Detailed fraud scenario with transaction patterns",
                    agent=agents["red_team_adversary"],
                ),
                Task(
                    description="Execute the scenario against the test database",
                    expected_output="Confirmation of scenario execution",
                    agent=agents["red_team_adversary"],
                    context=[tasks[0]] if tasks else None,
                ),
                Task(
                    description="Analyze the database for suspicious patterns",
                    expected_output="Analysis of detected patterns",
                    agent=agents["graph_analyst"],
                    context=[tasks[1]] if len(tasks) > 1 else None,
                ),
                Task(
                    description="Identify fraud patterns in the simulation",
                    expected_output="List of detected fraud patterns",
                    agent=agents["fraud_pattern_hunter"],
                    context=[tasks[2]] if len(tasks) > 2 else None,
                ),
                Task(
                    description="Generate simulation results report",
                    expected_output="Report comparing the actual scenario with detected patterns",
                    agent=agents["report_writer"],
                    context=[tasks[0], tasks[3]] if len(tasks) > 3 else None,
                ),
            ]
        
        elif crew_name == "crypto_investigation":
            # Create tasks for crypto investigation crew
            tasks = [
                Task(
                    description="Collect and normalize data from multiple blockchain sources",
                    expected_output="Structured blockchain data from multiple sources",
                    agent=agents["crypto_data_collector"],
                ),
                Task(
                    description="Trace transactions and identify related addresses",
                    expected_output="Transaction flow analysis and address clusters",
                    agent=agents["blockchain_detective"],
                    context=[tasks[0]] if tasks else None,
                ),
                Task(
                    description="Analyze DeFi protocol interactions and risks",
                    expected_output="DeFi protocol analysis with risk assessment",
                    agent=agents["defi_analyst"],
                    context=[tasks[0], tasks[1]] if len(tasks) > 1 else None,
                ),
                Task(
                    description="Track large holder movements and their impact",
                    expected_output="Whale activity analysis with potential market impact",
                    agent=agents["whale_tracker"],
                    context=[tasks[1]] if len(tasks) > 1 else None,
                ),
                Task(
                    description="Analyze smart contract interactions and protocol behavior",
                    expected_output="Smart contract analysis with potential vulnerabilities",
                    agent=agents["protocol_investigator"],
                    context=[tasks[2], tasks[3]] if len(tasks) > 3 else None,
                ),
                Task(
                    description="Produce comprehensive crypto investigation report",
                    expected_output="Markdown report with blockchain analysis findings",
                    agent=agents["report_writer"],
                    context=[tasks[1], tasks[2], tasks[3], tasks[4]] if len(tasks) > 4 else None,
                ),
            ]
        
        else:
            logger.warning(f"No predefined tasks for crew: {crew_name}")
        
        logger.info(f"Created {len(tasks)} tasks for crew: {crew_name}")
        return tasks
    
    async def create_crew(self, crew_name: str) -> Crew:
        """
        Create a CrewAI crew based on configuration.
        
        Args:
            crew_name: Name of the crew to create
            
        Returns:
            CrewAI Crew instance
        """
        # Check cache first
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
        
        # Determine the process type
        process_type = Process.sequential
        if config.process_type.lower() == "hierarchical":
            process_type = Process.hierarchical
        
        # Determine the manager
        manager = None
        if config.manager and config.manager in agents:
            manager = agents[config.manager]
        
        # Create the crew
        crew = Crew(
            agents=list(agents.values()),
            tasks=tasks,
            process=process_type,
            manager=manager,
            verbose=config.verbose,
            max_rpm=config.max_rpm,
            memory=config.memory,
            cache=config.cache,
        )
        
        # Cache the crew
        self.crews_cache[crew_name] = crew
        
        logger.info(f"Created crew: {crew_name}")
        return crew
    
    @classmethod
    async def load(cls, crew_name: str) -> Crew:
        """
        Load a crew by name.
        
        Args:
            crew_name: Name of the crew to load
            
        Returns:
            CrewAI Crew instance
        """
        factory = cls()
        await factory.connect()
        return await factory.create_crew(crew_name)
    
    async def run_crew(self, crew_name: str, inputs: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Run a crew with the given inputs.
        
        Args:
            crew_name: Name of the crew to run
            inputs: Dictionary of input values
            
        Returns:
            Dictionary containing the crew results
        """
        try:
            # Create the crew
            crew = await self.create_crew(crew_name)
            
            # Run the crew
            logger.info(f"Running crew: {crew_name}")
            result = crew.kickoff(inputs=inputs or {})
            
            # Process the result
            if isinstance(result, TaskOutput):
                output = {
                    "success": True,
                    "result": result.raw_output,
                    "task_id": result.task_id,
                    "agent_id": result.agent_id
                }
            else:
                output = {
                    "success": True,
                    "result": result
                }
            
            logger.info(f"Crew {crew_name} completed successfully")
            return output
            
        except Exception as e:
            logger.error(f"Error running crew {crew_name}: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }
        finally:
            # Close connections
            await self.close()
    
    @staticmethod
    def get_available_crews() -> List[str]:
        """
        Get a list of available crew names.
        
        Returns:
            List of available crew names
        """
        from backend.agents.config import DEFAULT_CREW_CONFIGS
        return list(DEFAULT_CREW_CONFIGS.keys())
