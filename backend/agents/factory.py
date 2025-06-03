"""
CrewAI Factory for building and managing crews.

This module provides a factory for creating and managing CrewAI crews,
including agent creation, tool assignment, task definition, and crew
orchestration.
"""

import os
import uuid
import logging
import asyncio
from typing import Dict, List, Optional, Any, Union, Set
from datetime import datetime
import traceback

from crewai import Agent, Task, Crew, Process
from crewai.agent import TaskOutput

from backend.agents.config import (
    load_agent_config,
    load_crew_config,
    get_available_crews,
    AGENT_CONFIGS_CREWS_DIR
)
from backend.agents.tools import (
    GraphQueryTool,
    SandboxExecTool,
    CodeGenTool,
    PatternLibraryTool,
    Neo4jSchemaTool,
    PolicyDocsTool,
    TemplateEngineTool,
    GNNFraudDetectionTool,
    GNNTrainingTool,
    GraphQLQueryTool,
    CryptoAnomalyTool,
    CryptoCSVLoaderTool,
    RandomTxGeneratorTool,
    create_crypto_tools
)
from backend.integrations.neo4j_client import Neo4jClient
from backend.integrations.gemini_client import GeminiClient
from backend.integrations.e2b_client import E2BClient
from backend.agents.llm import GeminiLLMProvider
from backend.core.metrics import increment_counter, observe_value

# Configure logging
logger = logging.getLogger(__name__)

# Global dictionary to track running crews
RUNNING_CREWS = {}


def get_all_tools():
    """
    Get all available tools.
    
    Returns:
        Dict[str, Any]: Dictionary of tool instances keyed by tool name.
    """
    tools = {}
    
    # Core tools
    tools["graph_query_tool"] = GraphQueryTool()
    tools["sandbox_exec_tool"] = SandboxExecTool()
    tools["code_gen_tool"] = CodeGenTool()
    tools["pattern_library_tool"] = PatternLibraryTool()
    tools["neo4j_schema_tool"] = Neo4jSchemaTool()
    tools["policy_docs_tool"] = PolicyDocsTool()
    tools["template_engine_tool"] = TemplateEngineTool()
    
    # GNN tools
    tools["gnn_fraud_detection_tool"] = GNNFraudDetectionTool()
    tools["gnn_training_tool"] = GNNTrainingTool()
    
    # GraphQL and API tools
    tools["graphql_query_tool"] = GraphQLQueryTool()
    
    # Crypto tools
    tools["crypto_anomaly_tool"] = CryptoAnomalyTool()
    tools["crypto_csv_loader_tool"] = CryptoCSVLoaderTool()
    tools["random_tx_generator_tool"] = RandomTxGeneratorTool()
    
    # Add crypto-specific tools
    crypto_tools = create_crypto_tools()
    tools.update(crypto_tools)
    
    return tools


class CrewFactory:
    """
    Factory for creating and managing CrewAI crews.
    
    This class provides methods for creating agents, tasks, and crews,
    as well as running crews and managing their lifecycle.
    """
    
    def __init__(self):
        """Initialize the CrewFactory."""
        # Initialize clients
        self.neo4j_client = Neo4jClient()
        self.gemini_client = GeminiClient()
        self.e2b_client = E2BClient()
        
        # Initialize LLM provider
        self.llm_provider = GeminiLLMProvider()
        
        # Initialize tools
        self.tools = get_all_tools()
        
        # Initialize caches
        self.agents_cache = {}
        self.crews_cache = {}
    
    async def connect(self):
        """
        Connect to external services.
        
        This method should be called before using the factory.
        """
        try:
            # Connect to Neo4j
            await self.neo4j_client.connect()
            
            # Note: Add other service connections here if needed
            
            logger.info("Connected to external services")
        except Exception as e:
            logger.error(f"Failed to connect to external services: {e}")
            raise
    
    async def close(self):
        """
        Close connections to external services.
        
        This method should be called when the factory is no longer needed.
        """
        try:
            # Close Neo4j connection if it exists
            if hasattr(self.neo4j_client, 'driver') and self.neo4j_client.driver:
                await self.neo4j_client.close()
            
            # Close E2B sandboxes
            if hasattr(self.e2b_client, 'close_all_sandboxes'):
                await self.e2b_client.close_all_sandboxes()
            
            logger.info("Closed connections to external services")
        except Exception as e:
            logger.error(f"Failed to close connections: {e}")
    
    def reload(self):
        """
        Reload configurations and tools.
        
        This method should be called when configurations or tools have changed.
        """
        logger.info("Reloading CrewFactory configurations and tools")
        
        # Clear caches
        self.agents_cache = {}
        self.crews_cache = {}
        
        # Reload tools
        self.tools = get_all_tools()
        
        # Note: We don't need to reload configurations as they're loaded on-demand
        
        logger.info("CrewFactory reloaded")
    
    def get_tool(self, tool_name: str) -> Optional[Any]:
        """
        Get a tool by name.
        
        Args:
            tool_name (str): Name of the tool.
            
        Returns:
            Optional[Any]: Tool instance or None if not found.
        """
        return self.tools.get(tool_name.lower())
    
    def create_agent(self, agent_id: str) -> Agent:
        """
        Create a CrewAI agent.
        
        Args:
            agent_id (str): ID of the agent configuration to use.
            
        Returns:
            Agent: CrewAI agent instance.
        """
        # Check if agent is already cached
        if agent_id in self.agents_cache:
            return self.agents_cache[agent_id]
        
        # Load agent configuration
        agent_config = load_agent_config(agent_id)
        
        # Get tools for agent
        agent_tools = []
        for tool_name in agent_config.tools:
            tool = self.get_tool(tool_name)
            if tool:
                agent_tools.append(tool)
            else:
                logger.warning(f"Tool '{tool_name}' not found for agent '{agent_id}'")
        
        # Create agent
        agent = Agent(
            id=agent_id,
            role=agent_config.role,
            goal=agent_config.goal,
            backstory=agent_config.backstory,
            verbose=agent_config.verbose,
            allow_delegation=agent_config.allow_delegation,
            tools=agent_tools,
            llm=self.llm_provider
        )
        
        # Cache agent
        self.agents_cache[agent_id] = agent
        
        return agent
    
    def create_tasks(self, crew_name: str, agents: Dict[str, Agent]) -> List[Task]:
        """
        Create tasks for a crew.
        
        Args:
            crew_name (str): Name of the crew.
            agents (Dict[str, Agent]): Dictionary of agents keyed by agent ID.
            
        Returns:
            List[Task]: List of tasks for the crew.
        """
        tasks = []
        
        # Create tasks based on crew type
        if crew_name == "fraud_investigation":
            # NLQ Translator task
            tasks.append(Task(
                description="Understand the user's query and translate it into a structured investigation plan. Identify entities, relationships, and potential fraud patterns to investigate.",
                expected_output="A structured investigation plan with entities, relationships, and potential fraud patterns.",
                agent=agents["nlq_translator"]
            ))
            
            # Graph Analyst task
            tasks.append(Task(
                description="Analyze the graph database to identify suspicious patterns and relationships between entities. Focus on transaction flows, unusual connections, and known fraud indicators.",
                expected_output="A detailed analysis of suspicious patterns and relationships in the graph.",
                agent=agents["graph_analyst"]
            ))
            
            # Fraud Pattern Hunter task
            tasks.append(Task(
                description="Identify specific fraud patterns in the data. Look for known typologies such as money laundering, structuring, round-tripping, and shell company patterns.",
                expected_output="A list of detected fraud patterns with evidence and confidence scores.",
                agent=agents["fraud_pattern_hunter"]
            ))
            
            # Sandbox Coder task
            tasks.append(Task(
                description="Generate and execute Python code to perform advanced analysis on the data. Create visualizations and statistical tests to support the investigation.",
                expected_output="Analysis results, visualizations, and statistical findings.",
                agent=agents["sandbox_coder"]
            ))
            
            # Compliance Checker task
            tasks.append(Task(
                description="Evaluate the findings against compliance requirements and regulations. Identify potential regulatory violations and recommend actions.",
                expected_output="Compliance assessment and recommendations for regulatory considerations.",
                agent=agents["compliance_checker"]
            ))
            
            # Report Writer task
            tasks.append(Task(
                description="Compile all findings into a comprehensive investigation report. Include executive summary, detailed findings, evidence, and recommendations.",
                expected_output="A complete investigation report with all findings and recommendations.",
                agent=agents["report_writer"]
            ))
        
        elif crew_name == "alert_enrichment":
            # NLQ Translator task
            tasks.append(Task(
                description="Understand the alert details and translate them into a structured enrichment plan. Identify what additional data is needed.",
                expected_output="A structured enrichment plan with data requirements.",
                agent=agents["nlq_translator"]
            ))
            
            # Graph Analyst task
            tasks.append(Task(
                description="Enrich the alert with additional context from the graph database. Find related entities and transactions that provide context.",
                expected_output="Enriched alert data with graph context.",
                agent=agents["graph_analyst"]
            ))
            
            # Fraud Pattern Hunter task
            tasks.append(Task(
                description="Evaluate the enriched alert against known fraud patterns. Determine if the alert matches any known typologies.",
                expected_output="Pattern matching results with confidence scores.",
                agent=agents["fraud_pattern_hunter"]
            ))
            
            # Compliance Checker task
            tasks.append(Task(
                description="Assess the regulatory implications of the alert. Identify any compliance requirements triggered by the alert.",
                expected_output="Regulatory assessment and compliance recommendations.",
                agent=agents["compliance_checker"]
            ))
            
            # Report Writer task
            tasks.append(Task(
                description="Create a concise alert enrichment report. Summarize the additional context and recommendations for the analyst.",
                expected_output="A concise alert enrichment report.",
                agent=agents["report_writer"]
            ))
        
        elif crew_name == "red_blue_simulation":
            # Red Team Adversary task
            tasks.append(Task(
                description="Simulate a sophisticated financial criminal. Design a scheme to evade detection using the current system.",
                expected_output="A detailed financial crime scheme designed to evade detection.",
                agent=agents["red_team_adversary"]
            ))
            
            # Graph Analyst task (first pass)
            tasks.append(Task(
                description="Analyze the proposed scheme using current detection methods. Identify what would be detected and what would be missed.",
                expected_output="Analysis of detection capabilities against the proposed scheme.",
                agent=agents["graph_analyst"]
            ))
            
            # Fraud Pattern Hunter task
            tasks.append(Task(
                description="Develop new detection patterns based on the red team simulation. Create rules to catch the simulated scheme.",
                expected_output="New detection patterns and rules.",
                agent=agents["fraud_pattern_hunter"]
            ))
            
            # Graph Analyst task (second pass)
            tasks.append(Task(
                description="Test the new detection patterns against the simulated scheme. Evaluate effectiveness and potential false positives.",
                expected_output="Evaluation of new detection patterns.",
                agent=agents["graph_analyst"]
            ))
            
            # Report Writer task
            tasks.append(Task(
                description="Document the red-blue simulation exercise. Include the scheme, detection gaps, new patterns, and recommendations.",
                expected_output="A comprehensive red-blue simulation report.",
                agent=agents["report_writer"]
            ))
        
        elif crew_name == "crypto_investigation":
            # Crypto Data Collector task
            tasks.append(Task(
                description="Collect relevant blockchain and cryptocurrency data for the investigation. Focus on addresses, transactions, and token movements.",
                expected_output="Comprehensive dataset of relevant crypto transactions and entities.",
                agent=agents["crypto_data_collector"]
            ))
            
            # Blockchain Detective task
            tasks.append(Task(
                description="Analyze on-chain activity to identify suspicious patterns. Look for mixing services, privacy coins, and suspicious address clustering.",
                expected_output="Analysis of suspicious on-chain activities and patterns.",
                agent=agents["blockchain_detective"]
            ))
            
            # DeFi Analyst task
            tasks.append(Task(
                description="Investigate DeFi protocol interactions. Look for flash loans, MEV, and other DeFi-specific exploits or suspicious activities.",
                expected_output="Analysis of DeFi protocol interactions and potential exploits.",
                agent=agents["defi_analyst"]
            ))
            
            # Whale Tracker task
            tasks.append(Task(
                description="Identify and analyze large transactions and whale wallet activities. Determine impact on markets and potential manipulation.",
                expected_output="Analysis of whale activities and market impact.",
                agent=agents["whale_tracker"]
            ))
            
            # Protocol Investigator task
            tasks.append(Task(
                description="Examine protocol-specific risks and vulnerabilities. Identify governance attacks, economic exploits, and protocol weaknesses.",
                expected_output="Assessment of protocol-specific risks and vulnerabilities.",
                agent=agents["protocol_investigator"]
            ))
            
            # Report Writer task
            tasks.append(Task(
                description="Compile all crypto investigation findings into a comprehensive report. Include technical details and layperson explanations.",
                expected_output="A comprehensive crypto investigation report.",
                agent=agents["report_writer"]
            ))
        
        # Add more crew types as needed
        
        return tasks
    
    async def create_crew(self, crew_name: str) -> Crew:
        """
        Create a CrewAI crew.
        
        Args:
            crew_name (str): Name of the crew configuration to use.
            
        Returns:
            Crew: CrewAI crew instance.
        """
        # Check if crew is already cached
        if crew_name in self.crews_cache:
            return self.crews_cache[crew_name]
        
        # Load crew configuration
        crew_config = load_crew_config(crew_name)
        
        # Create agents
        agents = {}
        for agent_id in crew_config.agents:
            agent = self.create_agent(agent_id)
            agents[agent_id] = agent
        
        # Create tasks
        tasks = self.create_tasks(crew_name, agents)
        
        # Determine process type
        process = Process.sequential
        if crew_config.process_type == "hierarchical":
            process = Process.hierarchical
        
        # Create crew
        crew = Crew(
            agents=list(agents.values()),
            tasks=tasks,
            process=process,
            verbose=crew_config.verbose,
            max_rpm=crew_config.max_rpm,
            memory=crew_config.memory,
            cache=crew_config.cache,
            manager_llm=self.llm_provider if crew_config.manager else None
        )
        
        # Cache crew
        self.crews_cache[crew_name] = crew
        
        return crew
    
    async def run_crew(self, crew_name: str, inputs: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Run a CrewAI crew.
        
        Args:
            crew_name (str): Name of the crew configuration to use.
            inputs (Dict[str, Any], optional): Inputs for the crew. Defaults to None.
            
        Returns:
            Dict[str, Any]: Result of the crew execution.
        """
        # Generate a unique task ID
        task_id = str(uuid.uuid4())
        
        # Initialize result
        result = {
            "success": False,
            "task_id": task_id,
            "crew_name": crew_name,
        }
        
        # Initialize inputs if None
        if inputs is None:
            inputs = {}
        
        # Track task in RUNNING_CREWS
        RUNNING_CREWS[task_id] = {
            "crew_name": crew_name,
            "state": "STARTING",
            "start_time": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "inputs": inputs,
            "current_agent": None,
            "context": {}  # Shared context for tools to store results
        }
        
        try:
            # Connect to external services
            await self.connect()
            
            # Create crew
            crew = await self.create_crew(crew_name)
            
            # Update task state
            RUNNING_CREWS[task_id]["state"] = "RUNNING"
            RUNNING_CREWS[task_id]["last_updated"] = datetime.now().isoformat()
            
            # Create a wrapper for kickoff to track context
            original_kickoff = crew.kickoff
            
            def kickoff_with_context(**kwargs):
                # Add context to inputs
                if "inputs" in kwargs and kwargs["inputs"] is not None:
                    if not isinstance(kwargs["inputs"], dict):
                        kwargs["inputs"] = {"input": kwargs["inputs"]}
                    
                    # Add context to inputs
                    kwargs["inputs"]["_context"] = RUNNING_CREWS[task_id]["context"]
                else:
                    kwargs["inputs"] = {"_context": RUNNING_CREWS[task_id]["context"]}
                
                # Run original kickoff
                output = original_kickoff(**kwargs)
                
                # Store context for future use
                if "_context" in kwargs["inputs"]:
                    RUNNING_CREWS[task_id]["context"] = kwargs["inputs"]["_context"]
                
                return output
            
            # Replace kickoff method with wrapper
            crew.kickoff = kickoff_with_context
            
            # Run crew
            output = crew.kickoff(inputs=inputs)
            
            # Update metrics
            increment_counter("crew_executions_total", {"crew": crew_name, "status": "success"})
            
            # Process output
            if isinstance(output, TaskOutput):
                result["result"] = output.raw_output
                result["agent_id"] = output.agent_id
                result["task_id"] = task_id  # Use our generated task_id
            else:
                result["result"] = str(output)
            
            # Mark as success
            result["success"] = True
            
            # Update task state
            RUNNING_CREWS[task_id]["state"] = "COMPLETED"
            RUNNING_CREWS[task_id]["completion_time"] = datetime.now().isoformat()
            RUNNING_CREWS[task_id]["last_updated"] = datetime.now().isoformat()
            RUNNING_CREWS[task_id]["result"] = result["result"]
            
        except Exception as e:
            # Log error
            logger.error(f"Failed to run crew '{crew_name}': {e}")
            logger.error(traceback.format_exc())
            
            # Update metrics
            increment_counter("crew_executions_total", {"crew": crew_name, "status": "error"})
            
            # Update result
            result["success"] = False
            result["error"] = str(e)
            
            # Update task state
            RUNNING_CREWS[task_id]["state"] = "ERROR"
            RUNNING_CREWS[task_id]["last_updated"] = datetime.now().isoformat()
            RUNNING_CREWS[task_id]["error"] = str(e)
            
        finally:
            # Close connections
            await self.close()
        
        return result
    
    @staticmethod
    def pause_crew(task_id: str, reason: str = None, review_id: str = None) -> bool:
        """
        Pause a running crew.
        
        Args:
            task_id (str): ID of the task to pause.
            reason (str, optional): Reason for pausing. Defaults to None.
            review_id (str, optional): ID of the associated review. Defaults to None.
            
        Returns:
            bool: True if successful, False otherwise.
        """
        if task_id not in RUNNING_CREWS:
            logger.warning(f"Cannot pause crew: Task ID '{task_id}' not found")
            return False
        
        if RUNNING_CREWS[task_id]["state"] != "RUNNING":
            logger.warning(f"Cannot pause crew: Task '{task_id}' is not in RUNNING state")
            return False
        
        # Update task state
        RUNNING_CREWS[task_id]["state"] = "PAUSED"
        RUNNING_CREWS[task_id]["paused_at"] = datetime.now().isoformat()
        RUNNING_CREWS[task_id]["pause_reason"] = reason
        RUNNING_CREWS[task_id]["review_id"] = review_id
        RUNNING_CREWS[task_id]["last_updated"] = datetime.now().isoformat()
        
        logger.info(f"Crew paused: Task ID '{task_id}'")
        return True
    
    @staticmethod
    def resume_crew(task_id: str, review_result: Dict[str, Any] = None) -> bool:
        """
        Resume a paused crew.
        
        Args:
            task_id (str): ID of the task to resume.
            review_result (Dict[str, Any], optional): Result of the review. Defaults to None.
            
        Returns:
            bool: True if successful, False otherwise.
        """
        if task_id not in RUNNING_CREWS:
            logger.warning(f"Cannot resume crew: Task ID '{task_id}' not found")
            return False
        
        if RUNNING_CREWS[task_id]["state"] != "PAUSED":
            logger.warning(f"Cannot resume crew: Task '{task_id}' is not in PAUSED state")
            return False
        
        # Update task state
        RUNNING_CREWS[task_id]["state"] = "RUNNING"
        RUNNING_CREWS[task_id]["resumed_at"] = datetime.now().isoformat()
        if review_result:
            RUNNING_CREWS[task_id]["review_result"] = review_result
        RUNNING_CREWS[task_id]["last_updated"] = datetime.now().isoformat()
        
        logger.info(f"Crew resumed: Task ID '{task_id}'")
        return True
    
    @staticmethod
    async def load(crew_name: str) -> Crew:
        """
        Load a crew by name.
        
        This is a convenience method for creating a factory, connecting to
        external services, and creating a crew.
        
        Args:
            crew_name (str): Name of the crew configuration to use.
            
        Returns:
            Crew: CrewAI crew instance.
        """
        factory = CrewFactory()
        await factory.connect()
        return await factory.create_crew(crew_name)
    
    @staticmethod
    def get_available_crews() -> List[str]:
        """
        Get a list of available crew names.
        
        Returns:
            List[str]: List of available crew names.
        """
        return get_available_crews()
