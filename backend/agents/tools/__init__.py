"""
CrewAI Tools for Analyst Augmentation Agent.

This package contains custom tools that can be used by CrewAI agents
to interact with external systems and services, including:

- GraphQueryTool: Execute Cypher queries against Neo4j
- SandboxExecTool: Run Python code in isolated e2b sandboxes
- CodeGenTool: Generate Python code using Gemini
- PatternLibraryTool: Access fraud pattern templates
- PolicyDocsTool: Retrieve compliance policies and regulations
- TemplateEngineTool: Generate reports from templates
- Neo4jSchemaTool: Retrieve Neo4j database schema
- RandomTxGeneratorTool: Generate random transactions for testing
"""

from backend.agents.tools.graph_query_tool import GraphQueryTool
from backend.agents.tools.sandbox_exec_tool import SandboxExecTool
from backend.agents.tools.code_gen_tool import CodeGenTool
from backend.agents.tools.pattern_library_tool import PatternLibraryTool
from backend.agents.tools.policy_docs_tool import PolicyDocsTool
from backend.agents.tools.template_engine_tool import TemplateEngineTool
from backend.agents.tools.neo4j_schema_tool import Neo4jSchemaTool
from backend.agents.tools.random_tx_generator_tool import RandomTxGeneratorTool

# Export all tools
__all__ = [
    "GraphQueryTool",
    "SandboxExecTool",
    "CodeGenTool",
    "PatternLibraryTool",
    "PolicyDocsTool",
    "TemplateEngineTool",
    "Neo4jSchemaTool",
    "RandomTxGeneratorTool",
]
