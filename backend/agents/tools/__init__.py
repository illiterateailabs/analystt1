"""
CrewAI Tools for Analyst Augmentation Agent.

This package contains custom tools that can be used by CrewAI agents
to interact with external systems and services.
"""

# Only import tools that actually exist to prevent import errors
try:
    from backend.agents.tools.graph_query_tool import GraphQueryTool
except ImportError:
    GraphQueryTool = None

try:
    from backend.agents.tools.sandbox_exec_tool import SandboxExecTool
except ImportError:
    SandboxExecTool = None

try:
    from backend.agents.tools.code_gen_tool import CodeGenTool
except ImportError:
    CodeGenTool = None

try:
    from backend.agents.tools.pattern_library_tool import PatternLibraryTool
except ImportError:
    PatternLibraryTool = None

try:
    from backend.agents.tools.policy_docs_tool import PolicyDocsTool
except ImportError:
    PolicyDocsTool = None

try:
    from backend.agents.tools.template_engine_tool import TemplateEngineTool
except ImportError:
    TemplateEngineTool = None

try:
    from backend.agents.tools.neo4j_schema_tool import Neo4jSchemaTool
except ImportError:
    Neo4jSchemaTool = None

try:
    from backend.agents.tools.random_tx_generator_tool import RandomTxGeneratorTool
except ImportError:
    RandomTxGeneratorTool = None

# Export only the tools that successfully imported
__all__ = [
    name for name in [
        "GraphQueryTool",
        "SandboxExecTool",
        "CodeGenTool",
        "PatternLibraryTool",
        "PolicyDocsTool",
        "TemplateEngineTool",
        "Neo4jSchemaTool",
        "RandomTxGeneratorTool",
    ]
    if globals().get(name) is not None
]
