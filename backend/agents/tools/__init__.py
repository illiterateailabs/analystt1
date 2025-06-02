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

# Import GNN tools
try:
    from backend.agents.tools.gnn_fraud_detection_tool import GNNFraudDetectionTool
except ImportError:
    GNNFraudDetectionTool = None

try:
    from backend.agents.tools.gnn_training_tool import GNNTrainingTool
except ImportError:
    GNNTrainingTool = None

# Import MCP client for tool integration
try:
    from backend.mcp import get_mcp_tools_for_factory
except ImportError:
    get_mcp_tools_for_factory = None

# Function to get all available tools (including MCP tools)
def get_all_tools():
    """
    Get all available tools, including traditional tools and MCP tools.
    
    Returns:
        List of all available tools.
    """
    tools = []
    
    # Add traditional tools
    for tool_class in [
        GraphQueryTool,
        SandboxExecTool,
        CodeGenTool,
        PatternLibraryTool,
        PolicyDocsTool,
        TemplateEngineTool,
        Neo4jSchemaTool,
        RandomTxGeneratorTool,
        GNNFraudDetectionTool,
        GNNTrainingTool,
    ]:
        if tool_class is not None:
            try:
                tools.append(tool_class())
            except Exception:
                # Skip tools that fail to initialize
                pass
    
    # Add MCP tools if available
    if get_mcp_tools_for_factory is not None:
        try:
            mcp_tools = get_mcp_tools_for_factory()
            tools.extend(mcp_tools)
        except Exception:
            # Skip MCP tools if they fail to initialize
            pass
    
    return tools

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
        "GNNFraudDetectionTool",
        "GNNTrainingTool",
        "get_all_tools",
    ]
    if globals().get(name) is not None
]
