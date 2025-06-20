"""
Transaction Flow Analysis Tool

This tool analyzes transaction flows between wallets using Sim APIs to detect patterns,
visualize money movement, and identify potentially suspicious activities like peel chains,
circular flows, and layering. It supports both single wallet and multi-wallet network analysis,
with integration to the graph database for persistence.

The tool provides flow metrics, risk scoring, and supports both real-time and batch processing.
"""

import logging
import time
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple, Union
from pydantic import BaseModel, Field, validator
import networkx as nx

from crewai_tools import BaseTool
from backend.integrations.sim_client import SimClient
from backend.core.metrics import record_tool_usage, record_tool_error
from backend.core.events import emit_event, GraphAddEvent

logger = logging.getLogger(__name__)

# Constants for transaction flow analysis
MAX_FLOW_DEPTH = 10  # Maximum depth for transaction flow analysis
FLOW_VALUE_THRESHOLD = 10000  # $10k+ for significant flows
SUSPICIOUS_CYCLE_MIN_LENGTH = 3  # Minimum length for suspicious cycles
SUSPICIOUS_CYCLE_MAX_LENGTH = 7  # Maximum length for suspicious cycles
LAYERING_THRESHOLD = 5  # Minimum transfers for potential layering
PEEL_CHAIN_THRESHOLD = 4  # Minimum hops for potential peel chain
PEEL_CHAIN_DECAY_RATE = 0.7  # Expected value decay in peel chains (each hop ~70% of previous)
TIME_WINDOW_HOURS = 24  # Default time window for analysis (24 hours)
MAX_WALLETS = 100  # Maximum number of wallets to analyze in one batch


class TransactionNode(BaseModel):
    """Model representing a node (wallet or contract) in the transaction flow."""
    
    address: str = Field(..., description="Wallet or contract address")
    type: str = Field("wallet", description="Node type: wallet, contract, token, etc.")
    first_seen: str = Field(..., description="Timestamp when first seen in the analysis")
    last_seen: str = Field(..., description="Timestamp when last seen in the analysis")
    total_in_value_usd: float = Field(0, description="Total USD value received")
    total_out_value_usd: float = Field(0, description="Total USD value sent")
    net_flow_usd: float = Field(0, description="Net flow (in - out) in USD")
    transaction_count: int = Field(0, description="Number of transactions")
    chains: List[str] = Field(default_factory=list, description="Chains where this node is active")
    risk_score: Optional[float] = Field(None, description="Risk score if calculated")
    labels: List[str] = Field(default_factory=list, description="Node labels/categories")


class TransactionEdge(BaseModel):
    """Model representing a directed edge (transaction) in the transaction flow."""
    
    source: str = Field(..., description="Source address")
    target: str = Field(..., description="Target address")
    transaction_hash: str = Field(..., description="Transaction hash")
    value_usd: float = Field(..., description="Transaction value in USD")
    timestamp: str = Field(..., description="Transaction timestamp")
    chain: str = Field(..., description="Blockchain network")
    token_address: Optional[str] = Field(None, description="Token address if not native")
    token_symbol: Optional[str] = Field(None, description="Token symbol")
    transaction_type: str = Field(..., description="Type of transaction (SEND, RECEIVE, etc.)")
    block_number: Optional[int] = Field(None, description="Block number")
    gas_used: Optional[int] = Field(None, description="Gas used")
    edge_id: str = Field(..., description="Unique edge identifier")


class FlowPattern(BaseModel):
    """Model representing a detected flow pattern in the transaction graph."""
    
    pattern_id: str = Field(..., description="Unique identifier for the pattern")
    pattern_type: str = Field(..., description="Type of pattern (PEEL_CHAIN, CYCLE, LAYERING, etc.)")
    addresses: List[str] = Field(..., description="Addresses involved in the pattern")
    transactions: List[str] = Field(..., description="Transaction hashes in the pattern")
    start_time: str = Field(..., description="Start time of the pattern")
    end_time: str = Field(..., description="End time of the pattern")
    total_value_usd: float = Field(..., description="Total USD value in the pattern")
    confidence: float = Field(..., description="Confidence score for the pattern detection")
    description: str = Field(..., description="Human-readable description of the pattern")
    risk_score: float = Field(..., description="Risk score for the pattern (0-100)")
    path: Optional[List[str]] = Field(None, description="Sequential path of addresses in the pattern")


class FlowMetrics(BaseModel):
    """Model representing metrics calculated from the transaction flow."""
    
    total_transactions: int = Field(..., description="Total number of transactions analyzed")
    total_value_usd: float = Field(..., description="Total USD value of all transactions")
    unique_addresses: int = Field(..., description="Number of unique addresses in the flow")
    unique_chains: int = Field(..., description="Number of unique chains in the flow")
    average_transaction_value_usd: float = Field(..., description="Average transaction value in USD")
    max_transaction_value_usd: float = Field(..., description="Maximum transaction value in USD")
    time_span_hours: float = Field(..., description="Time span of the flow in hours")
    transaction_density: float = Field(..., description="Transactions per hour")
    value_density_usd: float = Field(..., description="USD value per hour")
    graph_density: float = Field(..., description="Graph density (0-1)")
    average_path_length: Optional[float] = Field(None, description="Average shortest path length")
    diameter: Optional[int] = Field(None, description="Graph diameter (longest shortest path)")
    clustering_coefficient: Optional[float] = Field(None, description="Average clustering coefficient")


class TransactionFlowInput(BaseModel):
    """Input parameters for transaction flow analysis."""
    
    wallet_addresses: List[str] = Field(
        ..., 
        description="List of wallet addresses to analyze",
        min_items=1,
        max_items=MAX_WALLETS
    )
    time_window_hours: int = Field(
        TIME_WINDOW_HOURS,
        description=f"Time window for analysis in hours (default: {TIME_WINDOW_HOURS})"
    )
    max_depth: int = Field(
        MAX_FLOW_DEPTH,
        description=f"Maximum depth for transaction flow analysis (default: {MAX_FLOW_DEPTH})"
    )
    value_threshold_usd: float = Field(
        FLOW_VALUE_THRESHOLD,
        description=f"Minimum USD value for significant flows (default: ${FLOW_VALUE_THRESHOLD})"
    )
    detect_patterns: bool = Field(
        True,
        description="Whether to detect suspicious patterns in the flow"
    )
    include_contracts: bool = Field(
        True,
        description="Whether to include contract interactions in the flow"
    )
    chain_ids: Optional[str] = Field(
        None,
        description="Optional comma-separated list of chain IDs to analyze, or 'all' for all chains"
    )
    emit_graph_events: bool = Field(
        True,
        description="Whether to emit graph events for Neo4j integration"
    )
    batch_mode: bool = Field(
        False,
        description="Whether to run in batch mode (higher throughput, less real-time)"
    )


class TransactionFlowOutput(BaseModel):
    """Output from transaction flow analysis."""
    
    nodes: List[TransactionNode] = Field(..., description="Nodes in the transaction flow")
    edges: List[TransactionEdge] = Field(..., description="Edges in the transaction flow")
    patterns: List[FlowPattern] = Field(..., description="Detected patterns in the flow")
    metrics: FlowMetrics = Field(..., description="Flow metrics")
    risk_score: float = Field(..., description="Overall risk score for the flow (0-100)")
    risk_factors: List[str] = Field(..., description="Risk factors contributing to the risk score")
    start_time: str = Field(..., description="Start time of the analysis")
    end_time: str = Field(..., description="End time of the analysis")
    wallet_addresses: List[str] = Field(..., description="Wallet addresses analyzed")
    error: Optional[str] = Field(None, description="Error message if analysis failed")


class TransactionFlowTool(BaseTool):
    """
    Tool for analyzing transaction flows between wallets.
    
    This tool fetches transaction activity for multiple wallet addresses,
    builds transaction flow networks, detects suspicious patterns, and
    provides flow metrics and risk scoring. It integrates with the graph
    database for persistence and supports both real-time and batch processing.
    """
    
    name: str = "transaction_flow_tool"
    description: str = """
    Analyzes transaction flows between wallets to detect patterns, visualize money movement,
    and identify potentially suspicious activities. Supports both single wallet and multi-wallet
    network analysis, with integration to the graph database for persistence.
    
    Useful for financial crime investigation, AML compliance, and on-chain forensics.
    """
    
    sim_client: SimClient
    _flow_graphs: Dict[str, nx.DiGraph] = {}  # Cache for flow graphs
    
    def __init__(self, sim_client: Optional[SimClient] = None):
        """
        Initialize the TransactionFlowTool.
        
        Args:
            sim_client: Optional SimClient instance. If not provided, a new one will be created.
        """
        super().__init__()
        self.sim_client = sim_client or SimClient()
    
    async def _execute(
        self,
        wallet_addresses: List[str],
        time_window_hours: int = TIME_WINDOW_HOURS,
        max_depth: int = MAX_FLOW_DEPTH,
        value_threshold_usd: float = FLOW_VALUE_THRESHOLD,
        detect_patterns: bool = True,
        include_contracts: bool = True,
        chain_ids: Optional[str] = None,
        emit_graph_events: bool = True,
        batch_mode: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute transaction flow analysis.
        
        Args:
            wallet_addresses: List of wallet addresses to analyze
            time_window_hours: Time window for analysis in hours
            max_depth: Maximum depth for transaction flow analysis
            value_threshold_usd: Minimum USD value for significant flows
            detect_patterns: Whether to detect suspicious patterns
            include_contracts: Whether to include contract interactions
            chain_ids: Optional comma-separated list of chain IDs
            emit_graph_events: Whether to emit graph events for Neo4j
            batch_mode: Whether to run in batch mode
            
        Returns:
            Dictionary containing transaction flow analysis results
            
        Raises:
            Exception: If there's an error during analysis
        """
        try:
            # Record tool usage for metrics
            record_tool_usage(self.name)
            
            # Validate input
            if not wallet_addresses:
                return {
                    "error": "No wallet addresses provided",
                    "nodes": [],
                    "edges": [],
                    "patterns": [],
                    "metrics": {
                        "total_transactions": 0,
                        "total_value_usd": 0,
                        "unique_addresses": 0,
                        "unique_chains": 0,
                        "average_transaction_value_usd": 0,
                        "max_transaction_value_usd": 0,
                        "time_span_hours": 0,
                        "transaction_density": 0,
                        "value_density_usd": 0,
                        "graph_density": 0
                    },
                    "risk_score": 0,
                    "risk_factors": [],
                    "start_time": datetime.now().isoformat(),
                    "end_time": datetime.now().isoformat(),
                    "wallet_addresses": wallet_addresses
                }
            
            # Limit the number of wallets to analyze
            if len(wallet_addresses) > MAX_WALLETS:
                wallet_addresses = wallet_addresses[:MAX_WALLETS]
                logger.warning(f"Limited analysis to {MAX_WALLETS} wallets")
            
            # Record start time
            start_time = datetime.now().isoformat()
            
            # Build the transaction flow graph
            flow_graph, transactions = await self._build_transaction_flow_graph(
                wallet_addresses=wallet_addresses,
                time_window_hours=time_window_hours,
                max_depth=max_depth,
                value_threshold_usd=value_threshold_usd,
                include_contracts=include_contracts,
                chain_ids=chain_ids,
                batch_mode=batch_mode
            )
            
            # Extract nodes and edges from the graph
            nodes = self._extract_nodes_from_graph(flow_graph)
            edges = self._extract_edges_from_graph(flow_graph, transactions)
            
            # Detect patterns if requested
            patterns = []
            if detect_patterns and flow_graph:
                patterns = self._detect_flow_patterns(flow_graph, transactions)
            
            # Calculate flow metrics
            metrics = self._calculate_flow_metrics(flow_graph, transactions)
            
            # Calculate risk score
            risk_score, risk_factors = self._calculate_risk_score(flow_graph, patterns, metrics)
            
            # Record end time
            end_time = datetime.now().isoformat()
            
            # Prepare result
            result = {
                "nodes": [node.dict() for node in nodes],
                "edges": [edge.dict() for edge in edges],
                "patterns": [pattern.dict() for pattern in patterns],
                "metrics": metrics.dict(),
                "risk_score": risk_score,
                "risk_factors": risk_factors,
                "start_time": start_time,
                "end_time": end_time,
                "wallet_addresses": wallet_addresses
            }
            
            # Emit graph events for Neo4j integration
            if emit_graph_events:
                self._emit_graph_events(result)
            
            return result
            
        except Exception as e:
            error_msg = f"Error in transaction flow analysis: {str(e)}"
            logger.error(error_msg, exc_info=True)
            record_tool_error(self.name, str(e))
            
            return {
                "error": error_msg,
                "nodes": [],
                "edges": [],
                "patterns": [],
                "metrics": {
                    "total_transactions": 0,
                    "total_value_usd": 0,
                    "unique_addresses": 0,
                    "unique_chains": 0,
                    "average_transaction_value_usd": 0,
                    "max_transaction_value_usd": 0,
                    "time_span_hours": 0,
                    "transaction_density": 0,
                    "value_density_usd": 0,
                    "graph_density": 0
                },
                "risk_score": 0,
                "risk_factors": [],
                "start_time": start_time if 'start_time' in locals() else datetime.now().isoformat(),
                "end_time": datetime.now().isoformat(),
                "wallet_addresses": wallet_addresses
            }
    
    async def _build_transaction_flow_graph(
        self,
        wallet_addresses: List[str],
        time_window_hours: int,
        max_depth: int,
        value_threshold_usd: float,
        include_contracts: bool,
        chain_ids: Optional[str],
        batch_mode: bool
    ) -> Tuple[nx.DiGraph, List[Dict[str, Any]]]:
        """
        Build a directed graph representing transaction flows between wallets.
        
        Args:
            wallet_addresses: List of wallet addresses to analyze
            time_window_hours: Time window for analysis in hours
            max_depth: Maximum depth for transaction flow analysis
            value_threshold_usd: Minimum USD value for significant flows
            include_contracts: Whether to include contract interactions
            chain_ids: Optional comma-separated list of chain IDs
            batch_mode: Whether to run in batch mode
            
        Returns:
            Tuple of (DiGraph, list of transactions)
        """
        # Initialize the directed graph
        G = nx.DiGraph()
        
        # Initialize transactions list and processed addresses set
        all_transactions = []
        processed_addresses = set()
        addresses_to_process = set(wallet_addresses)
        current_depth = 0
        
        # Process addresses up to max_depth
        while addresses_to_process and current_depth <= max_depth:
            # Get the next batch of addresses to process
            current_batch = list(addresses_to_process)
            processed_addresses.update(current_batch)
            addresses_to_process = set()
            
            # Process each address in the current batch
            for address in current_batch:
                # Skip if already in cache (for batch mode)
                cache_key = f"{address}_{time_window_hours}_{value_threshold_usd}_{chain_ids}"
                if batch_mode and cache_key in self._flow_graphs:
                    # Merge cached graph into the main graph
                    cached_graph = self._flow_graphs[cache_key]
                    G.add_nodes_from(cached_graph.nodes(data=True))
                    G.add_edges_from(cached_graph.edges(data=True))
                    continue
                
                # Fetch activity for this address
                try:
                    activity_data = await self.sim_client.get_activity(
                        address,
                        limit=100  # Get up to 100 recent activities
                    )
                    
                    activities = activity_data.get("activity", [])
                    
                    # Filter activities by time window and value threshold
                    cutoff_time = datetime.now() - timedelta(hours=time_window_hours)
                    cutoff_timestamp = cutoff_time.timestamp()
                    
                    filtered_activities = []
                    for activity in activities:
                        # Skip if outside time window
                        if "block_time" not in activity or activity["block_time"] < cutoff_timestamp:
                            continue
                        
                        # Skip if below value threshold
                        value_usd = float(activity.get("value_usd", 0))
                        if value_usd < value_threshold_usd:
                            continue
                        
                        # Skip contract interactions if not included
                        if not include_contracts and activity.get("type") == "call":
                            continue
                        
                        filtered_activities.append(activity)
                    
                    # Process each filtered activity
                    for activity in filtered_activities:
                        # Extract transaction details
                        tx_hash = activity.get("transaction_hash", "unknown")
                        from_address = activity.get("from", address)
                        to_address = activity.get("to", "unknown")
                        value_usd = float(activity.get("value_usd", 0))
                        timestamp = datetime.fromtimestamp(activity["block_time"]).isoformat()
                        chain = activity.get("chain", "unknown")
                        token_address = activity.get("address", None)
                        token_symbol = activity.get("symbol", None)
                        tx_type = activity.get("type", "UNKNOWN").upper()
                        block_number = activity.get("block_number")
                        gas_used = activity.get("gas_used")
                        
                        # Create nodes if they don't exist
                        if from_address not in G:
                            G.add_node(from_address, 
                                       type="wallet", 
                                       first_seen=timestamp, 
                                       last_seen=timestamp,
                                       total_in_value_usd=0,
                                       total_out_value_usd=0,
                                       transaction_count=0,
                                       chains=set())
                        
                        if to_address not in G:
                            G.add_node(to_address, 
                                       type="wallet", 
                                       first_seen=timestamp, 
                                       last_seen=timestamp,
                                       total_in_value_usd=0,
                                       total_out_value_usd=0,
                                       transaction_count=0,
                                       chains=set())
                        
                        # Update node attributes
                        G.nodes[from_address]["last_seen"] = max(G.nodes[from_address]["last_seen"], timestamp)
                        G.nodes[to_address]["last_seen"] = max(G.nodes[to_address]["last_seen"], timestamp)
                        
                        G.nodes[from_address]["total_out_value_usd"] += value_usd
                        G.nodes[to_address]["total_in_value_usd"] += value_usd
                        
                        G.nodes[from_address]["transaction_count"] += 1
                        G.nodes[to_address]["transaction_count"] += 1
                        
                        G.nodes[from_address]["chains"] = G.nodes[from_address].get("chains", set())
                        G.nodes[from_address]["chains"].add(chain)
                        G.nodes[to_address]["chains"] = G.nodes[to_address].get("chains", set())
                        G.nodes[to_address]["chains"].add(chain)
                        
                        # Add edge for the transaction
                        edge_id = f"{tx_hash}_{from_address}_{to_address}"
                        G.add_edge(from_address, to_address, 
                                  transaction_hash=tx_hash,
                                  value_usd=value_usd,
                                  timestamp=timestamp,
                                  chain=chain,
                                  token_address=token_address,
                                  token_symbol=token_symbol,
                                  transaction_type=tx_type,
                                  block_number=block_number,
                                  gas_used=gas_used,
                                  edge_id=edge_id)
                        
                        # Add transaction to the list
                        transaction = {
                            "transaction_hash": tx_hash,
                            "from_address": from_address,
                            "to_address": to_address,
                            "value_usd": value_usd,
                            "timestamp": timestamp,
                            "chain": chain,
                            "token_address": token_address,
                            "token_symbol": token_symbol,
                            "transaction_type": tx_type,
                            "block_number": block_number,
                            "gas_used": gas_used,
                            "edge_id": edge_id
                        }
                        all_transactions.append(transaction)
                        
                        # Add counterparties to addresses to process for next depth
                        if current_depth < max_depth:
                            if from_address != address and from_address not in processed_addresses:
                                addresses_to_process.add(from_address)
                            if to_address != address and to_address not in processed_addresses:
                                addresses_to_process.add(to_address)
                    
                    # Cache this address's subgraph for batch mode
                    if batch_mode:
                        subgraph = G.subgraph([address] + list(G.predecessors(address)) + list(G.successors(address))).copy()
                        self._flow_graphs[cache_key] = subgraph
                    
                except Exception as e:
                    logger.error(f"Error fetching activity for {address}: {str(e)}")
                    continue
            
            # Increment depth
            current_depth += 1
        
        # Calculate net flow for each node
        for node in G.nodes():
            in_value = G.nodes[node].get("total_in_value_usd", 0)
            out_value = G.nodes[node].get("total_out_value_usd", 0)
            G.nodes[node]["net_flow_usd"] = in_value - out_value
            
            # Convert chain sets to lists for serialization
            G.nodes[node]["chains"] = list(G.nodes[node].get("chains", set()))
        
        return G, all_transactions
    
    def _extract_nodes_from_graph(self, G: nx.DiGraph) -> List[TransactionNode]:
        """
        Extract node models from the graph.
        
        Args:
            G: NetworkX DiGraph
            
        Returns:
            List of TransactionNode models
        """
        nodes = []
        for address, attrs in G.nodes(data=True):
            # Calculate node-level risk score
            risk_score = self._calculate_node_risk_score(G, address)
            
            # Create node model
            node = TransactionNode(
                address=address,
                type=attrs.get("type", "wallet"),
                first_seen=attrs.get("first_seen", ""),
                last_seen=attrs.get("last_seen", ""),
                total_in_value_usd=attrs.get("total_in_value_usd", 0),
                total_out_value_usd=attrs.get("total_out_value_usd", 0),
                net_flow_usd=attrs.get("net_flow_usd", 0),
                transaction_count=attrs.get("transaction_count", 0),
                chains=attrs.get("chains", []),
                risk_score=risk_score,
                labels=[]  # Will be populated later if needed
            )
            
            # Add labels based on node characteristics
            if node.net_flow_usd > 0 and node.net_flow_usd > 0.8 * node.total_in_value_usd:
                node.labels.append("ACCUMULATOR")
            elif node.net_flow_usd < 0 and abs(node.net_flow_usd) > 0.8 * node.total_out_value_usd:
                node.labels.append("SOURCE")
            
            if len(node.chains) > 1:
                node.labels.append("MULTI_CHAIN")
            
            # Add to nodes list
            nodes.append(node)
        
        return nodes
    
    def _extract_edges_from_graph(
        self, 
        G: nx.DiGraph, 
        transactions: List[Dict[str, Any]]
    ) -> List[TransactionEdge]:
        """
        Extract edge models from the graph.
        
        Args:
            G: NetworkX DiGraph
            transactions: List of transaction dictionaries
            
        Returns:
            List of TransactionEdge models
        """
        edges = []
        for tx in transactions:
            edge = TransactionEdge(
                source=tx["from_address"],
                target=tx["to_address"],
                transaction_hash=tx["transaction_hash"],
                value_usd=tx["value_usd"],
                timestamp=tx["timestamp"],
                chain=tx["chain"],
                token_address=tx["token_address"],
                token_symbol=tx["token_symbol"],
                transaction_type=tx["transaction_type"],
                block_number=tx["block_number"],
                gas_used=tx["gas_used"],
                edge_id=tx["edge_id"]
            )
            edges.append(edge)
        
        return edges
    
    def _detect_flow_patterns(
        self, 
        G: nx.DiGraph, 
        transactions: List[Dict[str, Any]]
    ) -> List[FlowPattern]:
        """
        Detect suspicious patterns in the transaction flow.
        
        Args:
            G: NetworkX DiGraph
            transactions: List of transaction dictionaries
            
        Returns:
            List of FlowPattern models
        """
        patterns = []
        
        # 1. Detect peel chains (sequential transfers with decreasing values)
        peel_chains = self._detect_peel_chains(G, transactions)
        patterns.extend(peel_chains)
        
        # 2. Detect circular flows (cycles in the graph)
        circular_flows = self._detect_circular_flows(G)
        patterns.extend(circular_flows)
        
        # 3. Detect layering (multiple hops to obscure source/destination)
        layering_patterns = self._detect_layering(G, transactions)
        patterns.extend(layering_patterns)
        
        return patterns
    
    def _detect_peel_chains(
        self, 
        G: nx.DiGraph, 
        transactions: List[Dict[str, Any]]
    ) -> List[FlowPattern]:
        """
        Detect peel chain patterns (sequential transfers with decreasing values).
        
        Args:
            G: NetworkX DiGraph
            transactions: List of transaction dictionaries
            
        Returns:
            List of FlowPattern models
        """
        import uuid
        
        # Sort transactions by timestamp
        sorted_txs = sorted(transactions, key=lambda x: x["timestamp"])
        
        # Group transactions by chain
        chain_txs = {}
        for tx in sorted_txs:
            chain = tx["chain"]
            if chain not in chain_txs:
                chain_txs[chain] = []
            chain_txs[chain].append(tx)
        
        patterns = []
        
        # Process each chain separately
        for chain, chain_transactions in chain_txs.items():
            # Build a directed graph of transactions by timestamp
            chain_graph = nx.DiGraph()
            
            # Add all transactions as edges
            for tx in chain_transactions:
                from_addr = tx["from_address"]
                to_addr = tx["to_address"]
                chain_graph.add_edge(from_addr, to_addr, **tx)
            
            # Find all simple paths of length >= PEEL_CHAIN_THRESHOLD
            for source in chain_graph.nodes():
                for target in chain_graph.nodes():
                    if source == target:
                        continue
                    
                    # Use a generator to avoid memory issues with all_simple_paths
                    simple_paths = nx.all_simple_paths(
                        chain_graph, 
                        source, 
                        target, 
                        cutoff=MAX_FLOW_DEPTH
                    )
                    
                    for path in simple_paths:
                        if len(path) < PEEL_CHAIN_THRESHOLD:
                            continue
                        
                        # Check if this is a peel chain (decreasing values)
                        path_edges = []
                        path_txs = []
                        decreasing = True
                        prev_value = float('inf')
                        
                        for i in range(len(path) - 1):
                            u, v = path[i], path[i + 1]
                            if chain_graph.has_edge(u, v):
                                edge_data = chain_graph.get_edge_data(u, v)
                                # There might be multiple edges, get the one with the right transaction hash
                                for key, data in edge_data.items():
                                    if isinstance(data, dict) and "value_usd" in data:
                                        path_edges.append((u, v, data))
                                        path_txs.append(data)
                                        
                                        # Check if value is decreasing
                                        current_value = data["value_usd"]
                                        if current_value > prev_value * PEEL_CHAIN_DECAY_RATE:
                                            decreasing = False
                                            break
                                        
                                        prev_value = current_value
                        
                        # If we have a valid peel chain
                        if decreasing and len(path_edges) >= PEEL_CHAIN_THRESHOLD - 1:
                            # Calculate total value
                            total_value = sum(tx["value_usd"] for tx in path_txs)
                            
                            # Get timestamps
                            timestamps = [tx["timestamp"] for tx in path_txs]
                            start_time = min(timestamps)
                            end_time = max(timestamps)
                            
                            # Calculate confidence based on path length and value decay
                            confidence = min(0.95, 0.5 + (len(path) - PEEL_CHAIN_THRESHOLD) * 0.1)
                            
                            # Calculate risk score
                            risk_score = min(100, 50 + (len(path) - PEEL_CHAIN_THRESHOLD) * 10)
                            
                            # Create pattern
                            pattern = FlowPattern(
                                pattern_id=str(uuid.uuid4()),
                                pattern_type="PEEL_CHAIN",
                                addresses=path,
                                transactions=[tx["transaction_hash"] for tx in path_txs],
                                start_time=start_time,
                                end_time=end_time,
                                total_value_usd=total_value,
                                confidence=confidence,
                                description=f"Peel chain of {len(path)} wallets with decreasing values",
                                risk_score=risk_score,
                                path=path
                            )
                            
                            patterns.append(pattern)
        
        return patterns
    
    def _detect_circular_flows(self, G: nx.DiGraph) -> List[FlowPattern]:
        """
        Detect circular flow patterns (cycles in the graph).
        
        Args:
            G: NetworkX DiGraph
            
        Returns:
            List of FlowPattern models
        """
        import uuid
        
        patterns = []
        
        # Find simple cycles in the graph
        try:
            cycles = list(nx.simple_cycles(G))
        except:
            # Fallback if simple_cycles fails (e.g., for large graphs)
            cycles = []
            for node in G.nodes():
                try:
                    for cycle in nx.find_cycle(G, node):
                        cycles.append([node for node, _ in cycle])
                except:
                    continue
        
        # Filter cycles by length
        filtered_cycles = [
            cycle for cycle in cycles 
            if SUSPICIOUS_CYCLE_MIN_LENGTH <= len(cycle) <= SUSPICIOUS_CYCLE_MAX_LENGTH
        ]
        
        # Process each cycle
        for cycle in filtered_cycles:
            # Get transactions in the cycle
            cycle_txs = []
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i + 1) % len(cycle)]
                if G.has_edge(u, v):
                    edge_data = G.get_edge_data(u, v)
                    for key, data in edge_data.items():
                        if isinstance(data, dict):
                            cycle_txs.append(data)
            
            if not cycle_txs:
                continue
            
            # Calculate total value
            total_value = sum(tx.get("value_usd", 0) for tx in cycle_txs)
            
            # Get timestamps
            timestamps = [tx.get("timestamp", "") for tx in cycle_txs if "timestamp" in tx]
            if not timestamps:
                continue
                
            start_time = min(timestamps)
            end_time = max(timestamps)
            
            # Calculate time span in hours
            try:
                start_dt = datetime.fromisoformat(start_time)
                end_dt = datetime.fromisoformat(end_time)
                time_span_hours = (end_dt - start_dt).total_seconds() / 3600
            except:
                time_span_hours = 24  # Default if parsing fails
            
            # Calculate confidence based on cycle length and time span
            # Higher confidence for shorter time spans and longer cycles
            time_factor = max(0, 1 - (time_span_hours / 24))  # 1.0 for instant, 0 for 24+ hours
            length_factor = min(1, (len(cycle) - 2) / 5)  # 0 for length 3, 1.0 for length 8+
            confidence = 0.5 + (time_factor * 0.25) + (length_factor * 0.25)
            
            # Calculate risk score
            risk_score = min(100, 60 + (time_factor * 20) + (length_factor * 20))
            
            # Create pattern
            pattern = FlowPattern(
                pattern_id=str(uuid.uuid4()),
                pattern_type="CIRCULAR_FLOW",
                addresses=cycle,
                transactions=[tx.get("transaction_hash", "unknown") for tx in cycle_txs],
                start_time=start_time,
                end_time=end_time,
                total_value_usd=total_value,
                confidence=confidence,
                description=f"Circular flow through {len(cycle)} wallets in {time_span_hours:.1f} hours",
                risk_score=risk_score,
                path=cycle + [cycle[0]]  # Add first node at end to complete the cycle
            )
            
            patterns.append(pattern)
        
        return patterns
    
    def _detect_layering(
        self, 
        G: nx.DiGraph, 
        transactions: List[Dict[str, Any]]
    ) -> List[FlowPattern]:
        """
        Detect layering patterns (multiple hops to obscure source/destination).
        
        Args:
            G: NetworkX DiGraph
            transactions: List of transaction dictionaries
            
        Returns:
            List of FlowPattern models
        """
        import uuid
        
        patterns = []
        
        # Sort transactions by timestamp
        sorted_txs = sorted(transactions, key=lambda x: x["timestamp"])
        
        # Find nodes with high in-degree and out-degree (potential layering nodes)
        potential_layering_nodes = []
        for node in G.nodes():
            in_degree = G.in_degree(node)
            out_degree = G.out_degree(node)
            
            # Potential layering nodes have multiple inputs and outputs
            if in_degree >= 2 and out_degree >= 2:
                potential_layering_nodes.append(node)
        
        # Process each potential layering node
        for node in potential_layering_nodes:
            # Get incoming and outgoing transactions
            incoming_edges = list(G.in_edges(node, data=True))
            outgoing_edges = list(G.out_edges(node, data=True))
            
            if len(incoming_edges) < 2 or len(outgoing_edges) < 2:
                continue
            
            # Get timestamps for incoming and outgoing transactions
            in_timestamps = [
                datetime.fromisoformat(edge[2].get("timestamp", "2000-01-01T00:00:00"))
                for edge in incoming_edges if "timestamp" in edge[2]
            ]
            
            out_timestamps = [
                datetime.fromisoformat(edge[2].get("timestamp", "2000-01-01T00:00:00"))
                for edge in outgoing_edges if "timestamp" in edge[2]
            ]
            
            if not in_timestamps or not out_timestamps:
                continue
            
            # Calculate time windows
            in_start = min(in_timestamps)
            in_end = max(in_timestamps)
            out_start = min(out_timestamps)
            out_end = max(out_timestamps)
            
            # Check if this looks like layering
            # Layering typically has funds coming in, then going out
            if in_end <= out_start:
                # This is a perfect layering pattern (all inputs before any outputs)
                layering_confidence = 0.9
            else:
                # There's some overlap, calculate confidence based on overlap
                total_span = max((out_end - in_start).total_seconds(), 1)
                overlap = min(in_end, out_end) - max(in_start, out_start)
                overlap_seconds = max(overlap.total_seconds(), 0)
                overlap_ratio = overlap_seconds / total_span
                
                layering_confidence = 0.7 * (1 - overlap_ratio)
            
            # Only consider as layering if confidence is high enough
            if layering_confidence < 0.5:
                continue
            
            # Calculate total values
            in_value = sum(edge[2].get("value_usd", 0) for edge in incoming_edges)
            out_value = sum(edge[2].get("value_usd", 0) for edge in outgoing_edges)
            
            # Check if values are roughly balanced (typical for layering)
            value_ratio = min(in_value, out_value) / max(in_value, out_value) if max(in_value, out_value) > 0 else 0
            if value_ratio < 0.7:  # Less than 70% of funds moved through
                continue
            
            # Get all addresses and transactions involved
            addresses = set([node])
            transactions = []
            
            for u, v, data in incoming_edges:
                addresses.add(u)
                if "transaction_hash" in data:
                    transactions.append(data["transaction_hash"])
            
            for u, v, data in outgoing_edges:
                addresses.add(v)
                if "transaction_hash" in data:
                    transactions.append(data["transaction_hash"])
            
            # Calculate risk score
            risk_score = min(100, 50 + (layering_confidence * 30) + (value_ratio * 20))
            
            # Create pattern
            pattern = FlowPattern(
                pattern_id=str(uuid.uuid4()),
                pattern_type="LAYERING",
                addresses=list(addresses),
                transactions=transactions,
                start_time=in_start.isoformat(),
                end_time=out_end.isoformat(),
                total_value_usd=max(in_value, out_value),
                confidence=layering_confidence,
                description=f"Layering through {node} with {len(incoming_edges)} inputs and {len(outgoing_edges)} outputs",
                risk_score=risk_score,
                path=None  # No specific path for layering
            )
            
            patterns.append(pattern)
        
        return patterns
    
    def _calculate_flow_metrics(
        self, 
        G: nx.DiGraph, 
        transactions: List[Dict[str, Any]]
    ) -> FlowMetrics:
        """
        Calculate metrics for the transaction flow.
        
        Args:
            G: NetworkX DiGraph
            transactions: List of transaction dictionaries
            
        Returns:
            FlowMetrics model
        """
        if not G or not transactions:
            return FlowMetrics(
                total_transactions=0,
                total_value_usd=0,
                unique_addresses=0,
                unique_chains=0,
                average_transaction_value_usd=0,
                max_transaction_value_usd=0,
                time_span_hours=0,
                transaction_density=0,
                value_density_usd=0,
                graph_density=0
            )
        
        # Basic metrics
        total_transactions = len(transactions)
        total_value_usd = sum(tx["value_usd"] for tx in transactions)
        unique_addresses = G.number_of_nodes()
        
        # Get unique chains
        unique_chains = set()
        for tx in transactions:
            if "chain" in tx:
                unique_chains.add(tx["chain"])
        
        # Transaction value metrics
        if total_transactions > 0:
            average_transaction_value_usd = total_value_usd / total_transactions
            max_transaction_value_usd = max(tx["value_usd"] for tx in transactions)
        else:
            average_transaction_value_usd = 0
            max_transaction_value_usd = 0
        
        # Time span metrics
        timestamps = [
            datetime.fromisoformat(tx["timestamp"]) 
            for tx in transactions if "timestamp" in tx
        ]
        
        if timestamps:
            min_time = min(timestamps)
            max_time = max(timestamps)
            time_span_seconds = (max_time - min_time).total_seconds()
            time_span_hours = time_span_seconds / 3600
            
            # Density metrics
            if time_span_hours > 0:
                transaction_density = total_transactions / time_span_hours
                value_density_usd = total_value_usd / time_span_hours
            else:
                transaction_density = total_transactions
                value_density_usd = total_value_usd
        else:
            time_span_hours = 0
            transaction_density = 0
            value_density_usd = 0
        
        # Graph metrics
        graph_density = nx.density(G)
        
        # Advanced graph metrics (with error handling for disconnected graphs)
        try:
            average_path_length = nx.average_shortest_path_length(G)
        except:
            average_path_length = None
        
        try:
            diameter = nx.diameter(G)
        except:
            diameter = None
        
        try:
            clustering_coefficient = nx.average_clustering(G)
        except:
            clustering_coefficient = None
        
        return FlowMetrics(
            total_transactions=total_transactions,
            total_value_usd=total_value_usd,
            unique_addresses=unique_addresses,
            unique_chains=len(unique_chains),
            average_transaction_value_usd=average_transaction_value_usd,
            max_transaction_value_usd=max_transaction_value_usd,
            time_span_hours=time_span_hours,
            transaction_density=transaction_density,
            value_density_usd=value_density_usd,
            graph_density=graph_density,
            average_path_length=average_path_length,
            diameter=diameter,
            clustering_coefficient=clustering_coefficient
        )
    
    def _calculate_risk_score(
        self, 
        G: nx.DiGraph, 
        patterns: List[FlowPattern], 
        metrics: FlowMetrics
    ) -> Tuple[float, List[str]]:
        """
        Calculate overall risk score for the transaction flow.
        
        Args:
            G: NetworkX DiGraph
            patterns: List of detected patterns
            metrics: Flow metrics
            
        Returns:
            Tuple of (risk_score, risk_factors)
        """
        risk_score = 0
        risk_factors = []
        
        # Base risk from patterns
        if patterns:
            pattern_risk = max(pattern.risk_score for pattern in patterns)
            risk_score += pattern_risk * 0.5  # Patterns contribute 50% of risk
            
            # Add risk factors for patterns
            for pattern in patterns:
                if pattern.risk_score >= 70:
                    risk_factors.append(
                        f"High-risk {pattern.pattern_type} pattern detected "
                        f"({pattern.confidence:.0%} confidence)"
                    )
                elif pattern.risk_score >= 50:
                    risk_factors.append(
                        f"Medium-risk {pattern.pattern_type} pattern detected "
                        f"({pattern.confidence:.0%} confidence)"
                    )
        
        # Risk from graph structure
        if G and G.number_of_nodes() > 0:
            # Check for high-value transfers
            high_value_edges = [
                (u, v, data["value_usd"]) 
                for u, v, data in G.edges(data=True) 
                if "value_usd" in data and data["value_usd"] >= FLOW_VALUE_THRESHOLD * 10
            ]
            
            if high_value_edges:
                risk_score += min(25, len(high_value_edges) * 5)
                risk_factors.append(
                    f"Large transfers detected ({len(high_value_edges)} transfers over ${FLOW_VALUE_THRESHOLD * 10:,.0f})"
                )
            
            # Check for complex flow structure
            if metrics.graph_density is not None and metrics.graph_density < 0.1 and G.number_of_nodes() >= 10:
                risk_score += 10
                risk_factors.append("Complex flow structure with low graph density")
            
            # Check for multi-chain activity
            chains_per_node = [len(data.get("chains", [])) for _, data in G.nodes(data=True)]
            multi_chain_nodes = sum(1 for count in chains_per_node if count > 1)
            
            if multi_chain_nodes >= 3:
                risk_score += 15
                risk_factors.append(f"Multi-chain activity detected ({multi_chain_nodes} wallets)")
            
            # Check for high transaction velocity
            if metrics.transaction_density >= 10:  # More than 10 transactions per hour
                risk_score += 15
                risk_factors.append(f"High transaction velocity ({metrics.transaction_density:.1f} tx/hour)")
        
        # Ensure risk score is between 0 and 100
        risk_score = min(100, max(0, risk_score))
        
        # Add default risk factor if none found
        if not risk_factors:
            if risk_score >= 70:
                risk_factors.append("Multiple high-risk flow characteristics detected")
            elif risk_score >= 40:
                risk_factors.append("Some suspicious flow characteristics detected")
            else:
                risk_factors.append("No significant risk factors detected")
        
        return risk_score, risk_factors
    
    def _calculate_node_risk_score(self, G: nx.DiGraph, node: str) -> float:
        """
        Calculate risk score for a specific node in the graph.
        
        Args:
            G: NetworkX DiGraph
            node: Node address
            
        Returns:
            Risk score (0-100)
        """
        if not G or node not in G:
            return 0
        
        risk_score = 0
        
        # Check in-degree and out-degree
        in_degree = G.in_degree(node)
        out_degree = G.out_degree(node)
        
        # High fan-in or fan-out is suspicious
        if in_degree >= 5:
            risk_score += min(20, in_degree)
        
        if out_degree >= 5:
            risk_score += min(20, out_degree)
        
        # Check multi-chain activity
        chains = G.nodes[node].get("chains", [])
        if len(chains) > 1:
            risk_score += len(chains) * 5
        
        # Check transaction value
        in_value = G.nodes[node].get("total_in_value_usd", 0)
        out_value = G.nodes[node].get("total_out_value_usd", 0)
        
        # High value is higher risk
        total_value = in_value + out_value
        if total_value > FLOW_VALUE_THRESHOLD * 100:
            risk_score += 25
        elif total_value > FLOW_VALUE_THRESHOLD * 10:
            risk_score += 15
        elif total_value > FLOW_VALUE_THRESHOLD:
            risk_score += 5
        
        # Check balance of in/out (close balance is suspicious)
        if max(in_value, out_value) > 0:
            balance_ratio = min(in_value, out_value) / max(in_value, out_value)
            if balance_ratio > 0.9:  # Very balanced
                risk_score += 20
            elif balance_ratio > 0.7:  # Somewhat balanced
                risk_score += 10
        
        # Ensure risk score is between 0 and 100
        risk_score = min(100, max(0, risk_score))
        
        return risk_score
    
    def _emit_graph_events(self, result: Dict[str, Any]) -> None:
        """
        Emit graph events for Neo4j integration.
        
        Args:
            result: Transaction flow analysis result
        """
        try:
            # Prepare data for Neo4j
            graph_data = {
                "nodes": result["nodes"],
                "edges": result["edges"],
                "patterns": result["patterns"],
                "metrics": result["metrics"],
                "risk_score": result["risk_score"],
                "risk_factors": result["risk_factors"],
                "timestamp": int(time.time())
            }
            
            # Emit event for graph processing
            emit_event(
                GraphAddEvent(
                    type="transaction_flow",
                    data=graph_data
                )
            )
            
            logger.debug(f"Emitted graph events for transaction flow analysis")
            
        except Exception as e:
            logger.error(f"Failed to emit graph events: {str(e)}")
            # Don't re-raise, as this is a non-critical operation
