"""
Advanced Graph Tool - Enhanced graph algorithms for fraud detection and network analysis

This tool extends the basic GNN capabilities with advanced graph algorithms specifically
designed for financial crime detection:

1. Enhanced Graph Attention Networks (GAT) with multi-head attention
2. Community detection using Neo4j GDS (Louvain, Label Propagation)
3. Risk propagation algorithm that spreads risk scores through the graph
4. Real-time anomaly detection using graph structure changes
5. Centrality-based fraud pattern detection

The tool is designed to work with the CrewAI framework and integrates with
the existing Neo4j graph database and GNN infrastructure.
"""

import os
import json
import logging
import pickle
import time
from typing import Dict, List, Optional, Union, Tuple, Any, Set, Callable
from datetime import datetime
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, HeteroData
from torch_geometric.nn import GATConv, GATv2Conv
from torch_geometric.loader import NeighborLoader
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import remove_self_loops, add_self_loops
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from langchain.tools import BaseTool
from crewai import Tool

from backend.core.logging import get_logger
from backend.core.metrics import REGISTRY, Counter, Histogram
from backend.core.events import EventBus, subscribe
from backend.integrations.neo4j_client import Neo4jClient
from backend.agents.tools.gnn_fraud_detection_tool import GNNModel, GraphDataProcessor
from backend.core.evidence import EvidenceBundle

# Configure logger
logger = get_logger(__name__)

# Prometheus metrics
COMMUNITY_DETECTION_COUNT = Counter(
    "community_detection_total",
    "Total number of community detection operations",
    ["algorithm", "result"]
)
RISK_PROPAGATION_COUNT = Counter(
    "risk_propagation_total", 
    "Total number of risk propagation operations",
    ["status"]
)
ADVANCED_GRAPH_LATENCY = Histogram(
    "advanced_graph_operation_seconds",
    "Latency of advanced graph operations",
    ["operation"]
)

# Constants
MODEL_DIR = Path("models/advanced_gnn")
DEFAULT_HEADS = 4  # Number of attention heads for multi-head GAT
DEFAULT_COMMUNITY_SIZE_THRESHOLD = 3  # Minimum nodes for a valid community
DEFAULT_RISK_DECAY_FACTOR = 0.5  # Risk decay per hop in propagation
DEFAULT_RISK_THRESHOLD = 0.7  # Threshold for high-risk nodes
DEFAULT_MAX_COMMUNITIES = 10  # Maximum communities to return

# Ensure model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)


class CommunityAlgorithm(str, Enum):
    """Supported community detection algorithms"""
    LOUVAIN = "louvain"
    LABEL_PROPAGATION = "labelPropagation"
    CONNECTED_COMPONENTS = "connectedComponents"
    STRONGLY_CONNECTED = "stronglyConnectedComponents"


class AdvancedGATModel(nn.Module):
    """
    Enhanced Graph Attention Network with multi-head attention
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        heads: int = DEFAULT_HEADS,
        dropout: float = 0.2,
        residual: bool = True,
        v2_attention: bool = True
    ):
        """
        Initialize enhanced GAT model with multi-head attention
        
        Args:
            in_channels: Number of input features
            hidden_channels: Number of hidden features
            out_channels: Number of output features
            num_layers: Number of GAT layers
            heads: Number of attention heads
            dropout: Dropout probability
            residual: Whether to use residual connections
            v2_attention: Whether to use GATv2 (more expressive) or original GAT
        """
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.residual = residual
        
        # Select GAT implementation
        conv_layer = GATv2Conv if v2_attention else GATConv
        
        # Input layer
        self.convs = nn.ModuleList()
        self.convs.append(
            conv_layer(
                in_channels, 
                hidden_channels // heads,  # Divide by heads to keep param count similar
                heads=heads,
                dropout=dropout
            )
        )
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(
                conv_layer(
                    hidden_channels,
                    hidden_channels // heads,
                    heads=heads,
                    dropout=dropout
                )
            )
        
        # Output layer (single head for final prediction)
        if num_layers > 1:
            self.convs.append(
                conv_layer(
                    hidden_channels,
                    hidden_channels,
                    heads=1,
                    concat=False,
                    dropout=dropout
                )
            )
        
        # Final MLP for prediction
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels)
        )
        
        # For residual connections
        if residual:
            self.skip_connections = nn.ModuleList()
            self.skip_connections.append(nn.Linear(in_channels, hidden_channels))
            for _ in range(num_layers - 2):
                self.skip_connections.append(nn.Identity())
            if num_layers > 1:
                self.skip_connections.append(nn.Linear(hidden_channels, hidden_channels))
    
    def forward(self, x, edge_index):
        """Forward pass through the enhanced GAT model"""
        # Process through GAT layers
        for i in range(self.num_layers):
            # Apply GAT convolution
            if i == 0 or i == self.num_layers - 1:
                # First and last layers need special handling due to dimension changes
                new_x = self.convs[i](x, edge_index)
            else:
                new_x = self.convs[i](x, edge_index)
            
            # Apply residual connection if enabled
            if self.residual:
                # Skip connection
                skip = self.skip_connections[i](x)
                x = new_x + skip
            else:
                x = new_x
            
            # Apply non-linearity except at final layer
            if i < self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Final prediction
        x = self.mlp(x)
        return x


class RiskPropagator:
    """
    Propagates risk scores through the graph network
    """
    
    def __init__(
        self, 
        neo4j_client: Neo4jClient,
        decay_factor: float = DEFAULT_RISK_DECAY_FACTOR,
        max_hops: int = 3,
        risk_threshold: float = DEFAULT_RISK_THRESHOLD
    ):
        """
        Initialize risk propagator
        
        Args:
            neo4j_client: Neo4j client for database access
            decay_factor: Risk decay per hop (0-1)
            max_hops: Maximum propagation distance
            risk_threshold: Threshold for high-risk nodes
        """
        self.neo4j_client = neo4j_client
        self.decay_factor = decay_factor
        self.max_hops = max_hops
        self.risk_threshold = risk_threshold
    
    def propagate_risk(
        self,
        seed_nodes: List[str],
        node_type: str = "Wallet",
        risk_property: str = "risk_score",
        weight_property: Optional[str] = "amount_usd",
        direction: str = "both"
    ) -> Dict[str, Any]:
        """
        Propagate risk from seed nodes through the graph
        
        Args:
            seed_nodes: List of node IDs to start propagation from
            node_type: Type of nodes to propagate risk to
            risk_property: Node property containing risk score
            weight_property: Edge property for weighted propagation
            direction: Direction of propagation ('both', 'in', 'out')
            
        Returns:
            Dictionary with propagation results
        """
        start_time = time.time()
        
        # Validate inputs
        if not seed_nodes:
            raise ValueError("Must provide seed nodes for risk propagation")
        
        # Direction mapping for Cypher
        direction_map = {
            "both": "",
            "in": "<",
            "out": ">"
        }
        if direction not in direction_map:
            raise ValueError(f"Invalid direction: {direction}. Must be 'both', 'in', or 'out'")
        
        direction_arrow = direction_map[direction]
        
        # Build weight clause if needed
        weight_clause = ""
        if weight_property:
            weight_clause = f", weight: coalesce(r.{weight_property}, 1.0)"
        
        # Cypher query for risk propagation
        query = f"""
        MATCH (seed:{node_type})
        WHERE seed.id IN $seed_nodes
        
        // Initialize algorithm with seed nodes
        WITH collect(seed) AS seeds
        CALL gds.graph.project.cypher(
            'risk_propagation',
            'MATCH (n:{node_type}) RETURN id(n) AS id, n.{risk_property} AS risk, n.id AS node_id',
            'MATCH (n:{node_type}){direction_arrow}-[r]-{direction_arrow}(m:{node_type}) 
             RETURN id(n) AS source, id(m) AS target{weight_clause}'
        )
        YIELD graphName
        
        // Run personalized PageRank from seed nodes
        CALL gds.pageRank.stream('risk_propagation', {{
            sourceNodes: seeds,
            maxIterations: $max_hops,
            dampingFactor: $decay_factor
        }})
        YIELD nodeId, score
        
        // Get node properties
        WITH gds.util.asNode(nodeId) AS node, score
        WHERE score >= $risk_threshold
        
        // Return results
        RETURN 
            node.id AS node_id,
            labels(node) AS labels,
            score AS propagated_risk,
            node.{risk_property} AS original_risk
        ORDER BY score DESC
        LIMIT 1000
        """
        
        # Execute query
        try:
            params = {
                "seed_nodes": seed_nodes,
                "max_hops": self.max_hops,
                "decay_factor": self.decay_factor,
                "risk_threshold": self.risk_threshold
            }
            
            result = self.neo4j_client.query(query, params)
            
            # Clean up the projected graph
            cleanup_query = "CALL gds.graph.drop('risk_propagation', false)"
            self.neo4j_client.query(cleanup_query)
            
            # Process results
            high_risk_nodes = []
            for record in result:
                high_risk_nodes.append({
                    "node_id": record["node_id"],
                    "labels": record["labels"],
                    "propagated_risk": record["propagated_risk"],
                    "original_risk": record["original_risk"],
                    "risk_delta": record["propagated_risk"] - (record["original_risk"] or 0)
                })
            
            # Record metrics
            duration = time.time() - start_time
            ADVANCED_GRAPH_LATENCY.labels(operation="risk_propagation").observe(duration)
            RISK_PROPAGATION_COUNT.labels(status="success").inc()
            
            return {
                "seed_nodes": seed_nodes,
                "high_risk_nodes": high_risk_nodes,
                "total_affected": len(high_risk_nodes),
                "max_propagated_risk": max([n["propagated_risk"] for n in high_risk_nodes]) if high_risk_nodes else 0,
                "execution_time_seconds": duration
            }
            
        except Exception as e:
            logger.error(f"Error in risk propagation: {str(e)}")
            RISK_PROPAGATION_COUNT.labels(status="error").inc()
            raise
    
    def update_risk_scores(
        self,
        propagation_result: Dict[str, Any],
        update_property: str = "propagated_risk_score",
        update_threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Update node properties with propagated risk scores
        
        Args:
            propagation_result: Result from propagate_risk method
            update_property: Property to store propagated risk
            update_threshold: Minimum risk delta to update
            
        Returns:
            Dictionary with update results
        """
        high_risk_nodes = propagation_result.get("high_risk_nodes", [])
        if not high_risk_nodes:
            return {"nodes_updated": 0}
        
        # Filter nodes with significant risk delta
        nodes_to_update = [
            node for node in high_risk_nodes 
            if node.get("risk_delta", 0) >= update_threshold
        ]
        
        if not nodes_to_update:
            return {"nodes_updated": 0}
        
        # Create parameter lists
        node_ids = [node["node_id"] for node in nodes_to_update]
        risk_scores = [node["propagated_risk"] for node in nodes_to_update]
        
        # Update query
        query = f"""
        UNWIND $updates AS update
        MATCH (n) WHERE n.id = update.node_id
        SET n.{update_property} = update.risk,
            n.risk_updated_at = datetime()
        RETURN count(n) AS updated
        """
        
        # Execute update
        try:
            updates = [{"node_id": node_id, "risk": risk} 
                      for node_id, risk in zip(node_ids, risk_scores)]
            
            result = self.neo4j_client.query(query, {"updates": updates})
            nodes_updated = result[0]["updated"] if result else 0
            
            return {
                "nodes_updated": nodes_updated,
                "update_property": update_property
            }
            
        except Exception as e:
            logger.error(f"Error updating risk scores: {str(e)}")
            raise


class CommunityDetector:
    """
    Detects communities in the graph using Neo4j GDS algorithms
    """
    
    def __init__(
        self,
        neo4j_client: Neo4jClient,
        min_community_size: int = DEFAULT_COMMUNITY_SIZE_THRESHOLD,
        max_communities: int = DEFAULT_MAX_COMMUNITIES
    ):
        """
        Initialize community detector
        
        Args:
            neo4j_client: Neo4j client for database access
            min_community_size: Minimum nodes for a valid community
            max_communities: Maximum communities to return
        """
        self.neo4j_client = neo4j_client
        self.min_community_size = min_community_size
        self.max_communities = max_communities
    
    def detect_communities(
        self,
        node_type: str = "Wallet",
        relationship_types: Optional[List[str]] = None,
        algorithm: CommunityAlgorithm = CommunityAlgorithm.LOUVAIN,
        store_results: bool = True,
        community_property: str = "community_id",
        weight_property: Optional[str] = "amount_usd"
    ) -> Dict[str, Any]:
        """
        Detect communities in the graph
        
        Args:
            node_type: Type of nodes to analyze
            relationship_types: Types of relationships to include
            algorithm: Community detection algorithm to use
            store_results: Whether to store community IDs in the graph
            community_property: Property name for storing community ID
            weight_property: Edge property to use as weight
            
        Returns:
            Dictionary with community detection results
        """
        start_time = time.time()
        
        # Default relationship types if none provided
        if not relationship_types:
            relationship_types = ["SENDS_TO", "RECEIVES_FROM", "INTERACTS_WITH"]
        
        # Relationship projection string
        rel_projection = "{" + ", ".join([f"'{r}': {{orientation: 'NATURAL'}}" for r in relationship_types]) + "}"
        
        # Weight configuration
        weight_config = ""
        if weight_property:
            weight_config = f", relationshipProperties: ['{weight_property}']"
        
        # Project graph for community detection
        projection_query = f"""
        CALL gds.graph.project(
            'community_detection',
            '{node_type}',
            {rel_projection}{weight_config}
        )
        YIELD graphName, nodeCount, relationshipCount
        RETURN graphName, nodeCount, relationshipCount
        """
        
        try:
            # Project graph
            projection_result = self.neo4j_client.query(projection_query)
            if not projection_result:
                raise ValueError("Failed to project graph for community detection")
            
            node_count = projection_result[0]["nodeCount"]
            relationship_count = projection_result[0]["relationshipCount"]
            
            # Run community detection algorithm
            algo_name = algorithm.value
            
            # Configure algorithm-specific parameters
            algo_params = ""
            if algorithm == CommunityAlgorithm.LOUVAIN:
                algo_params = ", tolerance: 0.0001, maxIterations: 20"
            elif algorithm == CommunityAlgorithm.LABEL_PROPAGATION:
                algo_params = ", maxIterations: 10"
            
            # Weight parameter if needed
            weight_param = f", relationshipWeightProperty: '{weight_property}'" if weight_property else ""
            
            # Execute community detection
            if store_results:
                # Write mode - store results in the graph
                detection_query = f"""
                CALL gds.{algo_name}.write('community_detection', {{
                    writeProperty: '{community_property}'{weight_param}{algo_params}
                }})
                YIELD communityCount, modularity, modularities
                RETURN communityCount, modularity, modularities
                """
            else:
                # Stream mode - return results without storing
                detection_query = f"""
                CALL gds.{algo_name}.stream('community_detection'{weight_param}{algo_params})
                YIELD nodeId, communityId
                WITH gds.util.asNode(nodeId) AS node, communityId
                RETURN communityId, collect(node.id) AS node_ids, count(*) AS community_size
                ORDER BY community_size DESC
                LIMIT {self.max_communities}
                """
            
            # Execute algorithm
            detection_result = self.neo4j_client.query(detection_query)
            
            # Process results based on mode
            if store_results:
                # For write mode, get summary statistics
                community_count = detection_result[0]["communityCount"] if detection_result else 0
                modularity = detection_result[0]["modularity"] if detection_result else 0
                
                # Query to get community sizes
                community_stats_query = f"""
                MATCH (n:{node_type})
                WHERE n.{community_property} IS NOT NULL
                WITH n.{community_property} AS community_id, count(*) AS size
                WHERE size >= {self.min_community_size}
                RETURN community_id, size
                ORDER BY size DESC
                LIMIT {self.max_communities}
                """
                
                community_stats = self.neo4j_client.query(community_stats_query)
                communities = [
                    {"community_id": record["community_id"], "size": record["size"]}
                    for record in community_stats
                ]
                
                result = {
                    "algorithm": algorithm.value,
                    "community_count": community_count,
                    "modularity": modularity,
                    "top_communities": communities,
                    "node_count": node_count,
                    "relationship_count": relationship_count,
                    "community_property": community_property,
                    "execution_time_seconds": time.time() - start_time
                }
            else:
                # For stream mode, return community membership
                communities = []
                for record in detection_result:
                    if record["community_size"] >= self.min_community_size:
                        communities.append({
                            "community_id": record["communityId"],
                            "size": record["community_size"],
                            "node_ids": record["node_ids"]
                        })
                
                result = {
                    "algorithm": algorithm.value,
                    "community_count": len(communities),
                    "top_communities": communities,
                    "node_count": node_count,
                    "relationship_count": relationship_count,
                    "execution_time_seconds": time.time() - start_time
                }
            
            # Clean up projected graph
            cleanup_query = "CALL gds.graph.drop('community_detection', false)"
            self.neo4j_client.query(cleanup_query)
            
            # Record metrics
            duration = time.time() - start_time
            ADVANCED_GRAPH_LATENCY.labels(operation=f"community_{algorithm.value}").observe(duration)
            COMMUNITY_DETECTION_COUNT.labels(
                algorithm=algorithm.value,
                result="success"
            ).inc()
            
            return result
            
        except Exception as e:
            logger.error(f"Error in community detection: {str(e)}")
            COMMUNITY_DETECTION_COUNT.labels(
                algorithm=algorithm.value if isinstance(algorithm, CommunityAlgorithm) else "unknown",
                result="error"
            ).inc()
            
            # Attempt to clean up projected graph
            try:
                cleanup_query = "CALL gds.graph.drop('community_detection', false)"
                self.neo4j_client.query(cleanup_query)
            except:
                pass
                
            raise
    
    def analyze_community(
        self,
        community_id: Union[int, str],
        community_property: str = "community_id",
        node_type: str = "Wallet",
        include_centrality: bool = True
    ) -> Dict[str, Any]:
        """
        Analyze a specific community in detail
        
        Args:
            community_id: ID of the community to analyze
            community_property: Property name storing community ID
            node_type: Type of nodes in the community
            include_centrality: Whether to compute centrality metrics
            
        Returns:
            Dictionary with community analysis
        """
        start_time = time.time()
        
        # Query to get community members
        members_query = f"""
        MATCH (n:{node_type})
        WHERE n.{community_property} = $community_id
        RETURN n.id AS node_id, labels(n) AS labels, 
               properties(n) AS properties
        """
        
        try:
            # Get community members
            members_result = self.neo4j_client.query(members_query, {"community_id": community_id})
            
            if not members_result:
                return {
                    "community_id": community_id,
                    "members": [],
                    "size": 0,
                    "error": "Community not found or empty"
                }
            
            # Process member data
            members = []
            node_ids = []
            for record in members_result:
                node_ids.append(record["node_id"])
                members.append({
                    "node_id": record["node_id"],
                    "labels": record["labels"],
                    "properties": {
                        k: v for k, v in record["properties"].items()
                        if k not in ["community_id", "pagerank", "betweenness"]
                    }
                })
            
            # Get intra-community relationships
            relationships_query = f"""
            MATCH (n:{node_type})-[r]->(m:{node_type})
            WHERE n.{community_property} = $community_id 
              AND m.{community_property} = $community_id
              AND n.id IN $node_ids AND m.id IN $node_ids
            RETURN type(r) AS type, count(r) AS count,
                   sum(coalesce(r.amount_usd, 0)) AS total_value
            """
            
            relationships_result = self.neo4j_client.query(
                relationships_query, 
                {"community_id": community_id, "node_ids": node_ids}
            )
            
            relationships = [
                {
                    "type": record["type"],
                    "count": record["count"],
                    "total_value": record["total_value"]
                }
                for record in relationships_result
            ]
            
            # Calculate centrality if requested
            centrality = {}
            if include_centrality and node_ids:
                # Project subgraph for this community
                subgraph_query = f"""
                CALL gds.graph.project(
                    'community_analysis',
                    '{node_type}',
                    '*',
                    {{
                        nodeFilter: 'n.{community_property} = $community_id',
                        relationshipProperties: ['amount_usd']
                    }}
                )
                YIELD graphName
                RETURN graphName
                """
                
                self.neo4j_client.query(subgraph_query, {"community_id": community_id})
                
                # Calculate PageRank centrality
                pagerank_query = """
                CALL gds.pageRank.stream('community_analysis', {
                    relationshipWeightProperty: 'amount_usd',
                    maxIterations: 20
                })
                YIELD nodeId, score
                WITH gds.util.asNode(nodeId) AS node, score
                RETURN node.id AS node_id, score AS pagerank
                ORDER BY pagerank DESC
                """
                
                pagerank_result = self.neo4j_client.query(pagerank_query)
                
                # Calculate betweenness centrality
                betweenness_query = """
                CALL gds.betweenness.stream('community_analysis')
                YIELD nodeId, score
                WITH gds.util.asNode(nodeId) AS node, score
                RETURN node.id AS node_id, score AS betweenness
                ORDER BY betweenness DESC
                """
                
                betweenness_result = self.neo4j_client.query(betweenness_query)
                
                # Clean up
                cleanup_query = "CALL gds.graph.drop('community_analysis', false)"
                self.neo4j_client.query(cleanup_query)
                
                # Process centrality results
                pagerank = {r["node_id"]: r["pagerank"] for r in pagerank_result}
                betweenness = {r["node_id"]: r["betweenness"] for r in betweenness_result}
                
                # Add centrality to members
                for member in members:
                    node_id = member["node_id"]
                    member["centrality"] = {
                        "pagerank": pagerank.get(node_id, 0),
                        "betweenness": betweenness.get(node_id, 0)
                    }
                
                # Calculate community-level centrality metrics
                centrality = {
                    "max_pagerank": max(pagerank.values()) if pagerank else 0,
                    "max_betweenness": max(betweenness.values()) if betweenness else 0,
                    "centralization": self._calculate_centralization(pagerank.values()) if pagerank else 0
                }
            
            # Record metrics
            duration = time.time() - start_time
            ADVANCED_GRAPH_LATENCY.labels(operation="community_analysis").observe(duration)
            
            return {
                "community_id": community_id,
                "size": len(members),
                "members": members,
                "relationships": relationships,
                "centrality": centrality,
                "execution_time_seconds": duration
            }
            
        except Exception as e:
            logger.error(f"Error analyzing community: {str(e)}")
            
            # Attempt to clean up projected graph
            try:
                cleanup_query = "CALL gds.graph.drop('community_analysis', false)"
                self.neo4j_client.query(cleanup_query)
            except:
                pass
                
            raise
    
    def _calculate_centralization(self, centrality_values: List[float]) -> float:
        """Calculate network centralization index"""
        if not centrality_values:
            return 0
            
        values = list(centrality_values)
        n = len(values)
        if n <= 1:
            return 0
            
        max_val = max(values)
        sum_diff = sum(max_val - val for val in values)
        
        # Normalize by theoretical maximum
        max_sum_diff = (n - 1) * (n - 1)
        if max_sum_diff == 0:
            return 0
            
        return sum_diff / max_sum_diff


class AdvancedGraphTool(BaseTool):
    """
    Advanced Graph Algorithm Tool for CrewAI
    
    Provides enhanced graph analysis capabilities for fraud detection:
    - Multi-head attention GAT models
    - Community detection
    - Risk propagation
    - Centrality-based analysis
    """
    
    name = "advanced_graph_tool"
    description = """
    Advanced graph analysis tool for financial crime detection.
    
    Capabilities:
    - Detect communities of related entities using algorithms like Louvain
    - Propagate risk scores through the network to find hidden high-risk entities
    - Analyze network structure using centrality and community metrics
    - Identify suspicious patterns using enhanced graph neural networks
    
    Use this tool when you need to:
    - Find clusters of suspicious activity
    - Trace risk through complex transaction networks
    - Identify central actors in fraudulent schemes
    - Analyze the structure of financial communities
    """
    
    def __init__(self, neo4j_client: Neo4jClient):
        """
        Initialize the advanced graph tool
        
        Args:
            neo4j_client: Neo4j client for database access
        """
        super().__init__()
        self.neo4j_client = neo4j_client
        self.community_detector = CommunityDetector(neo4j_client)
        self.risk_propagator = RiskPropagator(neo4j_client)
    
    def _run(
        self,
        operation: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run the specified graph operation
        
        Args:
            operation: Operation to perform (detect_communities, propagate_risk, analyze_community)
            **kwargs: Operation-specific parameters
            
        Returns:
            Dictionary with operation results
        """
        logger.info(f"Running advanced graph operation: {operation}")
        
        if operation == "detect_communities":
            return self._detect_communities(**kwargs)
        elif operation == "propagate_risk":
            return self._propagate_risk(**kwargs)
        elif operation == "analyze_community":
            return self._analyze_community(**kwargs)
        elif operation == "enhanced_gnn_prediction":
            return self._enhanced_gnn_prediction(**kwargs)
        elif operation == "generate_evidence":
            return self._generate_evidence(**kwargs)
        else:
            raise ValueError(f"Unsupported operation: {operation}")
    
    def _detect_communities(self, **kwargs) -> Dict[str, Any]:
        """Detect communities in the graph"""
        # Extract parameters with defaults
        node_type = kwargs.get("node_type", "Wallet")
        relationship_types = kwargs.get("relationship_types")
        algorithm_name = kwargs.get("algorithm", "louvain")
        store_results = kwargs.get("store_results", True)
        community_property = kwargs.get("community_property", "community_id")
        weight_property = kwargs.get("weight_property", "amount_usd")
        min_community_size = kwargs.get("min_community_size", DEFAULT_COMMUNITY_SIZE_THRESHOLD)
        
        # Update detector configuration
        self.community_detector.min_community_size = min_community_size
        
        # Convert algorithm name to enum
        try:
            algorithm = CommunityAlgorithm(algorithm_name)
        except ValueError:
            raise ValueError(f"Unsupported algorithm: {algorithm_name}. " +
                           f"Must be one of {[a.value for a in CommunityAlgorithm]}")
        
        # Run community detection
        result = self.community_detector.detect_communities(
            node_type=node_type,
            relationship_types=relationship_types,
            algorithm=algorithm,
            store_results=store_results,
            community_property=community_property,
            weight_property=weight_property
        )
        
        return result
    
    def _propagate_risk(self, **kwargs) -> Dict[str, Any]:
        """Propagate risk through the graph"""
        # Extract parameters with defaults
        seed_nodes = kwargs.get("seed_nodes")
        if not seed_nodes:
            raise ValueError("Must provide seed_nodes parameter")
            
        node_type = kwargs.get("node_type", "Wallet")
        risk_property = kwargs.get("risk_property", "risk_score")
        weight_property = kwargs.get("weight_property", "amount_usd")
        direction = kwargs.get("direction", "both")
        decay_factor = kwargs.get("decay_factor", DEFAULT_RISK_DECAY_FACTOR)
        max_hops = kwargs.get("max_hops", 3)
        risk_threshold = kwargs.get("risk_threshold", DEFAULT_RISK_THRESHOLD)
        update_scores = kwargs.get("update_scores", False)
        
        # Update propagator configuration
        self.risk_propagator.decay_factor = decay_factor
        self.risk_propagator.max_hops = max_hops
        self.risk_propagator.risk_threshold = risk_threshold
        
        # Run risk propagation
        result = self.risk_propagator.propagate_risk(
            seed_nodes=seed_nodes,
            node_type=node_type,
            risk_property=risk_property,
            weight_property=weight_property,
            direction=direction
        )
        
        # Update scores in the graph if requested
        if update_scores:
            update_property = kwargs.get("update_property", "propagated_risk_score")
            update_threshold = kwargs.get("update_threshold", 0.5)
            
            update_result = self.risk_propagator.update_risk_scores(
                propagation_result=result,
                update_property=update_property,
                update_threshold=update_threshold
            )
            
            result["update_result"] = update_result
        
        return result
    
    def _analyze_community(self, **kwargs) -> Dict[str, Any]:
        """Analyze a specific community"""
        # Extract parameters with defaults
        community_id = kwargs.get("community_id")
        if community_id is None:
            raise ValueError("Must provide community_id parameter")
            
        community_property = kwargs.get("community_property", "community_id")
        node_type = kwargs.get("node_type", "Wallet")
        include_centrality = kwargs.get("include_centrality", True)
        
        # Run community analysis
        result = self.community_detector.analyze_community(
            community_id=community_id,
            community_property=community_property,
            node_type=node_type,
            include_centrality=include_centrality
        )
        
        return result
    
    def _enhanced_gnn_prediction(self, **kwargs) -> Dict[str, Any]:
        """Run enhanced GNN model for fraud prediction"""
        # This would be implemented to run the AdvancedGATModel
        # For now, return a placeholder
        return {
            "message": "Enhanced GNN prediction not yet implemented",
            "status": "pending"
        }
    
    def _generate_evidence(self, **kwargs) -> Dict[str, Any]:
        """Generate evidence bundle from graph analysis results"""
        # Extract parameters
        operation_results = kwargs.get("operation_results", {})
        operation_type = kwargs.get("operation_type", "")
        title = kwargs.get("title", f"Advanced Graph Analysis: {operation_type}")
        description = kwargs.get("description", "Evidence from advanced graph algorithms")
        
        # Create evidence bundle
        evidence = EvidenceBundle(
            title=title,
            description=description
        )
        
        # Add operation results as evidence
        evidence.add_evidence("graph_analysis_results", operation_results)
        
        # Add Cypher query for reproducibility if available
        if "cypher_query" in kwargs:
            evidence.add_evidence("cypher_query", kwargs["cypher_query"])
        
        # Add visualization data if available
        if "visualization_data" in kwargs:
            evidence.add_evidence("visualization_data", kwargs["visualization_data"])
        
        # Return serialized evidence
        return evidence.to_dict()


# Factory function for CrewAI integration
def get_advanced_graph_tool(neo4j_client: Neo4jClient) -> Tool:
    """
    Create an instance of the advanced graph tool for CrewAI
    
    Args:
        neo4j_client: Neo4j client for database access
        
    Returns:
        CrewAI Tool instance
    """
    tool = AdvancedGraphTool(neo4j_client)
    
    return Tool(
        name="advanced_graph_tool",
        description=tool.description,
        func=tool._run
    )
