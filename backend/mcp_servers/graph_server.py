"""
Graph MCP Server - Neo4j and Graph Query Tools for MCP

This MCP server wraps the Neo4j client and graph query functionality from analystt1,
exposing graph database operations through the Model Context Protocol (MCP).

Features:
- Cypher query execution
- Subgraph extraction
- Fraud pattern analysis
- Graph visualization data preparation

This server integrates with the existing Neo4j client and follows MCP protocol
standards for schema definition and error handling.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any, Union
import asyncio
from datetime import datetime

from mcpengine import Server, Tool, Context
from pydantic import BaseModel, Field, validator

from backend.core.logging import get_logger
from backend.integrations.neo4j_client import Neo4jClient
from backend.agents.tools.pattern_library_tool import PatternLibraryTool

# Configure logger
logger = get_logger(__name__)

# Initialize the MCP server
server = Server(
    name="graph-server",
    description="MCP server for Neo4j graph database operations and fraud detection"
)

# Initialize Neo4j client
neo4j_client = Neo4jClient()

# Initialize Pattern Library Tool for fraud detection
try:
    pattern_tool = PatternLibraryTool()
except Exception as e:
    logger.warning(f"Failed to initialize PatternLibraryTool: {e}")
    pattern_tool = None


class CypherQueryParams(BaseModel):
    """Parameters for executing a Cypher query."""
    
    query: str = Field(
        ..., 
        description="Cypher query to execute against the Neo4j database"
    )
    params: Optional[Dict[str, Any]] = Field(
        None, 
        description="Parameters to use in the query"
    )
    include_stats: bool = Field(
        False, 
        description="Whether to include query execution statistics"
    )
    
    @validator('query')
    def validate_query(cls, v):
        """Validate the Cypher query for security."""
        # Basic validation - prevent destructive operations
        dangerous_keywords = ['DELETE', 'DETACH DELETE', 'DROP', 'REMOVE', 'SET']
        
        # Check if any dangerous keywords appear as standalone operations
        query_upper = v.upper()
        for keyword in dangerous_keywords:
            if f" {keyword} " in f" {query_upper} ":
                # Allow if explicitly permitted via environment variable
                if os.environ.get("MCP_ALLOW_WRITE_OPERATIONS") != "1":
                    raise ValueError(
                        f"Write operation '{keyword}' not allowed. "
                        "Set MCP_ALLOW_WRITE_OPERATIONS=1 to enable."
                    )
        return v


class SubgraphParams(BaseModel):
    """Parameters for extracting a subgraph."""
    
    node_ids: Optional[List[str]] = Field(
        None, 
        description="List of node IDs to include in the subgraph"
    )
    node_labels: Optional[List[str]] = Field(
        None, 
        description="List of node labels to filter by"
    )
    relationship_types: Optional[List[str]] = Field(
        None, 
        description="List of relationship types to include"
    )
    max_hops: int = Field(
        2, 
        description="Maximum number of hops from seed nodes",
        ge=1,
        le=5
    )
    limit: int = Field(
        1000, 
        description="Maximum number of nodes to return",
        ge=1,
        le=10000
    )
    
    @validator('node_ids', 'node_labels', 'relationship_types')
    def validate_lists(cls, v, values):
        """Validate that at least one filter is provided."""
        if v is None and 'node_ids' not in values and 'node_labels' not in values:
            raise ValueError("At least one of node_ids or node_labels must be provided")
        return v


class FraudPatternParams(BaseModel):
    """Parameters for fraud pattern analysis."""
    
    transaction_ids: Optional[List[str]] = Field(
        None, 
        description="List of transaction IDs to analyze"
    )
    entity_ids: Optional[List[str]] = Field(
        None, 
        description="List of entity IDs to analyze"
    )
    pattern_types: Optional[List[str]] = Field(
        None, 
        description="Types of fraud patterns to look for"
    )
    time_window_days: Optional[int] = Field(
        30, 
        description="Time window for analysis in days",
        ge=1,
        le=365
    )
    
    @validator('transaction_ids', 'entity_ids')
    def validate_ids(cls, v, values):
        """Validate that at least one ID list is provided."""
        if v is None and 'transaction_ids' not in values and 'entity_ids' not in values:
            raise ValueError("At least one of transaction_ids or entity_ids must be provided")
        return v


@server.tool(
    name="cypher_query",
    description="Execute a Cypher query against the Neo4j database",
    input_schema=CypherQueryParams.schema()
)
async def cypher_query(ctx: Context, query: str, params: Optional[Dict[str, Any]] = None, include_stats: bool = False) -> Dict[str, Any]:
    """
    Execute a Cypher query against the Neo4j database.
    
    Args:
        ctx: MCP context
        query: Cypher query to execute
        params: Parameters to use in the query
        include_stats: Whether to include query execution statistics
        
    Returns:
        Query results and optional statistics
    """
    try:
        logger.info(f"Executing Cypher query: {query}")
        start_time = datetime.now()
        
        # Execute the query
        result = neo4j_client.execute_query(query, params)
        
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        # Format the result
        formatted_result = {
            "success": True,
            "records": result,
            "record_count": len(result)
        }
        
        # Include stats if requested
        if include_stats:
            formatted_result["stats"] = {
                "execution_time_seconds": execution_time,
                "query_length": len(query),
                "timestamp": end_time.isoformat()
            }
        
        logger.info(f"Cypher query executed successfully, returned {len(result)} records")
        return formatted_result
        
    except Exception as e:
        logger.error(f"Error executing Cypher query: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "query": query
        }


@server.tool(
    name="subgraph_extract",
    description="Extract a subgraph from the Neo4j database",
    input_schema=SubgraphParams.schema()
)
async def subgraph_extract(
    ctx: Context,
    node_ids: Optional[List[str]] = None,
    node_labels: Optional[List[str]] = None,
    relationship_types: Optional[List[str]] = None,
    max_hops: int = 2,
    limit: int = 1000
) -> Dict[str, Any]:
    """
    Extract a subgraph from the Neo4j database.
    
    Args:
        ctx: MCP context
        node_ids: List of node IDs to include in the subgraph
        node_labels: List of node labels to filter by
        relationship_types: List of relationship types to include
        max_hops: Maximum number of hops from seed nodes
        limit: Maximum number of nodes to return
        
    Returns:
        Subgraph data with nodes and relationships
    """
    try:
        logger.info(f"Extracting subgraph with {len(node_ids or [])} node IDs, {len(node_labels or [])} labels")
        
        # Build the Cypher query for subgraph extraction
        match_clauses = []
        where_clauses = []
        
        # Add node ID matching
        if node_ids and len(node_ids) > 0:
            id_list = json.dumps(node_ids)
            match_clauses.append("MATCH (n)")
            where_clauses.append(f"n.id IN {id_list}")
        
        # Add node label matching
        if node_labels and len(node_labels) > 0:
            label_conditions = []
            for label in node_labels:
                label_conditions.append(f"n:{label}")
            
            if not match_clauses:
                match_clauses.append("MATCH (n)")
            
            if label_conditions:
                where_clauses.append("(" + " OR ".join(label_conditions) + ")")
        
        # Build the complete match clause
        match_clause = " ".join(match_clauses)
        
        # Build the where clause
        where_clause = ""
        if where_clauses:
            where_clause = "WHERE " + " AND ".join(where_clauses)
        
        # Build relationship filter
        rel_filter = ""
        if relationship_types and len(relationship_types) > 0:
            rel_types = "|".join([t.replace(":", "") for t in relationship_types])
            rel_filter = f":{rel_types}"
        
        # Build the complete query
        query = f"""
        {match_clause}
        {where_clause}
        CALL apoc.path.subgraphAll(n, {{relationshipFilter: "{rel_filter}", maxLevel: {max_hops}, limit: {limit}}})
        YIELD nodes, relationships
        RETURN nodes, relationships
        LIMIT 1
        """
        
        # Execute the query
        result = neo4j_client.execute_query(query)
        
        if not result or len(result) == 0:
            return {
                "success": True,
                "nodes": [],
                "relationships": [],
                "node_count": 0,
                "relationship_count": 0,
                "message": "No subgraph found matching the criteria"
            }
        
        # Process the result
        nodes = []
        relationships = []
        
        for record in result:
            # Process nodes
            for node in record.get("nodes", []):
                node_data = {
                    "id": node.get("id", str(node.id)),
                    "labels": list(node.labels),
                    "properties": dict(node)
                }
                nodes.append(node_data)
            
            # Process relationships
            for rel in record.get("relationships", []):
                rel_data = {
                    "id": rel.get("id", str(rel.id)),
                    "type": rel.type,
                    "start_node": rel.start_node.get("id", str(rel.start_node.id)),
                    "end_node": rel.end_node.get("id", str(rel.end_node.id)),
                    "properties": dict(rel)
                }
                relationships.append(rel_data)
        
        # Prepare visualization data
        vis_data = {
            "nodes": [
                {
                    "id": node["id"],
                    "label": node["properties"].get("name", node["id"]),
                    "group": node["labels"][0] if node["labels"] else "Unknown"
                }
                for node in nodes
            ],
            "edges": [
                {
                    "from": rel["start_node"],
                    "to": rel["end_node"],
                    "label": rel["type"],
                    "arrows": "to"
                }
                for rel in relationships
            ]
        }
        
        logger.info(f"Subgraph extracted successfully: {len(nodes)} nodes, {len(relationships)} relationships")
        
        return {
            "success": True,
            "nodes": nodes,
            "relationships": relationships,
            "node_count": len(nodes),
            "relationship_count": len(relationships),
            "visualization_data": vis_data
        }
        
    except Exception as e:
        logger.error(f"Error extracting subgraph: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "node_ids": node_ids,
            "node_labels": node_labels
        }


@server.tool(
    name="fraud_pattern_detect",
    description="Analyze graph data for potential fraud patterns",
    input_schema=FraudPatternParams.schema()
)
async def fraud_pattern_detect(
    ctx: Context,
    transaction_ids: Optional[List[str]] = None,
    entity_ids: Optional[List[str]] = None,
    pattern_types: Optional[List[str]] = None,
    time_window_days: int = 30
) -> Dict[str, Any]:
    """
    Analyze graph data for potential fraud patterns.
    
    Args:
        ctx: MCP context
        transaction_ids: List of transaction IDs to analyze
        entity_ids: List of entity IDs to analyze
        pattern_types: Types of fraud patterns to look for
        time_window_days: Time window for analysis in days
        
    Returns:
        Detected fraud patterns and risk scores
    """
    try:
        logger.info(f"Analyzing fraud patterns for {len(transaction_ids or [])} transactions, {len(entity_ids or [])} entities")
        
        # First, extract the relevant subgraph
        subgraph_params = {
            "node_ids": transaction_ids or entity_ids,
            "max_hops": 3,  # Wider context for fraud detection
            "limit": 5000   # Larger limit for comprehensive analysis
        }
        
        # Use the subgraph_extract tool to get the data
        subgraph_result = await subgraph_extract(
            ctx,
            node_ids=subgraph_params["node_ids"],
            max_hops=subgraph_params["max_hops"],
            limit=subgraph_params["limit"]
        )
        
        if not subgraph_result["success"]:
            return {
                "success": False,
                "error": "Failed to extract subgraph for fraud analysis",
                "details": subgraph_result.get("error")
            }
        
        # Check if PatternLibraryTool is available
        if pattern_tool is None:
            # Fallback to basic pattern detection
            patterns = detect_basic_fraud_patterns(
                subgraph_result["nodes"],
                subgraph_result["relationships"],
                pattern_types
            )
        else:
            # Use the PatternLibraryTool for advanced detection
            patterns = detect_advanced_fraud_patterns(
                subgraph_result["nodes"],
                subgraph_result["relationships"],
                pattern_types,
                time_window_days
            )
        
        # Calculate overall risk score
        risk_score = calculate_risk_score(patterns)
        
        logger.info(f"Fraud analysis complete: {len(patterns)} patterns detected, risk score: {risk_score}")
        
        return {
            "success": True,
            "patterns": patterns,
            "risk_score": risk_score,
            "analyzed_nodes": subgraph_result["node_count"],
            "analyzed_relationships": subgraph_result["relationship_count"],
            "time_window_days": time_window_days,
            "visualization_data": subgraph_result["visualization_data"]
        }
        
    except Exception as e:
        logger.error(f"Error in fraud pattern detection: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "transaction_ids": transaction_ids,
            "entity_ids": entity_ids
        }


def detect_basic_fraud_patterns(nodes, relationships, pattern_types=None):
    """
    Perform basic fraud pattern detection on a subgraph.
    
    This is a fallback when PatternLibraryTool is not available.
    
    Args:
        nodes: List of nodes in the subgraph
        relationships: List of relationships in the subgraph
        pattern_types: Types of patterns to look for
        
    Returns:
        List of detected patterns
    """
    patterns = []
    
    # Map nodes by ID for quick lookup
    node_map = {node["id"]: node for node in nodes}
    
    # Check for circular transactions
    # A->B->C->A pattern
    circular_paths = find_circular_paths(relationships, node_map)
    if circular_paths:
        patterns.append({
            "type": "circular_transaction",
            "severity": "high",
            "description": "Circular flow of funds detected",
            "instances": circular_paths,
            "confidence": 0.85
        })
    
    # Check for rapid succession transactions
    rapid_txs = find_rapid_succession_transactions(nodes)
    if rapid_txs:
        patterns.append({
            "type": "rapid_succession",
            "severity": "medium",
            "description": "Multiple transactions in rapid succession",
            "instances": rapid_txs,
            "confidence": 0.75
        })
    
    # Check for unusual amounts
    unusual_amounts = find_unusual_amounts(nodes)
    if unusual_amounts:
        patterns.append({
            "type": "unusual_amount",
            "severity": "medium",
            "description": "Transactions with unusual amounts",
            "instances": unusual_amounts,
            "confidence": 0.65
        })
    
    # Filter by requested pattern types
    if pattern_types:
        patterns = [p for p in patterns if p["type"] in pattern_types]
    
    return patterns


def detect_advanced_fraud_patterns(nodes, relationships, pattern_types=None, time_window_days=30):
    """
    Use PatternLibraryTool for advanced fraud pattern detection.
    
    Args:
        nodes: List of nodes in the subgraph
        relationships: List of relationships in the subgraph
        pattern_types: Types of patterns to look for
        time_window_days: Time window for analysis in days
        
    Returns:
        List of detected patterns
    """
    # Convert nodes and relationships to the format expected by PatternLibraryTool
    graph_data = {
        "nodes": nodes,
        "edges": relationships
    }
    
    # Call PatternLibraryTool
    result = pattern_tool.run(
        graph_data=graph_data,
        pattern_types=pattern_types or "all",
        time_window_days=time_window_days
    )
    
    # Process and return the patterns
    if result and "patterns" in result:
        return result["patterns"]
    
    return []


def find_circular_paths(relationships, node_map, max_path_length=5):
    """Find circular paths in the relationship graph."""
    # Build adjacency list
    adjacency = {}
    for rel in relationships:
        start_id = rel["start_node"]
        end_id = rel["end_node"]
        
        if start_id not in adjacency:
            adjacency[start_id] = []
        adjacency[start_id].append(end_id)
    
    # Find circular paths
    circular_paths = []
    
    def dfs(node_id, path, visited):
        if len(path) > max_path_length:
            return
        
        if node_id in visited and node_id == path[0] and len(path) > 2:
            # Found circular path
            circular_paths.append(path.copy())
            return
        
        if node_id in visited:
            return
        
        visited.add(node_id)
        path.append(node_id)
        
        if node_id in adjacency:
            for neighbor in adjacency[node_id]:
                dfs(neighbor, path, visited.copy())
        
        path.pop()
    
    # Start DFS from each node
    for node_id in node_map:
        if node_id in adjacency:
            dfs(node_id, [], set())
    
    # Format the results
    formatted_paths = []
    for path in circular_paths:
        formatted_path = []
        for node_id in path:
            if node_id in node_map:
                node = node_map[node_id]
                formatted_path.append({
                    "id": node_id,
                    "type": node["labels"][0] if node["labels"] else "Unknown",
                    "properties": {
                        k: v for k, v in node["properties"].items()
                        if k in ["amount", "timestamp", "name", "id"]
                    }
                })
        
        formatted_paths.append({
            "path": formatted_path,
            "length": len(formatted_path)
        })
    
    return formatted_paths


def find_rapid_succession_transactions(nodes):
    """Find transactions that occurred in rapid succession."""
    # Filter for transaction nodes
    transactions = [
        node for node in nodes 
        if "Transaction" in node.get("labels", []) and "timestamp" in node.get("properties", {})
    ]
    
    # Sort by timestamp
    transactions.sort(key=lambda x: x["properties"].get("timestamp", ""))
    
    # Find rapid succession (transactions within 1 minute of each other)
    rapid_groups = []
    current_group = []
    
    for i in range(len(transactions) - 1):
        current_tx = transactions[i]
        next_tx = transactions[i + 1]
        
        # Add current to group if empty
        if not current_group:
            current_group.append(current_tx)
        
        # Check time difference
        try:
            current_time = datetime.fromisoformat(current_tx["properties"].get("timestamp").replace("Z", "+00:00"))
            next_time = datetime.fromisoformat(next_tx["properties"].get("timestamp").replace("Z", "+00:00"))
            
            time_diff = (next_time - current_time).total_seconds()
            
            if time_diff < 60:  # Within 1 minute
                current_group.append(next_tx)
            else:
                # End current group if it has multiple transactions
                if len(current_group) > 1:
                    rapid_groups.append(current_group)
                current_group = []
        except (ValueError, AttributeError):
            # Skip if timestamp parsing fails
            continue
    
    # Add the last group if not empty
    if len(current_group) > 1:
        rapid_groups.append(current_group)
    
    # Format the results
    formatted_groups = []
    for group in rapid_groups:
        formatted_group = []
        for tx in group:
            formatted_group.append({
                "id": tx["id"],
                "timestamp": tx["properties"].get("timestamp"),
                "amount": tx["properties"].get("amount"),
                "sender": tx["properties"].get("sender"),
                "receiver": tx["properties"].get("receiver")
            })
        
        formatted_groups.append({
            "transactions": formatted_group,
            "count": len(formatted_group),
            "time_span_seconds": calculate_time_span(formatted_group)
        })
    
    return formatted_groups


def find_unusual_amounts(nodes):
    """Find transactions with unusual amounts."""
    # Filter for transaction nodes with amount
    transactions = [
        node for node in nodes 
        if "Transaction" in node.get("labels", []) and "amount" in node.get("properties", {})
    ]
    
    if not transactions:
        return []
    
    # Calculate statistics
    amounts = [float(tx["properties"].get("amount", 0)) for tx in transactions]
    avg_amount = sum(amounts) / len(amounts)
    
    # Calculate standard deviation
    variance = sum((x - avg_amount) ** 2 for x in amounts) / len(amounts)
    std_dev = variance ** 0.5
    
    # Find unusual amounts (> 2 standard deviations from mean)
    unusual = []
    for tx in transactions:
        amount = float(tx["properties"].get("amount", 0))
        if abs(amount - avg_amount) > 2 * std_dev:
            unusual.append({
                "id": tx["id"],
                "amount": amount,
                "timestamp": tx["properties"].get("timestamp"),
                "deviation": abs(amount - avg_amount) / std_dev if std_dev > 0 else 0
            })
    
    return unusual


def calculate_time_span(transactions):
    """Calculate the time span of a group of transactions in seconds."""
    timestamps = []
    for tx in transactions:
        try:
            timestamp = datetime.fromisoformat(tx.get("timestamp", "").replace("Z", "+00:00"))
            timestamps.append(timestamp)
        except (ValueError, AttributeError):
            continue
    
    if len(timestamps) < 2:
        return 0
    
    timestamps.sort()
    return (timestamps[-1] - timestamps[0]).total_seconds()


def calculate_risk_score(patterns):
    """
    Calculate an overall risk score based on detected patterns.
    
    Args:
        patterns: List of detected fraud patterns
        
    Returns:
        Risk score between 0 and 1
    """
    if not patterns:
        return 0.0
    
    # Define severity weights
    severity_weights = {
        "critical": 1.0,
        "high": 0.8,
        "medium": 0.5,
        "low": 0.3
    }
    
    # Calculate weighted score
    total_weight = 0
    weighted_sum = 0
    
    for pattern in patterns:
        severity = pattern.get("severity", "medium")
        confidence = pattern.get("confidence", 0.5)
        instances = len(pattern.get("instances", []))
        
        weight = severity_weights.get(severity, 0.5) * confidence * min(instances, 5) / 5
        weighted_sum += weight
        total_weight += 1
    
    # Normalize to 0-1 range
    if total_weight == 0:
        return 0.0
    
    raw_score = weighted_sum / total_weight
    
    # Apply sigmoid function to emphasize mid-range scores
    # score = 1 / (1 + math.exp(-10 * (raw_score - 0.5)))
    
    # Cap at 0.95 to avoid absolute certainty
    return min(raw_score, 0.95)


if __name__ == "__main__":
    """Run the MCP server."""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Run the server
    logger.info("Starting Graph MCP Server...")
    server.run()
