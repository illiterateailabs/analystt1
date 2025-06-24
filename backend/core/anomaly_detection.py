"""
Anomaly Detection Service

This module provides a comprehensive anomaly detection service for identifying
suspicious patterns, transactions, and behaviors in blockchain and financial data.
It supports multiple detection algorithms, configurable thresholds, and integration
with the evidence system for automatic documentation of findings.

Features:
- Multiple detection algorithms (statistical, ML-based, graph-based)
- Fraud pattern matching and heuristic rules
- Automatic evidence creation for detected anomalies
- Real-time scoring and alert generation
- Configurable thresholds and detection strategies
- Support for different data types (transactions, addresses, patterns)
- Integration with graph tools and Redis for caching
- Background processing for continuous monitoring
"""

import asyncio
import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
from pydantic import BaseModel, Field, validator

from backend.core.events import EventPriority, publish_event
from backend.core.evidence import (
    EvidenceBundle, AnomalyEvidence, EvidenceSource, 
    create_evidence_bundle, GraphElementEvidence
)
from backend.core.metrics import BusinessMetrics
from backend.core.redis_client import RedisClient, RedisDb, SerializationFormat
from backend.integrations.neo4j_client import Neo4jClient

# Configure module logger
logger = logging.getLogger(__name__)


# --- Data Models ---

class AnomalyType(str, Enum):
    """Types of anomalies that can be detected."""
    STATISTICAL_OUTLIER = "statistical_outlier"
    PATTERN_MATCH = "pattern_match"
    GRAPH_STRUCTURE = "graph_structure"
    TEMPORAL_PATTERN = "temporal_pattern"
    BEHAVIORAL_CHANGE = "behavioral_change"
    ML_PREDICTION = "ml_prediction"
    RULE_VIOLATION = "rule_violation"
    COMMUNITY_OUTLIER = "community_outlier"


class AnomalySeverity(str, Enum):
    """Severity levels for detected anomalies."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DetectionMethod(str, Enum):
    """Methods used for anomaly detection."""
    STATISTICAL = "statistical"
    MACHINE_LEARNING = "machine_learning"
    GRAPH_NEURAL_NETWORK = "graph_neural_network"
    PATTERN_MATCHING = "pattern_matching"
    RULE_BASED = "rule_based"
    COMMUNITY_DETECTION = "community_detection"
    TEMPORAL_ANALYSIS = "temporal_analysis"
    HYBRID = "hybrid"


class DataEntityType(str, Enum):
    """Types of data entities that can be analyzed."""
    ADDRESS = "address"
    TRANSACTION = "transaction"
    TOKEN = "token"
    CONTRACT = "contract"
    SUBGRAPH = "subgraph"
    WALLET_CLUSTER = "wallet_cluster"
    EXCHANGE = "exchange"
    PROTOCOL = "protocol"


class AlertStatus(str, Enum):
    """Status of an anomaly alert."""
    NEW = "new"
    INVESTIGATING = "investigating"
    CONFIRMED = "confirmed"
    FALSE_POSITIVE = "false_positive"
    RESOLVED = "resolved"
    IGNORED = "ignored"


class DetectionStrategy(BaseModel):
    """Configuration for a detection strategy."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: Optional[str] = None
    method: DetectionMethod
    entity_types: List[DataEntityType]
    enabled: bool = True
    threshold_config: Dict[str, Any] = Field(default_factory=dict)
    parameters: Dict[str, Any] = Field(default_factory=dict)
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    created_by: Optional[str] = None
    
    class Config:
        schema_extra = {
            "example": {
                "name": "Large Transaction Detection",
                "description": "Detects unusually large transactions",
                "method": "statistical",
                "entity_types": ["transaction"],
                "threshold_config": {
                    "z_score_threshold": 3.0,
                    "min_transaction_value": 10000
                },
                "parameters": {
                    "lookback_days": 30,
                    "min_samples": 100
                }
            }
        }


class AnomalyDetectionResult(BaseModel):
    """Result of an anomaly detection run."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    anomaly_type: AnomalyType
    severity: AnomalySeverity
    confidence: float = Field(ge=0.0, le=1.0)
    entity_type: DataEntityType
    entity_id: str
    detection_method: DetectionMethod
    strategy_id: str
    detection_time: datetime = Field(default_factory=datetime.now)
    score: float
    threshold: float
    details: Dict[str, Any] = Field(default_factory=dict)
    related_entities: List[Dict[str, Any]] = Field(default_factory=list)
    evidence_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = self.dict()
        result["anomaly_type"] = self.anomaly_type.value
        result["severity"] = self.severity.value
        result["entity_type"] = self.entity_type.value
        result["detection_method"] = self.detection_method.value
        result["detection_time"] = self.detection_time.isoformat()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AnomalyDetectionResult":
        """Create from dictionary."""
        if "detection_time" in data and isinstance(data["detection_time"], str):
            data["detection_time"] = datetime.fromisoformat(data["detection_time"])
        return cls(**data)


class AnomalyAlert(BaseModel):
    """Alert generated from an anomaly detection result."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    anomaly_id: str
    title: str
    description: str
    severity: AnomalySeverity
    status: AlertStatus = AlertStatus.NEW
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    assigned_to: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    entity_type: DataEntityType
    entity_id: str
    evidence_id: Optional[str] = None
    investigation_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = self.dict()
        result["severity"] = self.severity.value
        result["status"] = self.status.value
        result["entity_type"] = self.entity_type.value
        result["created_at"] = self.created_at.isoformat()
        result["updated_at"] = self.updated_at.isoformat()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AnomalyAlert":
        """Create from dictionary."""
        if "created_at" in data and isinstance(data["created_at"], str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        if "updated_at" in data and isinstance(data["updated_at"], str):
            data["updated_at"] = datetime.fromisoformat(data["updated_at"])
        return cls(**data)


# --- Detection Algorithms ---

class StatisticalDetector:
    """Statistical methods for anomaly detection."""
    
    @staticmethod
    def z_score_detection(
        values: List[float],
        threshold: float = 3.0,
        min_samples: int = 10
    ) -> List[Tuple[int, float]]:
        """
        Detect outliers using Z-score method.
        
        Args:
            values: List of numerical values
            threshold: Z-score threshold for outlier detection
            min_samples: Minimum number of samples required
            
        Returns:
            List of (index, z_score) tuples for outliers
        """
        if len(values) < min_samples:
            logger.warning(f"Not enough samples for z-score detection: {len(values)} < {min_samples}")
            return []
        
        try:
            values_array = np.array(values, dtype=float)
            mean = np.mean(values_array)
            std = np.std(values_array)
            
            if std == 0:
                logger.warning("Standard deviation is zero, cannot compute z-scores")
                return []
            
            z_scores = np.abs((values_array - mean) / std)
            outliers = [(i, z_scores[i]) for i in range(len(z_scores)) if z_scores[i] > threshold]
            
            return outliers
        except Exception as e:
            logger.error(f"Error in z_score_detection: {e}")
            return []
    
    @staticmethod
    def iqr_detection(
        values: List[float],
        threshold: float = 1.5,
        min_samples: int = 10
    ) -> List[Tuple[int, float]]:
        """
        Detect outliers using Interquartile Range (IQR) method.
        
        Args:
            values: List of numerical values
            threshold: IQR multiplier for outlier detection
            min_samples: Minimum number of samples required
            
        Returns:
            List of (index, distance) tuples for outliers
        """
        if len(values) < min_samples:
            logger.warning(f"Not enough samples for IQR detection: {len(values)} < {min_samples}")
            return []
        
        try:
            values_array = np.array(values, dtype=float)
            q1 = np.percentile(values_array, 25)
            q3 = np.percentile(values_array, 75)
            iqr = q3 - q1
            
            if iqr == 0:
                logger.warning("IQR is zero, cannot compute outliers")
                return []
            
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            
            outliers = []
            for i, value in enumerate(values_array):
                if value < lower_bound or value > upper_bound:
                    # Calculate distance as number of IQRs from the nearest bound
                    if value < lower_bound:
                        distance = (lower_bound - value) / iqr
                    else:
                        distance = (value - upper_bound) / iqr
                    outliers.append((i, distance))
            
            return outliers
        except Exception as e:
            logger.error(f"Error in iqr_detection: {e}")
            return []
    
    @staticmethod
    def moving_average_detection(
        values: List[float],
        window_size: int = 10,
        threshold: float = 2.0,
        min_samples: int = 20
    ) -> List[Tuple[int, float]]:
        """
        Detect outliers using moving average method.
        
        Args:
            values: List of numerical values (time series)
            window_size: Size of the moving window
            threshold: Threshold multiplier for outlier detection
            min_samples: Minimum number of samples required
            
        Returns:
            List of (index, deviation) tuples for outliers
        """
        if len(values) < min_samples:
            logger.warning(f"Not enough samples for moving average detection: {len(values)} < {min_samples}")
            return []
        
        try:
            values_array = np.array(values, dtype=float)
            outliers = []
            
            # Calculate moving average and standard deviation
            for i in range(window_size, len(values_array)):
                window = values_array[i-window_size:i]
                mean = np.mean(window)
                std = np.std(window)
                
                if std == 0:
                    continue
                
                deviation = abs((values_array[i] - mean) / std)
                if deviation > threshold:
                    outliers.append((i, deviation))
            
            return outliers
        except Exception as e:
            logger.error(f"Error in moving_average_detection: {e}")
            return []


class GraphBasedDetector:
    """Graph-based methods for anomaly detection."""
    
    def __init__(self, neo4j_client: Optional[Neo4jClient] = None):
        """
        Initialize the graph-based detector.
        
        Args:
            neo4j_client: Optional Neo4jClient instance
        """
        self.neo4j_client = neo4j_client or Neo4jClient()
    
    def detect_unusual_patterns(
        self,
        entity_id: str,
        entity_type: DataEntityType,
        pattern_query: str,
        parameters: Dict[str, Any] = None,
        threshold: float = 0.8
    ) -> Dict[str, Any]:
        """
        Detect unusual patterns in the graph.
        
        Args:
            entity_id: ID of the entity to analyze
            entity_type: Type of entity
            pattern_query: Cypher query to detect patterns
            parameters: Query parameters
            threshold: Threshold for pattern matching
            
        Returns:
            Detection results
        """
        try:
            # Prepare parameters
            params = parameters or {}
            params["entity_id"] = entity_id
            
            # Execute query
            results = self.neo4j_client.execute_query(pattern_query, params)
            
            # Process results
            if not results:
                return {
                    "detected": False,
                    "score": 0.0,
                    "details": {}
                }
            
            # Extract score from results if available
            score = 0.0
            if "score" in results[0]:
                score = float(results[0]["score"])
            
            # Determine if pattern is detected based on threshold
            detected = score >= threshold
            
            return {
                "detected": detected,
                "score": score,
                "threshold": threshold,
                "details": results[0],
                "related_entities": results
            }
        except Exception as e:
            logger.error(f"Error in detect_unusual_patterns: {e}")
            return {
                "detected": False,
                "score": 0.0,
                "error": str(e),
                "details": {}
            }
    
    def detect_community_outliers(
        self,
        entity_id: str,
        community_query: str,
        parameters: Dict[str, Any] = None,
        threshold: float = 0.7
    ) -> Dict[str, Any]:
        """
        Detect entities that are outliers within their communities.
        
        Args:
            entity_id: ID of the entity to analyze
            community_query: Cypher query to identify community and outliers
            parameters: Query parameters
            threshold: Threshold for outlier detection
            
        Returns:
            Detection results
        """
        try:
            # Prepare parameters
            params = parameters or {}
            params["entity_id"] = entity_id
            
            # Execute query
            results = self.neo4j_client.execute_query(community_query, params)
            
            # Process results
            if not results:
                return {
                    "detected": False,
                    "score": 0.0,
                    "details": {}
                }
            
            # Extract community information and outlier score
            community_size = results[0].get("community_size", 0)
            outlier_score = results[0].get("outlier_score", 0.0)
            
            # Determine if entity is an outlier
            is_outlier = outlier_score >= threshold
            
            return {
                "detected": is_outlier,
                "score": outlier_score,
                "threshold": threshold,
                "community_size": community_size,
                "details": results[0],
                "related_entities": results[:10]  # Limit to 10 related entities
            }
        except Exception as e:
            logger.error(f"Error in detect_community_outliers: {e}")
            return {
                "detected": False,
                "score": 0.0,
                "error": str(e),
                "details": {}
            }
    
    def detect_unusual_subgraph(
        self,
        center_node_id: str,
        depth: int = 2,
        parameters: Dict[str, Any] = None,
        threshold: float = 0.6
    ) -> Dict[str, Any]:
        """
        Detect unusual subgraph structures centered around a node.
        
        Args:
            center_node_id: ID of the center node
            depth: Depth of subgraph exploration
            parameters: Additional parameters
            threshold: Threshold for unusualness detection
            
        Returns:
            Detection results
        """
        try:
            # Query to extract subgraph and compute metrics
            query = f"""
            MATCH (center {{id: $center_node_id}})
            CALL apoc.path.subgraphAll(center, {{maxLevel: {depth}}})
            YIELD nodes, relationships
            WITH nodes, relationships,
                 size(nodes) AS node_count,
                 size(relationships) AS rel_count
            
            // Calculate basic metrics
            WITH nodes, relationships, node_count, rel_count,
                 CASE WHEN node_count > 0 THEN toFloat(rel_count) / node_count ELSE 0 END AS density
            
            // Detect unusual patterns in the subgraph
            WITH nodes, relationships, node_count, rel_count, density,
                 size([n IN nodes WHERE size((n)--()) > 5]) AS high_degree_nodes,
                 size([r IN relationships WHERE r.value > 10000]) AS high_value_txs
            
            // Calculate unusualness score
            WITH nodes, relationships, node_count, rel_count, density, high_degree_nodes, high_value_txs,
                 (toFloat(high_degree_nodes) / node_count) * 0.5 + 
                 (toFloat(high_value_txs) / CASE WHEN rel_count > 0 THEN rel_count ELSE 1 END) * 0.5 AS unusualness_score
            
            RETURN node_count, rel_count, density, high_degree_nodes, high_value_txs, unusualness_score,
                   [n IN nodes | n.id][..10] AS sample_node_ids
            """
            
            # Prepare parameters
            params = parameters or {}
            params["center_node_id"] = center_node_id
            
            # Execute query
            results = self.neo4j_client.execute_query(query, params)
            
            # Process results
            if not results:
                return {
                    "detected": False,
                    "score": 0.0,
                    "details": {}
                }
            
            # Extract metrics
            unusualness_score = results[0].get("unusualness_score", 0.0)
            
            # Determine if subgraph is unusual
            is_unusual = unusualness_score >= threshold
            
            return {
                "detected": is_unusual,
                "score": unusualness_score,
                "threshold": threshold,
                "node_count": results[0].get("node_count", 0),
                "relationship_count": results[0].get("rel_count", 0),
                "density": results[0].get("density", 0.0),
                "high_degree_nodes": results[0].get("high_degree_nodes", 0),
                "high_value_transactions": results[0].get("high_value_txs", 0),
                "sample_node_ids": results[0].get("sample_node_ids", []),
                "details": results[0]
            }
        except Exception as e:
            logger.error(f"Error in detect_unusual_subgraph: {e}")
            return {
                "detected": False,
                "score": 0.0,
                "error": str(e),
                "details": {}
            }


class PatternMatcher:
    """Pattern matching for known fraud patterns."""
    
    def __init__(self, redis_client: Optional[RedisClient] = None):
        """
        Initialize the pattern matcher.
        
        Args:
            redis_client: Optional RedisClient instance
        """
        self.redis_client = redis_client or RedisClient()
        self.patterns = self._load_patterns()
    
    def _load_patterns(self) -> Dict[str, Dict[str, Any]]:
        """
        Load fraud patterns from Redis or default patterns.
        
        Returns:
            Dictionary of patterns
        """
        patterns_key = "anomaly:patterns"
        patterns_data = self.redis_client.get(
            key=patterns_key,
            db=RedisDb.CACHE,
            format=SerializationFormat.JSON
        )
        
        if patterns_data:
            logger.info(f"Loaded {len(patterns_data)} fraud patterns from Redis")
            return patterns_data
        
        # Default patterns if none found in Redis
        default_patterns = {
            "wash_trading": {
                "name": "Wash Trading",
                "description": "Artificial trading activity between related accounts",
                "entity_type": DataEntityType.ADDRESS.value,
                "detection_query": """
                MATCH (a:Address {address: $address})-[t1:TRANSFERRED]->(b:Address)
                MATCH (b)-[t2:TRANSFERRED]->(a)
                WHERE t1.timestamp <= t2.timestamp + duration('P1D')
                  AND t1.timestamp >= t2.timestamp - duration('P1D')
                RETURN count(t1) AS cycle_count,
                       sum(t1.value) AS total_value,
                       collect(t1.hash)[..5] AS sample_tx_hashes,
                       collect(DISTINCT b.address)[..5] AS counterparties,
                       CASE WHEN count(t1) > 5 THEN true ELSE false END AS is_suspicious,
                       CASE 
                         WHEN count(t1) > 20 THEN 0.9
                         WHEN count(t1) > 10 THEN 0.7
                         WHEN count(t1) > 5 THEN 0.5
                         ELSE 0.3
                       END AS confidence
                """,
                "threshold": 5,
                "severity_rules": {
                    "low": {"cycle_count": 5, "total_value": 1000},
                    "medium": {"cycle_count": 10, "total_value": 10000},
                    "high": {"cycle_count": 20, "total_value": 100000},
                    "critical": {"cycle_count": 50, "total_value": 1000000}
                }
            },
            "smurfing": {
                "name": "Smurfing",
                "description": "Breaking down large transactions into many smaller ones",
                "entity_type": DataEntityType.ADDRESS.value,
                "detection_query": """
                MATCH (a:Address {address: $address})-[t:TRANSFERRED]->(b:Address)
                WITH a, b, collect(t) AS transactions
                WHERE size(transactions) > 10
                  AND max(transactions[i IN range(0, size(transactions)-1) | i].timestamp) - 
                      min(transactions[i IN range(0, size(transactions)-1) | i].timestamp) < duration('P1D')
                RETURN b.address AS recipient,
                       size(transactions) AS tx_count,
                       sum(t.value) AS total_value,
                       avg(t.value) AS avg_value,
                       collect(t.hash)[..5] AS sample_tx_hashes,
                       CASE WHEN size(transactions) > 20 THEN true ELSE false END AS is_suspicious,
                       CASE 
                         WHEN size(transactions) > 50 THEN 0.9
                         WHEN size(transactions) > 30 THEN 0.7
                         WHEN size(transactions) > 20 THEN 0.5
                         ELSE 0.3
                       END AS confidence
                """,
                "threshold": 20,
                "severity_rules": {
                    "low": {"tx_count": 20, "total_value": 10000},
                    "medium": {"tx_count": 30, "total_value": 50000},
                    "high": {"tx_count": 50, "total_value": 100000},
                    "critical": {"tx_count": 100, "total_value": 500000}
                }
            },
            "round_amount": {
                "name": "Round Amount Transactions",
                "description": "Suspiciously round transaction amounts",
                "entity_type": DataEntityType.TRANSACTION.value,
                "detection_query": """
                MATCH (a:Address)-[t:TRANSFERRED {hash: $tx_hash}]->(b:Address)
                WITH a, b, t,
                     t.value % 1000 AS remainder_1000,
                     t.value % 10000 AS remainder_10000,
                     t.value % 100000 AS remainder_100000
                WHERE (remainder_1000 = 0 OR remainder_10000 = 0 OR remainder_100000 = 0)
                  AND t.value >= 10000
                RETURN a.address AS sender,
                       b.address AS recipient,
                       t.value AS amount,
                       t.hash AS tx_hash,
                       CASE 
                         WHEN remainder_100000 = 0 THEN 0.8
                         WHEN remainder_10000 = 0 THEN 0.6
                         WHEN remainder_1000 = 0 THEN 0.4
                         ELSE 0.2
                       END AS confidence,
                       CASE
                         WHEN remainder_100000 = 0 AND t.value >= 1000000 THEN true
                         WHEN remainder_10000 = 0 AND t.value >= 100000 THEN true
                         WHEN remainder_1000 = 0 AND t.value >= 50000 THEN true
                         ELSE false
                       END AS is_suspicious
                """,
                "threshold": 0.5,
                "severity_rules": {
                    "low": {"amount": 10000},
                    "medium": {"amount": 50000},
                    "high": {"amount": 100000},
                    "critical": {"amount": 1000000}
                }
            },
            "layering": {
                "name": "Layering",
                "description": "Funds passing through multiple intermediaries",
                "entity_type": DataEntityType.ADDRESS.value,
                "detection_query": """
                MATCH path = (source:Address)-[t1:TRANSFERRED]->(a:Address {address: $address})-[t2:TRANSFERRED]->(dest:Address)
                WHERE t1.timestamp <= t2.timestamp + duration('PT1H')
                  AND t1.timestamp >= t2.timestamp - duration('PT1H')
                  AND abs(t1.value - t2.value) / t1.value < 0.1
                WITH source, a, dest, t1, t2
                RETURN source.address AS original_source,
                       dest.address AS final_destination,
                       t1.hash AS incoming_tx,
                       t2.hash AS outgoing_tx,
                       t1.value AS incoming_value,
                       t2.value AS outgoing_value,
                       abs(t1.timestamp - t2.timestamp) AS time_difference_ms,
                       abs(t1.value - t2.value) / t1.value AS value_difference_ratio,
                       CASE
                         WHEN abs(t1.value - t2.value) / t1.value < 0.05 AND 
                              abs(t1.timestamp - t2.timestamp) < duration('PT10M') THEN true
                         ELSE false
                       END AS is_suspicious,
                       CASE
                         WHEN abs(t1.value - t2.value) / t1.value < 0.01 AND 
                              abs(t1.timestamp - t2.timestamp) < duration('PT1M') THEN 0.9
                         WHEN abs(t1.value - t2.value) / t1.value < 0.05 AND 
                              abs(t1.timestamp - t2.timestamp) < duration('PT10M') THEN 0.7
                         WHEN abs(t1.value - t2.value) / t1.value < 0.1 AND 
                              abs(t1.timestamp - t2.timestamp) < duration('PT1H') THEN 0.5
                         ELSE 0.3
                       END AS confidence
                """,
                "threshold": 0.7,
                "severity_rules": {
                    "low": {"value_difference_ratio": 0.1, "time_difference_ms": 3600000},
                    "medium": {"value_difference_ratio": 0.05, "time_difference_ms": 600000},
                    "high": {"value_difference_ratio": 0.01, "time_difference_ms": 60000},
                    "critical": {"value_difference_ratio": 0.005, "time_difference_ms": 10000}
                }
            }
        }
        
        # Store default patterns in Redis
        self.redis_client.set(
            key=patterns_key,
            value=default_patterns,
            ttl_seconds=None,  # No expiration
            db=RedisDb.CACHE,
            format=SerializationFormat.JSON
        )
        
        logger.info(f"Stored {len(default_patterns)} default fraud patterns in Redis")
        return default_patterns
    
    def match_pattern(
        self,
        pattern_id: str,
        entity_id: str,
        neo4j_client: Optional[Neo4jClient] = None
    ) -> Dict[str, Any]:
        """
        Match an entity against a specific fraud pattern.
        
        Args:
            pattern_id: ID of the pattern to match
            entity_id: ID of the entity to check
            neo4j_client: Optional Neo4jClient instance
            
        Returns:
            Pattern matching results
        """
        if pattern_id not in self.patterns:
            logger.warning(f"Pattern {pattern_id} not found")
            return {
                "matched": False,
                "confidence": 0.0,
                "details": {"error": "Pattern not found"}
            }
        
        pattern = self.patterns[pattern_id]
        entity_type = pattern.get("entity_type")
        detection_query = pattern.get("detection_query")
        
        if not detection_query:
            logger.warning(f"No detection query found for pattern {pattern_id}")
            return {
                "matched": False,
                "confidence": 0.0,
                "details": {"error": "No detection query"}
            }
        
        # Use provided Neo4j client or create a new one
        client = neo4j_client or Neo4jClient()
        
        try:
            # Prepare parameters based on entity type
            params = {}
            if entity_type == DataEntityType.ADDRESS.value:
                params["address"] = entity_id
            elif entity_type == DataEntityType.TRANSACTION.value:
                params["tx_hash"] = entity_id
            else:
                params["entity_id"] = entity_id
            
            # Execute detection query
            results = client.execute_query(detection_query, params)
            
            # Process results
            if not results:
                return {
                    "matched": False,
                    "confidence": 0.0,
                    "pattern_id": pattern_id,
                    "pattern_name": pattern.get("name"),
                    "details": {}
                }
            
            # Extract matching information
            is_suspicious = results[0].get("is_suspicious", False)
            confidence = results[0].get("confidence", 0.0)
            
            # Determine severity based on rules
            severity = self._determine_severity(pattern, results[0])
            
            return {
                "matched": is_suspicious,
                "confidence": confidence,
                "severity": severity,
                "pattern_id": pattern_id,
                "pattern_name": pattern.get("name"),
                "pattern_description": pattern.get("description"),
                "details": results[0],
                "related_entities": results[:10]  # Limit to 10 related entities
            }
        except Exception as e:
            logger.error(f"Error matching pattern {pattern_id}: {e}")
            return {
                "matched": False,
                "confidence": 0.0,
                "pattern_id": pattern_id,
                "pattern_name": pattern.get("name"),
                "details": {"error": str(e)}
            }
    
    def _determine_severity(self, pattern: Dict[str, Any], result: Dict[str, Any]) -> str:
        """
        Determine the severity of a pattern match based on rules.
        
        Args:
            pattern: Pattern definition
            result: Match result
            
        Returns:
            Severity level
        """
        severity_rules = pattern.get("severity_rules", {})
        
        # Start with lowest severity
        severity = AnomalySeverity.LOW.value
        
        # Check each severity level
        for level, rules in severity_rules.items():
            # Check if all rules are satisfied
            if all(result.get(key, 0) >= value for key, value in rules.items()):
                severity = level
        
        return severity
    
    def add_pattern(self, pattern_id: str, pattern_definition: Dict[str, Any]) -> bool:
        """
        Add or update a fraud pattern.
        
        Args:
            pattern_id: ID of the pattern
            pattern_definition: Pattern definition
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Validate pattern definition
            required_fields = ["name", "description", "entity_type", "detection_query"]
            for field in required_fields:
                if field not in pattern_definition:
                    logger.error(f"Missing required field {field} in pattern definition")
                    return False
            
            # Update patterns dictionary
            self.patterns[pattern_id] = pattern_definition
            
            # Update patterns in Redis
            patterns_key = "anomaly:patterns"
            self.redis_client.set(
                key=patterns_key,
                value=self.patterns,
                ttl_seconds=None,  # No expiration
                db=RedisDb.CACHE,
                format=SerializationFormat.JSON
            )
            
            logger.info(f"Added/updated pattern {pattern_id}: {pattern_definition['name']}")
            return True
        except Exception as e:
            logger.error(f"Error adding pattern {pattern_id}: {e}")
            return False
    
    def get_patterns(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all available fraud patterns.
        
        Returns:
            Dictionary of patterns
        """
        return self.patterns
    
    def delete_pattern(self, pattern_id: str) -> bool:
        """
        Delete a fraud pattern.
        
        Args:
            pattern_id: ID of the pattern to delete
            
        Returns:
            True if successful, False otherwise
        """
        if pattern_id not in self.patterns:
            logger.warning(f"Pattern {pattern_id} not found")
            return False
        
        try:
            # Remove from patterns dictionary
            del self.patterns[pattern_id]
            
            # Update patterns in Redis
            patterns_key = "anomaly:patterns"
            self.redis_client.set(
                key=patterns_key,
                value=self.patterns,
                ttl_seconds=None,  # No expiration
                db=RedisDb.CACHE,
                format=SerializationFormat.JSON
            )
            
            logger.info(f"Deleted pattern {pattern_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting pattern {pattern_id}: {e}")
            return False


# --- Main Service ---

class AnomalyDetectionService:
    """
    Main service for anomaly detection across different data types and methods.
    """
    
    def __init__(
        self,
        redis_client: Optional[RedisClient] = None,
        neo4j_client: Optional[Neo4jClient] = None
    ):
        """
        Initialize the anomaly detection service.
        
        Args:
            redis_client: Optional RedisClient instance
            neo4j_client: Optional Neo4jClient instance
        """
        self.redis_client = redis_client or RedisClient()
        self.neo4j_client = neo4j_client or Neo4jClient()
        self.statistical_detector = StatisticalDetector()
        self.graph_detector = GraphBasedDetector(neo4j_client=self.neo4j_client)
        self.pattern_matcher = PatternMatcher(redis_client=self.redis_client)
        self.detection_strategies = self._load_strategies()
        
        logger.info("AnomalyDetectionService initialized")
    
    def _load_strategies(self) -> Dict[str, DetectionStrategy]:
        """
        Load detection strategies from Redis or create defaults.
        
        Returns:
            Dictionary of detection strategies
        """
        strategies_key = "anomaly:strategies"
        strategies_data = self.redis_client.get(
            key=strategies_key,
            db=RedisDb.CACHE,
            format=SerializationFormat.JSON
        )
        
        if strategies_data:
            strategies = {s["id"]: DetectionStrategy(**s) for s in strategies_data}
            logger.info(f"Loaded {len(strategies)} detection strategies from Redis")
            return strategies
        
        # Default strategies if none found in Redis
        default_strategies = {
            "large_transaction": DetectionStrategy(
                name="Large Transaction Detection",
                description="Detects unusually large transactions",
                method=DetectionMethod.STATISTICAL,
                entity_types=[DataEntityType.TRANSACTION],
                threshold_config={"z_score_threshold": 3.0, "min_transaction_value": 10000},
                parameters={"lookback_days": 30, "min_samples": 100}
            ),
            "wash_trading": DetectionStrategy(
                name="Wash Trading Detection",
                description="Detects artificial trading between related accounts",
                method=DetectionMethod.PATTERN_MATCHING,
                entity_types=[DataEntityType.ADDRESS],
                threshold_config={"confidence_threshold": 0.7},
                parameters={"pattern_id": "wash_trading"}
            ),
            "unusual_subgraph": DetectionStrategy(
                name="Unusual Subgraph Detection",
                description="Detects unusual network structures",
                method=DetectionMethod.GRAPH_NEURAL_NETWORK,
                entity_types=[DataEntityType.ADDRESS, DataEntityType.SUBGRAPH],
                threshold_config={"unusualness_threshold": 0.6},
                parameters={"depth": 2}
            ),
            "round_amount": DetectionStrategy(
                name="Round Amount Detection",
                description="Detects suspiciously round transaction amounts",
                method=DetectionMethod.RULE_BASED,
                entity_types=[DataEntityType.TRANSACTION],
                threshold_config={"confidence_threshold": 0.5},
                parameters={"pattern_id": "round_amount"}
            ),
            "layering": DetectionStrategy(
                name="Layering Detection",
                description="Detects funds passing through intermediaries",
                method=DetectionMethod.PATTERN_MATCHING,
                entity_types=[DataEntityType.ADDRESS],
                threshold_config={"confidence_threshold": 0.7},
                parameters={"pattern_id": "layering"}
            ),
            "community_outlier": DetectionStrategy(
                name="Community Outlier Detection",
                description="Detects entities that behave differently from their community",
                method=DetectionMethod.COMMUNITY_DETECTION,
                entity_types=[DataEntityType.ADDRESS],
                threshold_config={"outlier_threshold": 0.7},
                parameters={"community_depth": 2}
            )
        }
        
        # Convert to dictionary
        strategies = {s.id: s for s in default_strategies.values()}
        
        # Store in Redis
        self.redis_client.set(
            key=strategies_key,
            value=[s.dict() for s in strategies.values()],
            ttl_seconds=None,  # No expiration
            db=RedisDb.CACHE,
            format=SerializationFormat.JSON
        )
        
        logger.info(f"Created {len(strategies)} default detection strategies")
        return strategies
    
    async def detect_anomalies(
        self,
        entity_id: str,
        entity_type: DataEntityType,
        strategies: Optional[List[str]] = None,
        create_evidence: bool = True
    ) -> List[AnomalyDetectionResult]:
        """
        Detect anomalies for a specific entity using selected strategies.
        
        Args:
            entity_id: ID of the entity to analyze
            entity_type: Type of entity
            strategies: Optional list of strategy IDs to use (uses all applicable if None)
            create_evidence: Whether to create evidence for detected anomalies
            
        Returns:
            List of anomaly detection results
        """
        # Select strategies to use
        selected_strategies = []
        if strategies:
            # Use specified strategies if they exist and are applicable
            for strategy_id in strategies:
                if strategy_id in self.detection_strategies:
                    strategy = self.detection_strategies[strategy_id]
                    if entity_type in strategy.entity_types and strategy.enabled:
                        selected_strategies.append(strategy)
        else:
            # Use all applicable strategies
            for strategy in self.detection_strategies.values():
                if entity_type in strategy.entity_types and strategy.enabled:
                    selected_strategies.append(strategy)
        
        if not selected_strategies:
            logger.warning(f"No applicable detection strategies found for {entity_type.value} {entity_id}")
            return []
        
        # Run detection for each strategy
        results = []
        for strategy in selected_strategies:
            try:
                # Run detection based on method
                detection_result = await self._run_detection(entity_id, entity_type, strategy)
                
                if detection_result:
                    # Create evidence if requested and anomaly detected
                    if create_evidence and detection_result.score >= detection_result.threshold:
                        evidence_id = await self._create_evidence(detection_result)
                        detection_result.evidence_id = evidence_id
                    
                    # Add to results
                    results.append(detection_result)
                    
                    # Track metrics
                    BusinessMetrics.record_fraud_detection(
                        detection_type=detection_result.anomaly_type.value,
                        confidence_level=detection_result.severity.value,
                    )
                    
                    # Publish event
                    publish_event(
                        event_type="anomaly.detected",
                        data={
                            "anomaly_id": detection_result.id,
                            "entity_id": entity_id,
                            "entity_type": entity_type.value,
                            "anomaly_type": detection_result.anomaly_type.value,
                            "severity": detection_result.severity.value,
                            "confidence": detection_result.confidence,
                            "score": detection_result.score,
                            "threshold": detection_result.threshold,
                            "detection_method": detection_result.detection_method.value,
                            "strategy_id": detection_result.strategy_id,
                            "evidence_id": detection_result.evidence_id,
                            "timestamp": detection_result.detection_time.isoformat(),
                        },
                        priority=EventPriority.HIGH if detection_result.severity in [AnomalySeverity.HIGH, AnomalySeverity.CRITICAL] else EventPriority.NORMAL,
                    )
            except Exception as e:
                logger.error(f"Error running detection strategy {strategy.id}: {e}")
        
        # Store results in Redis
        if results:
            await self._store_detection_results(results)
        
        return results
    
    async def _run_detection(
        self,
        entity_id: str,
        entity_type: DataEntityType,
        strategy: DetectionStrategy
    ) -> Optional[AnomalyDetectionResult]:
        """
        Run detection for a specific entity using a strategy.
        
        Args:
            entity_id: ID of the entity to analyze
            entity_type: Type of entity
            strategy: Detection strategy to use
            
        Returns:
            Anomaly detection result or None if no anomaly detected
        """
        method = strategy.method
        parameters = strategy.parameters
        threshold_config = strategy.threshold_config
        
        # Run appropriate detection method
        if method == DetectionMethod.STATISTICAL:
            return await self._run_statistical_detection(entity_id, entity_type, strategy)
        elif method == DetectionMethod.GRAPH_NEURAL_NETWORK:
            return await self._run_graph_detection(entity_id, entity_type, strategy)
        elif method == DetectionMethod.PATTERN_MATCHING:
            return await self._run_pattern_matching(entity_id, entity_type, strategy)
        elif method == DetectionMethod.RULE_BASED:
            return await self._run_rule_based_detection(entity_id, entity_type, strategy)
        elif method == DetectionMethod.COMMUNITY_DETECTION:
            return await self._run_community_detection(entity_id, entity_type, strategy)
        else:
            logger.warning(f"Unsupported detection method: {method}")
            return None
    
    async def _run_statistical_detection(
        self,
        entity_id: str,
        entity_type: DataEntityType,
        strategy: DetectionStrategy
    ) -> Optional[AnomalyDetectionResult]:
        """
        Run statistical detection.
        
        Args:
            entity_id: ID of the entity to analyze
            entity_type: Type of entity
            strategy: Detection strategy to use
            
        Returns:
            Anomaly detection result or None if no anomaly detected
        """
        # Extract parameters
        lookback_days = strategy.parameters.get("lookback_days", 30)
        min_samples = strategy.parameters.get("min_samples", 100)
        z_score_threshold = strategy.threshold_config.get("z_score_threshold", 3.0)
        min_transaction_value = strategy.threshold_config.get("min_transaction_value", 0)
        
        # Query to get historical data
        if entity_type == DataEntityType.TRANSACTION:
            # For transaction, compare with similar transactions
            query = """
            MATCH (tx:Transaction {hash: $tx_hash})
            MATCH (similar:Transaction)
            WHERE similar.timestamp >= datetime() - duration('P30D')
              AND similar.value >= $min_value
            RETURN similar.value AS value
            LIMIT 1000
            """
            params = {"tx_hash": entity_id, "min_value": min_transaction_value}
            
            # Get transaction details
            tx_query = "MATCH (tx:Transaction {hash: $tx_hash}) RETURN tx.value AS value"
            tx_result = self.neo4j_client.execute_query(tx_query, {"tx_hash": entity_id})
            if not tx_result:
                logger.warning(f"Transaction {entity_id} not found")
                return None
            
            current_value = tx_result[0]["value"]
            
        elif entity_type == DataEntityType.ADDRESS:
            # For address, analyze transaction values
            query = """
            MATCH (a:Address {address: $address})-[tx:TRANSFERRED]->()
            WHERE tx.timestamp >= datetime() - duration('P30D')
              AND tx.value >= $min_value
            RETURN tx.value AS value
            LIMIT 1000
            """
            params = {"address": entity_id, "min_value": min_transaction_value}
            
            # Get latest transaction
            latest_tx_query = """
            MATCH (a:Address {address: $address})-[tx:TRANSFERRED]->()
            RETURN tx.value AS value
            ORDER BY tx.timestamp DESC
            LIMIT 1
            """
            latest_tx = self.neo4j_client.execute_query(latest_tx_query, {"address": entity_id})
            if not latest_tx:
                logger.warning(f"No transactions found for address {entity_id}")
                return None
            
            current_value = latest_tx[0]["value"]
        else:
            logger.warning(f"Statistical detection not supported for entity type {entity_type}")
            return None
        
        # Execute query
        results = self.neo4j_client.execute_query(query, params)
        
        # Extract values
        values = [r["value"] for r in results]
        
        # Check if we have enough samples
        if len(values) < min_samples:
            logger.warning(f"Not enough samples for statistical detection: {len(values)} < {min_samples}")
            return None
        
        # Detect outliers using Z-score
        outliers = self.statistical_detector.z_score_detection(
            values=values,
            threshold=z_score_threshold,
            min_samples=min_samples
        )
        
        # Calculate Z-score for current value
        values_array = np.array(values, dtype=float)
        mean = np.mean(values_array)
        std = np.std(values_array)
        
        if std == 0:
            logger.warning("Standard deviation is zero, cannot compute z-score")
            return None
        
        current_z_score = abs((current_value - mean) / std)
        
        # Check if current value is an outlier
        is_outlier = current_z_score > z_score_threshold
        
        if not is_outlier:
            logger.debug(f"No statistical anomaly detected for {entity_type.value} {entity_id}")
            return None
        
        # Determine severity based on Z-score
        severity = AnomalySeverity.LOW
        if current_z_score > z_score_threshold * 3:
            severity = AnomalySeverity.CRITICAL
        elif current_z_score > z_score_threshold * 2:
            severity = AnomalySeverity.HIGH
        elif current_z_score > z_score_threshold * 1.5:
            severity = AnomalySeverity.MEDIUM
        
        # Calculate confidence based on Z-score and sample size
        confidence = min(0.5 + (current_z_score / (z_score_threshold * 4)) * 0.5, 0.95)
        
        # Create detection result
        return AnomalyDetectionResult(
            anomaly_type=AnomalyType.STATISTICAL_OUTLIER,
            severity=severity,
            confidence=confidence,
            entity_type=entity_type,
            entity_id=entity_id,
            detection_method=DetectionMethod.STATISTICAL,
            strategy_id=strategy.id,
            score=current_z_score,
            threshold=z_score_threshold,
            details={
                "current_value": current_value,
                "mean_value": float(mean),
                "std_dev": float(std),
                "z_score": float(current_z_score),
                "sample_size": len(values),
                "outlier_count": len(outliers),
                "percentile": float(np.percentile(values_array, 100 * (1 - (len(outliers) / len(values)))))
            }
        )
    
    async def _run_graph_detection(
        self,
        entity_id: str,
        entity_type: DataEntityType,
        strategy: DetectionStrategy
    ) -> Optional[AnomalyDetectionResult]:
        """
        Run graph-based detection.
        
        Args:
            entity_id: ID of the entity to analyze
            entity_type: Type of entity
            strategy: Detection strategy to use
            
        Returns:
            Anomaly detection result or None if no anomaly detected
        """
        # Extract parameters
        depth = strategy.parameters.get("depth", 2)
        unusualness_threshold = strategy.threshold_config.get("unusualness_threshold", 0.6)
        
        # Run detection
        result = self.graph_detector.detect_unusual_subgraph(
            center_node_id=entity_id,
            depth=depth,
            threshold=unusualness_threshold
        )
        
        # Check if anomaly detected
        if not result.get("detected", False):
            logger.debug(f"No graph anomaly detected for {entity_type.value} {entity_id}")
            return None
        
        # Extract metrics
        score = result.get("score", 0.0)
        node_count = result.get("node_count", 0)
        relationship_count = result.get("relationship_count", 0)
        density = result.get("density", 0.0)
        high_degree_nodes = result.get("high_degree_nodes", 0)
        high_value_transactions = result.get("high_value_transactions", 0)
        sample_node_ids = result.get("sample_node_ids", [])
        
        # Determine severity based on score
        severity = AnomalySeverity.LOW
        if score > unusualness_threshold * 1.5:
            severity = AnomalySeverity.CRITICAL
        elif score > unusualness_threshold * 1.3:
            severity = AnomalySeverity.HIGH
        elif score > unusualness_threshold * 1.1:
            severity = AnomalySeverity.MEDIUM
        
        # Calculate confidence based on score and subgraph size
        confidence = min(0.5 + (score / (unusualness_threshold * 2)) * 0.5, 0.95)
        
        # Create related entities
        related_entities = []
        for node_id in sample_node_ids:
            related_entities.append({
                "entity_id": node_id,
                "entity_type": "node",
                "relationship": "part_of_subgraph"
            })
        
        # Create detection result
        return AnomalyDetectionResult(
            anomaly_type=AnomalyType.GRAPH_STRUCTURE,
            severity=severity,
            confidence=confidence,
            entity_type=entity_type,
            entity_id=entity_id,
            detection_method=DetectionMethod.GRAPH_NEURAL_NETWORK,
            strategy_id=strategy.id,
            score=score,
            threshold=unusualness_threshold,
            details={
                "node_count": node_count,
                "relationship_count": relationship_count,
                "density": density,
                "high_degree_nodes": high_degree_nodes,
                "high_value_transactions": high_value_transactions,
                "sample_node_ids": sample_node_ids
            },
            related_entities=related_entities
        )
    
    async def _run_pattern_matching(
        self,
        entity_id: str,
        entity_type: DataEntityType,
        strategy: DetectionStrategy
    ) -> Optional[AnomalyDetectionResult]:
        """
        Run pattern matching detection.
        
        Args:
            entity_id: ID of the entity to analyze
            entity_type: Type of entity
            strategy: Detection strategy to use
            
        Returns:
            Anomaly detection result or None if no anomaly detected
        """
        # Extract parameters
        pattern_id = strategy.parameters.get("pattern_id")
        confidence_threshold = strategy.threshold_config.get("confidence_threshold", 0.7)
        
        if not pattern_id:
            logger.warning(f"No pattern ID specified for strategy {strategy.id}")
            return None
        
        # Run pattern matching
        result = self.pattern_matcher.match_pattern(
            pattern_id=pattern_id,
            entity_id=entity_id,
            neo4j_client=self.neo4j_client
        )
        
        # Check if pattern matched
        if not result.get("matched", False):
            logger.debug(f"No pattern match for {entity_type.value} {entity_id} with pattern {pattern_id}")
            return None
        
        # Extract metrics
        confidence = result.get("confidence", 0.0)
        severity = result.get("severity", AnomalySeverity.LOW.value)
        pattern_name = result.get("pattern_name", "Unknown Pattern")
        pattern_description = result.get("pattern_description", "")
        details = result.get("details", {})
        related_entities = result.get("related_entities", [])
        
        # Check if confidence meets threshold
        if confidence < confidence_threshold:
            logger.debug(f"Pattern match confidence too low: {confidence} < {confidence_threshold}")
            return None
        
        # Create detection result
        return AnomalyDetectionResult(
            anomaly_type=AnomalyType.PATTERN_MATCH,
            severity=AnomalySeverity(severity),
            confidence=confidence,
            entity_type=entity_type,
            entity_id=entity_id,
            detection_method=DetectionMethod.PATTERN_MATCHING,
            strategy_id=strategy.id,
            score=confidence,
            threshold=confidence_threshold,
            details={
                "pattern_id": pattern_id,
                "pattern_name": pattern_name,
                "pattern_description": pattern_description,
                **details
            },
            related_entities=[
                {
                    "entity_id": entity.get("id", "unknown"),
                    "entity_type": entity.get("type", "unknown"),
                    "relationship": "related_to_pattern"
                }
                for entity in related_entities if isinstance(entity, dict)
            ]
        )
    
    async def _run_rule_based_detection(
        self,
        entity_id: str,
        entity_type: DataEntityType,
        strategy: DetectionStrategy
    ) -> Optional[AnomalyDetectionResult]:
        """
        Run rule-based detection.
        
        Args:
            entity_id: ID of the entity to analyze
            entity_type: Type of entity
            strategy: Detection strategy to use
            
        Returns:
            Anomaly detection result or None if no anomaly detected
        """
        # For now, rule-based detection uses pattern matching
        # This can be extended with more sophisticated rule engines
        pattern_id = strategy.parameters.get("pattern_id")
        if pattern_id:
            return await self._run_pattern_matching(entity_id, entity_type, strategy)
        
        logger.warning(f"No pattern ID specified for rule-based strategy {strategy.id}")
        return None
    
    async def _run_community_detection(
        self,
        entity_id: str,
        entity_type: DataEntityType,
        strategy: DetectionStrategy
    ) -> Optional[AnomalyDetectionResult]:
        """
        Run community outlier detection.
        
        Args:
            entity_id: ID of the entity to analyze
            entity_type: Type of entity
            strategy: Detection strategy to use
            
        Returns:
            Anomaly detection result or None if no anomaly detected
        """
        # Extract parameters
        community_depth = strategy.parameters.get("community_depth", 2)
        outlier_threshold = strategy.threshold_config.get("outlier_threshold", 0.7)
        
        # Community detection query
        community_query = f"""
        MATCH (center {{id: $entity_id}})
        
        // Find the community around the center node
        CALL apoc.path.subgraphNodes(center, {{maxLevel: {community_depth}}})
        YIELD node
        
        // Calculate community metrics
        WITH collect(node) AS community_nodes
        
        // For each node, calculate how different it is from the community average
        UNWIND community_nodes AS node
        MATCH (node)-[r]-()
        WITH node, count(r) AS connection_count, size(community_nodes) AS community_size,
             avg(r.value) AS avg_value, stddev(r.value) AS std_value
        
        // Calculate outlier score based on connection patterns and transaction values
        WITH node, connection_count, community_size,
             CASE WHEN std_value > 0 THEN abs(avg_value - apoc.node.degree(node)) / std_value ELSE 0 END AS value_z_score
        
        // Return the node and its outlier metrics
        WHERE node.id = $entity_id
        RETURN node.id AS node_id,
               connection_count,
               community_size,
               value_z_score,
               (value_z_score * 0.7 + (toFloat(connection_count) / community_size) * 0.3) AS outlier_score
        """
        
        # Execute query
        params = {"entity_id": entity_id}
        results = self.neo4j_client.execute_query(community_query, params)
        
        if not results:
            logger.warning(f"No community data found for {entity_type.value} {entity_id}")
            return None
        
        # Extract metrics
        community_size = results[0].get("community_size", 0)
        connection_count = results[0].get("connection_count", 0)
        value_z_score = results[0].get("value_z_score", 0.0)
        outlier_score = results[0].get("outlier_score", 0.0)
        
        # Check if outlier
        if outlier_score < outlier_threshold:
            logger.debug(f"No community outlier detected for {entity_type.value} {entity_id}")
            return None
        
        # Determine severity based on outlier score
        severity = AnomalySeverity.LOW
        if outlier_score > outlier_threshold * 1.5:
            severity = AnomalySeverity.CRITICAL
        elif outlier_score > outlier_threshold * 1.3:
            severity = AnomalySeverity.HIGH
        elif outlier_score > outlier_threshold * 1.1:
            severity = AnomalySeverity.MEDIUM
        
        # Calculate confidence based on outlier score and community size
        confidence = min(0.5 + (outlier_score / (outlier_threshold * 2)) * 0.5, 0.95)
        
        # Create detection result
        return AnomalyDetectionResult(
            anomaly_type=AnomalyType.COMMUNITY_OUTLIER,
            severity=severity,
            confidence=confidence,
            entity_type=entity_type,
            entity_id=entity_id,
            detection_method=DetectionMethod.COMMUNITY_DETECTION,
            strategy_id=strategy.id,
            score=outlier_score,
            threshold=outlier_threshold,
            details={
                "community_size": community_size,
                "connection_count": connection_count,
                "value_z_score": float(value_z_score),
                "outlier_score": float(outlier_score)
            }
        )
    
    async def _create_evidence(self, detection_result: AnomalyDetectionResult) -> Optional[str]:
        """
        Create evidence for a detected anomaly.
        
        Args:
            detection_result: Anomaly detection result
            
        Returns:
            Evidence ID or None if creation failed
        """
        try:
            # Create evidence bundle
            bundle = create_evidence_bundle(
                narrative=f"Anomaly Detection: {detection_result.anomaly_type.value} for {detection_result.entity_type.value} {detection_result.entity_id}",
                metadata={
                    "anomaly_id": detection_result.id,
                    "detection_time": detection_result.detection_time.isoformat(),
                    "detection_method": detection_result.detection_method.value,
                    "strategy_id": detection_result.strategy_id
                }
            )
            
            # Create anomaly evidence
            evidence = AnomalyEvidence(
                anomaly_type=detection_result.anomaly_type.value,
                severity=detection_result.severity.value,
                affected_entities=[detection_result.entity_id] + [
                    e.get("entity_id") for e in detection_result.related_entities
                    if isinstance(e, dict) and "entity_id" in e
                ],
                description=f"{detection_result.severity.value.capitalize()} {detection_result.anomaly_type.value} detected for {detection_result.entity_type.value} {detection_result.entity_id}",
                source=EvidenceSource.SYSTEM,
                confidence=detection_result.confidence,
                raw_data=detection_result.to_dict()
            )
            
            # Add to bundle
            bundle.add_evidence(evidence)
            
            # Add related entities as evidence
            for entity in detection_result.related_entities:
                if isinstance(entity, dict) and "entity_id" in entity:
                    bundle.add_raw_data(
                        data=entity,
                        description=f"Related entity: {entity.get('entity_id')}"
                    )
            
            # Synthesize narrative
            bundle.synthesize_narrative()
            
            logger.info(f"Created evidence bundle for anomaly {detection_result.id}")
            return evidence.id
        except Exception as e:
            logger.error(f"Error creating evidence for anomaly {detection_result.id}: {e}")
            return None
    
    async def _store_detection_results(self, results: List[AnomalyDetectionResult]) -> None:
        """
        Store detection results in Redis.
        
        Args:
            results: List of detection results
        """
        try:
            # Store each result individually
            for result in results:
                result_key = f"anomaly:result:{result.id}"
                self.redis_client.set(
                    key=result_key,
                    value=result.to_dict(),
                    ttl_seconds=86400 * 30,  # 30 days
                    db=RedisDb.CACHE,
                    format=SerializationFormat.JSON
                )
            
            # Store IDs in a list for each entity
            for result in results:
                entity_key = f"anomaly:entity:{result.entity_type.value}:{result.entity_id}"
                entity_results = self.redis_client.get(
                    key=entity_key,
                    db=RedisDb.CACHE,
                    format=SerializationFormat.JSON,
                    default=[]
                )
                entity_results.append(result.id)
                self.redis_client.set(
                    key=entity_key,
                    value=entity_results,
                    ttl_seconds=86400 * 30,  # 30 days
                    db=RedisDb.CACHE,
                    format=SerializationFormat.JSON
                )
            
            logger.info(f"Stored {len(results)} detection results in Redis")
        except Exception as e:
            logger.error(f"Error storing detection results: {e}")
    
    async def get_detection_result(self, result_id: str) -> Optional[AnomalyDetectionResult]:
        """
        Get a detection result by ID.
        
        Args:
            result_id: Result ID
            
        Returns:
            Detection result or None if not found
        """
        try:
            result_key = f"anomaly:result:{result_id}"
            result_data = self.redis_client.get(
                key=result_key,
                db=RedisDb.CACHE,
                format=SerializationFormat.JSON
            )
            
            if not result_data:
                logger.warning(f"Detection result {result_id} not found")
                return None
            
            return AnomalyDetectionResult.from_dict(result_data)
        except Exception as e:
            logger.error(f"Error getting detection result {result_id}: {e}")
            return None
    
    async def get_entity_results(
        self,
        entity_id: str,
        entity_type: DataEntityType,
        limit: int = 100
    ) -> List[AnomalyDetectionResult]:
        """
        Get detection results for a specific entity.
        
        Args:
            entity_id: Entity ID
            entity_type: Entity type
            limit: Maximum number of results to return
            
        Returns:
            List of detection results
        """
        try:
            entity_key = f"anomaly:entity:{entity_type.value}:{entity_id}"
            result_ids = self.redis_client.get(
                key=entity_key,
                db=RedisDb.CACHE,
                format=SerializationFormat.JSON,
                default=[]
            )
            
            results = []
            for result_id in result_ids[:limit]:
                result = await self.get_detection_result(result_id)
                if result:
                    results.append(result)
            
            return results
        except Exception as e:
            logger.error(f"Error getting entity results for {entity_type.value} {entity_id}: {e}")
            return []
    
    async def create_alert(
        self,
        anomaly_id: str,
        title: Optional[str] = None,
        description: Optional[str] = None,
        severity: Optional[AnomalySeverity] = None,
        tags: Optional[List[str]] = None
    ) -> Optional[AnomalyAlert]:
        """
        Create an alert from an anomaly detection result.
        
        Args:
            anomaly_id: Anomaly detection result ID
            title: Optional alert title
            description: Optional alert description
            severity: Optional alert severity
            tags: Optional alert tags
            
        Returns:
            Created alert or None if creation failed
        """
        try:
            # Get detection result
            detection_result = await self.get_detection_result(anomaly_id)
            if not detection_result:
                logger.warning(f"Cannot create alert: Detection result {anomaly_id} not found")
                return None
            
            # Create alert
            alert = AnomalyAlert(
                anomaly_id=anomaly_id,
                title=title or f"{detection_result.severity.value.capitalize()} {detection_result.anomaly_type.value} detected",
                description=description or f"{detection_result.severity.value.capitalize()} {detection_result.anomaly_type.value} detected for {detection_result.entity_type.value} {detection_result.entity_id}",
                severity=severity or detection_result.severity,
                entity_type=detection_result.entity_type,
                entity_id=detection_result.entity_id,
                evidence_id=detection_result.evidence_id,
                tags=tags or [detection_result.anomaly_type.value, detection_result.severity.value]
            )
            
            # Store alert in Redis
            alert_key = f"anomaly:alert:{alert.id}"
            self.redis_client.set(
                key=alert_key,
                value=alert.to_dict(),
                ttl_seconds=86400 * 30,  # 30 days
                db=RedisDb.CACHE,
                format=SerializationFormat.JSON
            )
            
            # Add to alert list
            alerts_key = "anomaly:alerts"
            alerts = self.redis_client.get(
                key=alerts_key,
                db=RedisDb.CACHE,
                format=SerializationFormat.JSON,
                default=[]
            )
            alerts.append(alert.id)
            self.redis_client.set(
                key=alerts_key,
                value=alerts,
                ttl_seconds=None,  # No expiration
                db=RedisDb.CACHE,
                format=SerializationFormat.JSON
            )
            
            # Publish event
            publish_event(
                event_type="anomaly.alert_created",
                data={
                    "alert_id": alert.id,
                    "anomaly_id": anomaly_id,
                    "title": alert.title,
                    "severity": alert.severity.value,
                    "entity_type": alert.entity_type.value,
                    "entity_id": alert.entity_id,
                    "timestamp": alert.created_at.isoformat(),
                },
                priority=EventPriority.HIGH if alert.severity in [AnomalySeverity.HIGH, AnomalySeverity.CRITICAL] else EventPriority.NORMAL,
            )
            
            logger.info(f"Created alert {alert.id} for anomaly {anomaly_id}")
            return alert
        except Exception as e:
            logger.error(f"Error creating alert for anomaly {anomaly_id}: {e}")
            return None
    
    async def get_alert(self, alert_id: str) -> Optional[AnomalyAlert]:
        """
        Get an alert by ID.
        
        Args:
            alert_id: Alert ID
            
        Returns:
            Alert or None if not found
        """
        try:
            alert_key = f"anomaly:alert:{alert_id}"
            alert_data = self.redis_client.get(
                key=alert_key,
                db=RedisDb.CACHE,
                format=SerializationFormat.JSON
            )
            
            if not alert_data:
                logger.warning(f"Alert {alert_id} not found")
                return None
            
            return AnomalyAlert.from_dict(alert_data)
        except Exception as e:
            logger.error(f"Error getting alert {alert_id}: {e}")
            return None
    
    async def get_alerts(
        self,
        status: Optional[AlertStatus] = None,
        severity: Optional[AnomalySeverity] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[AnomalyAlert]:
        """
        Get alerts with optional filtering.
        
        Args:
            status: Optional status filter
            severity: Optional severity filter
            limit: Maximum number of alerts to return
            offset: Offset for pagination
            
        Returns:
            List of alerts
        """
        try:
            alerts_key = "anomaly:alerts"
            alert_ids = self.redis_client.get(
                key=alerts_key,
                db=RedisDb.CACHE,
                format=SerializationFormat.JSON,
                default=[]
            )
            
            # Apply offset and limit
            alert_ids = alert_ids[offset:offset + limit]
            
            alerts = []
            for alert_id in alert_ids:
                alert = await self.get_alert(alert_id)
                if alert:
                    # Apply filters
                    if status and alert.status != status:
                        continue
                    if severity and alert.severity != severity:
                        continue
                    
                    alerts.append(alert)
            
            return alerts
        except Exception as e:
            logger.error(f"Error getting alerts: {e}")
            return []
    
    async def update_alert_status(
        self,
        alert_id: str,
        status: AlertStatus,
        assigned_to: Optional[str] = None
    ) -> bool:
        """
        Update the status of an alert.
        
        Args:
            alert_id: Alert ID
            status: New status
            assigned_to: Optional user to assign the alert to
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get alert
            alert = await self.get_alert(alert_id)
            if not alert:
                logger.warning(f"Cannot update alert: Alert {alert_id} not found")
                return False
            
            # Update status
            alert.status = status
            alert.updated_at = datetime.now()
            
            # Update assignment if provided
            if assigned_to is not None:
                alert.assigned_to = assigned_to
            
            # Store updated alert
            alert_key = f"anomaly:alert:{alert_id}"
            self.redis_client.set(
                key=alert_key,
                value=alert.to_dict(),
                ttl_seconds=86400 * 30,  # 30 days
                db=RedisDb.CACHE,
                format=SerializationFormat.JSON
            )
            
            # Publish event
            publish_event(
                event_type="anomaly.alert_updated",
                data={
                    "alert_id": alert_id,
                    "status": status.value,
                    "assigned_to": alert.assigned_to,
                    "timestamp": alert.updated_at.isoformat(),
                },
                priority=EventPriority.NORMAL,
            )
            
            logger.info(f"Updated alert {alert_id} status to {status.value}")
            return True
        except Exception as e:
            logger.error(f"Error updating alert {alert_id}: {e}")
            return False
    
    async def get_detection_strategies(self) -> Dict[str, DetectionStrategy]:
        """
        Get all detection strategies.
        
        Returns:
            Dictionary of detection strategies
        """
        return self.detection_strategies
    
    async def get_detection_strategy(self, strategy_id: str) -> Optional[DetectionStrategy]:
        """
        Get a detection strategy by ID.
        
        Args:
            strategy_id: Strategy ID
            
        Returns:
            Detection strategy or None if not found
        """
        return self.detection_strategies.get(strategy_id)
    
    async def add_detection_strategy(self, strategy: DetectionStrategy) -> bool:
        """
        Add or update a detection strategy.
        
        Args:
            strategy: Detection strategy
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Update strategies
