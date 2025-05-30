"""
PatternLibraryTool - A CrewAI tool for managing and converting fraud pattern definitions

This tool allows agents to:
1. Access a library of fraud detection patterns defined in YAML format
2. Search for patterns by various criteria (ID, category, risk level, tags)
3. Convert pattern definitions into executable Neo4j Cypher queries
4. Support both template-based and dynamic Cypher generation

The tool is used primarily by the fraud_pattern_hunter agent to identify
known financial crime patterns in transaction data.
"""

import os
import yaml
import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
from datetime import datetime, timedelta

from crewai_tools import BaseTool
from pydantic import BaseModel, Field

# Configure logging
logger = logging.getLogger(__name__)

# Path to pattern definitions
PATTERNS_DIR = Path(__file__).parent.parent / "patterns"


class PatternSearchParams(BaseModel):
    """Parameters for searching patterns"""
    pattern_id: Optional[str] = Field(None, description="Specific pattern ID to retrieve")
    category: Optional[str] = Field(None, description="Filter by pattern category")
    risk_level: Optional[str] = Field(None, description="Filter by risk level")
    tags: Optional[List[str]] = Field(None, description="Filter by tags (any match)")
    regulatory_implications: Optional[List[str]] = Field(None, description="Filter by regulatory implications")


class PatternConversionParams(BaseModel):
    """Parameters for converting a pattern to Cypher"""
    pattern_id: str = Field(..., description="Pattern ID to convert")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Parameters to substitute in the query")
    use_template: Optional[bool] = Field(True, description="Whether to use the template (True) or generate dynamically (False)")


class PatternLibraryTool(BaseTool):
    """
    Tool for accessing and converting fraud pattern definitions to Cypher queries.
    
    This tool loads pattern definitions from YAML files and provides methods to:
    - List all available patterns
    - Get a specific pattern by ID
    - Search patterns by category, risk level, or tags
    - Convert a pattern to an executable Cypher query
    """
    
    name: str = "PatternLibraryTool"
    description: str = "Access and convert fraud pattern definitions to Cypher queries"
    
    # Cache for loaded patterns
    _patterns_cache: Dict[str, Dict[str, Any]] = {}
    _schema: Dict[str, Any] = {}
    _last_load_time: datetime = datetime.min
    
    def __init__(self, **kwargs):
        """Initialize the PatternLibraryTool"""
        super().__init__(**kwargs)
        self._ensure_patterns_dir()
        self._load_patterns()
    
    def _ensure_patterns_dir(self) -> None:
        """Ensure the patterns directory exists"""
        PATTERNS_DIR.mkdir(parents=True, exist_ok=True)
        
        # Create a README if it doesn't exist
        readme_path = PATTERNS_DIR / "README.md"
        if not readme_path.exists():
            with open(readme_path, "w") as f:
                f.write("# Fraud Pattern Library\n\n")
                f.write("This directory contains YAML definitions of fraud patterns used by the PatternLibraryTool.\n")
                f.write("Each pattern defines a financial crime motif that can be detected in transaction data.\n")
    
    def _load_patterns(self, force: bool = False) -> None:
        """
        Load all pattern definitions from YAML files in the patterns directory
        
        Args:
            force: Force reload even if cache is recent
        """
        # Check if cache is still valid (less than 5 minutes old)
        if not force and self._patterns_cache and (datetime.now() - self._last_load_time < timedelta(minutes=5)):
            return
        
        # Clear existing cache
        self._patterns_cache = {}
        
        # Load schema first
        schema_path = PATTERNS_DIR / "fraud_motifs_schema.yaml"
        if schema_path.exists():
            try:
                with open(schema_path, "r") as f:
                    self._schema = yaml.safe_load(f)
                    logger.info(f"Loaded pattern schema from {schema_path}")
            except Exception as e:
                logger.error(f"Error loading pattern schema: {str(e)}")
                self._schema = {}
        
        # Load all pattern files
        pattern_files = list(PATTERNS_DIR.glob("*.yaml")) + list(PATTERNS_DIR.glob("*.yml"))
        pattern_files = [f for f in pattern_files if f.name != "fraud_motifs_schema.yaml"]
        
        for pattern_file in pattern_files:
            try:
                with open(pattern_file, "r") as f:
                    pattern_data = yaml.safe_load(f)
                    
                    # Handle different file formats
                    if "patterns" in pattern_data:
                        # Multiple patterns in one file
                        for pattern in pattern_data["patterns"]:
                            if "metadata" in pattern and "id" in pattern["metadata"]:
                                pattern_id = pattern["metadata"]["id"]
                                self._patterns_cache[pattern_id] = pattern
                    elif "metadata" in pattern_data and "id" in pattern_data["metadata"]:
                        # Single pattern file
                        pattern_id = pattern_data["metadata"]["id"]
                        self._patterns_cache[pattern_id] = pattern_data
                    else:
                        logger.warning(f"Skipping file {pattern_file} - invalid format")
                        continue
                    
                    logger.info(f"Loaded pattern from {pattern_file}")
            except Exception as e:
                logger.error(f"Error loading pattern file {pattern_file}: {str(e)}")
        
        # Also load example patterns from schema if available
        if self._schema and "example_patterns" in self._schema:
            for pattern in self._schema["example_patterns"]:
                if "metadata" in pattern and "id" in pattern["metadata"]:
                    pattern_id = pattern["metadata"]["id"]
                    if pattern_id not in self._patterns_cache:
                        self._patterns_cache[pattern_id] = pattern
                        logger.info(f"Loaded example pattern {pattern_id} from schema")
        
        self._last_load_time = datetime.now()
        logger.info(f"Loaded {len(self._patterns_cache)} patterns in total")
    
    def _search_patterns(self, params: PatternSearchParams) -> List[Dict[str, Any]]:
        """
        Search patterns based on search parameters
        
        Args:
            params: Search parameters
            
        Returns:
            List of matching patterns
        """
        # Reload patterns if needed
        self._load_patterns()
        
        # If pattern_id is specified, return just that pattern
        if params.pattern_id:
            pattern = self._patterns_cache.get(params.pattern_id)
            return [pattern] if pattern else []
        
        # Apply filters
        results = []
        for pattern_id, pattern in self._patterns_cache.items():
            metadata = pattern.get("metadata", {})
            
            # Filter by category
            if params.category and metadata.get("category") != params.category:
                continue
            
            # Filter by risk level
            if params.risk_level and metadata.get("risk_level") != params.risk_level:
                continue
            
            # Filter by tags (any match)
            if params.tags and metadata.get("tags"):
                if not any(tag in metadata["tags"] for tag in params.tags):
                    continue
            
            # Filter by regulatory implications (any match)
            if params.regulatory_implications and metadata.get("regulatory_implications"):
                if not any(reg in metadata["regulatory_implications"] for reg in params.regulatory_implications):
                    continue
            
            results.append(pattern)
        
        return results
    
    def _get_pattern_summary(self, pattern: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get a summary of a pattern (metadata only)
        
        Args:
            pattern: Full pattern definition
            
        Returns:
            Pattern summary with metadata
        """
        metadata = pattern.get("metadata", {})
        return {
            "id": metadata.get("id", "unknown"),
            "name": metadata.get("name", "Unnamed Pattern"),
            "description": metadata.get("description", ""),
            "category": metadata.get("category", ""),
            "risk_level": metadata.get("risk_level", ""),
            "tags": metadata.get("tags", []),
            "regulatory_implications": metadata.get("regulatory_implications", [])
        }
    
    def _convert_graph_pattern_to_cypher(self, graph_pattern: Dict[str, Any]) -> Tuple[str, str, Dict[str, Any]]:
        """
        Convert a graph pattern definition to Cypher MATCH and WHERE clauses
        
        Args:
            graph_pattern: Graph pattern definition from the pattern
            
        Returns:
            Tuple of (match_clause, where_clause, parameters)
        """
        match_parts = []
        where_parts = []
        parameters = {}
        
        # Process nodes
        if "nodes" in graph_pattern:
            for node in graph_pattern["nodes"]:
                node_id = node.get("id", "n")
                labels = node.get("labels", [])
                properties = node.get("properties", {})
                
                # Build node pattern
                if labels:
                    label_str = ":".join(labels)
                    node_pattern = f"({node_id}:{label_str})"
                else:
                    node_pattern = f"({node_id})"
                
                match_parts.append(node_pattern)
                
                # Process properties as WHERE conditions
                for prop, condition in properties.items():
                    if isinstance(condition, dict):
                        for op, value in condition.items():
                            # Convert operator to Cypher
                            if op == "$eq":
                                where_parts.append(f"{node_id}.{prop} = ${node_id}_{prop}_eq")
                                parameters[f"{node_id}_{prop}_eq"] = value
                            elif op == "$ne":
                                where_parts.append(f"{node_id}.{prop} <> ${node_id}_{prop}_ne")
                                parameters[f"{node_id}_{prop}_ne"] = value
                            elif op == "$gt":
                                where_parts.append(f"{node_id}.{prop} > ${node_id}_{prop}_gt")
                                parameters[f"{node_id}_{prop}_gt"] = value
                            elif op == "$gte":
                                where_parts.append(f"{node_id}.{prop} >= ${node_id}_{prop}_gte")
                                parameters[f"{node_id}_{prop}_gte"] = value
                            elif op == "$lt":
                                where_parts.append(f"{node_id}.{prop} < ${node_id}_{prop}_lt")
                                parameters[f"{node_id}_{prop}_lt"] = value
                            elif op == "$lte":
                                where_parts.append(f"{node_id}.{prop} <= ${node_id}_{prop}_lte")
                                parameters[f"{node_id}_{prop}_lte"] = value
                            elif op == "$in":
                                where_parts.append(f"{node_id}.{prop} IN ${node_id}_{prop}_in")
                                parameters[f"{node_id}_{prop}_in"] = value
                    else:
                        # Simple equality
                        where_parts.append(f"{node_id}.{prop} = ${node_id}_{prop}")
                        parameters[f"{node_id}_{prop}"] = condition
        
        # Process relationships
        if "relationships" in graph_pattern:
            for rel in graph_pattern["relationships"]:
                source = rel.get("source", "a")
                target = rel.get("target", "b")
                rel_type = rel.get("type", "RELATED_TO")
                direction = rel.get("direction", "OUTGOING")
                properties = rel.get("properties", {})
                
                # Build relationship pattern based on direction
                if direction == "OUTGOING":
                    rel_pattern = f"({source})-[:{rel_type}]->({target})"
                elif direction == "INCOMING":
                    rel_pattern = f"({source})<-[:{rel_type}]-({target})"
                else:  # BOTH
                    rel_pattern = f"({source})-[:{rel_type}]-({target})"
                
                match_parts.append(rel_pattern)
                
                # Process properties as WHERE conditions
                for prop, condition in properties.items():
                    rel_id = f"r_{source}_{target}"  # Generate a relationship variable
                    
                    # Update the match pattern to include the relationship variable
                    if direction == "OUTGOING":
                        rel_pattern = f"({source})-[{rel_id}:{rel_type}]->({target})"
                    elif direction == "INCOMING":
                        rel_pattern = f"({source})<-[{rel_id}:{rel_type}]-({target})"
                    else:  # BOTH
                        rel_pattern = f"({source})-[{rel_id}:{rel_type}]-({target})"
                    
                    # Replace the previous match part
                    match_parts[-1] = rel_pattern
                    
                    if isinstance(condition, dict):
                        for op, value in condition.items():
                            # Convert operator to Cypher
                            if op == "$eq":
                                where_parts.append(f"{rel_id}.{prop} = ${rel_id}_{prop}_eq")
                                parameters[f"{rel_id}_{prop}_eq"] = value
                            elif op == "$ne":
                                where_parts.append(f"{rel_id}.{prop} <> ${rel_id}_{prop}_ne")
                                parameters[f"{rel_id}_{prop}_ne"] = value
                            # Add other operators as needed
                    else:
                        # Simple equality
                        where_parts.append(f"{rel_id}.{prop} = ${rel_id}_{prop}")
                        parameters[f"{rel_id}_{prop}"] = condition
        
        # Process path patterns
        if "path_patterns" in graph_pattern:
            for path in graph_pattern["path_patterns"]:
                start_node = path.get("start_node", "a")
                end_node = path.get("end_node", "b")
                rel_types = path.get("relationship_types", [])
                min_length = path.get("min_length", 1)
                max_length = path.get("max_length", None)
                direction = path.get("direction", "OUTGOING")
                
                # Build relationship type string
                rel_type_str = "|".join(rel_types) if rel_types else ""
                
                # Build length constraint
                if min_length == max_length:
                    length_str = f"{{{min_length}}}"
                elif max_length is None:
                    length_str = f"{{{min_length},}}"
                else:
                    length_str = f"{{{min_length},{max_length}}}"
                
                # Build path pattern based on direction
                path_var = f"path_{start_node}_{end_node}"
                if direction == "OUTGOING":
                    if rel_type_str:
                        path_pattern = f"{path_var} = ({start_node})-[:{rel_type_str}{length_str}]->({end_node})"
                    else:
                        path_pattern = f"{path_var} = ({start_node})-[{length_str}]->({end_node})"
                elif direction == "INCOMING":
                    if rel_type_str:
                        path_pattern = f"{path_var} = ({start_node})<-[:{rel_type_str}{length_str}]-({end_node})"
                    else:
                        path_pattern = f"{path_var} = ({start_node})<-[{length_str}]-({end_node})"
                else:  # BOTH
                    if rel_type_str:
                        path_pattern = f"{path_var} = ({start_node})-[:{rel_type_str}{length_str}]-({end_node})"
                    else:
                        path_pattern = f"{path_var} = ({start_node})-[{length_str}]-({end_node})"
                
                match_parts.append(path_pattern)
        
        # Combine into MATCH and WHERE clauses
        match_clause = "MATCH " + ", ".join(match_parts)
        where_clause = "WHERE " + " AND ".join(where_parts) if where_parts else ""
        
        return match_clause, where_clause, parameters
    
    def _process_temporal_constraints(self, constraints: List[Dict[str, Any]]) -> Tuple[List[str], Dict[str, Any]]:
        """
        Process temporal constraints into Cypher WHERE conditions
        
        Args:
            constraints: List of temporal constraint definitions
            
        Returns:
            Tuple of (where_conditions, parameters)
        """
        where_parts = []
        parameters = {}
        
        for constraint in constraints:
            constraint_type = constraint.get("type")
            node_id = constraint.get("node_id", "n")
            property_name = constraint.get("property", "timestamp")
            params = constraint.get("parameters", {})
            
            if constraint_type == "TIME_WINDOW":
                window = params.get("window", "P30D")  # Default 30 days
                param_name = f"{node_id}_{property_name}_window"
                where_parts.append(f"{node_id}.{property_name} > datetime() - duration(${param_name})")
                parameters[param_name] = window
            
            elif constraint_type == "VELOCITY":
                max_duration = params.get("max_duration", "P7D")
                param_name = f"{node_id}_{property_name}_max_duration"
                # This will need to be adapted based on specific query structure
                where_parts.append(f"duration.between(min({node_id}.{property_name}), max({node_id}.{property_name})).days <= ${param_name}_days")
                # Convert ISO duration to days for simplicity
                days_match = re.search(r"P(\d+)D", max_duration)
                days = int(days_match.group(1)) if days_match else 7
                parameters[f"{param_name}_days"] = days
            
            elif constraint_type == "FREQUENCY":
                min_count = params.get("min_count", 3)
                time_period = params.get("time_period", "P30D")
                count_param = f"{node_id}_min_count"
                period_param = f"{node_id}_time_period"
                where_parts.append(f"count({node_id}) >= ${count_param}")
                where_parts.append(f"{node_id}.{property_name} > datetime() - duration(${period_param})")
                parameters[count_param] = min_count
                parameters[period_param] = time_period
            
            # Add other temporal constraint types as needed
        
        return where_parts, parameters
    
    def _process_value_constraints(self, constraints: List[Dict[str, Any]]) -> Tuple[List[str], Dict[str, Any]]:
        """
        Process value constraints into Cypher WHERE conditions
        
        Args:
            constraints: List of value constraint definitions
            
        Returns:
            Tuple of (where_conditions, parameters)
        """
        where_parts = []
        parameters = {}
        
        for constraint in constraints:
            constraint_type = constraint.get("type")
            node_id = constraint.get("node_id", "n")
            property_name = constraint.get("property", "amount")
            params = constraint.get("parameters", {})
            
            if constraint_type == "THRESHOLD":
                if "min" in params:
                    param_name = f"{node_id}_{property_name}_min"
                    where_parts.append(f"{node_id}.{property_name} >= ${param_name}")
                    parameters[param_name] = params["min"]
                
                if "max" in params:
                    param_name = f"{node_id}_{property_name}_max"
                    where_parts.append(f"{node_id}.{property_name} <= ${param_name}")
                    parameters[param_name] = params["max"]
            
            elif constraint_type == "RANGE":
                if "min" in params and "max" in params:
                    min_param = f"{node_id}_{property_name}_range_min"
                    max_param = f"{node_id}_{property_name}_range_max"
                    where_parts.append(f"{node_id}.{property_name} >= ${min_param} AND {node_id}.{property_name} <= ${max_param}")
                    parameters[min_param] = params["min"]
                    parameters[max_param] = params["max"]
            
            elif constraint_type == "STRUCTURING":
                threshold = params.get("threshold", 10000)
                margin = params.get("margin", 0.2)  # 20% below threshold
                min_value = threshold * (1 - margin)
                
                threshold_param = f"{node_id}_{property_name}_threshold"
                min_param = f"{node_id}_{property_name}_min"
                
                where_parts.append(f"{node_id}.{property_name} >= ${min_param} AND {node_id}.{property_name} < ${threshold_param}")
                parameters[threshold_param] = threshold
                parameters[min_param] = min_value
            
            elif constraint_type == "RATIO":
                if "min_ratio" in params:
                    param_name = f"{node_id}_{property_name}_min_ratio"
                    # This will need to be adapted based on specific query structure
                    where_parts.append(f"end_amount >= start_amount * ${param_name}")
                    parameters[param_name] = params["min_ratio"]
            
            # Add other value constraint types as needed
        
        return where_parts, parameters
    
    def _process_aggregation_rules(self, rules: List[Dict[str, Any]]) -> Tuple[str, str, Dict[str, Any]]:
        """
        Process aggregation rules into Cypher WITH and HAVING clauses
        
        Args:
            rules: List of aggregation rule definitions
            
        Returns:
            Tuple of (with_clause, having_clause, parameters)
        """
        with_parts = []
        having_parts = []
        parameters = {}
        
        for rule in rules:
            agg_type = rule.get("type", "COUNT")
            group_by = rule.get("group_by", [])
            having = rule.get("having", {})
            window = rule.get("window", {})
            
            # Build WITH clause
            if group_by:
                group_vars = ", ".join(group_by)
                with_parts.append(group_vars)
            
            # Add aggregation
            if agg_type == "COUNT":
                count_var = "count"
                count_expr = "count(*)"
                with_parts.append(f"{count_expr} as {count_var}")
                
                # Add HAVING conditions
                if "count" in having:
                    condition = having["count"]
                    if isinstance(condition, dict):
                        for op, value in condition.items():
                            if op == "$eq":
                                having_parts.append(f"{count_var} = ${count_var}_eq")
                                parameters[f"{count_var}_eq"] = value
                            elif op == "$gte":
                                having_parts.append(f"{count_var} >= ${count_var}_gte")
                                parameters[f"{count_var}_gte"] = value
                            # Add other operators as needed
            
            elif agg_type == "SUM":
                sum_var = "total"
                sum_expr = "sum(n.amount)"  # This might need to be parameterized
                with_parts.append(f"{sum_expr} as {sum_var}")
                
                # Add HAVING conditions
                if "sum" in having:
                    condition = having["sum"]
                    if isinstance(condition, dict):
                        for op, value in condition.items():
                            if op == "$eq":
                                having_parts.append(f"{sum_var} = ${sum_var}_eq")
                                parameters[f"{sum_var}_eq"] = value
                            elif op == "$gte":
                                having_parts.append(f"{sum_var} >= ${sum_var}_gte")
                                parameters[f"{sum_var}_gte"] = value
                            # Add other operators as needed
            
            # Add other aggregation types as needed
        
        # Combine into WITH and HAVING clauses
        with_clause = "WITH " + ", ".join(with_parts) if with_parts else ""
        having_clause = "HAVING " + " AND ".join(having_parts) if having_parts else ""
        
        return with_clause, having_clause, parameters
    
    def _convert_pattern_to_cypher_dynamic(self, pattern: Dict[str, Any], user_params: Dict[str, Any] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Dynamically convert a pattern to a Cypher query
        
        Args:
            pattern: Pattern definition
            user_params: User-provided parameters to override defaults
            
        Returns:
            Tuple of (cypher_query, parameters)
        """
        detection = pattern.get("detection", {})
        cypher_parts = []
        all_parameters = {}
        
        # Process graph pattern
        if "graph_pattern" in detection:
            match_clause, where_clause, graph_params = self._convert_graph_pattern_to_cypher(detection["graph_pattern"])
            cypher_parts.append(match_clause)
            if where_clause:
                cypher_parts.append(where_clause)
            all_parameters.update(graph_params)
        
        # Process temporal constraints
        if "temporal_constraints" in detection:
            temporal_where, temporal_params = self._process_temporal_constraints(detection["temporal_constraints"])
            if temporal_where:
                if "WHERE" in cypher_parts:
                    # Append to existing WHERE clause
                    where_index = cypher_parts.index("WHERE")
                    cypher_parts[where_index] = cypher_parts[where_index] + " AND " + " AND ".join(temporal_where)
                else:
                    # Add new WHERE clause
                    cypher_parts.append("WHERE " + " AND ".join(temporal_where))
            all_parameters.update(temporal_params)
        
        # Process value constraints
        if "value_constraints" in detection:
            value_where, value_params = self._process_value_constraints(detection["value_constraints"])
            if value_where:
                if "WHERE" in " ".join(cypher_parts):
                    # Append to existing WHERE clause
                    for i, part in enumerate(cypher_parts):
                        if part.startswith("WHERE "):
                            cypher_parts[i] = part + " AND " + " AND ".join(value_where)
                            break
                else:
                    # Add new WHERE clause
                    cypher_parts.append("WHERE " + " AND ".join(value_where))
            all_parameters.update(value_params)
        
        # Process aggregation rules
        if "aggregation_rules" in detection:
            with_clause, having_clause, agg_params = self._process_aggregation_rules(detection["aggregation_rules"])
            if with_clause:
                cypher_parts.append(with_clause)
            if having_clause:
                cypher_parts.append(having_clause)
            all_parameters.update(agg_params)
        
        # Add additional conditions
        if "additional_conditions" in detection:
            additional = detection["additional_conditions"]
            if additional:
                if "WHERE" in " ".join(cypher_parts):
                    # Append to existing WHERE clause
                    for i, part in enumerate(cypher_parts):
                        if part.startswith("WHERE "):
                            cypher_parts[i] = part + " AND " + additional
                            break
                else:
                    # Add new WHERE clause
                    cypher_parts.append("WHERE " + additional)
        
        # Add RETURN clause (simplified for now)
        cypher_parts.append("RETURN *")
        
        # Combine into final query
        cypher_query = "\n".join(cypher_parts)
        
        # Override with user parameters
        if user_params:
            all_parameters.update(user_params)
        
        return cypher_query, all_parameters
    
    def _convert_pattern_to_cypher_template(self, pattern: Dict[str, Any], user_params: Dict[str, Any] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Convert a pattern to a Cypher query using its template
        
        Args:
            pattern: Pattern definition
            user_params: User-provided parameters to override defaults
            
        Returns:
            Tuple of (cypher_query, parameters)
        """
        # Get the template
        cypher_template = pattern.get("cypher_template", "")
        if not cypher_template:
            raise ValueError(f"Pattern {pattern.get('metadata', {}).get('id', 'unknown')} has no cypher_template")
        
        # Prepare default parameters based on pattern definition
        default_params = {}
        
        # Extract parameters from detection section
        detection = pattern.get("detection", {})
        
        # Process temporal constraints for default parameters
        if "temporal_constraints" in detection:
            for constraint in detection["temporal_constraints"]:
                params = constraint.get("parameters", {})
                if "window" in params:
                    default_params["time_window"] = params["window"]
                if "max_duration" in params:
                    # Extract days from ISO duration
                    days_match = re.search(r"P(\d+)D", params["max_duration"])
                    if days_match:
                        default_params["max_days"] = int(days_match.group(1))
        
        # Process value constraints for default parameters
        if "value_constraints" in detection:
            for constraint in detection["value_constraints"]:
                constraint_type = constraint.get("type")
                params = constraint.get("parameters", {})
                
                if constraint_type == "THRESHOLD":
                    if "min" in params:
                        default_params["min_amount"] = params["min"]
                    if "max" in params:
                        default_params["max_amount"] = params["max"]
                    if "threshold" in params:
                        default_params["threshold"] = params["threshold"]
                
                elif constraint_type == "RATIO":
                    if "min_ratio" in params:
                        default_params["min_ratio"] = params["min_ratio"]
        
        # Process aggregation rules for default parameters
        if "aggregation_rules" in detection:
            for rule in detection["aggregation_rules"]:
                having = rule.get("having", {})
                
                if "count" in having and isinstance(having["count"], dict):
                    for op, value in having["count"].items():
                        if op == "$gte":
                            default_params["min_transactions"] = value
                
                if "sum" in having and isinstance(having["sum"], dict):
                    for op, value in having["sum"].items():
                        if op == "$gte":
                            default_params["min_total"] = value
        
        # Override with user parameters
        parameters = {**default_params}
        if user_params:
            parameters.update(user_params)
        
        return cypher_template, parameters
    
    def _convert_pattern_to_cypher(self, pattern_id: str, parameters: Dict[str, Any] = None, use_template: bool = True) -> Dict[str, Any]:
        """
        Convert a pattern to a Cypher query
        
        Args:
            pattern_id: ID of the pattern to convert
            parameters: Parameters to substitute in the query
            use_template: Whether to use the template or generate dynamically
            
        Returns:
            Dictionary with cypher query and parameters
        """
        # Reload patterns if needed
        self._load_patterns()
        
        # Get the pattern
        pattern = self._patterns_cache.get(pattern_id)
        if not pattern:
            return {
                "success": False,
                "error": f"Pattern not found: {pattern_id}"
            }
        
        try:
            if use_template:
                cypher_query, query_params = self._convert_pattern_to_cypher_template(pattern, parameters)
            else:
                cypher_query, query_params = self._convert_pattern_to_cypher_dynamic(pattern, parameters)
            
            return {
                "success": True,
                "pattern_id": pattern_id,
                "pattern_name": pattern.get("metadata", {}).get("name", ""),
                "cypher_query": cypher_query,
                "parameters": query_params,
                "generation_method": "template" if use_template else "dynamic"
            }
        except Exception as e:
            logger.error(f"Error converting pattern {pattern_id} to Cypher: {str(e)}")
            return {
                "success": False,
                "pattern_id": pattern_id,
                "error": str(e)
            }
    
    def _run(self, query: str) -> str:
        """
        Run the tool with the given query
        
        Args:
            query: JSON string with the query parameters
            
        Returns:
            JSON string with the result
        """
        try:
            # Parse the query
            query_data = json.loads(query)
            action = query_data.get("action", "list")
            
            # Handle different actions
            if action == "list":
                # List all patterns
                self._load_patterns(force=True)  # Force reload to get latest
                patterns = []
                for pattern_id, pattern in self._patterns_cache.items():
                    patterns.append(self._get_pattern_summary(pattern))
                
                return json.dumps({
                    "success": True,
                    "count": len(patterns),
                    "patterns": patterns
                }, indent=2)
            
            elif action == "get":
                # Get a specific pattern
                pattern_id = query_data.get("pattern_id")
                if not pattern_id:
                    return json.dumps({
                        "success": False,
                        "error": "Missing pattern_id parameter"
                    })
                
                self._load_patterns()
                pattern = self._patterns_cache.get(pattern_id)
                if not pattern:
                    return json.dumps({
                        "success": False,
                        "error": f"Pattern not found: {pattern_id}"
                    })
                
                return json.dumps({
                    "success": True,
                    "pattern": pattern
                }, indent=2)
            
            elif action == "search":
                # Search patterns
                search_params = PatternSearchParams(**query_data.get("params", {}))
                patterns = self._search_patterns(search_params)
                
                return json.dumps({
                    "success": True,
                    "count": len(patterns),
                    "patterns": [self._get_pattern_summary(p) for p in patterns]
                }, indent=2)
            
            elif action == "convert":
                # Convert pattern to Cypher
                pattern_id = query_data.get("pattern_id")
                if not pattern_id:
                    return json.dumps({
                        "success": False,
                        "error": "Missing pattern_id parameter"
                    })
                
                parameters = query_data.get("parameters", {})
                use_template = query_data.get("use_template", True)
                
                result = self._convert_pattern_to_cypher(pattern_id, parameters, use_template)
                return json.dumps(result, indent=2)
            
            else:
                return json.dumps({
                    "success": False,
                    "error": f"Unknown action: {action}"
                })
        
        except Exception as e:
            logger.error(f"Error in PatternLibraryTool: {str(e)}")
            return json.dumps({
                "success": False,
                "error": str(e)
            })
    
    def get_pattern(self, pattern_id: str) -> Dict[str, Any]:
        """
        Get a specific pattern by ID
        
        Args:
            pattern_id: ID of the pattern to retrieve
            
        Returns:
            Pattern definition or None if not found
        """
        self._load_patterns()
        return self._patterns_cache.get(pattern_id)
    
    def list_patterns(self) -> List[Dict[str, Any]]:
        """
        List all available patterns
        
        Returns:
            List of pattern summaries
        """
        self._load_patterns()
        return [self._get_pattern_summary(pattern) for pattern in self._patterns_cache.values()]
    
    def search_patterns(self, **kwargs) -> List[Dict[str, Any]]:
        """
        Search patterns by various criteria
        
        Args:
            **kwargs: Search parameters (pattern_id, category, risk_level, tags)
            
        Returns:
            List of matching pattern summaries
        """
        search_params = PatternSearchParams(**kwargs)
        patterns = self._search_patterns(search_params)
        return [self._get_pattern_summary(p) for p in patterns]
    
    def convert_pattern(self, pattern_id: str, parameters: Dict[str, Any] = None, use_template: bool = True) -> Dict[str, Any]:
        """
        Convert a pattern to a Cypher query
        
        Args:
            pattern_id: ID of the pattern to convert
            parameters: Parameters to substitute in the query
            use_template: Whether to use the template or generate dynamically
            
        Returns:
            Dictionary with cypher query and parameters
        """
        return self._convert_pattern_to_cypher(pattern_id, parameters, use_template)
