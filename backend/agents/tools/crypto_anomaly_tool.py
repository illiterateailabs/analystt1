"""
Crypto Anomaly Detection Tool for identifying suspicious patterns in cryptocurrency transactions.

This module provides a comprehensive tool for detecting various types of anomalies
in cryptocurrency transaction data, including:
- Time-series anomalies (volume spikes, level shifts, volatility changes)
- Wash trading patterns (circular transactions)
- Pump-and-dump schemes
- Clustering of similar addresses
- Graph-based anomaly detection

The tool supports both CSV data sources and Neo4j graph databases.
"""

import os
import logging
from typing import Dict, List, Union, Optional, Tuple, Any
from datetime import datetime, timedelta
import json
import tempfile

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import networkx as nx
import matplotlib.pyplot as plt
import io
import base64

# Import ADTK for time-series anomaly detection
try:
    from adtk.detector import ThresholdAD, QuantileAD, InterQuartileRangeAD, GeneralizedESDTestAD
    from adtk.detector import LevelShiftAD, VolatilityShiftAD, SeasonalAD, AutoregressionAD
    from adtk.data import validate_series
    ADTK_AVAILABLE = True
except ImportError:
    ADTK_AVAILABLE = False
    logging.warning("ADTK not installed. Time-series anomaly detection will not be available.")

# Import Neo4j client if available in the project
try:
    from backend.integrations.neo4j_client import Neo4jClient
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    logging.warning("Neo4j client not available. Neo4j data source will not be supported.")

class CryptoAnomalyTool:
    """
    Tool for detecting anomalies in cryptocurrency transaction data.
    
    This tool provides methods for identifying various types of suspicious patterns
    in cryptocurrency transactions, including time-series anomalies, wash trading,
    pump-and-dump schemes, and clustering of similar addresses.
    """
    
    def __init__(self, neo4j_client=None):
        """
        Initialize the CryptoAnomalyTool.
        
        Args:
            neo4j_client: Optional Neo4j client for database access
        """
        self.neo4j_client = neo4j_client
        self.logger = logging.getLogger(__name__)
        
        # Check for required dependencies
        if not ADTK_AVAILABLE:
            self.logger.warning("ADTK library not available. Install with: pip install adtk")
        
        if neo4j_client is None and NEO4J_AVAILABLE:
            try:
                self.neo4j_client = Neo4jClient()
                self.logger.info("Connected to Neo4j database")
            except Exception as e:
                self.logger.warning(f"Could not initialize Neo4j client: {str(e)}")
    
    def run(self, operation: str, **kwargs) -> Dict[str, Any]:
        """
        Run a specific anomaly detection operation.
        
        Args:
            operation: The operation to run. Options include:
                - 'time_series_anomaly': Detect anomalies in time-series data
                - 'wash_trading': Detect wash trading patterns
                - 'pump_and_dump': Detect pump-and-dump schemes
                - 'address_clustering': Cluster similar addresses
                - 'graph_metrics': Calculate graph-based metrics
                - 'all': Run all detection methods
            **kwargs: Additional arguments specific to each operation
                
        Returns:
            Dict containing the results of the operation
        
        Raises:
            ValueError: If an invalid operation is specified
            RuntimeError: If a required dependency is missing
        """
        self.logger.info(f"Running crypto anomaly detection: {operation}")
        
        try:
            if operation == 'time_series_anomaly':
                return self._detect_time_series_anomalies(**kwargs)
            elif operation == 'wash_trading':
                return self._detect_wash_trading(**kwargs)
            elif operation == 'pump_and_dump':
                return self._detect_pump_and_dump(**kwargs)
            elif operation == 'address_clustering':
                return self._cluster_addresses(**kwargs)
            elif operation == 'graph_metrics':
                return self._calculate_graph_metrics(**kwargs)
            elif operation == 'all':
                results = {}
                # Run all operations with the provided kwargs
                for op in ['time_series_anomaly', 'wash_trading', 'pump_and_dump', 
                          'address_clustering', 'graph_metrics']:
                    try:
                        results[op] = self.run(op, **kwargs)
                    except Exception as e:
                        self.logger.error(f"Error running {op}: {str(e)}")
                        results[op] = {"error": str(e)}
                return results
            else:
                raise ValueError(f"Invalid operation: {operation}")
        except Exception as e:
            self.logger.error(f"Error in CryptoAnomalyTool.run: {str(e)}")
            raise
    
    def _load_data(self, data_source: str, data_path: str, **kwargs) -> pd.DataFrame:
        """
        Load data from the specified source.
        
        Args:
            data_source: Source type ('csv', 'neo4j')
            data_path: Path to CSV file or Cypher query
            **kwargs: Additional arguments for data loading
            
        Returns:
            DataFrame containing the loaded data
            
        Raises:
            ValueError: If an invalid data source is specified
            FileNotFoundError: If the CSV file does not exist
            RuntimeError: If Neo4j client is not available
        """
        if data_source.lower() == 'csv':
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"CSV file not found: {data_path}")
            
            return pd.read_csv(data_path, **kwargs)
        
        elif data_source.lower() == 'neo4j':
            if not NEO4J_AVAILABLE or self.neo4j_client is None:
                raise RuntimeError("Neo4j client not available")
            
            # Execute Cypher query and convert results to DataFrame
            result = self.neo4j_client.run_query(data_path)
            return pd.DataFrame(result)
        
        else:
            raise ValueError(f"Invalid data source: {data_source}")
    
    def _detect_time_series_anomalies(self, 
                                     data_source: str, 
                                     data_path: str,
                                     timestamp_column: str,
                                     value_column: str,
                                     entity_column: Optional[str] = None,
                                     detection_methods: List[str] = None,
                                     **kwargs) -> Dict[str, Any]:
        """
        Detect anomalies in time-series cryptocurrency data.
        
        Args:
            data_source: Source type ('csv', 'neo4j')
            data_path: Path to CSV file or Cypher query
            timestamp_column: Name of the column containing timestamps
            value_column: Name of the column containing values to analyze
            entity_column: Optional column to group by (e.g., address, token)
            detection_methods: List of detection methods to use
                Options: 'seasonal', 'level_shift', 'volatility_shift', 'threshold', 
                         'quantile', 'iqr', 'esd', 'autoregression', 'all'
            **kwargs: Additional arguments for specific detectors
                
        Returns:
            Dict containing detected anomalies and visualization
            
        Raises:
            RuntimeError: If ADTK is not available
        """
        if not ADTK_AVAILABLE:
            raise RuntimeError("ADTK library is required for time-series anomaly detection")
        
        # Set default detection methods if not specified
        if detection_methods is None:
            detection_methods = ['seasonal', 'level_shift', 'volatility_shift']
        elif 'all' in detection_methods:
            detection_methods = ['seasonal', 'level_shift', 'volatility_shift', 
                                'threshold', 'quantile', 'iqr', 'esd', 'autoregression']
        
        # Load data
        df = self._load_data(data_source, data_path, **kwargs)
        
        # Convert timestamp to datetime if it's not already
        if pd.api.types.is_string_dtype(df[timestamp_column]):
            df[timestamp_column] = pd.to_datetime(df[timestamp_column])
        
        # Set up results dictionary
        results = {
            "anomalies": {},
            "visualization": None,
            "summary": {},
            "metadata": {
                "data_source": data_source,
                "timestamp_column": timestamp_column,
                "value_column": value_column,
                "entity_column": entity_column,
                "detection_methods": detection_methods,
                "total_records": len(df),
                "time_range": [df[timestamp_column].min().isoformat(), 
                              df[timestamp_column].max().isoformat()]
            }
        }
        
        # Process for each entity if entity_column is provided
        if entity_column is not None:
            entities = df[entity_column].unique()
            results["metadata"]["entity_count"] = len(entities)
            
            for entity in entities:
                entity_df = df[df[entity_column] == entity]
                entity_results = self._process_single_time_series(
                    entity_df, timestamp_column, value_column, detection_methods, **kwargs
                )
                results["anomalies"][str(entity)] = entity_results["anomalies"]
                results["summary"][str(entity)] = entity_results["summary"]
        else:
            # Process the entire dataset as a single time series
            ts_results = self._process_single_time_series(
                df, timestamp_column, value_column, detection_methods, **kwargs
            )
            results["anomalies"] = ts_results["anomalies"]
            results["summary"] = ts_results["summary"]
            results["visualization"] = ts_results["visualization"]
        
        return results
    
    def _process_single_time_series(self, 
                                   df: pd.DataFrame, 
                                   timestamp_column: str,
                                   value_column: str,
                                   detection_methods: List[str],
                                   **kwargs) -> Dict[str, Any]:
        """
        Process a single time series for anomaly detection.
        
        Args:
            df: DataFrame containing the time series
            timestamp_column: Name of the column containing timestamps
            value_column: Name of the column containing values to analyze
            detection_methods: List of detection methods to use
            **kwargs: Additional arguments for specific detectors
            
        Returns:
            Dict containing detected anomalies and visualization
        """
        # Set the timestamp as index
        ts_df = df.copy()
        ts_df = ts_df.set_index(timestamp_column)
        
        # Create a Series with the values to analyze
        ts = ts_df[value_column]
        ts = validate_series(ts)
        
        # Initialize results
        anomalies = {}
        all_anomalies = pd.Series(False, index=ts.index)
        
        # Apply each detection method
        for method in detection_methods:
            try:
                if method == 'seasonal':
                    # Detect seasonal anomalies
                    seasonal_period = kwargs.get('seasonal_period', 24)  # Default: 24 hours
                    seasonal_detector = SeasonalAD(seasonal_period=seasonal_period)
                    seasonal_anomalies = seasonal_detector.fit_detect(ts)
                    anomalies['seasonal'] = self._format_anomalies(seasonal_anomalies)
                    all_anomalies = all_anomalies | seasonal_anomalies
                
                elif method == 'level_shift':
                    # Detect level shifts (sudden changes in the mean)
                    window = kwargs.get('level_shift_window', 5)
                    level_shift_detector = LevelShiftAD(window=window, c=kwargs.get('level_shift_c', 6.0))
                    level_shift_anomalies = level_shift_detector.fit_detect(ts)
                    anomalies['level_shift'] = self._format_anomalies(level_shift_anomalies)
                    all_anomalies = all_anomalies | level_shift_anomalies
                
                elif method == 'volatility_shift':
                    # Detect volatility shifts (sudden changes in variance)
                    window = kwargs.get('volatility_shift_window', 5)
                    volatility_shift_detector = VolatilityShiftAD(window=window, c=kwargs.get('volatility_shift_c', 6.0))
                    volatility_shift_anomalies = volatility_shift_detector.fit_detect(ts)
                    anomalies['volatility_shift'] = self._format_anomalies(volatility_shift_anomalies)
                    all_anomalies = all_anomalies | volatility_shift_anomalies
                
                elif method == 'threshold':
                    # Detect threshold crossings
                    high = kwargs.get('threshold_high', ts.mean() + 3 * ts.std())
                    low = kwargs.get('threshold_low', ts.mean() - 3 * ts.std())
                    threshold_detector = ThresholdAD(high=high, low=low)
                    threshold_anomalies = threshold_detector.detect(ts)
                    anomalies['threshold'] = self._format_anomalies(threshold_anomalies)
                    all_anomalies = all_anomalies | threshold_anomalies
                
                elif method == 'quantile':
                    # Detect quantile-based anomalies
                    high = kwargs.get('quantile_high', 0.95)
                    low = kwargs.get('quantile_low', 0.05)
                    quantile_detector = QuantileAD(high=high, low=low)
                    quantile_anomalies = quantile_detector.fit_detect(ts)
                    anomalies['quantile'] = self._format_anomalies(quantile_anomalies)
                    all_anomalies = all_anomalies | quantile_anomalies
                
                elif method == 'iqr':
                    # Detect interquartile range anomalies
                    iqr_detector = InterQuartileRangeAD(c=kwargs.get('iqr_c', 3.0))
                    iqr_anomalies = iqr_detector.fit_detect(ts)
                    anomalies['iqr'] = self._format_anomalies(iqr_anomalies)
                    all_anomalies = all_anomalies | iqr_anomalies
                
                elif method == 'esd':
                    # Detect anomalies using Generalized ESD Test
                    esd_detector = GeneralizedESDTestAD(alpha=kwargs.get('esd_alpha', 0.05))
                    esd_anomalies = esd_detector.fit_detect(ts)
                    anomalies['esd'] = self._format_anomalies(esd_anomalies)
                    all_anomalies = all_anomalies | esd_anomalies
                
                elif method == 'autoregression':
                    # Detect anomalies using autoregression
                    ar_detector = AutoregressionAD(n_steps=kwargs.get('ar_steps', 1))
                    ar_anomalies = ar_detector.fit_detect(ts)
                    anomalies['autoregression'] = self._format_anomalies(ar_anomalies)
                    all_anomalies = all_anomalies | ar_anomalies
            
            except Exception as e:
                self.logger.error(f"Error applying {method} detector: {str(e)}")
                anomalies[method] = {"error": str(e)}
        
        # Create visualization
        visualization = self._create_time_series_visualization(ts, all_anomalies)
        
        # Create summary statistics
        summary = {
            "total_points": len(ts),
            "anomaly_count": all_anomalies.sum(),
            "anomaly_percentage": (all_anomalies.sum() / len(ts)) * 100 if len(ts) > 0 else 0,
            "value_stats": {
                "min": ts.min(),
                "max": ts.max(),
                "mean": ts.mean(),
                "median": ts.median(),
                "std": ts.std()
            }
        }
        
        return {
            "anomalies": anomalies,
            "visualization": visualization,
            "summary": summary
        }
    
    def _format_anomalies(self, anomaly_series: pd.Series) -> List[Dict[str, Any]]:
        """
        Format anomalies for output.
        
        Args:
            anomaly_series: Series of boolean values indicating anomalies
            
        Returns:
            List of dicts with timestamp and value for each anomaly
        """
        if anomaly_series is None:
            return []
        
        # Extract anomalies (True values)
        anomalies = anomaly_series[anomaly_series].index.to_list()
        
        # Format as list of dicts
        return [{"timestamp": ts.isoformat(), "score": 1.0} for ts in anomalies]
    
    def _create_time_series_visualization(self, 
                                         series: pd.Series, 
                                         anomalies: pd.Series) -> str:
        """
        Create a visualization of the time series with anomalies highlighted.
        
        Args:
            series: The time series data
            anomalies: Boolean series indicating anomalies
            
        Returns:
            Base64-encoded PNG image of the visualization
        """
        try:
            plt.figure(figsize=(12, 6))
            plt.plot(series.index, series.values, 'b-', label='Value')
            
            # Plot anomalies as red points
            if anomalies.sum() > 0:
                anomaly_indices = anomalies[anomalies].index
                plt.plot(anomaly_indices, series[anomaly_indices], 'ro', label='Anomaly')
            
            plt.title('Time Series Anomaly Detection')
            plt.xlabel('Timestamp')
            plt.ylabel('Value')
            plt.legend()
            plt.grid(True)
            
            # Save plot to a bytes buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            plt.close()
            
            # Encode the image as base64
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            
            return img_str
        
        except Exception as e:
            self.logger.error(f"Error creating visualization: {str(e)}")
            return None
    
    def _detect_wash_trading(self,
                            data_source: str,
                            data_path: str,
                            from_address_column: str,
                            to_address_column: str,
                            value_column: str,
                            timestamp_column: Optional[str] = None,
                            token_column: Optional[str] = None,
                            min_cycle_length: int = 2,
                            max_cycle_length: int = 5,
                            min_cycle_value: float = 0.0,
                            time_window_hours: Optional[int] = 24,
                            **kwargs) -> Dict[str, Any]:
        """
        Detect wash trading patterns in cryptocurrency transactions.
        
        Wash trading involves a trader buying and selling the same asset simultaneously
        to create artificial activity, often in a circular pattern between controlled addresses.
        
        Args:
            data_source: Source type ('csv', 'neo4j')
            data_path: Path to CSV file or Cypher query
            from_address_column: Name of the column containing source addresses
            to_address_column: Name of the column containing destination addresses
            value_column: Name of the column containing transaction values
            timestamp_column: Optional name of the column containing timestamps
            token_column: Optional name of the column containing token identifiers
            min_cycle_length: Minimum length of cycles to detect
            max_cycle_length: Maximum length of cycles to detect
            min_cycle_value: Minimum value of transactions in a cycle
            time_window_hours: Optional time window for transactions in hours
            **kwargs: Additional arguments for data loading
            
        Returns:
            Dict containing detected wash trading patterns
        """
        # Load data
        df = self._load_data(data_source, data_path, **kwargs)
        
        # Convert timestamp to datetime if provided and not already
        if timestamp_column is not None and pd.api.types.is_string_dtype(df[timestamp_column]):
            df[timestamp_column] = pd.to_datetime(df[timestamp_column])
        
        # Filter by time window if provided
        if timestamp_column is not None and time_window_hours is not None:
            max_time = df[timestamp_column].max()
            min_time = max_time - timedelta(hours=time_window_hours)
            df = df[(df[timestamp_column] >= min_time) & (df[timestamp_column] <= max_time)]
        
        # Initialize results
        results = {
            "wash_trading_cycles": [],
            "suspicious_addresses": [],
            "visualization": None,
            "summary": {},
            "metadata": {
                "data_source": data_source,
                "from_address_column": from_address_column,
                "to_address_column": to_address_column,
                "value_column": value_column,
                "timestamp_column": timestamp_column,
                "token_column": token_column,
                "min_cycle_length": min_cycle_length,
                "max_cycle_length": max_cycle_length,
                "min_cycle_value": min_cycle_value,
                "time_window_hours": time_window_hours,
                "total_transactions": len(df)
            }
        }
        
        # Process separately for each token if token_column is provided
        if token_column is not None:
            tokens = df[token_column].unique()
            results["metadata"]["token_count"] = len(tokens)
            
            for token in tokens:
                token_df = df[df[token_column] == token]
                token_results = self._detect_cycles_in_transactions(
                    token_df, from_address_column, to_address_column, value_column,
                    timestamp_column, min_cycle_length, max_cycle_length, min_cycle_value
                )
                
                # Add token identifier to each cycle
                for cycle in token_results["cycles"]:
                    cycle["token"] = token
                    results["wash_trading_cycles"].append(cycle)
                
                # Add token identifier to each suspicious address
                for address in token_results["suspicious_addresses"]:
                    address["token"] = token
                    results["suspicious_addresses"].append(address)
        else:
            # Process the entire dataset
            cycles_results = self._detect_cycles_in_transactions(
                df, from_address_column, to_address_column, value_column,
                timestamp_column, min_cycle_length, max_cycle_length, min_cycle_value
            )
            
            results["wash_trading_cycles"] = cycles_results["cycles"]
            results["suspicious_addresses"] = cycles_results["suspicious_addresses"]
            results["visualization"] = cycles_results["visualization"]
        
        # Create summary
        results["summary"] = {
            "total_cycles_detected": len(results["wash_trading_cycles"]),
            "total_suspicious_addresses": len(results["suspicious_addresses"]),
            "total_wash_trading_volume": sum(cycle["total_value"] for cycle in results["wash_trading_cycles"]),
            "average_cycle_length": np.mean([len(cycle["addresses"]) for cycle in results["wash_trading_cycles"]]) if results["wash_trading_cycles"] else 0
        }
        
        return results
    
    def _detect_cycles_in_transactions(self,
                                      df: pd.DataFrame,
                                      from_address_column: str,
                                      to_address_column: str,
                                      value_column: str,
                                      timestamp_column: Optional[str],
                                      min_cycle_length: int,
                                      max_cycle_length: int,
                                      min_cycle_value: float) -> Dict[str, Any]:
        """
        Detect cycles in transaction graph that may indicate wash trading.
        
        Args:
            df: DataFrame containing transaction data
            from_address_column: Name of the column containing source addresses
            to_address_column: Name of the column containing destination addresses
            value_column: Name of the column containing transaction values
            timestamp_column: Optional name of the column containing timestamps
            min_cycle_length: Minimum length of cycles to detect
            max_cycle_length: Maximum length of cycles to detect
            min_cycle_value: Minimum value of transactions in a cycle
            
        Returns:
            Dict containing detected cycles and suspicious addresses
        """
        # Create directed graph from transactions
        G = nx.DiGraph()
        
        # Add edges with attributes
        for _, row in df.iterrows():
            from_addr = row[from_address_column]
            to_addr = row[to_address_column]
            value = row[value_column]
            
            # Skip self-transactions (they're not part of cycles)
            if from_addr == to_addr:
                continue
            
            # Skip transactions below minimum value
            if value < min_cycle_value:
                continue
            
            # Add timestamp if available
            attrs = {"value": value}
            if timestamp_column is not None:
                attrs["timestamp"] = row[timestamp_column]
            
            # Add edge or update existing edge
            if G.has_edge(from_addr, to_addr):
                # If edge exists, update attributes (e.g., sum values)
                G[from_addr][to_addr]["value"] += value
                G[from_addr][to_addr]["count"] = G[from_addr][to_addr].get("count", 1) + 1
            else:
                attrs["count"] = 1
                G.add_edge(from_addr, to_addr, **attrs)
        
        # Find simple cycles in the graph
        cycles = []
        try:
            # Use NetworkX's simple_cycles to find all cycles
            for cycle in nx.simple_cycles(G):
                if min_cycle_length <= len(cycle) <= max_cycle_length:
                    # Calculate total value of the cycle
                    cycle_value = 0
                    cycle_edges = []
                    
                    # Add the last edge to close the cycle
                    cycle_with_close = cycle + [cycle[0]]
                    
                    # Calculate cycle properties
                    for i in range(len(cycle)):
                        from_addr = cycle[i]
                        to_addr = cycle_with_close[i+1]
                        
                        if G.has_edge(from_addr, to_addr):
                            edge_value = G[from_addr][to_addr]["value"]
                            cycle_value += edge_value
                            
                            edge_data = {
                                "from": from_addr,
                                "to": to_addr,
                                "value": edge_value,
                                "count": G[from_addr][to_addr].get("count", 1)
                            }
                            
                            if "timestamp" in G[from_addr][to_addr]:
                                edge_data["timestamp"] = G[from_addr][to_addr]["timestamp"].isoformat()
                            
                            cycle_edges.append(edge_data)
                    
                    # Add cycle to results
                    cycles.append({
                        "addresses": cycle,
                        "length": len(cycle),
                        "total_value": cycle_value,
                        "edges": cycle_edges,
                        "is_balanced": self._is_balanced_cycle(cycle_edges)
                    })
        except nx.NetworkXNoCycle:
            # No cycles found
            pass
        
        # Identify suspicious addresses (those involved in multiple cycles)
        address_cycle_count = {}
        for cycle in cycles:
            for address in cycle["addresses"]:
                address_cycle_count[address] = address_cycle_count.get(address, 0) + 1
        
        suspicious_addresses = [
            {"address": addr, "cycle_count": count}
            for addr, count in address_cycle_count.items()
            if count > 1
        ]
        
        # Sort by cycle count (most suspicious first)
        suspicious_addresses.sort(key=lambda x: x["cycle_count"], reverse=True)
        
        # Create visualization of the transaction graph with cycles highlighted
        visualization = self._create_graph_visualization(G, cycles)
        
        return {
            "cycles": cycles,
            "suspicious_addresses": suspicious_addresses,
            "visualization": visualization
        }
    
    def _is_balanced_cycle(self, cycle_edges: List[Dict[str, Any]]) -> bool:
        """
        Check if a cycle is balanced (all transactions have similar values).
        
        A balanced cycle is more likely to be wash trading.
        
        Args:
            cycle_edges: List of edges in the cycle with their values
            
        Returns:
            True if the cycle is balanced, False otherwise
        """
        if not cycle_edges:
            return False
        
        values = [edge["value"] for edge in cycle_edges]
        mean_value = np.mean(values)
        
        # Calculate coefficient of variation (CV)
        # CV = standard deviation / mean
        # A lower CV indicates more balanced values
        if mean_value > 0:
            cv = np.std(values) / mean_value
            return cv < 0.2  # Threshold for "balanced"
        
        return False
    
    def _create_graph_visualization(self, 
                                   G: nx.DiGraph, 
                                   cycles: List[Dict[str, Any]]) -> str:
        """
        Create a visualization of the transaction graph with cycles highlighted.
        
        Args:
            G: NetworkX DiGraph of transactions
            cycles: List of detected cycles
            
        Returns:
            Base64-encoded PNG image of the visualization
        """
        try:
            plt.figure(figsize=(12, 12))
            
            # Create a set of all addresses in cycles
            cycle_addresses = set()
            for cycle in cycles:
                cycle_addresses.update(cycle["addresses"])
            
            # Set node colors based on whether they're in cycles
            node_colors = []
            for node in G.nodes():
                if node in cycle_addresses:
                    node_colors.append('red')
                else:
                    node_colors.append('blue')
            
            # Set edge colors based on whether they're in cycles
            edge_colors = []
            for u, v in G.edges():
                in_cycle = False
                for cycle in cycles:
                    # Check if this edge is part of a cycle
                    cycle_with_close = cycle["addresses"] + [cycle["addresses"][0]]
                    for i in range(len(cycle["addresses"])):
                        if cycle_with_close[i] == u and cycle_with_close[i+1] == v:
                            in_cycle = True
                            break
                    if in_cycle:
                        break
                
                if in_cycle:
                    edge_colors.append('red')
                else:
                    edge_colors.append('blue')
            
            # Draw the graph
            pos = nx.spring_layout(G)
            nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=300, alpha=0.8)
            nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=1.0, alpha=0.5, arrows=True)
            nx.draw_networkx_labels(G, pos, font_size=8, font_color='black')
            
            plt.title('Transaction Graph with Wash Trading Cycles')
            plt.axis('off')
            
            # Save plot to a bytes buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            plt.close()
            
            # Encode the image as base64
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            
            return img_str
        
        except Exception as e:
            self.logger.error(f"Error creating graph visualization: {str(e)}")
            return None
    
    def _detect_pump_and_dump(self,
                             data_source: str,
                             data_path: str,
                             timestamp_column: str,
                             price_column: str,
                             volume_column: str,
                             token_column: Optional[str] = None,
                             lookback_window: int = 24,  # hours
                             price_increase_threshold: float = 50.0,  # percent
                             volume_increase_threshold: float = 200.0,  # percent
                             dump_threshold: float = 30.0,  # percent
                             **kwargs) -> Dict[str, Any]:
        """
        Detect pump-and-dump schemes in cryptocurrency price/volume data.
        
        Pump-and-dump schemes typically involve:
        1. A rapid price increase accompanied by volume spike
        2. Followed by a sharp price decrease as perpetrators sell
        
        Args:
            data_source: Source type ('csv', 'neo4j')
            data_path: Path to CSV file or Cypher query
            timestamp_column: Name of the column containing timestamps
            price_column: Name of the column containing price data
            volume_column: Name of the column containing volume data
            token_column: Optional name of the column containing token identifiers
            lookback_window: Hours to look back for baseline
            price_increase_threshold: Minimum percent price increase to flag
            volume_increase_threshold: Minimum percent volume increase to flag
            dump_threshold: Minimum percent price decrease after pump to flag
            **kwargs: Additional arguments for data loading
            
        Returns:
            Dict containing detected pump-and-dump patterns
        """
        # Load data
        df = self._load_data(data_source, data_path, **kwargs)
        
        # Convert timestamp to datetime if not already
        if pd.api.types.is_string_dtype(df[timestamp_column]):
            df[timestamp_column] = pd.to_datetime(df[timestamp_column])
        
        # Sort by timestamp
        df = df.sort_values(by=timestamp_column)
        
        # Initialize results
        results = {
            "pump_and_dump_events": [],
            "visualization": None,
            "summary": {},
            "metadata": {
                "data_source": data_source,
                "timestamp_column": timestamp_column,
                "price_column": price_column,
                "volume_column": volume_column,
                "token_column": token_column,
                "lookback_window": lookback_window,
                "price_increase_threshold": price_increase_threshold,
                "volume_increase_threshold": volume_increase_threshold,
                "dump_threshold": dump_threshold,
                "total_records": len(df),
                "time_range": [df[timestamp_column].min().isoformat(), 
                              df[timestamp_column].max().isoformat()]
            }
        }
        
        # Process separately for each token if token_column is provided
        if token_column is not None:
            tokens = df[token_column].unique()
            results["metadata"]["token_count"] = len(tokens)
            
            for token in tokens:
                token_df = df[df[token_column] == token]
                token_results = self._analyze_price_volume_patterns(
                    token_df, timestamp_column, price_column, volume_column,
                    lookback_window, price_increase_threshold, 
                    volume_increase_threshold, dump_threshold
                )
                
                # Add token identifier to each event
                for event in token_results["events"]:
                    event["token"] = token
                    results["pump_and_dump_events"].append(event)
                
                # If this is the first token, use its visualization
                if results["visualization"] is None and token_results["visualization"] is not None:
                    results["visualization"] = token_results["visualization"]
        else:
            # Process the entire dataset
            analysis_results = self._analyze_price_volume_patterns(
                df, timestamp_column, price_column, volume_column,
                lookback_window, price_increase_threshold, 
                volume_increase_threshold, dump_threshold
            )
            
            results["pump_and_dump_events"] = analysis_results["events"]
            results["visualization"] = analysis_results["visualization"]
        
        # Create summary
        results["summary"] = {
            "total_events_detected": len(results["pump_and_dump_events"]),
            "average_price_increase": np.mean([event["price_increase_percent"] for event in results["pump_and_dump_events"]]) if results["pump_and_dump_events"] else 0,
            "average_volume_increase": np.mean([event["volume_increase_percent"] for event in results["pump_and_dump_events"]]) if results["pump_and_dump_events"] else 0,
            "average_dump_decrease": np.mean([event["dump_decrease_percent"] for event in results["pump_and_dump_events"]]) if results["pump_and_dump_events"] else 0
        }
        
        return results
    
    def _analyze_price_volume_patterns(self,
                                      df: pd.DataFrame,
                                      timestamp_column: str,
                                      price_column: str,
                                      volume_column: str,
                                      lookback_window: int,
                                      price_increase_threshold: float,
                                      volume_increase_threshold: float,
                                      dump_threshold: float) -> Dict[str, Any]:
        """
        Analyze price and volume patterns to detect pump-and-dump schemes.
        
        Args:
            df: DataFrame containing price and volume data
            timestamp_column: Name of the column containing timestamps
            price_column: Name of the column containing price data
            volume_column: Name of the column containing volume data
            lookback_window: Hours to look back for baseline
            price_increase_threshold: Minimum percent price increase to flag
            volume_increase_threshold: Minimum percent volume increase to flag
            dump_threshold: Minimum percent price decrease after pump to flag
            
        Returns:
            Dict containing detected pump-and-dump events
        """
        if len(df) < 2:
            return {"events": [], "visualization": None}
        
        # Set timestamp as index
        df = df.set_index(timestamp_column)
        
        # Resample to hourly data if needed (for consistent analysis)
        if len(df) > 24 and (df.index[-1] - df.index[0]).total_seconds() > 86400:  # More than a day of data
            df_resampled = df.resample('1H').agg({
                price_column: 'last',
                volume_column: 'sum'
            }).dropna()
            
            # Only use resampled data if we have enough points
            if len(df_resampled) > lookback_window:
                df = df_resampled
        
        # Calculate rolling statistics
        df['price_pct_change'] = df[price_column].pct_change(periods=1) * 100
        df['volume_pct_change'] = df[volume_column].pct_change(periods=1) * 100
        
        # Calculate rolling baselines
        df['price_rolling_mean'] = df[price_column].rolling(window=lookback_window).mean()
        df['volume_rolling_mean'] = df[volume_column].rolling(window=lookback_window).mean()
        
        # Calculate percent change from baseline
        df['price_vs_baseline'] = ((df[price_column] / df['price_rolling_mean']) - 1) * 100
        df['volume_vs_baseline'] = ((df[volume_column] / df['volume_rolling_mean']) - 1) * 100
        
        # Drop NaN values
        df = df.dropna()
        
        # Detect potential pump events
        pump_events = []
        
        for i in range(lookback_window, len(df) - 1):
            # Check for pump conditions
            if (df['price_vs_baseline'].iloc[i] > price_increase_threshold and 
                df['volume_vs_baseline'].iloc[i] > volume_increase_threshold):
                
                # We found a potential pump, now look for the dump
                # Look at the next few periods (up to 48 hours or end of data)
                max_lookahead = min(48, len(df) - i - 1)
                
                dump_found = False
                dump_index = None
                max_price = df[price_column].iloc[i]
                
                for j in range(1, max_lookahead + 1):
                    current_price = df[price_column].iloc[i + j]
                    price_decrease = ((max_price - current_price) / max_price) * 100
                    
                    if price_decrease > dump_threshold:
                        dump_found = True
                        dump_index = i + j
                        break
                
                # If we found both pump and dump, add to events
                if dump_found:
                    pump_time = df.index[i]
                    dump_time = df.index[dump_index]
                    
                    event = {
                        "pump_timestamp": pump_time.isoformat(),
                        "dump_timestamp": dump_time.isoformat(),
                        "duration_hours": (dump_time - pump_time).total_seconds() / 3600,
                        "price_before_pump": df[price_column].iloc[i - lookback_window],
                        "price_at_pump": df[price_column].iloc[i],
                        "price_at_dump": df[price_column].iloc[dump_index],
                        "volume_before_pump": df[volume_column].iloc[i - lookback_window],
                        "volume_at_pump": df[volume_column].iloc[i],
                        "price_increase_percent": df['price_vs_baseline'].iloc[i],
                        "volume_increase_percent": df['volume_vs_baseline'].iloc[i],
                        "dump_decrease_percent": ((df[price_column].iloc[i] - df[price_column].iloc[dump_index]) / df[price_column].iloc[i]) * 100,
                        "anomaly_score": (df['price_vs_baseline'].iloc[i] + df['volume_vs_baseline'].iloc[i]) / 2
                    }
                    
                    pump_events.append(event)
                    
                    # Skip ahead past this dump to avoid duplicate detections
                    i = dump_index
        
        # Sort events by anomaly score (most suspicious first)
        pump_events.sort(key=lambda x: x["anomaly_score"], reverse=True)
        
        # Create visualization
        visualization = self._create_pump_dump_visualization(df, pump_events, price_column, volume_column)
        
        return {
            "events": pump_events,
            "visualization": visualization
        }
    
    def _create_pump_dump_visualization(self,
                                       df: pd.DataFrame,
                                       events: List[Dict[str, Any]],
                                       price_column: str,
                                       volume_column: str) -> str:
        """
        Create a visualization of price and volume with pump-and-dump events highlighted.
        
        Args:
            df: DataFrame containing price and volume data
            events: List of detected pump-and-dump events
            price_column: Name of the column containing price data
            volume_column: Name of the column containing volume data
            
        Returns:
            Base64-encoded PNG image of the visualization
        """
        try:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
            
            # Plot price
            ax1.plot(df.index, df[price_column], 'b-', label='Price')
            ax1.set_title('Price Chart with Pump-and-Dump Events')
            ax1.set_ylabel('Price')
            ax1.grid(True)
            
            # Plot volume
            ax2.bar(df.index, df[volume_column], color='gray', alpha=0.7, label='Volume')
            ax2.set_title('Volume Chart')
            ax2.set_xlabel('Timestamp')
            ax2.set_ylabel('Volume')
            ax2.grid(True)
            
            # Highlight pump-and-dump events
            for event in events:
                pump_time = pd.to_datetime(event["pump_timestamp"])
                dump_time = pd.to_datetime(event["dump_timestamp"])
                
                # Find indices
                if pump_time in df.index and dump_time in df.index:
                    # Highlight pump point
                    ax1.plot(pump_time, event["price_at_pump"], 'ro', markersize=8)
                    ax2.plot(pump_time, event["volume_at_pump"], 'ro', markersize=8)
                    
                    # Highlight dump point
                    ax1.plot(dump_time, event["price_at_dump"], 'go', markersize=8)
                    
                    # Add annotation
                    ax1.annotate(f"+{event['price_increase_percent']:.1f}%", 
                                (pump_time, event["price_at_pump"]),
                                textcoords="offset points",
                                xytext=(0, 10),
                                ha='center')
                    
                    ax1.annotate(f"-{event['dump_decrease_percent']:.1f}%", 
                                (dump_time, event["price_at_dump"]),
                                textcoords="offset points",
                                xytext=(0, -15),
                                ha='center')
            
            plt.tight_layout()
            
            # Save plot to a bytes buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            plt.close()
            
            # Encode the image as base64
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            
            return img_str
        
        except Exception as e:
            self.logger.error(f"Error creating pump-and-dump visualization: {str(e)}")
            return None
    
    def _cluster_addresses(self,
                          data_source: str,
                          data_path: str,
                          address_column: str,
                          feature_columns: List[str],
                          n_clusters: int = 5,
                          **kwargs) -> Dict[str, Any]:
        """
        Cluster cryptocurrency addresses based on their behavior.
        
        Args:
            data_source: Source type ('csv', 'neo4j')
            data_path: Path to CSV file or Cypher query
            address_column: Name of the column containing addresses
            feature_columns: List of columns to use as features for clustering
            n_clusters: Number of clusters to create
            **kwargs: Additional arguments for data loading
            
        Returns:
            Dict containing clustering results
        """
        # Load data
        df = self._load_data(data_source, data_path, **kwargs)
        
        # Check if required columns exist
        for col in [address_column] + feature_columns:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in data")
        
        # Initialize results
        results = {
            "clusters": [],
            "address_clusters": [],
            "visualization": None,
            "summary": {},
            "metadata": {
                "data_source": data_source,
                "address_column": address_column,
                "feature_columns": feature_columns,
                "n_clusters": n_clusters,
                "total_addresses": len(df)
            }
        }
        
        # Extract features
        X = df[feature_columns].copy()
        
        # Handle missing values
        X = X.fillna(0)
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        df['cluster'] = kmeans.fit_predict(X_scaled)
        
        # Calculate cluster centers in original feature space
        cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
        
        # Create cluster descriptions
        for i in range(n_clusters):
            # Get addresses in this cluster
            cluster_df = df[df['cluster'] == i]
            
            # Calculate cluster statistics
            cluster_stats = {}
            for col in feature_columns:
                cluster_stats[col] = {
                    "mean": cluster_df[col].mean(),
                    "median": cluster_df[col].median(),
                    "min": cluster_df[col].min(),
                    "max": cluster_df[col].max(),
                    "std": cluster_df[col].std()
                }
            
            # Create cluster description
            cluster = {
                "cluster_id": i,
                "size": len(cluster_df),
                "percentage": (len(cluster_df) / len(df)) * 100,
                "center": {feature_columns[j]: cluster_centers[i][j] for j in range(len(feature_columns))},
                "statistics": cluster_stats,
                "top_addresses": cluster_df.sort_values(by=feature_columns[0], ascending=False).head(10)[address_column].tolist()
            }
            
            results["clusters"].append(cluster)
        
        # Create address to cluster mapping
        for _, row in df.iterrows():
            results["address_clusters"].append({
                "address": row[address_column],
                "cluster": int(row['cluster']),
                "features": {col: row[col] for col in feature_columns}
            })
        
        # Create visualization
        results["visualization"] = self._create_cluster_visualization(X_scaled, df['cluster'], feature_columns)
        
        # Create summary
        results["summary"] = {
            "total_addresses": len(df),
            "number_of_clusters": n_clusters,
            "largest_cluster_size": max(cluster["size"] for cluster in results["clusters"]),
            "smallest_cluster_size": min(cluster["size"] for cluster in results["clusters"]),
            "silhouette_score": self._calculate_silhouette_score(X_scaled, df['cluster'])
        }
        
        return results
    
    def _calculate_silhouette_score(self, X: np.ndarray, labels: np.ndarray) -> float:
        """
        Calculate silhouette score for clustering evaluation.
        
        Args:
            X: Scaled feature matrix
            labels: Cluster labels
            
        Returns:
            Silhouette score (-1 to 1, higher is better)
        """
        try:
            from sklearn.metrics import silhouette_score
            return silhouette_score(X, labels)
        except Exception as e:
            self.logger.error(f"Error calculating silhouette score: {str(e)}")
            return 0.0
    
    def _create_cluster_visualization(self, 
                                     X_scaled: np.ndarray, 
                                     labels: np.ndarray,
                                     feature_names: List[str]) -> str:
        """
        Create a visualization of the clustering results.
        
        Args:
            X_scaled: Scaled feature matrix
            labels: Cluster labels
            feature_names: Names of the features
            
        Returns:
            Base64-encoded PNG image of the visualization
        """
        try:
            # If we have more than 2 dimensions, use PCA to reduce to 2D
            if X_scaled.shape[1] > 2:
                from sklearn.decomposition import PCA
                pca = PCA(n_components=2)
                X_2d = pca.fit_transform(X_scaled)
                
                # Calculate explained variance
                explained_variance = pca.explained_variance_ratio_ * 100
            else:
                # Use the first two features
                X_2d = X_scaled[:, :2]
                explained_variance = [100, 0]  # Not applicable
            
            # Create scatter plot
            plt.figure(figsize=(10, 8))
            
            # Plot each cluster with a different color
            unique_labels = np.unique(labels)
            colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
            
            for i, color in zip(unique_labels, colors):
                plt.scatter(
                    X_2d[labels == i, 0], X_2d[labels == i, 1],
                    color=color, label=f'Cluster {i}',
                    alpha=0.7, edgecolors='w', s=40
                )
            
            if X_scaled.shape[1] > 2:
                plt.title('PCA Projection of Address Clusters')
                plt.xlabel(f'PC1 ({explained_variance[0]:.1f}% variance)')
                plt.ylabel(f'PC2 ({explained_variance[1]:.1f}% variance)')
            else:
                plt.title('Address Clusters')
                plt.xlabel(feature_names[0])
                plt.ylabel(feature_names[1])
            
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Save plot to a bytes buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            plt.close()
            
            # Encode the image as base64
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            
            return img_str
        
        except Exception as e:
            self.logger.error(f"Error creating cluster visualization: {str(e)}")
            return None
    
    def _calculate_graph_metrics(self,
                                data_source: str,
                                data_path: str,
                                from_address_column: str,
                                to_address_column: str,
                                value_column: Optional[str] = None,
                                timestamp_column: Optional[str] = None,
                                metrics: List[str] = None,
                                top_n: int = 100,
                                **kwargs) -> Dict[str, Any]:
        """
        Calculate graph-based metrics for addresses in a transaction network.
        
        Args:
            data_source: Source type ('csv', 'neo4j')
            data_path: Path to CSV file or Cypher query
            from_address_column: Name of the column containing source addresses
            to_address_column: Name of the column containing destination addresses
            value_column: Optional name of the column containing transaction values
            timestamp_column: Optional name of the column containing timestamps
            metrics: List of metrics to calculate
                Options: 'degree', 'in_degree', 'out_degree', 'betweenness', 
                         'closeness', 'pagerank', 'clustering', 'all'
            top_n: Number of top addresses to return for each metric
            **kwargs: Additional arguments for data loading
            
        Returns:
            Dict containing calculated graph metrics
        """
        # Set default metrics if not specified
        if metrics is None:
            metrics = ['degree', 'pagerank', 'clustering']
        elif 'all' in metrics:
            metrics = ['degree', 'in_degree', 'out_degree', 'betweenness', 
                      'closeness', 'pagerank', 'clustering']
        
        # Load data
        df = self._load_data(data_source, data_path, **kwargs)
        
        # Initialize results
        results = {
            "address_metrics": [],
            "top_addresses": {},
            "visualization": None,
            "summary": {},
            "metadata": {
                "data_source": data_source,
                "from_address_column": from_address_column,
                "to_address_column": to_address_column,
                "value_column": value_column,
                "timestamp_column": timestamp_column,
                "metrics": metrics,
                "total_transactions": len(df)
            }
        }
        
        # Create directed graph from transactions
        G = nx.DiGraph()
        
        # Add edges with attributes
        for _, row in df.iterrows():
            from_addr = row[from_address_column]
            to_addr = row[to_address_column]
            
            # Skip self-transactions for some analyses
            if from_addr == to_addr:
                continue
            
            # Add edge attributes
            attrs = {}
            if value_column is not None:
                attrs["value"] = row[value_column]
            
            if timestamp_column is not None:
                attrs["timestamp"] = row[timestamp_column]
            
            # Add edge or update existing edge
            if G.has_edge(from_addr, to_addr):
                # If edge exists, update attributes (e.g., sum values)
                if value_column is not None:
                    G[from_addr][to_addr]["value"] = G[from_addr][to_addr].get("value", 0) + attrs["value"]
                G[from_addr][to_addr]["count"] = G[from_addr][to_addr].get("count", 1) + 1
            else:
                attrs["count"] = 1
                G.add_edge(from_addr, to_addr, **attrs)
        
        # Calculate requested metrics
        metric_results = {}
        
        if 'degree' in metrics:
            metric_results['degree'] = dict(G.degree())
        
        if 'in_degree' in metrics:
            metric_results['in_degree'] = dict(G.in_degree())
        
        if 'out_degree' in metrics:
            metric_results['out_degree'] = dict(G.out_degree())
        
        if 'betweenness' in metrics:
            try:
                # This can be computationally expensive for large graphs
                metric_results['betweenness'] = nx.betweenness_centrality(G)
            except Exception as e:
                self.logger.error(f"Error calculating betweenness centrality: {str(e)}")
                metric_results['betweenness'] = {}
        
        if 'closeness' in metrics:
            try:
                # This can be computationally expensive for large graphs
                metric_results['closeness'] = nx.closeness_centrality(G)
            except Exception as e:
                self.logger.error(f"Error calculating closeness centrality: {str(e)}")
                metric_results['closeness'] = {}
        
        if 'pagerank' in metrics:
            try:
                metric_results['pagerank'] = nx.pagerank(G)
            except Exception as e:
                self.logger.error(f"Error calculating PageRank: {str(e)}")
                metric_results['pagerank'] = {}
        
        if 'clustering' in metrics:
            try:
                metric_results['clustering'] = nx.clustering(G)
            except Exception as e:
                self.logger.error(f"Error calculating clustering coefficient: {str(e)}")
                metric_results['clustering'] = {}
        
        # Combine metrics for each address
        all_addresses = set(G.nodes())
        
        for address in all_addresses:
            address_data = {"address": address}
            
            # Add each calculated metric
            for metric_name, metric_dict in metric_results.items():
                address_data[metric_name] = metric_dict.get(address, 0)
            
            results["address_metrics"].append(address_data)
        
        # Find top addresses for each metric
        for metric_name in metric_results.keys():
            # Sort addresses by this metric (descending)
            sorted_addresses = sorted(
                results["address_metrics"],
                key=lambda x: x.get(metric_name, 0),
                reverse=True
            )
            
            # Take top N
            results["top_addresses"][metric_name] = sorted_addresses[:top_n]
        
        # Create visualization
        results["visualization"] = self._create_network_visualization(G, metric_results)
        
        # Create summary statistics
        results["summary"] = {
            "total_addresses": len(all_addresses),
            "total_transactions": len(df),
            "graph_density": nx.density(G),
            "average_degree": sum(dict(G.degree()).values()) / len(G) if len(G) > 0 else 0,
            "strongly_connected_components": nx.number_strongly_connected_components(G),
            "weakly_connected_components": nx.number_weakly_connected_components(G)
        }
        
        return results
    
    def _create_network_visualization(self, 
                                     G: nx.DiGraph, 
                                     metrics: Dict[str, Dict[str, float]]) -> str:
        """
        Create a visualization of the transaction network with node sizes based on metrics.
        
        Args:
            G: NetworkX DiGraph of transactions
            metrics: Dict of calculated metrics
            
        Returns:
            Base64-encoded PNG image of the visualization
        """
        try:
            # If graph is too large, sample a subgraph
            if len(G) > 500:
                # Use PageRank to find important nodes
                if 'pagerank' in metrics:
                    pagerank = metrics['pagerank']
                    top_nodes = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:300]
                    top_nodes = [node for node, _ in top_nodes]
                else:
                    # Use degree as fallback
                    degree = dict(G.degree())
                    top_nodes = sorted(degree.items(), key=lambda x: x[1], reverse=True)[:300]
                    top_nodes = [node for node, _ in top_nodes]
                
                # Create subgraph
                G = G.subgraph(top_nodes)
            
            plt.figure(figsize=(12, 12))
            
            # Use spring layout for node positioning
            pos = nx.spring_layout(G)
            
            # Use PageRank or degree for node sizes
            if 'pagerank' in metrics and metrics['pagerank']:
                node_size = [metrics['pagerank'].get(node, 0) * 10000 for node in G.nodes()]
                size_metric = 'pagerank'
            else:
                node_size = [G.degree(node) * 10 for node in G.nodes()]
                size_metric = 'degree'
            
            # Use in-degree for node color
            if 'in_degree' in metrics and metrics['in_degree']:
                node_color = [metrics['in_degree'].get(node, 0) for node in G.nodes()]
                color_metric = 'in_degree'
            else:
                node_color = [G.in_degree(node) for node in G.nodes()]
                color_metric = 'in_degree'
            
            # Draw the graph
            nodes = nx.draw_networkx_nodes(
                G, pos, 
                node_size=node_size,
                node_color=node_color,
                cmap=plt.cm.viridis,
                alpha=0.8
            )
            
            # Add colorbar
            plt.colorbar(nodes, label=f'Node color: {color_metric}')
            
            # Draw edges with alpha based on weight if available
            if any('value' in G[u][v] for u, v in G.edges()):
                edge_width = [G[u][v].get('value', 1) / 1000 + 0.1 for u, v in G.edges()]
                nx.draw_networkx_edges(G, pos, width=edge_width, alpha=0.3, arrows=True, arrowsize=5)
            else:
                nx.draw_networkx_edges(G, pos, alpha=0.3, arrows=True, arrowsize=5)
            
            # Draw labels for top nodes only
            if 'pagerank' in metrics and metrics['pagerank']:
                top_nodes = sorted(metrics['pagerank'].items(), key=lambda x: x[1], reverse=True)[:20]
                top_nodes = [node for node, _ in top_nodes]
                label_dict = {node: node for node in top_nodes if node in G.nodes()}
            else:
                top_degree_nodes = sorted(dict(G.degree()).items(), key=lambda x: x[1], reverse=True)[:20]
                top_nodes = [node for node, _ in top_degree_nodes]
                label_dict = {node: node for node in top_nodes}
            
            nx.draw_networkx_labels(G, pos, labels=label_dict, font_size=8, font_color='black')
            
            plt.title(f'Transaction Network (Node size: {size_metric})')
            plt.axis('off')
            
            # Save plot to a bytes buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            plt.close()
            
            # Encode the image as base64
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            
            return img_str
        
        except Exception as e:
            self.logger.error(f"Error creating network visualization: {str(e)}")
            return None
    
    def import_csv_to_neo4j(self,
                           csv_path: str,
                           from_address_column: str,
                           to_address_column: str,
                           value_column: Optional[str] = None,
                           timestamp_column: Optional[str] = None,
                           token_column: Optional[str] = None,
                           additional_columns: List[str] = None,
                           batch_size: int = 1000) -> Dict[str, Any]:
        """
        Import cryptocurrency transaction data from CSV to Neo4j.
        
        Args:
            csv_path: Path to CSV file
            from_address_column: Name of the column containing source addresses
            to_address_column: Name of the column containing destination addresses
            value_column: Optional name of the column containing transaction values
            timestamp_column: Optional name of the column containing timestamps
            token_column: Optional name of the column containing token identifiers
            additional_columns: Optional list of additional columns to import
            batch_size: Number of transactions to import in each batch
            
        Returns:
            Dict containing import results
            
        Raises:
            RuntimeError: If Neo4j client is not available
            FileNotFoundError: If the CSV file does not exist
        """
        if not NEO4J_AVAILABLE or self.neo4j_client is None:
            raise RuntimeError("Neo4j client not available")
        
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        # Read CSV file
        df = pd.read_csv(csv_path)
        
        # Check if required columns exist
        for col in [from_address_column, to_address_column]:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in CSV")
        
        # Initialize results
        results = {
            "total_transactions": len(df),
            "imported_transactions": 0,
            "errors": [],
            "summary": {}
        }
        
        # Process in batches
        for i in range(0, len(df), batch_size):
            batch_df = df.iloc[i:i+batch_size]
            
            # Create Cypher query for this batch
            cypher_queries = []
            
            for _, row in batch_df.iterrows():
                from_addr = row[from_address_column]
                to_addr = row[to_address_column]
                
                # Skip rows with missing addresses
                if pd.isna(from_addr) or pd.isna(to_addr):
                    continue
                
                # Create address nodes
                cypher_queries.append(
                    f"MERGE (from:Address {{address: '{from_addr}'}}) "
                    f"MERGE (to:Address {{address: '{to_addr}'}}) "
                )
                
                # Build transaction properties
                tx_props = []
                
                if value_column is not None and value_column in row and not pd.isna(row[value_column]):
                    tx_props.append(f"value: {row[value_column]}")
                
                if timestamp_column is not None and timestamp_column in row and not pd.isna(row[timestamp_column]):
                    # Format timestamp as Neo4j datetime
                    try:
                        timestamp = pd.to_datetime(row[timestamp_column])
                        tx_props.append(f"timestamp: datetime('{timestamp.isoformat()}')")
                    except:
                        pass
                
                if token_column is not None and token_column in row and not pd.isna(row[token_column]):
                    tx_props.append(f"token: '{row[token_column]}'")
                
                # Add additional columns
                if additional_columns is not None:
                    for col in additional_columns:
                        if col in row and not pd.isna(row[col]):
                            # Format based on data type
                            if isinstance(row[col], (int, float)):
                                tx_props.append(f"{col}: {row[col]}")
                            else:
                                tx_props.append(f"{col}: '{row[col]}'")
                
                # Create transaction relationship
                props_str = ", ".join(tx_props)
                cypher_queries.append(
                    f"MERGE (from)-[:TRANSFERS {{{props_str}}}]->(to) "
                )
            
            # Execute Cypher queries
            try:
                # Combine queries and execute as a single transaction
                combined_query = "\n".join(cypher_queries)
                self.neo4j_client.run_query(combined_query)
                results["imported_transactions"] += len(batch_df)
            except Exception as e:
                error_msg = f"Error importing batch {i//batch_size + 1}: {str(e)}"
                self.logger.error(error_msg)
                results["errors"].append(error_msg)
        
        # Create summary
        results["summary"] = {
            "success_rate": (results["imported_transactions"] / results["total_transactions"]) * 100 if results["total_transactions"] > 0 else 0,
            "error_count": len(results["errors"]),
            "unique_addresses": len(set(df[from_address_column]).union(set(df[to_address_column])))
        }
        
        return results
