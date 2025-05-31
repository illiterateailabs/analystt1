"""
Cryptocurrency CSV Loader Tool for importing transaction data into Neo4j.

This module provides a tool for loading cryptocurrency transaction data from
CSV files into Neo4j, creating appropriate graph schema, and calculating
derived features for analysis.

Supported CSV formats:
- Ethereum transactions (from_address, to_address, value, etc.)
- Token transfers (token_address, from_address, to_address, value, etc.)
- DEX trades (trader, token_bought, token_sold, amount_bought, amount_sold, etc.)
- Address-level data (address, balance, transaction_count, etc.)
"""

import os
import logging
from typing import Dict, List, Union, Optional, Any, Tuple
import time
from datetime import datetime
import pandas as pd
import numpy as np
import json
import tempfile
from pathlib import Path

# Import Neo4j client if available in the project
try:
    from backend.integrations.neo4j_client import Neo4jClient
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    logging.warning("Neo4j client not available. Neo4j data loading will not be supported.")

class CryptoCSVLoaderTool:
    """
    Tool for loading cryptocurrency transaction data from CSV files into Neo4j.
    
    This tool supports various CSV formats, creates appropriate graph schema,
    handles large files with batch processing, adds proper indexes for performance,
    calculates derived features, and generates import statistics.
    """
    
    # Define CSV format types
    CSV_FORMAT_ETH_TRANSACTIONS = "eth_transactions"
    CSV_FORMAT_TOKEN_TRANSFERS = "token_transfers"
    CSV_FORMAT_DEX_TRADES = "dex_trades"
    CSV_FORMAT_ADDRESS_DATA = "address_data"
    CSV_FORMAT_CUSTOM = "custom"
    
    # Define Neo4j label and relationship types
    LABEL_ADDRESS = "Address"
    LABEL_TRANSACTION = "Transaction"
    LABEL_TOKEN = "Token"
    LABEL_DEX = "DEX"
    REL_TRANSFERS = "TRANSFERS"
    REL_INTERACTS_WITH = "INTERACTS_WITH"
    REL_TRADES = "TRADES"
    REL_HOLDS = "HOLDS"
    
    def __init__(self, neo4j_client=None):
        """
        Initialize the CryptoCSVLoaderTool.
        
        Args:
            neo4j_client: Optional Neo4j client for database access
        """
        self.neo4j_client = neo4j_client
        self.logger = logging.getLogger(__name__)
        
        # Check for Neo4j client
        if neo4j_client is None and NEO4J_AVAILABLE:
            try:
                self.neo4j_client = Neo4jClient()
                self.logger.info("Connected to Neo4j database")
            except Exception as e:
                self.logger.warning(f"Could not initialize Neo4j client: {str(e)}")
                
        if not NEO4J_AVAILABLE or self.neo4j_client is None:
            self.logger.warning("Neo4j client not available. Tool will not be able to load data.")
    
    def run(self, operation: str, **kwargs) -> Dict[str, Any]:
        """
        Run a specific CSV loading operation.
        
        Args:
            operation: The operation to run. Options include:
                - 'load_csv': Load CSV data into Neo4j
                - 'create_schema': Create Neo4j schema (indexes, constraints)
                - 'calculate_metrics': Calculate derived metrics for addresses
                - 'validate_data': Validate CSV data without loading
                - 'get_schema_info': Get information about the current Neo4j schema
            **kwargs: Additional arguments specific to each operation
                
        Returns:
            Dict containing the results of the operation
        
        Raises:
            ValueError: If an invalid operation is specified
            RuntimeError: If Neo4j client is not available
        """
        self.logger.info(f"Running crypto CSV loader: {operation}")
        
        # Check if Neo4j client is available
        if not NEO4J_AVAILABLE or self.neo4j_client is None:
            raise RuntimeError("Neo4j client not available. Cannot run operation.")
        
        try:
            if operation == 'load_csv':
                return self._load_csv(**kwargs)
            elif operation == 'create_schema':
                return self._create_schema(**kwargs)
            elif operation == 'calculate_metrics':
                return self._calculate_metrics(**kwargs)
            elif operation == 'validate_data':
                return self._validate_data(**kwargs)
            elif operation == 'get_schema_info':
                return self._get_schema_info(**kwargs)
            else:
                raise ValueError(f"Invalid operation: {operation}")
        except Exception as e:
            self.logger.error(f"Error in CryptoCSVLoaderTool.run: {str(e)}")
            raise
    
    def _load_csv(self,
                 csv_path: str,
                 csv_format: str = CSV_FORMAT_ETH_TRANSACTIONS,
                 batch_size: int = 1000,
                 create_schema: bool = True,
                 create_indexes: bool = True,
                 calculate_metrics: bool = True,
                 column_mapping: Optional[Dict[str, str]] = None,
                 date_columns: Optional[List[str]] = None,
                 numeric_columns: Optional[List[str]] = None,
                 skip_validation: bool = False,
                 delimiter: str = ',',
                 encoding: str = 'utf-8',
                 **kwargs) -> Dict[str, Any]:
        """
        Load CSV data into Neo4j.
        
        Args:
            csv_path: Path to the CSV file
            csv_format: Format of the CSV file (eth_transactions, token_transfers, dex_trades, address_data, custom)
            batch_size: Number of rows to process in each batch
            create_schema: Whether to create schema (indexes, constraints) before loading
            create_indexes: Whether to create indexes on key properties
            calculate_metrics: Whether to calculate derived metrics after loading
            column_mapping: Optional mapping of CSV columns to Neo4j properties
            date_columns: Optional list of columns to parse as dates
            numeric_columns: Optional list of columns to ensure are numeric
            skip_validation: Whether to skip data validation
            delimiter: CSV delimiter character
            encoding: CSV file encoding
            **kwargs: Additional arguments
                
        Returns:
            Dict containing the results of the operation
            
        Raises:
            FileNotFoundError: If the CSV file does not exist
            ValueError: If the CSV format is invalid or required columns are missing
        """
        # Check if file exists
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        # Initialize results
        results = {
            "operation": "load_csv",
            "csv_path": csv_path,
            "csv_format": csv_format,
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "total_rows": 0,
            "processed_rows": 0,
            "skipped_rows": 0,
            "error_rows": 0,
            "batches": 0,
            "errors": [],
            "warnings": [],
            "schema_created": False,
            "indexes_created": False,
            "metrics_calculated": False,
            "summary": {}
        }
        
        # Start timing
        start_time = time.time()
        
        try:
            # Read CSV file (first few rows to get column info)
            df_sample = pd.read_csv(csv_path, nrows=5, delimiter=delimiter, encoding=encoding)
            
            # Get total number of rows (without loading entire file)
            with open(csv_path, 'r', encoding=encoding) as f:
                results["total_rows"] = sum(1 for _ in f) - 1  # Subtract header row
            
            # Validate CSV format and get required columns
            required_columns, optional_columns = self._get_required_columns(csv_format)
            
            # Apply column mapping if provided
            if column_mapping:
                # Create a mapping from CSV columns to required columns
                mapped_required = []
                for req_col in required_columns:
                    # Find the CSV column that maps to this required column
                    csv_col = next((csv_col for csv_col, neo4j_col in column_mapping.items() 
                                   if neo4j_col == req_col), None)
                    if csv_col:
                        mapped_required.append(csv_col)
                    else:
                        mapped_required.append(req_col)  # Keep original if no mapping
                
                # Replace required columns with mapped columns
                required_columns = mapped_required
            
            # Check if required columns exist
            missing_columns = [col for col in required_columns if col not in df_sample.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
            
            # Create schema if requested
            if create_schema:
                schema_result = self._create_schema(csv_format=csv_format)
                results["schema_created"] = schema_result.get("success", False)
                results["schema_details"] = schema_result
            
            # Create indexes if requested
            if create_indexes:
                index_result = self._create_indexes(csv_format=csv_format)
                results["indexes_created"] = index_result.get("success", False)
                results["index_details"] = index_result
            
            # Process CSV in batches
            with pd.read_csv(csv_path, chunksize=batch_size, delimiter=delimiter, encoding=encoding) as reader:
                for batch_num, df_chunk in enumerate(reader):
                    batch_result = self._process_batch(
                        df_chunk, 
                        csv_format=csv_format, 
                        column_mapping=column_mapping,
                        date_columns=date_columns,
                        numeric_columns=numeric_columns,
                        skip_validation=skip_validation,
                        batch_num=batch_num
                    )
                    
                    # Update results
                    results["processed_rows"] += batch_result.get("processed_rows", 0)
                    results["skipped_rows"] += batch_result.get("skipped_rows", 0)
                    results["error_rows"] += batch_result.get("error_rows", 0)
                    results["batches"] += 1
                    
                    # Add any errors or warnings
                    if "errors" in batch_result and batch_result["errors"]:
                        results["errors"].extend(batch_result["errors"])
                    if "warnings" in batch_result and batch_result["warnings"]:
                        results["warnings"].extend(batch_result["warnings"])
                    
                    # Log progress every 10 batches
                    if batch_num % 10 == 0:
                        self.logger.info(f"Processed {results['processed_rows']} rows in {results['batches']} batches")
            
            # Calculate metrics if requested
            if calculate_metrics:
                metrics_result = self._calculate_metrics(csv_format=csv_format)
                results["metrics_calculated"] = metrics_result.get("success", False)
                results["metrics_details"] = metrics_result
            
            # Create summary
            results["summary"] = {
                "success_rate": (results["processed_rows"] / results["total_rows"]) * 100 if results["total_rows"] > 0 else 0,
                "error_rate": (results["error_rows"] / results["total_rows"]) * 100 if results["total_rows"] > 0 else 0,
                "processing_time_seconds": time.time() - start_time,
                "rows_per_second": results["processed_rows"] / (time.time() - start_time) if (time.time() - start_time) > 0 else 0,
                "total_errors": len(results["errors"]),
                "total_warnings": len(results["warnings"])
            }
            
            # Add format-specific summary
            if csv_format == self.CSV_FORMAT_ETH_TRANSACTIONS:
                # Get counts of addresses and transactions
                address_count = self._count_nodes(self.LABEL_ADDRESS)
                transaction_count = self._count_relationships(self.REL_TRANSFERS)
                results["summary"]["unique_addresses"] = address_count
                results["summary"]["total_transactions"] = transaction_count
            
            elif csv_format == self.CSV_FORMAT_TOKEN_TRANSFERS:
                # Get counts of addresses, tokens, and transfers
                address_count = self._count_nodes(self.LABEL_ADDRESS)
                token_count = self._count_nodes(self.LABEL_TOKEN)
                transfer_count = self._count_relationships(self.REL_TRANSFERS)
                results["summary"]["unique_addresses"] = address_count
                results["summary"]["unique_tokens"] = token_count
                results["summary"]["total_transfers"] = transfer_count
            
            elif csv_format == self.CSV_FORMAT_DEX_TRADES:
                # Get counts of addresses, tokens, and trades
                address_count = self._count_nodes(self.LABEL_ADDRESS)
                token_count = self._count_nodes(self.LABEL_TOKEN)
                dex_count = self._count_nodes(self.LABEL_DEX)
                trade_count = self._count_relationships(self.REL_TRADES)
                results["summary"]["unique_addresses"] = address_count
                results["summary"]["unique_tokens"] = token_count
                results["summary"]["unique_dexes"] = dex_count
                results["summary"]["total_trades"] = trade_count
            
            elif csv_format == self.CSV_FORMAT_ADDRESS_DATA:
                # Get count of addresses
                address_count = self._count_nodes(self.LABEL_ADDRESS)
                results["summary"]["unique_addresses"] = address_count
            
        except Exception as e:
            self.logger.error(f"Error loading CSV: {str(e)}")
            results["errors"].append(str(e))
            results["success"] = False
        else:
            results["success"] = True
        
        # Set end time
        results["end_time"] = datetime.now().isoformat()
        
        return results
    
    def _process_batch(self,
                      df: pd.DataFrame,
                      csv_format: str,
                      column_mapping: Optional[Dict[str, str]] = None,
                      date_columns: Optional[List[str]] = None,
                      numeric_columns: Optional[List[str]] = None,
                      skip_validation: bool = False,
                      batch_num: int = 0) -> Dict[str, Any]:
        """
        Process a batch of CSV data and load it into Neo4j.
        
        Args:
            df: DataFrame containing the batch of data
            csv_format: Format of the CSV file
            column_mapping: Optional mapping of CSV columns to Neo4j properties
            date_columns: Optional list of columns to parse as dates
            numeric_columns: Optional list of columns to ensure are numeric
            skip_validation: Whether to skip data validation
            batch_num: Batch number for logging
            
        Returns:
            Dict containing the results of the batch processing
        """
        # Initialize batch results
        batch_results = {
            "batch_num": batch_num,
            "total_rows": len(df),
            "processed_rows": 0,
            "skipped_rows": 0,
            "error_rows": 0,
            "errors": [],
            "warnings": []
        }
        
        try:
            # Apply column mapping if provided
            if column_mapping:
                # Rename columns according to mapping
                df = df.rename(columns={csv_col: neo4j_col for csv_col, neo4j_col in column_mapping.items()
                                       if csv_col in df.columns})
            
            # Convert date columns
            if date_columns:
                for col in date_columns:
                    if col in df.columns:
                        try:
                            df[col] = pd.to_datetime(df[col])
                        except Exception as e:
                            batch_results["warnings"].append(f"Failed to convert column '{col}' to datetime: {str(e)}")
            
            # Ensure numeric columns are numeric
            if numeric_columns:
                for col in numeric_columns:
                    if col in df.columns:
                        try:
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                        except Exception as e:
                            batch_results["warnings"].append(f"Failed to convert column '{col}' to numeric: {str(e)}")
            
            # Validate data if not skipped
            if not skip_validation:
                validation_result = self._validate_batch(df, csv_format)
                if not validation_result["valid"]:
                    # Filter out invalid rows
                    invalid_indices = validation_result.get("invalid_indices", [])
                    if invalid_indices:
                        df = df.drop(invalid_indices)
                        batch_results["skipped_rows"] += len(invalid_indices)
                        batch_results["warnings"].append(f"Skipped {len(invalid_indices)} invalid rows")
            
            # Process based on CSV format
            if csv_format == self.CSV_FORMAT_ETH_TRANSACTIONS:
                result = self._process_eth_transactions(df)
            elif csv_format == self.CSV_FORMAT_TOKEN_TRANSFERS:
                result = self._process_token_transfers(df)
            elif csv_format == self.CSV_FORMAT_DEX_TRADES:
                result = self._process_dex_trades(df)
            elif csv_format == self.CSV_FORMAT_ADDRESS_DATA:
                result = self._process_address_data(df)
            elif csv_format == self.CSV_FORMAT_CUSTOM:
                result = self._process_custom(df)
            else:
                raise ValueError(f"Unsupported CSV format: {csv_format}")
            
            # Update batch results
            batch_results["processed_rows"] = result.get("processed_rows", 0)
            batch_results["error_rows"] += result.get("error_rows", 0)
            
            # Add any errors or warnings
            if "errors" in result and result["errors"]:
                batch_results["errors"].extend(result["errors"])
            if "warnings" in result and result["warnings"]:
                batch_results["warnings"].extend(result["warnings"])
            
        except Exception as e:
            self.logger.error(f"Error processing batch {batch_num}: {str(e)}")
            batch_results["errors"].append(str(e))
            batch_results["error_rows"] += len(df)
        
        return batch_results
    
    def _validate_batch(self, df: pd.DataFrame, csv_format: str) -> Dict[str, Any]:
        """
        Validate a batch of CSV data.
        
        Args:
            df: DataFrame containing the batch of data
            csv_format: Format of the CSV file
            
        Returns:
            Dict containing validation results
        """
        # Initialize validation results
        validation_results = {
            "valid": True,
            "invalid_indices": [],
            "warnings": []
        }
        
        # Get required columns for the format
        required_columns, _ = self._get_required_columns(csv_format)
        
        # Check for missing values in required columns
        for col in required_columns:
            if col in df.columns:
                missing_indices = df[df[col].isna()].index.tolist()
                if missing_indices:
                    validation_results["invalid_indices"].extend(missing_indices)
                    validation_results["warnings"].append(f"Missing values in required column '{col}'")
        
        # Format-specific validations
        if csv_format == self.CSV_FORMAT_ETH_TRANSACTIONS:
            # Validate addresses (should be 42 characters starting with 0x)
            if 'from_address' in df.columns:
                invalid_from = df[~df['from_address'].astype(str).str.match(r'^0x[a-fA-F0-9]{40}$')].index.tolist()
                validation_results["invalid_indices"].extend(invalid_from)
            
            if 'to_address' in df.columns:
                invalid_to = df[~df['to_address'].astype(str).str.match(r'^0x[a-fA-F0-9]{40}$')].index.tolist()
                validation_results["invalid_indices"].extend(invalid_to)
            
            # Validate transaction values (should be numeric and non-negative)
            if 'value' in df.columns:
                try:
                    df['value'] = pd.to_numeric(df['value'], errors='coerce')
                    invalid_value = df[(df['value'].isna()) | (df['value'] < 0)].index.tolist()
                    validation_results["invalid_indices"].extend(invalid_value)
                except Exception:
                    validation_results["warnings"].append("Failed to validate 'value' column")
        
        elif csv_format == self.CSV_FORMAT_TOKEN_TRANSFERS:
            # Validate token address
            if 'token_address' in df.columns:
                invalid_token = df[~df['token_address'].astype(str).str.match(r'^0x[a-fA-F0-9]{40}$')].index.tolist()
                validation_results["invalid_indices"].extend(invalid_token)
        
        # Remove duplicates from invalid_indices
        validation_results["invalid_indices"] = list(set(validation_results["invalid_indices"]))
        
        # Set valid flag
        validation_results["valid"] = len(validation_results["invalid_indices"]) == 0
        
        return validation_results
    
    def _process_eth_transactions(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Process Ethereum transactions and load them into Neo4j.
        
        Args:
            df: DataFrame containing Ethereum transactions
            
        Returns:
            Dict containing processing results
        """
        # Initialize results
        results = {
            "processed_rows": 0,
            "error_rows": 0,
            "errors": [],
            "warnings": []
        }
        
        try:
            # Create Cypher queries for this batch
            cypher_queries = []
            
            # Process each row
            for idx, row in df.iterrows():
                try:
                    # Get required fields
                    from_addr = str(row['from_address'])
                    to_addr = str(row['to_address'])
                    
                    # Skip rows with missing addresses
                    if pd.isna(from_addr) or pd.isna(to_addr):
                        results["error_rows"] += 1
                        continue
                    
                    # Create address nodes
                    cypher_queries.append(
                        f"MERGE (from:{self.LABEL_ADDRESS} {{address: '{from_addr}'}}) "
                        f"MERGE (to:{self.LABEL_ADDRESS} {{address: '{to_addr}'}}) "
                    )
                    
                    # Build transaction properties
                    tx_props = []
                    
                    # Add transaction hash if available
                    if 'hash' in row and not pd.isna(row['hash']):
                        tx_props.append(f"hash: '{row['hash']}'")
                    
                    # Add value if available
                    if 'value' in row and not pd.isna(row['value']):
                        tx_props.append(f"value: {float(row['value'])}")
                    
                    # Add gas-related properties if available
                    for gas_prop in ['gas', 'gas_price', 'gas_used']:
                        if gas_prop in row and not pd.isna(row[gas_prop]):
                            tx_props.append(f"{gas_prop}: {float(row[gas_prop])}")
                    
                    # Add timestamp if available
                    if 'timestamp' in row and not pd.isna(row['timestamp']):
                        # Format timestamp as Neo4j datetime
                        try:
                            timestamp = pd.to_datetime(row['timestamp'])
                            tx_props.append(f"timestamp: datetime('{timestamp.isoformat()}')")
                        except:
                            pass
                    
                    # Add block number if available
                    if 'block_number' in row and not pd.isna(row['block_number']):
                        tx_props.append(f"block_number: {int(row['block_number'])}")
                    
                    # Create transaction relationship
                    props_str = ", ".join(tx_props)
                    cypher_queries.append(
                        f"MERGE (from)-[:{self.REL_TRANSFERS} {{{props_str}}}]->(to) "
                    )
                    
                    results["processed_rows"] += 1
                    
                except Exception as e:
                    results["error_rows"] += 1
                    results["errors"].append(f"Error processing row {idx}: {str(e)}")
            
            # Execute Cypher queries
            if cypher_queries:
                # Combine queries and execute as a single transaction
                combined_query = "\n".join(cypher_queries)
                self.neo4j_client.run_query(combined_query)
            
        except Exception as e:
            self.logger.error(f"Error processing Ethereum transactions: {str(e)}")
            results["errors"].append(str(e))
            results["error_rows"] += len(df) - results["processed_rows"]
        
        return results
    
    def _process_token_transfers(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Process token transfers and load them into Neo4j.
        
        Args:
            df: DataFrame containing token transfers
            
        Returns:
            Dict containing processing results
        """
        # Initialize results
        results = {
            "processed_rows": 0,
            "error_rows": 0,
            "errors": [],
            "warnings": []
        }
        
        try:
            # Create Cypher queries for this batch
            cypher_queries = []
            
            # Process each row
            for idx, row in df.iterrows():
                try:
                    # Get required fields
                    token_addr = str(row['token_address'])
                    from_addr = str(row['from_address'])
                    to_addr = str(row['to_address'])
                    
                    # Skip rows with missing addresses
                    if pd.isna(token_addr) or pd.isna(from_addr) or pd.isna(to_addr):
                        results["error_rows"] += 1
                        continue
                    
                    # Create token node
                    token_props = [f"address: '{token_addr}'"]
                    
                    # Add token name and symbol if available
                    if 'token_name' in row and not pd.isna(row['token_name']):
                        token_props.append(f"name: '{row['token_name']}'")
                    
                    if 'token_symbol' in row and not pd.isna(row['token_symbol']):
                        token_props.append(f"symbol: '{row['token_symbol']}'")
                    
                    if 'token_decimals' in row and not pd.isna(row['token_decimals']):
                        token_props.append(f"decimals: {int(row['token_decimals'])}")
                    
                    token_props_str = ", ".join(token_props)
                    cypher_queries.append(
                        f"MERGE (token:{self.LABEL_TOKEN} {{{token_props_str}}}) "
                        f"MERGE (from:{self.LABEL_ADDRESS} {{address: '{from_addr}'}}) "
                        f"MERGE (to:{self.LABEL_ADDRESS} {{address: '{to_addr}'}}) "
                    )
                    
                    # Build transfer properties
                    transfer_props = []
                    
                    # Add value if available
                    if 'value' in row and not pd.isna(row['value']):
                        transfer_props.append(f"value: {float(row['value'])}")
                    
                    # Add timestamp if available
                    if 'timestamp' in row and not pd.isna(row['timestamp']):
                        # Format timestamp as Neo4j datetime
                        try:
                            timestamp = pd.to_datetime(row['timestamp'])
                            transfer_props.append(f"timestamp: datetime('{timestamp.isoformat()}')")
                        except:
                            pass
                    
                    # Add transaction hash if available
                    if 'transaction_hash' in row and not pd.isna(row['transaction_hash']):
                        transfer_props.append(f"transaction_hash: '{row['transaction_hash']}'")
                    
                    # Create transfer relationship
                    transfer_props_str = ", ".join(transfer_props)
                    cypher_queries.append(
                        f"MERGE (from)-[t:{self.REL_TRANSFERS} {{{transfer_props_str}}}]->(to) "
                        f"SET t.token_address = '{token_addr}' "
                    )
                    
                    # Create HOLDS relationships
                    cypher_queries.append(
                        f"MERGE (from)-[:{self.REL_HOLDS}]->(token) "
                        f"MERGE (to)-[:{self.REL_HOLDS}]->(token) "
                    )
                    
                    results["processed_rows"] += 1
                    
                except Exception as e:
                    results["error_rows"] += 1
                    results["errors"].append(f"Error processing row {idx}: {str(e)}")
            
            # Execute Cypher queries
            if cypher_queries:
                # Combine queries and execute as a single transaction
                combined_query = "\n".join(cypher_queries)
                self.neo4j_client.run_query(combined_query)
            
        except Exception as e:
            self.logger.error(f"Error processing token transfers: {str(e)}")
            results["errors"].append(str(e))
            results["error_rows"] += len(df) - results["processed_rows"]
        
        return results
    
    def _process_dex_trades(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Process DEX trades and load them into Neo4j.
        
        Args:
            df: DataFrame containing DEX trades
            
        Returns:
            Dict containing processing results
        """
        # Initialize results
        results = {
            "processed_rows": 0,
            "error_rows": 0,
            "errors": [],
            "warnings": []
        }
        
        try:
            # Create Cypher queries for this batch
            cypher_queries = []
            
            # Process each row
            for idx, row in df.iterrows():
                try:
                    # Get required fields
                    trader_addr = str(row['trader_address'])
                    dex_name = str(row['dex_name'])
                    token_bought = str(row['token_bought'])
                    token_sold = str(row['token_sold'])
                    
                    # Skip rows with missing data
                    if pd.isna(trader_addr) or pd.isna(dex_name) or pd.isna(token_bought) or pd.isna(token_sold):
                        results["error_rows"] += 1
                        continue
                    
                    # Create nodes
                    cypher_queries.append(
                        f"MERGE (trader:{self.LABEL_ADDRESS} {{address: '{trader_addr}'}}) "
                        f"MERGE (dex:{self.LABEL_DEX} {{name: '{dex_name}'}}) "
                        f"MERGE (bought:{self.LABEL_TOKEN} {{address: '{token_bought}'}}) "
                        f"MERGE (sold:{self.LABEL_TOKEN} {{address: '{token_sold}'}}) "
                    )
                    
                    # Build trade properties
                    trade_props = []
                    
                    # Add amounts if available
                    if 'amount_bought' in row and not pd.isna(row['amount_bought']):
                        trade_props.append(f"amount_bought: {float(row['amount_bought'])}")
                    
                    if 'amount_sold' in row and not pd.isna(row['amount_sold']):
                        trade_props.append(f"amount_sold: {float(row['amount_sold'])}")
                    
                    # Add timestamp if available
                    if 'timestamp' in row and not pd.isna(row['timestamp']):
                        # Format timestamp as Neo4j datetime
                        try:
                            timestamp = pd.to_datetime(row['timestamp'])
                            trade_props.append(f"timestamp: datetime('{timestamp.isoformat()}')")
                        except:
                            pass
                    
                    # Add transaction hash if available
                    if 'transaction_hash' in row and not pd.isna(row['transaction_hash']):
                        trade_props.append(f"transaction_hash: '{row['transaction_hash']}'")
                    
                    # Create trade relationship
                    trade_props_str = ", ".join(trade_props)
                    cypher_queries.append(
                        f"MERGE (trader)-[t:{self.REL_TRADES} {{{trade_props_str}}}]->(dex) "
                        f"SET t.token_bought = '{token_bought}', "
                        f"    t.token_sold = '{token_sold}' "
                    )
                    
                    # Create INTERACTS_WITH relationships
                    cypher_queries.append(
                        f"MERGE (trader)-[:{self.REL_INTERACTS_WITH}]->(bought) "
                        f"MERGE (trader)-[:{self.REL_INTERACTS_WITH}]->(sold) "
                    )
                    
                    results["processed_rows"] += 1
                    
                except Exception as e:
                    results["error_rows"] += 1
                    results["errors"].append(f"Error processing row {idx}: {str(e)}")
            
            # Execute Cypher queries
            if cypher_queries:
                # Combine queries and execute as a single transaction
                combined_query = "\n".join(cypher_queries)
                self.neo4j_client.run_query(combined_query)
            
        except Exception as e:
            self.logger.error(f"Error processing DEX trades: {str(e)}")
            results["errors"].append(str(e))
            results["error_rows"] += len(df) - results["processed_rows"]
        
        return results
    
    def _process_address_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Process address-level data and load it into Neo4j.
        
        Args:
            df: DataFrame containing address data
            
        Returns:
            Dict containing processing results
        """
        # Initialize results
        results = {
            "processed_rows": 0,
            "error_rows": 0,
            "errors": [],
            "warnings": []
        }
        
        try:
            # Create Cypher queries for this batch
            cypher_queries = []
            
            # Process each row
            for idx, row in df.iterrows():
                try:
                    # Get required fields
                    address = str(row['address'])
                    
                    # Skip rows with missing address
                    if pd.isna(address):
                        results["error_rows"] += 1
                        continue
                    
                    # Build address properties
                    addr_props = [f"address: '{address}'"]
                    
                    # Add all other columns as properties
                    for col in df.columns:
                        if col != 'address' and not pd.isna(row[col]):
                            # Format based on data type
                            if isinstance(row[col], (int, float, bool)):
                                addr_props.append(f"{col}: {row[col]}")
                            else:
                                # Escape single quotes in string values
                                str_val = str(row[col]).replace("'", "\\'")
                                addr_props.append(f"{col}: '{str_val}'")
                    
                    # Create or update address node
                    addr_props_str = ", ".join(addr_props)
                    cypher_queries.append(
                        f"MERGE (addr:{self.LABEL_ADDRESS} {{address: '{address}'}}) "
                        f"SET addr += {{{addr_props_str}}} "
                    )
                    
                    results["processed_rows"] += 1
                    
                except Exception as e:
                    results["error_rows"] += 1
                    results["errors"].append(f"Error processing row {idx}: {str(e)}")
            
            # Execute Cypher queries
            if cypher_queries:
                # Combine queries and execute as a single transaction
                combined_query = "\n".join(cypher_queries)
                self.neo4j_client.run_query(combined_query)
            
        except Exception as e:
            self.logger.error(f"Error processing address data: {str(e)}")
            results["errors"].append(str(e))
            results["error_rows"] += len(df) - results["processed_rows"]
        
        return results
    
    def _process_custom(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Process custom CSV format.
        
        Args:
            df: DataFrame containing custom data
            
        Returns:
            Dict containing processing results
        """
        # This is a placeholder for custom CSV processing
        # The implementation would depend on the specific custom format
        return {
            "processed_rows": 0,
            "error_rows": len(df),
            "errors": ["Custom CSV format processing not implemented"],
            "warnings": []
        }
    
    def _create_schema(self, csv_format: str = None) -> Dict[str, Any]:
        """
        Create Neo4j schema (constraints and indexes) for the specified CSV format.
        
        Args:
            csv_format: Format of the CSV file
            
        Returns:
            Dict containing the results of the operation
        """
        # Initialize results
        results = {
            "operation": "create_schema",
            "csv_format": csv_format,
            "constraints_created": 0,
            "indexes_created": 0,
            "errors": [],
            "success": False
        }
        
        try:
            # Create constraints based on CSV format
            constraints = []
            
            # Common constraints for all formats
            constraints.append(
                f"CREATE CONSTRAINT IF NOT EXISTS FOR (a:{self.LABEL_ADDRESS}) REQUIRE a.address IS UNIQUE"
            )
            
            # Format-specific constraints
            if csv_format == self.CSV_FORMAT_TOKEN_TRANSFERS or csv_format == self.CSV_FORMAT_DEX_TRADES:
                constraints.append(
                    f"CREATE CONSTRAINT IF NOT EXISTS FOR (t:{self.LABEL_TOKEN}) REQUIRE t.address IS UNIQUE"
                )
            
            if csv_format == self.CSV_FORMAT_DEX_TRADES:
                constraints.append(
                    f"CREATE CONSTRAINT IF NOT EXISTS FOR (d:{self.LABEL_DEX}) REQUIRE d.name IS UNIQUE"
                )
            
            # Execute constraints
            for constraint in constraints:
                try:
                    self.neo4j_client.run_query(constraint)
                    results["constraints_created"] += 1
                except Exception as e:
                    results["errors"].append(f"Error creating constraint: {str(e)}")
            
            results["success"] = len(results["errors"]) == 0
            
        except Exception as e:
            self.logger.error(f"Error creating schema: {str(e)}")
            results["errors"].append(str(e))
            results["success"] = False
        
        return results
    
    def _create_indexes(self, csv_format: str = None) -> Dict[str, Any]:
        """
        Create Neo4j indexes for the specified CSV format.
        
        Args:
            csv_format: Format of the CSV file
            
        Returns:
            Dict containing the results of the operation
        """
        # Initialize results
        results = {
            "operation": "create_indexes",
            "csv_format": csv_format,
            "indexes_created": 0,
            "errors": [],
            "success": False
        }
        
        try:
            # Create indexes based on CSV format
            indexes = []
            
            # Common indexes for all formats
            # Note: We don't need to create an index on address since we already have a constraint
            
            # Format-specific indexes
            if csv_format == self.CSV_FORMAT_ETH_TRANSACTIONS:
                indexes.append(
                    f"CREATE INDEX IF NOT EXISTS FOR ()-[t:{self.REL_TRANSFERS}]-() ON (t.hash)"
                )
                indexes.append(
                    f"CREATE INDEX IF NOT EXISTS FOR ()-[t:{self.REL_TRANSFERS}]-() ON (t.timestamp)"
                )
                indexes.append(
                    f"CREATE INDEX IF NOT EXISTS FOR ()-[t:{self.REL_TRANSFERS}]-() ON (t.block_number)"
                )
            
            elif csv_format == self.CSV_FORMAT_TOKEN_TRANSFERS:
                indexes.append(
                    f"CREATE INDEX IF NOT EXISTS FOR ()-[t:{self.REL_TRANSFERS}]-() ON (t.token_address)"
                )
                indexes.append(
                    f"CREATE INDEX IF NOT EXISTS FOR ()-[t:{self.REL_TRANSFERS}]-() ON (t.timestamp)"
                )
                indexes.append(
                    f"CREATE INDEX IF NOT EXISTS FOR ()-[t:{self.REL_TRANSFERS}]-() ON (t.transaction_hash)"
                )
            
            elif csv_format == self.CSV_FORMAT_DEX_TRADES:
                indexes.append(
                    f"CREATE INDEX IF NOT EXISTS FOR ()-[t:{self.REL_TRADES}]-() ON (t.token_bought)"
                )
                indexes.append(
                    f"CREATE INDEX IF NOT EXISTS FOR ()-[t:{self.REL_TRADES}]-() ON (t.token_sold)"
                )
                indexes.append(
                    f"CREATE INDEX IF NOT EXISTS FOR ()-[t:{self.REL_TRADES}]-() ON (t.timestamp)"
                )
                indexes.append(
                    f"CREATE INDEX IF NOT EXISTS FOR ()-[t:{self.REL_TRADES}]-() ON (t.transaction_hash)"
                )
            
            # Execute indexes
            for index in indexes:
                try:
                    self.neo4j_client.run_query(index)
                    results["indexes_created"] += 1
                except Exception as e:
                    results["errors"].append(f"Error creating index: {str(e)}")
            
            results["success"] = len(results["errors"]) == 0
            
        except Exception as e:
            self.logger.error(f"Error creating indexes: {str(e)}")
            results["errors"].append(str(e))
            results["success"] = False
        
        return results
    
    def _calculate_metrics(self, csv_format: str = None) -> Dict[str, Any]:
        """
        Calculate derived metrics for addresses based on the loaded data.
        
        Args:
            csv_format: Format of the CSV file
            
        Returns:
            Dict containing the results of the operation
        """
        # Initialize results
        results = {
            "operation": "calculate_metrics",
            "csv_format": csv_format,
            "metrics_calculated": 0,
            "addresses_updated": 0,
            "errors": [],
            "success": False
        }
        
        try:
            # Define metrics to calculate based on CSV format
            metrics = []
            
            # Common metrics for all formats
            metrics.append("""
                MATCH (a:Address)
                SET a.last_updated = datetime()
            """)
            
            # Format-specific metrics
            if csv_format == self.CSV_FORMAT_ETH_TRANSACTIONS:
                # Calculate transaction counts and volumes
                metrics.append("""
                    MATCH (a:Address)
                    OPTIONAL MATCH (a)-[t:TRANSFERS]->()
                    WITH a, count(t) as outgoing_count, sum(t.value) as outgoing_volume
                    SET a.outgoing_tx_count = outgoing_count,
                        a.outgoing_volume = outgoing_volume
                """)
                
                metrics.append("""
                    MATCH (a:Address)
                    OPTIONAL MATCH ()-[t:TRANSFERS]->(a)
                    WITH a, count(t) as incoming_count, sum(t.value) as incoming_volume
                    SET a.incoming_tx_count = incoming_count,
                        a.incoming_volume = incoming_volume
                """)
                
                # Calculate net flow
                metrics.append("""
                    MATCH (a:Address)
                    SET a.net_flow = coalesce(a.incoming_volume, 0) - coalesce(a.outgoing_volume, 0)
                """)
                
                # Calculate unique counterparties
                metrics.append("""
                    MATCH (a:Address)
                    OPTIONAL MATCH (a)-[:TRANSFERS]->(counter:Address)
                    WITH a, count(distinct counter) as unique_out
                    SET a.unique_outgoing_counterparties = unique_out
                """)
                
                metrics.append("""
                    MATCH (a:Address)
                    OPTIONAL MATCH (counter:Address)-[:TRANSFERS]->(a)
                    WITH a, count(distinct counter) as unique_in
                    SET a.unique_incoming_counterparties = unique_in
                """)
            
            elif csv_format == self.CSV_FORMAT_TOKEN_TRANSFERS:
                # Calculate token metrics
                metrics.append("""
                    MATCH (a:Address)
                    OPTIONAL MATCH (a)-[h:HOLDS]->(t:Token)
                    WITH a, count(t) as token_count
                    SET a.token_count = token_count
                """)
                
                # Calculate token transfer counts
                metrics.append("""
                    MATCH (a:Address)
                    OPTIONAL MATCH (a)-[t:TRANSFERS]->()
                    WITH a, count(t) as outgoing_count
                    SET a.outgoing_transfer_count = outgoing_count
                """)
                
                metrics.append("""
                    MATCH (a:Address)
                    OPTIONAL MATCH ()-[t:TRANSFERS]->(a)
                    WITH a, count(t) as incoming_count
                    SET a.incoming_transfer_count = incoming_count
                """)
            
            elif csv_format == self.CSV_FORMAT_DEX_TRADES:
                # Calculate DEX metrics
                metrics.append("""
                    MATCH (a:Address)
                    OPTIONAL MATCH (a)-[t:TRADES]->()
                    WITH a, count(t) as trade_count
                    SET a.trade_count = trade_count
                """)
                
                # Calculate unique DEXes used
                metrics.append("""
                    MATCH (a:Address)
                    OPTIONAL MATCH (a)-[:TRADES]->(d:DEX)
                    WITH a, count(distinct d) as dex_count
                    SET a.unique_dexes = dex_count
                """)
                
                # Calculate unique tokens traded
                metrics.append("""
                    MATCH (a:Address)
                    OPTIONAL MATCH (a)-[:INTERACTS_WITH]->(t:Token)
                    WITH a, count(distinct t) as token_count
                    SET a.traded_token_count = token_count
                """)
            
            # Execute metrics calculations
            for metric in metrics:
                try:
                    result = self.neo4j_client.run_query(metric)
                    results["metrics_calculated"] += 1
                except Exception as e:
                    results["errors"].append(f"Error calculating metric: {str(e)}")
            
            # Count updated addresses
            count_query = f"MATCH (a:{self.LABEL_ADDRESS}) WHERE a.last_updated IS NOT NULL RETURN count(a) as count"
            count_result = self.neo4j_client.run_query(count_query)
            results["addresses_updated"] = count_result[0]["count"] if count_result else 0
            
            results["success"] = len(results["errors"]) == 0
            
        except Exception as e:
            self.logger.error(f"Error calculating metrics: {str(e)}")
            results["errors"].append(str(e))
            results["success"] = False
        
        return results
    
    def _validate_data(self, 
                      csv_path: str,
                      csv_format: str = CSV_FORMAT_ETH_TRANSACTIONS,
                      sample_size: int = 1000,
                      **kwargs) -> Dict[str, Any]:
        """
        Validate CSV data without loading it into Neo4j.
        
        Args:
            csv_path: Path to the CSV file
            csv_format: Format of the CSV file
            sample_size: Number of rows to sample for validation
            **kwargs: Additional arguments for data loading
            
        Returns:
            Dict containing validation results
        """
        # Initialize results
        results = {
            "operation": "validate_data",
            "csv_path": csv_path,
            "csv_format": csv_format,
            "total_rows": 0,
            "valid_rows": 0,
            "invalid_rows": 0,
            "sample_size": sample_size,
            "issues": [],
            "column_stats": {},
            "success": False
        }
        
        try:
            # Check if file exists
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"CSV file not found: {csv_path}")
            
            # Get total number of rows (without loading entire file)
            with open(csv_path, 'r', encoding=kwargs.get('encoding', 'utf-8')) as f:
                results["total_rows"] = sum(1 for _ in f) - 1  # Subtract header row
            
            # Read sample of CSV file
            df_sample = pd.read_csv(
                csv_path, 
                nrows=sample_size,
                delimiter=kwargs.get('delimiter', ','),
                encoding=kwargs.get('encoding', 'utf-8')
            )
            
            # Validate CSV format and get required columns
            required_columns, optional_columns = self._get_required_columns(csv_format)
            
            # Check if required columns exist
            missing_columns = [col for col in required_columns if col not in df_sample.columns]
            if missing_columns:
                results["issues"].append(f"Missing required columns: {', '.join(missing_columns)}")
            
            # Validate sample data
            validation_result = self._validate_batch(df_sample, csv_format)
            results["valid_rows"] = sample_size - len(validation_result["invalid_indices"])
            results["invalid_rows"] = len(validation_result["invalid_indices"])
            
            # Add validation warnings to issues
            results["issues"].extend(validation_result.get("warnings", []))
            
            # Calculate column statistics
            results["column_stats"] = self._calculate_column_stats(df_sample)
            
            # Set success flag
            results["success"] = len(missing_columns) == 0
            
        except Exception as e:
            self.logger.error(f"Error validating data: {str(e)}")
            results["issues"].append(str(e))
            results["success"] = False
        
        return results
    
    def _calculate_column_stats(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """
        Calculate statistics for each column in the DataFrame.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dict containing statistics for each column
        """
        stats = {}
        
        for col in df.columns:
            col_stats = {
                "type": str(df[col].dtype),
                "missing_count": df[col].isna().sum(),
                "missing_percentage": (df[col].isna().sum() / len(df)) * 100 if len(df) > 0 else 0
            }
            
            # Add numeric statistics if column is numeric
            if pd.api.types.is_numeric_dtype(df[col]):
                col_stats.update({
                    "min": df[col].min(),
                    "max": df[col].max(),
                    "mean": df[col].mean(),
                    "median": df[col].median(),
                    "std": df[col].std()
                })
            
            # Add string statistics if column is string
            elif pd.api.types.is_string_dtype(df[col]):
                col_stats.update({
                    "unique_count": df[col].nunique(),
                    "unique_percentage": (df[col].nunique() / len(df)) * 100 if len(df) > 0 else 0,
                    "most_common": df[col].value_counts().head(5).to_dict()
                })
            
            stats[col] = col_stats
        
        return stats
    
    def _get_schema_info(self) -> Dict[str, Any]:
        """
        Get information about the current Neo4j schema.
        
        Returns:
            Dict containing schema information
        """
        # Initialize results
        results = {
            "operation": "get_schema_info",
            "node_labels": [],
            "relationship_types": [],
            "constraints": [],
            "indexes": [],
            "node_counts": {},
            "relationship_counts": {},
            "success": False
        }
        
        try:
            # Get node labels
            labels_query = "CALL db.labels()"
            labels_result = self.neo4j_client.run_query(labels_query)
            results["node_labels"] = [record["label"] for record in labels_result]
            
            # Get relationship types
            rel_types_query = "CALL db.relationshipTypes()"
            rel_types_result = self.neo4j_client.run_query(rel_types_query)
            results["relationship_types"] = [record["relationshipType"] for record in rel_types_result]
            
            # Get constraints
            constraints_query = "SHOW CONSTRAINTS"
            constraints_result = self.neo4j_client.run_query(constraints_query)
            results["constraints"] = [record["name"] for record in constraints_result]
            
            # Get indexes
            indexes_query = "SHOW INDEXES"
            indexes_result = self.neo4j_client.run_query(indexes_query)
            results["indexes"] = [record["name"] for record in indexes_result]
            
            # Get node counts for each label
            for label in results["node_labels"]:
                count_query = f"MATCH (:{label}) RETURN count(*) as count"
                count_result = self.neo4j_client.run_query(count_query)
                results["node_counts"][label] = count_result[0]["count"] if count_result else 0
            
            # Get relationship counts for each type
            for rel_type in results["relationship_types"]:
                count_query = f"MATCH ()-[:{rel_type}]->() RETURN count(*) as count"
                count_result = self.neo4j_client.run_query(count_query)
                results["relationship_counts"][rel_type] = count_result[0]["count"] if count_result else 0
            
            results["success"] = True
            
        except Exception as e:
            self.logger.error(f"Error getting schema info: {str(e)}")
            results["error"] = str(e)
            results["success"] = False
        
        return results
    
    def _get_required_columns(self, csv_format: str) -> Tuple[List[str], List[str]]:
        """
        Get the required and optional columns for a specific CSV format.
        
        Args:
            csv_format: Format of the CSV file
            
        Returns:
            Tuple of (required_columns, optional_columns)
            
        Raises:
            ValueError: If the CSV format is invalid
        """
        if csv_format == self.CSV_FORMAT_ETH_TRANSACTIONS:
            required_columns = ['from_address', 'to_address']
            optional_columns = ['hash', 'value', 'gas', 'gas_price', 'gas_used', 'timestamp', 'block_number']
        
        elif csv_format == self.CSV_FORMAT_TOKEN_TRANSFERS:
            required_columns = ['token_address', 'from_address', 'to_address']
            optional_columns = ['value', 'timestamp', 'transaction_hash', 'token_name', 'token_symbol', 'token_decimals']
        
        elif csv_format == self.CSV_FORMAT_DEX_TRADES:
            required_columns = ['trader_address', 'dex_name', 'token_bought', 'token_sold']
            optional_columns = ['amount_bought', 'amount_sold', 'timestamp', 'transaction_hash']
        
        elif csv_format == self.CSV_FORMAT_ADDRESS_DATA:
            required_columns = ['address']
            optional_columns = []  # All other columns are optional
        
        elif csv_format == self.CSV_FORMAT_CUSTOM:
            # For custom format, no required columns are enforced
            required_columns = []
            optional_columns = []
        
        else:
            raise ValueError(f"Invalid CSV format: {csv_format}")
        
        return required_columns, optional_columns
    
    def _count_nodes(self, label: str) -> int:
        """
        Count nodes with a specific label.
        
        Args:
            label: Node label to count
            
        Returns:
            Number of nodes with the label
        """
        query = f"MATCH (:{label}) RETURN count(*) as count"
        result = self.neo4j_client.run_query(query)
        return result[0]["count"] if result else 0
    
    def _count_relationships(self, rel_type: str) -> int:
        """
        Count relationships with a specific type.
        
        Args:
            rel_type: Relationship type to count
            
        Returns:
            Number of relationships with the type
        """
        query = f"MATCH ()-[:{rel_type}]->() RETURN count(*) as count"
        result = self.neo4j_client.run_query(query)
        return result[0]["count"] if result else 0
