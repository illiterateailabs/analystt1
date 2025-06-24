#!/usr/bin/env python3
"""
Demo Data Ingestion and Fraud Detection Script

This script demonstrates the platform's capabilities by:
1. Generating realistic blockchain transaction data with fraud patterns
2. Ingesting sample addresses, transactions, and relationships into Neo4j
3. Creating fraud scenarios (wash trading, smurfing, layering examples)
4. Setting up Redis cache with sample data
5. Demonstrating the platform's detection capabilities
6. Creating sample investigations and evidence

Usage:
    python demo_ingest.py [--addresses N] [--transactions M] [--cleanup]
    python demo_ingest.py --fraud-only  # Only run fraud detection on existing data
    python demo_ingest.py --evidence-only  # Only create evidence bundles

Options:
    --addresses N       Number of addresses to generate (default: 100)
    --transactions M    Number of transactions to generate (default: 1000)
    --cleanup           Remove generated data after demonstration
    --fraud-only        Skip data generation, only run fraud detection
    --evidence-only     Skip data generation, only create evidence
    --no-detection      Skip fraud detection step
    --verbose           Show detailed output
"""

import argparse
import asyncio
import json
import logging
import os
import random
import sys
import time
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import pandas as pd
from tqdm import tqdm

# Add parent directory to path to allow importing backend modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import backend modules
from backend.core.anomaly_detection import (
    AnomalyDetectionService, 
    AnomalyType,
    AnomalySeverity,
    DataEntityType,
    DetectionMethod
)
from backend.core.evidence import (
    EvidenceBundle, 
    AnomalyEvidence, 
    EvidenceSource, 
    create_evidence_bundle,
    create_pattern_match_evidence,
    create_transaction_evidence
)
from backend.core.redis_client import RedisClient, RedisDb, SerializationFormat
from backend.integrations.neo4j_client import Neo4jClient

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("demo_ingest")

# ANSI color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_header(message: str) -> None:
    """Print a header message."""
    print("\n" + "=" * 80)
    print(f"{Colors.HEADER}{Colors.BOLD}{message}{Colors.ENDC}")
    print("=" * 80)


def print_subheader(message: str) -> None:
    """Print a subheader message."""
    print(f"\n{Colors.BOLD}{message}{Colors.ENDC}")
    print("-" * 60)


def print_success(message: str) -> None:
    """Print a success message."""
    print(f"{Colors.GREEN}✓ {message}{Colors.ENDC}")


def print_warning(message: str) -> None:
    """Print a warning message."""
    print(f"{Colors.YELLOW}⚠ {message}{Colors.ENDC}")


def print_error(message: str) -> None:
    """Print an error message."""
    print(f"{Colors.RED}✗ {message}{Colors.ENDC}")


def print_info(message: str) -> None:
    """Print an info message."""
    print(f"{Colors.BLUE}ℹ {message}{Colors.ENDC}")


def print_step(message: str) -> None:
    """Print a step message."""
    print(f"{Colors.CYAN}➤ {message}{Colors.ENDC}")


def generate_eth_address() -> str:
    """Generate a random Ethereum address."""
    return "0x" + uuid.uuid4().hex[:40]


def generate_tx_hash() -> str:
    """Generate a random transaction hash."""
    return "0x" + uuid.uuid4().hex


def generate_timestamp(start_date: datetime, end_date: datetime) -> datetime:
    """Generate a random timestamp between start_date and end_date."""
    delta = end_date - start_date
    random_seconds = random.randint(0, int(delta.total_seconds()))
    return start_date + timedelta(seconds=random_seconds)


class DataGenerator:
    """Generate realistic blockchain data with fraud patterns."""
    
    def __init__(self, num_addresses: int = 100, num_transactions: int = 1000):
        self.num_addresses = num_addresses
        self.num_transactions = num_transactions
        self.addresses = []
        self.transactions = []
        self.fraud_scenarios = {
            "wash_trading": [],
            "smurfing": [],
            "layering": [],
            "round_amounts": [],
            "high_frequency": []
        }
        self.start_date = datetime.now() - timedelta(days=30)
        self.end_date = datetime.now()
    
    def generate_addresses(self) -> None:
        """Generate random addresses with balances."""
        print_step("Generating addresses...")
        self.addresses = []
        
        # Generate normal addresses
        for _ in tqdm(range(self.num_addresses - 10), desc="Normal addresses"):
            address = {
                "address": generate_eth_address(),
                "balance": round(random.uniform(0.1, 100.0), 6),
                "type": "normal",
                "created_at": generate_timestamp(self.start_date, self.end_date)
            }
            self.addresses.append(address)
        
        # Generate high-value addresses
        for _ in tqdm(range(5), desc="High-value addresses"):
            address = {
                "address": generate_eth_address(),
                "balance": round(random.uniform(1000.0, 10000.0), 6),
                "type": "high_value",
                "created_at": generate_timestamp(self.start_date, self.end_date)
            }
            self.addresses.append(address)
        
        # Generate exchange addresses
        for _ in tqdm(range(5), desc="Exchange addresses"):
            address = {
                "address": generate_eth_address(),
                "balance": round(random.uniform(10000.0, 100000.0), 6),
                "type": "exchange",
                "created_at": generate_timestamp(self.start_date, self.end_date)
            }
            self.addresses.append(address)
        
        print_success(f"Generated {len(self.addresses)} addresses")
    
    def generate_normal_transactions(self, count: int) -> None:
        """Generate normal transactions between addresses."""
        print_step(f"Generating {count} normal transactions...")
        
        for _ in tqdm(range(count), desc="Normal transactions"):
            # Select random sender and recipient
            sender = random.choice(self.addresses)
            recipient = random.choice(self.addresses)
            
            # Ensure sender and recipient are different
            while sender["address"] == recipient["address"]:
                recipient = random.choice(self.addresses)
            
            # Generate transaction
            amount = round(random.uniform(0.001, sender["balance"] * 0.8), 6)
            tx = {
                "hash": generate_tx_hash(),
                "from_address": sender["address"],
                "to_address": recipient["address"],
                "value": amount,
                "timestamp": generate_timestamp(self.start_date, self.end_date),
                "gas": round(random.uniform(21000, 100000)),
                "gas_price": round(random.uniform(1, 100)),
                "type": "normal"
            }
            self.transactions.append(tx)
    
    def generate_wash_trading(self, num_pairs: int = 3, trades_per_pair: int = 10) -> None:
        """Generate wash trading patterns (back-and-forth trading between related addresses)."""
        print_step(f"Generating wash trading scenario with {num_pairs} pairs...")
        
        for pair_idx in tqdm(range(num_pairs), desc="Wash trading pairs"):
            # Create or select two addresses for wash trading
            if random.random() < 0.5 and len(self.addresses) > 10:
                # Use existing addresses
                addr1 = random.choice(self.addresses)
                addr2 = random.choice(self.addresses)
                while addr1["address"] == addr2["address"]:
                    addr2 = random.choice(self.addresses)
            else:
                # Create new addresses specifically for wash trading
                addr1 = {
                    "address": generate_eth_address(),
                    "balance": round(random.uniform(10.0, 50.0), 6),
                    "type": "wash_trader",
                    "created_at": self.start_date + timedelta(days=random.randint(1, 15))
                }
                addr2 = {
                    "address": generate_eth_address(),
                    "balance": round(random.uniform(10.0, 50.0), 6),
                    "type": "wash_trader",
                    "created_at": self.start_date + timedelta(days=random.randint(1, 15))
                }
                self.addresses.extend([addr1, addr2])
            
            # Generate series of back-and-forth transactions
            wash_txs = []
            base_amount = round(random.uniform(1.0, 10.0), 6)
            
            # Start time for this wash trading activity
            start_time = generate_timestamp(self.start_date, self.end_date - timedelta(days=2))
            
            for i in range(trades_per_pair):
                # Alternate direction
                if i % 2 == 0:
                    from_addr = addr1["address"]
                    to_addr = addr2["address"]
                else:
                    from_addr = addr2["address"]
                    to_addr = addr1["address"]
                
                # Slightly vary the amount to make it look less suspicious
                variation = random.uniform(0.95, 1.05)
                amount = round(base_amount * variation, 6)
                
                # Create transaction with small time gap
                tx_time = start_time + timedelta(minutes=random.randint(10, 60))
                start_time = tx_time  # Update for next transaction
                
                tx = {
                    "hash": generate_tx_hash(),
                    "from_address": from_addr,
                    "to_address": to_addr,
                    "value": amount,
                    "timestamp": tx_time,
                    "gas": round(random.uniform(21000, 100000)),
                    "gas_price": round(random.uniform(1, 100)),
                    "type": "wash_trading"
                }
                wash_txs.append(tx)
            
            # Add to transactions and fraud scenarios
            self.transactions.extend(wash_txs)
            self.fraud_scenarios["wash_trading"].append({
                "addresses": [addr1["address"], addr2["address"]],
                "transactions": [tx["hash"] for tx in wash_txs],
                "description": f"Wash trading between {addr1['address']} and {addr2['address']}",
                "total_volume": sum(tx["value"] for tx in wash_txs)
            })
        
        print_success(f"Generated {num_pairs} wash trading pairs with {num_pairs * trades_per_pair} transactions")
    
    def generate_smurfing(self, num_scenarios: int = 2, txs_per_scenario: int = 20) -> None:
        """Generate smurfing patterns (breaking down large transactions into many smaller ones)."""
        print_step(f"Generating smurfing scenario with {num_scenarios} cases...")
        
        for scenario_idx in tqdm(range(num_scenarios), desc="Smurfing scenarios"):
            # Create a source address (the smurf)
            source_addr = {
                "address": generate_eth_address(),
                "balance": round(random.uniform(100.0, 1000.0), 6),
                "type": "smurf_source",
                "created_at": self.start_date + timedelta(days=random.randint(1, 10))
            }
            
            # Create a destination address (the ultimate recipient)
            dest_addr = {
                "address": generate_eth_address(),
                "balance": round(random.uniform(10.0, 50.0), 6),
                "type": "smurf_destination",
                "created_at": self.start_date + timedelta(days=random.randint(1, 10))
            }
            
            # Create intermediary addresses
            intermediaries = []
            for _ in range(random.randint(3, 6)):
                intermediary = {
                    "address": generate_eth_address(),
                    "balance": round(random.uniform(1.0, 10.0), 6),
                    "type": "smurf_intermediary",
                    "created_at": self.start_date + timedelta(days=random.randint(1, 15))
                }
                intermediaries.append(intermediary)
            
            # Add all addresses
            self.addresses.extend([source_addr, dest_addr] + intermediaries)
            
            # Generate smurfing transactions
            smurf_txs = []
            total_amount = round(random.uniform(50.0, 200.0), 6)
            individual_amount = round(total_amount / txs_per_scenario, 6)
            
            # Start time for this smurfing activity
            start_time = generate_timestamp(self.start_date, self.end_date - timedelta(days=5))
            
            # First phase: source to intermediaries
            for i, intermediary in enumerate(intermediaries):
                # Distribute funds to intermediaries
                tx_time = start_time + timedelta(minutes=random.randint(5, 30))
                
                # Calculate how many transactions to send through this intermediary
                txs_through_this_intermediary = txs_per_scenario // len(intermediaries)
                if i < txs_per_scenario % len(intermediaries):
                    txs_through_this_intermediary += 1
                
                # Send funds to intermediary
                amount = round(individual_amount * txs_through_this_intermediary, 6)
                
                tx = {
                    "hash": generate_tx_hash(),
                    "from_address": source_addr["address"],
                    "to_address": intermediary["address"],
                    "value": amount,
                    "timestamp": tx_time,
                    "gas": round(random.uniform(21000, 100000)),
                    "gas_price": round(random.uniform(1, 100)),
                    "type": "smurfing_phase1"
                }
                smurf_txs.append(tx)
            
            # Second phase: intermediaries to destination in small amounts
            txs_sent = 0
            for intermediary in intermediaries:
                # Calculate how many transactions to send through this intermediary
                txs_through_this_intermediary = txs_per_scenario // len(intermediaries)
                if txs_sent < txs_per_scenario % len(intermediaries):
                    txs_through_this_intermediary += 1
                    txs_sent += 1
                
                for _ in range(txs_through_this_intermediary):
                    tx_time = start_time + timedelta(hours=random.randint(1, 24))
                    start_time = tx_time  # Update for next transaction
                    
                    # Slightly vary the amount
                    variation = random.uniform(0.9, 1.1)
                    amount = round(individual_amount * variation, 6)
                    
                    tx = {
                        "hash": generate_tx_hash(),
                        "from_address": intermediary["address"],
                        "to_address": dest_addr["address"],
                        "value": amount,
                        "timestamp": tx_time,
                        "gas": round(random.uniform(21000, 100000)),
                        "gas_price": round(random.uniform(1, 100)),
                        "type": "smurfing_phase2"
                    }
                    smurf_txs.append(tx)
            
            # Add to transactions and fraud scenarios
            self.transactions.extend(smurf_txs)
            self.fraud_scenarios["smurfing"].append({
                "source": source_addr["address"],
                "destination": dest_addr["address"],
                "intermediaries": [addr["address"] for addr in intermediaries],
                "transactions": [tx["hash"] for tx in smurf_txs],
                "description": f"Smurfing from {source_addr['address']} to {dest_addr['address']} via {len(intermediaries)} intermediaries",
                "total_amount": total_amount
            })
        
        print_success(f"Generated {num_scenarios} smurfing scenarios with {len(self.fraud_scenarios['smurfing'][-1]['transactions'])} transactions")
    
    def generate_layering(self, num_scenarios: int = 2, layers: int = 4) -> None:
        """Generate layering patterns (funds passing through multiple intermediaries)."""
        print_step(f"Generating layering scenario with {num_scenarios} cases...")
        
        for scenario_idx in tqdm(range(num_scenarios), desc="Layering scenarios"):
            # Create addresses for the layering chain
            chain_addresses = []
            for i in range(layers + 1):  # +1 for the final destination
                addr_type = "layering_source" if i == 0 else "layering_destination" if i == layers else f"layering_layer{i}"
                addr = {
                    "address": generate_eth_address(),
                    "balance": round(random.uniform(10.0, 100.0), 6),
                    "type": addr_type,
                    "created_at": self.start_date + timedelta(days=random.randint(1, 20))
                }
                chain_addresses.append(addr)
            
            # Add addresses to the main list
            self.addresses.extend(chain_addresses)
            
            # Generate layering transactions
            layer_txs = []
            base_amount = round(random.uniform(50.0, 200.0), 6)
            
            # Start time for this layering activity
            start_time = generate_timestamp(self.start_date, self.end_date - timedelta(days=10))
            
            # Create transactions through each layer
            for i in range(layers):
                # Slightly decrease amount at each layer (to simulate fees)
                fee_percentage = random.uniform(0.001, 0.01)  # 0.1% to 1% fee
                amount = base_amount * (1 - fee_percentage) if i > 0 else base_amount
                base_amount = amount  # Update for next layer
                
                # Add some delay between layers
                tx_time = start_time + timedelta(hours=random.randint(1, 12))
                start_time = tx_time  # Update for next transaction
                
                tx = {
                    "hash": generate_tx_hash(),
                    "from_address": chain_addresses[i]["address"],
                    "to_address": chain_addresses[i+1]["address"],
                    "value": round(amount, 6),
                    "timestamp": tx_time,
                    "gas": round(random.uniform(21000, 100000)),
                    "gas_price": round(random.uniform(1, 100)),
                    "type": f"layering_layer{i+1}"
                }
                layer_txs.append(tx)
            
            # Add to transactions and fraud scenarios
            self.transactions.extend(layer_txs)
            self.fraud_scenarios["layering"].append({
                "path": [addr["address"] for addr in chain_addresses],
                "transactions": [tx["hash"] for tx in layer_txs],
                "description": f"Layering from {chain_addresses[0]['address']} through {layers} layers to {chain_addresses[-1]['address']}",
                "initial_amount": layer_txs[0]["value"],
                "final_amount": layer_txs[-1]["value"]
            })
        
        print_success(f"Generated {num_scenarios} layering scenarios with {layers} layers each")
    
    def generate_round_amount_transactions(self, count: int = 10) -> None:
        """Generate suspiciously round amount transactions."""
        print_step(f"Generating {count} round amount transactions...")
        
        round_txs = []
        for _ in tqdm(range(count), desc="Round amount transactions"):
            # Select random sender and recipient
            sender = random.choice(self.addresses)
            recipient = random.choice(self.addresses)
            
            # Ensure sender and recipient are different
            while sender["address"] == recipient["address"]:
                recipient = random.choice(self.addresses)
            
            # Generate round amount
            magnitude = random.choice([10, 100, 1000, 10000])
            amount = magnitude * random.randint(1, 10)
            
            # Generate transaction
            tx = {
                "hash": generate_tx_hash(),
                "from_address": sender["address"],
                "to_address": recipient["address"],
                "value": float(amount),
                "timestamp": generate_timestamp(self.start_date, self.end_date),
                "gas": round(random.uniform(21000, 100000)),
                "gas_price": round(random.uniform(1, 100)),
                "type": "round_amount"
            }
            round_txs.append(tx)
        
        # Add to transactions and fraud scenarios
        self.transactions.extend(round_txs)
        self.fraud_scenarios["round_amounts"] = [
            {
                "transaction": tx["hash"],
                "from_address": tx["from_address"],
                "to_address": tx["to_address"],
                "amount": tx["value"],
                "description": f"Round amount transaction of {tx['value']} ETH"
            }
            for tx in round_txs
        ]
        
        print_success(f"Generated {count} round amount transactions")
    
    def generate_high_frequency_trading(self, num_addresses: int = 2, num_txs: int = 50) -> None:
        """Generate high frequency trading patterns."""
        print_step(f"Generating high frequency trading scenario with {num_addresses} addresses...")
        
        # Create or select addresses for high frequency trading
        hft_addresses = []
        for _ in range(num_addresses):
            if random.random() < 0.3 and len(self.addresses) > 20:
                # Use existing address
                addr = random.choice(self.addresses)
                while any(a["address"] == addr["address"] for a in hft_addresses):
                    addr = random.choice(self.addresses)
                hft_addresses.append(addr)
            else:
                # Create new address
                addr = {
                    "address": generate_eth_address(),
                    "balance": round(random.uniform(100.0, 500.0), 6),
                    "type": "high_frequency_trader",
                    "created_at": self.start_date + timedelta(days=random.randint(1, 15))
                }
                self.addresses.append(addr)
                hft_addresses.append(addr)
        
        # Generate high frequency transactions
        hft_txs = []
        
        # Start time for this high frequency activity
        start_time = generate_timestamp(self.start_date, self.end_date - timedelta(hours=6))
        
        for _ in tqdm(range(num_txs), desc="High frequency transactions"):
            # Select random sender and recipient from HFT addresses
            sender = random.choice(hft_addresses)
            recipient = random.choice(hft_addresses)
            
            # Ensure sender and recipient are different
            while sender["address"] == recipient["address"]:
                recipient = random.choice(hft_addresses)
            
            # Generate amount
            amount = round(random.uniform(0.1, 5.0), 6)
            
            # Create transaction with very small time gap
            tx_time = start_time + timedelta(seconds=random.randint(10, 300))
            start_time = tx_time  # Update for next transaction
            
            tx = {
                "hash": generate_tx_hash(),
                "from_address": sender["address"],
                "to_address": recipient["address"],
                "value": amount,
                "timestamp": tx_time,
                "gas": round(random.uniform(21000, 100000)),
                "gas_price": round(random.uniform(1, 100)),
                "type": "high_frequency"
            }
            hft_txs.append(tx)
        
        # Add to transactions and fraud scenarios
        self.transactions.extend(hft_txs)
        self.fraud_scenarios["high_frequency"].append({
            "addresses": [addr["address"] for addr in hft_addresses],
            "transactions": [tx["hash"] for tx in hft_txs],
            "description": f"High frequency trading among {num_addresses} addresses with {num_txs} transactions",
            "time_span_seconds": (hft_txs[-1]["timestamp"] - hft_txs[0]["timestamp"]).total_seconds(),
            "average_interval_seconds": (hft_txs[-1]["timestamp"] - hft_txs[0]["timestamp"]).total_seconds() / num_txs
        })
        
        print_success(f"Generated high frequency trading scenario with {num_txs} transactions")
    
    def generate_all_data(self) -> Dict[str, Any]:
        """Generate all data including normal and fraudulent patterns."""
        print_header("Generating Blockchain Data")
        
        # Generate addresses
        self.generate_addresses()
        
        # Calculate how many transactions to allocate to each type
        remaining_txs = self.num_transactions
        
        # Allocate transactions for fraud scenarios
        wash_trading_pairs = 3
        wash_trading_txs_per_pair = 10
        wash_trading_txs = wash_trading_pairs * wash_trading_txs_per_pair
        
        smurfing_scenarios = 2
        smurfing_txs_per_scenario = 20
        smurfing_txs = smurfing_scenarios * smurfing_txs_per_scenario
        
        layering_scenarios = 2
        layering_layers = 4
        layering_txs = layering_scenarios * layering_layers
        
        round_amount_txs = 10
        
        high_frequency_addresses = 2
        high_frequency_txs = 50
        
        # Calculate normal transactions
        fraud_txs = wash_trading_txs + smurfing_txs + layering_txs + round_amount_txs + high_frequency_txs
        normal_txs = max(0, remaining_txs - fraud_txs)
        
        # Generate transactions
        self.generate_normal_transactions(normal_txs)
        self.generate_wash_trading(wash_trading_pairs, wash_trading_txs_per_pair)
        self.generate_smurfing(smurfing_scenarios, smurfing_txs_per_scenario)
        self.generate_layering(layering_scenarios, layering_layers)
        self.generate_round_amount_transactions(round_amount_txs)
        self.generate_high_frequency_trading(high_frequency_addresses, high_frequency_txs)
        
        # Return all generated data
        return {
            "addresses": self.addresses,
            "transactions": self.transactions,
            "fraud_scenarios": self.fraud_scenarios
        }


class Neo4jIngestor:
    """Ingest generated data into Neo4j."""
    
    def __init__(self, neo4j_client: Optional[Neo4jClient] = None):
        self.neo4j_client = neo4j_client or Neo4jClient()
    
    async def connect(self) -> None:
        """Connect to Neo4j database."""
        if not hasattr(self.neo4j_client, 'driver') or self.neo4j_client.driver is None:
            await self.neo4j_client.connect()
    
    async def create_constraints_and_indexes(self) -> None:
        """Create necessary constraints and indexes in Neo4j."""
        print_step("Creating constraints and indexes...")
        
        constraints = [
            "CREATE CONSTRAINT address_id IF NOT EXISTS FOR (a:Address) REQUIRE a.address IS UNIQUE",
            "CREATE CONSTRAINT transaction_id IF NOT EXISTS FOR (t:Transaction) REQUIRE t.hash IS UNIQUE"
        ]
        
        indexes = [
            "CREATE INDEX address_type_idx IF NOT EXISTS FOR (a:Address) ON (a.type)",
            "CREATE INDEX tx_timestamp_idx IF NOT EXISTS FOR (t:Transaction) ON (t.timestamp)",
            "CREATE INDEX tx_value_idx IF NOT EXISTS FOR (t:Transaction) ON (t.value)",
            "CREATE INDEX tx_type_idx IF NOT EXISTS FOR (t:Transaction) ON (t.type)"
        ]
        
        # Create constraints
        for constraint in constraints:
            self.neo4j_client.execute_query(constraint)
        
        # Create indexes
        for index in indexes:
            self.neo4j_client.execute_query(index)
        
        print_success("Created constraints and indexes")
    
    async def ingest_addresses(self, addresses: List[Dict[str, Any]]) -> None:
        """Ingest addresses into Neo4j."""
        print_step(f"Ingesting {len(addresses)} addresses...")
        
        # Process in batches to avoid memory issues
        batch_size = 100
        for i in tqdm(range(0, len(addresses), batch_size), desc="Address batches"):
            batch = addresses[i:i+batch_size]
            
            # Convert timestamps to strings for Neo4j
            for addr in batch:
                if isinstance(addr["created_at"], datetime):
                    addr["created_at"] = addr["created_at"].isoformat()
            
            # Create Cypher query
            query = """
            UNWIND $addresses AS addr
            MERGE (a:Address {address: addr.address})
            SET a.balance = addr.balance,
                a.type = addr.type,
                a.created_at = datetime(addr.created_at),
                a.updated_at = datetime()
            """
            
            # Execute query
            self.neo4j_client.execute_query(query, {"addresses": batch})
        
        print_success(f"Ingested {len(addresses)} addresses")
    
    async def ingest_transactions(self, transactions: List[Dict[str, Any]]) -> None:
        """Ingest transactions into Neo4j."""
        print_step(f"Ingesting {len(transactions)} transactions...")
        
        # Process in batches to avoid memory issues
        batch_size = 100
        for i in tqdm(range(0, len(transactions), batch_size), desc="Transaction batches"):
            batch = transactions[i:i+batch_size]
            
            # Convert timestamps to strings for Neo4j
            for tx in batch:
                if isinstance(tx["timestamp"], datetime):
                    tx["timestamp"] = tx["timestamp"].isoformat()
            
            # Create Cypher query
            query = """
            UNWIND $transactions AS tx
            MATCH (from:Address {address: tx.from_address})
            MATCH (to:Address {address: tx.to_address})
            CREATE (from)-[t:TRANSFERRED {
                hash: tx.hash,
                value: tx.value,
                timestamp: datetime(tx.timestamp),
                gas: tx.gas,
                gas_price: tx.gas_price,
                type: tx.type
            }]->(to)
            """
            
            # Execute query
            self.neo4j_client.execute_query(query, {"transactions": batch})
        
        print_success(f"Ingested {len(transactions)} transactions")
    
    async def create_fraud_labels(self, fraud_scenarios: Dict[str, List[Dict[str, Any]]]) -> None:
        """Create fraud labels in Neo4j based on generated scenarios."""
        print_step("Creating fraud labels...")
        
        # Wash trading
        if fraud_scenarios.get("wash_trading"):
            for scenario in tqdm(fraud_scenarios["wash_trading"], desc="Wash trading labels"):
                # Label addresses
                addr_query = """
                UNWIND $addresses AS addr
                MATCH (a:Address {address: addr})
                SET a:WashTrader
                SET a.fraud_risk = 'high'
                """
                self.neo4j_client.execute_query(addr_query, {"addresses": scenario["addresses"]})
                
                # Label transactions
                tx_query = """
                UNWIND $hashes AS hash
                MATCH ()-[t:TRANSFERRED {hash: hash}]->()
                SET t.fraud_type = 'wash_trading'
                SET t.fraud_scenario_id = $scenario_id
                """
                self.neo4j_client.execute_query(tx_query, {
                    "hashes": scenario["transactions"],
                    "scenario_id": str(uuid.uuid4())
                })
        
        # Smurfing
        if fraud_scenarios.get("smurfing"):
            for scenario in tqdm(fraud_scenarios["smurfing"], desc="Smurfing labels"):
                # Label source address
                source_query = """
                MATCH (a:Address {address: $address})
                SET a:Smurf
                SET a.fraud_risk = 'high'
                """
                self.neo4j_client.execute_query(source_query, {"address": scenario["source"]})
                
                # Label intermediaries
                if scenario.get("intermediaries"):
                    interm_query = """
                    UNWIND $addresses AS addr
                    MATCH (a:Address {address: addr})
                    SET a:SmurfIntermediary
                    SET a.fraud_risk = 'medium'
                    """
                    self.neo4j_client.execute_query(interm_query, {"addresses": scenario["intermediaries"]})
                
                # Label transactions
                tx_query = """
                UNWIND $hashes AS hash
                MATCH ()-[t:TRANSFERRED {hash: hash}]->()
                SET t.fraud_type = 'smurfing'
                SET t.fraud_scenario_id = $scenario_id
                """
                self.neo4j_client.execute_query(tx_query, {
                    "hashes": scenario["transactions"],
                    "scenario_id": str(uuid.uuid4())
                })
        
        # Layering
        if fraud_scenarios.get("layering"):
            for scenario in tqdm(fraud_scenarios["layering"], desc="Layering labels"):
                # Label addresses in the path
                path_query = """
                UNWIND $addresses AS addr
                MATCH (a:Address {address: addr})
                SET a:LayeringParticipant
                SET a.fraud_risk = CASE 
                    WHEN a.address = $source THEN 'high'
                    WHEN a.address = $destination THEN 'medium'
                    ELSE 'low'
                END
                """
                self.neo4j_client.execute_query(path_query, {
                    "addresses": scenario["path"],
                    "source": scenario["path"][0],
                    "destination": scenario["path"][-1]
                })
                
                # Label transactions
                tx_query = """
                UNWIND $hashes AS hash
                MATCH ()-[t:TRANSFERRED {hash: hash}]->()
                SET t.fraud_type = 'layering'
                SET t.fraud_scenario_id = $scenario_id
                """
                self.neo4j_client.execute_query(tx_query, {
                    "hashes": scenario["transactions"],
                    "scenario_id": str(uuid.uuid4())
                })
        
        # Round amounts
        if fraud_scenarios.get("round_amounts"):
            for tx in tqdm(fraud_scenarios["round_amounts"], desc="Round amount labels"):
                tx_query = """
                MATCH ()-[t:TRANSFERRED {hash: $hash}]->()
                SET t.fraud_type = 'round_amount'
                SET t.suspicious = true
                """
                self.neo4j_client.execute_query(tx_query, {"hash": tx["transaction"]})
        
        # High frequency
        if fraud_scenarios.get("high_frequency"):
            for scenario in tqdm(fraud_scenarios["high_frequency"], desc="High frequency labels"):
                # Label addresses
                addr_query = """
                UNWIND $addresses AS addr
                MATCH (a:Address {address: addr})
                SET a:HighFrequencyTrader
                SET a.fraud_risk = 'medium'
                """
                self.neo4j_client.execute_query(addr_query, {"addresses": scenario["addresses"]})
                
                # Label transactions
                tx_query = """
                UNWIND $hashes AS hash
                MATCH ()-[t:TRANSFERRED {hash: hash}]->()
                SET t.fraud_type = 'high_frequency'
                SET t.fraud_scenario_id = $scenario_id
                """
                self.neo4j_client.execute_query(tx_query, {
                    "hashes": scenario["transactions"],
                    "scenario_id": str(uuid.uuid4())
                })
        
        print_success("Created fraud labels")
    
    async def cleanup_data(self) -> None:
        """Remove all data created by this script."""
        print_step("Cleaning up data...")
        
        # Remove relationships first
        self.neo4j_client.execute_query("""
        MATCH ()-[t:TRANSFERRED]->()
        WHERE t.type IN ['normal', 'wash_trading', 'smurfing_phase1', 'smurfing_phase2', 
                        'layering_layer1', 'layering_layer2', 'layering_layer3', 'layering_layer4',
                        'round_amount', 'high_frequency']
        DELETE t
        """)
        
        # Remove addresses
        self.neo4j_client.execute_query("""
        MATCH (a:Address)
        WHERE a.type IN ['normal', 'high_value', 'exchange', 'wash_trader', 'smurf_source', 
                        'smurf_destination', 'smurf_intermediary', 'layering_source', 
                        'layering_destination', 'layering_layer1', 'layering_layer2', 
                        'layering_layer3', 'high_frequency_trader']
        DELETE a
        """)
        
        print_success("Cleaned up all demo data")
    
    async def ingest_all_data(self, data: Dict[str, Any], cleanup_first: bool = False) -> None:
        """Ingest all data into Neo4j."""
        print_header("Ingesting Data into Neo4j")
        
        # Connect to Neo4j
        await self.connect()
        
        # Clean up existing data if requested
        if cleanup_first:
            await self.cleanup_data()
        
        # Create constraints and indexes
        await self.create_constraints_and_indexes()
        
        # Ingest addresses
        await self.ingest_addresses(data["addresses"])
        
        # Ingest transactions
        await self.ingest_transactions(data["transactions"])
        
        # Create fraud labels
        await self.create_fraud_labels(data["fraud_scenarios"])
        
        print_success("All data ingested successfully")


class RedisSetup:
    """Set up Redis cache with sample data."""
    
    def __init__(self, redis_client: Optional[RedisClient] = None):
        self.redis_client = redis_client or RedisClient()
    
    def setup_cache_data(self, data: Dict[str, Any]) -> None:
        """Set up Redis cache with sample data."""
        print_header("Setting Up Redis Cache")
        
        # Store fraud scenarios
        print_step("Storing fraud scenarios in Redis...")
        self.redis_client.set(
            key="demo:fraud_scenarios",
            value=data["fraud_scenarios"],
            ttl_seconds=86400 * 7,  # 7 days
            db=RedisDb.CACHE,
            format=SerializationFormat.JSON
        )
        
        # Store sample addresses
        print_step("Storing sample addresses in Redis...")
        sample_addresses = random.sample(data["addresses"], min(20, len(data["addresses"])))
        self.redis_client.set(
            key="demo:sample_addresses",
            value=sample_addresses,
            ttl_seconds=86400 * 7,  # 7 days
            db=RedisDb.CACHE,
            format=SerializationFormat.JSON
        )
        
        # Store sample transactions
        print_step("Storing sample transactions in Redis...")
        sample_transactions = random.sample(data["transactions"], min(50, len(data["transactions"])))
        self.redis_client.set(
            key="demo:sample_transactions",
            value=sample_transactions,
            ttl_seconds=86400 * 7,  # 7 days
            db=RedisDb.CACHE,
            format=SerializationFormat.JSON
        )
        
        # Store detection patterns
        print_step("Storing detection patterns in Redis...")
        detection_patterns = {
            "wash_trading": {
                "name": "Wash Trading Detection",
                "description": "Detects artificial trading activity between related accounts",
                "cypher_query": """
                MATCH (a:Address)-[t1:TRANSFERRED]->(b:Address)
                MATCH (b)-[t2:TRANSFERRED]->(a)
                WHERE t1.timestamp <= t2.timestamp + duration('P1D')
                  AND t1.timestamp >= t2.timestamp - duration('P1D')
                WITH a, b, count(t1) AS cycle_count
                WHERE cycle_count >= 5
                RETURN a.address AS address_a, b.address AS address_b, 
                       cycle_count, 'wash_trading' AS pattern_type,
                       CASE 
                         WHEN cycle_count > 20 THEN 'high'
                         WHEN cycle_count > 10 THEN 'medium'
                         ELSE 'low'
                       END AS risk_level
                """
            },
            "smurfing": {
                "name": "Smurfing Detection",
                "description": "Detects breaking down of large transactions into many smaller ones",
                "cypher_query": """
                MATCH (a:Address)-[t:TRANSFERRED]->(b:Address)
                WITH a, b, collect(t) AS transactions
                WHERE size(transactions) > 10
                  AND max(transactions[i IN range(0, size(transactions)-1) | i].timestamp) - 
                      min(transactions[i IN range(0, size(transactions)-1) | i].timestamp) < duration('P1D')
                RETURN a.address AS source_address, b.address AS destination_address,
                       size(transactions) AS transaction_count, 
                       sum(t.value) AS total_value,
                       'smurfing' AS pattern_type,
                       CASE 
                         WHEN size(transactions) > 30 THEN 'high'
                         WHEN size(transactions) > 20 THEN 'medium'
                         ELSE 'low'
                       END AS risk_level
                """
            },
            "layering": {
                "name": "Layering Detection",
                "description": "Detects funds passing through multiple intermediaries",
                "cypher_query": """
                MATCH path = (source:Address)-[t1:TRANSFERRED]->(a:Address)-[t2:TRANSFERRED]->(dest:Address)
                WHERE t1.timestamp <= t2.timestamp + duration('PT1H')
                  AND t1.timestamp >= t2.timestamp - duration('PT1H')
                  AND abs(t1.value - t2.value) / t1.value < 0.1
                RETURN source.address AS source_address, dest.address AS destination_address,
                       t1.hash AS first_tx, t2.hash AS second_tx,
                       t1.value AS first_value, t2.value AS second_value,
                       'layering' AS pattern_type,
                       CASE
                         WHEN abs(t1.value - t2.value) / t1.value < 0.01 THEN 'high'
                         WHEN abs(t1.value - t2.value) / t1.value < 0.05 THEN 'medium'
                         ELSE 'low'
                       END AS risk_level
                """
            },
            "round_amounts": {
                "name": "Round Amount Detection",
                "description": "Detects suspiciously round transaction amounts",
                "cypher_query": """
                MATCH (a:Address)-[t:TRANSFERRED]->(b:Address)
                WHERE t.value % 1000 = 0 OR t.value % 10000 = 0 OR t.value % 100000 = 0
                  AND t.value >= 10000
                RETURN a.address AS sender, b.address AS recipient,
                       t.hash AS tx_hash, t.value AS amount,
                       'round_amount' AS pattern_type,
                       CASE 
                         WHEN t.value >= 1000000 THEN 'high'
                         WHEN t.value >= 100000 THEN 'medium'
                         ELSE 'low'
                       END AS risk_level
                """
            },
            "high_frequency": {
                "name": "High Frequency Trading Detection",
                "description": "Detects unusually high frequency of transactions",
                "cypher_query": """
                MATCH (a:Address)-[t:TRANSFERRED]->(b:Address)
                WITH a, count(t) AS tx_count
                WHERE tx_count > 20
                RETURN a.address AS address, tx_count,
                       'high_frequency' AS pattern_type,
                       CASE 
                         WHEN tx_count > 100 THEN 'high'
                         WHEN tx_count > 50 THEN 'medium'
                         ELSE 'low'
                       END AS risk_level
                """
            }
        }
        
        self.redis_client.set(
            key="demo:detection_patterns",
            value=detection_patterns,
            ttl_seconds=86400 * 7,  # 7 days
            db=RedisDb.CACHE,
            format=SerializationFormat.JSON
        )
        
        # Store demo metadata
        print_step("Storing demo metadata in Redis...")
        demo_metadata = {
            "generation_time": datetime.now().isoformat(),
            "address_count": len(data["addresses"]),
            "transaction_count": len(data["transactions"]),
            "fraud_scenarios": {
                "wash_trading": len(data["fraud_scenarios"].get("wash_trading", [])),
                "smurfing": len(data["fraud_scenarios"].get("smurfing", [])),
                "layering": len(data["fraud_scenarios"].get("layering", [])),
                "round_amounts": len(data["fraud_scenarios"].get("round_amounts", [])),
                "high_frequency": len(data["fraud_scenarios"].get("high_frequency", []))
            }
        }
        
        self.redis_client.set(
            key="demo:metadata",
            value=demo_metadata,
            ttl_seconds=86400 * 7,  # 7 days
            db=RedisDb.CACHE,
            format=SerializationFormat.JSON
        )
        
        print_success("Redis cache setup complete")
    
    def cleanup_cache(self) -> None:
        """Clean up Redis cache data."""
        print_step("Cleaning up Redis cache...")
        
        keys = [
            "demo:fraud_scenarios",
            "demo:sample_addresses",
            "demo:sample_transactions",
            "demo:detection_patterns",
            "demo:metadata"
        ]
        
        for key in keys:
            self.redis_client.delete(key, RedisDb.CACHE)
        
        print_success("Redis cache cleaned up")


class FraudDetector:
    """Demonstrate the platform's fraud detection capabilities."""
    
    def __init__(
        self,
        neo4j_client: Optional[Neo4jClient] = None,
        redis_client: Optional[RedisClient] = None
    ):
        self.neo4j_client = neo4j_client or Neo4jClient()
        self.redis_client = redis_client or RedisClient()
        self.anomaly_service = AnomalyDetectionService(
            neo4j_client=self.neo4j_client,
            redis_client=self.redis_client
        )
    
    async def detect_wash_trading(self) -> List[Dict[str, Any]]:
        """Detect wash trading patterns."""
        print_step("Detecting wash trading patterns...")
        
        query = """
        MATCH (a:Address)-[t1:TRANSFERRED]->(b:Address)
        MATCH (b)-[t2:TRANSFERRED]->(a)
        WHERE t1.timestamp <= t2.timestamp + duration('P1D')
          AND t1.timestamp >= t2.timestamp - duration('P1D')
        WITH a, b, count(t1) AS cycle_count, sum(t1.value) AS total_value
        WHERE cycle_count >= 5
        RETURN a.address AS address_a, b.address AS address_b, 
               cycle_count, total_value,
               CASE 
                 WHEN cycle_count > 20 THEN 'high'
                 WHEN cycle_count > 10 THEN 'medium'
                 ELSE 'low'
               END AS risk_level
        ORDER BY cycle_count DESC
        LIMIT 10
        """
        
        results = self.neo4j_client.execute_query(query)
        
        if results:
            print_success(f"Found {len(results)} wash trading patterns")
            for i, result in enumerate(results):
                print(f"  {i+1}. Between {result['address_a']} and {result['address_b']}")
                print(f"     {result['cycle_count']} cycles, {result['total_value']} ETH total volume")
                print(f"     Risk level: {result['risk_level']}")
        else:
            print_warning("No wash trading patterns detected")
        
        return results
    
    async def detect_smurfing(self) -> List[Dict[str, Any]]:
        """Detect smurfing patterns."""
        print_step("Detecting smurfing patterns...")
        
        query = """
        MATCH (a:Address)-[t:TRANSFERRED]->(b:Address)
        WITH a, b, collect(t) AS transactions
        WHERE size(transactions) > 10
          AND max(transactions[i IN range(0, size(transactions)-1) | i].timestamp) - 
              min(transactions[i IN range(0, size(transactions)-1) | i].timestamp) < duration('P1D')
        RETURN a.address AS source_address, b.address AS destination_address,
               size(transactions) AS transaction_count, 
               sum(t.value) AS total_value,
               avg(t.value) AS avg_value,
               CASE 
                 WHEN size(transactions) > 30 THEN 'high'
                 WHEN size(transactions) > 20 THEN 'medium'
                 ELSE 'low'
               END AS risk_level
        ORDER BY transaction_count DESC
        LIMIT 10
        """
        
        results = self.neo4j_client.execute_query(query)
        
        if results:
            print_success(f"Found {len(results)} smurfing patterns")
            for i, result in enumerate(results):
                print(f"  {i+1}. From {result['source_address']} to {result['destination_address']}")
                print(f"     {result['transaction_count']} transactions, {result['total_value']} ETH total")
                print(f"     Average transaction: {result['avg_value']} ETH")
                print(f"     Risk level: {result['risk_level']}")
        else:
            print_warning("No smurfing patterns detected")
        
        return results
    
    async def detect_layering(self) -> List[Dict[str, Any]]:
        """Detect layering patterns."""
        print_step("Detecting layering patterns...")
        
        query = """
        MATCH path = (source:Address)-[t1:TRANSFERRED]->(a:Address)-[t2:TRANSFERRED]->(dest:Address)
        WHERE t1.timestamp <= t2.timestamp + duration('PT1H')
          AND t1.timestamp >= t2.timestamp - duration('PT1H')
          AND abs(t1.value - t2.value) / t1.value < 0.1
        WITH source, a, dest, t1, t2, 
             abs(t1.value - t2.value) / t1.value AS value_difference_ratio,
             abs(t1.timestamp - t2.timestamp) AS time_difference
        RETURN source.address AS source_address, a.address AS intermediary_address,
               dest.address AS destination_address,
               t1.hash AS first_tx, t2.hash AS second_tx,
               t1.value AS first_value, t2.value AS second_value,
               value_difference_ratio,
               CASE
                 WHEN value_difference_ratio < 0.01 THEN 'high'
                 WHEN value_difference_ratio < 0.05 THEN 'medium'
                 ELSE 'low'
               END AS risk_level
        ORDER BY value_difference_ratio
        LIMIT 10
        """
        
        results = self.neo4j_client.execute_query(query)
        
        if results:
            print_success(f"Found {len(results)} layering patterns")
            for i, result in enumerate(results):
                print(f"  {i+1}. From {result['source_address']} through {result['intermediary_address']} to {result['destination_address']}")
                print(f"     First tx: {result['first_value']} ETH, Second tx: {result['second_value']} ETH")
                print(f"     Value difference: {result['value_difference_ratio']*100:.2f}%")
                print(f"     Risk level: {result['risk_level']}")
        else:
            print_warning("No layering patterns detected")
        
        return results
    
    async def detect_round_amounts(self) -> List[Dict[str, Any]]:
        """Detect suspiciously round transaction amounts."""
        print_step("Detecting round amount transactions...")
        
        query = """
        MATCH (a:Address)-[t:TRANSFERRED]->(b:Address)
        WHERE t.value % 1000 = 0 OR t.value % 10000 = 0 OR t.value % 100000 = 0
          AND t.value >= 10000
        RETURN a.address AS sender, b.address AS recipient,
               t.hash AS tx_hash, t.value AS amount,
               CASE 
                 WHEN t.value >= 1000000 THEN 'high'
                 WHEN t.value >= 100000 THEN 'medium'
                 ELSE 'low'
               END AS risk_level
        ORDER BY t.value DESC
        LIMIT 10
        """
        
        results = self.neo4j_client.execute_query(query)
        
        if results:
            print_success(f"Found {len(results)} round amount transactions")
            for i, result in enumerate(results):
                print(f"  {i+1}. {result['amount']} ETH from {result['sender']} to {result['recipient']}")
                print(f"     Transaction hash: {result['tx_hash']}")
                print(f"     Risk level: {result['risk_level']}")
        else:
            print_warning("No round amount transactions detected")
        
        return results
    
    async def detect_high_frequency(self) -> List[Dict[str, Any]]:
        """Detect high frequency trading patterns."""
        print_step("Detecting high frequency trading...")
        
        query = """
        MATCH (a:Address)-[t:TRANSFERRED]->(b:Address)
        WITH a, count(t) AS tx_count
        WHERE tx_count > 20
        RETURN a.address AS address, tx_count,
               CASE 
                 WHEN tx_count > 100 THEN 'high'
                 WHEN tx_count > 50 THEN 'medium'
                 ELSE 'low'
               END AS risk_level
        ORDER BY tx_count DESC
        LIMIT 10
        """
        
        results = self.neo4j_client.execute_query(query)
        
        if results:
            print_success(f"Found {len(results)} high frequency traders")
            for i, result in enumerate(results):
                print(f"  {i+1}. Address {result['address']} with {result['tx_count']} transactions")
                print(f"     Risk level: {result['risk_level']}")
        else:
            print_warning("No high frequency trading patterns detected")
        
        return results
    
    async def run_anomaly_detection(self, address: Optional[str] = None) -> Dict[str, Any]:
        """Run anomaly detection on an address."""
        if not address:
            # Get a random address
            addresses_query = """
            MATCH (a:Address)
            WHERE a.type IN ['wash_trader', 'smurf_source', 'layering_source', 'high_frequency_trader']
            RETURN a.address AS address
            LIMIT 10
            """
            addresses = self.neo4j_client.execute_query(addresses_query)
            if not addresses:
                print_error("No suitable addresses found for anomaly detection")
                return {}
            
            address = random.choice(addresses)["address"]
        
        print_step(f"Running anomaly detection on address: {address}")
        
        # Run anomaly detection
        results = await self.anomaly_service.detect_anomalies(
            entity_id=address,
            entity_type=DataEntityType.ADDRESS,
            create_evidence=True
        )
        
        if results:
            print_success(f"Found {len(results)} anomalies for address {address}")
            for i, result in enumerate(results):
                print(f"  {i+1}. {result.anomaly_type.value} ({result.severity.value})")
                print(f"     Score: {result.score:.4f}, Threshold: {result.threshold:.4f}")
                print(f"     Confidence: {result.confidence:.2f}")
                print(f"     Detection method: {result.detection_method.value}")
                if result.evidence_id:
                    print(f"     Evidence ID: {result.evidence_id}")
        else:
            print_warning(f"No anomalies detected for address {address}")
        
        return {"address": address, "anomalies": results}
    
    async def create_investigation(self, anomaly_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create a sample investigation from anomaly results."""
        if not anomaly_results or not anomaly_results.get("anomalies"):
            print_warning("No anomalies to create investigation from")
            return {}
        
        address = anomaly_results["address"]
        anomalies = anomaly_results["anomalies"]
        
        print_step(f"Creating investigation for address: {address}")
        
        # Create evidence bundle
        bundle = create_evidence_bundle(
            narrative=f"Investigation of suspicious activity for address {address}",
            investigation_id=f"INV-{uuid.uuid4().hex[:8]}",
            metadata={
                "address": address,
                "anomaly_count": len(anomalies),
                "created_at": datetime.now().isoformat()
            }
        )
        
        # Add evidence for each anomaly
        for anomaly in anomalies:
            # Create anomaly evidence
            evidence = AnomalyEvidence(
                anomaly_type=anomaly.anomaly_type.value,
                severity=anomaly.severity.value,
                affected_entities=[address],
                description=f"{anomaly.severity.value.capitalize()} {anomaly.anomaly_type.value} detected for address {address}",
                source=EvidenceSource.SYSTEM,
                confidence=anomaly.confidence,
                raw_data=anomaly.to_dict()
            )
            
            # Add to bundle
            bundle.add_evidence(evidence)
        
        # Add related transactions
        tx_query = """
        MATCH (a:Address {address: $address})-[t:TRANSFERRED]->(b:Address)
        RETURN t.hash AS hash, a.address AS from_address, b.address AS to_address,
               t.value AS value, t.timestamp AS timestamp, t.type AS type
        ORDER BY t.timestamp DESC
        LIMIT 10
        """
        
        transactions = self.neo4j_client.execute_query(tx_query, {"address": address})
        
        for tx in transactions:
            # Create transaction evidence
            tx_evidence = create_transaction_evidence(
                tx_hash=tx["hash"],
                chain="ethereum",
                from_address=tx["from_address"],
                to_address=tx["to_address"],
                amount=tx["value"],
                asset="ETH",
                description=f"Transaction {tx['hash']} from {tx['from_address']} to {tx['to_address']} of {tx['value']} ETH",
                source=EvidenceSource.GRAPH_ANALYSIS,
                confidence=0.9,
                raw_data=tx
            )
            
            # Add to bundle
            bundle.add_evidence(tx_evidence)
        
        # Synthesize narrative
        narrative = bundle.synthesize_narrative()
        
        print_success(f"Created investigation with ID: {bundle.investigation_id}")
        print_info(f"Evidence items: {len(bundle.evidence_items)}")
        print_info(f"Investigation narrative: {narrative[:100]}...")
        
        return {
            "investigation_id": bundle.investigation_id,
            "address": address,
            "evidence_count": len(bundle.evidence_items),
            "narrative": narrative
        }
    
    async def run_all_detections(self) -> Dict[str, Any]:
        """Run all fraud detection methods."""
        print_header("Demonstrating Fraud Detection Capabilities")
        
        # Connect to Neo4j
        if not hasattr(self.neo4j_client, 'driver') or self.neo4j_client.driver is None:
            await self.neo4j_client.connect()
        
        # Run all detection methods
        wash_trading = await self.detect_wash_trading()
        smurfing = await self.detect_smurfing()
        layering = await self.detect_layering()
        round_amounts = await self.detect_round_amounts()
        high_frequency = await self.detect_high_frequency()
        
        # Run anomaly detection on a random address
        anomaly_results = await self.run_anomaly_detection()
        
        # Create investigation
        investigation = await self.create_investigation(anomaly_results)
        
        # Return all results
        return {
            "wash_trading": wash_trading,
            "smurfing": smurfing,
            "layering": layering,
            "round_amounts": round_amounts,
            "high_frequency": high_frequency,
            "anomaly_detection": anomaly_results,
            "investigation": investigation
        }


async def main() -> None:
    """Main entry point for the demo script."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Demo data ingestion and fraud detection script")
    parser.add_argument("--addresses", type=int, default=100, help="Number of addresses to generate")
    parser.add_argument("--transactions", type=int, default=1000, help="Number of transactions to generate")
    parser.add_argument("--cleanup", action="store_true", help="Remove generated data after demonstration")
    parser.add_argument("--fraud-only", action="store_true", help="Skip data generation, only run fraud detection")
    parser.add_argument("--evidence-only", action="store_true", help="Skip data generation, only create evidence")
    parser.add_argument("--no-detection", action="store_true", help="Skip fraud detection step")
    parser.add_argument("--verbose", action="store_true", help="Show detailed output")
    args = parser.parse_args()
    
    # Set logging level based on verbosity
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize clients
    neo4j_client = Neo4jClient()
    redis_client = RedisClient()
    
    # Print welcome message
    print_header("Blockchain Fraud Detection Demo")
    print("This script demonstrates the platform's capabilities by generating realistic")
    print("blockchain data with fraud patterns, ingesting it into Neo4j, and running")
    print("fraud detection algorithms.")
    print()
    
    # Generate data if not fraud-only or evidence-only
    if not args.fraud_only and not args.evidence_only:
        # Generate data
        data_generator = DataGenerator(args.addresses, args.transactions)
        data = data_generator.generate_all_data()
        
        # Ingest data into Neo4j
        neo4j_ingestor = Neo4jIngestor(neo4j_client)
        await neo4j_ingestor.ingest_all_data(data)
        
        # Set up Redis cache
        redis_setup = RedisSetup(redis_client)
        redis_setup.setup_cache_data(data)
    
    # Run fraud detection if not skipped
    if not args.no_detection:
        # Run fraud detection
        fraud_detector = FraudDetector(neo4j_client, redis_client)
        await fraud_detector.run_all_detections()
    
    # Clean up if requested
    if args.cleanup:
        print_header("Cleaning Up")
        
        # Clean up Neo4j data
        neo4j_ingestor = Neo4jIngestor(neo4j_client)
        await neo4j_ingestor.cleanup_data()
        
        # Clean up Redis cache
        redis_setup = RedisSetup(redis_client)
        redis_setup.cleanup_cache()
    
    print_header("Demo Complete")
    print("You have successfully run the blockchain fraud detection demo.")
    print("The platform now contains sample data and fraud patterns that you can")
    print("explore through the web interface or API.")
    print()
    print_info("Next steps:")
    print("1. Open the web interface to explore the data")
    print("2. Use the API to query for fraud patterns")
    print("3. Check the evidence bundles created for detected anomalies")
    print()
    print_success("Happy fraud hunting!")


if __name__ == "__main__":
    asyncio.run(main())
