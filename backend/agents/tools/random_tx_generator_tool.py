"""
RandomTxGeneratorTool for generating synthetic financial transactions.

This tool provides CrewAI agents (particularly the red_team_adversary) with the
ability to generate synthetic financial transactions for testing fraud detection
systems. It can create various suspicious patterns like circular transactions,
structuring, and shell company networks.
"""

import json
import logging
import random
import uuid
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timedelta

from crewai_tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class TransactionGeneratorInput(BaseModel):
    """Input model for transaction generation."""
    
    pattern_type: str = Field(
        default="random",
        description="Type of transaction pattern to generate: 'random', 'circular', 'structuring', 'layering', 'shell_company'"
    )
    num_transactions: int = Field(
        default=10,
        description="Number of transactions to generate"
    )
    num_entities: int = Field(
        default=5,
        description="Number of entities (accounts, companies, persons) to involve"
    )
    min_amount: float = Field(
        default=1000.0,
        description="Minimum transaction amount"
    )
    max_amount: float = Field(
        default=10000.0,
        description="Maximum transaction amount"
    )
    time_period_days: int = Field(
        default=30,
        description="Time period in days for the transactions"
    )
    currency: str = Field(
        default="USD",
        description="Currency for the transactions"
    )
    include_metadata: bool = Field(
        default=True,
        description="Whether to include additional metadata in transactions"
    )
    seed: Optional[int] = Field(
        default=None,
        description="Random seed for reproducibility"
    )
    output_format: str = Field(
        default="json",
        description="Output format: 'json', 'cypher', or 'csv'"
    )


class RandomTxGeneratorTool(BaseTool):
    """
    Tool for generating synthetic financial transactions.
    
    This tool allows agents to generate synthetic financial transactions
    for testing fraud detection systems. It can create various suspicious
    patterns like circular transactions, structuring, and shell company networks.
    """
    
    name: str = "random_tx_generator_tool"
    description: str = """
    Generate synthetic financial transactions for testing fraud detection.
    
    Use this tool when you need to:
    - Create test data for fraud detection systems
    - Simulate suspicious transaction patterns
    - Generate synthetic financial networks
    - Test detection algorithms with controlled data
    - Create red-team scenarios for blue-team testing
    
    The tool can generate various patterns including:
    - Random transactions
    - Circular transaction patterns
    - Structuring (multiple small transactions)
    - Layering (complex chains of transactions)
    - Shell company networks
    
    You can customize parameters like amount ranges, time periods, and the number of entities involved.
    """
    args_schema: type[BaseModel] = TransactionGeneratorInput
    
    def __init__(self):
        """Initialize the RandomTxGeneratorTool."""
        super().__init__()
        self.entity_types = ["Person", "Company", "Account"]
        self.countries = ["US", "UK", "CA", "DE", "FR", "CH", "SG", "HK", "JP", "AU"]
        self.transaction_types = ["TRANSFER", "PAYMENT", "WITHDRAWAL", "DEPOSIT", "INVESTMENT", "LOAN"]
        self.company_types = ["LLC", "Inc", "Ltd", "GmbH", "SA", "BV", "Pte Ltd"]
        self.high_risk_countries = ["CY", "MT", "PA", "VG", "KY", "BZ", "SC", "TC"]
    
    async def _arun(
        self,
        pattern_type: str = "random",
        num_transactions: int = 10,
        num_entities: int = 5,
        min_amount: float = 1000.0,
        max_amount: float = 10000.0,
        time_period_days: int = 30,
        currency: str = "USD",
        include_metadata: bool = True,
        seed: Optional[int] = None,
        output_format: str = "json"
    ) -> str:
        """
        Generate synthetic transactions asynchronously.
        
        Args:
            pattern_type: Type of transaction pattern to generate
            num_transactions: Number of transactions to generate
            num_entities: Number of entities to involve
            min_amount: Minimum transaction amount
            max_amount: Maximum transaction amount
            time_period_days: Time period in days for the transactions
            currency: Currency for the transactions
            include_metadata: Whether to include additional metadata
            seed: Random seed for reproducibility
            output_format: Output format
            
        Returns:
            JSON string containing the generated transactions
        """
        try:
            # Set random seed if provided
            if seed is not None:
                random.seed(seed)
            
            # Generate entities
            entities = self._generate_entities(num_entities)
            
            # Generate transactions based on pattern type
            if pattern_type == "circular":
                transactions = self._generate_circular_transactions(
                    entities, num_transactions, min_amount, max_amount, 
                    time_period_days, currency, include_metadata
                )
            elif pattern_type == "structuring":
                transactions = self._generate_structuring_transactions(
                    entities, num_transactions, min_amount, max_amount, 
                    time_period_days, currency, include_metadata
                )
            elif pattern_type == "layering":
                transactions = self._generate_layering_transactions(
                    entities, num_transactions, min_amount, max_amount, 
                    time_period_days, currency, include_metadata
                )
            elif pattern_type == "shell_company":
                transactions = self._generate_shell_company_transactions(
                    entities, num_transactions, min_amount, max_amount, 
                    time_period_days, currency, include_metadata
                )
            else:  # random
                transactions = self._generate_random_transactions(
                    entities, num_transactions, min_amount, max_amount, 
                    time_period_days, currency, include_metadata
                )
            
            # Format output
            if output_format == "cypher":
                output = self._format_as_cypher(entities, transactions)
            elif output_format == "csv":
                output = self._format_as_csv(transactions)
            else:  # json
                output = {
                    "entities": entities,
                    "transactions": transactions,
                    "metadata": {
                        "pattern_type": pattern_type,
                        "num_transactions": len(transactions),
                        "num_entities": len(entities),
                        "generated_at": datetime.now().isoformat(),
                        "currency": currency,
                        "seed": seed
                    }
                }
            
            return json.dumps({
                "success": True,
                "pattern_type": pattern_type,
                "data": output
            }, default=str)
            
        except Exception as e:
            logger.error(f"Error generating transactions: {e}", exc_info=True)
            return json.dumps({
                "success": False,
                "error": str(e)
            })
    
    def _run(
        self,
        pattern_type: str = "random",
        num_transactions: int = 10,
        num_entities: int = 5,
        min_amount: float = 1000.0,
        max_amount: float = 10000.0,
        time_period_days: int = 30,
        currency: str = "USD",
        include_metadata: bool = True,
        seed: Optional[int] = None,
        output_format: str = "json"
    ) -> str:
        """
        Synchronous wrapper for _arun.
        
        This method exists for compatibility with synchronous CrewAI operations.
        It should not be called directly in an async context.
        """
        import asyncio
        
        # Create a new event loop if needed
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(
            self._arun(
                pattern_type, num_transactions, num_entities, min_amount, max_amount,
                time_period_days, currency, include_metadata, seed, output_format
            )
        )
    
    def _generate_entities(self, num_entities: int) -> List[Dict[str, Any]]:
        """
        Generate synthetic entities (persons, companies, accounts).
        
        Args:
            num_entities: Number of entities to generate
            
        Returns:
            List of entity dictionaries
        """
        entities = []
        
        # Ensure at least one of each type
        min_of_each = min(num_entities, 3)
        
        # Generate persons
        for i in range(min_of_each):
            person_id = f"P{i+1:04d}"
            person = {
                "id": person_id,
                "type": "Person",
                "name": f"Person {i+1}",
                "country": random.choice(self.countries),
                "risk_score": random.uniform(0, 1),
                "pep": random.random() < 0.1,  # 10% chance of being PEP
                "created_at": (datetime.now() - timedelta(days=random.randint(1, 365 * 5))).isoformat()
            }
            entities.append(person)
        
        # Generate companies
        for i in range(min_of_each):
            company_id = f"C{i+1:04d}"
            high_risk = random.random() < 0.2  # 20% chance of high risk
            company = {
                "id": company_id,
                "type": "Company",
                "name": f"Company {i+1} {random.choice(self.company_types)}",
                "country": random.choice(self.high_risk_countries if high_risk else self.countries),
                "incorporation_date": (datetime.now() - timedelta(days=random.randint(1, 365 * 10))).isoformat(),
                "risk_score": random.uniform(0.5, 1.0) if high_risk else random.uniform(0, 0.5),
                "industry": random.choice(["Finance", "Technology", "Real Estate", "Retail", "Manufacturing"]),
                "created_at": (datetime.now() - timedelta(days=random.randint(1, 365 * 5))).isoformat()
            }
            entities.append(company)
        
        # Generate accounts
        for i in range(min_of_each):
            account_id = f"A{i+1:04d}"
            account = {
                "id": account_id,
                "type": "Account",
                "account_number": f"ACC{random.randint(10000000, 99999999)}",
                "bank": f"Bank {i+1}",
                "country": random.choice(self.countries),
                "balance": round(random.uniform(1000, 1000000), 2),
                "currency": "USD",
                "status": "Active",
                "created_at": (datetime.now() - timedelta(days=random.randint(1, 365 * 3))).isoformat()
            }
            entities.append(account)
        
        # Generate remaining entities randomly
        for i in range(min_of_each * 3, num_entities):
            entity_type = random.choice(self.entity_types)
            
            if entity_type == "Person":
                person_id = f"P{len([e for e in entities if e['type'] == 'Person']) + 1:04d}"
                entity = {
                    "id": person_id,
                    "type": "Person",
                    "name": f"Person {len([e for e in entities if e['type'] == 'Person']) + 1}",
                    "country": random.choice(self.countries),
                    "risk_score": random.uniform(0, 1),
                    "pep": random.random() < 0.1,  # 10% chance of being PEP
                    "created_at": (datetime.now() - timedelta(days=random.randint(1, 365 * 5))).isoformat()
                }
            elif entity_type == "Company":
                company_id = f"C{len([e for e in entities if e['type'] == 'Company']) + 1:04d}"
                high_risk = random.random() < 0.2  # 20% chance of high risk
                entity = {
                    "id": company_id,
                    "type": "Company",
                    "name": f"Company {len([e for e in entities if e['type'] == 'Company']) + 1} {random.choice(self.company_types)}",
                    "country": random.choice(self.high_risk_countries if high_risk else self.countries),
                    "incorporation_date": (datetime.now() - timedelta(days=random.randint(1, 365 * 10))).isoformat(),
                    "risk_score": random.uniform(0.5, 1.0) if high_risk else random.uniform(0, 0.5),
                    "industry": random.choice(["Finance", "Technology", "Real Estate", "Retail", "Manufacturing"]),
                    "created_at": (datetime.now() - timedelta(days=random.randint(1, 365 * 5))).isoformat()
                }
            else:  # Account
                account_id = f"A{len([e for e in entities if e['type'] == 'Account']) + 1:04d}"
                entity = {
                    "id": account_id,
                    "type": "Account",
                    "account_number": f"ACC{random.randint(10000000, 99999999)}",
                    "bank": f"Bank {random.randint(1, 10)}",
                    "country": random.choice(self.countries),
                    "balance": round(random.uniform(1000, 1000000), 2),
                    "currency": "USD",
                    "status": "Active",
                    "created_at": (datetime.now() - timedelta(days=random.randint(1, 365 * 3))).isoformat()
                }
            
            entities.append(entity)
        
        # Create relationships between entities
        for entity in entities:
            if entity["type"] == "Account":
                # Assign an owner (Person or Company)
                potential_owners = [e for e in entities if e["type"] in ["Person", "Company"]]
                if potential_owners:
                    owner = random.choice(potential_owners)
                    entity["owner_id"] = owner["id"]
                    entity["owner_type"] = owner["type"]
            
            if entity["type"] == "Company":
                # Assign directors (Persons)
                potential_directors = [e for e in entities if e["type"] == "Person"]
                if potential_directors:
                    num_directors = min(random.randint(1, 3), len(potential_directors))
                    entity["directors"] = [random.choice(potential_directors)["id"] for _ in range(num_directors)]
        
        return entities
    
    def _generate_random_transactions(
        self,
        entities: List[Dict[str, Any]],
        num_transactions: int,
        min_amount: float,
        max_amount: float,
        time_period_days: int,
        currency: str,
        include_metadata: bool
    ) -> List[Dict[str, Any]]:
        """
        Generate random transactions between entities.
        
        Args:
            entities: List of entity dictionaries
            num_transactions: Number of transactions to generate
            min_amount: Minimum transaction amount
            max_amount: Maximum transaction amount
            time_period_days: Time period in days for the transactions
            currency: Currency for the transactions
            include_metadata: Whether to include additional metadata
            
        Returns:
            List of transaction dictionaries
        """
        transactions = []
        accounts = [e for e in entities if e["type"] == "Account"]
        
        if len(accounts) < 2:
            raise ValueError("Need at least 2 accounts to generate transactions")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=time_period_days)
        
        for i in range(num_transactions):
            # Select source and destination accounts
            source = random.choice(accounts)
            dest = random.choice([a for a in accounts if a["id"] != source["id"]])
            
            # Generate transaction details
            tx_id = f"TX{i+1:06d}"
            tx_type = random.choice(self.transaction_types)
            amount = round(random.uniform(min_amount, max_amount), 2)
            
            # Generate random timestamp within the time period
            seconds_diff = int((end_date - start_date).total_seconds())
            random_seconds = random.randint(0, seconds_diff)
            timestamp = start_date + timedelta(seconds=random_seconds)
            
            transaction = {
                "id": tx_id,
                "source_id": source["id"],
                "destination_id": dest["id"],
                "type": tx_type,
                "amount": amount,
                "currency": currency,
                "timestamp": timestamp.isoformat()
            }
            
            # Add additional metadata if requested
            if include_metadata:
                transaction["reference"] = f"REF{uuid.uuid4().hex[:8].upper()}"
                transaction["status"] = random.choice(["Completed", "Pending", "Failed"])
                transaction["description"] = f"{tx_type} from {source['id']} to {dest['id']}"
                
                # Add risk indicators randomly
                if random.random() < 0.1:  # 10% chance
                    transaction["risk_indicators"] = random.sample([
                        "Unusual amount", "Cross-border", "High-risk country",
                        "Unusual time", "Unusual frequency", "New counterparty"
                    ], random.randint(1, 3))
            
            transactions.append(transaction)
        
        return transactions
    
    def _generate_circular_transactions(
        self,
        entities: List[Dict[str, Any]],
        num_transactions: int,
        min_amount: float,
        max_amount: float,
        time_period_days: int,
        currency: str,
        include_metadata: bool
    ) -> List[Dict[str, Any]]:
        """
        Generate circular transaction patterns (money laundering).
        
        Args:
            entities: List of entity dictionaries
            num_transactions: Number of transactions to generate
            min_amount: Minimum transaction amount
            max_amount: Maximum transaction amount
            time_period_days: Time period in days for the transactions
            currency: Currency for the transactions
            include_metadata: Whether to include additional metadata
            
        Returns:
            List of transaction dictionaries
        """
        transactions = []
        accounts = [e for e in entities if e["type"] == "Account"]
        
        if len(accounts) < 3:
            raise ValueError("Need at least 3 accounts to generate circular transactions")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=time_period_days)
        
        # Determine number of circular patterns
        num_patterns = max(1, num_transactions // 10)
        
        for pattern_idx in range(num_patterns):
            # Select accounts for this circular pattern
            circle_size = random.randint(3, min(8, len(accounts)))
            circle_accounts = random.sample(accounts, circle_size)
            
            # Determine base amount for this pattern
            base_amount = round(random.uniform(min_amount, max_amount), 2)
            
            # Generate timestamps for this pattern (in chronological order)
            seconds_diff = int((end_date - start_date).total_seconds())
            pattern_start = start_date + timedelta(seconds=random.randint(0, seconds_diff // 2))
            
            # Create transactions in the circle
            for j in range(circle_size):
                source = circle_accounts[j]
                dest = circle_accounts[(j + 1) % circle_size]
                
                # Vary amount slightly to avoid exact matches
                amount = round(base_amount * random.uniform(0.95, 1.05), 2)
                
                # Add some time between transactions
                timestamp = pattern_start + timedelta(hours=j * random.randint(4, 24))
                
                tx_id = f"TX{len(transactions) + 1:06d}"
                tx_type = random.choice(self.transaction_types)
                
                transaction = {
                    "id": tx_id,
                    "source_id": source["id"],
                    "destination_id": dest["id"],
                    "type": tx_type,
                    "amount": amount,
                    "currency": currency,
                    "timestamp": timestamp.isoformat(),
                    "pattern": f"circular_{pattern_idx}"
                }
                
                # Add additional metadata if requested
                if include_metadata:
                    transaction["reference"] = f"REF{uuid.uuid4().hex[:8].upper()}"
                    transaction["status"] = "Completed"
                    transaction["description"] = f"{tx_type} from {source['id']} to {dest['id']}"
                    
                    # Add risk indicators
                    if j == circle_size - 1:  # Last transaction in the circle
                        transaction["risk_indicators"] = ["Circular pattern", "Money laundering risk"]
                
                transactions.append(transaction)
        
        # Fill remaining transactions with random ones
        remaining = num_transactions - len(transactions)
        if remaining > 0:
            random_txs = self._generate_random_transactions(
                entities, remaining, min_amount, max_amount, 
                time_period_days, currency, include_metadata
            )
            transactions.extend(random_txs)
        
        return transactions
    
    def _generate_structuring_transactions(
        self,
        entities: List[Dict[str, Any]],
        num_transactions: int,
        min_amount: float,
        max_amount: float,
        time_period_days: int,
        currency: str,
        include_metadata: bool
    ) -> List[Dict[str, Any]]:
        """
        Generate structuring transaction patterns (breaking large amounts into smaller ones).
        
        Args:
            entities: List of entity dictionaries
            num_transactions: Number of transactions to generate
            min_amount: Minimum transaction amount
            max_amount: Maximum transaction amount
            time_period_days: Time period in days for the transactions
            currency: Currency for the transactions
            include_metadata: Whether to include additional metadata
            
        Returns:
            List of transaction dictionaries
        """
        transactions = []
        accounts = [e for e in entities if e["type"] == "Account"]
        
        if len(accounts) < 2:
            raise ValueError("Need at least 2 accounts to generate structuring transactions")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=time_period_days)
        
        # Determine number of structuring patterns
        num_patterns = max(1, num_transactions // 15)
        
        for pattern_idx in range(num_patterns):
            # Select source and destination accounts
            source = random.choice(accounts)
            dest = random.choice([a for a in accounts if a["id"] != source["id"]])
            
            # Determine large amount to be broken down
            large_amount = round(random.uniform(max_amount * 2, max_amount * 5), 2)
            
            # Determine number of smaller transactions
            num_small_txs = random.randint(5, 15)
            
            # Calculate small transaction amounts (should sum to large_amount)
            small_amounts = []
            remaining = large_amount
            
            for i in range(num_small_txs - 1):
                # Keep amounts just below reporting thresholds
                amount = round(min(remaining * random.uniform(0.05, 0.2), 9900), 2)
                small_amounts.append(amount)
                remaining -= amount
            
            # Add the final amount
            small_amounts.append(round(remaining, 2))
            
            # Generate timestamps for this pattern (in chronological order)
            seconds_diff = int((end_date - start_date).total_seconds())
            pattern_start = start_date + timedelta(seconds=random.randint(0, seconds_diff // 2))
            
            # Create the transactions
            for j, amount in enumerate(small_amounts):
                # Add some time between transactions
                timestamp = pattern_start + timedelta(hours=j * random.randint(1, 12))
                
                tx_id = f"TX{len(transactions) + 1:06d}"
                tx_type = "TRANSFER"
                
                transaction = {
                    "id": tx_id,
                    "source_id": source["id"],
                    "destination_id": dest["id"],
                    "type": tx_type,
                    "amount": amount,
                    "currency": currency,
                    "timestamp": timestamp.isoformat(),
                    "pattern": f"structuring_{pattern_idx}"
                }
                
                # Add additional metadata if requested
                if include_metadata:
                    transaction["reference"] = f"REF{uuid.uuid4().hex[:8].upper()}"
                    transaction["status"] = "Completed"
                    transaction["description"] = f"{tx_type} from {source['id']} to {dest['id']}"
                    
                    # Add risk indicators
                    transaction["risk_indicators"] = ["Structuring", "Just below threshold"]
                
                transactions.append(transaction)
        
        # Fill remaining transactions with random ones
        remaining = num_transactions - len(transactions)
        if remaining > 0:
            random_txs = self._generate_random_transactions(
                entities, remaining, min_amount, max_amount, 
                time_period_days, currency, include_metadata
            )
            transactions.extend(random_txs)
        
        return transactions
    
    def _generate_layering_transactions(
        self,
        entities: List[Dict[str, Any]],
        num_transactions: int,
        min_amount: float,
        max_amount: float,
        time_period_days: int,
        currency: str,
        include_metadata: bool
    ) -> List[Dict[str, Any]]:
        """
        Generate layering transaction patterns (complex chains).
        
        Args:
            entities: List of entity dictionaries
            num_transactions: Number of transactions to generate
            min_amount: Minimum transaction amount
            max_amount: Maximum transaction amount
            time_period_days: Time period in days for the transactions
            currency: Currency for the transactions
            include_metadata: Whether to include additional metadata
            
        Returns:
            List of transaction dictionaries
        """
        transactions = []
        accounts = [e for e in entities if e["type"] == "Account"]
        
        if len(accounts) < 5:
            raise ValueError("Need at least 5 accounts to generate layering transactions")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=time_period_days)
        
        # Determine number of layering patterns
        num_patterns = max(1, num_transactions // 20)
        
        for pattern_idx in range(num_patterns):
            # Select source and final destination accounts
            source = random.choice(accounts)
            final_dest = random.choice([a for a in accounts if a["id"] != source["id"]])
            
            # Select intermediary accounts (3-5)
            num_intermediaries = random.randint(3, min(5, len(accounts) - 2))
            intermediaries = random.sample(
                [a for a in accounts if a["id"] != source["id"] and a["id"] != final_dest["id"]],
                num_intermediaries
            )
            
            # Create the full path
            path = [source] + intermediaries + [final_dest]
            
            # Determine initial amount
            initial_amount = round(random.uniform(max_amount, max_amount * 3), 2)
            
            # Generate timestamps for this pattern (in chronological order)
            seconds_diff = int((end_date - start_date).total_seconds())
            pattern_start = start_date + timedelta(seconds=random.randint(0, seconds_diff // 2))
            
            # Create the transactions along the path
            current_amount = initial_amount
            
            for j in range(len(path) - 1):
                src = path[j]
                dst = path[j + 1]
                
                # Reduce amount slightly at each step (fees)
                amount = round(current_amount * random.uniform(0.95, 0.99), 2)
                current_amount = amount
                
                # Add increasing time between transactions
                timestamp = pattern_start + timedelta(days=j * random.randint(1, 5))
                
                tx_id = f"TX{len(transactions) + 1:06d}"
                tx_type = random.choice(["TRANSFER", "PAYMENT"])
                
                transaction = {
                    "id": tx_id,
                    "source_id": src["id"],
                    "destination_id": dst["id"],
                    "type": tx_type,
                    "amount": amount,
                    "currency": currency,
                    "timestamp": timestamp.isoformat(),
                    "pattern": f"layering_{pattern_idx}",
                    "step": j + 1,
                    "path_length": len(path) - 1
                }
                
                # Add additional metadata if requested
                if include_metadata:
                    transaction["reference"] = f"REF{uuid.uuid4().hex[:8].upper()}"
                    transaction["status"] = "Completed"
                    transaction["description"] = f"{tx_type} from {src['id']} to {dst['id']}"
                    
                    # Add risk indicators
                    if j == 0:
                        transaction["risk_indicators"] = ["Layering start", "Large amount"]
                    elif j == len(path) - 2:
                        transaction["risk_indicators"] = ["Layering end", "Money laundering risk"]
                    else:
                        transaction["risk_indicators"] = ["Layering intermediary"]
                
                transactions.append(transaction)
        
        # Fill remaining transactions with random ones
        remaining = num_transactions - len(transactions)
        if remaining > 0:
            random_txs = self._generate_random_transactions(
                entities, remaining, min_amount, max_amount, 
                time_period_days, currency, include_metadata
            )
            transactions.extend(random_txs)
        
        return transactions
    
    def _generate_shell_company_transactions(
        self,
        entities: List[Dict[str, Any]],
        num_transactions: int,
        min_amount: float,
        max_amount: float,
        time_period_days: int,
        currency: str,
        include_metadata: bool
    ) -> List[Dict[str, Any]]:
        """
        Generate shell company transaction patterns.
        
        Args:
            entities: List of entity dictionaries
            num_transactions: Number of transactions to generate
            min_amount: Minimum transaction amount
            max_amount: Maximum transaction amount
            time_period_days: Time period in days for the transactions
            currency: Currency for the transactions
            include_metadata: Whether to include additional metadata
            
        Returns:
            List of transaction dictionaries
        """
        transactions = []
        
        # Get companies and accounts
        companies = [e for e in entities if e["type"] == "Company"]
        accounts = [e for e in entities if e["type"] == "Account"]
        
        if len(companies) < 3 or len(accounts) < 3:
            raise ValueError("Need at least 3 companies and 3 accounts to generate shell company transactions")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=time_period_days)
        
        # Create shell companies (high risk)
        shell_companies = []
        for i in range(min(3, len(companies))):
            company = companies[i]
            company["shell_company"] = True
            company["risk_score"] = random.uniform(0.7, 1.0)
            company["country"] = random.choice(self.high_risk_countries)
            shell_companies.append(company)
        
        # Assign accounts to shell companies
        shell_accounts = []
        for i, company in enumerate(shell_companies):
            if i < len(accounts):
                account = accounts[i]
                account["owner_id"] = company["id"]
                account["owner_type"] = "Company"
                shell_accounts.append(account)
        
        # Generate invoice fraud transactions
        num_patterns = max(1, num_transactions // 15)
        
        for pattern_idx in range(num_patterns):
            # Select source (legitimate) account
            legitimate_accounts = [a for a in accounts if a not in shell_accounts]
            if not legitimate_accounts:
                break
                
            source = random.choice(legitimate_accounts)
            
            # Select destination (shell company) account
            if not shell_accounts:
                break
                
            dest = random.choice(shell_accounts)
            
            # Determine invoice amount
            invoice_amount = round(random.uniform(max_amount, max_amount * 5), 2)
            
            # Generate timestamp
            seconds_diff = int((end_date - start_date).total_seconds())
            timestamp = start_date + timedelta(seconds=random.randint(0, seconds_diff))
            
            # Create the invoice payment transaction
            tx_id = f"TX{len(transactions) + 1:06d}"
            
            transaction = {
                "id": tx_id,
                "source_id": source["id"],
                "destination_id": dest["id"],
                "type": "PAYMENT",
                "amount": invoice_amount,
                "currency": currency,
                "timestamp": timestamp.isoformat(),
                "pattern": f"shell_company_{pattern_idx}"
            }
            
            # Add additional metadata if requested
            if include_metadata:
                transaction["reference"] = f"INV{random.randint(10000, 99999)}"
                transaction["status"] = "Completed"
                transaction["description"] = f"Invoice payment to {dest['id']}"
                transaction["invoice_number"] = f"INV-{random.randint(1000, 9999)}"
                transaction["invoice_date"] = (datetime.fromisoformat(timestamp) - timedelta(days=random.randint(7, 30))).isoformat()
                transaction["risk_indicators"] = ["Shell company recipient", "Unusual invoice amount"]
            
            transactions.append(transaction)
            
            # Generate follow-up transactions (moving money out of shell company)
            num_followups = random.randint(2, 5)
            remaining = invoice_amount
            
            for j in range(num_followups):
                # Determine amount for this transaction
                if j == num_followups - 1:
                    amount = round(remaining, 2)
                else:
                    amount = round(remaining * random.uniform(0.2, 0.5), 2)
                    remaining -= amount
                
                # Select destination (another shell company or offshore account)
                if j < num_followups - 1 and len(shell_accounts) > 1:
                    next_dest = random.choice([a for a in shell_accounts if a["id"] != dest["id"]])
                else:
                    # Final destination (offshore or back to legitimate but different account)
                    if random.random() < 0.7 and len(legitimate_accounts) > 1:
                        next_dest = random.choice([a for a in legitimate_accounts if a["id"] != source["id"]])
                    else:
                        next_dest = random.choice(shell_accounts)
                
                # Add time delay
                followup_timestamp = (datetime.fromisoformat(timestamp) + 
                                     timedelta(days=random.randint(1, 10))).isoformat()
                
                followup_tx_id = f"TX{len(transactions) + 1:06d}"
                
                followup_tx = {
                    "id": followup_tx_id,
                    "source_id": dest["id"],
                    "destination_id": next_dest["id"],
                    "type": random.choice(["TRANSFER", "WITHDRAWAL"]),
                    "amount": amount,
                    "currency": currency,
                    "timestamp": followup_timestamp,
                    "pattern": f"shell_company_{pattern_idx}",
                    "related_to": tx_id
                }
                
                # Add additional metadata if requested
                if include_metadata:
                    followup_tx["reference"] = f"REF{uuid.uuid4().hex[:8].upper()}"
                    followup_tx["status"] = "Completed"
                    followup_tx["description"] = f"Transfer from {dest['id']} to {next_dest['id']}"
                    followup_tx["risk_indicators"] = ["Shell company transfer", "Layering activity"]
                
                transactions.append(followup_tx)
                dest = next_dest  # For next iteration
        
        # Fill remaining transactions with random ones
        remaining = num_transactions - len(transactions)
        if remaining > 0:
            random_txs = self._generate_random_transactions(
                entities, remaining, min_amount, max_amount, 
                time_period_days, currency, include_metadata
            )
            transactions.extend(random_txs)
        
        return transactions
    
    def _format_as_cypher(
        self,
        entities: List[Dict[str, Any]],
        transactions: List[Dict[str, Any]]
    ) -> str:
        """
        Format entities and transactions as Cypher statements.
        
        Args:
            entities: List of entity dictionaries
            transactions: List of transaction dictionaries
            
        Returns:
            String of Cypher statements
        """
        cypher = []
        
        # Create entity nodes
        for entity in entities:
            if entity["type"] == "Person":
                props = {k: v for k, v in entity.items() if k not in ["type", "id"]}
                cypher.append(
                    f"CREATE (p:Person {{id: '{entity['id']}', {', '.join([f'{k}: {json.dumps(v)}' for k, v in props.items()])}}})"
                )
            elif entity["type"] == "Company":
                props = {k: v for k, v in entity.items() if k not in ["type", "id", "directors"]}
                cypher.append(
                    f"CREATE (c:Company {{id: '{entity['id']}', {', '.join([f'{k}: {json.dumps(v)}' for k, v in props.items()])}}})"
                )
            elif entity["type"] == "Account":
                props = {k: v for k, v in entity.items() if k not in ["type", "id", "owner_id", "owner_type"]}
                cypher.append(
                    f"CREATE (a:Account {{id: '{entity['id']}', {', '.join([f'{k}: {json.dumps(v)}' for k, v in props.items()])}}})"
                )
        
        # Create relationships between entities
        for entity in entities:
            if entity["type"] == "Account" and "owner_id" in entity and "owner_type" in entity:
                cypher.append(
                    f"MATCH (a:Account {{id: '{entity['id']}'}}), (o:{entity['owner_type']} {{id: '{entity['owner_id']}'}}) "
                    f"CREATE (o)-[:OWNS]->(a)"
                )
            
            if entity["type"] == "Company" and "directors" in entity:
                for director_id in entity["directors"]:
                    cypher.append(
                        f"MATCH (c:Company {{id: '{entity['id']}'}}), (p:Person {{id: '{director_id}'}}) "
                        f"CREATE (p)-[:DIRECTS]->(c)"
                    )
        
        # Create transaction relationships
        for tx in transactions:
            props = {k: v for k, v in tx.items() if k not in ["id", "source_id", "destination_id"]}
            cypher.append(
                f"MATCH (src:Account {{id: '{tx['source_id']}'}}), (dst:Account {{id: '{tx['destination_id']}'}}) "
                f"CREATE (src)-[:TRANSACTION {{id: '{tx['id']}', {', '.join([f'{k}: {json.dumps(v)}' for k, v in props.items()])}}]->(dst)"
            )
        
        return ";\n\n".join(cypher) + ";"
    
    def _format_as_csv(self, transactions: List[Dict[str, Any]]) -> str:
        """
        Format transactions as CSV.
        
        Args:
            transactions: List of transaction dictionaries
            
        Returns:
            CSV string
        """
        if not transactions:
            return ""
        
        # Get all possible headers from all transactions
        headers = set()
        for tx in transactions:
            headers.update(tx.keys())
        
        headers = sorted(list(headers))
        
        # Create CSV content
        csv_lines = [",".join(headers)]
        
        for tx in transactions:
            csv_lines.append(",".join([
                f'"{str(tx.get(header, ""))}"' for header in headers
            ]))
        
        return "\n".join(csv_lines)
