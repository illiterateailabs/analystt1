"""
Sim API Graph Ingestion Tool

This tool processes data from Sim APIs (wallet balances and activity) into Neo4j graph
relationships. It creates and updates nodes for wallets, tokens, transactions, and contracts,
and establishes relationships between them to enable advanced fraud detection and analysis.

Usage:
    tool = SimGraphIngestionTool()
    result = await tool.run(wallet_address="0xd8da6bf26964af9d7eed9e03e53415d37aa96045")
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Set

from pydantic import BaseModel, Field

from backend.agents.tools.base_tool import BaseTool
from backend.integrations.neo4j_client import Neo4jClient
from backend.integrations.sim_client import SimClient, SimApiError
from backend.config import settings

logger = logging.getLogger(__name__)

class SimGraphIngestionTool(BaseTool):
    """
    Tool for ingesting Sim API data (wallet balances and activity) into Neo4j.

    This tool processes data fetched from Sim APIs and transforms it into
    graph structures within Neo4j, creating nodes for wallets, tokens,
    transactions, and contracts, and establishing relationships between them.
    It supports batch processing for efficient data loading.
    """

    name = "sim_graph_ingestion_tool"
    description = """
    Ingests blockchain data from Sim APIs (wallet balances and activity) into Neo4j.
    Creates and updates nodes for CryptoWallets, Tokens, Transactions, and Contracts,
    and establishes relationships like OWNS_BALANCE, SENT, RECEIVED, INTERACTED_WITH.
    Use this tool to enrich the graph database with real-time blockchain data.
    """

    def __init__(self, neo4j_client: Optional[Neo4jClient] = None, sim_client: Optional[SimClient] = None):
        super().__init__()
        self.neo4j = neo4j_client or Neo4jClient(
            uri=settings.NEO4J_URI,
            username=settings.NEO4J_USERNAME,
            password=settings.NEO4J_PASSWORD
        )
        self.sim = sim_client or SimClient(
            api_key=settings.SIM_API_KEY,
            base_url=settings.SIM_API_URL
        )

    async def run(
        self, 
        wallet_address: str, 
        ingest_balances: bool = True, 
        ingest_activity: bool = True,
        limit_balances: int = 100,
        limit_activity: int = 50,
        chain_ids: str = "all",
        create_schema: bool = False
    ) -> Dict[str, Any]:
        """
        Ingests Sim API data for a given wallet address into Neo4j.

        Args:
            wallet_address: The blockchain wallet address to ingest data for.
            ingest_balances: Whether to ingest token balance data.
            ingest_activity: Whether to ingest transaction activity data.
            limit_balances: Maximum number of balances to fetch.
            limit_activity: Maximum number of activities to fetch.
            chain_ids: Comma-separated list of chain IDs or "all" for all chains.
            create_schema: Whether to create necessary Neo4j schema constraints and indexes.

        Returns:
            A dictionary summarizing the ingestion results.
        """
        if not wallet_address:
            raise ValueError("Wallet address cannot be empty.")

        results = {
            "wallet_address": wallet_address,
            "balances_ingested": 0,
            "activities_ingested": 0,
            "tokens_created": 0,
            "transactions_created": 0,
            "contracts_created": 0,
            "relationships_created": 0,
            "errors": []
        }

        try:
            # Create schema if requested
            if create_schema:
                await self._ensure_schema()
                results["schema_created"] = True

            # Ensure the CryptoWallet node exists
            await self._create_or_update_wallet(wallet_address)
            results["wallet_created"] = True

            # Process balances
            if ingest_balances:
                logger.info(f"Fetching and ingesting balances for {wallet_address}...")
                balances_response = await self.sim.get_balances(
                    wallet_address, 
                    limit=limit_balances,
                    chain_ids=chain_ids,
                    metadata="url,logo"
                )
                
                if balances_response and "balances" in balances_response:
                    balances_result = await self._ingest_balances(wallet_address, balances_response["balances"])
                    results.update(balances_result)
                    logger.info(f"Ingested {balances_result['balances_ingested']} balances for {wallet_address}.")
                else:
                    logger.info(f"No balances found for {wallet_address}.")

            # Process activity
            if ingest_activity:
                logger.info(f"Fetching and ingesting activity for {wallet_address}...")
                activity_response = await self.sim.get_activity(wallet_address, limit=limit_activity)
                
                if activity_response and "activity" in activity_response:
                    activity_result = await self._ingest_activity(wallet_address, activity_response["activity"])
                    results.update(activity_result)
                    logger.info(f"Ingested {activity_result['activities_ingested']} activities for {wallet_address}.")
                else:
                    logger.info(f"No activities found for {wallet_address}.")

        except SimApiError as e:
            error_msg = f"Sim API error during ingestion for {wallet_address}: {e.message} (Status: {e.status_code})"
            logger.error(error_msg)
            results["errors"].append(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error during ingestion for {wallet_address}: {str(e)}"
            logger.exception(error_msg)
            results["errors"].append(error_msg)

        return results

    async def _ensure_schema(self) -> None:
        """
        Ensures that the necessary Neo4j schema constraints and indexes exist.
        This includes constraints for CryptoWallet, Token, and Transaction nodes,
        and indexes for efficient querying.
        """
        logger.info("Creating Neo4j schema constraints and indexes for blockchain data...")
        
        schema_queries = [
            # Constraints
            """
            CREATE CONSTRAINT crypto_wallet_address_unique IF NOT EXISTS 
            FOR (w:CryptoWallet) REQUIRE w.address IS UNIQUE
            """,
            """
            CREATE CONSTRAINT token_chain_address_unique IF NOT EXISTS 
            FOR (t:Token) REQUIRE (t.chain_id, t.address) IS NODE KEY
            """,
            """
            CREATE CONSTRAINT transaction_hash_unique IF NOT EXISTS 
            FOR (tx:Transaction) REQUIRE tx.hash IS UNIQUE
            """,
            """
            CREATE CONSTRAINT contract_address_unique IF NOT EXISTS 
            FOR (c:Contract) REQUIRE (c.chain_id, c.address) IS NODE KEY
            """,
            
            # Indexes
            """
            CREATE INDEX token_symbol_index IF NOT EXISTS 
            FOR (t:Token) ON (t.symbol)
            """,
            """
            CREATE INDEX token_chain_index IF NOT EXISTS 
            FOR (t:Token) ON (t.chain_id)
            """,
            """
            CREATE INDEX transaction_block_time_index IF NOT EXISTS 
            FOR (tx:Transaction) ON (tx.block_time)
            """,
            """
            CREATE INDEX transaction_type_index IF NOT EXISTS 
            FOR (tx:Transaction) ON (tx.type)
            """,
            """
            CREATE INDEX contract_chain_index IF NOT EXISTS 
            FOR (c:Contract) ON (c.chain_id)
            """
        ]
        
        for query in schema_queries:
            try:
                await self.neo4j.execute_query(query)
            except Exception as e:
                logger.error(f"Error creating schema: {str(e)}")
                raise

    async def _create_or_update_wallet(self, wallet_address: str) -> None:
        """
        Creates or updates a CryptoWallet node in Neo4j.
        
        Args:
            wallet_address: The blockchain wallet address.
        """
        query = """
        MERGE (w:CryptoWallet {address: $wallet_address})
        ON CREATE SET 
            w.created_at = datetime(),
            w.updated_at = datetime(),
            w.last_seen = datetime()
        ON MATCH SET 
            w.updated_at = datetime(),
            w.last_seen = datetime()
        RETURN w
        """
        await self.neo4j.execute_query(query, {"wallet_address": wallet_address.lower()})
        logger.debug(f"Created or updated CryptoWallet node for {wallet_address}")

    async def _ingest_balances(self, wallet_address: str, balances: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Ingests token balances into Neo4j.
        
        Args:
            wallet_address: The wallet address.
            balances: List of token balance objects from Sim API.
            
        Returns:
            Dictionary with ingestion statistics.
        """
        if not balances:
            return {"balances_ingested": 0, "tokens_created": 0, "relationships_created": 0}
        
        # Prepare data for batching
        balance_data = []
        for balance in balances:
            token_address = balance.get("address", "native")
            
            # Handle native tokens specially
            if token_address == "native":
                token_address = f"native-{balance.get('chain_id', 0)}"
            
            balance_data.append({
                "wallet_address": wallet_address.lower(),
                "token_address": token_address.lower(),
                "chain": balance.get("chain"),
                "chain_id": balance.get("chain_id"),
                "amount": balance.get("amount"),
                "value_usd": balance.get("value_usd"),
                "symbol": balance.get("symbol"),
                "name": balance.get("name") or balance.get("symbol"),
                "decimals": balance.get("decimals"),
                "logo": balance.get("token_metadata", {}).get("logo"),
                "url": balance.get("token_metadata", {}).get("url"),
                "low_liquidity": balance.get("low_liquidity", False),
                "pool_size": balance.get("pool_size"),
                "price_usd": balance.get("price_usd"),
                "block_time": balance.get("block_time") or datetime.now().isoformat(),
                "is_native": token_address == f"native-{balance.get('chain_id', 0)}"
            })

        # Execute batch query to create/update tokens and relationships
        query = """
        UNWIND $balance_data AS data
        
        // Ensure wallet node exists
        MERGE (w:CryptoWallet {address: data.wallet_address})
        ON CREATE SET 
            w.created_at = datetime(),
            w.updated_at = datetime(),
            w.last_seen = datetime()
        ON MATCH SET 
            w.updated_at = datetime(),
            w.last_seen = datetime()
        
        // Create or update token node
        MERGE (t:Token {address: data.token_address, chain_id: data.chain_id})
        ON CREATE SET 
            t.created_at = datetime(),
            t.updated_at = datetime(),
            t.symbol = data.symbol,
            t.name = data.name,
            t.decimals = data.decimals,
            t.logo = data.logo,
            t.url = data.url,
            t.chain = data.chain,
            t.is_native = data.is_native
        ON MATCH SET 
            t.updated_at = datetime(),
            t.symbol = COALESCE(data.symbol, t.symbol),
            t.name = COALESCE(data.name, t.name),
            t.decimals = COALESCE(data.decimals, t.decimals),
            t.logo = COALESCE(data.logo, t.logo),
            t.url = COALESCE(data.url, t.url),
            t.chain = COALESCE(data.chain, t.chain)
        
        // Update token pricing information
        SET 
            t.price_usd = CASE WHEN data.price_usd IS NOT NULL THEN data.price_usd ELSE t.price_usd END,
            t.pool_size = CASE WHEN data.pool_size IS NOT NULL THEN data.pool_size ELSE t.pool_size END,
            t.low_liquidity = CASE WHEN data.low_liquidity IS NOT NULL THEN data.low_liquidity ELSE t.low_liquidity END
        
        // Create or update OWNS_BALANCE relationship
        MERGE (w)-[r:OWNS_BALANCE]->(t)
        SET 
            r.amount = data.amount,
            r.value_usd = data.value_usd,
            r.last_updated = datetime(),
            r.block_time = datetime(data.block_time)
            
        RETURN count(t) as tokens_created, count(r) as relationships_created
        """
        
        try:
            result = await self.neo4j.execute_query(query, {"balance_data": balance_data})
            
            # Extract statistics from result
            tokens_created = sum(record["tokens_created"] for record in result)
            relationships_created = sum(record["relationships_created"] for record in result)
            
            return {
                "balances_ingested": len(balances),
                "tokens_created": tokens_created,
                "relationships_created": relationships_created
            }
        except Exception as e:
            logger.error(f"Error ingesting balances: {str(e)}")
            raise

    async def _ingest_activity(self, wallet_address: str, activities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Ingests transaction activities into Neo4j.
        
        Args:
            wallet_address: The wallet address.
            activities: List of activity objects from Sim API.
            
        Returns:
            Dictionary with ingestion statistics.
        """
        if not activities:
            return {
                "activities_ingested": 0,
                "transactions_created": 0,
                "contracts_created": 0,
                "relationships_created": 0
            }
        
        # Track statistics
        stats = {
            "activities_ingested": len(activities),
            "transactions_created": 0,
            "contracts_created": 0,
            "relationships_created": 0
        }
        
        # Process each activity type
        for activity in activities:
            tx_hash = activity.get("transaction_hash")
            if not tx_hash:
                logger.warning(f"Skipping activity without transaction_hash: {activity}")
                continue
            
            # Create transaction node
            tx_created = await self._create_or_update_transaction(activity)
            if tx_created:
                stats["transactions_created"] += 1
            
            # Handle different activity types
            activity_type = activity.get("type")
            
            if activity_type in ["send", "receive"]:
                # Handle token transfers
                rel_created = await self._process_transfer(wallet_address, activity)
                stats["relationships_created"] += rel_created
                
            elif activity_type == "call":
                # Handle contract interactions
                contract_created, rel_created = await self._process_contract_call(wallet_address, activity)
                stats["contracts_created"] += contract_created
                stats["relationships_created"] += rel_created
                
            elif activity_type in ["mint", "burn"]:
                # Handle token minting/burning
                rel_created = await self._process_token_event(wallet_address, activity)
                stats["relationships_created"] += rel_created
                
            elif activity_type in ["swap", "approve"]:
                # Handle swaps and approvals
                rel_created = await self._process_token_action(wallet_address, activity)
                stats["relationships_created"] += rel_created
        
        return stats

    async def _create_or_update_transaction(self, activity: Dict[str, Any]) -> bool:
        """
        Creates or updates a Transaction node in Neo4j.
        
        Args:
            activity: The activity data containing transaction information.
            
        Returns:
            Boolean indicating if a new transaction was created.
        """
        tx_hash = activity.get("transaction_hash")
        
        query = """
        MERGE (tx:Transaction {hash: $hash})
        ON CREATE SET 
            tx.created_at = datetime(),
            tx.updated_at = datetime(),
            tx.type = $type,
            tx.chain = $chain,
            tx.chain_id = $chain_id,
            tx.block_number = $block_number,
            tx.block_time = datetime($block_time),
            tx.value_usd = $value_usd
        ON MATCH SET 
            tx.updated_at = datetime(),
            tx.type = COALESCE($type, tx.type),
            tx.chain = COALESCE($chain, tx.chain),
            tx.chain_id = COALESCE($chain_id, tx.chain_id),
            tx.block_number = COALESCE($block_number, tx.block_number),
            tx.block_time = COALESCE(datetime($block_time), tx.block_time),
            tx.value_usd = COALESCE($value_usd, tx.value_usd)
        RETURN tx.created_at = datetime() as created_new
        """
        
        params = {
            "hash": tx_hash.lower(),
            "type": activity.get("type"),
            "chain": activity.get("chain"),
            "chain_id": activity.get("chain_id"),
            "block_number": activity.get("block_number"),
            "block_time": activity.get("block_time"),
            "value_usd": activity.get("value_usd")
        }
        
        result = await self.neo4j.execute_query(query, params)
        return len(result) > 0 and result[0].get("created_new", False)

    async def _process_transfer(self, wallet_address: str, activity: Dict[str, Any]) -> int:
        """
        Processes a token transfer activity (send/receive).
        
        Args:
            wallet_address: The wallet address being processed.
            activity: The activity data containing transfer information.
            
        Returns:
            Number of relationships created.
        """
        tx_hash = activity.get("transaction_hash")
        from_address = activity.get("from_address")
        to_address = activity.get("to_address")
        token_address = activity.get("token_address")
        activity_type = activity.get("type")
        
        if not (tx_hash and (from_address or to_address)):
            logger.warning(f"Skipping transfer with missing data: {activity}")
            return 0
        
        # Create token node if token transfer
        token_created = False
        if token_address and activity.get("token_metadata"):
            token_created = await self._create_or_update_token(
                token_address,
                activity.get("chain_id"),
                activity.get("token_metadata")
            )
        
        # Create relationships based on activity type
        query = """
        MATCH (tx:Transaction {hash: $tx_hash})
        
        // Ensure from_wallet exists
        MERGE (from:CryptoWallet {address: $from_address})
        ON CREATE SET 
            from.created_at = datetime(),
            from.updated_at = datetime()
        ON MATCH SET 
            from.updated_at = datetime()
            
        // Ensure to_wallet exists
        MERGE (to:CryptoWallet {address: $to_address})
        ON CREATE SET 
            to.created_at = datetime(),
            to.updated_at = datetime()
        ON MATCH SET 
            to.updated_at = datetime()
        
        // Create SENT relationship
        MERGE (from)-[s:SENT]->(tx)
        ON CREATE SET 
            s.created_at = datetime(),
            s.amount = $amount,
            s.value_usd = $value_usd
        
        // Create RECEIVED relationship
        MERGE (tx)-[r:RECEIVED_BY]->(to)
        ON CREATE SET 
            r.created_at = datetime(),
            r.amount = $amount,
            r.value_usd = $value_usd
        """
        
        params = {
            "tx_hash": tx_hash.lower(),
            "from_address": (from_address or "unknown").lower(),
            "to_address": (to_address or "unknown").lower(),
            "amount": activity.get("amount"),
            "value_usd": activity.get("value_usd")
        }
        
        # Add token relationship if token transfer
        if token_address:
            query += """
            // Link transaction to token
            WITH tx
            MATCH (token:Token {address: $token_address, chain_id: $chain_id})
            MERGE (tx)-[tf:TRANSFERS_TOKEN]->(token)
            ON CREATE SET 
                tf.created_at = datetime(),
                tf.amount = $amount
            """
            params["token_address"] = token_address.lower()
            params["chain_id"] = activity.get("chain_id")
        
        await self.neo4j.execute_query(query, params)
        
        # Count relationships created (2 for from/to + 1 for token if applicable)
        return 2 + (1 if token_address else 0)

    async def _process_contract_call(self, wallet_address: str, activity: Dict[str, Any]) -> Tuple[bool, int]:
        """
        Processes a contract call activity.
        
        Args:
            wallet_address: The wallet address being processed.
            activity: The activity data containing contract call information.
            
        Returns:
            Tuple of (contract_created, relationships_created)
        """
        tx_hash = activity.get("transaction_hash")
        contract_address = activity.get("to_address")
        
        if not (tx_hash and contract_address):
            logger.warning(f"Skipping contract call with missing data: {activity}")
            return False, 0
        
        # Create contract node
        contract_created = await self._create_or_update_contract(
            contract_address,
            activity.get("chain_id"),
            activity.get("function", {}).get("name"),
            activity.get("function", {}).get("signature")
        )
        
        # Create relationships
        query = """
        MATCH (tx:Transaction {hash: $tx_hash})
        MATCH (wallet:CryptoWallet {address: $wallet_address})
        MATCH (contract:Contract {address: $contract_address, chain_id: $chain_id})
        
        // Create INITIATED relationship
        MERGE (wallet)-[i:INITIATED]->(tx)
        ON CREATE SET i.created_at = datetime()
        
        // Create CALLS relationship
        MERGE (tx)-[c:CALLS]->(contract)
        ON CREATE SET 
            c.created_at = datetime(),
            c.function_name = $function_name,
            c.function_signature = $function_signature,
            c.function_params = $function_params
        
        // Create direct INTERACTED_WITH relationship
        MERGE (wallet)-[iw:INTERACTED_WITH]->(contract)
        ON CREATE SET 
            iw.created_at = datetime(),
            iw.last_interaction = datetime()
        ON MATCH SET 
            iw.last_interaction = datetime(),
            iw.interaction_count = COALESCE(iw.interaction_count, 0) + 1
        """
        
        # Convert function parameters to JSON string if available
        function_params = None
        if activity.get("function", {}).get("parameters"):
            try:
                function_params = json.dumps(activity["function"]["parameters"])
            except:
                function_params = str(activity["function"]["parameters"])
        
        params = {
            "tx_hash": tx_hash.lower(),
            "wallet_address": wallet_address.lower(),
            "contract_address": contract_address.lower(),
            "chain_id": activity.get("chain_id"),
            "function_name": activity.get("function", {}).get("name"),
            "function_signature": activity.get("function", {}).get("signature"),
            "function_params": function_params
        }
        
        await self.neo4j.execute_query(query, params)
        
        # 3 relationships: INITIATED, CALLS, INTERACTED_WITH
        return contract_created, 3

    async def _process_token_event(self, wallet_address: str, activity: Dict[str, Any]) -> int:
        """
        Processes token mint/burn events.
        
        Args:
            wallet_address: The wallet address being processed.
            activity: The activity data containing mint/burn information.
            
        Returns:
            Number of relationships created.
        """
        tx_hash = activity.get("transaction_hash")
        token_address = activity.get("token_address")
        activity_type = activity.get("type")
        
        if not (tx_hash and token_address):
            logger.warning(f"Skipping {activity_type} with missing data: {activity}")
            return 0
        
        # Create token node
        await self._create_or_update_token(
            token_address,
            activity.get("chain_id"),
            activity.get("token_metadata")
        )
        
        # Create relationships
        query = """
        MATCH (tx:Transaction {hash: $tx_hash})
        MATCH (wallet:CryptoWallet {address: $wallet_address})
        MATCH (token:Token {address: $token_address, chain_id: $chain_id})
        
        // Create INITIATED relationship
        MERGE (wallet)-[i:INITIATED]->(tx)
        ON CREATE SET i.created_at = datetime()
        """
        
        # Add relationship based on activity type
        if activity_type == "mint":
            query += """
            // Create MINTS relationship
            MERGE (tx)-[m:MINTS]->(token)
            ON CREATE SET 
                m.created_at = datetime(),
                m.amount = $amount,
                m.value_usd = $value_usd
            
            // Create direct MINTED relationship
            MERGE (wallet)-[mt:MINTED]->(token)
            ON CREATE SET 
                mt.created_at = datetime(),
                mt.last_mint = datetime()
            ON MATCH SET 
                mt.last_mint = datetime(),
                mt.mint_count = COALESCE(mt.mint_count, 0) + 1
            """
        else:  # burn
            query += """
            // Create BURNS relationship
            MERGE (tx)-[b:BURNS]->(token)
            ON CREATE SET 
                b.created_at = datetime(),
                b.amount = $amount,
                b.value_usd = $value_usd
            
            // Create direct BURNED relationship
            MERGE (wallet)-[bt:BURNED]->(token)
            ON CREATE SET 
                bt.created_at = datetime(),
                bt.last_burn = datetime()
            ON MATCH SET 
                bt.last_burn = datetime(),
                bt.burn_count = COALESCE(bt.burn_count, 0) + 1
            """
        
        params = {
            "tx_hash": tx_hash.lower(),
            "wallet_address": wallet_address.lower(),
            "token_address": token_address.lower(),
            "chain_id": activity.get("chain_id"),
            "amount": activity.get("amount"),
            "value_usd": activity.get("value_usd")
        }
        
        await self.neo4j.execute_query(query, params)
        
        # 3 relationships: INITIATED, MINTS/BURNS, MINTED/BURNED
        return 3

    async def _process_token_action(self, wallet_address: str, activity: Dict[str, Any]) -> int:
        """
        Processes token swap/approve actions.
        
        Args:
            wallet_address: The wallet address being processed.
            activity: The activity data containing swap/approve information.
            
        Returns:
            Number of relationships created.
        """
        tx_hash = activity.get("transaction_hash")
        token_address = activity.get("token_address")
        activity_type = activity.get("type")
        to_address = activity.get("to_address")
        
        if not tx_hash:
            logger.warning(f"Skipping {activity_type} with missing data: {activity}")
            return 0
        
        # Create token node if available
        if token_address:
            await self._create_or_update_token(
                token_address,
                activity.get("chain_id"),
                activity.get("token_metadata")
            )
        
        # Create relationships
        query = """
        MATCH (tx:Transaction {hash: $tx_hash})
        MATCH (wallet:CryptoWallet {address: $wallet_address})
        
        // Create INITIATED relationship
        MERGE (wallet)-[i:INITIATED]->(tx)
        ON CREATE SET i.created_at = datetime()
        """
        
        params = {
            "tx_hash": tx_hash.lower(),
            "wallet_address": wallet_address.lower(),
            "chain_id": activity.get("chain_id"),
            "amount": activity.get("amount"),
            "value_usd": activity.get("value_usd")
        }
        
        # Add token relationship if token is available
        relationship_count = 1  # INITIATED
        
        if token_address:
            query += """
            WITH tx, wallet
            MATCH (token:Token {address: $token_address, chain_id: $chain_id})
            """
            params["token_address"] = token_address.lower()
            
            # Add specific relationship based on activity type
            if activity_type == "swap":
                query += """
                MERGE (tx)-[s:SWAPS]->(token)
                ON CREATE SET 
                    s.created_at = datetime(),
                    s.amount = $amount,
                    s.value_usd = $value_usd
                """
                relationship_count += 1
            elif activity_type == "approve":
                query += """
                MERGE (tx)-[a:APPROVES]->(token)
                ON CREATE SET 
                    a.created_at = datetime(),
                    a.amount = $amount,
                    a.value_usd = $value_usd
                """
                relationship_count += 1
        
        # Add spender relationship for approvals
        if activity_type == "approve" and to_address:
            query += """
            WITH tx, wallet
            MERGE (spender:CryptoWallet {address: $to_address})
            ON CREATE SET 
                spender.created_at = datetime(),
                spender.updated_at = datetime()
            ON MATCH SET 
                spender.updated_at = datetime()
                
            MERGE (wallet)-[auth:AUTHORIZED]->(spender)
            ON CREATE SET 
                auth.created_at = datetime(),
                auth.last_approval = datetime()
            ON MATCH SET 
                auth.last_approval = datetime(),
                auth.approval_count = COALESCE(auth.approval_count, 0) + 1
            """
            params["to_address"] = to_address.lower()
            relationship_count += 1
        
        await self.neo4j.execute_query(query, params)
        return relationship_count

    async def _create_or_update_token(
        self, 
        token_address: str, 
        chain_id: int, 
        token_metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Creates or updates a Token node in Neo4j.
        
        Args:
            token_address: The token contract address.
            chain_id: The blockchain chain ID.
            token_metadata: Optional metadata for the token.
            
        Returns:
            Boolean indicating if a new token was created.
        """
        if not token_address or not chain_id:
            return False
        
        # Handle native tokens specially
        is_native = token_address == "native"
        if is_native:
            token_address = f"native-{chain_id}"
        
        query = """
        MERGE (t:Token {address: $address, chain_id: $chain_id})
        ON CREATE SET 
            t.created_at = datetime(),
            t.updated_at = datetime(),
            t.symbol = $symbol,
            t.name = $name,
            t.decimals = $decimals,
            t.logo = $logo,
            t.is_native = $is_native
        ON MATCH SET 
            t.updated_at = datetime(),
            t.symbol = COALESCE($symbol, t.symbol),
            t.name = COALESCE($name, t.name),
            t.decimals = COALESCE($decimals, t.decimals),
            t.logo = COALESCE($logo, t.logo)
        RETURN t.created_at = datetime() as created_new
        """
        
        params = {
            "address": token_address.lower(),
            "chain_id": chain_id,
            "symbol": token_metadata.get("symbol") if token_metadata else None,
            "name": token_metadata.get("name") if token_metadata else None,
            "decimals": token_metadata.get("decimals") if token_metadata else None,
            "logo": token_metadata.get("logo") if token_metadata else None,
            "is_native": is_native
        }
        
        result = await self.neo4j.execute_query(query, params)
        return len(result) > 0 and result[0].get("created_new", False)

    async def _create_or_update_contract(
        self, 
        contract_address: str, 
        chain_id: int, 
        function_name: Optional[str] = None,
        function_signature: Optional[str] = None
    ) -> bool:
        """
        Creates or updates a Contract node in Neo4j.
        
        Args:
            contract_address: The contract address.
            chain_id: The blockchain chain ID.
            function_name: Optional function name for context.
            function_signature: Optional function signature.
            
        Returns:
            Boolean indicating if a new contract was created.
        """
        if not contract_address or not chain_id:
            return False
        
        query = """
        MERGE (c:Contract {address: $address, chain_id: $chain_id})
        ON CREATE SET 
            c.created_at = datetime(),
            c.updated_at = datetime(),
            c.chain_id = $chain_id,
            c.functions = CASE WHEN $function_name IS NOT NULL THEN [$function_name] ELSE [] END
        ON MATCH SET 
            c.updated_at = datetime(),
            c.functions = CASE 
                WHEN $function_name IS NOT NULL AND NOT $function_name IN c.functions 
                THEN c.functions + $function_name 
                ELSE c.functions 
            END
        RETURN c.created_at = datetime() as created_new
        """
        
        params = {
            "address": contract_address.lower(),
            "chain_id": chain_id,
            "function_name": function_name
        }
        
        result = await self.neo4j.execute_query(query, params)
        return len(result) > 0 and result[0].get("created_new", False)

    async def batch_ingest_wallets(self, wallet_addresses: List[str], **kwargs) -> Dict[str, Any]:
        """
        Batch ingests multiple wallet addresses.
        
        Args:
            wallet_addresses: List of wallet addresses to ingest.
            **kwargs: Additional arguments to pass to the run method.
            
        Returns:
            Dictionary with aggregated ingestion statistics.
        """
        if not wallet_addresses:
            return {"wallets_processed": 0, "errors": []}
        
        results = {
            "wallets_processed": 0,
            "balances_ingested": 0,
            "activities_ingested": 0,
            "tokens_created": 0,
            "transactions_created": 0,
            "contracts_created": 0,
            "relationships_created": 0,
            "errors": []
        }
        
        # Create schema only once
        if kwargs.get("create_schema", False):
            try:
                await self._ensure_schema()
                kwargs["create_schema"] = False  # Don't recreate for each wallet
                results["schema_created"] = True
            except Exception as e:
                error_msg = f"Error creating schema: {str(e)}"
                logger.error(error_msg)
                results["errors"].append(error_msg)
        
        # Process each wallet
        for wallet_address in wallet_addresses:
            try:
                wallet_result = await self.run(wallet_address=wallet_address, **kwargs)
                
                # Aggregate statistics
                results["wallets_processed"] += 1
                results["balances_ingested"] += wallet_result.get("balances_ingested", 0)
                results["activities_ingested"] += wallet_result.get("activities_ingested", 0)
                results["tokens_created"] += wallet_result.get("tokens_created", 0)
                results["transactions_created"] += wallet_result.get("transactions_created", 0)
                results["contracts_created"] += wallet_result.get("contracts_created", 0)
                results["relationships_created"] += wallet_result.get("relationships_created", 0)
                
                # Aggregate errors
                if wallet_result.get("errors"):
                    results["errors"].extend(wallet_result["errors"])
                    
            except Exception as e:
                error_msg = f"Error processing wallet {wallet_address}: {str(e)}"
                logger.exception(error_msg)
                results["errors"].append(error_msg)
        
        return results
