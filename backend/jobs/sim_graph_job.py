"""
Sim Graph Ingestion Job

This module defines a background job for ingesting blockchain data from Sim APIs
into Neo4j. It leverages the SimGraphIngestionTool to fetch wallet balances and
activity and transform them into graph structures.
"""

import logging
from typing import Dict, Any, List, Optional

from backend.integrations.neo4j_client import Neo4jClient
from backend.integrations.sim_client import SimClient
from backend.agents.tools.sim_graph_ingestion_tool import SimGraphIngestionTool
from backend.core.metrics import record_job_execution_time, record_job_status

logger = logging.getLogger(__name__)

async def run_sim_graph_ingestion_job(
    wallet_address: str,
    neo4j_client: Neo4jClient,
    sim_client: SimClient,
    ingest_balances: bool = True,
    ingest_activity: bool = True,
    limit_balances: int = 100,
    limit_activity: int = 50,
    chain_ids: str = "all",
    create_schema: bool = False,
) -> Dict[str, Any]:
    """
    Executes the Sim Graph Ingestion Job for a given wallet address.

    This job fetches blockchain data using Sim APIs and ingests it into Neo4j,
    creating and updating graph entities and relationships.

    Args:
        wallet_address: The blockchain wallet address to ingest data for.
        neo4j_client: An initialized Neo4jClient instance.
        sim_client: An initialized SimClient instance.
        ingest_balances: Whether to ingest token balance data.
        ingest_activity: Whether to ingest transaction activity data.
        limit_balances: Maximum number of balances to fetch.
        limit_activity: Maximum number of activities to fetch.
        chain_ids: Comma-separated list of chain IDs or "all" for all chains.
        create_schema: Whether to create necessary Neo4j schema constraints and indexes.

    Returns:
        A dictionary summarizing the ingestion results.
    """
    job_name = "sim_graph_ingestion"
    status = "failed"
    start_time = None
    try:
        logger.info(f"Starting Sim graph ingestion job for wallet: {wallet_address}")
        start_time = record_job_execution_time(job_name, "start")

        ingestion_tool = SimGraphIngestionTool(
            neo4j_client=neo4j_client,
            sim_client=sim_client
        )

        results = await ingestion_tool.run(
            wallet_address=wallet_address,
            ingest_balances=ingest_balances,
            ingest_activity=ingest_activity,
            limit_balances=limit_balances,
            limit_activity=limit_activity,
            chain_ids=chain_ids,
            create_schema=create_schema,
        )
        status = "completed"
        logger.info(f"Sim graph ingestion job completed for wallet {wallet_address}. Results: {results}")
        return results

    except Exception as e:
        logger.exception(f"Sim graph ingestion job failed for wallet {wallet_address}: {e}")
        return {"status": "error", "message": str(e)}
    finally:
        if start_time is not None:
            record_job_execution_time(job_name, status, start_time)
        record_job_status(job_name, status)

async def batch_sim_graph_ingestion_job(
    wallet_addresses: List[str],
    neo4j_client: Neo4jClient,
    sim_client: SimClient,
    **kwargs
) -> Dict[str, Any]:
    """
    Executes a batch ingestion job for multiple wallet addresses.
    
    Args:
        wallet_addresses: List of wallet addresses to process.
        neo4j_client: An initialized Neo4jClient instance.
        sim_client: An initialized SimClient instance.
        **kwargs: Additional parameters to pass to the individual job runs.
        
    Returns:
        A dictionary summarizing the batch ingestion results.
    """
    job_name = "batch_sim_graph_ingestion"
    status = "failed"
    start_time = None
    
    try:
        logger.info(f"Starting batch Sim graph ingestion job for {len(wallet_addresses)} wallets")
        start_time = record_job_execution_time(job_name, "start")
        
        ingestion_tool = SimGraphIngestionTool(
            neo4j_client=neo4j_client,
            sim_client=sim_client
        )
        
        # Create schema only once for the batch
        if kwargs.get("create_schema", False):
            await ingestion_tool._ensure_schema()
            kwargs["create_schema"] = False
        
        # Process wallets in batch
        results = await ingestion_tool.batch_ingest_wallets(
            wallet_addresses=wallet_addresses,
            **kwargs
        )
        
        status = "completed"
        logger.info(f"Batch Sim graph ingestion job completed. Processed {results.get('wallets_processed', 0)} wallets.")
        return results
        
    except Exception as e:
        logger.exception(f"Batch Sim graph ingestion job failed: {e}")
        return {"status": "error", "message": str(e), "wallets_processed": 0}
    finally:
        if start_time is not None:
            record_job_execution_time(job_name, status, start_time)
        record_job_status(job_name, status)
