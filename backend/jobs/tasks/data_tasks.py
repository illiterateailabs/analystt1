"""
Celery Tasks for Data Processing and Ingestion

This module contains Celery tasks responsible for data-intensive operations,
such as fetching data from external APIs, loading it into the graph database,
and performing bulk embedding operations.

These tasks are designed to be run asynchronously by Celery workers, offloading
long-running processes from the main API thread to improve responsiveness and
enable horizontal scaling.
"""

import logging
from typing import List, Dict, Any

from backend.core.neo4j_loader import Neo4jLoader
from backend.core.graph_rag import GraphRAG
from backend.integrations.sim_client import SimApiClient
from backend.jobs.celery_app import celery_app
from backend.core.telemetry import trace

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, name="data_tasks.ingest_sim_data_for_address")
@trace(name="celery.task.ingest_sim_data")
def ingest_sim_data_for_address(self, address: str, chain: str):
    """
    Fetches activity for a given address from the SIM API and loads it into Neo4j.

    This task acts as a pipeline for ingesting on-chain data for a specific entity.

    Args:
        address: The blockchain address to process.
        chain: The blockchain network (e.g., "ethereum").

    Returns:
        A summary dictionary of the ingestion results.
    """
    logger.info(f"Starting SIM data ingestion for address: {address} on chain: {chain}")
    self.update_state(state='PROGRESS', meta={'step': 'Initializing clients'})

    try:
        sim_client = SimApiClient()
        neo4j_loader = Neo4jLoader()

        self.update_state(state='PROGRESS', meta={'step': 'Fetching SIM activity'})
        activity_data = sim_client.get_activity(address=address, chain=chain, limit=200)

        if not activity_data or not activity_data.get("data"):
            logger.warning(f"No activity data found for address {address}")
            return {"address": address, "status": "NO_DATA", "processed_transactions": 0}

        self.update_state(state='PROGRESS', meta={'step': 'Loading data into Neo4j'})
        summary = neo4j_loader.load_sim_activity(activity_data["data"])
        logger.info(f"Successfully ingested data for address: {address}. Summary: {summary}")

        return {
            "address": address,
            "status": "SUCCESS",
            **summary
        }
    except Exception as e:
        logger.error(f"Failed to ingest SIM data for address {address}: {e}", exc_info=True)
        self.update_state(state='FAILURE', meta={'error': str(e)})
        # Re-raise the exception to let Celery know the task failed
        raise


@celery_app.task(bind=True, name="data_tasks.batch_embed_graph_nodes")
@trace(name="celery.task.batch_embed_nodes")
def batch_embed_graph_nodes(self, node_ids: List[str]):
    """
    Performs bulk vector embedding for a list of graph nodes.

    This task is essential for offloading the CPU/GPU-intensive process of
    creating vector embeddings from the main application.

    Args:
        node_ids: A list of node unique identifiers to embed.

    Returns:
        A summary of the batch embedding operation.
    """
    total_nodes = len(node_ids)
    logger.info(f"Starting batch embedding for {total_nodes} nodes.")
    self.update_state(state='PROGRESS', meta={'total': total_nodes, 'processed': 0})

    try:
        graph_rag = GraphRAG()
        success_count = 0
        failed_ids = []

        # The GraphRAG service already handles batching, but we can report progress here.
        # For this example, we assume batch_embed_nodes is synchronous within the async task runner.
        # In a fully async worker, we'd await it.
        results = graph_rag.batch_embed_nodes(node_ids=node_ids)

        for node_id, embedding in results.items():
            if embedding:
                success_count += 1
            else:
                failed_ids.append(node_id)
        
        self.update_state(state='SUCCESS', meta={'total': total_nodes, 'processed': success_count})
        summary = {
            "status": "SUCCESS",
            "total_nodes": total_nodes,
            "successfully_embedded": success_count,
            "failed_count": len(failed_ids),
            "failed_ids": failed_ids,
        }
        logger.info(f"Batch embedding complete. {summary}")
        return summary

    except Exception as e:
        logger.error(f"Batch embedding task failed: {e}", exc_info=True)
        self.update_state(state='FAILURE', meta={'error': str(e)})
        raise


@celery_app.task(name="data_tasks.periodic_sim_graph_ingestion")
@trace(name="celery.task.periodic_ingestion")
def periodic_sim_graph_ingestion():
    """
    A periodic task that triggers data ingestion for a predefined set of addresses.

    This task can be scheduled via Celery Beat to keep the graph data fresh.
    """
    # In a real application, these would come from a config file, database, or discovery service.
    addresses_to_monitor = {
        "ethereum": [
            "0xdAC17F958D2ee523a2206206994597C13D831ec7",  # Tether USD (USDT)
            "0x7Fc66500c84A76Ad7e9c93437bFc5Ac33E2DDaE9",  # Aave Token (AAVE)
        ],
        "polygon": [
            "0x0d500B1d8E8eF31E21C99d1Db9A6444d3ADf1270", # Wrapped MATIC (WMATIC)
        ]
    }
    logger.info("Starting periodic SIM graph ingestion for monitored addresses.")

    for chain, addresses in addresses_to_monitor.items():
        for address in addresses:
            # Launch a separate task for each address to run them in parallel.
            ingest_sim_data_for_address.delay(address=address, chain=chain)
            logger.debug(f"Queued ingestion task for {address} on {chain}.")

    return {"status": "SUCCESS", "message": "All periodic ingestion tasks have been queued."}
