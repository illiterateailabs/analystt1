"""
Celery Tasks for Advanced Analysis

This module contains Celery tasks for computationally intensive analysis, such as
machine learning model training, fraud detection inference, and vision analysis.
By offloading these to background workers, the API remains responsive.
"""

import logging
import random
import time
import base64
import io
from typing import Dict, Any, List

from backend.jobs.celery_app import celery_app
from backend.core.telemetry import trace
from backend.integrations.gemini_client import GeminiClient
from backend.integrations.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, name="analysis_tasks.train_gnn_model")
@trace(name="celery.task.train_gnn_model")
def train_gnn_model_task(self, graph_query: str) -> Dict[str, Any]:
    """
    A long-running task to simulate training a Graph Neural Network (GNN).

    Args:
        graph_query: A Cypher query to fetch the training data from Neo4j.

    Returns:
        A dictionary with the trained model's ID and performance metrics.
    """
    logger.info("Starting GNN model training task.")
    total_epochs = 20
    try:
        neo4j_client = Neo4jClient()
        self.update_state(state='PROGRESS', meta={'step': 'Fetching training data', 'progress': 5})
        
        # Simulate fetching a large dataset from Neo4j
        training_data = neo4j_client.execute_query(graph_query)
        if not training_data:
            raise ValueError("No training data returned from the graph query.")
        
        logger.info(f"Fetched {len(training_data)} records for GNN training.")
        time.sleep(5)  # Simulate data preprocessing

        # Simulate the training loop
        for epoch in range(1, total_epochs + 1):
            self.update_state(
                state='PROGRESS',
                meta={
                    'step': 'Training',
                    'epoch': epoch,
                    'total_epochs': total_epochs,
                    'progress': int(10 + (epoch / total_epochs) * 80)
                }
            )
            # Simulate training work for one epoch
            time.sleep(random.uniform(2, 5))
            logger.debug(f"GNN training epoch {epoch}/{total_epochs} complete.")

        self.update_state(state='PROGRESS', meta={'step': 'Evaluating model', 'progress': 95})
        time.sleep(3) # Simulate final evaluation

        model_id = f"gnn_model_{int(time.time())}"
        metrics = {
            "accuracy": round(random.uniform(0.85, 0.98), 4),
            "precision": round(random.uniform(0.88, 0.99), 4),
            "recall": round(random.uniform(0.82, 0.95), 4),
            "f1_score": round(random.uniform(0.85, 0.97), 4),
        }
        
        logger.info(f"GNN model training complete. Model ID: {model_id}, Metrics: {metrics}")
        return {"status": "SUCCESS", "model_id": model_id, "metrics": metrics}

    except Exception as e:
        logger.error(f"GNN model training task failed: {e}", exc_info=True)
        self.update_state(state='FAILURE', meta={'error': str(e)})
        raise


@celery_app.task(bind=True, name="analysis_tasks.run_gnn_fraud_detection")
@trace(name="celery.task.run_gnn_fraud_detection")
def run_gnn_fraud_detection_task(self, center_node_id: str, model_id: str) -> Dict[str, Any]:
    """
    Runs fraud detection on a subgraph using a pre-trained GNN model.

    Args:
        center_node_id: The ID of the node at the center of the subgraph to analyze.
        model_id: The ID of the GNN model to use for inference.

    Returns:
        A dictionary containing the list of detected fraudulent nodes.
    """
    logger.info(f"Running GNN fraud detection on subgraph for node {center_node_id} using model {model_id}.")
    self.update_state(state='PROGRESS', meta={'step': 'Loading model and data'})
    
    try:
        neo4j_client = Neo4jClient()
        # Simulate loading the GNN model
        time.sleep(2)

        # Fetch the subgraph to analyze
        subgraph_query = f"""
        MATCH (center {{id: '{center_node_id}'}})
        CALL apoc.path.subgraphNodes(center, {{maxLevel: 2}}) YIELD node
        RETURN node.id as id, labels(node) as labels
        LIMIT 100
        """
        subgraph_nodes = neo4j_client.execute_query(subgraph_query)
        
        if not subgraph_nodes:
            return {"status": "SUCCESS", "detected_nodes": [], "message": "Subgraph is empty."}

        self.update_state(state='PROGRESS', meta={'step': 'Running inference', 'node_count': len(subgraph_nodes)})
        time.sleep(len(subgraph_nodes) * 0.1) # Simulate inference time

        # Simulate detection results
        detected_nodes = []
        for node in subgraph_nodes:
            if random.random() < 0.1: # 10% chance of being detected as fraudulent
                detected_nodes.append({
                    "node_id": node['id'],
                    "labels": node['labels'],
                    "confidence": round(random.uniform(0.7, 0.99), 3)
                })

        logger.info(f"Fraud detection complete. Found {len(detected_nodes)} suspicious nodes.")
        return {"status": "SUCCESS", "detected_nodes": detected_nodes}

    except Exception as e:
        logger.error(f"GNN fraud detection task failed for node {center_node_id}: {e}", exc_info=True)
        self.update_state(state='FAILURE', meta={'error': str(e)})
        raise


@celery_app.task(bind=True, name="analysis_tasks.analyze_image")
@trace(name="celery.task.analyze_image")
def analyze_image_task(self, image_data_b64: str, prompt: str) -> Dict[str, Any]:
    """
    Performs image analysis using the Gemini Vision model in the background.

    Args:
        image_data_b64: The base64-encoded string of the image data.
        prompt: The user prompt to guide the image analysis.

    Returns:
        A dictionary containing the analysis results.
    """
    logger.info("Starting background image analysis task.")
    self.update_state(state='PROGRESS', meta={'step': 'Decoding image'})

    try:
        gemini_client = GeminiClient()
        
        # Decode the base64 image data
        image_data = base64.b64decode(image_data_b64)

        self.update_state(state='PROGRESS', meta={'step': 'Analyzing with Gemini Vision'})
        analysis_text = gemini_client.analyze_image(image_data=image_data, prompt=prompt)

        logger.info("Image analysis complete.")
        return {"status": "SUCCESS", "analysis": analysis_text}

    except Exception as e:
        logger.error(f"Image analysis task failed: {e}", exc_info=True)
        self.update_state(state='FAILURE', meta={'error': str(e)})
        raise
