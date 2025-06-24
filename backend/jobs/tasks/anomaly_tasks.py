"""
Celery Tasks for Anomaly Detection and Fraud Monitoring

This module defines asynchronous Celery tasks for the Anomaly Detection Service.
It includes tasks for:
- Continuous monitoring (periodic scanning for anomalies)
- On-demand single entity analysis
- Bulk analysis for multiple entities
- Alert generation and notification
- Management of detection strategies

These tasks integrate with the core AnomalyDetectionService and ensure proper
error handling, progress tracking, and observability.
"""

import logging
import asyncio
import uuid
from typing import Any, Dict, List, Optional, Tuple

from backend.jobs.celery_app import celery_app
from backend.core.telemetry import trace
from backend.core.anomaly_detection import (
    AnomalyDetectionService,
    AnomalyDetectionResult,
    AnomalyAlert,
    AnomalySeverity,
    DetectionStrategy,
    DataEntityType,
    AlertStatus,
)
from backend.core.events import publish_event, EventPriority

logger = logging.getLogger(__name__)

# Initialize the AnomalyDetectionService (it handles its own Redis/Neo4j clients)
anomaly_service = AnomalyDetectionService()


@celery_app.task(bind=True, name="anomaly_tasks.detect_single_entity_anomaly")
@trace(name="celery.task.detect_single_entity_anomaly")
async def detect_single_entity_anomaly_task(
    self,
    entity_id: str,
    entity_type: str,  # Passed as string, convert to Enum
    strategies: Optional[List[str]] = None,
    create_evidence: bool = True,
) -> List[Dict[str, Any]]:
    """
    Performs on-demand anomaly detection for a single entity.

    Args:
        entity_id: The ID of the entity to analyze (e.g., wallet address, transaction hash).
        entity_type: The type of the entity (e.g., 'address', 'transaction').
        strategies: Optional list of strategy IDs to use. If None, all applicable
                    strategies will be used.
        create_evidence: Whether to create evidence bundles for detected anomalies.

    Returns:
        A list of dictionaries, each representing an AnomalyDetectionResult.
    """
    logger.info(f"Starting anomaly detection for single entity: {entity_type}:{entity_id}")
    self.update_state(state='PROGRESS', meta={'step': 'Detecting anomalies', 'progress': 10})

    try:
        # Convert entity_type string to Enum
        entity_type_enum = DataEntityType(entity_type)

        results: List[AnomalyDetectionResult] = await anomaly_service.detect_anomalies(
            entity_id=entity_id,
            entity_type=entity_type_enum,
            strategies=strategies,
            create_evidence=create_evidence,
        )

        self.update_state(state='PROGRESS', meta={'step': 'Processing results', 'progress': 80})

        processed_results = [r.to_dict() for r in results]
        logger.info(f"Completed anomaly detection for {entity_type}:{entity_id}. Found {len(results)} anomalies.")
        return {"status": "SUCCESS", "results": processed_results}

    except ValueError as ve:
        logger.error(f"Invalid entity type '{entity_type}': {ve}", exc_info=True)
        self.update_state(state='FAILURE', meta={'error': f"Invalid entity type: {entity_type}"})
        raise
    except Exception as e:
        logger.error(f"Anomaly detection task failed for {entity_type}:{entity_id}: {e}", exc_info=True)
        self.update_state(state='FAILURE', meta={'error': str(e)})
        raise


@celery_app.task(bind=True, name="anomaly_tasks.detect_bulk_anomalies")
@trace(name="celery.task.detect_bulk_anomalies")
async def detect_bulk_anomalies_task(
    self,
    entity_ids_and_types: List[Tuple[str, str]],  # List of (entity_id, entity_type_str)
    strategies: Optional[List[str]] = None,
    create_evidence: bool = True,
) -> List[Dict[str, Any]]:
    """
    Performs anomaly detection for a list of entities in bulk.

    Args:
        entity_ids_and_types: A list of tuples, where each tuple is (entity_id, entity_type_string).
        strategies: Optional list of strategy IDs to use.
        create_evidence: Whether to create evidence bundles for detected anomalies.

    Returns:
        A list of dictionaries, each representing an AnomalyDetectionResult.
    """
    logger.info(f"Starting bulk anomaly detection for {len(entity_ids_and_types)} entities.")
    all_results = []
    total_entities = len(entity_ids_and_types)

    for i, (entity_id, entity_type_str) in enumerate(entity_ids_and_types):
        self.update_state(
            state='PROGRESS',
            meta={
                'step': f'Processing entity {i+1}/{total_entities}',
                'progress': int((i / total_entities) * 100)
            }
        )
        try:
            entity_type_enum = DataEntityType(entity_type_str)
            results: List[AnomalyDetectionResult] = await anomaly_service.detect_anomalies(
                entity_id=entity_id,
                entity_type=entity_type_enum,
                strategies=strategies,
                create_evidence=create_evidence,
            )
            all_results.extend([r.to_dict() for r in results])
            logger.debug(f"Processed {entity_type_str}:{entity_id}. Found {len(results)} anomalies.")
        except Exception as e:
            logger.error(f"Failed to detect anomalies for {entity_type_str}:{entity_id}: {e}", exc_info=True)
            # Continue processing other entities even if one fails

    self.update_state(state='PROGRESS', meta={'step': 'Bulk processing complete', 'progress': 100})
    logger.info(f"Completed bulk anomaly detection. Total anomalies found: {len(all_results)}.")
    return {"status": "SUCCESS", "results": all_results}


@celery_app.task(bind=True, name="anomaly_tasks.continuous_anomaly_monitoring")
@trace(name="celery.task.continuous_anomaly_monitoring")
async def continuous_anomaly_monitoring_task(
    self,
    scan_interval_minutes: int = 60,
    entity_types_to_scan: Optional[List[str]] = None,
    limit_per_type: int = 100,
) -> Dict[str, Any]:
    """
    Periodically scans for anomalies across a set of entities.
    This task is intended to be scheduled (e.g., via Celery Beat).

    Args:
        scan_interval_minutes: How often the scan should theoretically run (for logging).
        entity_types_to_scan: List of entity types to include in the scan (e.g., ['address', 'transaction']).
                              If None, a default set will be used.
        limit_per_type: Maximum number of entities to scan per type in one run.

    Returns:
        A summary of the monitoring run.
    """
    logger.info(f"Starting continuous anomaly monitoring run. Interval: {scan_interval_minutes} min.")
    self.update_state(state='PROGRESS', meta={'step': 'Initializing scan', 'progress': 0})

    if not entity_types_to_scan:
        entity_types_to_scan = [DataEntityType.ADDRESS.value, DataEntityType.TRANSACTION.value]
        logger.info(f"No specific entity types provided, scanning default: {entity_types_to_scan}")

    total_anomalies_found = 0
    scan_summary = {}

    for i, entity_type_str in enumerate(entity_types_to_scan):
        self.update_state(
            state='PROGRESS',
            meta={
                'step': f'Scanning {entity_type_str} entities',
                'progress': int((i / len(entity_types_to_scan)) * 100)
            }
        )
        try:
            entity_type_enum = DataEntityType(entity_type_str)
            # In a real system, you'd query your database for recent/relevant entities
            # For this example, we'll simulate fetching some entities
            # Example: Fetch recent addresses from Neo4j or Redis
            # For now, let's assume we have a way to get a list of entity IDs
            
            # Placeholder: Fetch some dummy entity IDs for demonstration
            # In a real scenario, this would involve querying Neo4j for active addresses/transactions
            # or pulling from a stream.
            dummy_entity_ids = [f"0x{uuid.uuid4().hex}" for _ in range(limit_per_type)]
            
            # Create a list of (entity_id, entity_type_str) tuples for bulk processing
            entities_for_scan = [(eid, entity_type_str) for eid in dummy_entity_ids]

            if not entities_for_scan:
                logger.info(f"No {entity_type_str} entities found for scanning.")
                scan_summary[entity_type_str] = {"scanned": 0, "anomalies_found": 0}
                continue

            logger.info(f"Scanning {len(entities_for_scan)} {entity_type_str} entities...")
            bulk_results = await detect_bulk_anomalies_task(
                entity_ids_and_types=entities_for_scan,
                create_evidence=True,
            )
            
            num_anomalies_for_type = len(bulk_results.get("results", []))
            total_anomalies_found += num_anomalies_for_type
            scan_summary[entity_type_str] = {
                "scanned": len(entities_for_scan),
                "anomalies_found": num_anomalies_for_type
            }
            logger.info(f"Scan for {entity_type_str} completed. Found {num_anomalies_for_type} anomalies.")

        except Exception as e:
            logger.error(f"Error during continuous monitoring for {entity_type_str}: {e}", exc_info=True)
            scan_summary[entity_type_str] = {"scanned": 0, "anomalies_found": 0, "error": str(e)}
            # Continue to next entity type

    self.update_state(state='PROGRESS', meta={'step': 'Monitoring complete', 'progress': 100})
    logger.info(f"Continuous anomaly monitoring run finished. Total anomalies found: {total_anomalies_found}.")
    return {"status": "SUCCESS", "total_anomalies_found": total_anomalies_found, "summary": scan_summary}


@celery_app.task(bind=True, name="anomaly_tasks.generate_anomaly_alert")
@trace(name="celery.task.generate_anomaly_alert")
async def generate_anomaly_alert_task(
    self,
    anomaly_id: str,
    title: Optional[str] = None,
    description: Optional[str] = None,
    severity: Optional[str] = None,  # Passed as string, convert to Enum
    tags: Optional[List[str]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Generates an alert for a detected anomaly.

    Args:
        anomaly_id: The ID of the AnomalyDetectionResult to create an alert for.
        title: Optional title for the alert.
        description: Optional description for the alert.
        severity: Optional severity for the alert (e.g., 'high', 'medium').
        tags: Optional list of tags for the alert.

    Returns:
        A dictionary representing the created AnomalyAlert, or None if failed.
    """
    logger.info(f"Generating alert for anomaly ID: {anomaly_id}")
    self.update_state(state='PROGRESS', meta={'step': 'Creating alert', 'progress': 20})

    try:
        severity_enum = AnomalySeverity(severity) if severity else None
        alert: Optional[AnomalyAlert] = await anomaly_service.create_alert(
            anomaly_id=anomaly_id,
            title=title,
            description=description,
            severity=severity_enum,
            tags=tags,
        )

        if alert:
            self.update_state(state='PROGRESS', meta={'step': 'Alert created', 'progress': 100})
            logger.info(f"Alert {alert.id} created for anomaly {anomaly_id}.")
            return {"status": "SUCCESS", "alert": alert.to_dict()}
        else:
            self.update_state(state='FAILURE', meta={'error': 'Failed to create alert'})
            logger.error(f"Failed to create alert for anomaly {anomaly_id}.")
            return {"status": "FAILURE", "error": "Failed to create alert"}

    except ValueError as ve:
        logger.error(f"Invalid severity '{severity}': {ve}", exc_info=True)
        self.update_state(state='FAILURE', meta={'error': f"Invalid severity: {severity}"})
        raise
    except Exception as e:
        logger.error(f"Failed to generate alert for anomaly {anomaly_id}: {e}", exc_info=True)
        self.update_state(state='FAILURE', meta={'error': str(e)})
        raise


@celery_app.task(bind=True, name="anomaly_tasks.update_alert_status")
@trace(name="celery.task.update_alert_status")
async def update_alert_status_task(
    self,
    alert_id: str,
    status: str,  # Passed as string, convert to Enum
    assigned_to: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Updates the status of an existing anomaly alert.

    Args:
        alert_id: The ID of the alert to update.
        status: The new status for the alert (e.g., 'investigating', 'resolved').
        assigned_to: Optional user to assign the alert to.

    Returns:
        A dictionary indicating success or failure.
    """
    logger.info(f"Updating status for alert {alert_id} to {status}")
    self.update_state(state='PROGRESS', meta={'step': 'Updating alert status', 'progress': 20})

    try:
        status_enum = AlertStatus(status)
        success = await anomaly_service.update_alert_status(
            alert_id=alert_id,
            status=status_enum,
            assigned_to=assigned_to,
        )

        if success:
            self.update_state(state='PROGRESS', meta={'step': 'Alert status updated', 'progress': 100})
            logger.info(f"Alert {alert_id} status updated to {status}.")
            return {"status": "SUCCESS", "message": f"Alert {alert_id} status updated to {status}"}
        else:
            self.update_state(state='FAILURE', meta={'error': 'Failed to update alert status'})
            logger.error(f"Failed to update alert {alert_id} status to {status}.")
            return {"status": "FAILURE", "error": "Failed to update alert status"}

    except ValueError as ve:
        logger.error(f"Invalid alert status '{status}': {ve}", exc_info=True)
        self.update_state(state='FAILURE', meta={'error': f"Invalid alert status: {status}"})
        raise
    except Exception as e:
        logger.error(f"Failed to update alert {alert_id} status: {e}", exc_info=True)
        self.update_state(state='FAILURE', meta={'error': str(e)})
        raise


@celery_app.task(bind=True, name="anomaly_tasks.add_detection_strategy")
@trace(name="celery.task.add_detection_strategy")
async def add_detection_strategy_task(
    self,
    strategy_data: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Adds a new anomaly detection strategy.

    Args:
        strategy_data: Dictionary containing the strategy definition.

    Returns:
        A dictionary indicating success or failure.
    """
    logger.info(f"Adding new detection strategy: {strategy_data.get('name', 'Unnamed')}")
    self.update_state(state='PROGRESS', meta={'step': 'Creating strategy', 'progress': 20})

    try:
        # Create DetectionStrategy object from data
        strategy = DetectionStrategy(**strategy_data)
        
        # Add strategy to service
        strategies = await anomaly_service.get_detection_strategies()
        strategies[strategy.id] = strategy
        
        # Store updated strategies
        # In a real implementation, this would call a method on the service
        # For now, we'll assume the service handles persistence internally
        
        self.update_state(state='PROGRESS', meta={'step': 'Strategy added', 'progress': 100})
        logger.info(f"Detection strategy {strategy.id} added: {strategy.name}")
        
        return {
            "status": "SUCCESS",
            "strategy_id": strategy.id,
            "message": f"Strategy '{strategy.name}' added successfully"
        }

    except Exception as e:
        logger.error(f"Failed to add detection strategy: {e}", exc_info=True)
        self.update_state(state='FAILURE', meta={'error': str(e)})
        raise


@celery_app.task(bind=True, name="anomaly_tasks.update_detection_strategy")
@trace(name="celery.task.update_detection_strategy")
async def update_detection_strategy_task(
    self,
    strategy_id: str,
    strategy_updates: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Updates an existing anomaly detection strategy.

    Args:
        strategy_id: ID of the strategy to update.
        strategy_updates: Dictionary containing the fields to update.

    Returns:
        A dictionary indicating success or failure.
    """
    logger.info(f"Updating detection strategy: {strategy_id}")
    self.update_state(state='PROGRESS', meta={'step': 'Updating strategy', 'progress': 20})

    try:
        # Get existing strategy
        strategy = await anomaly_service.get_detection_strategy(strategy_id)
        if not strategy:
            self.update_state(state='FAILURE', meta={'error': f"Strategy {strategy_id} not found"})
            return {"status": "FAILURE", "error": f"Strategy {strategy_id} not found"}
        
        # Update strategy fields
        for key, value in strategy_updates.items():
            if hasattr(strategy, key):
                setattr(strategy, key, value)
        
        # Update timestamp
        strategy.updated_at = datetime.now()
        
        # Update in service
        strategies = await anomaly_service.get_detection_strategies()
        strategies[strategy.id] = strategy
        
        # Store updated strategies
        # In a real implementation, this would call a method on the service
        
        self.update_state(state='PROGRESS', meta={'step': 'Strategy updated', 'progress': 100})
        logger.info(f"Detection strategy {strategy_id} updated: {strategy.name}")
        
        return {
            "status": "SUCCESS",
            "strategy_id": strategy.id,
            "message": f"Strategy '{strategy.name}' updated successfully"
        }

    except Exception as e:
        logger.error(f"Failed to update detection strategy {strategy_id}: {e}", exc_info=True)
        self.update_state(state='FAILURE', meta={'error': str(e)})
        raise


@celery_app.task(bind=True, name="anomaly_tasks.delete_detection_strategy")
@trace(name="celery.task.delete_detection_strategy")
async def delete_detection_strategy_task(
    self,
    strategy_id: str,
) -> Dict[str, Any]:
    """
    Deletes an anomaly detection strategy.

    Args:
        strategy_id: ID of the strategy to delete.

    Returns:
        A dictionary indicating success or failure.
    """
    logger.info(f"Deleting detection strategy: {strategy_id}")
    self.update_state(state='PROGRESS', meta={'step': 'Deleting strategy', 'progress': 20})

    try:
        # Get existing strategy
        strategy = await anomaly_service.get_detection_strategy(strategy_id)
        if not strategy:
            self.update_state(state='FAILURE', meta={'error': f"Strategy {strategy_id} not found"})
            return {"status": "FAILURE", "error": f"Strategy {strategy_id} not found"}
        
        # Delete from service
        strategies = await anomaly_service.get_detection_strategies()
        if strategy_id in strategies:
            del strategies[strategy_id]
        
        # Store updated strategies
        # In a real implementation, this would call a method on the service
        
        self.update_state(state='PROGRESS', meta={'step': 'Strategy deleted', 'progress': 100})
        logger.info(f"Detection strategy {strategy_id} deleted: {strategy.name}")
        
        return {
            "status": "SUCCESS",
            "strategy_id": strategy_id,
            "message": f"Strategy '{strategy.name}' deleted successfully"
        }

    except Exception as e:
        logger.error(f"Failed to delete detection strategy {strategy_id}: {e}", exc_info=True)
        self.update_state(state='FAILURE', meta={'error': str(e)})
        raise


@celery_app.task(bind=True, name="anomaly_tasks.notify_anomaly_alert")
@trace(name="celery.task.notify_anomaly_alert")
async def notify_anomaly_alert_task(
    self,
    alert_id: str,
    notification_channels: List[str],
    custom_message: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Sends notifications for an anomaly alert through specified channels.

    Args:
        alert_id: ID of the alert to notify about.
        notification_channels: List of channels to notify (e.g., 'email', 'slack', 'webhook').
        custom_message: Optional custom message to include in the notification.

    Returns:
        A dictionary indicating success or failure for each channel.
    """
    logger.info(f"Sending notifications for alert {alert_id} via {notification_channels}")
    self.update_state(state='PROGRESS', meta={'step': 'Preparing notifications', 'progress': 10})

    try:
        # Get alert
        alert = await anomaly_service.get_alert(alert_id)
        if not alert:
            self.update_state(state='FAILURE', meta={'error': f"Alert {alert_id} not found"})
            return {"status": "FAILURE", "error": f"Alert {alert_id} not found"}
        
        # Get anomaly details
        anomaly = await anomaly_service.get_detection_result(alert.anomaly_id)
        
        # Prepare notification content
        notification_content = {
            "alert_id": alert.id,
            "title": alert.title,
            "description": alert.description,
            "severity": alert.severity.value,
            "entity_type": alert.entity_type.value,
            "entity_id": alert.entity_id,
            "created_at": alert.created_at.isoformat(),
            "custom_message": custom_message,
            "anomaly_details": anomaly.to_dict() if anomaly else None,
        }
        
        # Send notifications to each channel
        results = {}
        total_channels = len(notification_channels)
        
        for i, channel in enumerate(notification_channels):
            self.update_state(
                state='PROGRESS',
                meta={
                    'step': f'Sending notification via {channel}',
                    'progress': 10 + int(90 * (i / total_channels))
                }
            )
            
            try:
                # In a real implementation, this would call different notification methods
                # For now, we'll just simulate success for all channels
                
                # Example of how this would work:
                # if channel == 'email':
                #     success = await send_email_notification(alert, notification_content)
                # elif channel == 'slack':
                #     success = await send_slack_notification(alert, notification_content)
                # elif channel == 'webhook':
                #     success = await send_webhook_notification(alert, notification_content)
                
                # Simulate notification
                logger.info(f"Simulating {channel} notification for alert {alert_id}")
                
                # Publish event
                publish_event(
                    event_type=f"anomaly.notification.{channel}",
                    data={
                        "alert_id": alert_id,
                        "channel": channel,
                        "content": notification_content,
                    },
                    priority=EventPriority.HIGH if alert.severity in [AnomalySeverity.HIGH, AnomalySeverity.CRITICAL] else EventPriority.NORMAL,
                )
                
                results[channel] = {"status": "SUCCESS"}
                
            except Exception as e:
                logger.error(f"Failed to send {channel} notification for alert {alert_id}: {e}", exc_info=True)
                results[channel] = {"status": "FAILURE", "error": str(e)}
        
        self.update_state(state='PROGRESS', meta={'step': 'Notifications sent', 'progress': 100})
        
        # Determine overall status
        overall_status = "SUCCESS"
        if any(r["status"] == "FAILURE" for r in results.values()):
            overall_status = "PARTIAL_SUCCESS" if any(r["status"] == "SUCCESS" for r in results.values()) else "FAILURE"
        
        return {
            "status": overall_status,
            "alert_id": alert_id,
            "channels": results
        }

    except Exception as e:
        logger.error(f"Failed to send notifications for alert {alert_id}: {e}", exc_info=True)
        self.update_state(state='FAILURE', meta={'error': str(e)})
        raise
