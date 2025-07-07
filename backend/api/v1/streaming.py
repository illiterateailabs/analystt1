"""
Streaming API - WebSocket endpoints for real-time transaction monitoring

This module provides WebSocket endpoints for live streaming of blockchain transactions,
including real-time filtering, alerting, and multi-tenant support. It integrates
with Redis Streams for high-throughput data ingestion and processing.

Key features:
- Live transaction stream via WebSocket
- Connection management with tenant isolation
- Real-time filtering based on chain, risk, and other criteria
- Integration with Redis Streams for data consumption
- Proper error handling and cleanup
- Metrics collection for active connections and message flow
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query, Depends, HTTPException, status
from pydantic import BaseModel, Field

from backend.core.logging import get_logger
from backend.core.metrics import REGISTRY, Counter, Gauge
from backend.core.ws_manager import get_ws_manager, ConnectionManager
from backend.streaming import get_stream_client, StreamBackend
from backend.streaming.redis_stream import RedisStreamClient
from backend.tenancy import get_tenant_context, TenantContext
from backend.auth.dependencies import get_current_user, RoleChecker
from backend.auth.rbac import Role

# Configure logger
logger = get_logger(__name__)

# Create router
router = APIRouter(prefix="/ws", tags=["Streaming"])

# Prometheus Metrics
STREAM_CONNECTIONS_TOTAL = Counter(
    "stream_connections_total",
    "Total number of WebSocket connections to the streaming endpoint",
    ["endpoint", "status", "tenant"]
)
STREAM_MESSAGES_SENT_TOTAL = Counter(
    "stream_messages_sent_total",
    "Total number of messages sent through the streaming WebSocket",
    ["endpoint", "tenant", "chain_id"]
)
STREAM_ACTIVE_CONNECTIONS = Gauge(
    "stream_active_connections",
    "Number of active WebSocket connections to the streaming endpoint",
    ["endpoint", "tenant"]
)
STREAM_ERRORS_TOTAL = Counter(
    "stream_errors_total",
    "Total number of errors encountered in streaming operations",
    ["endpoint", "operation", "error_type", "tenant"]
)

# Role checker for analyst access
analyst_role = RoleChecker([Role.ANALYST, Role.ADMIN])

class Transaction(BaseModel):
    """Data model for transaction messages"""
    id: str
    timestamp: str
    from_address: str
    to_address: str
    amount: float
    amount_usd: float
    chain_id: str
    transaction_type: str
    risk_score: Optional[float] = None
    is_high_risk: Optional[bool] = None
    tenant_id: Optional[str] = None
    tags: Optional[List[str]] = None
    method: Optional[str] = None

@router.websocket("/tx_stream")
async def transaction_stream_websocket(
    websocket: WebSocket,
    tenant_id: Optional[str] = Query(None, description="Filter stream by tenant ID"),
    chain_id: Optional[str] = Query(None, description="Filter stream by blockchain ID"),
    high_risk_only: bool = Query(False, description="Filter for high-risk transactions only"),
    # current_user: Dict = Depends(get_current_user), # Uncomment if authentication is needed directly on WS
    # role_check: bool = Depends(analyst_role) # Uncomment if role check is needed directly on WS
):
    """
    WebSocket endpoint for real-time streaming of blockchain transactions.

    Clients can connect to this endpoint to receive a live feed of transactions,
    optionally filtered by tenant, chain, and risk level.
    """
    # Use the tenant context from the query parameter or default
    # In a real multi-tenant setup, tenant_id would likely come from JWT
    # For now, we'll use the query param as per frontend's expectation
    effective_tenant_id = tenant_id if tenant_id else "default"
    
    # Create a TenantContext for this connection
    # This ensures that any downstream services (like RedisStreamClient)
    # can operate with the correct tenant context.
    with TenantContext(tenant_id=effective_tenant_id) as tenant_ctx:
        endpoint_name = "tx_stream"
        ws_manager: ConnectionManager = get_ws_manager()
        stream_client = get_stream_client()

        if not isinstance(stream_client, RedisStreamClient):
            logger.error(f"Unsupported stream backend for WebSocket: {stream_client.__class__.__name__}")
            await websocket.close(code=status.WS_1011_INTERNAL_ERROR, reason="Unsupported stream backend")
            return

        try:
            await ws_manager.connect(websocket)
            STREAM_CONNECTIONS_TOTAL.labels(endpoint=endpoint_name, status="connected", tenant=tenant_ctx.id).inc()
            STREAM_ACTIVE_CONNECTIONS.labels(endpoint=endpoint_name, tenant=tenant_ctx.id).inc()
            logger.info(f"WebSocket connected: {websocket.client.host}:{websocket.client.port} for tenant {tenant_ctx.id}")

            # Ensure the stream and consumer group exist
            stream_name = stream_client.get_stream_name(endpoint_name, tenant_ctx.id)
            await stream_client.create_stream(endpoint_name, tenant_ctx.id, create_consumer_group=True)

            # Use a unique consumer name for this WebSocket connection
            consumer_name = f"consumer_{websocket.client.host}_{websocket.client.port}"

            while True:
                try:
                    # Read messages from the stream using the consumer group
                    # Block for a short period if no messages are available
                    messages = await stream_client.read_group(
                        stream_name=endpoint_name,
                        consumer_name=consumer_name,
                        tenant_id=tenant_ctx.id,
                        count=10, # Read up to 10 messages at a time
                        block_ms=1000 # Block for 1 second
                    )

                    if messages:
                        for message_id, message_data in messages:
                            try:
                                # Apply backend filters
                                if chain_id and message_data.get("chain_id") != chain_id:
                                    continue
                                if high_risk_only and not message_data.get("is_high_risk", False):
                                    continue
                                
                                # Ensure the message is a valid Transaction model
                                transaction = Transaction(**message_data)
                                
                                start_send_time = asyncio.get_event_loop().time()
                                await websocket.send_json(transaction.dict())
                                end_send_time = asyncio.get_event_loop().time()
                                
                                STREAM_MESSAGES_SENT_TOTAL.labels(
                                    endpoint=endpoint_name, 
                                    tenant=tenant_ctx.id, 
                                    chain_id=transaction.chain_id
                                ).inc()
                                
                                # Acknowledge message after successful processing and sending
                                await stream_client.acknowledge_messages(endpoint_name, [message_id], tenant_ctx.id)

                            except Exception as e:
                                logger.error(f"Error processing or sending message {message_id} for tenant {tenant_ctx.id}: {e}")
                                STREAM_ERRORS_TOTAL.labels(
                                    endpoint=endpoint_name, 
                                    operation="send_message", 
                                    error_type=type(e).__name__, 
                                    tenant=tenant_ctx.id
                                ).inc()
                                # Move to DLQ if processing fails
                                await stream_client.move_to_dead_letter(endpoint_name, message_id, message_data, str(e), tenant_ctx.id)
                                # Acknowledge the message from the main stream so it's not reprocessed
                                await stream_client.acknowledge_messages(endpoint_name, [message_id], tenant_ctx.id)
                                
                        # Claim any pending messages that might have been left by crashed consumers
                        await stream_client.claim_pending_messages(endpoint_name, consumer_name, tenant_id=tenant_ctx.id)

                except asyncio.CancelledError:
                    logger.info(f"WebSocket read task cancelled for tenant {tenant_ctx.id}")
                    break # Exit loop if task is cancelled
                except Exception as e:
                    logger.error(f"Error reading from Redis Stream for tenant {tenant_ctx.id}: {e}")
                    STREAM_ERRORS_TOTAL.labels(
                        endpoint=endpoint_name, 
                        operation="read_stream", 
                        error_type=type(e).__name__, 
                        tenant=tenant_ctx.id
                    ).inc()
                    await asyncio.sleep(1) # Prevent tight loop on persistent errors

        except WebSocketDisconnect:
            logger.info(f"WebSocket disconnected: {websocket.client.host}:{websocket.client.port} for tenant {tenant_ctx.id}")
            STREAM_CONNECTIONS_TOTAL.labels(endpoint=endpoint_name, status="disconnected", tenant=tenant_ctx.id).inc()
        except Exception as e:
            logger.error(f"WebSocket error for tenant {tenant_ctx.id}: {e}")
            STREAM_ERRORS_TOTAL.labels(
                endpoint=endpoint_name, 
                operation="connection_error", 
                error_type=type(e).__name__, 
                tenant=tenant_ctx.id
            ).inc()
        finally:
            STREAM_ACTIVE_CONNECTIONS.labels(endpoint=endpoint_name, tenant=tenant_ctx.id).dec()
            await ws_manager.disconnect(websocket) # Ensure connection is properly removed
