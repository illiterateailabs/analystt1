"""
Whale Detection API Endpoints

This module provides API endpoints for detecting and tracking cryptocurrency whale activity
across multiple blockchains. It integrates with the WhaleDetectionTool to identify large
wallets, monitor significant transactions, and detect potentially coordinated movements.
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple
from fastapi import APIRouter, HTTPException, Depends, Request, Query
from pydantic import BaseModel, Field

from backend.agents.tools.whale_detection_tool import (
    WhaleDetectionTool,
    WhaleDetectionInput,
    WhaleWallet,
    WhaleMovement,
    CoordinationGroup,
    TIER1_WHALE_THRESHOLD,
    TIER2_WHALE_THRESHOLD,
    LARGE_TX_THRESHOLD,
    WHALE_MONITORING_LOOKBACK
)
from backend.auth.rbac import require_roles, Roles, RoleSets
from backend.integrations.sim_client import SimClient, SimApiError

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/whale", tags=["whale"])

# Request/Response Models
class WhaleDetectionRequest(BaseModel):
    """Request model for whale detection."""
    
    wallet_address: Optional[str] = Field(
        None, 
        description="Optional specific wallet to analyze for whale activity"
    )
    lookback_days: int = Field(
        WHALE_MONITORING_LOOKBACK,
        description=f"Days of history to analyze (default: {WHALE_MONITORING_LOOKBACK})"
    )
    tier1_threshold: float = Field(
        TIER1_WHALE_THRESHOLD,
        description=f"USD threshold for Tier 1 whale classification (default: ${TIER1_WHALE_THRESHOLD:,})"
    )
    tier2_threshold: float = Field(
        TIER2_WHALE_THRESHOLD,
        description=f"USD threshold for Tier 2 whale classification (default: ${TIER2_WHALE_THRESHOLD:,})"
    )
    tx_threshold: float = Field(
        LARGE_TX_THRESHOLD,
        description=f"USD threshold for large transaction detection (default: ${LARGE_TX_THRESHOLD:,})"
    )
    detect_coordination: bool = Field(
        True,
        description="Whether to detect coordination between whale wallets"
    )
    chain_ids: Optional[str] = Field(
        None,
        description="Optional comma-separated list of chain IDs to monitor, or 'all' for all chains"
    )

class WhaleMovementRequest(BaseModel):
    """Request model for whale movement detection."""
    
    lookback_days: int = Field(
        WHALE_MONITORING_LOOKBACK,
        description=f"Days of history to analyze (default: {WHALE_MONITORING_LOOKBACK})"
    )
    tx_threshold: float = Field(
        LARGE_TX_THRESHOLD,
        description=f"USD threshold for large transaction detection (default: ${LARGE_TX_THRESHOLD:,})"
    )
    chain_ids: Optional[str] = Field(
        None,
        description="Optional comma-separated list of chain IDs to monitor, or 'all' for all chains"
    )

class WhaleMonitorRequest(BaseModel):
    """Request model for real-time whale monitoring."""
    
    wallets: List[str] = Field(
        ...,
        description="List of wallet addresses to monitor",
        min_items=1,
        max_items=100
    )
    alert_threshold_usd: float = Field(
        100000,
        description="USD threshold for movement alerts (default: $100,000)"
    )
    coordination_detection: bool = Field(
        True,
        description="Whether to detect coordination between monitored wallets"
    )
    chain_ids: Optional[str] = Field(
        None,
        description="Optional comma-separated list of chain IDs to monitor, or 'all' for all chains"
    )

class WhaleStatsRequest(BaseModel):
    """Request model for whale statistics."""
    
    time_period: str = Field(
        "24h",
        description="Time period for statistics (1h, 24h, 7d, 30d)"
    )
    min_tier: str = Field(
        "TIER2",
        description="Minimum whale tier to include (TIER1, TIER2, ACTIVE)"
    )
    chain_ids: Optional[str] = Field(
        None,
        description="Optional comma-separated list of chain IDs, or 'all' for all chains"
    )

class WhaleDetectionResponse(BaseModel):
    """Response model for whale detection."""
    
    whales: List[Dict[str, Any]] = Field(
        ...,
        description="List of detected whale wallets"
    )
    movements: List[Dict[str, Any]] = Field(
        ...,
        description="List of significant whale movements"
    )
    coordination_groups: List[Dict[str, Any]] = Field(
        ...,
        description="List of detected coordination groups"
    )
    stats: Dict[str, Any] = Field(
        ...,
        description="Detection statistics"
    )
    error: Optional[str] = Field(
        None,
        description="Error message if detection failed"
    )

class WhaleMovementResponse(BaseModel):
    """Response model for whale movements."""
    
    wallet_address: str = Field(
        ...,
        description="Wallet address analyzed"
    )
    movements: List[Dict[str, Any]] = Field(
        ...,
        description="List of significant movements"
    )
    stats: Dict[str, Any] = Field(
        ...,
        description="Movement statistics"
    )
    error: Optional[str] = Field(
        None,
        description="Error message if detection failed"
    )

class WhaleMonitorResponse(BaseModel):
    """Response model for real-time whale monitoring."""
    
    monitor_id: str = Field(
        ...,
        description="Unique identifier for this monitoring session"
    )
    wallets_monitored: List[str] = Field(
        ...,
        description="List of wallet addresses being monitored"
    )
    alerts: List[Dict[str, Any]] = Field(
        ...,
        description="List of movement alerts"
    )
    stats: Dict[str, Any] = Field(
        ...,
        description="Monitoring statistics"
    )
    error: Optional[str] = Field(
        None,
        description="Error message if monitoring failed"
    )

class WhaleStatsResponse(BaseModel):
    """Response model for whale statistics."""
    
    time_period: str = Field(
        ...,
        description="Time period for statistics"
    )
    whale_counts: Dict[str, int] = Field(
        ...,
        description="Counts of whales by tier"
    )
    total_value_usd: float = Field(
        ...,
        description="Total USD value held by whales"
    )
    movement_stats: Dict[str, Any] = Field(
        ...,
        description="Statistics on whale movements"
    )
    chain_distribution: Dict[str, int] = Field(
        ...,
        description="Distribution of whales across chains"
    )
    error: Optional[str] = Field(
        None,
        description="Error message if statistics generation failed"
    )

# Dependency functions
async def get_whale_detection_tool(request: Request) -> WhaleDetectionTool:
    """Dependency for getting the WhaleDetectionTool instance."""
    # Get SimClient from app state
    sim_client = request.app.state.sim
    # Create and return WhaleDetectionTool with SimClient
    return WhaleDetectionTool(sim_client=sim_client)

async def get_sim_client(request: Request) -> SimClient:
    """Dependency for getting the SimClient instance."""
    return request.app.state.sim

# API Endpoints
@router.post("/detect", response_model=WhaleDetectionResponse)
@require_roles(RoleSets.ANALYSTS_AND_ADMIN)
async def detect_whales(
    request: WhaleDetectionRequest,
    whale_tool: WhaleDetectionTool = Depends(get_whale_detection_tool)
):
    """
    Detect cryptocurrency whale wallets and their movements.
    
    This endpoint identifies wallets with large holdings (whales), tracks their
    movements, and detects potentially coordinated activities across multiple wallets.
    It can analyze a specific wallet or scan for whales based on configurable thresholds.
    """
    logger.info(f"Whale detection requested: {request.dict()}")
    try:
        # Execute whale detection using the tool
        result = await whale_tool._execute(
            wallet_address=request.wallet_address,
            lookback_days=request.lookback_days,
            tier1_threshold=request.tier1_threshold,
            tier2_threshold=request.tier2_threshold,
            tx_threshold=request.tx_threshold,
            detect_coordination=request.detect_coordination,
            chain_ids=request.chain_ids
        )
        
        logger.info(f"Whale detection completed: {result['stats']['total_whales_detected']} whales found")
        return WhaleDetectionResponse(**result)
        
    except Exception as e:
        logger.error(f"Error in whale detection: {str(e)}")
        return WhaleDetectionResponse(
            whales=[],
            movements=[],
            coordination_groups=[],
            stats={
                "total_whales_detected": 0,
                "new_whales_detected": 0,
                "large_movements_detected": 0,
                "coordination_groups_detected": 0,
                "total_value_monitored_usd": 0,
            },
            error=str(e)
        )

@router.get("/movements/{wallet}", response_model=WhaleMovementResponse)
@require_roles(RoleSets.ANALYSTS_AND_ADMIN)
async def get_whale_movements(
    wallet: str,
    lookback_days: int = WHALE_MONITORING_LOOKBACK,
    tx_threshold: float = LARGE_TX_THRESHOLD,
    chain_ids: Optional[str] = None,
    whale_tool: WhaleDetectionTool = Depends(get_whale_detection_tool)
):
    """
    Get significant movements for a specific wallet address.
    
    This endpoint retrieves large transactions and significant movements
    for the specified wallet address, regardless of whether it qualifies
    as a whale or not. It's useful for tracking specific addresses of interest.
    """
    logger.info(f"Whale movements requested for wallet {wallet}")
    try:
        # Get movements using the tool's internal method
        movements = await whale_tool._get_whale_movements(
            wallet_address=wallet,
            lookback_days=lookback_days,
            tx_threshold=tx_threshold,
            chain_ids=chain_ids
        )
        
        # Calculate statistics
        total_value = sum(m.value_usd for m in movements)
        movement_types = {}
        chains = {}
        
        for m in movements:
            # Count by movement type
            movement_types[m.movement_type] = movement_types.get(m.movement_type, 0) + 1
            
            # Count by chain
            chains[m.chain] = chains.get(m.chain, 0) + 1
        
        stats = {
            "total_movements": len(movements),
            "total_value_usd": total_value,
            "movement_types": movement_types,
            "chains": chains,
            "average_value_usd": total_value / len(movements) if movements else 0
        }
        
        logger.info(f"Found {len(movements)} significant movements for wallet {wallet}")
        return WhaleMovementResponse(
            wallet_address=wallet,
            movements=[m.dict() for m in movements],
            stats=stats
        )
        
    except Exception as e:
        logger.error(f"Error getting whale movements for {wallet}: {str(e)}")
        return WhaleMovementResponse(
            wallet_address=wallet,
            movements=[],
            stats={
                "total_movements": 0,
                "total_value_usd": 0,
                "movement_types": {},
                "chains": {},
                "average_value_usd": 0
            },
            error=str(e)
        )

@router.post("/monitor", response_model=WhaleMonitorResponse)
@require_roles(RoleSets.ANALYSTS_AND_ADMIN)
async def monitor_whales(
    request: WhaleMonitorRequest,
    whale_tool: WhaleDetectionTool = Depends(get_whale_detection_tool),
    sim: SimClient = Depends(get_sim_client)
):
    """
    Set up real-time monitoring for a list of whale wallets.
    
    This endpoint establishes monitoring for a list of wallet addresses,
    checking for new significant movements and potential coordination.
    It returns the current state and any recent alerts.
    
    In a production environment, this would typically initiate a
    background task or webhook registration for continuous monitoring.
    """
    logger.info(f"Whale monitoring requested for {len(request.wallets)} wallets")
    try:
        import uuid
        monitor_id = str(uuid.uuid4())
        
        # For each wallet, check recent movements
        all_movements = []
        alerts = []
        
        for wallet in request.wallets:
            try:
                # Get recent activity
                activity_response = await sim.get_activity(
                    wallet,
                    limit=10  # Get most recent activities
                )
                
                activities = activity_response.get("activity", [])
                
                # Check for large transactions
                for activity in activities:
                    value_usd = float(activity.get("value_usd", 0))
                    if value_usd >= request.alert_threshold_usd:
                        # Create alert
                        alert = {
                            "wallet_address": wallet,
                            "transaction_hash": activity.get("transaction_hash", "unknown"),
                            "type": activity.get("type", "unknown"),
                            "value_usd": value_usd,
                            "timestamp": datetime.fromtimestamp(activity.get("block_time", time.time())).isoformat(),
                            "chain": activity.get("chain", "unknown"),
                            "token_symbol": activity.get("symbol", "unknown"),
                            "alert_level": "HIGH" if value_usd >= request.alert_threshold_usd * 10 else "MEDIUM"
                        }
                        alerts.append(alert)
                        
                        # Add to movements for coordination detection
                        movement = WhaleMovement(
                            transaction_hash=activity.get("transaction_hash", "unknown"),
                            from_address=activity.get("from", wallet),
                            to_address=activity.get("to", "unknown"),
                            value_usd=value_usd,
                            timestamp=datetime.fromtimestamp(activity.get("block_time", time.time())).isoformat(),
                            chain=activity.get("chain", "unknown"),
                            token_address=activity.get("address", None),
                            token_symbol=activity.get("symbol", None),
                            movement_type=activity.get("type", "UNKNOWN").upper(),
                            is_coordinated=False,
                            coordination_group=None
                        )
                        all_movements.append(movement)
            
            except Exception as e:
                logger.error(f"Error monitoring wallet {wallet}: {str(e)}")
                continue
        
        # Detect coordination if enabled and we have multiple movements
        coordination_groups = []
        if request.coordination_detection and len(all_movements) > 1:
            coordination_groups = whale_tool._detect_coordination_patterns(all_movements)
            
            # Add coordination alerts
            for group in coordination_groups:
                alert = {
                    "type": "COORDINATION",
                    "pattern_type": group.pattern_type,
                    "wallets_involved": len(group.wallets),
                    "total_value_usd": group.total_value_usd,
                    "confidence": group.confidence,
                    "start_time": group.start_time,
                    "end_time": group.end_time,
                    "alert_level": "HIGH" if group.confidence > 0.8 else "MEDIUM",
                    "group_id": group.group_id
                }
                alerts.append(alert)
        
        # Calculate statistics
        stats = {
            "wallets_monitored": len(request.wallets),
            "alerts_generated": len(alerts),
            "coordination_groups_detected": len(coordination_groups),
            "alert_threshold_usd": request.alert_threshold_usd,
            "monitor_start_time": datetime.now().isoformat()
        }
        
        logger.info(f"Whale monitoring initialized with ID {monitor_id}: {len(alerts)} alerts generated")
        return WhaleMonitorResponse(
            monitor_id=monitor_id,
            wallets_monitored=request.wallets,
            alerts=alerts,
            stats=stats
        )
        
    except Exception as e:
        logger.error(f"Error in whale monitoring: {str(e)}")
        return WhaleMonitorResponse(
            monitor_id="error",
            wallets_monitored=request.wallets,
            alerts=[],
            stats={
                "wallets_monitored": len(request.wallets),
                "alerts_generated": 0,
                "coordination_groups_detected": 0,
                "alert_threshold_usd": request.alert_threshold_usd,
                "monitor_start_time": datetime.now().isoformat()
            },
            error=str(e)
        )

@router.get("/stats", response_model=WhaleStatsResponse)
@require_roles(RoleSets.ANALYSTS_AND_ADMIN)
async def get_whale_stats(
    time_period: str = "24h",
    min_tier: str = "TIER2",
    chain_ids: Optional[str] = None,
    whale_tool: WhaleDetectionTool = Depends(get_whale_detection_tool)
):
    """
    Get statistics about whale activity and distribution.
    
    This endpoint provides aggregated statistics about whale wallets,
    including counts by tier, total value held, movement patterns,
    and distribution across different blockchains.
    """
    logger.info(f"Whale statistics requested: period={time_period}, min_tier={min_tier}")
    try:
        # Convert time period to days for lookback
        lookback_days = {
            "1h": 1/24,
            "24h": 1,
            "7d": 7,
            "30d": 30
        }.get(time_period, 1)
        
        # Get known whales from the tool
        whales = list(whale_tool._known_whales.values())
        
        # Filter by tier
        tier_priority = {"TIER1": 3, "TIER2": 2, "ACTIVE": 1}
        min_tier_priority = tier_priority.get(min_tier, 1)
        filtered_whales = [w for w in whales if tier_priority.get(w.tier, 0) >= min_tier_priority]
        
        # Count whales by tier
        whale_counts = {}
        for whale in filtered_whales:
            whale_counts[whale.tier] = whale_counts.get(whale.tier, 0) + 1
        
        # Calculate total value
        total_value_usd = sum(w.total_value_usd for w in filtered_whales)
        
        # Count whales by chain
        chain_distribution = {}
        for whale in filtered_whales:
            for chain in whale.chains:
                chain_distribution[chain] = chain_distribution.get(chain, 0) + 1
        
        # Get recent movements
        recent_movements = whale_tool._recent_large_movements
        
        # Filter movements by time period
        cutoff_time = datetime.now() - timedelta(days=lookback_days)
        filtered_movements = [
            m for m in recent_movements 
            if datetime.fromisoformat(m.timestamp) >= cutoff_time
        ]
        
        # Calculate movement statistics
        movement_stats = {
            "total_movements": len(filtered_movements),
            "total_value_usd": sum(m.value_usd for m in filtered_movements),
            "average_value_usd": sum(m.value_usd for m in filtered_movements) / len(filtered_movements) if filtered_movements else 0,
            "movement_types": {},
            "coordinated_movements": sum(1 for m in filtered_movements if m.is_coordinated)
        }
        
        # Count movement types
        for movement in filtered_movements:
            movement_type = movement.movement_type
            movement_stats["movement_types"][movement_type] = movement_stats["movement_types"].get(movement_type, 0) + 1
        
        logger.info(f"Whale statistics generated: {len(filtered_whales)} whales, {len(filtered_movements)} movements")
        return WhaleStatsResponse(
            time_period=time_period,
            whale_counts=whale_counts,
            total_value_usd=total_value_usd,
            movement_stats=movement_stats,
            chain_distribution=chain_distribution
        )
        
    except Exception as e:
        logger.error(f"Error generating whale statistics: {str(e)}")
        return WhaleStatsResponse(
            time_period=time_period,
            whale_counts={},
            total_value_usd=0,
            movement_stats={
                "total_movements": 0,
                "total_value_usd": 0,
                "average_value_usd": 0,
                "movement_types": {},
                "coordinated_movements": 0
            },
            chain_distribution={},
            error=str(e)
        )
