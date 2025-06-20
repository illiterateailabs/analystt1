"""
Cross-Chain Identity Analysis Tool

This tool analyzes wallet identities and transaction patterns across multiple blockchain networks
to detect cross-chain movements, identify potential identity clusters, and assess associated risks.
It leverages Sim APIs to gather multi-chain data and integrates with the graph database for persistence.
"""

import logging
import time
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple, Union
from pydantic import BaseModel, Field, validator
import networkx as nx
import uuid

from crewai_tools import BaseTool
from backend.integrations.sim_client import SimClient
from backend.core.metrics import record_tool_usage, record_tool_error
from backend.core.events import emit_event, GraphAddEvent

logger = logging.getLogger(__name__)

# Constants for cross-chain analysis
CROSS_CHAIN_TIME_WINDOW = 15  # Minutes to consider for cross-chain movements
BRIDGE_VALUE_THRESHOLD = 1000  # Minimum USD value to consider for bridge transactions
IDENTITY_CONFIDENCE_THRESHOLD = 0.7  # Minimum confidence for identity clustering
MAX_WALLETS_PER_REQUEST = 50  # Maximum wallets to process in one request
LOOKBACK_DAYS_DEFAULT = 7  # Default lookback period for activity data

# Known bridge contracts (simplified list, would be more comprehensive in production)
KNOWN_BRIDGE_CONTRACTS = {
    # Ethereum → Polygon
    "0xa0c68c638235ee32657e8f720a23cec1bfc77c77": "Polygon Bridge",
    # Ethereum → Arbitrum
    "0x8315177ab297ba92a06054ce80a67ed4dbd7ed3a": "Arbitrum Bridge",
    # Ethereum → Optimism
    "0x99c9fc46f92e8a1c0dec1b1747d010903e884be1": "Optimism Bridge",
    # Ethereum → Base
    "0x3154cf16ccdb4c6d922629664174b904d80f2c35": "Base Bridge",
    # Add more known bridge contracts as needed
}

# Chain ID mapping for common EVM chains
CHAIN_ID_TO_NAME = {
    "1": "ethereum",
    "137": "polygon",
    "42161": "arbitrum",
    "10": "optimism",
    "56": "bnb",
    "8453": "base",
    "43114": "avalanche",
    "250": "fantom",
    # Add more chains as needed
}

# Chain name to ID mapping (reverse of above)
CHAIN_NAME_TO_ID = {v: k for k, v in CHAIN_ID_TO_NAME.items()}


class CrossChainWallet(BaseModel):
    """Model representing a wallet's presence on a specific blockchain."""
    address: str = Field(..., description="Wallet address")
    chain_id: str = Field(..., description="Chain ID where this wallet exists")
    chain_name: str = Field(..., description="Chain name (e.g., 'ethereum', 'polygon')")
    first_seen: Optional[str] = Field(None, description="First seen timestamp")
    last_seen: Optional[str] = Field(None, description="Last seen timestamp")
    total_value_usd: float = Field(0, description="Total portfolio value in USD")
    transaction_count: int = Field(0, description="Number of transactions")
    token_count: int = Field(0, description="Number of tokens held")
    is_contract: bool = Field(False, description="Whether this address is a contract")


class CrossChainMovement(BaseModel):
    """Model representing a detected cross-chain movement (e.g., bridge transaction)."""
    id: str = Field(..., description="Unique identifier for this movement")
    source_chain_id: str = Field(..., description="Source chain ID")
    source_chain_name: str = Field(..., description="Source chain name")
    destination_chain_id: str = Field(..., description="Destination chain ID")
    destination_chain_name: str = Field(..., description="Destination chain name")
    source_transaction: Optional[Dict[str, Any]] = Field(None, description="Source transaction details")
    destination_transaction: Optional[Dict[str, Any]] = Field(None, description="Destination transaction details")
    wallet_address: str = Field(..., description="Wallet address involved")
    bridge_address: Optional[str] = Field(None, description="Bridge contract address if known")
    bridge_name: Optional[str] = Field(None, description="Bridge name if known")
    value_usd: float = Field(..., description="USD value of the movement")
    source_time: str = Field(..., description="Source transaction timestamp")
    destination_time: Optional[str] = Field(None, description="Destination transaction timestamp")
    time_difference_minutes: Optional[float] = Field(None, description="Time difference in minutes")
    confidence: float = Field(..., description="Confidence score for this movement (0-1)")
    risk_score: float = Field(0, description="Risk score for this movement (0-100)")
    risk_factors: List[str] = Field(default_factory=list, description="Risk factors for this movement")


class IdentityCluster(BaseModel):
    """Model representing a cluster of wallets identified as belonging to the same entity."""
    cluster_id: str = Field(..., description="Unique identifier for this cluster")
    wallets: List[CrossChainWallet] = Field(..., description="Wallets in this cluster")
    main_address: str = Field(..., description="Main address for this cluster")
    total_value_usd: float = Field(0, description="Total value across all wallets in USD")
    chains: List[str] = Field(..., description="Chains where this entity has presence")
    first_seen: Optional[str] = Field(None, description="First seen timestamp across all wallets")
    last_seen: Optional[str] = Field(None, description="Last seen timestamp across all wallets")
    cross_chain_movements: List[str] = Field(default_factory=list, description="IDs of cross-chain movements")
    confidence: float = Field(..., description="Confidence score for this cluster (0-1)")
    risk_score: float = Field(0, description="Risk score for this cluster (0-100)")
    risk_factors: List[str] = Field(default_factory=list, description="Risk factors for this cluster")
    coordination_patterns: List[str] = Field(default_factory=list, description="Detected coordination patterns")


class BridgeUsage(BaseModel):
    """Model representing bridge usage patterns."""
    bridge_address: str = Field(..., description="Bridge contract address")
    bridge_name: Optional[str] = Field(None, description="Bridge name if known")
    source_chain_id: str = Field(..., description="Source chain ID")
    destination_chain_id: str = Field(..., description="Destination chain ID")
    usage_count: int = Field(0, description="Number of times this bridge was used")
    total_value_usd: float = Field(0, description="Total USD value bridged")
    average_value_usd: float = Field(0, description="Average USD value per bridge transaction")
    first_used: Optional[str] = Field(None, description="First usage timestamp")
    last_used: Optional[str] = Field(None, description="Last usage timestamp")
    risk_score: float = Field(0, description="Risk score for this bridge usage (0-100)")
    risk_factors: List[str] = Field(default_factory=list, description="Risk factors for this bridge usage")


class CrossChainIdentityInput(BaseModel):
    """Input parameters for cross-chain identity analysis."""
    wallet_addresses: List[str] = Field(
        ..., 
        description="List of wallet addresses to analyze",
        min_items=1,
        max_items=MAX_WALLETS_PER_REQUEST
    )
    lookback_days: int = Field(
        LOOKBACK_DAYS_DEFAULT,
        description=f"Number of days to look back for activity data (default: {LOOKBACK_DAYS_DEFAULT})"
    )
    chain_ids: Optional[str] = Field(
        "all",
        description="Comma-separated list of chain IDs to include, or 'all' for all supported chains"
    )
    detect_clusters: bool = Field(
        True,
        description="Whether to detect identity clusters"
    )
    detect_cross_chain_movements: bool = Field(
        True,
        description="Whether to detect cross-chain movements"
    )
    analyze_bridge_usage: bool = Field(
        True,
        description="Whether to analyze bridge usage patterns"
    )
    detect_coordination: bool = Field(
        True,
        description="Whether to detect coordination patterns"
    )
    min_bridge_value_usd: float = Field(
        BRIDGE_VALUE_THRESHOLD,
        description=f"Minimum USD value for bridge transactions (default: ${BRIDGE_VALUE_THRESHOLD})"
    )
    cross_chain_time_window: int = Field(
        CROSS_CHAIN_TIME_WINDOW,
        description=f"Time window in minutes for cross-chain movements (default: {CROSS_CHAIN_TIME_WINDOW})"
    )
    emit_graph_events: bool = Field(
        True,
        description="Whether to emit graph events for Neo4j integration"
    )


class CrossChainIdentityOutput(BaseModel):
    """Output from cross-chain identity analysis."""
    wallets: List[CrossChainWallet] = Field(..., description="Wallets analyzed across chains")
    cross_chain_movements: List[CrossChainMovement] = Field(..., description="Detected cross-chain movements")
    identity_clusters: List[IdentityCluster] = Field(..., description="Detected identity clusters")
    bridge_usage: List[BridgeUsage] = Field(..., description="Bridge usage analysis")
    overall_risk_score: float = Field(..., description="Overall risk score (0-100)")
    overall_risk_factors: List[str] = Field(..., description="Overall risk factors")
    wallet_addresses: List[str] = Field(..., description="Input wallet addresses")
    start_time: str = Field(..., description="Analysis start time")
    end_time: str = Field(..., description="Analysis end time")
    error: Optional[str] = Field(None, description="Error message if any")


class CrossChainIdentityTool(BaseTool):
    """
    Tool for analyzing cross-chain identity and detecting related wallets across multiple blockchains.
    
    This tool tracks wallet identities across multiple blockchain networks, detects bridge transactions,
    identifies potential identity clusters based on timing and amount patterns, and provides
    risk scoring for cross-chain activities.
    """
    
    name: str = "cross_chain_identity_tool"
    description: str = """
    Analyzes wallet identities across multiple blockchain networks to detect cross-chain movements,
    identify potential identity clusters, and assess associated risks.
    
    Useful for:
    - Tracking funds across multiple chains
    - Identifying related wallets belonging to the same entity
    - Detecting suspicious cross-chain coordination patterns
    - Analyzing bridge usage and associated risks
    - Generating compliance reports for cross-chain activities
    """
    
    sim_client: SimClient
    
    def __init__(self, sim_client: Optional[SimClient] = None):
        """
        Initialize the CrossChainIdentityTool.
        
        Args:
            sim_client: Optional SimClient instance. If not provided, a new one will be created.
        """
        super().__init__()
        self.sim_client = sim_client or SimClient()
    
    async def _execute(
        self,
        wallet_addresses: List[str],
        lookback_days: int = LOOKBACK_DAYS_DEFAULT,
        chain_ids: Optional[str] = "all",
        detect_clusters: bool = True,
        detect_cross_chain_movements: bool = True,
        analyze_bridge_usage: bool = True,
        detect_coordination: bool = True,
        min_bridge_value_usd: float = BRIDGE_VALUE_THRESHOLD,
        cross_chain_time_window: int = CROSS_CHAIN_TIME_WINDOW,
        emit_graph_events: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute cross-chain identity analysis.
        
        Args:
            wallet_addresses: List of wallet addresses to analyze
            lookback_days: Number of days to look back for activity data
            chain_ids: Comma-separated list of chain IDs to include, or 'all'
            detect_clusters: Whether to detect identity clusters
            detect_cross_chain_movements: Whether to detect cross-chain movements
            analyze_bridge_usage: Whether to analyze bridge usage patterns
            detect_coordination: Whether to detect coordination patterns
            min_bridge_value_usd: Minimum USD value for bridge transactions
            cross_chain_time_window: Time window in minutes for cross-chain movements
            emit_graph_events: Whether to emit graph events for Neo4j integration
            
        Returns:
            Dictionary containing cross-chain identity analysis results
        """
        try:
            # Record tool usage for metrics
            record_tool_usage(self.name)
            
            # Record start time
            start_time = datetime.now().isoformat()
            
            # Validate input
            if not wallet_addresses:
                return {
                    "wallets": [],
                    "cross_chain_movements": [],
                    "identity_clusters": [],
                    "bridge_usage": [],
                    "overall_risk_score": 0,
                    "overall_risk_factors": ["No wallet addresses provided"],
                    "wallet_addresses": [],
                    "start_time": start_time,
                    "end_time": datetime.now().isoformat(),
                    "error": "No wallet addresses provided"
                }
            
            # Limit the number of wallets to analyze
            if len(wallet_addresses) > MAX_WALLETS_PER_REQUEST:
                wallet_addresses = wallet_addresses[:MAX_WALLETS_PER_REQUEST]
                logger.warning(f"Limited analysis to {MAX_WALLETS_PER_REQUEST} wallets")
            
            # 1. Fetch multi-chain data
            wallet_data = await self._fetch_multi_chain_data(
                wallet_addresses=wallet_addresses,
                lookback_days=lookback_days,
                chain_ids=chain_ids
            )
            
            # Extract wallet data
            wallets = wallet_data["wallets"]
            activities = wallet_data["activities"]
            balances = wallet_data["balances"]
            
            # 2. Detect cross-chain movements
            cross_chain_movements = []
            if detect_cross_chain_movements:
                cross_chain_movements = await self._detect_cross_chain_movements(
                    wallets=wallets,
                    activities=activities,
                    min_bridge_value_usd=min_bridge_value_usd,
                    cross_chain_time_window=cross_chain_time_window
                )
            
            # 3. Analyze bridge usage
            bridge_usage = []
            if analyze_bridge_usage and cross_chain_movements:
                bridge_usage = self._analyze_bridge_usage(cross_chain_movements)
            
            # 4. Identify identity clusters
            identity_clusters = []
            if detect_clusters:
                identity_clusters = await self._identify_identity_clusters(
                    wallets=wallets,
                    activities=activities,
                    balances=balances,
                    cross_chain_movements=cross_chain_movements
                )
            
            # 5. Detect coordination patterns
            if detect_coordination and identity_clusters:
                self._detect_coordination_patterns(
                    identity_clusters=identity_clusters,
                    cross_chain_movements=cross_chain_movements,
                    activities=activities
                )
            
            # 6. Calculate overall risk score
            overall_risk_score, overall_risk_factors = self._calculate_overall_risk(
                wallets=wallets,
                cross_chain_movements=cross_chain_movements,
                identity_clusters=identity_clusters,
                bridge_usage=bridge_usage
            )
            
            # 7. Emit graph events if requested
            if emit_graph_events:
                self._emit_graph_events(
                    wallets=wallets,
                    cross_chain_movements=cross_chain_movements,
                    identity_clusters=identity_clusters,
                    bridge_usage=bridge_usage
                )
            
            # Record end time
            end_time = datetime.now().isoformat()
            
            # Prepare result
            result = {
                "wallets": wallets,
                "cross_chain_movements": cross_chain_movements,
                "identity_clusters": identity_clusters,
                "bridge_usage": bridge_usage,
                "overall_risk_score": overall_risk_score,
                "overall_risk_factors": overall_risk_factors,
                "wallet_addresses": wallet_addresses,
                "start_time": start_time,
                "end_time": end_time
            }
            
            return result
            
        except Exception as e:
            error_msg = f"Error in cross-chain identity analysis: {str(e)}"
            logger.error(error_msg, exc_info=True)
            record_tool_error(self.name, str(e))
            
            # Return empty result with error
            return {
                "wallets": [],
                "cross_chain_movements": [],
                "identity_clusters": [],
                "bridge_usage": [],
                "overall_risk_score": 0,
                "overall_risk_factors": [f"Error: {str(e)}"],
                "wallet_addresses": wallet_addresses,
                "start_time": start_time if 'start_time' in locals() else datetime.now().isoformat(),
                "end_time": datetime.now().isoformat(),
                "error": str(e)
            }
    
    async def _fetch_multi_chain_data(
        self,
        wallet_addresses: List[str],
        lookback_days: int,
        chain_ids: Optional[str]
    ) -> Dict[str, Any]:
        """
        Fetch data for multiple wallets across multiple chains.
        
        Args:
            wallet_addresses: List of wallet addresses to fetch data for
            lookback_days: Number of days to look back for activity data
            chain_ids: Comma-separated list of chain IDs to include, or 'all'
            
        Returns:
            Dictionary containing wallets, activities, and balances data
        """
        # Initialize result containers
        wallets = []
        all_activities = []
        all_balances = []
        
        # Process each wallet address
        for wallet_address in wallet_addresses:
            try:
                # Fetch balances
                balances_response = await self.sim_client.get_balances(
                    wallet_address,
                    chain_ids=chain_ids,
                    limit=100,
                    metadata="url,logo"
                )
                
                balances = balances_response.get("balances", [])
                all_balances.extend(balances)
                
                # Process balances to create wallet objects for each chain
                chain_data = {}
                for balance in balances:
                    chain_id = str(balance.get("chain_id", ""))
                    chain_name = balance.get("chain", "")
                    
                    if chain_id not in chain_data:
                        chain_data[chain_id] = {
                            "address": wallet_address,
                            "chain_id": chain_id,
                            "chain_name": chain_name,
                            "total_value_usd": 0,
                            "token_count": 0,
                            "first_seen": None,
                            "last_seen": None
                        }
                    
                    # Update chain data
                    chain_data[chain_id]["total_value_usd"] += float(balance.get("value_usd", 0))
                    chain_data[chain_id]["token_count"] += 1
                
                # Fetch activity data
                activity_response = await self.sim_client.get_activity(
                    wallet_address,
                    limit=100
                )
                
                activities = activity_response.get("activity", [])
                
                # Filter activities by lookback period
                cutoff_time = datetime.now() - timedelta(days=lookback_days)
                cutoff_timestamp = cutoff_time.timestamp()
                
                filtered_activities = []
                for activity in activities:
                    if "block_time" in activity and activity["block_time"] >= cutoff_timestamp:
                        # Add wallet address to activity for reference
                        activity["wallet_address"] = wallet_address
                        filtered_activities.append(activity)
                
                all_activities.extend(filtered_activities)
                
                # Update chain data with activity information
                for activity in filtered_activities:
                    chain_id = str(activity.get("chain_id", ""))
                    block_time = activity.get("block_time", 0)
                    timestamp = datetime.fromtimestamp(block_time).isoformat()
                    
                    if chain_id in chain_data:
                        # Update first/last seen
                        if chain_data[chain_id]["first_seen"] is None or timestamp < chain_data[chain_id]["first_seen"]:
                            chain_data[chain_id]["first_seen"] = timestamp
                        
                        if chain_data[chain_id]["last_seen"] is None or timestamp > chain_data[chain_id]["last_seen"]:
                            chain_data[chain_id]["last_seen"] = timestamp
                
                # Create wallet objects for each chain
                for chain_id, data in chain_data.items():
                    # Count transactions for this chain
                    tx_count = sum(1 for a in filtered_activities if str(a.get("chain_id", "")) == chain_id)
                    
                    wallet = CrossChainWallet(
                        address=data["address"],
                        chain_id=data["chain_id"],
                        chain_name=data["chain_name"],
                        first_seen=data["first_seen"],
                        last_seen=data["last_seen"],
                        total_value_usd=data["total_value_usd"],
                        transaction_count=tx_count,
                        token_count=data["token_count"],
                        is_contract=False  # Would need additional API call to determine
                    )
                    
                    wallets.append(wallet)
                
            except Exception as e:
                logger.error(f"Error fetching data for wallet {wallet_address}: {str(e)}")
                # Continue with next wallet
        
        return {
            "wallets": wallets,
            "activities": all_activities,
            "balances": all_balances
        }
    
    async def _detect_cross_chain_movements(
        self,
        wallets: List[CrossChainWallet],
        activities: List[Dict[str, Any]],
        min_bridge_value_usd: float,
        cross_chain_time_window: int
    ) -> List[CrossChainMovement]:
        """
        Detect cross-chain movements (bridge transactions) from activity data.
        
        Args:
            wallets: List of wallet objects
            activities: List of activity dictionaries
            min_bridge_value_usd: Minimum USD value for bridge transactions
            cross_chain_time_window: Time window in minutes for cross-chain movements
            
        Returns:
            List of detected cross-chain movements
        """
        cross_chain_movements = []
        
        # Group activities by wallet address
        wallet_activities = {}
        for activity in activities:
            wallet_address = activity.get("wallet_address", "")
            if wallet_address:
                if wallet_address not in wallet_activities:
                    wallet_activities[wallet_address] = []
                wallet_activities[wallet_address].append(activity)
        
        # Process each wallet's activities
        for wallet_address, wallet_activities_list in wallet_activities.items():
            # Sort activities by block time
            sorted_activities = sorted(wallet_activities_list, key=lambda x: x.get("block_time", 0))
            
            # Find outgoing transactions (potential source of cross-chain movement)
            outgoing_txs = [
                tx for tx in sorted_activities 
                if tx.get("type") in ["send", "call"] and float(tx.get("value_usd", 0)) >= min_bridge_value_usd
            ]
            
            # Find incoming transactions (potential destination of cross-chain movement)
            incoming_txs = [
                tx for tx in sorted_activities 
                if tx.get("type") == "receive" and float(tx.get("value_usd", 0)) >= min_bridge_value_usd
            ]
            
            # Match outgoing and incoming transactions
            for outgoing_tx in outgoing_txs:
                outgoing_time = datetime.fromtimestamp(outgoing_tx.get("block_time", 0))
                outgoing_chain_id = str(outgoing_tx.get("chain_id", ""))
                outgoing_chain_name = outgoing_tx.get("chain", "")
                outgoing_value_usd = float(outgoing_tx.get("value_usd", 0))
                
                # Check if this is a known bridge contract
                to_address = outgoing_tx.get("to", "").lower()
                is_known_bridge = to_address in KNOWN_BRIDGE_CONTRACTS
                
                # Look for matching incoming transactions
                for incoming_tx in incoming_txs:
                    incoming_time = datetime.fromtimestamp(incoming_tx.get("block_time", 0))
                    incoming_chain_id = str(incoming_tx.get("chain_id", ""))
                    incoming_chain_name = incoming_tx.get("chain", "")
                    incoming_value_usd = float(incoming_tx.get("value_usd", 0))
                    
                    # Skip if same chain
                    if outgoing_chain_id == incoming_chain_id:
                        continue
                    
                    # Check if within time window
                    time_diff = incoming_time - outgoing_time
                    time_diff_minutes = time_diff.total_seconds() / 60
                    
                    if 0 <= time_diff_minutes <= cross_chain_time_window:
                        # Check for value similarity (allow for fees)
                        value_ratio = min(outgoing_value_usd, incoming_value_usd) / max(outgoing_value_usd, incoming_value_usd)
                        
                        # Calculate confidence based on multiple factors
                        confidence = 0.5  # Base confidence
                        
                        # Known bridge increases confidence
                        if is_known_bridge:
                            confidence += 0.3
                        
                        # Value similarity increases confidence
                        if value_ratio > 0.9:
                            confidence += 0.2
                        elif value_ratio > 0.7:
                            confidence += 0.1
                        
                        # Short time window increases confidence
                        if time_diff_minutes < 5:
                            confidence += 0.1
                        
                        # Cap confidence at 1.0
                        confidence = min(1.0, confidence)
                        
                        # Calculate risk score
                        risk_score = 0
                        risk_factors = []
                        
                        # High value increases risk
                        if outgoing_value_usd > 100000:
                            risk_score += 30
                            risk_factors.append(f"High value movement: ${outgoing_value_usd:.2f}")
                        elif outgoing_value_usd > 10000:
                            risk_score += 10
                        
                        # Unknown bridge increases risk
                        if not is_known_bridge:
                            risk_score += 20
                            risk_factors.append("Unknown bridge contract")
                        
                        # Very fast bridging increases risk
                        if time_diff_minutes < 2:
                            risk_score += 10
                            risk_factors.append(f"Very fast cross-chain movement: {time_diff_minutes:.1f} minutes")
                        
                        # Create cross-chain movement
                        movement = CrossChainMovement(
                            id=str(uuid.uuid4()),
                            source_chain_id=outgoing_chain_id,
                            source_chain_name=outgoing_chain_name,
                            destination_chain_id=incoming_chain_id,
                            destination_chain_name=incoming_chain_name,
                            source_transaction=outgoing_tx,
                            destination_transaction=incoming_tx,
                            wallet_address=wallet_address,
                            bridge_address=to_address if is_known_bridge else None,
                            bridge_name=KNOWN_BRIDGE_CONTRACTS.get(to_address) if is_known_bridge else None,
                            value_usd=outgoing_value_usd,
                            source_time=outgoing_time.isoformat(),
                            destination_time=incoming_time.isoformat(),
                            time_difference_minutes=time_diff_minutes,
                            confidence=confidence,
                            risk_score=min(100, risk_score),
                            risk_factors=risk_factors
                        )
                        
                        cross_chain_movements.append(movement)
        
        return cross_chain_movements
    
    async def _identify_identity_clusters(
        self,
        wallets: List[CrossChainWallet],
        activities: List[Dict[str, Any]],
        balances: List[Dict[str, Any]],
        cross_chain_movements: List[CrossChainMovement]
    ) -> List[IdentityCluster]:
        """
        Identify clusters of wallets that likely belong to the same entity.
        
        Args:
            wallets: List of wallet objects
            activities: List of activity dictionaries
            balances: List of balance dictionaries
            cross_chain_movements: List of cross-chain movement objects
            
        Returns:
            List of identity clusters
        """
        # Group wallets by address
        address_to_wallets = {}
        for wallet in wallets:
            if wallet.address not in address_to_wallets:
                address_to_wallets[wallet.address] = []
            address_to_wallets[wallet.address].append(wallet)
        
        # Create initial clusters (one per address)
        initial_clusters = []
        for address, address_wallets in address_to_wallets.items():
            # Calculate total value and chains
            total_value = sum(w.total_value_usd for w in address_wallets)
            chains = [w.chain_name for w in address_wallets]
            
            # Find first and last seen
            first_seen_dates = [w.first_seen for w in address_wallets if w.first_seen]
            last_seen_dates = [w.last_seen for w in address_wallets if w.last_seen]
            
            first_seen = min(first_seen_dates) if first_seen_dates else None
            last_seen = max(last_seen_dates) if last_seen_dates else None
            
            # Create cluster
            cluster = IdentityCluster(
                cluster_id=str(uuid.uuid4()),
                wallets=address_wallets,
                main_address=address,
                total_value_usd=total_value,
                chains=chains,
                first_seen=first_seen,
                last_seen=last_seen,
                cross_chain_movements=[],
                confidence=1.0,  # Initial confidence is 1.0 (same address)
                risk_score=0,
                risk_factors=[],
                coordination_patterns=[]
            )
            
            initial_clusters.append(cluster)
        
        # Merge clusters based on cross-chain movements
        merged_clusters = self._merge_clusters_by_movements(initial_clusters, cross_chain_movements)
        
        # Merge clusters based on activity patterns
        final_clusters = await self._merge_clusters_by_patterns(merged_clusters, activities, balances)
        
        # Calculate risk scores for each cluster
        for cluster in final_clusters:
            risk_score, risk_factors = self._calculate_cluster_risk(cluster, cross_chain_movements)
            cluster.risk_score = risk_score
            cluster.risk_factors = risk_factors
        
        return final_clusters
    
    def _merge_clusters_by_movements(
        self,
        clusters: List[IdentityCluster],
        cross_chain_movements: List[CrossChainMovement]
    ) -> List[IdentityCluster]:
        """
        Merge clusters based on cross-chain movements.
        
        Args:
            clusters: List of identity clusters
            cross_chain_movements: List of cross-chain movement objects
            
        Returns:
            List of merged identity clusters
        """
        if not cross_chain_movements:
            return clusters
        
        # Create a mapping of address to cluster
        address_to_cluster = {cluster.main_address: cluster for cluster in clusters}
        
        # Create a union-find data structure for merging clusters
        parent = {cluster.cluster_id: cluster.cluster_id for cluster in clusters}
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            parent[find(x)] = find(y)
        
        # Merge clusters based on cross-chain movements
        for movement in cross_chain_movements:
            wallet_address = movement.wallet_address
            if wallet_address in address_to_cluster:
                cluster = address_to_cluster[wallet_address]
                
                # Add cross-chain movement to cluster
                if movement.id not in cluster.cross_chain_movements:
                    cluster.cross_chain_movements.append(movement.id)
                
                # No merging needed here, just recording movements
        
        # Collect merged clusters
        merged_clusters = []
        cluster_map = {}
        
        for cluster in clusters:
            root_id = find(cluster.cluster_id)
            if root_id not in cluster_map:
                cluster_map[root_id] = cluster
            else:
                # Merge this cluster into the root cluster
                root_cluster = cluster_map[root_id]
                
                # Merge wallets
                for wallet in cluster.wallets:
                    if wallet not in root_cluster.wallets:
                        root_cluster.wallets.append(wallet)
                
                # Merge chains
                for chain in cluster.chains:
                    if chain not in root_cluster.chains:
                        root_cluster.chains.append(chain)
                
                # Merge cross-chain movements
                for movement_id in cluster.cross_chain_movements:
                    if movement_id not in root_cluster.cross_chain_movements:
                        root_cluster.cross_chain_movements.append(movement_id)
                
                # Update total value
                root_cluster.total_value_usd += cluster.total_value_usd
                
                # Update first/last seen
                if cluster.first_seen and (not root_cluster.first_seen or cluster.first_seen < root_cluster.first_seen):
                    root_cluster.first_seen = cluster.first_seen
                
                if cluster.last_seen and (not root_cluster.last_seen or cluster.last_seen > root_cluster.last_seen):
                    root_cluster.last_seen = cluster.last_seen
                
                # Update confidence (take the minimum)
                root_cluster.confidence = min(root_cluster.confidence, cluster.confidence)
        
        # Collect final merged clusters
        merged_clusters = list(cluster_map.values())
        
        return merged_clusters
    
    async def _merge_clusters_by_patterns(
        self,
        clusters: List[IdentityCluster],
        activities: List[Dict[str, Any]],
        balances: List[Dict[str, Any]]
    ) -> List[IdentityCluster]:
        """
        Merge clusters based on activity and balance patterns.
        
        Args:
            clusters: List of identity clusters
            activities: List of activity dictionaries
            balances: List of balance dictionaries
            
        Returns:
            List of merged identity clusters
        """
        if len(clusters) <= 1:
            return clusters
        
        # Build a similarity graph
        G = nx.Graph()
        
        # Add nodes for each cluster
        for cluster in clusters:
            G.add_node(cluster.cluster_id, cluster=cluster)
        
        # Calculate similarity between each pair of clusters
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                cluster1 = clusters[i]
                cluster2 = clusters[j]
                
                # Skip if already connected by cross-chain movements
                if any(movement_id in cluster2.cross_chain_movements for movement_id in cluster1.cross_chain_movements):
                    continue
                
                # Calculate similarity based on activity patterns
                similarity = await self._calculate_cluster_similarity(cluster1, cluster2, activities, balances)
                
                # Add edge if similarity is above threshold
                if similarity >= IDENTITY_CONFIDENCE_THRESHOLD:
                    G.add_edge(
                        cluster1.cluster_id,
                        cluster2.cluster_id,
                        similarity=similarity
                    )
        
        # Find connected components (clusters of clusters)
        connected_components = list(nx.connected_components(G))
        
        # Merge clusters within each connected component
        merged_clusters = []
        
        for component in connected_components:
            if len(component) == 1:
                # Single cluster, no merging needed
                cluster_id = list(component)[0]
                cluster = next(c for c in clusters if c.cluster_id == cluster_id)
                merged_clusters.append(cluster)
            else:
                # Merge multiple clusters
                component_clusters = [next(c for c in clusters if c.cluster_id == cluster_id) for cluster_id in component]
                
                # Create merged cluster
                merged_cluster = self._merge_cluster_group(component_clusters)
                merged_clusters.append(merged_cluster)
        
        return merged_clusters
    
    async def _calculate_cluster_similarity(
        self,
        cluster1: IdentityCluster,
        cluster2: IdentityCluster,
        activities: List[Dict[str, Any]],
        balances: List[Dict[str, Any]]
    ) -> float:
        """
        Calculate similarity between two clusters based on activity and balance patterns.
        
        Args:
            cluster1: First identity cluster
            cluster2: Second identity cluster
            activities: List of activity dictionaries
            balances: List of balance dictionaries
            
        Returns:
            Similarity score (0-1)
        """
        similarity_scores = []
        
        # 1. Check for common tokens with similar balances
        cluster1_addresses = [w.address for w in cluster1.wallets]
        cluster2_addresses = [w.address for w in cluster2.wallets]
        
        cluster1_balances = [b for b in balances if b.get("address") in cluster1_addresses]
        cluster2_balances = [b for b in balances if b.get("address") in cluster2_addresses]
        
        # Group balances by token address
        cluster1_tokens = {}
        for balance in cluster1_balances:
            token_address = balance.get("token_address", "")
            if token_address:
                if token_address not in cluster1_tokens:
                    cluster1_tokens[token_address] = []
                cluster1_tokens[token_address].append(balance)
        
        cluster2_tokens = {}
        for balance in cluster2_balances:
            token_address = balance.get("token_address", "")
            if token_address:
                if token_address not in cluster2_tokens:
                    cluster2_tokens[token_address] = []
                cluster2_tokens[token_address].append(balance)
        
        # Find common tokens
        common_tokens = set(cluster1_tokens.keys()) & set(cluster2_tokens.keys())
        
        if common_tokens:
            token_similarity = len(common_tokens) / max(len(cluster1_tokens), len(cluster2_tokens))
            similarity_scores.append(token_similarity)
        
        # 2. Check for similar transaction patterns
        cluster1_activities = [a for a in activities if a.get("wallet_address") in cluster1_addresses]
        cluster2_activities = [a for a in activities if a.get("wallet_address") in cluster2_addresses]
        
        # Group activities by hour
        cluster1_hourly = self._group_activities_by_hour(cluster1_activities)
        cluster2_hourly = self._group_activities_by_hour(cluster2_activities)
        
        # Calculate hourly pattern similarity
        if cluster1_hourly and cluster2_hourly:
            all_hours = set(cluster1_hourly.keys()) | set(cluster2_hourly.keys())
            
            if all_hours:
                hour_similarity = sum(
                    1 for hour in all_hours 
                    if hour in cluster1_hourly and hour in cluster2_hourly
                ) / len(all_hours)
                
                similarity_scores.append(hour_similarity)
        
        # 3. Check for similar counterparties
        cluster1_counterparties = self._extract_counterparties(cluster1_activities)
        cluster2_counterparties = self._extract_counterparties(cluster2_activities)
        
        common_counterparties = cluster1_counterparties & cluster2_counterparties
        
        if common_counterparties:
            counterparty_similarity = len(common_counterparties) / max(len(cluster1_counterparties), len(cluster2_counterparties))
            similarity_scores.append(counterparty_similarity)
        
        # Calculate overall similarity
        if similarity_scores:
            # Weight the scores (token similarity is most important)
            weights = [0.5, 0.3, 0.2]  # Adjust weights as needed
            
            # Pad weights if needed
            if len(weights) > len(similarity_scores):
                weights = weights[:len(similarity_scores)]
            else:
                weights.extend([0.1] * (len(similarity_scores) - len(weights)))
                
            # Normalize weights
            total_weight = sum(weights)
            if total_weight > 0:
                weights = [w / total_weight for w in weights]
            
            # Calculate weighted average
            overall_similarity = sum(score * weight for score, weight in zip(similarity_scores, weights))
            
            return overall_similarity
        
        return 0.0
    
    def _group_activities_by_hour(self, activities: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Group activities by hour of day to find patterns.
        
        Args:
            activities: List of activity dictionaries
            
        Returns:
            Dictionary mapping hour to activity count
        """
        hourly_counts = {}
        
        for activity in activities:
            if "block_time" in activity:
                timestamp = activity["block_time"]
                dt = datetime.fromtimestamp(timestamp)
                hour = dt.hour
                
                if hour not in hourly_counts:
                    hourly_counts[hour] = 0
                
                hourly_counts[hour] += 1
        
        return hourly_counts
    
    def _extract_counterparties(self, activities: List[Dict[str, Any]]) -> Set[str]:
        """
        Extract counterparty addresses from activities.
        
        Args:
            activities: List of activity dictionaries
            
        Returns:
            Set of counterparty addresses
        """
        counterparties = set()
        
        for activity in activities:
            wallet_address = activity.get("wallet_address", "").lower()
            from_address = activity.get("from", "").lower()
            to_address = activity.get("to", "").lower()
            
            if from_address and from_address != wallet_address:
                counterparties.add(from_address)
            
            if to_address and to_address != wallet_address:
                counterparties.add(to_address)
        
        return counterparties
    
    def _merge_cluster_group(self, clusters: List[IdentityCluster]) -> IdentityCluster:
        """
        Merge a group of clusters into a single cluster.
        
        Args:
            clusters: List of clusters to merge
            
        Returns:
            Merged identity cluster
        """
        if not clusters:
            raise ValueError("Cannot merge empty cluster list")
        
        if len(clusters) == 1:
            return clusters[0]
        
        # Find the main cluster (highest total value)
        main_cluster = max(clusters, key=lambda c: c.total_value_usd)
        
        # Create a new merged cluster
        merged_cluster = IdentityCluster(
            cluster_id=str(uuid.uuid4()),
            wallets=[],
            main_address=main_cluster.main_address,
            total_value_usd=0,
            chains=[],
            first_seen=None,
            last_seen=None,
            cross_chain_movements=[],
            confidence=1.0,
            risk_score=0,
            risk_factors=[],
            coordination_patterns=[]
        )
        
        # Merge data from all clusters
        for cluster in clusters:
            # Merge wallets
            for wallet in cluster.wallets:
                if wallet not in merged_cluster.wallets:
                    merged_cluster.wallets.append(wallet)
            
            # Merge chains
            for chain in cluster.chains:
                if chain not in merged_cluster.chains:
                    merged_cluster.chains.append(chain)
            
            # Merge cross-chain movements
            for movement_id in cluster.cross_chain_movements:
                if movement_id not in merged_cluster.cross_chain_movements:
                    merged_cluster.cross_chain_movements.append(movement_id)
            
            # Merge coordination patterns
            for pattern in cluster.coordination_patterns:
                if pattern not in merged_cluster.coordination_patterns:
                    merged_cluster.coordination_patterns.append(pattern)
            
            # Update total value
            merged_cluster.total_value_usd += cluster.total_value_usd
            
            # Update first/last seen
            if cluster.first_seen and (not merged_cluster.first_seen or cluster.first_seen < merged_cluster.first_seen):
                merged_cluster.first_seen = cluster.first_seen
            
            if cluster.last_seen and (not merged_cluster.last_seen or cluster.last_seen > merged_cluster.last_seen):
                merged_cluster.last_seen = cluster.last_seen
            
            # Update confidence (take the minimum)
            merged_cluster.confidence = min(merged_cluster.confidence, cluster.confidence * 0.9)  # Slightly reduce confidence for merged clusters
        
        return merged_cluster
    
    def _analyze_bridge_usage(self, cross_chain_movements: List[CrossChainMovement]) -> List[BridgeUsage]:
        """
        Analyze bridge usage patterns from cross-chain movements.
        
        Args:
            cross_chain_movements: List of cross-chain movement objects
            
        Returns:
            List of bridge usage analysis objects
        """
        if not cross_chain_movements:
            return []
        
        # Group movements by bridge
        bridge_movements = {}
        
        for movement in cross_chain_movements:
            bridge_key = f"{movement.source_chain_id}_{movement.destination_chain_id}"
            
            if movement.bridge_address:
                bridge_key = f"{bridge_key}_{movement.bridge_address}"
            
            if bridge_key not in bridge_movements:
                bridge_movements[bridge_key] = []
            
            bridge_movements[bridge_key].append(movement)
        
        # Analyze each bridge
        bridge_usage_list = []
        
        for bridge_key, movements in bridge_movements.items():
            # Extract bridge details
            sample_movement = movements[0]
            bridge_address = sample_movement.bridge_address or "unknown"
            bridge_name = sample_movement.bridge_name
            source_chain_id = sample_movement.source_chain_id
            destination_chain_id = sample_movement.destination_chain_id
            
            # Calculate usage statistics
            usage_count = len(movements)
            total_value_usd = sum(m.value_usd for m in movements)
            average_value_usd = total_value_usd / usage_count if usage_count > 0 else 0
            
            # Find first/last usage
            source_times = [datetime.fromisoformat(m.source_time) for m in movements]
            first_used = min(source_times).isoformat() if source_times else None
            last_used = max(source_times).isoformat() if source_times else None
            
            # Calculate risk score
            risk_score = 0
            risk_factors = []
            
            # Unknown bridge is higher risk
            if not bridge_name:
                risk_score += 30
                risk_factors.append("Unknown bridge")
            
            # High usage count is higher risk
            if usage_count > 10:
                risk_score += 20
                risk_factors.append(f"High usage frequency: {usage_count} times")
            elif usage_count > 5:
                risk_score += 10
            
            # High average value is higher risk
            if average_value_usd > 50000:
                risk_score += 20
                risk_factors.append(f"High average value: ${average_value_usd:.2f}")
            elif average_value_usd > 10000:
                risk_score += 10
            
            # Create bridge usage object
            bridge_usage = BridgeUsage(
                bridge_address=bridge_address,
                bridge_name=bridge_name,
                source_chain_id=source_chain_id,
                destination_chain_id=destination_chain_id,
                usage_count=usage_count,
                total_value_usd=total_value_usd,
                average_value_usd=average_value_usd,
                first_used=first_used,
                last_used=last_used,
                risk_score=min(100, risk_score),
                risk_factors=risk_factors
            )
            
            bridge_usage_list.append(bridge_usage)
        
        return bridge_usage_list
    
    def _detect_coordination_patterns(
        self,
        identity_clusters: List[IdentityCluster],
        cross_chain_movements: List[CrossChainMovement],
        activities: List[Dict[str, Any]]
    ) -> None:
        """
        Detect coordination patterns across chains.
        
        Args:
            identity_clusters: List of identity clusters
            cross_chain_movements: List of cross-chain movement objects
            activities: List of activity dictionaries
        """
        for cluster in identity_clusters:
            # Skip clusters with only one wallet
            if len(cluster.wallets) <= 1:
                continue
            
            # Get all addresses in this cluster
            cluster_addresses = [w.address.lower() for w in cluster.wallets]
            
            # Get activities for this cluster
            cluster_activities = [a for a in activities if a.get("wallet_address", "").lower() in cluster_addresses]
            
            # Sort activities by time
            sorted_activities = sorted(cluster_activities, key=lambda x: x.get("block_time", 0))
            
            # Check for coordination patterns
            
            # 1. Check for parallel transactions (same action across multiple chains within a short time)
            parallel_txs = self._detect_parallel_transactions(sorted_activities, cluster_addresses)
            if parallel_txs:
                cluster.coordination_patterns.append("PARALLEL_TRANSACTIONS")
            
            # 2. Check for sequential chain hopping
            chain_hopping = self._detect_chain_hopping(sorted_activities, cluster_addresses)
            if chain_hopping:
                cluster.coordination_patterns.append("CHAIN_HOPPING")
            
            # 3. Check for circular flows
            circular_flows = self._detect_circular_flows(sorted_activities, cluster_addresses)
            if circular_flows:
                cluster.coordination_patterns.append("CIRCULAR_FLOWS")
    
    def _detect_parallel_transactions(
        self,
        activities: List[Dict[str, Any]],
        addresses: List[str]
    ) -> bool:
        """
        Detect parallel transactions across multiple chains.
        
        Args:
            activities: List of activity dictionaries
            addresses: List of addresses to consider
            
        Returns:
            True if parallel transactions detected, False otherwise
        """
        if len(activities) < 2:
            return False
        
        # Group activities by time window (5 minute windows)
        time_windows = {}
        
        for activity in activities:
            block_time = activity.get("block_time", 0)
            window = block_time // 300  # 5 minute windows
            
            if window not in time_windows:
                time_windows[window] = []
            
            time_windows[window].append(activity)
        
        # Check each time window for parallel transactions
        for window, window_activities in time_windows.items():
            # Skip windows with only one activity
            if len(window_activities) < 2:
                continue
            
            # Group by chain
            chains = {}
            
            for activity in window_activities:
                chain_id = str(activity.get("chain_id", ""))
                
                if chain_id not in chains:
                    chains[chain_id] = []
                
                chains[chain_id].append(activity)
            
            # If activities on multiple chains in the same window, consider it parallel
            if len(chains) >= 2:
                return True
        
        return False
    
    def _detect_chain_hopping(
        self,
        activities: List[Dict[str, Any]],
        addresses: List[str]
    ) -> bool:
        """
        Detect sequential chain hopping.
        
        Args:
            activities: List of activity dictionaries
            addresses: List of addresses to consider
            
        Returns:
            True if chain hopping detected, False otherwise
        """
        if len(activities) < 3:
            return False
        
        # Track chain sequence
        chain_sequence = []
        
        for activity in activities:
            chain_id = str(activity.get("chain_id", ""))
            
            if chain_id:
                chain_sequence.append(chain_id)
        
        # Check for at least 3 different chains in sequence
        unique_chains = set()
        chain_changes = 0
        
        for i in range(1, len(chain_sequence)):
            if chain_sequence[i] != chain_sequence[i-1]:
                chain_changes += 1
                unique_chains.add(chain_sequence[i])
                unique_chains.add(chain_sequence[i-1])
        
        # If at least 3 different chains and at least 2 chain changes, consider it chain hopping
        return len(unique_chains) >= 3 and chain_changes >= 2
    
    def _detect_circular_flows(
        self,
        activities: List[Dict[str, Any]],
        addresses: List[str]
    ) -> bool:
        """
        Detect circular flows across chains.
        
        Args:
            activities: List of activity dictionaries
            addresses: List of addresses to consider
            
        Returns:
            True if circular flows detected, False otherwise
        """
        if len(activities) < 3:
            return False
        
        # Build a directed graph of value flows
        G = nx.DiGraph()
        
        for activity in activities:
            if activity.get("type") in ["send", "receive"]:
                from_address = activity.get("from", "").lower()
                to_address = activity.get("to", "").lower()
                
                if from_address and to_address:
                    G.add_edge(from_address, to_address)
        
        # Check for cycles
        try:
            cycles = list(nx.simple_cycles(G))
            
            # If any cycle involves at least one address from our cluster, consider it a circular flow
            for cycle in cycles:
                if any(address.lower() in cycle for address in addresses):
                    return True
                
            return False
            
        except:
            # nx.simple_cycles can raise an exception for large graphs
            return False
    
    def _calculate_cluster_risk(
        self,
        cluster: IdentityCluster,
        cross_chain_movements: List[CrossChainMovement]
    ) -> Tuple[float, List[str]]:
        """
        Calculate risk score for an identity cluster.
        
        Args:
            cluster: Identity cluster
            cross_chain_movements: List of cross-chain movement objects
            
        Returns:
            Tuple of (risk_score, risk_factors)
        """
        risk_score = 0
        risk_factors = []
        
        # 1. Check number of chains
        if len(cluster.chains) > 5:
            risk_score += 20
            risk_factors.append(f"Active on {len(cluster.chains)} chains")
        elif len(cluster.chains) > 3:
            risk_score += 10
        
        # 2. Check total value
        if cluster.total_value_usd > 1000000:
            risk_score += 20
            risk_factors.append(f"High total value: ${cluster.total_value_usd:.2f}")
        elif cluster.total_value_usd > 100000:
            risk_score += 10
        
        # 3. Check cross-chain movements
        cluster_movements = [
            m for m in cross_chain_movements 
            if m.id in cluster.cross_chain_movements
        ]
        
        if len(cluster_movements) > 10:
            risk_score += 20
            risk_factors.append(f"High number of cross-chain movements: {len(cluster_movements)}")
        elif len(cluster_movements) > 5:
            risk_score += 10
        
        # 4. Check coordination patterns
        if "CIRCULAR_FLOWS" in cluster.coordination_patterns:
            risk_score += 30
            risk_factors.append("Circular flow pattern detected")
        
        if "CHAIN_HOPPING" in cluster.coordination_patterns:
            risk_score += 20
            risk_factors.append("Chain hopping pattern detected")
        
        if "PARALLEL_TRANSACTIONS" in cluster.coordination_patterns:
            risk_score += 10
            risk_factors.append("Parallel transaction pattern detected")
        
        # 5. Check confidence (lower confidence = higher risk)
        confidence_factor = (1 - cluster.confidence) * 20
        risk_score += confidence_factor
        
        # Ensure risk score is between 0 and 100
        risk_score = min(100, max(0, risk_score))
        
        return risk_score, risk_factors
    
    def _calculate_overall_risk(
        self,
        wallets: List[CrossChainWallet],
        cross_chain_movements: List[CrossChainMovement],
        identity_clusters: List[IdentityCluster],
        bridge_usage: List[BridgeUsage]
    ) -> Tuple[float, List[str]]:
        """
        Calculate overall risk score for the cross-chain identity analysis.
        
        Args:
            wallets: List of wallet objects
            cross_chain_movements: List of cross-chain movement objects
            identity_clusters: List of identity clusters
            bridge_usage: List of bridge usage objects
            
        Returns:
            Tuple of (risk_score, risk_factors)
        """
        risk_score = 0
        risk_factors = []
        
        # 1. Check cross-chain movements
        if cross_chain_movements:
            # Average risk score of movements
            movement_risk = sum(m.risk_score for m in cross_chain_movements) / len(cross_chain_movements)
            risk_score += movement_risk * 0.3  # 30% weight
            
            # High risk movements
            high_risk_movements = [m for m in cross_chain_movements if m.risk_score >= 70]
            if high_risk_movements:
                risk_factors.append(f"High-risk cross-chain movements detected: {len(high_risk_movements)}")
        
        # 2. Check identity clusters
        if identity_clusters:
            # Average risk score of clusters
            cluster_risk = sum(c.risk_score for c in identity_clusters) / len(identity_clusters)
            risk_score += cluster_risk * 0.4  # 40% weight
            
            # High risk clusters
            high_risk_clusters = [c for c in identity_clusters if c.risk_score >= 70]
            if high_risk_clusters:
                risk_factors.append(f"High-risk identity clusters detected: {len(high_risk_clusters)}")
            
            # Multi-chain clusters
            multi_chain_clusters = [c for c in identity_clusters if len(c.chains) >= 3]
            if multi_chain_clusters:
                risk_factors.append(f"Multi-chain identity clusters detected: {len(multi_chain_clusters)}")
        
        # 3. Check bridge usage
        if bridge_usage:
            # Average risk score of bridge usage
            bridge_risk = sum(b.risk_score for b in bridge_usage) / len(bridge_usage)
            risk_score += bridge_risk * 0.3  # 30% weight
            
            # High risk bridge usage
            high_risk_bridges = [b for b in bridge_usage if b.risk_score >= 70]
            if high_risk_bridges:
                risk_factors.append(f"High-risk bridge usage detected: {len(high_risk_bridges)}")
            
            # Unknown bridges
            unknown_bridges = [b for b in bridge_usage if not b.bridge_name]
            if unknown_bridges:
                risk_factors.append(f"Unknown bridges used: {len(unknown_bridges)}")
        
        # Ensure risk score is between 0 and 100
        risk_score = min(100, max(0, risk_score))
        
        # Add default risk factor if none found
        if not risk_factors:
            if risk_score >= 70:
                risk_factors.append("High overall cross-chain risk profile")
            elif risk_score >= 40:
                risk_factors.append("Moderate cross-chain risk profile")
            else:
                risk_factors.append("Low cross-chain risk profile")
        
        return risk_score, risk_factors
    
    def _emit_graph_events(
        self,
        wallets: List[CrossChainWallet],
        cross_chain_movements: List[CrossChainMovement],
        identity_clusters: List[IdentityCluster],
        bridge_usage: List[BridgeUsage]
    ) -> None:
        """
        Emit graph events for Neo4j integration.
        
        Args:
            wallets: List of wallet objects
            cross_chain_movements: List of cross-chain movement objects
            identity_clusters: List of identity clusters
            bridge_usage: List of bridge usage objects
        """
        try:
            # Prepare data for Neo4j
            graph_data = {
                "wallets": [wallet.dict() for wallet in wallets],
                "cross_chain_movements": [movement.dict() for movement in cross_chain_movements],
                "identity_clusters": [cluster.dict() for cluster in identity_clusters],
                "bridge_usage": [bridge.dict() for bridge in bridge_usage],
                "timestamp": int(time.time())
            }
            
            # Emit graph event
            emit_event(
                GraphAddEvent(
                    type="cross_chain_identity",
                    data=graph_data
                )
            )
            
            logger.debug("Emitted graph events for cross-chain identity analysis")
            
        except Exception as e:
            logger.error(f"Failed to emit graph events: {str(e)}")
            # Don't re-raise, as this is a non-critical operation
