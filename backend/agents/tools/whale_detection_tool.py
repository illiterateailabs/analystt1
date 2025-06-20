"""
Whale Detection Tool

This tool identifies and tracks cryptocurrency "whales" - wallets with large holdings
or significant transaction activity. It uses Sim APIs to detect large transactions,
monitor known whale wallets, and identify coordinated movement patterns.

The tool provides real-time whale movement monitoring and classification based on
configurable thresholds for portfolio value and transaction activity.
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple
from pydantic import BaseModel, Field, validator

from crewai_tools import BaseTool
from backend.integrations.sim_client import SimClient
from backend.core.metrics import record_tool_usage, record_tool_error
from backend.core.events import emit_event, GraphAddEvent

logger = logging.getLogger(__name__)

# Constants for whale classification
TIER1_WHALE_THRESHOLD = 10_000_000  # $10M+ portfolio value
TIER2_WHALE_THRESHOLD = 1_000_000   # $1M+ portfolio value
ACTIVE_WHALE_TX_THRESHOLD = 100_000  # $100k+ in transactions
LARGE_TX_THRESHOLD = 100_000        # $100k+ single transaction
COORDINATED_MOVEMENT_WINDOW = 3600  # 1 hour window for coordination detection
WHALE_MONITORING_LOOKBACK = 7       # 7 days of transaction history


class WhaleWallet(BaseModel):
    """Model representing a whale wallet with classification and metrics."""
    
    address: str = Field(..., description="Wallet address")
    tier: str = Field(..., description="Whale tier classification (TIER1, TIER2, ACTIVE)")
    total_value_usd: float = Field(..., description="Total portfolio value in USD")
    last_active: Optional[str] = Field(None, description="Timestamp of last activity")
    large_transactions: int = Field(0, description="Count of large transactions")
    chains: List[str] = Field(default_factory=list, description="Active chains")
    tokens: Dict[str, float] = Field(default_factory=dict, description="Top token holdings by value")
    connected_wallets: List[str] = Field(default_factory=list, description="Potentially connected wallets")
    risk_score: Optional[float] = Field(None, description="Risk score if available")
    first_seen: str = Field(..., description="Timestamp when first detected as whale")


class WhaleMovement(BaseModel):
    """Model representing a significant whale movement/transaction."""
    
    transaction_hash: str = Field(..., description="Transaction hash")
    from_address: str = Field(..., description="Sender address")
    to_address: str = Field(..., description="Recipient address")
    value_usd: float = Field(..., description="Transaction value in USD")
    timestamp: str = Field(..., description="Transaction timestamp")
    chain: str = Field(..., description="Blockchain network")
    token_address: Optional[str] = Field(None, description="Token address if not native")
    token_symbol: Optional[str] = Field(None, description="Token symbol")
    movement_type: str = Field(..., description="Type of movement (SEND, RECEIVE, SWAP, etc.)")
    is_coordinated: bool = Field(False, description="Whether this appears to be part of coordinated movement")
    coordination_group: Optional[str] = Field(None, description="ID of coordination group if applicable")


class CoordinationGroup(BaseModel):
    """Model representing a group of coordinated whale movements."""
    
    group_id: str = Field(..., description="Unique identifier for the coordination group")
    wallets: List[str] = Field(..., description="List of wallet addresses involved")
    start_time: str = Field(..., description="Start time of coordination window")
    end_time: str = Field(..., description="End time of coordination window")
    total_value_usd: float = Field(..., description="Total USD value of coordinated movements")
    movement_count: int = Field(..., description="Number of transactions in the group")
    pattern_type: str = Field(..., description="Detected pattern type (DISTRIBUTION, ACCUMULATION, CIRCULAR)")
    confidence: float = Field(..., description="Confidence score for the coordination detection")


class WhaleDetectionInput(BaseModel):
    """Input parameters for whale detection."""
    
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


class WhaleDetectionTool(BaseTool):
    """
    Tool for detecting and tracking cryptocurrency whale activity.
    
    This tool identifies wallets with large holdings (whales), tracks their
    movements, detects large transactions, and identifies potentially
    coordinated activities across multiple wallets. It leverages Sim APIs
    to monitor on-chain activity in real-time across multiple blockchains.
    """
    
    name: str = "whale_detection_tool"
    description: str = """
    Detects and tracks cryptocurrency whale activity across multiple blockchains.
    Identifies large wallets, monitors significant transactions, and detects
    coordinated movements between whales. Useful for market analysis, fraud
    detection, and understanding large capital flows in the crypto ecosystem.
    """
    
    sim_client: SimClient
    _known_whales: Dict[str, WhaleWallet] = {}
    _recent_large_movements: List[WhaleMovement] = []
    _coordination_groups: Dict[str, CoordinationGroup] = {}
    
    def __init__(self, sim_client: Optional[SimClient] = None):
        """
        Initialize the WhaleDetectionTool.
        
        Args:
            sim_client: Optional SimClient instance. If not provided, a new one will be created.
        """
        super().__init__()
        self.sim_client = sim_client or SimClient()
    
    async def _execute(
        self,
        wallet_address: Optional[str] = None,
        lookback_days: int = WHALE_MONITORING_LOOKBACK,
        tier1_threshold: float = TIER1_WHALE_THRESHOLD,
        tier2_threshold: float = TIER2_WHALE_THRESHOLD,
        tx_threshold: float = LARGE_TX_THRESHOLD,
        detect_coordination: bool = True,
        chain_ids: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute whale detection analysis.
        
        Args:
            wallet_address: Optional specific wallet to analyze
            lookback_days: Days of history to analyze
            tier1_threshold: USD threshold for Tier 1 whale classification
            tier2_threshold: USD threshold for Tier 2 whale classification
            tx_threshold: USD threshold for large transaction detection
            detect_coordination: Whether to detect coordination between whales
            chain_ids: Optional comma-separated list of chain IDs to monitor
            
        Returns:
            Dictionary containing detected whales, movements, and coordination groups
            
        Raises:
            Exception: If there's an error during whale detection
        """
        try:
            # Record tool usage for metrics
            record_tool_usage(self.name)
            
            results = {
                "whales": [],
                "movements": [],
                "coordination_groups": [],
                "stats": {
                    "total_whales_detected": 0,
                    "new_whales_detected": 0,
                    "large_movements_detected": 0,
                    "coordination_groups_detected": 0,
                    "total_value_monitored_usd": 0,
                }
            }
            
            # If specific wallet provided, analyze just that wallet
            if wallet_address:
                whale_data = await self._analyze_wallet_for_whale_activity(
                    wallet_address, 
                    tier1_threshold,
                    tier2_threshold,
                    tx_threshold,
                    lookback_days,
                    chain_ids
                )
                
                if whale_data:
                    results["whales"].append(whale_data.dict())
                    results["stats"]["total_whales_detected"] = 1
                    results["stats"]["new_whales_detected"] = 1 if wallet_address not in self._known_whales else 0
                    
                    # Get recent movements for this whale
                    movements = await self._get_whale_movements(
                        wallet_address,
                        lookback_days,
                        tx_threshold,
                        chain_ids
                    )
                    
                    results["movements"] = [m.dict() for m in movements]
                    results["stats"]["large_movements_detected"] = len(movements)
                    results["stats"]["total_value_monitored_usd"] = whale_data.total_value_usd
            
            # Otherwise, scan for whales and their movements
            else:
                # This would typically involve more complex scanning logic
                # For now, we'll implement a simplified version that checks known whales
                # and looks for new ones based on large transactions
                
                # 1. Check for large transactions to identify potential new whales
                new_whale_candidates = await self._scan_for_large_transactions(
                    lookback_days,
                    tx_threshold,
                    chain_ids
                )
                
                # 2. Analyze each candidate to confirm whale status
                confirmed_whales = []
                for candidate in new_whale_candidates:
                    whale_data = await self._analyze_wallet_for_whale_activity(
                        candidate,
                        tier1_threshold,
                        tier2_threshold,
                        tx_threshold,
                        lookback_days,
                        chain_ids
                    )
                    
                    if whale_data:
                        confirmed_whales.append(whale_data)
                
                # 3. Update known whales list and get their movements
                all_movements = []
                total_value = 0
                
                for whale in confirmed_whales:
                    self._known_whales[whale.address] = whale
                    movements = await self._get_whale_movements(
                        whale.address,
                        lookback_days,
                        tx_threshold,
                        chain_ids
                    )
                    all_movements.extend(movements)
                    total_value += whale.total_value_usd
                
                # 4. Detect coordination if enabled
                coordination_groups = []
                if detect_coordination and len(confirmed_whales) > 1:
                    coordination_groups = self._detect_coordination_patterns(all_movements)
                
                # 5. Populate results
                results["whales"] = [w.dict() for w in confirmed_whales]
                results["movements"] = [m.dict() for m in all_movements]
                results["coordination_groups"] = [g.dict() for g in coordination_groups]
                
                results["stats"]["total_whales_detected"] = len(confirmed_whales)
                results["stats"]["new_whales_detected"] = sum(1 for w in confirmed_whales if w.address not in self._known_whales)
                results["stats"]["large_movements_detected"] = len(all_movements)
                results["stats"]["coordination_groups_detected"] = len(coordination_groups)
                results["stats"]["total_value_monitored_usd"] = total_value
            
            # Emit graph events for Neo4j integration
            if results["whales"]:
                self._emit_graph_events(results)
            
            return results
            
        except Exception as e:
            error_msg = f"Error in whale detection: {str(e)}"
            logger.error(error_msg)
            record_tool_error(self.name, str(e))
            
            return {
                "error": error_msg,
                "whales": [],
                "movements": [],
                "coordination_groups": [],
                "stats": {
                    "total_whales_detected": 0,
                    "new_whales_detected": 0,
                    "large_movements_detected": 0,
                    "coordination_groups_detected": 0,
                    "total_value_monitored_usd": 0,
                }
            }
    
    async def _analyze_wallet_for_whale_activity(
        self,
        wallet_address: str,
        tier1_threshold: float,
        tier2_threshold: float,
        tx_threshold: float,
        lookback_days: int,
        chain_ids: Optional[str]
    ) -> Optional[WhaleWallet]:
        """
        Analyze a wallet to determine if it qualifies as a whale.
        
        Args:
            wallet_address: The wallet address to analyze
            tier1_threshold: USD threshold for Tier 1 classification
            tier2_threshold: USD threshold for Tier 2 classification
            tx_threshold: USD threshold for large transaction detection
            lookback_days: Days of history to analyze
            chain_ids: Optional comma-separated list of chain IDs
            
        Returns:
            WhaleWallet object if the wallet qualifies as a whale, None otherwise
        """
        try:
            # 1. Get wallet balances
            balances_response = await self.sim_client.get_balances(
                wallet_address,
                limit=100,  # Get up to 100 token balances
                chain_ids=chain_ids or "all",
                metadata="url,logo"
            )
            
            balances = balances_response.get("balances", [])
            
            # 2. Calculate total portfolio value
            total_value_usd = 0
            tokens = {}
            chains = set()
            
            for balance in balances:
                value_usd = float(balance.get("value_usd", 0))
                total_value_usd += value_usd
                
                # Track top tokens by value
                if value_usd > 0:
                    symbol = balance.get("symbol", "UNKNOWN")
                    tokens[symbol] = value_usd
                    
                    # Track active chains
                    chain = balance.get("chain", "unknown")
                    chains.add(chain)
            
            # 3. Get recent activity to check for large transactions
            activity_response = await self.sim_client.get_activity(
                wallet_address,
                limit=50  # Get up to 50 recent activities
            )
            
            activities = activity_response.get("activity", [])
            
            # 4. Count large transactions and find last active timestamp
            large_tx_count = 0
            last_active = None
            
            for activity in activities:
                # Check if this is a recent transaction (within lookback period)
                if "block_time" in activity:
                    block_time = activity["block_time"]
                    tx_time = datetime.fromtimestamp(block_time)
                    now = datetime.now()
                    
                    # Update last active time
                    if last_active is None or tx_time > datetime.fromisoformat(last_active):
                        last_active = tx_time.isoformat()
                    
                    # Only count recent transactions for large tx detection
                    if (now - tx_time).days <= lookback_days:
                        # Check if this is a large transaction
                        value_usd = float(activity.get("value_usd", 0))
                        if value_usd >= tx_threshold:
                            large_tx_count += 1
            
            # 5. Determine whale classification
            if total_value_usd >= tier1_threshold:
                tier = "TIER1"
            elif total_value_usd >= tier2_threshold:
                tier = "TIER2"
            elif large_tx_count > 0:
                tier = "ACTIVE"
            else:
                # Not a whale by our criteria
                return None
            
            # 6. Create and return WhaleWallet object
            whale = WhaleWallet(
                address=wallet_address,
                tier=tier,
                total_value_usd=total_value_usd,
                last_active=last_active,
                large_transactions=large_tx_count,
                chains=list(chains),
                tokens=dict(sorted(tokens.items(), key=lambda x: x[1], reverse=True)[:10]),  # Top 10 tokens
                connected_wallets=[],  # Will be populated in coordination detection
                first_seen=datetime.now().isoformat()
            )
            
            # 7. If this is a known whale, preserve first_seen timestamp
            if wallet_address in self._known_whales:
                whale.first_seen = self._known_whales[wallet_address].first_seen
            
            return whale
            
        except Exception as e:
            logger.error(f"Error analyzing wallet {wallet_address} for whale activity: {str(e)}")
            return None
    
    async def _get_whale_movements(
        self,
        wallet_address: str,
        lookback_days: int,
        tx_threshold: float,
        chain_ids: Optional[str]
    ) -> List[WhaleMovement]:
        """
        Get significant movements for a whale wallet.
        
        Args:
            wallet_address: The whale wallet address
            lookback_days: Days of history to analyze
            tx_threshold: USD threshold for large transaction detection
            chain_ids: Optional comma-separated list of chain IDs
            
        Returns:
            List of WhaleMovement objects representing significant transactions
        """
        try:
            # Get wallet activity
            activity_response = await self.sim_client.get_activity(
                wallet_address,
                limit=100  # Get up to 100 recent activities
            )
            
            activities = activity_response.get("activity", [])
            movements = []
            
            # Calculate lookback timestamp
            lookback_time = datetime.now() - timedelta(days=lookback_days)
            lookback_timestamp = lookback_time.timestamp()
            
            # Process activities to find significant movements
            for activity in activities:
                # Skip if outside lookback period
                if "block_time" not in activity or activity["block_time"] < lookback_timestamp:
                    continue
                
                # Check if this is a significant transaction
                value_usd = float(activity.get("value_usd", 0))
                if value_usd < tx_threshold:
                    continue
                
                # Create WhaleMovement object
                movement = WhaleMovement(
                    transaction_hash=activity.get("transaction_hash", "unknown"),
                    from_address=activity.get("from", wallet_address),
                    to_address=activity.get("to", "unknown"),
                    value_usd=value_usd,
                    timestamp=datetime.fromtimestamp(activity["block_time"]).isoformat(),
                    chain=activity.get("chain", "unknown"),
                    token_address=activity.get("address", None),
                    token_symbol=activity.get("symbol", None),
                    movement_type=activity.get("type", "UNKNOWN").upper(),
                    is_coordinated=False,  # Will be updated in coordination detection
                    coordination_group=None
                )
                
                movements.append(movement)
                
                # Add to recent large movements cache for coordination detection
                self._recent_large_movements.append(movement)
                
                # Trim cache if it gets too large (keep last 1000 movements)
                if len(self._recent_large_movements) > 1000:
                    self._recent_large_movements = self._recent_large_movements[-1000:]
            
            return movements
            
        except Exception as e:
            logger.error(f"Error getting whale movements for {wallet_address}: {str(e)}")
            return []
    
    async def _scan_for_large_transactions(
        self,
        lookback_days: int,
        tx_threshold: float,
        chain_ids: Optional[str]
    ) -> List[str]:
        """
        Scan for large transactions to identify potential new whales.
        
        This is a simplified implementation. In a production environment,
        this would typically involve more sophisticated scanning techniques
        or integration with external data sources.
        
        Args:
            lookback_days: Days of history to analyze
            tx_threshold: USD threshold for large transaction detection
            chain_ids: Optional comma-separated list of chain IDs
            
        Returns:
            List of wallet addresses that might be whales
        """
        # In a real implementation, this would scan transaction data
        # For now, we'll return known whales plus any from recent movements
        potential_whales = set(self._known_whales.keys())
        
        # Add addresses from recent large movements
        for movement in self._recent_large_movements:
            potential_whales.add(movement.from_address)
            potential_whales.add(movement.to_address)
        
        # In a production system, we would add additional scanning logic here
        # such as querying for top token holders or recent large transactions
        
        return list(potential_whales)
    
    def _detect_coordination_patterns(self, movements: List[WhaleMovement]) -> List[CoordinationGroup]:
        """
        Detect coordination patterns between whale wallets.
        
        Args:
            movements: List of whale movements to analyze
            
        Returns:
            List of detected coordination groups
        """
        if not movements or len(movements) < 2:
            return []
        
        coordination_groups = []
        processed_txs = set()
        
        # Sort movements by timestamp
        sorted_movements = sorted(movements, key=lambda m: m.timestamp)
        
        # Look for temporal clusters of movements
        for i, movement in enumerate(sorted_movements):
            # Skip if already processed as part of a group
            if movement.transaction_hash in processed_txs:
                continue
            
            # Define time window for potential coordination
            movement_time = datetime.fromisoformat(movement.timestamp)
            window_end = movement_time + timedelta(seconds=COORDINATED_MOVEMENT_WINDOW)
            
            # Find other movements in the same time window
            window_movements = [movement]
            for j in range(i + 1, len(sorted_movements)):
                next_movement = sorted_movements[j]
                next_time = datetime.fromisoformat(next_movement.timestamp)
                
                if next_time <= window_end:
                    window_movements.append(next_movement)
                else:
                    break
            
            # If we have multiple movements in the window, check for patterns
            if len(window_movements) > 1:
                # Look for patterns like:
                # 1. Multiple sends from same address (DISTRIBUTION)
                # 2. Multiple receives to same address (ACCUMULATION)
                # 3. Circular transfers (CIRCULAR)
                
                # Group by sender and recipient
                senders = {}
                recipients = {}
                wallet_connections = {}
                
                for m in window_movements:
                    # Track senders
                    if m.from_address not in senders:
                        senders[m.from_address] = []
                    senders[m.from_address].append(m)
                    
                    # Track recipients
                    if m.to_address not in recipients:
                        recipients[m.to_address] = []
                    recipients[m.to_address].append(m)
                    
                    # Track wallet connections
                    key = f"{m.from_address}->{m.to_address}"
                    if key not in wallet_connections:
                        wallet_connections[key] = []
                    wallet_connections[key].append(m)
                
                # Detect distribution pattern (one sender, multiple recipients)
                for sender, sender_movements in senders.items():
                    if len(sender_movements) >= 3:  # At least 3 outgoing transactions
                        unique_recipients = set(m.to_address for m in sender_movements)
                        if len(unique_recipients) >= 3:  # To at least 3 different addresses
                            # This looks like a distribution pattern
                            group = self._create_coordination_group(
                                sender_movements,
                                "DISTRIBUTION",
                                0.8 if len(sender_movements) >= 5 else 0.6
                            )
                            coordination_groups.append(group)
                            
                            # Mark these transactions as processed
                            for m in sender_movements:
                                processed_txs.add(m.transaction_hash)
                                m.is_coordinated = True
                                m.coordination_group = group.group_id
                
                # Detect accumulation pattern (multiple senders, one recipient)
                for recipient, recipient_movements in recipients.items():
                    if len(recipient_movements) >= 3:  # At least 3 incoming transactions
                        unique_senders = set(m.from_address for m in recipient_movements)
                        if len(unique_senders) >= 3:  # From at least 3 different addresses
                            # This looks like an accumulation pattern
                            group = self._create_coordination_group(
                                recipient_movements,
                                "ACCUMULATION",
                                0.8 if len(recipient_movements) >= 5 else 0.6
                            )
                            coordination_groups.append(group)
                            
                            # Mark these transactions as processed
                            for m in recipient_movements:
                                processed_txs.add(m.transaction_hash)
                                m.is_coordinated = True
                                m.coordination_group = group.group_id
                
                # Detect circular pattern
                # This is more complex and would require graph analysis
                # For now, we'll use a simplified approach looking for A->B->C->A patterns
                if len(window_movements) >= 3:
                    # Build directed graph
                    graph = {}
                    for m in window_movements:
                        if m.from_address not in graph:
                            graph[m.from_address] = set()
                        graph[m.from_address].add(m.to_address)
                    
                    # Look for cycles
                    cycles = self._find_cycles(graph)
                    for cycle in cycles:
                        if len(cycle) >= 3:  # At least 3 wallets in the cycle
                            # Find movements that are part of this cycle
                            cycle_movements = []
                            for m in window_movements:
                                for i in range(len(cycle) - 1):
                                    if m.from_address == cycle[i] and m.to_address == cycle[i + 1]:
                                        cycle_movements.append(m)
                                # Check the wrap-around connection
                                if m.from_address == cycle[-1] and m.to_address == cycle[0]:
                                    cycle_movements.append(m)
                            
                            if len(cycle_movements) >= 3:
                                # This looks like a circular pattern
                                group = self._create_coordination_group(
                                    cycle_movements,
                                    "CIRCULAR",
                                    0.9  # High confidence for circular patterns
                                )
                                coordination_groups.append(group)
                                
                                # Mark these transactions as processed
                                for m in cycle_movements:
                                    processed_txs.add(m.transaction_hash)
                                    m.is_coordinated = True
                                    m.coordination_group = group.group_id
        
        # Update the coordination groups cache
        for group in coordination_groups:
            self._coordination_groups[group.group_id] = group
        
        return coordination_groups
    
    def _create_coordination_group(
        self,
        movements: List[WhaleMovement],
        pattern_type: str,
        confidence: float
    ) -> CoordinationGroup:
        """
        Create a coordination group from a list of movements.
        
        Args:
            movements: List of movements in the coordination group
            pattern_type: Type of coordination pattern detected
            confidence: Confidence score for the detection
            
        Returns:
            CoordinationGroup object
        """
        import uuid
        
        # Get unique wallets involved
        wallets = set()
        for m in movements:
            wallets.add(m.from_address)
            wallets.add(m.to_address)
        
        # Get time range
        timestamps = [datetime.fromisoformat(m.timestamp) for m in movements]
        start_time = min(timestamps).isoformat()
        end_time = max(timestamps).isoformat()
        
        # Calculate total value
        total_value = sum(m.value_usd for m in movements)
        
        # Create group
        return CoordinationGroup(
            group_id=str(uuid.uuid4()),
            wallets=list(wallets),
            start_time=start_time,
            end_time=end_time,
            total_value_usd=total_value,
            movement_count=len(movements),
            pattern_type=pattern_type,
            confidence=confidence
        )
    
    def _find_cycles(self, graph: Dict[str, Set[str]]) -> List[List[str]]:
        """
        Find cycles in a directed graph using DFS.
        
        Args:
            graph: Directed graph represented as adjacency list
            
        Returns:
            List of cycles found in the graph
        """
        cycles = []
        visited = set()
        
        def dfs(node, path, start_node):
            if node in path:
                # Found a cycle
                cycle_start = path.index(node)
                cycle = path[cycle_start:] + [node]
                cycles.append(cycle)
                return
            
            if node in visited:
                return
            
            visited.add(node)
            path.append(node)
            
            if node in graph:
                for neighbor in graph[node]:
                    dfs(neighbor, path.copy(), start_node)
            
            path.pop()
        
        # Start DFS from each node
        for node in graph:
            dfs(node, [], node)
        
        return cycles
    
    def _emit_graph_events(self, results: Dict[str, Any]) -> None:
        """
        Emit graph events for Neo4j integration.
        
        Args:
            results: Whale detection results
        """
        try:
            # Prepare data for Neo4j
            graph_data = {
                "whales": results["whales"],
                "movements": results["movements"],
                "coordination_groups": results["coordination_groups"],
                "timestamp": int(time.time())
            }
            
            # Emit event for graph processing
            emit_event(
                GraphAddEvent(
                    type="whale_detection",
                    data=graph_data
                )
            )
            
            logger.debug(f"Emitted graph events for {len(results['whales'])} whales and {len(results['movements'])} movements")
            
        except Exception as e:
            logger.error(f"Failed to emit graph events: {str(e)}")
            # Don't re-raise, as this is a non-critical operation
