/**
 * TypeScript type definitions for the Whale Detection API
 * These types match the Python Pydantic models from the backend
 */

/**
 * Whale wallet with classification and metrics
 */
export interface WhaleWallet {
  address: string;
  tier: 'TIER1' | 'TIER2' | 'ACTIVE';
  total_value_usd: number;
  last_active?: string;
  large_transactions: number;
  chains: string[];
  tokens: Record<string, number>;
  connected_wallets: string[];
  risk_score?: number;
  first_seen: string;
}

/**
 * Significant whale movement/transaction
 */
export interface WhaleMovement {
  transaction_hash: string;
  from_address: string;
  to_address: string;
  value_usd: number;
  timestamp: string;
  chain: string;
  token_address?: string;
  token_symbol?: string;
  movement_type: string;
  is_coordinated: boolean;
  coordination_group?: string;
}

/**
 * Group of coordinated whale movements
 */
export interface CoordinationGroup {
  group_id: string;
  wallets: string[];
  start_time: string;
  end_time: string;
  total_value_usd: number;
  movement_count: number;
  pattern_type: 'DISTRIBUTION' | 'ACCUMULATION' | 'CIRCULAR';
  confidence: number;
}

/**
 * Response from whale detection API
 */
export interface WhaleDetectionResponse {
  whales: WhaleWallet[];
  movements: WhaleMovement[];
  coordination_groups: CoordinationGroup[];
  stats: {
    total_whales_detected: number;
    new_whales_detected: number;
    large_movements_detected: number;
    coordination_groups_detected: number;
    total_value_monitored_usd: number;
  };
  error?: string;
}

/**
 * Response from whale movements API
 */
export interface WhaleMovementResponse {
  wallet_address: string;
  movements: WhaleMovement[];
  stats: {
    total_movements: number;
    total_value_usd: number;
    movement_types: Record<string, number>;
    chains: Record<string, number>;
    average_value_usd: number;
  };
  error?: string;
}

/**
 * Alert object in the whale monitor response
 */
export interface WhaleAlert {
  wallet_address?: string;
  transaction_hash?: string;
  type: string;
  value_usd?: number;
  timestamp?: string;
  chain?: string;
  token_symbol?: string;
  alert_level: 'HIGH' | 'MEDIUM' | 'LOW';
  pattern_type?: string;
  wallets_involved?: number;
  confidence?: number;
  group_id?: string;
}

/**
 * Response from whale monitor API
 */
export interface WhaleMonitorResponse {
  monitor_id: string;
  wallets_monitored: string[];
  alerts: WhaleAlert[];
  stats: {
    wallets_monitored: number;
    alerts_generated: number;
    coordination_groups_detected: number;
    alert_threshold_usd: number;
    monitor_start_time: string;
  };
  error?: string;
}

/**
 * Response from whale stats API
 */
export interface WhaleStatsResponse {
  time_period: string;
  whale_counts: Record<string, number>;
  total_value_usd: number;
  movement_stats: {
    total_movements: number;
    total_value_usd: number;
    average_value_usd: number;
    movement_types: Record<string, number>;
    coordinated_movements: number;
  };
  chain_distribution: Record<string, number>;
  error?: string;
}

/**
 * Options for whale detection API
 */
export interface WhaleDetectionOptions {
  wallet_address?: string;
  lookback_days?: number;
  tier1_threshold?: number;
  tier2_threshold?: number;
  tx_threshold?: number;
  detect_coordination?: boolean;
  chain_ids?: string;
}

/**
 * Options for whale monitor API
 */
export interface WhaleMonitorOptions {
  wallets: string[];
  alert_threshold_usd?: number;
  coordination_detection?: boolean;
  chain_ids?: string;
}
