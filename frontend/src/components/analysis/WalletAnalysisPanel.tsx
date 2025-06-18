'use client';

import React, { useState } from 'react';
import { useQuery } from 'react-query';
import { analysisAPI, handleAPIError } from '@/lib/api';
import { useToast } from '@/hooks/useToast';
import {
  Wallet,
  Activity,
  DollarSign,
  Loader2,
  AlertTriangle,
  Search,
  ArrowUpRight,
  ArrowDownLeft,
  RefreshCw,
  Link as LinkIcon,
  Coins,
  Tag,
  Zap,
  CheckCircle,
  XCircle,
  Info,
} from 'lucide-react';
import { cn } from '@/lib/utils';

// --- TypeScript Interfaces for Sim API Data ---

interface TokenMetadata {
  symbol: string;
  name?: string;
  decimals: number;
  logo?: string;
  url?: string;
}

interface TokenBalance {
  address: string;
  amount: string;
  chain: string;
  chain_id: number;
  decimals: number;
  name?: string;
  symbol: string;
  price_usd?: number;
  value_usd?: number;
  token_metadata?: TokenMetadata;
  low_liquidity?: boolean;
  pool_size?: number;
}

interface BalancesResponse {
  balances: TokenBalance[];
  wallet_address: string;
  next_offset?: string;
  request_time?: string;
  response_time?: string;
}

interface FunctionParameter {
  name: string;
  type: string;
  value: any;
}

interface FunctionInfo {
  name: string;
  signature?: string;
  parameters?: FunctionParameter[];
}

interface ActivityItem {
  id?: string;
  type: 'send' | 'receive' | 'mint' | 'burn' | 'swap' | 'approve' | 'call';
  chain: string;
  chain_id: number;
  block_number: number;
  block_time: string;
  transaction_hash: string;
  from_address?: string;
  to_address?: string;
  asset_type?: string;
  amount?: string;
  value?: string;
  value_usd?: number;
  token_address?: string;
  token_id?: string;
  token_metadata?: TokenMetadata;
  function?: FunctionInfo;
}

interface ActivityResponse {
  activity: ActivityItem[];
  wallet_address: string;
  next_offset?: string;
  request_time?: string;
  response_time?: string;
}

// --- Helper Functions ---

const formatUsd = (value?: number) => {
  if (value === undefined || value === null) return 'N/A';
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  }).format(value);
};

const formatAmount = (amount: string, decimals: number) => {
  if (!amount || isNaN(Number(amount))) return amount;
  return (Number(amount) / Math.pow(10, decimals)).toFixed(4);
};

const getActivityIcon = (type: ActivityItem['type']) => {
  switch (type) {
    case 'send':
      return <ArrowUpRight className="h-4 w-4 text-red-500" />;
    case 'receive':
      return <ArrowDownLeft className="h-4 w-4 text-green-500" />;
    case 'swap':
      return <RefreshCw className="h-4 w-4 text-blue-500" />;
    case 'approve':
      return <CheckCircle className="h-4 w-4 text-purple-500" />;
    case 'call':
      return <Zap className="h-4 w-4 text-yellow-500" />;
    case 'mint':
      return <Coins className="h-4 w-4 text-indigo-500" />;
    case 'burn':
      return <XCircle className="h-4 w-4 text-gray-500" />;
    default:
      return <Info className="h-4 w-4 text-gray-500" />;
  }
};

export function WalletAnalysisPanel() {
  const [walletAddress, setWalletAddress] = useState<string>('');
  const [currentWallet, setCurrentWallet] = useState<string>('');
  const { toast } = useToast();
  
  // Fetch balances
  const {
    data: balancesData,
    isLoading: isLoadingBalances,
    isFetching: isFetchingBalances,
    error: balancesError,
    refetch: refetchBalances,
  } = useQuery<BalancesResponse, Error>(
    ['simBalances', currentWallet],
    async () => {
      if (!currentWallet) return { balances: [], wallet_address: '' };
      const response = await analysisAPI.getSimBalances(currentWallet);
      return response.data;
    },
    {
      enabled: !!currentWallet,
      onError: (err) => {
        const errorInfo = handleAPIError(err);
        toast({
          description: `Failed to fetch balances: ${errorInfo.message}`,
          variant: 'destructive',
        });
      },
    }
  );

  // Fetch activity
  const {
    data: activityData,
    isLoading: isLoadingActivity,
    isFetching: isFetchingActivity,
    error: activityError,
    refetch: refetchActivity,
  } = useQuery<ActivityResponse, Error>(
    ['simActivity', currentWallet],
    async () => {
      if (!currentWallet) return { activity: [], wallet_address: '' };
      const response = await analysisAPI.getSimActivity(currentWallet);
      return response.data;
    },
    {
      enabled: !!currentWallet,
      onError: (err) => {
        const errorInfo = handleAPIError(err);
        toast({
          description: `Failed to fetch activity: ${errorInfo.message}`,
          variant: 'destructive',
        });
      },
    }
  );

  const handleAnalyzeWallet = () => {
    if (!walletAddress.trim()) {
      toast({
        description: 'Please enter a wallet address.',
        variant: 'destructive',
      });
      return;
    }
    setCurrentWallet(walletAddress.trim());
  };

  const totalUsdValue = balancesData?.balances.reduce((sum, b) => sum + (b.value_usd || 0), 0) || 0;

  return (
    <div className="flex flex-col h-full bg-gray-50 dark:bg-gray-950">
      {/* Header */}
      <div className="p-6 border-b border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900">
        <h2 className="text-xl font-semibold text-gray-900 dark:text-gray-100 flex items-center">
          <Wallet className="h-6 w-6 mr-2 text-blue-500" />
          Wallet Analysis
        </h2>
        <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
          Investigate crypto wallet balances and activity using Sim APIs.
        </p>
      </div>

      {/* Wallet Input */}
      <div className="p-6 border-b border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900">
        <div className="flex space-x-3">
          <input
            type="text"
            value={walletAddress}
            onChange={(e) => setWalletAddress(e.target.value)}
            placeholder="Enter wallet address (e.g., 0xd8da...)"
            className="flex-1 px-4 py-2 border border-gray-300 dark:border-gray-700 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500 bg-gray-50 dark:bg-gray-800 text-gray-900 dark:text-gray-100 placeholder-gray-400"
          />
          <button
            onClick={handleAnalyzeWallet}
            className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
            disabled={isLoadingBalances || isLoadingActivity || isFetchingBalances || isFetchingActivity}
          >
            {(isLoadingBalances || isLoadingActivity || isFetchingBalances || isFetchingActivity) ? (
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
            ) : (
              <Search className="mr-2 h-4 w-4" />
            )}
            Analyze
          </button>
        </div>
        {currentWallet && (
          <p className="mt-2 text-sm text-gray-600 dark:text-gray-300">
            Analyzing: <span className="font-mono text-blue-600 dark:text-blue-400">{currentWallet}</span>
          </p>
        )}
      </div>

      {/* Analysis Results */}
      <div className="flex-1 overflow-y-auto p-6 space-y-6">
        {!currentWallet ? (
          <div className="flex flex-col items-center justify-center h-full text-gray-500 dark:text-gray-400">
            <Wallet className="h-16 w-16 mb-4" />
            <p className="text-lg">Enter a wallet address to begin analysis.</p>
          </div>
        ) : (
          <>
            {/* Balances Section */}
            <div className="bg-white dark:bg-gray-900 shadow-md rounded-lg overflow-hidden">
              <div className="p-5 border-b border-gray-200 dark:border-gray-800 flex items-center justify-between">
                <h3 className="text-lg font-medium text-gray-900 dark:text-gray-100 flex items-center">
                  <DollarSign className="h-5 w-5 mr-2 text-green-500" />
                  Balances
                </h3>
                <span className="text-xl font-bold text-green-600 dark:text-green-400">
                  {formatUsd(totalUsdValue)}
                </span>
              </div>
              {(isLoadingBalances || isFetchingBalances) ? (
                <div className="flex justify-center items-center p-8">
                  <Loader2 className="h-8 w-8 animate-spin text-blue-500" />
                  <span className="ml-3 text-gray-600 dark:text-gray-400">Loading balances...</span>
                </div>
              ) : balancesError ? (
                <div className="p-5 text-red-600 dark:text-red-400 flex items-center">
                  <AlertTriangle className="h-5 w-5 mr-2" />
                  Error: {balancesError.message}
                </div>
              ) : (
                <div className="divide-y divide-gray-200 dark:divide-gray-800">
                  {balancesData?.balances.length === 0 ? (
                    <p className="p-5 text-gray-500 dark:text-gray-400 text-center">No balances found for this wallet.</p>
                  ) : (
                    balancesData?.balances.map((balance, index) => (
                      <div key={index} className="p-4 hover:bg-gray-50 dark:hover:bg-gray-800 flex items-center justify-between">
                        <div className="flex items-center">
                          {balance.token_metadata?.logo && (
                            <img src={balance.token_metadata.logo} alt={balance.symbol} className="h-6 w-6 rounded-full mr-3" />
                          )}
                          <div>
                            <p className="font-medium text-gray-900 dark:text-gray-100">{balance.symbol} ({balance.chain})</p>
                            <p className="text-sm text-gray-500 dark:text-gray-400">
                              {formatAmount(balance.amount, balance.decimals)}
                            </p>
                          </div>
                        </div>
                        <div className="text-right">
                          <p className="font-semibold text-gray-900 dark:text-gray-100">{formatUsd(balance.value_usd)}</p>
                          {balance.low_liquidity && (
                            <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200 mt-1">
                              <AlertTriangle className="h-3 w-3 mr-1" /> Low Liquidity
                            </span>
                          )}
                        </div>
                      </div>
                    ))
                  )}
                </div>
              )}
            </div>

            {/* Activity Section */}
            <div className="bg-white dark:bg-gray-900 shadow-md rounded-lg overflow-hidden">
              <div className="p-5 border-b border-gray-200 dark:border-gray-800">
                <h3 className="text-lg font-medium text-gray-900 dark:text-gray-100 flex items-center">
                  <Activity className="h-5 w-5 mr-2 text-purple-500" />
                  Recent Activity
                </h3>
              </div>
              {(isLoadingActivity || isFetchingActivity) ? (
                <div className="flex justify-center items-center p-8">
                  <Loader2 className="h-8 w-8 animate-spin text-blue-500" />
                  <span className="ml-3 text-gray-600 dark:text-gray-400">Loading activity...</span>
                </div>
              ) : activityError ? (
                <div className="p-5 text-red-600 dark:text-red-400 flex items-center">
                  <AlertTriangle className="h-5 w-5 mr-2" />
                  Error: {activityError.message}
                </div>
              ) : (
                <div className="divide-y divide-gray-200 dark:divide-gray-800">
                  {activityData?.activity.length === 0 ? (
                    <p className="p-5 text-gray-500 dark:text-gray-400 text-center">No activity found for this wallet.</p>
                  ) : (
                    activityData?.activity.map((activity, index) => (
                      <div key={index} className="p-4 hover:bg-gray-50 dark:hover:bg-gray-800 flex items-center justify-between">
                        <div className="flex items-center">
                          <div className="h-8 w-8 rounded-full bg-gray-100 dark:bg-gray-800 flex items-center justify-center mr-3">
                            {getActivityIcon(activity.type)}
                          </div>
                          <div>
                            <p className="font-medium text-gray-900 dark:text-gray-100 flex items-center">
                              {activity.type.charAt(0).toUpperCase() + activity.type.slice(1)}
                              {activity.type === 'call' && activity.function && (
                                <span className="ml-1 text-gray-500 dark:text-gray-400">
                                  : {activity.function.name}
                                </span>
                              )}
                            </p>
                            <div className="flex text-xs text-gray-500 dark:text-gray-400 space-x-2">
                              <span>{new Date(activity.block_time).toLocaleString()}</span>
                              <span>•</span>
                              <span className="font-mono">{activity.transaction_hash.substring(0, 8)}...</span>
                            </div>
                          </div>
                        </div>
                        <div className="text-right">
                          {activity.value_usd !== undefined && (
                            <p className={cn(
                              "font-semibold",
                              activity.type === 'receive' ? 'text-green-600 dark:text-green-400' : 
                              activity.type === 'send' ? 'text-red-600 dark:text-red-400' : 
                              'text-gray-900 dark:text-gray-100'
                            )}>
                              {activity.type === 'receive' ? '+' : activity.type === 'send' ? '-' : ''}
                              {formatUsd(activity.value_usd)}
                            </p>
                          )}
                          {activity.token_metadata && (
                            <p className="text-sm text-gray-500 dark:text-gray-400">
                              {activity.token_metadata.symbol}
                            </p>
                          )}
                        </div>
                      </div>
                    ))
                  )}
                </div>
              )}
            </div>

            {/* Risk Assessment Section */}
            <div className="bg-white dark:bg-gray-900 shadow-md rounded-lg overflow-hidden">
              <div className="p-5 border-b border-gray-200 dark:border-gray-800">
                <h3 className="text-lg font-medium text-gray-900 dark:text-gray-100 flex items-center">
                  <AlertTriangle className="h-5 w-5 mr-2 text-orange-500" />
                  Risk Assessment
                </h3>
              </div>
              <div className="p-5">
                <div className="space-y-4">
                  {/* Risk Indicators */}
                  <div>
                    <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">Key Risk Indicators</h4>
                    <div className="grid grid-cols-2 gap-4">
                      <div className="bg-gray-50 dark:bg-gray-800 p-3 rounded-lg border border-gray-200 dark:border-gray-700">
                        <div className="text-sm text-gray-500 dark:text-gray-400">Total Value</div>
                        <div className="text-lg font-semibold text-gray-900 dark:text-gray-100">{formatUsd(totalUsdValue)}</div>
                      </div>
                      <div className="bg-gray-50 dark:bg-gray-800 p-3 rounded-lg border border-gray-200 dark:border-gray-700">
                        <div className="text-sm text-gray-500 dark:text-gray-400">Low Liquidity Tokens</div>
                        <div className="text-lg font-semibold text-gray-900 dark:text-gray-100">
                          {balancesData?.balances.filter(b => b.low_liquidity).length || 0}
                        </div>
                      </div>
                      <div className="bg-gray-50 dark:bg-gray-800 p-3 rounded-lg border border-gray-200 dark:border-gray-700">
                        <div className="text-sm text-gray-500 dark:text-gray-400">Unique Chains</div>
                        <div className="text-lg font-semibold text-gray-900 dark:text-gray-100">
                          {new Set(balancesData?.balances.map(b => b.chain)).size || 0}
                        </div>
                      </div>
                      <div className="bg-gray-50 dark:bg-gray-800 p-3 rounded-lg border border-gray-200 dark:border-gray-700">
                        <div className="text-sm text-gray-500 dark:text-gray-400">Transaction Count</div>
                        <div className="text-lg font-semibold text-gray-900 dark:text-gray-100">
                          {activityData?.activity.length || 0}
                        </div>
                      </div>
                    </div>
                  </div>
                  
                  {/* Potential Risk Flags */}
                  <div>
                    <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">Potential Risk Flags</h4>
                    <ul className="space-y-2">
                      {totalUsdValue > 100000 && (
                        <li className="flex items-center text-sm text-amber-600 dark:text-amber-400">
                          <AlertTriangle className="h-4 w-4 mr-2" />
                          High value wallet ({formatUsd(totalUsdValue)})
                        </li>
                      )}
                      {balancesData?.balances.filter(b => b.low_liquidity).length > 0 && (
                        <li className="flex items-center text-sm text-amber-600 dark:text-amber-400">
                          <AlertTriangle className="h-4 w-4 mr-2" />
                          Contains {balancesData?.balances.filter(b => b.low_liquidity).length} low liquidity tokens
                        </li>
                      )}
                      {new Set(balancesData?.balances.map(b => b.chain)).size > 3 && (
                        <li className="flex items-center text-sm text-amber-600 dark:text-amber-400">
                          <AlertTriangle className="h-4 w-4 mr-2" />
                          Active across {new Set(balancesData?.balances.map(b => b.chain)).size} different chains
                        </li>
                      )}
                      {activityData?.activity.filter(a => a.type === 'call').length > 5 && (
                        <li className="flex items-center text-sm text-amber-600 dark:text-amber-400">
                          <AlertTriangle className="h-4 w-4 mr-2" />
                          High contract interaction frequency
                        </li>
                      )}
                    </ul>
                  </div>
                  
                  {/* Actions */}
                  <div className="pt-4 flex space-x-3">
                    <button className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors flex items-center">
                      <LinkIcon className="h-4 w-4 mr-2" />
                      Add to Investigation
                    </button>
                    <button className="px-4 py-2 bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200 rounded-md hover:bg-gray-300 dark:hover:bg-gray-600 transition-colors flex items-center">
                      <Tag className="h-4 w-4 mr-2" />
                      Flag for Review
                    </button>
                  </div>
                </div>
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
