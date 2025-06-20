import React, { useState, useEffect } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
import {
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
} from '@/components/ui/tabs';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import {
  Table,
  TableBody,
  TableCaption,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import {
  AlertCircle,
  AlertTriangle,
  ArrowDownCircle,
  ArrowUpCircle,
  BarChart4,
  Clock,
  Coins,
  DollarSign,
  LineChart,
  Loader2,
  Network,
  RefreshCcw,
  Search,
  Settings,
  Wallet,
} from 'lucide-react';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Slider } from '@/components/ui/slider';
import { Switch } from '@/components/ui/switch';
import { Skeleton } from '@/components/ui/skeleton';
import { Progress } from '@/components/ui/progress';
import { Separator } from '@/components/ui/separator';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { useToast } from '@/hooks/useToast';
import { formatDistanceToNow } from 'date-fns';
import { api } from '@/lib/api';

// TypeScript interfaces for whale data structures
interface WhaleWallet {
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

interface WhaleMovement {
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

interface CoordinationGroup {
  group_id: string;
  wallets: string[];
  start_time: string;
  end_time: string;
  total_value_usd: number;
  movement_count: number;
  pattern_type: 'DISTRIBUTION' | 'ACCUMULATION' | 'CIRCULAR';
  confidence: number;
}

interface WhaleDetectionResponse {
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

interface WhaleMovementResponse {
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

interface WhaleMonitorResponse {
  monitor_id: string;
  wallets_monitored: string[];
  alerts: Array<{
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
  }>;
  stats: {
    wallets_monitored: number;
    alerts_generated: number;
    coordination_groups_detected: number;
    alert_threshold_usd: number;
    monitor_start_time: string;
  };
  error?: string;
}

interface WhaleStatsResponse {
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

interface WhaleDetectionOptions {
  wallet_address?: string;
  lookback_days?: number;
  tier1_threshold?: number;
  tier2_threshold?: number;
  tx_threshold?: number;
  detect_coordination?: boolean;
  chain_ids?: string;
}

interface WhaleMonitorOptions {
  wallets: string[];
  alert_threshold_usd?: number;
  coordination_detection?: boolean;
  chain_ids?: string;
}

// Custom hooks for whale data fetching
const useWhaleDetection = (options: WhaleDetectionOptions = {}) => {
  return useQuery<WhaleDetectionResponse>({
    queryKey: ['whaleDetection', options],
    queryFn: async () => {
      const response = await api.post('/api/v1/analysis/whale/detect', options);
      return response.data;
    },
    enabled: !!options.wallet_address || options.wallet_address === '',
    staleTime: 30000, // 30 seconds
  });
};

const useWhaleMovements = (wallet: string, options: { lookback_days?: number; tx_threshold?: number; chain_ids?: string } = {}) => {
  return useQuery<WhaleMovementResponse>({
    queryKey: ['whaleMovements', wallet, options],
    queryFn: async () => {
      const params = new URLSearchParams();
      if (options.lookback_days) params.append('lookback_days', options.lookback_days.toString());
      if (options.tx_threshold) params.append('tx_threshold', options.tx_threshold.toString());
      if (options.chain_ids) params.append('chain_ids', options.chain_ids);
      
      const response = await api.get(`/api/v1/analysis/whale/movements/${wallet}?${params.toString()}`);
      return response.data;
    },
    enabled: !!wallet,
    staleTime: 30000, // 30 seconds
  });
};

const useWhaleMonitor = (options: WhaleMonitorOptions) => {
  return useQuery<WhaleMonitorResponse>({
    queryKey: ['whaleMonitor', options],
    queryFn: async () => {
      const response = await api.post('/api/v1/analysis/whale/monitor', options);
      return response.data;
    },
    enabled: options.wallets.length > 0,
    refetchInterval: 60000, // Refetch every minute for real-time monitoring
  });
};

const useWhaleStats = (options: { time_period?: string; min_tier?: string; chain_ids?: string } = {}) => {
  return useQuery<WhaleStatsResponse>({
    queryKey: ['whaleStats', options],
    queryFn: async () => {
      const params = new URLSearchParams();
      if (options.time_period) params.append('time_period', options.time_period);
      if (options.min_tier) params.append('min_tier', options.min_tier);
      if (options.chain_ids) params.append('chain_ids', options.chain_ids);
      
      const response = await api.get(`/api/v1/analysis/whale/stats?${params.toString()}`);
      return response.data;
    },
    staleTime: 300000, // 5 minutes
  });
};

// Helper components
const WhaleClassificationBadge = ({ tier }: { tier: string }) => {
  switch (tier) {
    case 'TIER1':
      return <Badge className="bg-red-500 hover:bg-red-600">Tier 1 Whale</Badge>;
    case 'TIER2':
      return <Badge className="bg-orange-500 hover:bg-orange-600">Tier 2 Whale</Badge>;
    case 'ACTIVE':
      return <Badge className="bg-blue-500 hover:bg-blue-600">Active Whale</Badge>;
    default:
      return <Badge>Unknown</Badge>;
  }
};

const MovementTypeBadge = ({ type }: { type: string }) => {
  switch (type) {
    case 'SEND':
      return <Badge variant="outline" className="border-red-500 text-red-500"><ArrowUpCircle className="mr-1 h-3 w-3" /> Send</Badge>;
    case 'RECEIVE':
      return <Badge variant="outline" className="border-green-500 text-green-500"><ArrowDownCircle className="mr-1 h-3 w-3" /> Receive</Badge>;
    case 'SWAP':
      return <Badge variant="outline" className="border-purple-500 text-purple-500">Swap</Badge>;
    case 'CALL':
      return <Badge variant="outline" className="border-blue-500 text-blue-500">Contract Call</Badge>;
    default:
      return <Badge variant="outline">{type}</Badge>;
  }
};

const formatUSD = (value: number) => {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    maximumFractionDigits: 0
  }).format(value);
};

const formatAddress = (address: string) => {
  if (!address || address.length < 10) return address;
  return `${address.substring(0, 6)}...${address.substring(address.length - 4)}`;
};

// Main Whale Dashboard Component
const WhaleDashboard: React.FC = () => {
  const { toast } = useToast();
  const queryClient = useQueryClient();
  
  // State for configuration options
  const [activeTab, setActiveTab] = useState('overview');
  const [searchWallet, setSearchWallet] = useState('');
  const [searchQuery, setSearchQuery] = useState('');
  const [monitoredWallets, setMonitoredWallets] = useState<string[]>([]);
  const [alertThreshold, setAlertThreshold] = useState(100000); // $100k default
  const [detectCoordination, setDetectCoordination] = useState(true);
  const [timePeriod, setTimePeriod] = useState('24h');
  const [minTier, setMinTier] = useState('TIER2');
  const [selectedChains, setSelectedChains] = useState('all');
  const [lookbackDays, setLookbackDays] = useState(7);
  
  // Queries
  const whaleDetection = useWhaleDetection({
    wallet_address: searchQuery,
    lookback_days: lookbackDays,
    detect_coordination: detectCoordination,
    chain_ids: selectedChains
  });
  
  const whaleStats = useWhaleStats({
    time_period: timePeriod,
    min_tier: minTier,
    chain_ids: selectedChains
  });
  
  const whaleMonitor = useWhaleMonitor({
    wallets: monitoredWallets,
    alert_threshold_usd: alertThreshold,
    coordination_detection: detectCoordination,
    chain_ids: selectedChains
  });
  
  // Handle search
  const handleSearch = () => {
    setSearchQuery(searchWallet);
  };
  
  // Handle monitoring
  const addToMonitoring = (wallet: string) => {
    if (!monitoredWallets.includes(wallet)) {
      setMonitoredWallets([...monitoredWallets, wallet]);
      toast({
        title: "Wallet added to monitoring",
        description: `Now tracking ${formatAddress(wallet)}`,
      });
    }
  };
  
  const removeFromMonitoring = (wallet: string) => {
    setMonitoredWallets(monitoredWallets.filter(w => w !== wallet));
    toast({
      title: "Wallet removed from monitoring",
      description: `No longer tracking ${formatAddress(wallet)}`,
    });
  };
  
  // Handle refresh
  const handleRefresh = () => {
    queryClient.invalidateQueries({ queryKey: ['whaleDetection'] });
    queryClient.invalidateQueries({ queryKey: ['whaleStats'] });
    queryClient.invalidateQueries({ queryKey: ['whaleMonitor'] });
    toast({
      title: "Data refreshed",
      description: "Whale tracking data has been updated",
    });
  };
  
  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <div>
          <h2 className="text-3xl font-bold tracking-tight">Whale Movement Tracker</h2>
          <p className="text-muted-foreground">
            Monitor large wallet movements and detect coordination patterns across multiple blockchains
          </p>
        </div>
        <Button onClick={handleRefresh} variant="outline" className="gap-2">
          <RefreshCcw className="h-4 w-4" />
          Refresh
        </Button>
      </div>
      
      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="grid grid-cols-4 w-[600px]">
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="whales">Whale Wallets</TabsTrigger>
          <TabsTrigger value="movements">Large Movements</TabsTrigger>
          <TabsTrigger value="coordination">Coordination</TabsTrigger>
        </TabsList>
        
        {/* Overview Tab */}
        <TabsContent value="overview" className="space-y-6">
          {/* Stats Cards */}
          <div className="grid grid-cols-4 gap-4">
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium text-muted-foreground">Total Whales</CardTitle>
              </CardHeader>
              <CardContent>
                {whaleStats.isLoading ? (
                  <Skeleton className="h-8 w-20" />
                ) : (
                  <div className="text-2xl font-bold">
                    {whaleStats.data?.whale_counts ? 
                      Object.values(whaleStats.data.whale_counts).reduce((a, b) => a + b, 0) : 
                      '0'}
                  </div>
                )}
              </CardContent>
            </Card>
            
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium text-muted-foreground">Total Value</CardTitle>
              </CardHeader>
              <CardContent>
                {whaleStats.isLoading ? (
                  <Skeleton className="h-8 w-28" />
                ) : (
                  <div className="text-2xl font-bold">
                    {whaleStats.data ? formatUSD(whaleStats.data.total_value_usd) : '$0'}
                  </div>
                )}
              </CardContent>
            </Card>
            
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium text-muted-foreground">Large Movements</CardTitle>
              </CardHeader>
              <CardContent>
                {whaleStats.isLoading ? (
                  <Skeleton className="h-8 w-20" />
                ) : (
                  <div className="text-2xl font-bold">
                    {whaleStats.data?.movement_stats?.total_movements || 0}
                  </div>
                )}
              </CardContent>
            </Card>
            
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium text-muted-foreground">Coordination Groups</CardTitle>
              </CardHeader>
              <CardContent>
                {whaleStats.isLoading ? (
                  <Skeleton className="h-8 w-20" />
                ) : (
                  <div className="text-2xl font-bold">
                    {whaleStats.data?.movement_stats?.coordinated_movements || 0}
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
          
          {/* Whale Distribution */}
          <Card>
            <CardHeader>
              <CardTitle>Whale Classification Distribution</CardTitle>
              <CardDescription>Breakdown of whales by tier and chain</CardDescription>
            </CardHeader>
            <CardContent>
              {whaleStats.isLoading ? (
                <div className="space-y-2">
                  <Skeleton className="h-4 w-full" />
                  <Skeleton className="h-4 w-full" />
                  <Skeleton className="h-4 w-full" />
                </div>
              ) : whaleStats.error ? (
                <Alert variant="destructive">
                  <AlertCircle className="h-4 w-4" />
                  <AlertTitle>Error</AlertTitle>
                  <AlertDescription>
                    Failed to load whale statistics: {whaleStats.error.toString()}
                  </AlertDescription>
                </Alert>
              ) : (
                <div className="space-y-6">
                  {/* Tier distribution */}
                  <div>
                    <h4 className="text-sm font-medium mb-2">Whale Tiers</h4>
                    <div className="space-y-2">
                      {whaleStats.data?.whale_counts && Object.entries(whaleStats.data.whale_counts).map(([tier, count]) => (
                        <div key={tier} className="flex items-center gap-2">
                          <div className="w-24">
                            <WhaleClassificationBadge tier={tier} />
                          </div>
                          <Progress 
                            value={count / Object.values(whaleStats.data.whale_counts).reduce((a, b) => a + b, 0) * 100} 
                            className="h-2"
                          />
                          <span className="text-sm font-medium">{count}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                  
                  {/* Chain distribution */}
                  <div>
                    <h4 className="text-sm font-medium mb-2">Chain Distribution</h4>
                    <div className="space-y-2">
                      {whaleStats.data?.chain_distribution && Object.entries(whaleStats.data.chain_distribution)
                        .sort((a, b) => b[1] - a[1])
                        .slice(0, 5)
                        .map(([chain, count]) => (
                          <div key={chain} className="flex items-center gap-2">
                            <div className="w-24 truncate">
                              <Badge variant="outline">{chain}</Badge>
                            </div>
                            <Progress 
                              value={count / Object.values(whaleStats.data.chain_distribution).reduce((a, b) => a + b, 0) * 100} 
                              className="h-2"
                            />
                            <span className="text-sm font-medium">{count}</span>
                          </div>
                      ))}
                    </div>
                  </div>
                </div>
              )}
            </CardContent>
            <CardFooter className="flex justify-between">
              <div className="flex items-center gap-2">
                <Label htmlFor="time-period">Time Period:</Label>
                <Select value={timePeriod} onValueChange={setTimePeriod}>
                  <SelectTrigger className="w-24">
                    <SelectValue placeholder="Period" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="1h">1 hour</SelectItem>
                    <SelectItem value="24h">24 hours</SelectItem>
                    <SelectItem value="7d">7 days</SelectItem>
                    <SelectItem value="30d">30 days</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="flex items-center gap-2">
                <Label htmlFor="min-tier">Min Tier:</Label>
                <Select value={minTier} onValueChange={setMinTier}>
                  <SelectTrigger className="w-24">
                    <SelectValue placeholder="Min Tier" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="TIER1">Tier 1</SelectItem>
                    <SelectItem value="TIER2">Tier 2</SelectItem>
                    <SelectItem value="ACTIVE">Active</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </CardFooter>
          </Card>
          
          {/* Recent Alerts */}
          <Card>
            <CardHeader>
              <CardTitle>Recent Alerts</CardTitle>
              <CardDescription>Latest whale movement alerts from monitored wallets</CardDescription>
            </CardHeader>
            <CardContent>
              {whaleMonitor.isLoading ? (
                <div className="space-y-2">
                  <Skeleton className="h-12 w-full" />
                  <Skeleton className="h-12 w-full" />
                  <Skeleton className="h-12 w-full" />
                </div>
              ) : !monitoredWallets.length ? (
                <div className="text-center py-6 text-muted-foreground">
                  <AlertCircle className="mx-auto h-8 w-8 mb-2" />
                  <p>No wallets being monitored</p>
                  <p className="text-sm">Add wallets to the monitoring list to receive alerts</p>
                </div>
              ) : whaleMonitor.data?.alerts && whaleMonitor.data.alerts.length > 0 ? (
                <div className="space-y-2">
                  {whaleMonitor.data.alerts.slice(0, 5).map((alert, index) => (
                    <Alert key={index} variant={alert.alert_level === 'HIGH' ? 'destructive' : 'default'} className="py-2">
                      <div className="flex items-center gap-2">
                        {alert.type === 'COORDINATION' ? (
                          <Network className="h-4 w-4" />
                        ) : (
                          <AlertTriangle className="h-4 w-4" />
                        )}
                        <AlertTitle className="text-sm font-medium">
                          {alert.type === 'COORDINATION' ? 
                            `${alert.pattern_type} Pattern Detected` : 
                            `Large ${alert.type} of ${alert.token_symbol || 'tokens'}`}
                        </AlertTitle>
                      </div>
                      <AlertDescription className="text-xs mt-1">
                        {alert.type === 'COORDINATION' ? (
                          <span>
                            {alert.wallets_involved} wallets involved, {formatUSD(alert.value_usd || 0)} total value, 
                            {alert.confidence && ` ${Math.round(alert.confidence * 100)}% confidence`}
                          </span>
                        ) : (
                          <span>
                            {alert.wallet_address && `${formatAddress(alert.wallet_address)} • `}
                            {alert.value_usd && formatUSD(alert.value_usd)} • 
                            {alert.chain && ` ${alert.chain} • `}
                            {alert.timestamp && formatDistanceToNow(new Date(alert.timestamp), { addSuffix: true })}
                          </span>
                        )}
                      </AlertDescription>
                    </Alert>
                  ))}
                </div>
              ) : (
                <div className="text-center py-6 text-muted-foreground">
                  <Clock className="mx-auto h-8 w-8 mb-2" />
                  <p>No alerts detected</p>
                  <p className="text-sm">Monitoring {monitoredWallets.length} wallets</p>
                </div>
              )}
            </CardContent>
            <CardFooter className="flex justify-between">
              <div className="text-sm text-muted-foreground">
                {whaleMonitor.data?.stats && (
                  <>Monitoring {whaleMonitor.data.stats.wallets_monitored} wallets</>
                )}
              </div>
              <Button variant="outline" size="sm" disabled={!monitoredWallets.length} onClick={() => setActiveTab('movements')}>
                View All Alerts
              </Button>
            </CardFooter>
          </Card>
        </TabsContent>
        
        {/* Whales Tab */}
        <TabsContent value="whales" className="space-y-6">
          {/* Search and Configuration */}
          <Card>
            <CardHeader>
              <CardTitle>Search Whale Wallets</CardTitle>
              <CardDescription>
                Search for specific wallets or detect new whales based on transaction activity
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="flex gap-2">
                <div className="flex-1">
                  <Input
                    placeholder="Enter wallet address..."
                    value={searchWallet}
                    onChange={(e) => setSearchWallet(e.target.value)}
                    onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
                  />
                </div>
                <Button onClick={handleSearch} className="gap-2">
                  <Search className="h-4 w-4" />
                  Search
                </Button>
              </div>
              
              <Separator className="my-4" />
              
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label htmlFor="lookback-days">Lookback Period (days)</Label>
                  <div className="flex items-center gap-2">
                    <Slider
                      id="lookback-days"
                      min={1}
                      max={30}
                      step={1}
                      value={[lookbackDays]}
                      onValueChange={(value) => setLookbackDays(value[0])}
                      className="flex-1"
                    />
                    <span className="w-8 text-center">{lookbackDays}</span>
                  </div>
                </div>
                
                <div className="space-y-2">
                  <Label htmlFor="chains">Blockchain Networks</Label>
                  <Select value={selectedChains} onValueChange={setSelectedChains}>
                    <SelectTrigger id="chains">
                      <SelectValue placeholder="Select chains" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="all">All Chains</SelectItem>
                      <SelectItem value="1">Ethereum</SelectItem>
                      <SelectItem value="137">Polygon</SelectItem>
                      <SelectItem value="56">BSC</SelectItem>
                      <SelectItem value="42161">Arbitrum</SelectItem>
                      <SelectItem value="10">Optimism</SelectItem>
                      <SelectItem value="1,137,56,42161,10">Major Chains</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <Label htmlFor="detect-coordination">Detect Coordination</Label>
                    <Switch
                      id="detect-coordination"
                      checked={detectCoordination}
                      onCheckedChange={setDetectCoordination}
                    />
                  </div>
                  <p className="text-xs text-muted-foreground">
                    Identify patterns of coordinated activity between whale wallets
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
          
          {/* Whale List */}
          <Card>
            <CardHeader>
              <CardTitle>Detected Whale Wallets</CardTitle>
              <CardDescription>
                Wallets with large holdings or significant transaction activity
              </CardDescription>
            </CardHeader>
            <CardContent>
              {whaleDetection.isLoading ? (
                <div className="space-y-2">
                  <Skeleton className="h-16 w-full" />
                  <Skeleton className="h-16 w-full" />
                  <Skeleton className="h-16 w-full" />
                </div>
              ) : whaleDetection.error ? (
                <Alert variant="destructive">
                  <AlertCircle className="h-4 w-4" />
                  <AlertTitle>Error</AlertTitle>
                  <AlertDescription>
                    Failed to load whale data: {whaleDetection.error.toString()}
                  </AlertDescription>
                </Alert>
              ) : !whaleDetection.data?.whales.length ? (
                <div className="text-center py-6 text-muted-foreground">
                  <Wallet className="mx-auto h-8 w-8 mb-2" />
                  <p>No whale wallets detected</p>
                  <p className="text-sm">Try adjusting your search criteria or lookback period</p>
                </div>
              ) : (
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Wallet</TableHead>
                      <TableHead>Classification</TableHead>
                      <TableHead>Value</TableHead>
                      <TableHead>Last Active</TableHead>
                      <TableHead>Actions</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {whaleDetection.data.whales.map((whale) => (
                      <TableRow key={whale.address}>
                        <TableCell className="font-mono">{formatAddress(whale.address)}</TableCell>
                        <TableCell>
                          <WhaleClassificationBadge tier={whale.tier} />
                        </TableCell>
                        <TableCell>{formatUSD(whale.total_value_usd)}</TableCell>
                        <TableCell>
                          {whale.last_active ? 
                            formatDistanceToNow(new Date(whale.last_active), { addSuffix: true }) : 
                            'Unknown'}
                        </TableCell>
                        <TableCell>
                          <div className="flex gap-2">
                            <Button 
                              variant="outline" 
                              size="sm"
                              onClick={() => {
                                setSearchWallet(whale.address);
                                setSearchQuery(whale.address);
                              }}
                            >
                              Details
                            </Button>
                            {monitoredWallets.includes(whale.address) ? (
                              <Button 
                                variant="destructive" 
                                size="sm"
                                onClick={() => removeFromMonitoring(whale.address)}
                              >
                                Unmonitor
                              </Button>
                            ) : (
                              <Button 
                                variant="default" 
                                size="sm"
                                onClick={() => addToMonitoring(whale.address)}
                              >
                                Monitor
                              </Button>
                            )}
                          </div>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              )}
            </CardContent>
            <CardFooter>
              <div className="text-sm text-muted-foreground">
                {whaleDetection.data?.stats && (
                  <>
                    {whaleDetection.data.stats.total_whales_detected} whales detected
                    {whaleDetection.data.stats.new_whales_detected > 0 && 
                      ` (${whaleDetection.data.stats.new_whales_detected} new)`}
                  </>
                )}
              </div>
            </CardFooter>
          </Card>
          
          {/* Whale Details (if specific wallet searched) */}
          {searchQuery && whaleDetection.data?.whales.length === 1 && (
            <Card>
              <CardHeader>
                <div className="flex justify-between">
                  <div>
                    <CardTitle>Whale Details</CardTitle>
                    <CardDescription className="font-mono">{searchQuery}</CardDescription>
                  </div>
                  <WhaleClassificationBadge tier={whaleDetection.data.whales[0].tier} />
                </div>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-2 gap-6">
                  <div>
                    <h4 className="text-sm font-medium mb-2">Portfolio Overview</h4>
                    <div className="space-y-2">
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Total Value:</span>
                        <span className="font-medium">{formatUSD(whaleDetection.data.whales[0].total_value_usd)}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">First Seen:</span>
                        <span>{formatDistanceToNow(new Date(whaleDetection.data.whales[0].first_seen), { addSuffix: true })}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Large Transactions:</span>
                        <span>{whaleDetection.data.whales[0].large_transactions}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Active Chains:</span>
                        <span>{whaleDetection.data.whales[0].chains.length}</span>
                      </div>
                    </div>
                    
                    <h4 className="text-sm font-medium mt-4 mb-2">Active Chains</h4>
                    <div className="flex flex-wrap gap-2">
                      {whaleDetection.data.whales[0].chains.map((chain) => (
                        <Badge key={chain} variant="outline">{chain}</Badge>
                      ))}
                    </div>
                  </div>
                  
                  <div>
                    <h4 className="text-sm font-medium mb-2">Top Holdings</h4>
                    <div className="space-y-2">
                      {Object.entries(whaleDetection.data.whales[0].tokens)
                        .sort(([, a], [, b]) => b - a)
                        .slice(0, 5)
                        .map(([symbol, value]) => (
                          <div key={symbol} className="flex justify-between items-center">
                            <div className="flex items-center gap-2">
                              <Coins className="h-4 w-4 text-muted-foreground" />
                              <span>{symbol}</span>
                            </div>
                            <span>{formatUSD(value)}</span>
                          </div>
                        ))}
                    </div>
                    
                    {whaleDetection.data.whales[0].connected_wallets.length > 0 && (
                      <>
                        <h4 className="text-sm font-medium mt-4 mb-2">Connected Wallets</h4>
                        <div className="space-y-1">
                          {whaleDetection.data.whales[0].connected_wallets.map((wallet) => (
                            <div key={wallet} className="font-mono text-xs">
                              {formatAddress(wallet)}
                            </div>
                          ))}
                        </div>
                      </>
                    )}
                  </div>
                </div>
              </CardContent>
              <CardFooter className="flex justify-end gap-2">
                {monitoredWallets.includes(searchQuery) ? (
                  <Button 
                    variant="destructive"
                    onClick={() => removeFromMonitoring(searchQuery)}
                  >
                    Remove from Monitoring
                  </Button>
                ) : (
                  <Button 
                    variant="default"
                    onClick={() => addToMonitoring(searchQuery)}
                  >
                    Add to Monitoring
                  </Button>
                )}
              </CardFooter>
            </Card>
          )}
        </TabsContent>
        
        {/* Movements Tab */}
        <TabsContent value="movements" className="space-y-6">
          {/* Movement Configuration */}
          <Card>
            <CardHeader>
              <CardTitle>Large Movement Monitoring</CardTitle>
              <CardDescription>
                Track significant token transfers and wallet activity
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label htmlFor="alert-threshold">Alert Threshold (USD)</Label>
                  <div className="flex items-center gap-4">
                    <Slider
                      id="alert-threshold"
                      min={10000}
                      max={1000000}
                      step={10000}
                      value={[alertThreshold]}
                      onValueChange={(value) => setAlertThreshold(value[0])}
                      className="flex-1"
                    />
                    <span className="w-24 text-right">{formatUSD(alertThreshold)}</span>
                  </div>
                  <p className="text-xs text-muted-foreground">
                    Minimum USD value for movement alerts
                  </p>
                </div>
                
                <div className="space-y-2">
                  <Label>Monitored Wallets</Label>
                  <div className="flex items-center justify-between">
                    <span className="text-sm">{monitoredWallets.length} wallets</span>
                    <Button 
                      variant="outline" 
                      size="sm" 
                      disabled={!monitoredWallets.length}
                      onClick={() => setMonitoredWallets([])}
                    >
                      Clear All
                    </Button>
                  </div>
                  <div className="flex flex-wrap gap-2 mt-2">
                    {monitoredWallets.slice(0, 3).map((wallet) => (
                      <Badge key={wallet} variant="secondary" className="font-mono">
                        {formatAddress(wallet)}
                      </Badge>
                    ))}
                    {monitoredWallets.length > 3 && (
                      <Badge variant="secondary">
                        +{monitoredWallets.length - 3} more
                      </Badge>
                    )}
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
          
          {/* Movement Feed */}
          <Card>
            <CardHeader>
              <CardTitle>Large Movement Feed</CardTitle>
              <CardDescription>
                Recent significant transactions from monitored wallets
              </CardDescription>
            </CardHeader>
            <CardContent>
              {whaleMonitor.isLoading ? (
                <div className="space-y-2">
                  <Skeleton className="h-16 w-full" />
                  <Skeleton className="h-16 w-full" />
                  <Skeleton className="h-16 w-full" />
                </div>
              ) : !monitoredWallets.length ? (
                <div className="text-center py-6 text-muted-foreground">
                  <AlertCircle className="mx-auto h-8 w-8 mb-2" />
                  <p>No wallets being monitored</p>
                  <p className="text-sm">Add wallets from the "Whales" tab to track movements</p>
                </div>
              ) : !whaleMonitor.data?.alerts.length ? (
                <div className="text-center py-6 text-muted-foreground">
                  <Clock className="mx-auto h-8 w-8 mb-2" />
                  <p>No large movements detected</p>
                  <p className="text-sm">Monitoring {monitoredWallets.length} wallets</p>
                </div>
              ) : (
                <div className="space-y-4">
                  {whaleMonitor.data.alerts
                    .filter(alert => alert.type !== 'COORDINATION')
                    .map((alert, index) => (
                      <div key={index} className="flex items-start gap-4 p-4 rounded-lg border">
                        <div className={`p-2 rounded-full ${
                          alert.alert_level === 'HIGH' ? 'bg-red-100' : 
                          alert.alert_level === 'MEDIUM' ? 'bg-orange-100' : 'bg-blue-100'
                        }`}>
                          {alert.type === 'SEND' ? (
                            <ArrowUpCircle className={`h-6 w-6 ${
                              alert.alert_level === 'HIGH' ? 'text-red-600' : 
                              alert.alert_level === 'MEDIUM' ? 'text-orange-600' : 'text-blue-600'
                            }`} />
                          ) : (
                            <ArrowDownCircle className={`h-6 w-6 ${
                              alert.alert_level === 'HIGH' ? 'text-red-600' : 
                              alert.alert_level === 'MEDIUM' ? 'text-orange-600' : 'text-blue-600'
                            }`} />
                          )}
                        </div>
                        
                        <div className="flex-1">
                          <div className="flex justify-between">
                            <div>
                              <h4 className="font-medium">
                                {alert.type} {alert.token_symbol || 'Tokens'}
                              </h4>
                              <p className="text-sm text-muted-foreground">
                                {alert.wallet_address && formatAddress(alert.wallet_address)}
                                {alert.chain && ` on ${alert.chain}`}
                              </p>
                            </div>
                            <div className="text-right">
                              <p className="font-medium">{alert.value_usd && formatUSD(alert.value_usd)}</p>
                              <p className="text-sm text-muted-foreground">
                                {alert.timestamp && formatDistanceToNow(new Date(alert.timestamp), { addSuffix: true })}
                              </p>
                            </div>
                          </div>
                          
                          {alert.transaction_hash && (
                            <div className="mt-2 text-xs font-mono text-muted-foreground">
                              TX: {formatAddress(alert.transaction_hash)}
                            </div>
                          )}
                        </div>
                      </div>
                    ))}
                </div>
              )}
            </CardContent>
            <CardFooter className="flex justify-between">
              <div className="text-sm text-muted-foreground">
                {whaleMonitor.data?.stats && (
                  <>Showing {whaleMonitor.data.alerts.filter(a => a.type !== 'COORDINATION').length} movements</>
                )}
              </div>
              <Button variant="outline" size="sm" onClick={handleRefresh}>
                <RefreshCcw className="h-4 w-4 mr-2" />
                Refresh
              </Button>
            </CardFooter>
          </Card>
        </TabsContent>
        
        {/* Coordination Tab */}
        <TabsContent value="coordination" className="space-y-6">
          {/* Coordination Overview */}
          <Card>
            <CardHeader>
              <CardTitle>Whale Coordination Patterns</CardTitle>
              <CardDescription>
                Detected patterns of coordinated activity between whale wallets
              </CardDescription>
            </CardHeader>
            <CardContent>
              {whaleDetection.isLoading || whaleMonitor.isLoading ? (
                <div className="space-y-2">
                  <Skeleton className="h-16 w-full" />
                  <Skeleton className="h-16 w-full" />
                  <Skeleton className="h-16 w-full" />
                </div>
              ) : (!whaleDetection.data?.coordination_groups.length && 
                   !whaleMonitor.data?.alerts.filter(a => a.type === 'COORDINATION').length) ? (
                <div className="text-center py-6 text-muted-foreground">
                  <Network className="mx-auto h-8 w-8 mb-2" />
                  <p>No coordination patterns detected</p>
                  <p className="text-sm">Try adjusting detection settings or monitoring more wallets</p>
                </div>
              ) : (
                <div className="space-y-6">
                  {/* Coordination groups from whale detection */}
                  {whaleDetection.data?.coordination_groups.map((group, index) => (
                    <div key={index} className="p-4 rounded-lg border">
                      <div className="flex justify-between items-start mb-2">
                        <div>
                          <h4 className="font-medium flex items-center gap-2">
                            <Network className="h-4 w-4" />
                            {group.pattern_type} Pattern
                            <Badge variant={group.confidence > 0.8 ? "destructive" : "outline"}>
                              {Math.round(group.confidence * 100)}% confidence
                            </Badge>
                          </h4>
                          <p className="text-sm text-muted-foreground">
                            {group.wallets.length} wallets, {group.movement_count} movements
                          </p>
                        </div>
                        <div className="text-right">
                          <p className="font-medium">{formatUSD(group.total_value_usd)}</p>
                          <p className="text-xs text-muted-foreground">
                            {formatDistanceToNow(new Date(group.start_time), { addSuffix: true })}
                          </p>
                        </div>
                      </div>
                      
                      <div className="mt-2">
                        <h5 className="text-sm font-medium mb-1">Involved Wallets</h5>
                        <div className="flex flex-wrap gap-2">
                          {group.wallets.map((wallet) => (
                            <Badge key={wallet} variant="secondary" className="font-mono">
                              {formatAddress(wallet)}
                            </Badge>
                          ))}
                        </div>
                      </div>
                      
                      <div className="mt-4 flex justify-end">
                        <Button variant="outline" size="sm">
                          View Details
                        </Button>
                      </div>
                    </div>
                  ))}
                  
                  {/* Coordination alerts from monitoring */}
                  {whaleMonitor.data?.alerts
                    .filter(alert => alert.type === 'COORDINATION')
                    .map((alert, index) => (
                      <div key={`monitor-${index}`} className="p-4 rounded-lg border">
                        <div className="flex justify-between items-start mb-2">
                          <div>
                            <h4 className="font-medium flex items-center gap-2">
                              <Network className="h-4 w-4" />
                              {alert.pattern_type} Pattern
                              {alert.confidence && (
                                <Badge variant={alert.confidence > 0.8 ? "destructive" : "outline"}>
                                  {Math.round(alert.confidence * 100)}% confidence
                                </Badge>
                              )}
                            </h4>
                            <p className="text-sm text-muted-foreground">
                              {alert.wallets_involved} wallets involved
                            </p>
                          </div>
                          <div className="text-right">
                            <p className="font-medium">{formatUSD(alert.value_usd || 0)}</p>
                            <p className="text-xs text-muted-foreground">
                              {alert.timestamp && formatDistanceToNow(new Date(alert.timestamp), { addSuffix: true })}
                            </p>
                          </div>
                        </div>
                        
                        <div className="mt-4 flex justify-end">
                          <Button variant="outline" size="sm">
                            View Details
                          </Button>
                        </div>
                      </div>
                    ))}
                </div>
              )}
            </CardContent>
          </Card>
          
          {/* Pattern Types */}
          <Card>
            <CardHeader>
              <CardTitle>Coordination Pattern Types</CardTitle>
              <CardDescription>
                Understanding different whale coordination patterns
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid md:grid-cols-3 gap-4">
                <div className="p-4 rounded-lg border">
                  <h4 className="font-medium flex items-center gap-2 mb-2">
                    <Badge variant="outline">DISTRIBUTION</Badge>
                  </h4>
                  <p className="text-sm">
                    One wallet sends funds to multiple recipients in a short time window.
                    Often indicates token distribution or splitting funds to hide their source.
                  </p>
                </div>
                
                <div className="p-4 rounded-lg border">
                  <h4 className="font-medium flex items-center gap-2 mb-2">
                    <Badge variant="outline">ACCUMULATION</Badge>
                  </h4>
                  <p className="text-sm">
                    Multiple wallets send funds to a single recipient in a short time window.
                    May indicate fund consolidation or coordinated buying.
                  </p>
                </div>
                
                <div className="p-4 rounded-lg border">
                  <h4 className="font-medium flex items-center gap-2 mb-2">
                    <Badge variant="outline">CIRCULAR</Badge>
                  </h4>
                  <p className="text-sm">
                    Funds move in a circular pattern through multiple wallets, eventually returning to the origin.
                    Often associated with wash trading or layering techniques.
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default WhaleDashboard;
