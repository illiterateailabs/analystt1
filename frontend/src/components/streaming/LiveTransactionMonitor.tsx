import React, { useState, useEffect, useRef, useCallback } from 'react';
import { 
  Card, 
  CardContent, 
  CardHeader, 
  CardTitle, 
  CardDescription 
} from '@/components/ui/card';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Switch } from '@/components/ui/switch';
import { Slider } from '@/components/ui/slider';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover';
import { Bell, AlertTriangle, Filter, RefreshCw, Wifi, WifiOff, ExternalLink, BarChart2 } from 'lucide-react';
import { toast } from '@/hooks/useToast';
import { useAuth } from '@/hooks/useAuth';
import { Progress } from '@/components/ui/progress';

// Chart components
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, LineChart, Line } from 'recharts';

// Transaction types
interface Transaction {
  id: string;
  timestamp: string;
  from_address: string;
  to_address: string;
  amount: number;
  amount_usd: number;
  chain_id: string;
  transaction_type: string;
  risk_score?: number;
  is_high_risk?: boolean;
  tenant_id?: string;
  tags?: string[];
  method?: string;
}

interface TransactionAlert {
  id: string;
  timestamp: string;
  message: string;
  severity: 'low' | 'medium' | 'high';
  transaction_id: string;
  acknowledged: boolean;
}

interface FilterState {
  minAmount: number;
  maxAmount: number;
  minRiskScore: number;
  transactionTypes: string[];
  chains: string[];
  highRiskOnly: boolean;
  searchQuery: string;
}

interface ChartData {
  name: string;
  value: number;
  risk?: number;
}

interface LiveTransactionMonitorProps {
  tenantId?: string;
  defaultFilters?: Partial<FilterState>;
  autoConnect?: boolean;
  maxTransactions?: number;
  showRiskScores?: boolean;
  onTransactionSelected?: (transaction: Transaction) => void;
  onAlertTriggered?: (alert: TransactionAlert) => void;
}

const DEFAULT_WS_URL = process.env.NEXT_PUBLIC_WS_TX_STREAM_URL || 'ws://localhost:8000/api/v1/ws/tx_stream';
const MAX_TRANSACTIONS = 100;
const RISK_THRESHOLD = 0.7;

const LiveTransactionMonitor: React.FC<LiveTransactionMonitorProps> = ({
  tenantId,
  defaultFilters,
  autoConnect = true,
  maxTransactions = MAX_TRANSACTIONS,
  showRiskScores = true,
  onTransactionSelected,
  onAlertTriggered,
}) => {
  // State
  const [transactions, setTransactions] = useState<Transaction[]>([]);
  const [alerts, setAlerts] = useState<TransactionAlert[]>([]);
  const [isConnected, setIsConnected] = useState<boolean>(false);
  const [isConnecting, setIsConnecting] = useState<boolean>(false);
  const [connectionAttempts, setConnectionAttempts] = useState<number>(0);
  const [activeTab, setActiveTab] = useState<string>('live');
  const [chartData, setChartData] = useState<ChartData[]>([]);
  const [alertsEnabled, setAlertsEnabled] = useState<boolean>(true);
  const [alertThreshold, setAlertThreshold] = useState<number>(RISK_THRESHOLD);
  const [selectedChain, setSelectedChain] = useState<string>('all');
  const [selectedTenant, setSelectedTenant] = useState<string>(tenantId || 'all');
  
  // Filters
  const [filters, setFilters] = useState<FilterState>({
    minAmount: 0,
    maxAmount: 1000000,
    minRiskScore: 0,
    transactionTypes: [],
    chains: [],
    highRiskOnly: false,
    searchQuery: '',
    ...defaultFilters,
  });

  // Refs
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const { user } = useAuth();

  // Available chains and transaction types (would be fetched from API in a real app)
  const availableChains = [
    { id: 'ethereum', name: 'Ethereum' },
    { id: 'polygon', name: 'Polygon' },
    { id: 'binance', name: 'BNB Chain' },
    { id: 'arbitrum', name: 'Arbitrum' },
    { id: 'optimism', name: 'Optimism' },
  ];

  const transactionTypes = [
    { id: 'transfer', name: 'Transfer' },
    { id: 'swap', name: 'Swap' },
    { id: 'mint', name: 'Mint' },
    { id: 'burn', name: 'Burn' },
    { id: 'approve', name: 'Approve' },
  ];

  // Connect to WebSocket
  const connectWebSocket = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;
    
    setIsConnecting(true);
    setConnectionAttempts(prev => prev + 1);
    
    // Build WebSocket URL with query parameters
    let wsUrl = DEFAULT_WS_URL;
    const params = new URLSearchParams();
    
    if (selectedTenant && selectedTenant !== 'all') {
      params.append('tenant_id', selectedTenant);
    }
    
    if (selectedChain && selectedChain !== 'all') {
      params.append('chain_id', selectedChain);
    }
    
    if (filters.highRiskOnly) {
      params.append('high_risk_only', 'true');
    }
    
    if (params.toString()) {
      wsUrl += '?' + params.toString();
    }
    
    const ws = new WebSocket(wsUrl);
    
    ws.onopen = () => {
      console.log('WebSocket connected');
      setIsConnected(true);
      setIsConnecting(false);
      setConnectionAttempts(0);
    };
    
    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        handleIncomingTransaction(data);
      } catch (error) {
        console.error('Error parsing WebSocket message:', error);
      }
    };
    
    ws.onclose = () => {
      console.log('WebSocket disconnected');
      setIsConnected(false);
      setIsConnecting(false);
      
      // Attempt to reconnect after delay
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      
      reconnectTimeoutRef.current = setTimeout(() => {
        if (connectionAttempts < 5) {
          connectWebSocket();
        } else {
          toast({
            title: 'Connection failed',
            description: 'Could not connect to transaction stream after multiple attempts.',
            variant: 'destructive',
          });
        }
      }, 2000 * Math.min(connectionAttempts, 10));
    };
    
    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      toast({
        title: 'Connection error',
        description: 'Error connecting to transaction stream.',
        variant: 'destructive',
      });
    };
    
    wsRef.current = ws;
  }, [selectedTenant, selectedChain, filters.highRiskOnly, connectionAttempts]);

  // Disconnect WebSocket
  const disconnectWebSocket = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }
    
    setIsConnected(false);
    setIsConnecting(false);
  }, []);

  // Handle incoming transaction
  const handleIncomingTransaction = useCallback((transaction: Transaction) => {
    // Add timestamp if not present
    if (!transaction.timestamp) {
      transaction.timestamp = new Date().toISOString();
    }
    
    // Apply filters
    if (shouldFilterTransaction(transaction)) {
      return;
    }
    
    // Check for alerts
    if (alertsEnabled && shouldTriggerAlert(transaction)) {
      const alert: TransactionAlert = {
        id: `alert-${Date.now()}-${transaction.id}`,
        timestamp: new Date().toISOString(),
        message: `High-risk transaction detected: ${transaction.amount_usd.toFixed(2)} USD`,
        severity: getSeverity(transaction),
        transaction_id: transaction.id,
        acknowledged: false,
      };
      
      setAlerts(prev => [alert, ...prev].slice(0, 50));
      
      if (onAlertTriggered) {
        onAlertTriggered(alert);
      }
      
      // Show toast notification
      toast({
        title: 'High Risk Transaction',
        description: `${transaction.amount_usd.toFixed(2)} USD - Risk score: ${(transaction.risk_score || 0).toFixed(2)}`,
        variant: 'destructive',
      });
    }
    
    // Add to transactions list
    setTransactions(prev => {
      const newTransactions = [transaction, ...prev].slice(0, maxTransactions);
      updateChartData(newTransactions);
      return newTransactions;
    });
  }, [alertsEnabled, maxTransactions, onAlertTriggered]);

  // Update chart data
  const updateChartData = useCallback((transactions: Transaction[]) => {
    // Group transactions by hour
    const hourlyData: Record<string, { value: number, risk: number, count: number }> = {};
    
    transactions.forEach(tx => {
      const date = new Date(tx.timestamp);
      const hour = date.getHours();
      const key = `${hour}:00`;
      
      if (!hourlyData[key]) {
        hourlyData[key] = { value: 0, risk: 0, count: 0 };
      }
      
      hourlyData[key].value += tx.amount_usd;
      hourlyData[key].risk += tx.risk_score || 0;
      hourlyData[key].count += 1;
    });
    
    // Convert to chart data format
    const chartData: ChartData[] = Object.entries(hourlyData)
      .map(([name, data]) => ({
        name,
        value: data.value,
        risk: data.count > 0 ? data.risk / data.count : 0,
      }))
      .sort((a, b) => {
        const hourA = parseInt(a.name.split(':')[0]);
        const hourB = parseInt(b.name.split(':')[0]);
        return hourA - hourB;
      });
    
    setChartData(chartData);
  }, []);

  // Filter transaction
  const shouldFilterTransaction = useCallback((transaction: Transaction) => {
    // Amount filter
    if (transaction.amount_usd < filters.minAmount || 
        (filters.maxAmount > 0 && transaction.amount_usd > filters.maxAmount)) {
      return true;
    }
    
    // Risk score filter
    if (showRiskScores && 
        transaction.risk_score !== undefined && 
        transaction.risk_score < filters.minRiskScore) {
      return true;
    }
    
    // High risk only filter
    if (filters.highRiskOnly && 
        (!transaction.is_high_risk && 
         (!transaction.risk_score || transaction.risk_score < RISK_THRESHOLD))) {
      return true;
    }
    
    // Transaction type filter
    if (filters.transactionTypes.length > 0 && 
        !filters.transactionTypes.includes(transaction.transaction_type)) {
      return true;
    }
    
    // Chain filter
    if (filters.chains.length > 0 && 
        !filters.chains.includes(transaction.chain_id)) {
      return true;
    }
    
    // Search query filter
    if (filters.searchQuery) {
      const query = filters.searchQuery.toLowerCase();
      const searchableFields = [
        transaction.from_address,
        transaction.to_address,
        transaction.transaction_type,
        transaction.chain_id,
        transaction.id,
      ].map(field => (field || '').toLowerCase());
      
      if (!searchableFields.some(field => field.includes(query))) {
        return true;
      }
    }
    
    return false;
  }, [filters, showRiskScores]);

  // Alert triggers
  const shouldTriggerAlert = useCallback((transaction: Transaction) => {
    // High risk score
    if (transaction.risk_score && transaction.risk_score >= alertThreshold) {
      return true;
    }
    
    // Explicitly marked as high risk
    if (transaction.is_high_risk) {
      return true;
    }
    
    // Large transaction (over $100,000)
    if (transaction.amount_usd >= 100000) {
      return true;
    }
    
    return false;
  }, [alertThreshold]);

  // Get alert severity
  const getSeverity = (transaction: Transaction): 'low' | 'medium' | 'high' => {
    if (!transaction.risk_score) return 'low';
    
    if (transaction.risk_score >= 0.9) return 'high';
    if (transaction.risk_score >= 0.7) return 'medium';
    return 'low';
  };

  // Acknowledge alert
  const acknowledgeAlert = (alertId: string) => {
    setAlerts(prev => 
      prev.map(alert => 
        alert.id === alertId ? { ...alert, acknowledged: true } : alert
      )
    );
  };

  // Clear all alerts
  const clearAlerts = () => {
    setAlerts([]);
  };

  // Update filter
  const updateFilter = (key: keyof FilterState, value: any) => {
    setFilters(prev => ({ ...prev, [key]: value }));
  };

  // Reset filters
  const resetFilters = () => {
    setFilters({
      minAmount: 0,
      maxAmount: 1000000,
      minRiskScore: 0,
      transactionTypes: [],
      chains: [],
      highRiskOnly: false,
      searchQuery: '',
      ...defaultFilters,
    });
  };

  // Format address for display
  const formatAddress = (address: string) => {
    if (!address) return '';
    return `${address.substring(0, 6)}...${address.substring(address.length - 4)}`;
  };

  // Format timestamp for display
  const formatTimestamp = (timestamp: string) => {
    try {
      const date = new Date(timestamp);
      return date.toLocaleTimeString();
    } catch (e) {
      return timestamp;
    }
  };

  // Get risk color
  const getRiskColor = (riskScore?: number) => {
    if (riskScore === undefined) return 'bg-gray-200';
    if (riskScore >= 0.8) return 'bg-red-500';
    if (riskScore >= 0.6) return 'bg-orange-500';
    if (riskScore >= 0.4) return 'bg-yellow-500';
    if (riskScore >= 0.2) return 'bg-green-500';
    return 'bg-blue-500';
  };

  // Connect/disconnect on mount/unmount
  useEffect(() => {
    if (autoConnect) {
      connectWebSocket();
    }
    
    return () => {
      disconnectWebSocket();
    };
  }, [autoConnect, connectWebSocket, disconnectWebSocket]);

  // Reconnect when tenant or chain selection changes
  useEffect(() => {
    if (isConnected) {
      disconnectWebSocket();
      connectWebSocket();
    }
  }, [selectedTenant, selectedChain, disconnectWebSocket, connectWebSocket, isConnected]);

  return (
    <div className="w-full space-y-4">
      <div className="flex justify-between items-center">
        <div className="flex items-center space-x-2">
          <h2 className="text-2xl font-bold">Live Transaction Monitor</h2>
          {isConnected ? (
            <Badge variant="outline" className="bg-green-100 text-green-800 flex items-center">
              <Wifi className="h-3 w-3 mr-1" /> Connected
            </Badge>
          ) : isConnecting ? (
            <Badge variant="outline" className="bg-yellow-100 text-yellow-800 flex items-center">
              <RefreshCw className="h-3 w-3 mr-1 animate-spin" /> Connecting...
            </Badge>
          ) : (
            <Badge variant="outline" className="bg-red-100 text-red-800 flex items-center">
              <WifiOff className="h-3 w-3 mr-1" /> Disconnected
            </Badge>
          )}
        </div>
        
        <div className="flex items-center space-x-2">
          {!isConnected && !isConnecting ? (
            <Button onClick={connectWebSocket} size="sm" variant="default">
              Connect
            </Button>
          ) : (
            <Button onClick={disconnectWebSocket} size="sm" variant="outline">
              Disconnect
            </Button>
          )}
          
          <Popover>
            <PopoverTrigger asChild>
              <Button size="sm" variant="outline">
                <Filter className="h-4 w-4 mr-1" /> Filters
              </Button>
            </PopoverTrigger>
            <PopoverContent className="w-80">
              <div className="space-y-4">
                <h3 className="font-medium">Transaction Filters</h3>
                
                <div className="space-y-2">
                  <Label>Amount Range (USD)</Label>
                  <div className="flex items-center space-x-2">
                    <Input
                      type="number"
                      value={filters.minAmount}
                      onChange={(e) => updateFilter('minAmount', Number(e.target.value))}
                      placeholder="Min"
                      className="w-1/2"
                    />
                    <span>to</span>
                    <Input
                      type="number"
                      value={filters.maxAmount}
                      onChange={(e) => updateFilter('maxAmount', Number(e.target.value))}
                      placeholder="Max"
                      className="w-1/2"
                    />
                  </div>
                </div>
                
                {showRiskScores && (
                  <div className="space-y-2">
                    <Label>Minimum Risk Score</Label>
                    <div className="flex items-center space-x-2">
                      <Slider
                        value={[filters.minRiskScore]}
                        min={0}
                        max={1}
                        step={0.1}
                        onValueChange={(value) => updateFilter('minRiskScore', value[0])}
                      />
                      <span className="w-8 text-right">{filters.minRiskScore.toFixed(1)}</span>
                    </div>
                  </div>
                )}
                
                <div className="space-y-2">
                  <Label>Chain</Label>
                  <Select
                    value={selectedChain}
                    onValueChange={setSelectedChain}
                  >
                    <SelectTrigger>
                      <SelectValue placeholder="Select chain" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="all">All Chains</SelectItem>
                      {availableChains.map((chain) => (
                        <SelectItem key={chain.id} value={chain.id}>
                          {chain.name}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                
                {user?.role === 'admin' && (
                  <div className="space-y-2">
                    <Label>Tenant</Label>
                    <Select
                      value={selectedTenant}
                      onValueChange={setSelectedTenant}
                    >
                      <SelectTrigger>
                        <SelectValue placeholder="Select tenant" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="all">All Tenants</SelectItem>
                        <SelectItem value="tenant1">Tenant 1</SelectItem>
                        <SelectItem value="tenant2">Tenant 2</SelectItem>
                        <SelectItem value="tenant3">Tenant 3</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                )}
                
                <div className="flex items-center space-x-2">
                  <Switch
                    id="high-risk-only"
                    checked={filters.highRiskOnly}
                    onCheckedChange={(checked) => updateFilter('highRiskOnly', checked)}
                  />
                  <Label htmlFor="high-risk-only">High risk only</Label>
                </div>
                
                <div className="pt-2">
                  <Button onClick={resetFilters} variant="outline" size="sm" className="w-full">
                    Reset Filters
                  </Button>
                </div>
              </div>
            </PopoverContent>
          </Popover>
          
          <Popover>
            <PopoverTrigger asChild>
              <Button size="sm" variant="outline" className="relative">
                <Bell className="h-4 w-4" />
                {alerts.filter(a => !a.acknowledged).length > 0 && (
                  <span className="absolute -top-1 -right-1 bg-red-500 text-white rounded-full w-4 h-4 text-xs flex items-center justify-center">
                    {alerts.filter(a => !a.acknowledged).length}
                  </span>
                )}
              </Button>
            </PopoverTrigger>
            <PopoverContent className="w-80">
              <div className="space-y-4">
                <div className="flex justify-between items-center">
                  <h3 className="font-medium">Alerts</h3>
                  <div className="flex items-center space-x-2">
                    <Switch
                      id="alerts-enabled"
                      checked={alertsEnabled}
                      onCheckedChange={setAlertsEnabled}
                    />
                    <Label htmlFor="alerts-enabled">Enabled</Label>
                  </div>
                </div>
                
                <div className="space-y-2">
                  <Label>Alert Threshold</Label>
                  <div className="flex items-center space-x-2">
                    <Slider
                      value={[alertThreshold]}
                      min={0}
                      max={1}
                      step={0.1}
                      onValueChange={(value) => setAlertThreshold(value[0])}
                    />
                    <span className="w-8 text-right">{alertThreshold.toFixed(1)}</span>
                  </div>
                </div>
                
                <div className="max-h-60 overflow-y-auto space-y-2">
                  {alerts.length === 0 ? (
                    <p className="text-sm text-gray-500">No alerts</p>
                  ) : (
                    alerts.map((alert) => (
                      <Alert
                        key={alert.id}
                        variant={alert.acknowledged ? "outline" : "default"}
                        className={
                          alert.severity === 'high'
                            ? 'border-red-500 bg-red-50'
                            : alert.severity === 'medium'
                            ? 'border-orange-500 bg-orange-50'
                            : 'border-yellow-500 bg-yellow-50'
                        }
                      >
                        <AlertTriangle className="h-4 w-4" />
                        <AlertTitle className="text-sm font-medium">
                          {alert.severity.charAt(0).toUpperCase() + alert.severity.slice(1)} Risk
                        </AlertTitle>
                        <AlertDescription className="text-xs">
                          <div className="flex justify-between">
                            <span>{alert.message}</span>
                            <span className="text-xs text-gray-500">
                              {formatTimestamp(alert.timestamp)}
                            </span>
                          </div>
                          {!alert.acknowledged && (
                            <Button
                              size="sm"
                              variant="ghost"
                              className="mt-1 h-6 text-xs"
                              onClick={() => acknowledgeAlert(alert.id)}
                            >
                              Acknowledge
                            </Button>
                          )}
                        </AlertDescription>
                      </Alert>
                    ))
                  )}
                </div>
                
                {alerts.length > 0 && (
                  <Button onClick={clearAlerts} variant="outline" size="sm" className="w-full">
                    Clear All
                  </Button>
                )}
              </div>
            </PopoverContent>
          </Popover>
        </div>
      </div>
      
      <div className="flex items-center space-x-2">
        <Input
          placeholder="Search transactions..."
          value={filters.searchQuery}
          onChange={(e) => updateFilter('searchQuery', e.target.value)}
          className="max-w-sm"
        />
        <Badge variant="outline">
          {transactions.length} transactions
        </Badge>
      </div>
      
      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList>
          <TabsTrigger value="live">Live Feed</TabsTrigger>
          <TabsTrigger value="analytics">Analytics</TabsTrigger>
          <TabsTrigger value="alerts">Alerts</TabsTrigger>
        </TabsList>
        
        <TabsContent value="live" className="space-y-4">
          <Card>
            <CardContent className="p-0">
              <div className="rounded-md border">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Time</TableHead>
                      <TableHead>From</TableHead>
                      <TableHead>To</TableHead>
                      <TableHead>Amount (USD)</TableHead>
                      <TableHead>Type</TableHead>
                      <TableHead>Chain</TableHead>
                      {showRiskScores && <TableHead>Risk</TableHead>}
                      <TableHead></TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {transactions.length === 0 ? (
                      <TableRow>
                        <TableCell colSpan={showRiskScores ? 8 : 7} className="text-center text-muted-foreground h-24">
                          No transactions yet
                        </TableCell>
                      </TableRow>
                    ) : (
                      transactions.map((tx) => (
                        <TableRow 
                          key={tx.id}
                          className={tx.is_high_risk ? 'bg-red-50' : ''}
                          onClick={() => onTransactionSelected?.(tx)}
                        >
                          <TableCell>{formatTimestamp(tx.timestamp)}</TableCell>
                          <TableCell className="font-mono text-xs">
                            {formatAddress(tx.from_address)}
                          </TableCell>
                          <TableCell className="font-mono text-xs">
                            {formatAddress(tx.to_address)}
                          </TableCell>
                          <TableCell>
                            ${tx.amount_usd.toLocaleString(undefined, { 
                              minimumFractionDigits: 2,
                              maximumFractionDigits: 2
                            })}
                          </TableCell>
                          <TableCell>
                            <Badge variant="outline">
                              {tx.transaction_type || 'Transfer'}
                            </Badge>
                          </TableCell>
                          <TableCell>
                            <Badge variant="secondary">
                              {tx.chain_id}
                            </Badge>
                          </TableCell>
                          {showRiskScores && (
                            <TableCell>
                              {tx.risk_score !== undefined ? (
                                <div className="flex items-center space-x-2">
                                  <div className="w-full bg-gray-200 rounded-full h-2.5">
                                    <div 
                                      className={`h-2.5 rounded-full ${getRiskColor(tx.risk_score)}`}
                                      style={{ width: `${(tx.risk_score * 100)}%` }}
                                    ></div>
                                  </div>
                                  <span className="text-xs font-medium">
                                    {(tx.risk_score * 100).toFixed(0)}%
                                  </span>
                                </div>
                              ) : (
                                <span className="text-xs text-gray-500">N/A</span>
                              )}
                            </TableCell>
                          )}
                          <TableCell>
                            <Button 
                              variant="ghost" 
                              size="sm"
                              onClick={(e) => {
                                e.stopPropagation();
                                window.open(`/transaction/${tx.id}`, '_blank');
                              }}
                            >
                              <ExternalLink className="h-4 w-4" />
                            </Button>
                          </TableCell>
                        </TableRow>
                      ))
                    )}
                  </TableBody>
                </Table>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
        
        <TabsContent value="analytics">
          <Card>
            <CardHeader>
              <CardTitle>Transaction Volume</CardTitle>
              <CardDescription>
                Transaction volume by hour with risk overlay
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="h-80">
                {chartData.length > 0 ? (
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart
                      data={chartData}
                      margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="name" />
                      <YAxis yAxisId="left" orientation="left" stroke="#8884d8" />
                      <YAxis yAxisId="right" orientation="right" stroke="#82ca9d" />
                      <Tooltip />
                      <Bar yAxisId="left" dataKey="value" fill="#8884d8" name="Volume (USD)" />
                      <Line yAxisId="right" type="monotone" dataKey="risk" stroke="#ff7300" name="Avg Risk" />
                    </BarChart>
                  </ResponsiveContainer>
                ) : (
                  <div className="h-full flex items-center justify-center text-muted-foreground">
                    No data available
                  </div>
                )}
              </div>
            </CardContent>
          </Card>
        </TabsContent>
        
        <TabsContent value="alerts">
          <Card>
            <CardHeader>
              <CardTitle>Alert History</CardTitle>
              <CardDescription>
                Recent alerts and notifications
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {alerts.length === 0 ? (
                  <div className="text-center p-4 text-muted-foreground">
                    No alerts yet
                  </div>
                ) : (
                  alerts.map((alert) => (
                    <Alert
                      key={alert.id}
                      variant={alert.acknowledged ? "outline" : "default"}
                      className={
                        alert.severity === 'high'
                          ? 'border-red-500 bg-red-50'
                          : alert.severity === 'medium'
                          ? 'border-orange-500 bg-orange-50'
                          : 'border-yellow-500 bg-yellow-50'
                      }
                    >
                      <AlertTriangle className="h-4 w-4" />
                      <AlertTitle>
                        {alert.severity.charAt(0).toUpperCase() + alert.severity.slice(1)} Risk Alert
                      </AlertTitle>
                      <AlertDescription>
                        <div className="flex justify-between">
                          <span>{alert.message}</span>
                          <span className="text-sm text-gray-500">
                            {formatTimestamp(alert.timestamp)}
                          </span>
                        </div>
                        <div className="flex space-x-2 mt-2">
                          {!alert.acknowledged && (
                            <Button
                              size="sm"
                              variant="outline"
                              onClick={() => acknowledgeAlert(alert.id)}
                            >
                              Acknowledge
                            </Button>
                          )}
                          <Button
                            size="sm"
                            variant="outline"
                            onClick={() => {
                              const tx = transactions.find(t => t.id === alert.transaction_id);
                              if (tx) {
                                onTransactionSelected?.(tx);
                              }
                            }}
                          >
                            View Transaction
                          </Button>
                        </div>
                      </AlertDescription>
                    </Alert>
                  ))
                )}
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default LiveTransactionMonitor;
