import React, { useState, useEffect, useCallback } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  TextField,
  Button,
  CircularProgress,
  Alert,
  AlertTitle,
  Slider,
  Switch,
  FormControlLabel,
  Grid,
  Chip,
  Paper,
  Divider,
  IconButton,
  Tooltip,
  InputAdornment,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  List,
  ListItem,
  ListItemText,
} from '@mui/material';
import { styled } from '@mui/material/styles';
import { useMutation } from '@tanstack/react-query';
import {
  analyzeCrossChainIdentity,
  CrossChainIdentityAnalysis,
  CrossChainWallet,
  CrossChainMovement,
  IdentityCluster,
  BridgeUsage,
} from '../../lib/api';
import { formatAddress, formatUSD } from '../../lib/utils';
import SearchIcon from '@mui/icons-material/Search';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import ErrorIcon from '@mui/icons-material/Error';
import WarningIcon from '@mui/icons-material/Warning';
import InfoIcon from '@mui/icons-material/Info';
import LinkIcon from '@mui/icons-material/Link';
import AccountBalanceWalletIcon from '@mui/icons-material/AccountBalanceWallet';
import SwapHorizIcon from '@mui/icons-material/SwapHoriz';
import PublicIcon from '@mui/icons-material/Public';
import ErrorBoundary from '../ui/ErrorBoundary';
import LoadingSpinner from '../ui/LoadingSpinner';

// Styled components for consistency
const StyledPanel = styled(Card)(({ theme }) => ({
  height: '100%',
  display: 'flex',
  flexDirection: 'column',
}));

const StyledPanelContent = styled(CardContent)(({ theme }) => ({
  flexGrow: 1,
  overflowY: 'auto',
  padding: theme.spacing(3),
}));

const MetricsContainer = styled(Paper)(({ theme }) => ({
  marginTop: theme.spacing(3),
  padding: theme.spacing(2),
  borderRadius: theme.shape.borderRadius,
}));

const MetricItem = styled(Box)(({ theme }) => ({
  display: 'flex',
  justifyContent: 'space-between',
  padding: theme.spacing(0.5, 0),
}));

const RiskScoreContainer = styled(Box)(({ theme }) => ({
  display: 'flex',
  alignItems: 'center',
  marginBottom: theme.spacing(1),
}));

const RiskScoreBar = styled(Box)<{ risk: number }>(({ theme, risk }) => ({
  height: 10,
  borderRadius: 5,
  width: '100%',
  backgroundColor: theme.palette.grey[200],
  position: 'relative',
  '&::after': {
    content: '""',
    position: 'absolute',
    top: 0,
    left: 0,
    height: '100%',
    width: `${risk}%`,
    borderRadius: 5,
    backgroundColor:
      risk >= 75 ? theme.palette.error.main :
      risk >= 40 ? theme.palette.warning.main :
      theme.palette.success.main,
  }
}));

const ChainBadge = styled(Chip)(({ theme, color }) => ({
  backgroundColor: color || theme.palette.primary.main,
  color: theme.palette.getContrastText(color || theme.palette.primary.main),
  marginRight: theme.spacing(0.5),
  marginBottom: theme.spacing(0.5),
}));

const getChainColor = (chainName: string) => {
  switch (chainName.toLowerCase()) {
    case 'ethereum': return '#627EEA';
    case 'polygon': return '#8247E5';
    case 'arbitrum': return '#28A0F0';
    case 'optimism': return '#FF0420';
    case 'bnb': return '#F3BA2F';
    case 'base': return '#0052FF';
    case 'avalanche': return '#E84142';
    case 'fantom': return '#13B5EC';
    case 'solana': return '#9945FF';
    default: return '#607D8B'; // Grey
  }
};

interface CrossChainIdentityPanelProps {
  initialWalletAddress?: string;
}

const CrossChainIdentityPanel: React.FC<CrossChainIdentityPanelProps> = ({ initialWalletAddress }) => {
  // Form state
  const [walletInput, setWalletInput] = useState<string>(initialWalletAddress || '');
  const [lookbackDays, setLookbackDays] = useState<number>(7);
  const [detectClusters, setDetectClusters] = useState<boolean>(true);
  const [detectMovements, setDetectMovements] = useState<boolean>(true);
  const [analyzeBridgeUsage, setAnalyzeBridgeUsage] = useState<boolean>(true);
  const [detectCoordination, setDetectCoordination] = useState<boolean>(true);
  const [minBridgeValue, setMinBridgeValue] = useState<number>(1000);
  const [crossChainTimeWindow, setCrossChainTimeWindow] = useState<number>(15);

  // Analysis mutation
  const {
    mutate: runAnalysis,
    isLoading,
    error,
    data: analysisResult,
    reset: resetAnalysis,
  } = useMutation<CrossChainIdentityAnalysis>({
    mutationFn: (params: {
      wallet_addresses: string[];
      lookback_days: number;
      detect_clusters: boolean;
      detect_cross_chain_movements: boolean;
      analyze_bridge_usage: boolean;
      detect_coordination: boolean;
      min_bridge_value_usd: number;
      cross_chain_time_window: number;
    }) => analyzeCrossChainIdentity(params),
  });

  // Handle analysis
  const handleAnalyze = () => {
    if (walletInput.trim()) {
      const addresses = walletInput.split(',').map(addr => addr.trim()).filter(Boolean);
      runAnalysis({
        wallet_addresses: addresses,
        lookback_days: lookbackDays,
        detect_clusters: detectClusters,
        detect_cross_chain_movements: detectMovements,
        analyze_bridge_usage: analyzeBridgeUsage,
        detect_coordination: detectCoordination,
        min_bridge_value_usd: minBridgeValue,
        cross_chain_time_window: crossChainTimeWindow,
      });
    }
  };

  // Render risk level chip
  const renderRiskLevelChip = (score: number) => {
    if (score >= 75) {
      return <Chip size="small" label="HIGH RISK" color="error" icon={<ErrorIcon />} />;
    } else if (score >= 40) {
      return <Chip size="small" label="MEDIUM RISK" color="warning" icon={<WarningIcon />} />;
    } else {
      return <Chip size="small" label="LOW RISK" color="success" icon={<InfoIcon />} />;
    }
  };

  return (
    <ErrorBoundary>
      <StyledPanel>
        <StyledPanelContent>
          <Typography variant="h5" gutterBottom>
            Cross-Chain Identity Analysis
          </Typography>
          <Typography variant="body2" color="text.secondary" paragraph>
            Analyze wallet identities and transaction patterns across multiple blockchain networks to detect cross-chain movements, identify potential identity clusters, and assess associated risks.
          </Typography>

          {/* Input Controls */}
          <Grid container spacing={2} alignItems="center">
            <Grid item xs={12} md={8}>
              <TextField
                label="Wallet Addresses (comma-separated)"
                fullWidth
                value={walletInput}
                onChange={(e) => setWalletInput(e.target.value)}
                placeholder="e.g., 0xabc..., 0xdef..."
                multiline
                rows={2}
                variant="outlined"
                InputProps={{
                  endAdornment: (
                    <InputAdornment position="end">
                      <Tooltip title="Analyze Identity">
                        <IconButton
                          edge="end"
                          onClick={handleAnalyze}
                          disabled={isLoading || !walletInput.trim()}
                        >
                          {isLoading ? <CircularProgress size={24} /> : <SearchIcon />}
                        </IconButton>
                      </Tooltip>
                    </InputAdornment>
                  ),
                }}
              />
            </Grid>
            <Grid item xs={12} md={4}>
              <Button
                variant="contained"
                color="primary"
                fullWidth
                onClick={handleAnalyze}
                disabled={isLoading || !walletInput.trim()}
                startIcon={isLoading ? <CircularProgress size={20} color="inherit" /> : <SearchIcon />}
              >
                {isLoading ? 'Analyzing...' : 'Analyze Identity'}
              </Button>
            </Grid>
            <Grid item xs={12} sm={6}>
              <Typography gutterBottom>Lookback Period: {lookbackDays} days</Typography>
              <Slider
                value={lookbackDays}
                onChange={(_e, newValue) => setLookbackDays(newValue as number)}
                aria-labelledby="lookback-days-slider"
                valueLabelDisplay="auto"
                min={1}
                max={30}
                step={1}
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <Typography gutterBottom>Min Bridge Value: {formatUSD(minBridgeValue)}</Typography>
              <Slider
                value={minBridgeValue}
                onChange={(_e, newValue) => setMinBridgeValue(newValue as number)}
                aria-labelledby="min-bridge-value-slider"
                valueLabelDisplay="auto"
                min={100}
                max={100000}
                step={100}
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <Typography gutterBottom>Cross-Chain Time Window: {crossChainTimeWindow} min</Typography>
              <Slider
                value={crossChainTimeWindow}
                onChange={(_e, newValue) => setCrossChainTimeWindow(newValue as number)}
                aria-labelledby="cross-chain-time-window-slider"
                valueLabelDisplay="auto"
                min={1}
                max={60}
                step={1}
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <FormControlLabel
                control={
                  <Switch
                    checked={detectClusters}
                    onChange={(e) => setDetectClusters(e.target.checked)}
                    name="detectClusters"
                    color="primary"
                  />
                }
                label="Detect Identity Clusters"
              />
              <FormControlLabel
                control={
                  <Switch
                    checked={detectMovements}
                    onChange={(e) => setDetectMovements(e.target.checked)}
                    name="detectMovements"
                    color="primary"
                  />
                }
                label="Detect Cross-Chain Movements"
              />
              <FormControlLabel
                control={
                  <Switch
                    checked={analyzeBridgeUsage}
                    onChange={(e) => setAnalyzeBridgeUsage(e.target.checked)}
                    name="analyzeBridgeUsage"
                    color="primary"
                  />
                }
                label="Analyze Bridge Usage"
              />
              <FormControlLabel
                control={
                  <Switch
                    checked={detectCoordination}
                    onChange={(e) => setDetectCoordination(e.target.checked)}
                    name="detectCoordination"
                    color="primary"
                  />
                }
                label="Detect Coordination Patterns"
              />
            </Grid>
            <Grid item xs={12} sx={{ display: 'flex', justifyContent: 'flex-end' }}>
              <Button
                variant="outlined"
                color="primary"
                onClick={() => resetAnalysis()}
                sx={{ mr: 1 }}
              >
                Clear
              </Button>
              {/* Export functionality can be added here */}
            </Grid>
          </Grid>

          {/* Error Display */}
          {error && (
            <Alert severity="error" sx={{ mt: 3 }}>
              <AlertTitle>Analysis Error</AlertTitle>
              {error instanceof Error ? error.message : 'An unexpected error occurred during analysis.'}
            </Alert>
          )}

          {/* Results Display */}
          {analysisResult ? (
            <>
              {/* Overall Risk Assessment */}
              <MetricsContainer elevation={1} sx={{ mt: 3 }}>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                  <Typography variant="h6">Overall Risk Assessment</Typography>
                  {renderRiskLevelChip(analysisResult.overall_risk_score)}
                </Box>
                <Divider sx={{ mb: 2 }} />
                <RiskScoreContainer>
                  <Typography variant="body2" color="text.secondary" sx={{ mr: 2, minWidth: '80px' }}>
                    Risk Score:
                  </Typography>
                  <Box sx={{ flexGrow: 1 }}>
                    <RiskScoreBar risk={analysisResult.overall_risk_score} />
                  </Box>
                  <Typography variant="body2" sx={{ ml: 2, fontWeight: 'bold' }}>
                    {analysisResult.overall_risk_score.toFixed(0)}/100
                  </Typography>
                </RiskScoreContainer>
                <Typography variant="subtitle2" gutterBottom sx={{ mt: 2 }}>Risk Factors:</Typography>
                {analysisResult.overall_risk_factors.length > 0 ? (
                  <ul style={{ margin: '8px 0', paddingLeft: '20px' }}>
                    {analysisResult.overall_risk_factors.map((factor, index) => (
                      <li key={index}>
                        <Typography variant="body2" color="text.secondary">
                          {factor}
                        </Typography>
                      </li>
                    ))}
                  </ul>
                ) : (
                  <Typography variant="body2" color="text.secondary">
                    No significant risk factors detected.
                  </Typography>
                )}
              </MetricsContainer>

              {/* Identity Clusters */}
              {detectClusters && analysisResult.identity_clusters.length > 0 && (
                <Box sx={{ mt: 3 }}>
                  <Typography variant="h6" gutterBottom>
                    Identity Clusters ({analysisResult.identity_clusters.length})
                  </Typography>
                  <Typography variant="body2" color="text.secondary" paragraph>
                    Groups of wallets that likely belong to the same entity based on cross-chain activity patterns.
                  </Typography>

                  {analysisResult.identity_clusters.map((cluster, index) => (
                    <Accordion key={cluster.cluster_id} sx={{ mb: 2 }}>
                      <AccordionSummary
                        expandIcon={<ExpandMoreIcon />}
                        aria-controls={`cluster-${index}-content`}
                        id={`cluster-${index}-header`}
                      >
                        <Box sx={{ display: 'flex', alignItems: 'center', width: '100%', justifyContent: 'space-between' }}>
                          <Box sx={{ display: 'flex', alignItems: 'center' }}>
                            <AccountBalanceWalletIcon sx={{ mr: 1 }} />
                            <Typography>
                              Cluster {index + 1}: {formatAddress(cluster.main_address)}
                            </Typography>
                            <Chip
                              size="small"
                              label={`${cluster.wallets.length} wallets`}
                              sx={{ ml: 1 }}
                            />
                            <Box sx={{ ml: 2 }}>
                              {cluster.chains.map(chain => (
                                <ChainBadge
                                  key={chain}
                                  label={chain}
                                  size="small"
                                  color={getChainColor(chain)}
                                />
                              ))}
                            </Box>
                          </Box>
                          <Box>
                            {renderRiskLevelChip(cluster.risk_score)}
                          </Box>
                        </Box>
                      </AccordionSummary>
                      <AccordionDetails>
                        <Grid container spacing={2}>
                          <Grid item xs={12} md={6}>
                            <Typography variant="subtitle2" gutterBottom>Wallets in Cluster:</Typography>
                            <Paper variant="outlined" sx={{ p: 2, maxHeight: '200px', overflow: 'auto' }}>
                              <List dense>
                                {cluster.wallets.map(wallet => (
                                  <ListItem key={`${wallet.address}-${wallet.chain_id}`} divider>
                                    <ListItemText
                                      primary={formatAddress(wallet.address)}
                                      secondary={`${wallet.chain_name} • ${formatUSD(wallet.total_value_usd)} • ${wallet.transaction_count} txs`}
                                    />
                                    <ChainBadge
                                      label={wallet.chain_name}
                                      size="small"
                                      color={getChainColor(wallet.chain_name)}
                                    />
                                  </ListItem>
                                ))}
                              </List>
                            </Paper>
                          </Grid>
                          <Grid item xs={12} md={6}>
                            <Typography variant="subtitle2" gutterBottom>Cluster Details:</Typography>
                            <Paper variant="outlined" sx={{ p: 2 }}>
                              <MetricItem>
                                <Typography variant="body2" color="text.secondary">Total Value:</Typography>
                                <Typography variant="body2">{formatUSD(cluster.total_value_usd)}</Typography>
                              </MetricItem>
                              <MetricItem>
                                <Typography variant="body2" color="text.secondary">First Seen:</Typography>
                                <Typography variant="body2">{cluster.first_seen ? new Date(cluster.first_seen).toLocaleString() : 'Unknown'}</Typography>
                              </MetricItem>
                              <MetricItem>
                                <Typography variant="body2" color="text.secondary">Last Seen:</Typography>
                                <Typography variant="body2">{cluster.last_seen ? new Date(cluster.last_seen).toLocaleString() : 'Unknown'}</Typography>
                              </MetricItem>
                              <MetricItem>
                                <Typography variant="body2" color="text.secondary">Confidence:</Typography>
                                <Typography variant="body2">{(cluster.confidence * 100).toFixed(0)}%</Typography>
                              </MetricItem>
                              <MetricItem>
                                <Typography variant="body2" color="text.secondary">Risk Score:</Typography>
                                <Typography variant="body2">{cluster.risk_score.toFixed(0)}/100</Typography>
                              </MetricItem>
                              {cluster.risk_factors.length > 0 && (
                                <>
                                  <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>Risk Factors:</Typography>
                                  <ul style={{ margin: '4px 0', paddingLeft: '20px' }}>
                                    {cluster.risk_factors.map((factor, i) => (
                                      <li key={i}>
                                        <Typography variant="body2">{factor}</Typography>
                                      </li>
                                    ))}
                                  </ul>
                                </>
                              )}
                              {cluster.coordination_patterns.length > 0 && (
                                <>
                                  <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>Coordination Patterns:</Typography>
                                  <Box sx={{ mt: 0.5 }}>
                                    {cluster.coordination_patterns.map((pattern, i) => (
                                      <Chip
                                        key={i}
                                        label={pattern.replace('_', ' ')}
                                        size="small"
                                        color="warning"
                                        sx={{ mr: 0.5, mb: 0.5 }}
                                      />
                                    ))}
                                  </Box>
                                </>
                              )}
                            </Paper>
                          </Grid>
                        </Grid>
                      </AccordionDetails>
                    </Accordion>
                  ))}
                </Box>
              )}

              {/* Cross-Chain Movements */}
              {detectMovements && analysisResult.cross_chain_movements.length > 0 && (
                <Box sx={{ mt: 3 }}>
                  <Typography variant="h6" gutterBottom>
                    Cross-Chain Movements ({analysisResult.cross_chain_movements.length})
                  </Typography>
                  <Typography variant="body2" color="text.secondary" paragraph>
                    Detected movements of assets between different blockchain networks.
                  </Typography>

                  <Grid container spacing={2}>
                    {analysisResult.cross_chain_movements.map((movement, index) => (
                      <Grid item xs={12} md={6} key={movement.id}>
                        <Paper
                          variant="outlined"
                          sx={{
                            p: 2,
                            borderColor: movement.risk_score >= 70 ? 'error.main' : movement.risk_score >= 40 ? 'warning.main' : 'divider'
                          }}
                        >
                          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 1 }}>
                            <Box sx={{ display: 'flex', alignItems: 'center' }}>
                              <SwapHorizIcon sx={{ mr: 1 }} />
                              <Typography variant="subtitle1">
                                {movement.source_chain_name} → {movement.destination_chain_name}
                              </Typography>
                            </Box>
                            {renderRiskLevelChip(movement.risk_score)}
                          </Box>

                          <Divider sx={{ my: 1 }} />

                          <MetricItem>
                            <Typography variant="body2" color="text.secondary">Wallet:</Typography>
                            <Typography variant="body2">{formatAddress(movement.wallet_address)}</Typography>
                          </MetricItem>
                          <MetricItem>
                            <Typography variant="body2" color="text.secondary">Value:</Typography>
                            <Typography variant="body2">{formatUSD(movement.value_usd)}</Typography>
                          </MetricItem>
                          <MetricItem>
                            <Typography variant="body2" color="text.secondary">Time Difference:</Typography>
                            <Typography variant="body2">
                              {movement.time_difference_minutes?.toFixed(1) || '?'} minutes
                            </Typography>
                          </MetricItem>
                          {movement.bridge_name && (
                            <MetricItem>
                              <Typography variant="body2" color="text.secondary">Bridge:</Typography>
                              <Typography variant="body2">
                                <Box sx={{ display: 'flex', alignItems: 'center' }}>
                                  <LinkIcon fontSize="small" sx={{ mr: 0.5 }} />
                                  {movement.bridge_name}
                                </Box>
                              </Typography>
                            </MetricItem>
                          )}
                          <MetricItem>
                            <Typography variant="body2" color="text.secondary">Source Time:</Typography>
                            <Typography variant="body2">
                              {new Date(movement.source_time).toLocaleString()}
                            </Typography>
                          </MetricItem>
                          <MetricItem>
                            <Typography variant="body2" color="text.secondary">Destination Time:</Typography>
                            <Typography variant="body2">
                              {movement.destination_time ? new Date(movement.destination_time).toLocaleString() : 'Unknown'}
                            </Typography>
                          </MetricItem>
                          <MetricItem>
                            <Typography variant="body2" color="text.secondary">Confidence:</Typography>
                            <Typography variant="body2">{(movement.confidence * 100).toFixed(0)}%</Typography>
                          </MetricItem>

                          {movement.risk_factors.length > 0 && (
                            <>
                              <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>Risk Factors:</Typography>
                              <ul style={{ margin: '4px 0', paddingLeft: '20px' }}>
                                {movement.risk_factors.map((factor, i) => (
                                  <li key={i}>
                                    <Typography variant="body2">{factor}</Typography>
                                  </li>
                                ))}
                              </ul>
                            </>
                          )}
                        </Paper>
                      </Grid>
                    ))}
                  </Grid>
                </Box>
              )}

              {/* Bridge Usage */}
              {analyzeBridgeUsage && analysisResult.bridge_usage.length > 0 && (
                <Box sx={{ mt: 3 }}>
                  <Typography variant="h6" gutterBottom>
                    Bridge Usage Analysis ({analysisResult.bridge_usage.length})
                  </Typography>
                  <Typography variant="body2" color="text.secondary" paragraph>
                    Statistics on bridge usage patterns between different blockchain networks.
                  </Typography>

                  <Grid container spacing={2}>
                    {analysisResult.bridge_usage.map((bridge, index) => (
                      <Grid item xs={12} md={6} lg={4} key={`${bridge.bridge_address}-${index}`}>
                        <Paper
                          variant="outlined"
                          sx={{
                            p: 2,
                            borderColor: bridge.risk_score >= 70 ? 'error.main' : bridge.risk_score >= 40 ? 'warning.main' : 'divider'
                          }}
                        >
                          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 1 }}>
                            <Box sx={{ display: 'flex', alignItems: 'center' }}>
                              <LinkIcon sx={{ mr: 1 }} />
                              <Typography variant="subtitle1">
                                {bridge.bridge_name || 'Unknown Bridge'}
                              </Typography>
                            </Box>
                            {renderRiskLevelChip(bridge.risk_score)}
                          </Box>

                          <Box sx={{ display: 'flex', alignItems: 'center', my: 1 }}>
                            <ChainBadge
                              label={bridge.source_chain_id}
                              size="small"
                              color={getChainColor(bridge.source_chain_id)}
                            />
                            <SwapHorizIcon sx={{ mx: 1 }} />
                            <ChainBadge
                              label={bridge.destination_chain_id}
                              size="small"
                              color={getChainColor(bridge.destination_chain_id)}
                            />
                          </Box>

                          <Divider sx={{ my: 1 }} />

                          <MetricItem>
                            <Typography variant="body2" color="text.secondary">Usage Count:</Typography>
                            <Typography variant="body2">{bridge.usage_count}</Typography>
                          </MetricItem>
                          <MetricItem>
                            <Typography variant="body2" color="text.secondary">Total Value:</Typography>
                            <Typography variant="body2">{formatUSD(bridge.total_value_usd)}</Typography>
                          </MetricItem>
                          <MetricItem>
                            <Typography variant="body2" color="text.secondary">Average Value:</Typography>
                            <Typography variant="body2">{formatUSD(bridge.average_value_usd)}</Typography>
                          </MetricItem>
                          <MetricItem>
                            <Typography variant="body2" color="text.secondary">First Used:</Typography>
                            <Typography variant="body2">
                              {bridge.first_used ? new Date(bridge.first_used).toLocaleString() : 'Unknown'}
                            </Typography>
                          </MetricItem>
                          <MetricItem>
                            <Typography variant="body2" color="text.secondary">Last Used:</Typography>
                            <Typography variant="body2">
                              {bridge.last_used ? new Date(bridge.last_used).toLocaleString() : 'Unknown'}
                            </Typography>
                          </MetricItem>

                          {bridge.risk_factors.length > 0 && (
                            <>
                              <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>Risk Factors:</Typography>
                              <ul style={{ margin: '4px 0', paddingLeft: '20px' }}>
                                {bridge.risk_factors.map((factor, i) => (
                                  <li key={i}>
                                    <Typography variant="body2">{factor}</Typography>
                                  </li>
                                ))}
                              </ul>
                            </>
                          )}
                        </Paper>
                      </Grid>
                    ))}
                  </Grid>
                </Box>
              )}

              {/* Multi-Chain Wallet Presence */}
              {analysisResult.wallets.length > 0 && (
                <Box sx={{ mt: 3 }}>
                  <Typography variant="h6" gutterBottom>
                    Multi-Chain Wallet Presence
                  </Typography>
                  <Typography variant="body2" color="text.secondary" paragraph>
                    Overview of wallet presence across different blockchain networks.
                  </Typography>

                  <Paper variant="outlined" sx={{ p: 2 }}>
                    <Grid container spacing={2}>
                      {analysisResult.wallets.map((wallet) => (
                        <Grid item xs={12} sm={6} md={4} key={`${wallet.address}-${wallet.chain_id}`}>
                          <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                            <PublicIcon sx={{ mr: 1 }} />
                            <Typography variant="subtitle2">
                              {formatAddress(wallet.address)}
                            </Typography>
                          </Box>
                          <Box sx={{ display: 'flex', alignItems: 'center', mb: 0.5 }}>
                            <ChainBadge
                              label={wallet.chain_name}
                              size="small"
                              color={getChainColor(wallet.chain_name)}
                            />
                          </Box>
                          <Box sx={{ pl: 1 }}>
                            <Typography variant="body2" color="text.secondary">
                              Value: {formatUSD(wallet.total_value_usd)}
                            </Typography>
                            <Typography variant="body2" color="text.secondary">
                              Transactions: {wallet.transaction_count}
                            </Typography>
                            <Typography variant="body2" color="text.secondary">
                              Tokens: {wallet.token_count}
                            </Typography>
                          </Box>
                        </Grid>
                      ))}
                    </Grid>
                  </Paper>
                </Box>
              )}
            </>
          ) : (
            isLoading ? (
              <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '400px' }}>
                <LoadingSpinner size="large" />
              </Box>
            ) : (
              <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '400px', flexDirection: 'column' }}>
                <InfoIcon sx={{ fontSize: 64, color: 'text.secondary', mb: 2 }} />
                <Typography variant="h6" color="text.secondary">
                  Enter wallet addresses and click "Analyze Identity" to detect cross-chain patterns
                </Typography>
              </Box>
            )
          )}
        </StyledPanelContent>
      </StyledPanel>
    </ErrorBoundary>
  );
};

export default CrossChainIdentityPanel;
