import React, { useState, useEffect, useRef, useCallback } from 'react';
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
} from '@mui/material';
import { styled } from '@mui/material/styles';
import { useMutation } from '@tanstack/react-query';
import { analyzeTransactionFlow, TransactionFlowAnalysis, TransactionFlowMetrics } from '../../lib/api';
import { formatAddress, formatUSD } from '../../lib/utils';
import { Network } from 'vis-network/standalone';
import { DataSet } from 'vis-data';
import SearchIcon from '@mui/icons-material/Search';
import ZoomInIcon from '@mui/icons-material/ZoomIn';
import ZoomOutIcon from '@mui/icons-material/ZoomOut';
import RefreshIcon from '@mui/icons-material/Refresh';
import DownloadIcon from '@mui/icons-material/Download';
import SaveAltIcon from '@mui/icons-material/SaveAlt';
import InfoIcon from '@mui/icons-material/Info';
import WarningIcon from '@mui/icons-material/Warning';
import ErrorIcon from '@mui/icons-material/Error';
import FitScreenIcon from '@mui/icons-material/FitScreen';
import ErrorBoundary from '../ui/ErrorBoundary';
import LoadingSpinner from '../ui/LoadingSpinner';

// Styled components for consistency with WalletAnalysisPanel
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

const GraphContainer = styled(Box)(({ theme }) => ({
  height: '600px',
  border: `1px solid ${theme.palette.divider}`,
  borderRadius: theme.shape.borderRadius,
  marginTop: theme.spacing(2),
  marginBottom: theme.spacing(2),
  position: 'relative',
}));

const ControlsContainer = styled(Box)(({ theme }) => ({
  position: 'absolute',
  top: theme.spacing(1),
  right: theme.spacing(1),
  zIndex: 10,
  display: 'flex',
  flexDirection: 'column',
  gap: theme.spacing(1),
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

const PatternItem = styled(Box)(({ theme }) => ({
  display: 'flex',
  alignItems: 'center',
  marginBottom: theme.spacing(1),
}));

const PatternChip = styled(Chip)(({ theme }) => ({
  marginRight: theme.spacing(1),
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

interface TransactionFlowPanelProps {
  initialWalletAddress?: string;
}

const TransactionFlowPanel: React.FC<TransactionFlowPanelProps> = ({ initialWalletAddress }) => {
  // Network visualization refs
  const graphRef = useRef<HTMLDivElement>(null);
  const networkRef = useRef<Network | null>(null);

  // Form state
  const [walletInput, setWalletInput] = useState<string>(initialWalletAddress || '');
  const [timeWindow, setTimeWindow] = useState<number>(24); // hours
  const [detectPatterns, setDetectPatterns] = useState<boolean>(true);
  const [valueThreshold, setValueThreshold] = useState<number>(10000); // USD

  // Analysis mutation
  const {
    mutate: runAnalysis,
    isLoading,
    error,
    data: analysisResult,
    reset: resetAnalysis,
  } = useMutation<TransactionFlowAnalysis>({
    mutationFn: (params: { wallet_addresses: string[], time_window_hours: number, detect_patterns: boolean, value_threshold_usd: number }) => 
      analyzeTransactionFlow(params),
  });

  // Handle analysis
  const handleAnalyze = () => {
    if (walletInput.trim()) {
      const addresses = walletInput.split(',').map(addr => addr.trim()).filter(Boolean);
      runAnalysis({
        wallet_addresses: addresses,
        time_window_hours: timeWindow,
        detect_patterns: detectPatterns,
        value_threshold_usd: valueThreshold,
      });
    }
  };

  // Initialize network when data is available
  useEffect(() => {
    if (analysisResult && graphRef.current) {
      drawNetwork();
    }
  }, [analysisResult]);

  // Draw the network visualization
  const drawNetwork = useCallback(() => {
    if (!graphRef.current || !analysisResult) return;

    // Create nodes dataset
    const nodes = new DataSet(
      analysisResult.nodes.map(node => ({
        id: node.address,
        label: formatAddress(node.address),
        title: `<div style="max-width:300px;padding:10px;">
          <strong>Address:</strong> ${node.address}<br>
          <strong>Total In:</strong> ${formatUSD(node.total_in_value_usd)}<br>
          <strong>Total Out:</strong> ${formatUSD(node.total_out_value_usd)}<br>
          <strong>Net Flow:</strong> ${formatUSD(node.net_flow_usd)}<br>
          <strong>Transactions:</strong> ${node.transaction_count}<br>
          <strong>Chains:</strong> ${node.chains.join(', ')}<br>
          <strong>Risk Score:</strong> ${node.risk_score?.toFixed(0) || 'N/A'}<br>
          ${node.labels?.length ? `<strong>Labels:</strong> ${node.labels.join(', ')}` : ''}
        </div>`,
        color: {
          background: node.risk_score && node.risk_score >= 70
            ? '#EF5350' // Red for high risk
            : node.risk_score && node.risk_score >= 40
              ? '#FFB300' // Orange for medium risk
              : '#66BB6A', // Green for low risk
          border: node.labels?.includes('ACCUMULATOR') 
            ? '#7E57C2' // Purple for accumulator
            : node.labels?.includes('SOURCE')
              ? '#42A5F5' // Blue for source
              : undefined,
          highlight: {
            background: '#90CAF9',
            border: '#1976D2'
          }
        },
        font: { color: '#FFFFFF' },
        shape: 'dot',
        size: 10 + Math.log(node.transaction_count + 1) * 5, // Size based on transaction count
      }))
    );

    // Create edges dataset
    const edges = new DataSet(
      analysisResult.edges.map(edge => ({
        id: edge.edge_id,
        from: edge.source,
        to: edge.target,
        value: edge.value_usd, // Used for thickness
        title: `<div style="max-width:300px;padding:10px;">
          <strong>From:</strong> ${formatAddress(edge.source)}<br>
          <strong>To:</strong> ${formatAddress(edge.target)}<br>
          <strong>Value:</strong> ${formatUSD(edge.value_usd)}<br>
          <strong>Chain:</strong> ${edge.chain}<br>
          <strong>Type:</strong> ${edge.transaction_type}<br>
          <strong>Hash:</strong> ${edge.transaction_hash}<br>
          <strong>Time:</strong> ${new Date(edge.timestamp).toLocaleString()}
        </div>`,
        arrows: 'to',
        color: { inherit: 'from' },
        width: 1 + Math.log(edge.value_usd / 1000 + 1) * 2, // Thickness based on value
      }))
    );

    // Network data
    const data = { nodes, edges };

    // Network options
    const options = {
      nodes: {
        borderWidth: 2,
        shadow: true,
      },
      edges: {
        smooth: {
          type: 'continuous',
        },
        shadow: true,
      },
      physics: {
        enabled: true,
        barnesHut: {
          gravitationalConstant: -2000,
          centralGravity: 0.3,
          springLength: 95,
          springConstant: 0.04,
          damping: 0.09,
          avoidOverlap: 0.5,
        },
        maxVelocity: 50,
        minVelocity: 0.1,
        solver: 'barnesHut',
        stabilization: {
          enabled: true,
          iterations: 1000,
          updateInterval: 25,
          fit: true,
        },
      },
      interaction: {
        navigationButtons: true,
        zoomView: true,
        dragView: true,
        tooltipDelay: 200,
        hover: true,
      },
    };

    // Clean up previous network
    if (networkRef.current) {
      networkRef.current.destroy();
    }

    // Create new network
    networkRef.current = new Network(graphRef.current, data, options);

    // Highlight patterns if available
    if (detectPatterns && analysisResult.patterns && analysisResult.patterns.length > 0) {
      // Delay pattern highlighting to allow network to stabilize
      setTimeout(() => {
        analysisResult.patterns.forEach(pattern => {
          if (pattern.path) {
            // Highlight nodes in the pattern
            const pathNodes = pattern.path.map(addr => {
              const node = nodes.get(addr);
              return {
                id: addr,
                borderWidth: 4,
                color: {
                  border: pattern.pattern_type === 'CIRCULAR_FLOW' 
                    ? '#FFD700' // Gold for circular
                    : pattern.pattern_type === 'PEEL_CHAIN'
                      ? '#00BFFF' // DeepSkyBlue for peel chains
                      : '#FF00FF', // Magenta for other patterns
                }
              };
            });
            nodes.update(pathNodes);

            // Highlight edges in the pattern
            for (let i = 0; i < pattern.path.length - 1; i++) {
              const from = pattern.path[i];
              const to = pattern.path[i+1];
              
              // Find edges between consecutive nodes in the path
              const edgeIds = edges.getIds({
                filter: item => {
                  const edge = edges.get(item);
                  return edge.from === from && edge.to === to;
                }
              });

              if (edgeIds.length > 0) {
                edges.update(edgeIds.map(id => ({
                  id,
                  color: pattern.pattern_type === 'CIRCULAR_FLOW' 
                    ? '#FFD700' // Gold for circular
                    : pattern.pattern_type === 'PEEL_CHAIN'
                      ? '#00BFFF' // DeepSkyBlue for peel chains
                      : '#FF00FF', // Magenta for other patterns
                  width: edges.get(id).width + 2,
                })));
              }
            }
          }
        });
      }, 1000);
    }
  }, [analysisResult, detectPatterns]);

  // Network control handlers
  const handleZoomIn = () => {
    if (networkRef.current) {
      networkRef.current.zoom(1.2);
    }
  };

  const handleZoomOut = () => {
    if (networkRef.current) {
      networkRef.current.zoom(0.8);
    }
  };

  const handleFitNetwork = () => {
    if (networkRef.current) {
      networkRef.current.fit();
    }
  };

  const handleRefreshLayout = () => {
    if (networkRef.current) {
      networkRef.current.setOptions({ physics: { enabled: true } });
      setTimeout(() => {
        if (networkRef.current) {
          networkRef.current.setOptions({ physics: { enabled: false } });
        }
      }, 3000);
    }
  };

  // Export handlers
  const handleExportImage = () => {
    if (networkRef.current) {
      const canvas = networkRef.current.canvas.getContext().canvas;
      const image = canvas.toDataURL('image/png');
      const a = document.createElement('a');
      a.href = image;
      a.download = `transaction_flow_${new Date().toISOString()}.png`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
    }
  };

  const handleExportData = () => {
    if (analysisResult) {
      const dataStr = JSON.stringify(analysisResult, null, 2);
      const blob = new Blob([dataStr], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `transaction_flow_data_${new Date().toISOString()}.json`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
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
            Transaction Flow Networks
          </Typography>
          <Typography variant="body2" color="text.secondary" paragraph>
            Visualize how funds move between wallets and contracts across time and chains. Detect suspicious patterns like peel chains, circular flows, and layering.
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
                      <Tooltip title="Analyze Flow">
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
                {isLoading ? 'Analyzing...' : 'Analyze Flow'}
              </Button>
            </Grid>
            <Grid item xs={12} sm={6}>
              <Typography gutterBottom>Time Window: {timeWindow} hours</Typography>
              <Slider
                value={timeWindow}
                onChange={(_e, newValue) => setTimeWindow(newValue as number)}
                aria-labelledby="time-window-slider"
                valueLabelDisplay="auto"
                min={1}
                max={72}
                step={1}
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <Typography gutterBottom>Min Transaction Value: {formatUSD(valueThreshold)}</Typography>
              <Slider
                value={valueThreshold}
                onChange={(_e, newValue) => setValueThreshold(newValue as number)}
                aria-labelledby="value-threshold-slider"
                valueLabelDisplay="auto"
                min={100}
                max={100000}
                step={100}
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <FormControlLabel
                control={
                  <Switch
                    checked={detectPatterns}
                    onChange={(e) => setDetectPatterns(e.target.checked)}
                    name="detectPatterns"
                    color="primary"
                  />
                }
                label="Detect Suspicious Patterns"
              />
            </Grid>
            <Grid item xs={12} sm={6} sx={{ display: 'flex', justifyContent: 'flex-end' }}>
              <Button
                variant="outlined"
                color="primary"
                onClick={() => resetAnalysis()}
                sx={{ mr: 1 }}
              >
                Clear
              </Button>
              <Button
                variant="outlined"
                color="primary"
                onClick={handleExportData}
                disabled={!analysisResult}
                startIcon={<SaveAltIcon />}
              >
                Export Data
              </Button>
            </Grid>
          </Grid>

          {/* Error Display */}
          {error && (
            <Alert severity="error" sx={{ mt: 3 }}>
              <AlertTitle>Analysis Error</AlertTitle>
              {error instanceof Error ? error.message : 'An unexpected error occurred during analysis.'}
            </Alert>
          )}

          {/* Graph Visualization */}
          {analysisResult ? (
            <>
              <GraphContainer ref={graphRef}>
                {/* Network visualization will be rendered here */}
                {analysisResult.nodes.length === 0 && (
                  <Box sx={{ position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%, -50%)', textAlign: 'center' }}>
                    <InfoIcon sx={{ fontSize: 48, color: 'text.secondary' }} />
                    <Typography variant="h6" color="text.secondary" mt={2}>
                      No transactions found for the given criteria.
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Try adjusting the time window or value threshold.
                    </Typography>
                  </Box>
                )}
                <ControlsContainer>
                  <Tooltip title="Zoom In">
                    <IconButton onClick={handleZoomIn} size="small" sx={{ bgcolor: 'background.paper' }}>
                      <ZoomInIcon />
                    </IconButton>
                  </Tooltip>
                  <Tooltip title="Zoom Out">
                    <IconButton onClick={handleZoomOut} size="small" sx={{ bgcolor: 'background.paper' }}>
                      <ZoomOutIcon />
                    </IconButton>
                  </Tooltip>
                  <Tooltip title="Fit View">
                    <IconButton onClick={handleFitNetwork} size="small" sx={{ bgcolor: 'background.paper' }}>
                      <FitScreenIcon />
                    </IconButton>
                  </Tooltip>
                  <Tooltip title="Refresh Layout">
                    <IconButton onClick={handleRefreshLayout} size="small" sx={{ bgcolor: 'background.paper' }}>
                      <RefreshIcon />
                    </IconButton>
                  </Tooltip>
                  <Tooltip title="Export as Image">
                    <IconButton onClick={handleExportImage} size="small" sx={{ bgcolor: 'background.paper' }}>
                      <DownloadIcon />
                    </IconButton>
                  </Tooltip>
                </ControlsContainer>
              </GraphContainer>

              {/* Results Display */}
              <Grid container spacing={3}>
                {/* Flow Metrics */}
                <Grid item xs={12} md={6}>
                  <MetricsContainer elevation={1}>
                    <Typography variant="h6" gutterBottom>Flow Metrics</Typography>
                    <Divider sx={{ mb: 2 }} />
                    <MetricItem>
                      <Typography variant="body2" color="text.secondary">Total Transactions:</Typography>
                      <Typography variant="body2">{analysisResult.metrics.total_transactions}</Typography>
                    </MetricItem>
                    <MetricItem>
                      <Typography variant="body2" color="text.secondary">Total Value:</Typography>
                      <Typography variant="body2">{formatUSD(analysisResult.metrics.total_value_usd)}</Typography>
                    </MetricItem>
                    <MetricItem>
                      <Typography variant="body2" color="text.secondary">Unique Addresses:</Typography>
                      <Typography variant="body2">{analysisResult.metrics.unique_addresses}</Typography>
                    </MetricItem>
                    <MetricItem>
                      <Typography variant="body2" color="text.secondary">Unique Chains:</Typography>
                      <Typography variant="body2">{analysisResult.metrics.unique_chains}</Typography>
                    </MetricItem>
                    <MetricItem>
                      <Typography variant="body2" color="text.secondary">Avg. Transaction Value:</Typography>
                      <Typography variant="body2">{formatUSD(analysisResult.metrics.average_transaction_value_usd)}</Typography>
                    </MetricItem>
                    <MetricItem>
                      <Typography variant="body2" color="text.secondary">Max Transaction Value:</Typography>
                      <Typography variant="body2">{formatUSD(analysisResult.metrics.max_transaction_value_usd)}</Typography>
                    </MetricItem>
                    <MetricItem>
                      <Typography variant="body2" color="text.secondary">Time Span:</Typography>
                      <Typography variant="body2">{analysisResult.metrics.time_span_hours.toFixed(1)} hours</Typography>
                    </MetricItem>
                    <MetricItem>
                      <Typography variant="body2" color="text.secondary">Transaction Density:</Typography>
                      <Typography variant="body2">{analysisResult.metrics.transaction_density.toFixed(2)} tx/hour</Typography>
                    </MetricItem>
                    <MetricItem>
                      <Typography variant="body2" color="text.secondary">Value Density:</Typography>
                      <Typography variant="body2">{formatUSD(analysisResult.metrics.value_density_usd)} /hour</Typography>
                    </MetricItem>
                    <MetricItem>
                      <Typography variant="body2" color="text.secondary">Graph Density:</Typography>
                      <Typography variant="body2">{analysisResult.metrics.graph_density.toFixed(3)}</Typography>
                    </MetricItem>
                  </MetricsContainer>
                </Grid>

                {/* Risk Assessment */}
                <Grid item xs={12} md={6}>
                  <MetricsContainer elevation={1}>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                      <Typography variant="h6">Risk Assessment</Typography>
                      {renderRiskLevelChip(analysisResult.risk_score)}
                    </Box>
                    <Divider sx={{ mb: 2 }} />
                    
                    <RiskScoreContainer>
                      <Typography variant="body2" color="text.secondary" sx={{ mr: 2, minWidth: '80px' }}>
                        Risk Score:
                      </Typography>
                      <Box sx={{ flexGrow: 1 }}>
                        <RiskScoreBar risk={analysisResult.risk_score} />
                      </Box>
                      <Typography variant="body2" sx={{ ml: 2, fontWeight: 'bold' }}>
                        {analysisResult.risk_score.toFixed(0)}/100
                      </Typography>
                    </RiskScoreContainer>

                    <Typography variant="subtitle2" gutterBottom sx={{ mt: 2 }}>Risk Factors:</Typography>
                    {analysisResult.risk_factors.length > 0 ? (
                      <ul style={{ margin: '8px 0', paddingLeft: '20px' }}>
                        {analysisResult.risk_factors.map((factor, index) => (
                          <li key={index}>
                            <Typography variant="body2">{factor}</Typography>
                          </li>
                        ))}
                      </ul>
                    ) : (
                      <Typography variant="body2" color="text.secondary">No significant risk factors detected.</Typography>
                    )}

                    {/* Detected Patterns */}
                    {analysisResult.patterns && analysisResult.patterns.length > 0 && (
                      <>
                        <Typography variant="subtitle2" gutterBottom sx={{ mt: 2 }}>Detected Patterns:</Typography>
                        {analysisResult.patterns.map((pattern, index) => (
                          <PatternItem key={index}>
                            <PatternChip 
                              label={pattern.pattern_type.replace('_', ' ')} 
                              color={
                                pattern.pattern_type === 'CIRCULAR_FLOW' ? 'warning' :
                                pattern.pattern_type === 'PEEL_CHAIN' ? 'info' : 'default'
                              }
                              size="small"
                            />
                            <Typography variant="body2">
                              {pattern.description} ({(pattern.confidence * 100).toFixed(0)}% confidence)
                            </Typography>
                          </PatternItem>
                        ))}
                      </>
                    )}
                  </MetricsContainer>
                </Grid>
              </Grid>
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
                  Enter wallet addresses and click "Analyze Flow" to visualize transaction networks
                </Typography>
              </Box>
            )
          )}
        </StyledPanelContent>
      </StyledPanel>
    </ErrorBoundary>
  );
};

export default TransactionFlowPanel;
