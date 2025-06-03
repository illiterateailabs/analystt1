'use client';

import React, { useState, useEffect } from 'react';
import { useParams, useRouter } from 'next/navigation';
import ProtectedRoute from '@/components/auth/ProtectedRoute';
import { getAuthHeaders } from '@/lib/auth';
import { API_BASE_URL, USER_ROLES } from '@/lib/constants';
import ReactMarkdown from 'react-markdown';
import {
  Box,
  Typography,
  Paper,
  CircularProgress,
  Alert,
  Button,
  Card,
  CardContent,
  Divider,
  Grid,
  Chip,
  IconButton,
  Tooltip,
  Container,
  Stack,
} from '@mui/material';
import {
  ArrowBack as ArrowBackIcon,
  Download as DownloadIcon,
  ContentCopy as CopyIcon,
  Refresh as RefreshIcon,
  Info as InfoIcon,
  Warning as WarningIcon,
  CheckCircle as CheckCircleIcon,
  BarChart as BarChartIcon,
  FileDownload as FileDownloadIcon,
} from '@mui/icons-material';
import GraphVisualization from '@/components/graph/GraphVisualization';

// Define TaskResult interface
interface Visualization {
  filename: string;
  type: string;
  data: string;
}

interface TaskResult {
  task_id: string;
  crew_name: string;
  state: string;
  start_time: string;
  completion_time?: string;
  result?: string;
  report?: string;
  visualizations?: Visualization[];
  metadata?: {
    inputs?: any;
    paused_duration?: number;
    risk_score?: number;
    confidence?: number;
    [key: string]: any;
  };
}

export default function AnalysisResultsPage() {
  const { taskId } = useParams();
  const router = useRouter();
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [taskResult, setTaskResult] = useState<TaskResult | null>(null);
  const [activeTab, setActiveTab] = useState('summary');
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    fetchTaskResult();
  }, [taskId]);

  const fetchTaskResult = async () => {
    try {
      setLoading(true);
      setError(null);

      const response = await fetch(`${API_BASE_URL}/crew/${taskId}/result`, {
        headers: getAuthHeaders(),
      });

      if (!response.ok) {
        throw new Error(`Error fetching task result: ${response.statusText}`);
      }

      const data = await response.json();
      setTaskResult(data);
    } catch (err) {
      console.error('Failed to fetch task result:', err);
      setError(err instanceof Error ? err.message : 'Failed to fetch task result');
    } finally {
      setLoading(false);
    }
  };

  const handleExportJSON = () => {
    if (!taskResult) return;
    
    const dataStr = JSON.stringify(taskResult, null, 2);
    const dataUri = `data:application/json;charset=utf-8,${encodeURIComponent(dataStr)}`;
    
    const exportFileDefaultName = `analysis-${taskResult.task_id}.json`;
    
    const linkElement = document.createElement('a');
    linkElement.setAttribute('href', dataUri);
    linkElement.setAttribute('download', exportFileDefaultName);
    linkElement.click();
  };

  const handleExportReport = () => {
    if (!taskResult?.report) return;
    
    const dataStr = taskResult.report;
    const dataUri = `data:text/markdown;charset=utf-8,${encodeURIComponent(dataStr)}`;
    
    const exportFileDefaultName = `report-${taskResult.task_id}.md`;
    
    const linkElement = document.createElement('a');
    linkElement.setAttribute('href', dataUri);
    linkElement.setAttribute('download', exportFileDefaultName);
    linkElement.click();
  };

  const handleCopyTaskId = () => {
    if (taskId) {
      navigator.clipboard.writeText(taskId as string);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  const downloadVisualization = (visualization: Visualization) => {
    const linkElement = document.createElement('a');
    linkElement.setAttribute('href', `data:${visualization.type};base64,${visualization.data}`);
    linkElement.setAttribute('download', visualization.filename);
    linkElement.click();
  };

  const renderRiskIndicator = () => {
    const riskScore = taskResult?.metadata?.risk_score || 0;
    let color = 'success.main';
    let icon = <CheckCircleIcon />;
    let label = 'Low Risk';
    
    if (riskScore > 70) {
      color = 'error.main';
      icon = <WarningIcon />;
      label = 'High Risk';
    } else if (riskScore > 30) {
      color = 'warning.main';
      icon = <WarningIcon />;
      label = 'Medium Risk';
    }
    
    return (
      <Box display="flex" alignItems="center" sx={{ color }}>
        {icon}
        <Typography variant="body1" ml={1} fontWeight="bold">
          {label} ({riskScore}/100)
        </Typography>
      </Box>
    );
  };

  const renderConfidenceIndicator = () => {
    const confidence = taskResult?.metadata?.confidence || 0;
    let color = 'error.main';
    
    if (confidence > 0.7) {
      color = 'success.main';
    } else if (confidence > 0.4) {
      color = 'warning.main';
    }
    
    return (
      <Box display="flex" alignItems="center" sx={{ color }}>
        <InfoIcon />
        <Typography variant="body1" ml={1} fontWeight="bold">
          Confidence: {(confidence * 100).toFixed(0)}%
        </Typography>
      </Box>
    );
  };

  const renderExecutionTime = () => {
    if (!taskResult?.start_time || !taskResult?.completion_time) return null;
    
    try {
      const start = new Date(taskResult.start_time);
      const end = new Date(taskResult.completion_time);
      const durationMs = end.getTime() - start.getTime();
      const durationMinutes = Math.floor(durationMs / 60000);
      const durationSeconds = Math.floor((durationMs % 60000) / 1000);
      
      return (
        <Typography variant="body2" color="text.secondary">
          Execution time: {durationMinutes}m {durationSeconds}s
        </Typography>
      );
    } catch (e) {
      return null;
    }
  };

  const renderStatusChip = () => {
    if (!taskResult) return null;
    
    let color: 'success' | 'error' | 'warning' | 'default' = 'default';
    
    switch (taskResult.state) {
      case 'COMPLETED':
        color = 'success';
        break;
      case 'ERROR':
        color = 'error';
        break;
      case 'PAUSED':
        color = 'warning';
        break;
    }
    
    return (
      <Chip 
        label={taskResult.state} 
        color={color} 
        size="small" 
        sx={{ fontWeight: 'bold' }}
      />
    );
  };

  const renderVisualizations = () => {
    if (!taskResult?.visualizations || taskResult.visualizations.length === 0) {
      return (
        <Alert severity="info" sx={{ mt: 2 }}>
          No visualizations were generated during this analysis.
        </Alert>
      );
    }
    
    return (
      <Grid container spacing={3} sx={{ mt: 1 }}>
        {taskResult.visualizations.map((vis, index) => (
          <Grid item xs={12} md={6} key={index}>
            <Card variant="outlined">
              <CardContent>
                <Typography variant="subtitle1" gutterBottom>
                  {vis.filename}
                </Typography>
                <Box 
                  sx={{ 
                    display: 'flex', 
                    justifyContent: 'center', 
                    alignItems: 'center',
                    border: '1px solid #eee',
                    borderRadius: 1,
                    p: 1,
                    mb: 2,
                    minHeight: 200
                  }}
                >
                  <img 
                    src={`data:${vis.type};base64,${vis.data}`} 
                    alt={vis.filename}
                    style={{ maxWidth: '100%', maxHeight: 400 }}
                  />
                </Box>
                <Button
                  startIcon={<DownloadIcon />}
                  variant="outlined"
                  size="small"
                  onClick={() => downloadVisualization(vis)}
                >
                  Download
                </Button>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>
    );
  };

  const renderGraph = () => {
    // Check if there's graph data in the metadata
    const graphData = taskResult?.metadata?.graph_data;
    
    if (!graphData) {
      return (
        <Alert severity="info" sx={{ mt: 2 }}>
          No graph data available for this analysis.
        </Alert>
      );
    }
    
    return (
      <Box sx={{ height: 600, border: '1px solid #eee', borderRadius: 1, p: 2 }}>
        <GraphVisualization data={graphData} />
      </Box>
    );
  };

  return (
    <ProtectedRoute requiredRoles={[USER_ROLES.ANALYST, USER_ROLES.ADMIN]}>
      <Box sx={{ p: 3 }}>
        {/* Header */}
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
          <Button 
            startIcon={<ArrowBackIcon />} 
            onClick={() => router.push('/analysis')}
          >
            Back to Analysis
          </Button>
          
          <Box sx={{ display: 'flex', gap: 2 }}>
            <Button
              startIcon={<RefreshIcon />}
              variant="outlined"
              onClick={fetchTaskResult}
              disabled={loading}
            >
              Refresh
            </Button>
            <Button
              startIcon={<FileDownloadIcon />}
              variant="contained"
              onClick={handleExportJSON}
              disabled={!taskResult}
            >
              Export Results
            </Button>
          </Box>
        </Box>
        
        {/* Loading state */}
        {loading && (
          <Box sx={{ display: 'flex', justifyContent: 'center', my: 4 }}>
            <CircularProgress />
          </Box>
        )}
        
        {/* Error state */}
        {error && (
          <Alert severity="error" sx={{ mb: 3 }}>
            {error}
          </Alert>
        )}
        
        {/* Result display */}
        {!loading && !error && taskResult && (
          <>
            {/* Task info card */}
            <Paper sx={{ p: 3, mb: 3 }}>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 2 }}>
                <Box>
                  <Typography variant="h5" gutterBottom>
                    {taskResult.crew_name.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                  </Typography>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                    <Typography variant="body2" color="text.secondary">
                      Task ID: {taskResult.task_id}
                    </Typography>
                    <Tooltip title={copied ? "Copied!" : "Copy Task ID"}>
                      <IconButton size="small" onClick={handleCopyTaskId}>
                        <CopyIcon fontSize="small" />
                      </IconButton>
                    </Tooltip>
                    {renderStatusChip()}
                  </Box>
                  {renderExecutionTime()}
                </Box>
                
                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1, alignItems: 'flex-end' }}>
                  {taskResult.metadata?.risk_score && renderRiskIndicator()}
                  {taskResult.metadata?.confidence && renderConfidenceIndicator()}
                </Box>
              </Box>
              
              <Divider sx={{ my: 2 }} />
              
              {/* Navigation tabs */}
              <Box sx={{ display: 'flex', gap: 2, mb: 3 }}>
                <Button 
                  variant={activeTab === 'summary' ? 'contained' : 'outlined'}
                  onClick={() => setActiveTab('summary')}
                >
                  Summary
                </Button>
                {taskResult.report && (
                  <Button 
                    variant={activeTab === 'report' ? 'contained' : 'outlined'}
                    onClick={() => setActiveTab('report')}
                    endIcon={taskResult.report && <DownloadIcon onClick={handleExportReport} />}
                  >
                    Report
                  </Button>
                )}
                {taskResult.visualizations && taskResult.visualizations.length > 0 && (
                  <Button 
                    variant={activeTab === 'visualizations' ? 'contained' : 'outlined'}
                    onClick={() => setActiveTab('visualizations')}
                    startIcon={<BarChartIcon />}
                  >
                    Visualizations
                  </Button>
                )}
                {taskResult.metadata?.graph_data && (
                  <Button 
                    variant={activeTab === 'graph' ? 'contained' : 'outlined'}
                    onClick={() => setActiveTab('graph')}
                  >
                    Graph
                  </Button>
                )}
              </Box>
              
              {/* Tab content */}
              <Box sx={{ mt: 3 }}>
                {activeTab === 'summary' && (
                  <Box>
                    <Typography variant="h6" gutterBottom>
                      Executive Summary
                    </Typography>
                    <Paper variant="outlined" sx={{ p: 2, bgcolor: 'background.default' }}>
                      <Typography variant="body1" component="div" sx={{ whiteSpace: 'pre-wrap' }}>
                        {taskResult.result || "No summary available."}
                      </Typography>
                    </Paper>
                    
                    {taskResult.metadata?.inputs && (
                      <Box sx={{ mt: 3 }}>
                        <Typography variant="h6" gutterBottom>
                          Analysis Inputs
                        </Typography>
                        <Paper variant="outlined" sx={{ p: 2, bgcolor: 'background.default' }}>
                          <pre style={{ margin: 0, overflow: 'auto' }}>
                            {JSON.stringify(taskResult.metadata.inputs, null, 2)}
                          </pre>
                        </Paper>
                      </Box>
                    )}
                  </Box>
                )}
                
                {activeTab === 'report' && taskResult.report && (
                  <Box>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                      <Typography variant="h6">
                        Detailed Report
                      </Typography>
                      <Button 
                        startIcon={<DownloadIcon />}
                        variant="outlined"
                        size="small"
                        onClick={handleExportReport}
                      >
                        Download Report
                      </Button>
                    </Box>
                    <Paper 
                      variant="outlined" 
                      sx={{ 
                        p: 3, 
                        bgcolor: 'background.default',
                        '& img': { maxWidth: '100%' },
                        '& pre': { 
                          backgroundColor: '#f5f5f5', 
                          padding: 2, 
                          borderRadius: 1,
                          overflow: 'auto'
                        },
                        '& table': {
                          borderCollapse: 'collapse',
                          width: '100%',
                          marginBottom: 2
                        },
                        '& th, & td': {
                          border: '1px solid #ddd',
                          padding: 1
                        },
                        '& th': {
                          backgroundColor: '#f5f5f5'
                        }
                      }}
                    >
                      <ReactMarkdown>
                        {taskResult.report}
                      </ReactMarkdown>
                    </Paper>
                  </Box>
                )}
                
                {activeTab === 'visualizations' && renderVisualizations()}
                
                {activeTab === 'graph' && renderGraph()}
              </Box>
            </Paper>
          </>
        )}
        
        {/* No result found */}
        {!loading && !error && !taskResult && (
          <Alert severity="info" className="mb-6">
            No results found for this task ID. The task may still be running or doesn't exist.
          </Alert>
        )}
      </Box>
    </ProtectedRoute>
  );
}
