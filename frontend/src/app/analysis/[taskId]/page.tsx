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
} from '@mui/material';
import {
  ArrowBack as ArrowBackIcon,
  Download as DownloadIcon,
  ContentCopy as CopyIcon,
  Refresh as RefreshIcon,
  Info as InfoIcon,
} from '@mui/icons-material';

interface Visualization {
  filename: string;
  content: string; // base64 encoded image
}

interface TaskResult {
  task_id: string;
  crew_name: string;
  state: string;
  start_time: string;
  completion_time?: string;
  result: any;
  report?: string;
  visualizations?: Visualization[];
  metadata?: {
    execution_time?: number;
    agent_count?: number;
    paused_duration?: number;
    [key: string]: any;
  };
}

export default function AnalysisResultPage() {
  const { taskId } = useParams();
  const router = useRouter();
  const [taskResult, setTaskResult] = useState<TaskResult | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);

  // Fetch task result
  useEffect(() => {
    const fetchTaskResult = async () => {
      setLoading(true);
      setError(null);
      try {
        const headers = await getAuthHeaders();
        const response = await fetch(`${API_BASE_URL}/crew/${taskId}/result`, { headers });
        
        if (!response.ok) {
          const errorData = await response.json();
          throw new Error(errorData.detail || 'Failed to fetch task result');
        }
        
        const data = await response.json();
        setTaskResult(data);
      } catch (err: any) {
        console.error('Error fetching task result:', err);
        setError(err.message || 'An unexpected error occurred while fetching the task result');
      } finally {
        setLoading(false);
      }
    };

    if (taskId) {
      fetchTaskResult();
    }
  }, [taskId]);

  // Handle refreshing the result
  const handleRefresh = async () => {
    if (!taskId) return;
    
    setLoading(true);
    setError(null);
    try {
      const headers = await getAuthHeaders();
      const response = await fetch(`${API_BASE_URL}/crew/${taskId}/result`, { headers });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to refresh task result');
      }
      
      const data = await response.json();
      setTaskResult(data);
    } catch (err: any) {
      console.error('Error refreshing task result:', err);
      setError(err.message || 'An unexpected error occurred while refreshing the task result');
    } finally {
      setLoading(false);
    }
  };

  // Handle copying report to clipboard
  const handleCopyReport = () => {
    if (taskResult?.report) {
      navigator.clipboard.writeText(taskResult.report);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  // Handle exporting report as markdown file
  const handleExportReport = () => {
    if (!taskResult?.report) return;
    
    const blob = new Blob([taskResult.report], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `report-${taskResult.task_id}.md`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  // Handle exporting visualizations
  const handleExportVisualization = (visualization: Visualization) => {
    const byteString = atob(visualization.content);
    const ab = new ArrayBuffer(byteString.length);
    const ia = new Uint8Array(ab);
    
    for (let i = 0; i < byteString.length; i++) {
      ia[i] = byteString.charCodeAt(i);
    }
    
    const blob = new Blob([ab], { type: 'image/png' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = visualization.filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  // Format date for display
  const formatDate = (dateString?: string) => {
    if (!dateString) return 'N/A';
    return new Date(dateString).toLocaleString();
  };

  // Format duration for display
  const formatDuration = (seconds?: number) => {
    if (!seconds) return 'N/A';
    
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = Math.floor(seconds % 60);
    
    if (minutes === 0) {
      return `${remainingSeconds}s`;
    }
    
    return `${minutes}m ${remainingSeconds}s`;
  };

  return (
    <ProtectedRoute roles={[USER_ROLES.ADMIN, USER_ROLES.ANALYST, USER_ROLES.COMPLIANCE, USER_ROLES.AUDITOR]}>
      <Box className="min-h-screen bg-gray-100 p-4 sm:p-6 lg:p-8">
        <Box className="max-w-7xl mx-auto">
          {/* Header with navigation */}
          <Box className="flex justify-between items-center mb-6">
            <Button
              variant="outlined"
              startIcon={<ArrowBackIcon />}
              onClick={() => router.push('/dashboard')}
            >
              Back to Dashboard
            </Button>
            
            <Box>
              <Button
                variant="outlined"
                startIcon={<RefreshIcon />}
                onClick={handleRefresh}
                disabled={loading}
                sx={{ mr: 2 }}
              >
                Refresh
              </Button>
              
              {taskResult?.report && (
                <Button
                  variant="contained"
                  startIcon={<DownloadIcon />}
                  onClick={handleExportReport}
                  disabled={loading}
                >
                  Export Report
                </Button>
              )}
            </Box>
          </Box>

          {/* Loading state */}
          {loading && (
            <Box className="flex justify-center items-center py-12">
              <CircularProgress />
              <Typography variant="h6" className="ml-4">
                Loading investigation results...
              </Typography>
            </Box>
          )}

          {/* Error state */}
          {error && (
            <Alert severity="error" className="mb-6">
              {error}
            </Alert>
          )}

          {/* Result content */}
          {!loading && !error && taskResult && (
            <Grid container spacing={4}>
              {/* Metadata card */}
              <Grid item xs={12}>
                <Paper className="p-6 mb-6 shadow-md rounded-lg">
                  <Typography variant="h4" component="h1" gutterBottom>
                    Investigation Results
                  </Typography>
                  
                  <Grid container spacing={2} className="mb-4">
                    <Grid item xs={12} md={6}>
                      <Typography variant="subtitle1" className="font-bold">
                        Task ID:
                      </Typography>
                      <Typography variant="body1" className="mb-2">
                        {taskResult.task_id}
                      </Typography>
                      
                      <Typography variant="subtitle1" className="font-bold">
                        Crew:
                      </Typography>
                      <Typography variant="body1" className="mb-2">
                        {taskResult.crew_name}
                      </Typography>
                      
                      <Typography variant="subtitle1" className="font-bold">
                        Status:
                      </Typography>
                      <Chip
                        label={taskResult.state}
                        color={taskResult.state === 'COMPLETED' ? 'success' : 'default'}
                        className="mb-2"
                      />
                    </Grid>
                    
                    <Grid item xs={12} md={6}>
                      <Typography variant="subtitle1" className="font-bold">
                        Started:
                      </Typography>
                      <Typography variant="body1" className="mb-2">
                        {formatDate(taskResult.start_time)}
                      </Typography>
                      
                      <Typography variant="subtitle1" className="font-bold">
                        Completed:
                      </Typography>
                      <Typography variant="body1" className="mb-2">
                        {formatDate(taskResult.completion_time)}
                      </Typography>
                      
                      {taskResult.metadata && (
                        <>
                          <Typography variant="subtitle1" className="font-bold">
                            Execution Time:
                          </Typography>
                          <Typography variant="body1" className="mb-2">
                            {formatDuration(taskResult.metadata.execution_time)}
                          </Typography>
                          
                          {taskResult.metadata.paused_duration && (
                            <>
                              <Typography variant="subtitle1" className="font-bold">
                                HITL Review Duration:
                              </Typography>
                              <Typography variant="body1" className="mb-2">
                                {formatDuration(taskResult.metadata.paused_duration)}
                              </Typography>
                            </>
                          )}
                        </>
                      )}
                    </Grid>
                  </Grid>
                </Paper>
              </Grid>

              {/* Report card */}
              {taskResult.report && (
                <Grid item xs={12} lg={8}>
                  <Paper className="p-6 shadow-md rounded-lg">
                    <Box className="flex justify-between items-center mb-4">
                      <Typography variant="h5" component="h2">
                        Investigation Report
                      </Typography>
                      
                      <Box>
                        <Tooltip title={copied ? "Copied!" : "Copy to clipboard"}>
                          <IconButton onClick={handleCopyReport}>
                            <CopyIcon />
                          </IconButton>
                        </Tooltip>
                        
                        <Tooltip title="Export as markdown">
                          <IconButton onClick={handleExportReport}>
                            <DownloadIcon />
                          </IconButton>
                        </Tooltip>
                      </Box>
                    </Box>
                    
                    <Divider className="mb-4" />
                    
                    <Box className="prose max-w-none">
                      <ReactMarkdown>
                        {taskResult.report}
                      </ReactMarkdown>
                    </Box>
                  </Paper>
                </Grid>
              )}

              {/* Visualizations and metadata */}
              <Grid item xs={12} lg={taskResult.report ? 4 : 12}>
                {/* Visualizations */}
                {taskResult.visualizations && taskResult.visualizations.length > 0 && (
                  <Paper className="p-6 shadow-md rounded-lg mb-6">
                    <Typography variant="h5" component="h2" className="mb-4">
                      Visualizations
                    </Typography>
                    
                    <Divider className="mb-4" />
                    
                    {taskResult.visualizations.map((viz, index) => (
                      <Box key={index} className="mb-4">
                        <Typography variant="subtitle1" className="mb-2">
                          {viz.filename}
                          <IconButton 
                            size="small" 
                            onClick={() => handleExportVisualization(viz)}
                            sx={{ ml: 1 }}
                          >
                            <DownloadIcon fontSize="small" />
                          </IconButton>
                        </Typography>
                        
                        <Box className="border rounded-lg overflow-hidden">
                          <img 
                            src={`data:image/png;base64,${viz.content}`} 
                            alt={viz.filename}
                            className="w-full h-auto"
                          />
                        </Box>
                      </Box>
                    ))}
                  </Paper>
                )}

                {/* Raw result data */}
                <Paper className="p-6 shadow-md rounded-lg">
                  <Box className="flex justify-between items-center mb-4">
                    <Typography variant="h5" component="h2">
                      Raw Result Data
                    </Typography>
                    
                    <Tooltip title="Technical data from the investigation">
                      <IconButton>
                        <InfoIcon />
                      </IconButton>
                    </Tooltip>
                  </Box>
                  
                  <Divider className="mb-4" />
                  
                  <Box className="bg-gray-100 p-4 rounded-lg overflow-auto max-h-96">
                    <pre className="whitespace-pre-wrap break-words">
                      {JSON.stringify(taskResult.result, null, 2)}
                    </pre>
                  </Box>
                </Paper>
              </Grid>
            </Grid>
          )}

          {/* No result found */}
          {!loading && !error && !taskResult && (
            <Alert severity="info" className="mb-6">
              No results found for this task ID. The task may still be running or doesn't exist.
            </Alert>
          )}
        </Box>
      </Box>
    </ProtectedRoute>
  );
}
