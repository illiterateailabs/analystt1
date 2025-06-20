'use client'

import { useState } from 'react'
import { useQuery, useMutation } from '@tanstack/react-query' // Changed from 'react-query' to '@tanstack/react-query'
import { analysisAPI, handleAPIError } from '@/lib/api'
import toast from 'react-hot-toast'
import {
  Box,
  Tabs,
  Tab,
  Typography,
  TextField,
  Button,
  CircularProgress,
  Alert,
  AlertTitle,
  Paper,
  Divider,
} from '@mui/material'; // Material-UI imports
import { styled } from '@mui/material/styles'; // For styled components

// Material-UI Icons
import {
  Wallet as WalletIcon,
  SwapHoriz as SwapHorizIcon,
  Public as PublicIcon,
  ChartBar as ChartBarIcon,
  BugReport as BugReportIcon, // Using BugReport for Fraud Detection
  Code as CodeIcon,
  Info as InfoIcon,
} from '@mui/icons-material';

// Import new analysis components
import WalletAnalysisPanel from './WalletAnalysisPanel';
import TransactionFlowPanel from './TransactionFlowPanel';
import CrossChainIdentityPanel from './CrossChainIdentityPanel';

// Styled components for consistency
const StyledPanel = styled(Box)(({ theme }) => ({
  display: 'flex',
  height: '100%',
  backgroundColor: theme.palette.background.paper,
}));

const LeftPanel = styled(Box)(({ theme }) => ({
  width: '350px', // Fixed width for the left panel
  borderRight: `1px solid ${theme.palette.divider}`,
  display: 'flex',
  flexDirection: 'column',
  [theme.breakpoints.down('md')]: {
    width: '100%', // Full width on smaller screens
    borderRight: 'none',
    borderBottom: `1px solid ${theme.palette.divider}`,
  },
}));

const RightPanel = styled(Box)(({ theme }) => ({
  flexGrow: 1,
  display: 'flex',
  flexDirection: 'column',
  overflow: 'hidden', // Ensure content within is scrollable if needed
}));

const TabContent = styled(Box)(({ theme }) => ({
  flexGrow: 1,
  overflowY: 'auto', // Enable scrolling for tab content
  padding: theme.spacing(3),
}));

type ActiveTab = 'wallet' | 'transaction-flow' | 'cross-chain' | 'analysis' | 'fraud' | 'code';

export function AnalysisPanel() {
  const [analysisTask, setAnalysisTask] = useState('');
  const [codeToExecute, setCodeToExecute] = useState('');
  const [activeTab, setActiveTab] = useState<ActiveTab>('wallet'); // Default to Wallet Analysis
  const [walletAddressInput, setWalletAddressInput] = useState(''); // State for wallet address input

  // Fraud detection query
  const { data: fraudData, isLoading: fraudLoading, refetch: refetchFraud } = useQuery(
    ['fraud-patterns'],
    () => analysisAPI.detectFraudPatterns('money_laundering', 50),
    {
      enabled: false, // Don't auto-fetch
      onError: (error) => {
        const errorInfo = handleAPIError(error);
        toast.error(errorInfo.message);
      },
    }
  );

  // Analysis mutation
  const analysisMutation = useMutation(
    (task: string) => analysisAPI.performAnalysis(task, 'graph'),
    {
      onSuccess: (response) => {
        toast.success('Analysis completed successfully');
        console.log('Analysis results:', response.data);
      },
      onError: (error) => {
        const errorInfo = handleAPIError(error);
        toast.error(errorInfo.message);
      },
    }
  );

  // Code execution mutation
  const codeExecutionMutation = useMutation(
    (code: string) => analysisAPI.executeCode(code, ['pandas', 'numpy', 'matplotlib']),
    {
      onSuccess: (response) => {
        if (response.data.success) {
          toast.success('Code executed successfully');
        } else {
          toast.error('Code execution failed');
        }
        console.log('Execution results:', response.data);
      },
      onError: (error) => {
        const errorInfo = handleAPIError(error);
        toast.error(errorInfo.message);
      },
    }
  );

  const handleAnalysis = () => {
    if (!analysisTask.trim()) return;
    analysisMutation.mutate(analysisTask);
  };

  const handleCodeExecution = () => {
    if (!codeToExecute.trim()) return;
    codeExecutionMutation.mutate(codeToExecute);
  };

  const handleFraudDetection = () => {
    refetchFraud();
  };

  const handleTabChange = (_event: React.SyntheticEvent, newValue: ActiveTab) => {
    setActiveTab(newValue);
  };

  const renderActiveTabContent = () => {
    switch (activeTab) {
      case 'wallet':
        return (
          <Box sx={{ p: 3 }}>
            <TextField
              label="Wallet Address"
              fullWidth
              value={walletAddressInput}
              onChange={(e) => setWalletAddressInput(e.target.value)}
              placeholder="e.g., 0xd8da6bf26964af9d7eed9e03e53415d37aa96045"
              sx={{ mb: 3 }}
            />
            {walletAddressInput ? (
              <WalletAnalysisPanel walletAddress={walletAddressInput} />
            ) : (
              <Alert severity="info">
                <AlertTitle>Enter a Wallet Address</AlertTitle>
                Please enter a wallet address to view its analysis.
              </Alert>
            )}
          </Box>
        );
      case 'transaction-flow':
        return (
          <Box sx={{ p: 3 }}>
            <TransactionFlowPanel />
          </Box>
        );
      case 'cross-chain':
        return (
          <Box sx={{ p: 3 }}>
            <CrossChainIdentityPanel />
          </Box>
        );
      case 'analysis':
        return (
          <Box sx={{ p: 3 }}>
            <div className="space-y-4">
              <div>
                <Typography variant="subtitle1" gutterBottom>
                  Analysis Task Description
                </Typography>
                <TextField
                  value={analysisTask}
                  onChange={(e) => setAnalysisTask(e.target.value)}
                  placeholder="e.g., Analyze transaction patterns to identify potential money laundering schemes"
                  fullWidth
                  multiline
                  rows={4}
                  variant="outlined"
                />
              </div>

              <Button
                variant="contained"
                color="primary"
                fullWidth
                onClick={handleAnalysis}
                disabled={!analysisTask.trim() || analysisMutation.isLoading}
                startIcon={analysisMutation.isLoading ? <CircularProgress size={20} color="inherit" /> : <ChartBarIcon />}
              >
                {analysisMutation.isLoading ? 'Analyzing...' : 'Start Analysis'}
              </Button>

              {/* Quick analysis templates */}
              <Box sx={{ mt: 3 }}>
                <Typography variant="subtitle2" gutterBottom>Quick Templates</Typography>
                <div className="space-y-2">
                  {[
                    'Identify high-risk transaction patterns',
                    'Analyze network centrality for key actors',
                    'Detect circular money flows',
                    'Find suspicious account clustering',
                  ].map((template) => (
                    <Button
                      key={template}
                      fullWidth
                      variant="outlined"
                      onClick={() => setAnalysisTask(template)}
                      sx={{ justifyContent: 'flex-start', textTransform: 'none' }}
                    >
                      {template}
                    </Button>
                  ))}
                </div>
              </Box>
            </div>
          </Box>
        );
      case 'fraud':
        return (
          <Box sx={{ p: 3 }}>
            <div className="space-y-4">
              <div>
                <Typography variant="h6" gutterBottom>
                  Fraud Detection Patterns
                </Typography>

                <Button
                  variant="contained"
                  color="primary"
                  fullWidth
                  onClick={handleFraudDetection}
                  disabled={fraudLoading}
                  startIcon={fraudLoading ? <CircularProgress size={20} color="inherit" /> : <BugReportIcon />}
                >
                  {fraudLoading ? 'Detecting...' : 'Detect Money Laundering Patterns'}
                </Button>

                {fraudData?.data && (
                  <Alert severity="warning" sx={{ mt: 3 }}>
                    <AlertTitle>Detection Results</AlertTitle>
                    <Typography variant="body2">
                      Found {fraudData.data.patterns_found} potential patterns
                    </Typography>
                    {fraudData.data.explanation && (
                      <Typography variant="body2" sx={{ mt: 1 }}>
                        {fraudData.data.explanation}
                      </Typography>
                    )}
                  </Alert>
                )}
              </div>

              {/* Fraud detection options */}
              <Box sx={{ mt: 3 }}>
                <Typography variant="subtitle2" gutterBottom>Detection Types</Typography>
                <div className="space-y-2">
                  {[
                    'Money Laundering Schemes',
                    'Circular Transactions',
                    'Suspicious Velocity',
                    'Account Takeover Patterns',
                  ].map((type) => (
                    <Button
                      key={type}
                      fullWidth
                      variant="outlined"
                      sx={{ justifyContent: 'flex-start', textTransform: 'none' }}
                    >
                      {type}
                    </Button>
                  ))}
                </div>
              </Box>
            </div>
          </Box>
        );
      case 'code':
        return (
          <Box sx={{ p: 3 }}>
            <div className="space-y-4">
              <div>
                <Typography variant="subtitle1" gutterBottom>
                  Python Code
                </Typography>
                <TextField
                  value={codeToExecute}
                  onChange={(e) => setCodeToExecute(e.target.value)}
                  placeholder="import pandas as pd\nimport numpy as np\n\n# Your analysis code here"
                  fullWidth
                  multiline
                  rows={8}
                  variant="outlined"
                  InputProps={{ style: { fontFamily: 'monospace', fontSize: '0.875rem' } }}
                />
              </div>

              <Button
                variant="contained"
                color="primary"
                fullWidth
                onClick={handleCodeExecution}
                disabled={!codeToExecute.trim() || codeExecutionMutation.isLoading}
                startIcon={codeExecutionMutation.isLoading ? <CircularProgress size={20} color="inherit" /> : <CodeIcon />}
              >
                {codeExecutionMutation.isLoading ? 'Executing...' : 'Execute Code'}
              </Button>

              {/* Code templates */}
              <Box sx={{ mt: 3 }}>
                <Typography variant="subtitle2" gutterBottom>Code Templates</Typography>
                <div className="space-y-2">
                  {[
                    'Basic data analysis',
                    'Graph metrics calculation',
                    'Visualization generation',
                    'Statistical analysis',
                  ].map((template) => (
                    <Button
                      key={template}
                      fullWidth
                      variant="outlined"
                      sx={{ justifyContent: 'flex-start', textTransform: 'none' }}
                    >
                      {template}
                    </Button>
                  ))}
                </div>
              </Box>
            </div>
          </Box>
        );
      default:
        return null;
    }
  };

  return (
    <StyledPanel>
      {/* Left panel - Controls and Tabs */}
      <LeftPanel>
        <Box sx={{ p: 3, borderBottom: `1px solid ${theme => theme.palette.divider}` }}>
          <Typography variant="h5" gutterBottom>
            Data Analysis
          </Typography>
          <Typography variant="body2" color="text.secondary">
            AI-powered analysis and fraud detection
          </Typography>
        </Box>

        {/* Tabs */}
        <Tabs
          orientation="vertical"
          variant="scrollable"
          value={activeTab}
          onChange={handleTabChange}
          aria-label="Analysis tabs"
          sx={{ borderRight: 1, borderColor: 'divider', flexGrow: 1 }}
        >
          <Tab label="Wallet Analysis" value="wallet" icon={<WalletIcon />} iconPosition="start" />
          <Tab label="Transaction Flow" value="transaction-flow" icon={<SwapHorizIcon />} iconPosition="start" />
          <Tab label="Cross-Chain Identity" value="cross-chain" icon={<PublicIcon />} iconPosition="start" />
          <Divider sx={{ my: 1 }} />
          <Tab label="General Analysis" value="analysis" icon={<ChartBarIcon />} iconPosition="start" />
          <Tab label="Fraud Detection" value="fraud" icon={<BugReportIcon />} iconPosition="start" />
          <Tab label="Code Execution" value="code" icon={<CodeIcon />} iconPosition="start" />
        </Tabs>
      </LeftPanel>

      {/* Right panel - Results */}
      <RightPanel>
        <Box sx={{ p: 3, borderBottom: `1px solid ${theme => theme.palette.divider}` }}>
          <Typography variant="h6" gutterBottom>
            Analysis Results
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Results and visualizations will appear here
          </Typography>
        </Box>

        <TabContent>
          {renderActiveTabContent()}
        </TabContent>
      </RightPanel>
    </StyledPanel>
  );
}
