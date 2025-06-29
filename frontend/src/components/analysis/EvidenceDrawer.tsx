import React, { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import {
  Drawer,
  Box,
  Typography,
  IconButton,
  Tabs,
  Tab,
  CircularProgress,
  Button,
  Chip,
  Alert,
  Divider,
  Paper,
  Tooltip,
} from '@mui/material';
import { styled } from '@mui/material/styles';
import CloseIcon from '@mui/icons-material/Close';
import DownloadIcon from '@mui/icons-material/Download';
import CodeIcon from '@mui/icons-material/Code';
import ArticleIcon from '@mui/icons-material/Article';
import HubIcon from '@mui/icons-material/Hub';
import PsychologyIcon from '@mui/icons-material/Psychology';

import { analysisAPI, EvidenceBundle, EvidenceItem } from '../../lib/api';
import { handleAPIError } from '../../lib/api';

interface EvidenceDrawerProps {
  evidenceBundleId: string | null;
  open: boolean;
  onClose: () => void;
}

const DrawerHeader = styled('div')(({ theme }) => ({
  display: 'flex',
  alignItems: 'center',
  padding: theme.spacing(1, 2),
  ...theme.mixins.toolbar,
  justifyContent: 'space-between',
  borderBottom: `1px solid ${theme.palette.divider}`,
}));

const DrawerContent = styled(Box)(({ theme }) => ({
  padding: theme.spacing(2),
  height: '100%',
  overflowY: 'auto',
}));

const EvidenceItemCard = styled(Paper)(({ theme }) => ({
  padding: theme.spacing(2),
  marginBottom: theme.spacing(2),
}));

const CodeBlock = styled('pre')(({ theme }) => ({
  backgroundColor: theme.palette.grey[100],
  border: `1px solid ${theme.palette.divider}`,
  borderRadius: theme.shape.borderRadius,
  padding: theme.spacing(2),
  whiteSpace: 'pre-wrap',
  wordBreak: 'break-all',
  fontFamily: 'monospace',
  maxHeight: '60vh',
  overflowY: 'auto',
}));

const EvidenceDrawer: React.FC<EvidenceDrawerProps> = ({ evidenceBundleId, open, onClose }) => {
  const [activeTab, setActiveTab] = useState(0);

  const {
    data: evidenceBundle,
    isLoading,
    isError,
    error,
  } = useQuery({
    queryKey: ['evidenceBundle', evidenceBundleId],
    queryFn: () => {
      if (!evidenceBundleId) {
        throw new Error('No Evidence Bundle ID provided');
      }
      return analysisAPI.getEvidenceBundle(evidenceBundleId);
    },
    enabled: !!evidenceBundleId && open,
    staleTime: 10 * 60 * 1000, // 10 minutes
  });

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setActiveTab(newValue);
  };

  const handleDownload = () => {
    if (!evidenceBundle) return;
    const jsonString = JSON.stringify(evidenceBundle, null, 2);
    const blob = new Blob([jsonString], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `evidence-bundle-${evidenceBundle.bundle_id}.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  const renderConfidenceChip = (confidence: number) => {
    let color: 'success' | 'warning' | 'error' | 'default' = 'default';
    if (confidence >= 0.8) color = 'success';
    else if (confidence >= 0.5) color = 'warning';
    else if (confidence > 0) color = 'error';

    return <Chip label={`Confidence: ${(confidence * 100).toFixed(0)}%`} color={color} size="small" />;
  };

  const renderContent = () => {
    if (isLoading) {
      return (
        <Box display="flex" justifyContent="center" alignItems="center" height="100%">
          <CircularProgress />
        </Box>
      );
    }

    if (isError) {
      const apiError = handleAPIError(error);
      return <Alert severity="error">Error fetching evidence: {apiError.message}</Alert>;
    }

    if (!evidenceBundle) {
      return <Alert severity="info">No evidence bundle data available.</Alert>;
    }

    // Extract Cypher queries from evidence items
    const cypherQueries = evidenceBundle.items
      .map(item => item.raw_data?.query_text)
      .filter(Boolean);

    return (
      <>
        <Tabs value={activeTab} onChange={handleTabChange} variant="fullWidth">
          <Tab icon={<ArticleIcon />} label="Summary" />
          <Tab icon={<CodeIcon />} label="Raw JSON" />
          {cypherQueries.length > 0 && <Tab icon={<HubIcon />} label="Cypher Query" />}
        </Tabs>
        <DrawerContent>
          {activeTab === 0 && (
            <Box>
              <Typography variant="h6" gutterBottom>
                Executive Summary
              </Typography>
              <Typography variant="body1" paragraph>
                {evidenceBundle.summary || 'No summary provided.'}
              </Typography>
              <Divider sx={{ my: 2 }} />
              <Typography variant="h6" gutterBottom>
                Evidence Items ({evidenceBundle.items.length})
              </Typography>
              {evidenceBundle.items.map((item, index) => (
                <EvidenceItemCard key={item.id || index} elevation={2}>
                  <Box display="flex" justifyContent="space-between" alignItems="center" mb={1}>
                    <Typography variant="subtitle1" fontWeight="bold">
                      {item.description}
                    </Typography>
                    {renderConfidenceChip(item.confidence)}
                  </Box>
                  <Typography variant="caption" color="text.secondary">
                    Source: {item.source} | ID: {item.id}
                  </Typography>
                </EvidenceItemCard>
              ))}
            </Box>
          )}
          {activeTab === 1 && (
            <Box>
              <Typography variant="h6" gutterBottom>
                Raw Evidence Bundle JSON
              </Typography>
              <CodeBlock>{JSON.stringify(evidenceBundle, null, 2)}</CodeBlock>
            </Box>
          )}
          {activeTab === 2 && cypherQueries.length > 0 && (
            <Box>
              <Typography variant="h6" gutterBottom>
                Associated Cypher Query
              </Typography>
              <CodeBlock>{cypherQueries[0]}</CodeBlock>
              <Box mt={2} display="flex" justifyContent="flex-end">
                <Tooltip title="Explain this Cypher query using AI (coming soon)">
                  <span>
                    <Button
                      variant="outlined"
                      startIcon={<PsychologyIcon />}
                      disabled // To be enabled when functionality is ready
                    >
                      Explain Query
                    </Button>
                  </span>
                </Tooltip>
              </Box>
            </Box>
          )}
        </DrawerContent>
      </>
    );
  };

  return (
    <Drawer anchor="right" open={open} onClose={onClose} PaperProps={{ sx: { width: '50%', minWidth: 400 } }}>
      <DrawerHeader>
        <Box>
          <Typography variant="h6" component="div">
            Evidence Bundle
          </Typography>
          <Typography variant="caption" color="text.secondary">
            ID: {evidenceBundleId}
          </Typography>
        </Box>
        <Box>
          <Tooltip title="Download as JSON">
            <span>
              <IconButton onClick={handleDownload} disabled={!evidenceBundle}>
                <DownloadIcon />
              </IconButton>
            </span>
          </Tooltip>
          <Tooltip title="Close">
            <IconButton onClick={onClose}>
              <CloseIcon />
            </IconButton>
          </Tooltip>
        </Box>
      </DrawerHeader>
      {renderContent()}
    </Drawer>
  );
};

export default EvidenceDrawer;
