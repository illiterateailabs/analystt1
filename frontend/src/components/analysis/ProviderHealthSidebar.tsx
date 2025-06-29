import React, { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import {
  Box,
  Paper,
  Typography,
  IconButton,
  Collapse,
  LinearProgress,
  Tooltip,
  Chip,
  CircularProgress,
  Alert,
  Divider,
} from '@mui/material';
import { styled } from '@mui/material/styles';
import ChevronLeftIcon from '@mui/icons-material/ChevronLeft';
import ChevronRightIcon from '@mui/icons-material/ChevronRight';
import LanIcon from '@mui/icons-material/Lan';
import CheckCircleOutlineIcon from '@mui/icons-material/CheckCircleOutline';
import WarningAmberIcon from '@mui/icons-material/WarningAmber';
import ErrorOutlineIcon from '@mui/icons-material/ErrorOutline';

import { backpressureAPI, ProviderStatus } from '../../lib/api';

const SidebarContainer = styled(Paper)<{ collapsed: number }>(({ theme, collapsed }) => ({
  position: 'fixed',
  top: '50%',
  right: collapsed ? -340 : 0,
  transform: 'translateY(-50%)',
  width: 380,
  height: 'auto',
  maxHeight: '80vh',
  display: 'flex',
  flexDirection: 'row',
  alignItems: 'flex-start',
  transition: 'right 0.3s ease-in-out',
  zIndex: theme.zIndex.drawer - 1,
  boxShadow: theme.shadows[5],
  borderTopLeftRadius: theme.shape.borderRadius,
  borderBottomLeftRadius: theme.shape.borderRadius,
}));

const ToggleButton = styled(Box)(({ theme }) => ({
  width: 40,
  height: 60,
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  backgroundColor: theme.palette.primary.main,
  color: theme.palette.primary.contrastText,
  cursor: 'pointer',
  borderTopLeftRadius: theme.shape.borderRadius,
  borderBottomLeftRadius: theme.shape.borderRadius,
}));

const ContentContainer = styled(Box)({
  width: 340,
  height: 'auto',
  maxHeight: '80vh',
  overflowY: 'auto',
  padding: '16px',
});

const ProviderCard = styled(Paper)(({ theme }) => ({
  padding: theme.spacing(2),
  marginBottom: theme.spacing(2),
}));

const ProgressBar = styled(LinearProgress)<{ value: number }>(({ theme, value }) => ({
  height: 10,
  borderRadius: 5,
  marginTop: theme.spacing(0.5),
  '& .MuiLinearProgress-bar': {
    backgroundColor:
      value > 90
        ? theme.palette.error.main
        : value > 75
        ? theme.palette.warning.main
        : theme.palette.success.main,
  },
}));

const StatusRow = styled(Box)({
  display: 'flex',
  justifyContent: 'space-between',
  alignItems: 'center',
  marginBottom: '8px',
});

const ProviderStatusCard: React.FC<{ status: ProviderStatus }> = ({ status }) => {
  const dailyBudgetUsage =
    status.budget.daily_limit_usd > 0
      ? (status.budget.daily_spent_usd / status.budget.daily_limit_usd) * 100
      : 0;
  const monthlyBudgetUsage =
    status.budget.monthly_limit_usd > 0
      ? (status.budget.monthly_spent_usd / status.budget.monthly_limit_usd) * 100
      : 0;
  const dailyRequestUsage =
    status.requests.daily_limit > 0
      ? (status.requests.daily_count / status.requests.daily_limit) * 100
      : 0;
  const minuteRequestUsage =
    status.requests.minute_limit > 0
      ? (status.requests.requests_this_minute / status.requests.minute_limit) * 100
      : 0;

  const getCircuitColor = () => {
    switch (status.circuit_breaker.state) {
      case 'open':
        return 'error';
      case 'half_open':
        return 'warning';
      default:
        return 'success';
    }
  };

  return (
    <ProviderCard variant="outlined">
      <Box display="flex" alignItems="center" mb={2}>
        <LanIcon sx={{ mr: 1.5 }} color="primary" />
        <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
          {status.provider_id.charAt(0).toUpperCase() + status.provider_id.slice(1)}
        </Typography>
        <Tooltip title={`Circuit Breaker: ${status.circuit_breaker.state}`}>
          <Chip
            label={status.circuit_breaker.state}
            color={getCircuitColor()}
            size="small"
            variant="filled"
          />
        </Tooltip>
      </Box>
      <Divider sx={{ mb: 2 }} />
      
      <Typography variant="subtitle2" gutterBottom>Budget (USD)</Typography>
      <StatusRow>
        <Typography variant="body2">Daily:</Typography>
        <Typography variant="body2">${status.budget.daily_spent_usd.toFixed(2)} / ${status.budget.daily_limit_usd.toFixed(2)}</Typography>
      </StatusRow>
      <ProgressBar variant="determinate" value={dailyBudgetUsage} />

      <StatusRow sx={{ mt: 1 }}>
        <Typography variant="body2">Monthly:</Typography>
        <Typography variant="body2">${status.budget.monthly_spent_usd.toFixed(2)} / ${status.budget.monthly_limit_usd.toFixed(2)}</Typography>
      </StatusRow>
      <ProgressBar variant="determinate" value={monthlyBudgetUsage} />

      <Typography variant="subtitle2" gutterBottom sx={{ mt: 2 }}>Requests</Typography>
      <StatusRow>
        <Typography variant="body2">Per Minute:</Typography>
        <Typography variant="body2">{status.requests.requests_this_minute} / {status.requests.minute_limit}</Typography>
      </StatusRow>
      <ProgressBar variant="determinate" value={minuteRequestUsage} />

      <StatusRow sx={{ mt: 1 }}>
        <Typography variant="body2">Daily:</Typography>
        <Typography variant="body2">{status.requests.daily_count} / {status.requests.daily_limit}</Typography>
      </StatusRow>
      <ProgressBar variant="determinate" value={dailyRequestUsage} />
    </ProviderCard>
  );
};

const ProviderHealthSidebar: React.FC = () => {
  const [isCollapsed, setIsCollapsed] = useState(true);

  const { data, isLoading, isError, error } = useQuery({
    queryKey: ['providerStatus'],
    queryFn: backpressureAPI.getAllProviderStatus,
    refetchInterval: 15000, // Refetch every 15 seconds
  });

  const toggleSidebar = () => {
    setIsCollapsed(!isCollapsed);
  };

  const overallStatusIcon = () => {
    if (isLoading) return <CircularProgress size={24} color="inherit" />;
    if (isError) return <ErrorOutlineIcon />;

    const hasIssues = Object.values(data || {}).some(
      p => p.circuit_breaker.state !== 'closed' ||
           (p.budget.daily_limit_usd > 0 && (p.budget.daily_spent_usd / p.budget.daily_limit_usd) > 0.9)
    );

    if (hasIssues) return <WarningAmberIcon />;
    return <CheckCircleOutlineIcon />;
  };

  return (
    <SidebarContainer collapsed={isCollapsed ? 1 : 0}>
      <Tooltip title={isCollapsed ? "Show Provider Health" : "Hide Provider Health"} placement="left">
        <ToggleButton onClick={toggleSidebar}>
          {isCollapsed ? <ChevronLeftIcon /> : <ChevronRightIcon />}
        </ToggleButton>
      </Tooltip>
      <Collapse in={!isCollapsed} orientation="horizontal">
        <ContentContainer>
          <Typography variant="h5" gutterBottom>
            Provider Health
          </Typography>
          {isLoading && <CircularProgress />}
          {isError && <Alert severity="error">Failed to load provider status.</Alert>}
          {data && Object.keys(data).length > 0 ? (
            Object.values(data).map(status => (
              <ProviderStatusCard key={status.provider_id} status={status} />
            ))
          ) : (
            !isLoading && <Typography>No providers configured.</Typography>
          )}
        </ContentContainer>
      </Collapse>
      {isCollapsed && (
         <Tooltip title="Provider Health Status" placement="left">
            <Box sx={{
                position: 'absolute',
                top: '50%',
                left: -20,
                transform: 'translateY(-50%)',
                backgroundColor: 'primary.main',
                color: 'white',
                borderRadius: '50%',
                width: 40,
                height: 40,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                boxShadow: 3,
            }}>
                {overallStatusIcon()}
            </Box>
         </Tooltip>
      )}
    </SidebarContainer>
  );
};

export default ProviderHealthSidebar;
