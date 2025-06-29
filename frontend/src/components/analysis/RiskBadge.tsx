import React from 'react';
import { useQuery } from '@tanstack/react-query';
import { Chip, Tooltip, CircularProgress, Box } from '@mui/material';
import { styled } from '@mui/material/styles';
import SecurityIcon from '@mui/icons-material/Security';
import WarningAmberIcon from '@mui/icons-material/WarningAmber';
import GppBadIcon from '@mui/icons-material/GppBad';
import HelpOutlineIcon from '@mui/icons-material/HelpOutline';

import { analysisAPI } from '../../lib/api';
import { handleAPIError } from '../../lib/api';

interface RiskBadgeProps {
  walletAddress: string;
}

const StyledChip = styled(Chip)(({ theme }) => ({
  fontWeight: 'bold',
  textTransform: 'uppercase',
  fontSize: '0.75rem',
  height: '24px',
}));

const RiskBadge: React.FC<RiskBadgeProps> = ({ walletAddress }) => {
  const { data: riskData, isLoading, isError, error } = useQuery({
    queryKey: ['riskScore', walletAddress],
    queryFn: () => analysisAPI.getRiskScore(walletAddress),
    enabled: !!walletAddress, // Only run query if walletAddress is provided
    staleTime: 5 * 60 * 1000, // 5 minutes
    refetchOnWindowFocus: false,
  });

  if (isLoading) {
    return (
      <Tooltip title="Fetching risk score...">
        <Box display="flex" alignItems="center">
          <CircularProgress size={20} />
        </Box>
      </Tooltip>
    );
  }

  if (isError) {
    const apiError = handleAPIError(error);
    return (
      <Tooltip title={`Error fetching risk score: ${apiError.message}`}>
        <StyledChip
          icon={<HelpOutlineIcon />}
          label="Unknown Risk"
          color="default"
          size="small"
        />
      </Tooltip>
    );
  }

  if (!riskData) {
    return null; // Or a placeholder
  }

  const { risk_level, risk_score, risk_factors } = riskData;

  const getRiskConfig = () => {
    switch (risk_level) {
      case 'LOW':
        return {
          color: 'success' as const,
          icon: <SecurityIcon />,
          label: `Low Risk (${risk_score})`,
        };
      case 'MEDIUM':
        return {
          color: 'warning' as const,
          icon: <WarningAmberIcon />,
          label: `Medium Risk (${risk_score})`,
        };
      case 'HIGH':
        return {
          color: 'error' as const,
          icon: <GppBadIcon />,
          label: `High Risk (${risk_score})`,
        };
      default:
        return {
          color: 'default' as const,
          icon: <HelpOutlineIcon />,
          label: `Unknown Risk (${risk_score})`,
        };
    }
  };

  const config = getRiskConfig();

  const tooltipTitle = (
    <div>
      <strong>Risk Factors:</strong>
      {risk_factors && risk_factors.length > 0 ? (
        <ul>
          {risk_factors.map((factor, index) => (
            <li key={index}>{factor}</li>
          ))}
        </ul>
      ) : (
        <p>No specific risk factors identified.</p>
      )}
    </div>
  );

  return (
    <Tooltip title={tooltipTitle} arrow>
      <StyledChip
        icon={config.icon}
        label={config.label}
        color={config.color}
        size="small"
        variant="filled"
      />
    </Tooltip>
  );
};

export default RiskBadge;
