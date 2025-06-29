import React, { useState, useEffect, useCallback, useRef } from 'react';
import { Box, Card, CardContent, Typography, Tabs, Tab, CircularProgress, Grid, Chip, Drawer, Button, Alert, AlertTitle, Divider, LinearProgress, IconButton, Badge } from '@mui/material';
import { styled } from '@mui/material/styles';
import { useQuery, useInfiniteQuery } from '@tanstack/react-query';
import { getSimBalances, getSimActivity, getSimCollectibles, getSimTokenInfo } from '../../lib/api';
import { SimTokenBalance, SimActivityItem, SimCollectible, SimTokenInfo } from '../../lib/api';
import { formatAddress, formatAmount, formatUSD } from '../../lib/utils';
import InfoIcon from '@mui/icons-material/Info';
import CloseIcon from '@mui/icons-material/Close';
import WarningIcon from '@mui/icons-material/Warning';
import SecurityIcon from '@mui/icons-material/Security';
import OpenInNewIcon from '@mui/icons-material/OpenInNew';
import ErrorBoundary from '../ui/ErrorBoundary';
import LoadingSpinner from '../ui/LoadingSpinner';
import RiskBadge from './RiskBadge';
import EvidenceDrawer from './EvidenceDrawer';

// Styled components
const StyledTabPanel = styled(Box)(({ theme }) => ({
  padding: theme.spacing(3),
  height: '100%',
  overflowY: 'auto',
}));

const TokenItem = styled(Box, {
  shouldForwardProp: (prop) => prop !== 'lowLiquidity',
})<{ lowLiquidity?: boolean }>(({ theme, lowLiquidity }) => ({
  display: 'flex',
  alignItems: 'center',
  padding: theme.spacing(2),
  borderBottom: `1px solid ${theme.palette.divider}`,
  cursor: 'pointer',
  '&:hover': {
    backgroundColor: theme.palette.action.hover,
  },
  ...(lowLiquidity && {
    opacity: 0.6,
  }),
}));

const TokenIcon = styled('img')({
  width: 36,
  height: 36,
  borderRadius: '50%',
  marginRight: 16,
  objectFit: 'contain',
});

const TokenPlaceholder = styled(Box)(({ theme }) => ({
  width: 36,
  height: 36,
  borderRadius: '50%',
  marginRight: 16,
  backgroundColor: theme.palette.grey[300],
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  color: theme.palette.text.secondary,
  fontSize: '0.75rem',
  fontWeight: 'bold',
}));

const ActivityItem = styled(Box)(({ theme }) => ({
  display: 'flex',
  alignItems: 'flex-start',
  padding: theme.spacing(2),
  borderBottom: `1px solid ${theme.palette.divider}`,
}));

const ActivityIcon = styled(Box)(({ theme }) => ({
  width: 36,
  height: 36,
  borderRadius: '50%',
  marginRight: 16,
  backgroundColor: theme.palette.grey[300],
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  color: theme.palette.text.secondary,
  fontSize: '1rem',
}));

const CollectibleGrid = styled(Grid)(({ theme }) => ({
  marginTop: theme.spacing(2),
}));

const CollectibleCard = styled(Card)(({ theme }) => ({
  height: '100%',
  display: 'flex',
  flexDirection: 'column',
  cursor: 'pointer',
  transition: 'transform 0.2s ease-in-out',
  '&:hover': {
    transform: 'translateY(-4px)',
    boxShadow: theme.shadows[4],
  },
}));

const CollectibleImage = styled('img')({
  width: '100%',
  height: 200,
  objectFit: 'cover',
});

const CollectiblePlaceholder = styled(Box)(({ theme }) => ({
  width: '100%',
  height: 200,
  backgroundColor: theme.palette.grey[300],
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  color: theme.palette.text.secondary,
}));

const DrawerContent = styled(Box)(({ theme }) => ({
  width: 400,
  padding: theme.spacing(3),
  [theme.breakpoints.down('sm')]: {
    width: '100%',
  },
}));

const InfoRow = styled(Box)(({ theme }) => ({
  display: 'flex',
  justifyContent: 'space-between',
  padding: theme.spacing(1, 0),
  borderBottom: `1px solid ${theme.palette.divider}`,
}));

// Interface definitions
interface WalletAnalysisPanelProps {
  walletAddress: string;
}

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;

  return (
    <StyledTabPanel
      role="tabpanel"
      hidden={value !== index}
      id={`wallet-tabpanel-${index}`}
      aria-labelledby={`wallet-tab-${index}`}
      {...other}
    >
      {value === index && <>{children}</>}
    </StyledTabPanel>
  );
}

function a11yProps(index: number) {
  return {
    id: `wallet-tab-${index}`,
    'aria-controls': `wallet-tabpanel-${index}`,
  };
}

export default function WalletAnalysisPanel({ walletAddress }: WalletAnalysisPanelProps) {
  // State
  const [tabValue, setTabValue] = useState(0);
  const [selectedToken, setSelectedToken] = useState<SimTokenBalance | null>(null);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [evidenceDrawerOpen, setEvidenceDrawerOpen] = useState(false);
  const [selectedEvidenceId, setSelectedEvidenceId] = useState<string | null>(null);
  
  // Refs
  const activityContainerRef = useRef<HTMLDivElement>(null);

  // Queries
  const {
    data: balancesData,
    isLoading: balancesLoading,
    error: balancesError,
  } = useQuery({
    queryKey: ['simBalances', walletAddress],
    queryFn: () => getSimBalances(walletAddress),
    enabled: !!walletAddress,
  });

  const {
    data: activityData,
    isLoading: activityLoading,
    error: activityError,
    fetchNextPage,
    hasNextPage,
    isFetchingNextPage,
  } = useInfiniteQuery({
    queryKey: ['simActivity', walletAddress],
    queryFn: ({ pageParam }) => getSimActivity(walletAddress, 25, pageParam),
    getNextPageParam: (lastPage) => lastPage.next_offset || undefined,
    enabled: !!walletAddress && tabValue === 1,
  });

  const {
    data: collectiblesData,
    isLoading: collectiblesLoading,
    error: collectiblesError,
  } = useQuery({
    queryKey: ['simCollectibles', walletAddress],
    queryFn: () => getSimCollectibles(walletAddress),
    enabled: !!walletAddress && tabValue === 2,
  });

  const {
    data: tokenInfoData,
    isLoading: tokenInfoLoading,
    error: tokenInfoError,
  } = useQuery({
    queryKey: ['simTokenInfo', selectedToken?.address, selectedToken?.chain_id],
    queryFn: () => {
      if (!selectedToken) return null;
      return getSimTokenInfo(selectedToken.address, selectedToken.chain_id.toString());
    },
    enabled: !!selectedToken && drawerOpen,
  });

  // Handlers
  const handleTabChange = (_event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };

  const handleTokenClick = (token: SimTokenBalance) => {
    setSelectedToken(token);
    setDrawerOpen(true);
  };

  const handleCloseDrawer = () => {
    setDrawerOpen(false);
  };

  const handleActivityClick = (activity: SimActivityItem) => {
    // Placeholder logic: open evidence drawer for potentially risky activities
    const riskyTypes = ['swap', 'burn', 'approve'];
    if (riskyTypes.includes(activity.type)) {
      // In a real app, you'd get this ID from the activity item itself
      // or from a related analysis endpoint.
      setSelectedEvidenceId(`evt-${activity.transaction_hash}`);
      setEvidenceDrawerOpen(true);
    }
  };

  // Infinite scroll handler for activity tab
  const handleScroll = useCallback(() => {
    if (activityContainerRef.current) {
      const { scrollTop, scrollHeight, clientHeight } = activityContainerRef.current;
      if (scrollTop + clientHeight >= scrollHeight - 100 && hasNextPage && !isFetchingNextPage) {
        fetchNextPage();
      }
    }
  }, [fetchNextPage, hasNextPage, isFetchingNextPage]);

  // Add scroll event listener
  useEffect(() => {
    const currentRef = activityContainerRef.current;
    if (currentRef && tabValue === 1) {
      currentRef.addEventListener('scroll', handleScroll);
      return () => {
        currentRef.removeEventListener('scroll', handleScroll);
      };
    }
  }, [tabValue, handleScroll]);

  // Render functions
  const renderTokens = () => {
    if (balancesLoading) {
      return <LoadingSpinner size="medium" />;
    }

    if (balancesError) {
      return (
        <Alert severity="error">
          <AlertTitle>Error loading balances</AlertTitle>
          Failed to load token balances. Please try again.
        </Alert>
      );
    }

    if (!balancesData?.balances?.length) {
      return (
        <Alert severity="info">
          <AlertTitle>No tokens found</AlertTitle>
          This wallet doesn't have any tokens or balances.
        </Alert>
      );
    }

    return (
      <>
        {balancesData.balances.map((token) => (
          <TokenItem 
            key={`${token.chain_id}-${token.address}`} 
            onClick={() => handleTokenClick(token)}
            lowLiquidity={token.low_liquidity}
          >
            {token.token_metadata?.logo ? (
              <TokenIcon src={token.token_metadata.logo} alt={token.symbol} />
            ) : (
              <TokenPlaceholder>{token.symbol?.substring(0, 2) || '?'}</TokenPlaceholder>
            )}
            <Box sx={{ flexGrow: 1 }}>
              <Typography variant="subtitle1">
                {token.symbol}
                {token.low_liquidity && (
                  <Chip 
                    size="small" 
                    label="Low Liquidity" 
                    color="default" 
                    sx={{ ml: 1, opacity: 0.7 }} 
                  />
                )}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                {formatAmount(token.amount, token.decimals)} • {token.chain}
              </Typography>
            </Box>
            <Box sx={{ textAlign: 'right' }}>
              <Typography variant="subtitle1">{formatUSD(token.value_usd)}</Typography>
              <Typography variant="body2" color="text.secondary">
                {formatUSD(token.price_usd)} per token
              </Typography>
            </Box>
          </TokenItem>
        ))}
      </>
    );
  };

  const renderActivity = () => {
    if (activityLoading && !activityData) {
      return <LoadingSpinner size="medium" />;
    }

    if (activityError) {
      return (
        <Alert severity="error">
          <AlertTitle>Error loading activity</AlertTitle>
          Failed to load transaction activity. Please try again.
        </Alert>
      );
    }

    const allActivities = activityData?.pages.flatMap(page => page.activity) || [];

    if (!allActivities.length) {
      return (
        <Alert severity="info">
          <AlertTitle>No activity found</AlertTitle>
          This wallet doesn't have any recent transaction activity.
        </Alert>
      );
    }

    return (
      <Box ref={activityContainerRef} sx={{ height: '100%', overflowY: 'auto' }}>
        {allActivities.map((activity, index) => {
          const isRisky = ['swap', 'burn', 'approve'].includes(activity.type);
          return (
            <ActivityItem 
              key={`${activity.transaction_hash}-${index}`}
              onClick={() => handleActivityClick(activity)}
              sx={{ 
                cursor: isRisky ? 'pointer' : 'default',
                '&:hover': {
                  backgroundColor: isRisky ? 'action.hover' : 'transparent',
                }
              }}
            >
              <ActivityIcon>
                {activity.type === 'send' && '↑'}
                {activity.type === 'receive' && '↓'}
                {activity.type === 'swap' && '↔'}
                {activity.type !== 'send' && activity.type !== 'receive' && activity.type !== 'swap' && '•'}
              </ActivityIcon>
              <Box sx={{ flexGrow: 1 }}>
                <Typography variant="subtitle1">
                  {activity.type.charAt(0).toUpperCase() + activity.type.slice(1)}
                  {activity.function?.name && `: ${activity.function.name}`}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  {activity.type === 'send' && `To: ${formatAddress(activity.to_address)}`}
                  {activity.type === 'receive' && `From: ${formatAddress(activity.from_address)}`}
                  {activity.type !== 'send' && activity.type !== 'receive' && 
                    `${formatAddress(activity.from_address)} → ${formatAddress(activity.to_address)}`}
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  {new Date(activity.block_time).toLocaleString()} • {activity.chain}
                </Typography>
              </Box>
              {activity.amount && (
                <Box sx={{ textAlign: 'right' }}>
                  <Typography 
                    variant="subtitle1" 
                    color={activity.type === 'send' ? 'error.main' : activity.type === 'receive' ? 'success.main' : 'inherit'}
                  >
                    {activity.type === 'send' ? '-' : activity.type === 'receive' ? '+' : ''}
                    {formatAmount(activity.amount, activity.token_metadata?.decimals || 18)} {activity.token_metadata?.symbol || ''}
                  </Typography>
                  {activity.value_usd && (
                    <Typography variant="body2" color="text.secondary">
                      {formatUSD(activity.value_usd)}
                    </Typography>
                  )}
                </Box>
              )}
            </ActivityItem>
          );
        })}
        {isFetchingNextPage && (
          <Box sx={{ display: 'flex', justifyContent: 'center', p: 2 }}>
            <CircularProgress size={24} />
          </Box>
        )}
        {!hasNextPage && allActivities.length > 0 && (
          <Box sx={{ textAlign: 'center', p: 2 }}>
            <Typography variant="body2" color="text.secondary">
              No more activity to load
            </Typography>
          </Box>
        )}
      </Box>
    );
  };

  const renderCollectibles = () => {
    if (collectiblesLoading) {
      return <LoadingSpinner size="medium" />;
    }

    if (collectiblesError) {
      return (
        <Alert severity="error">
          <AlertTitle>Error loading collectibles</AlertTitle>
          Failed to load NFT collectibles. Please try again.
        </Alert>
      );
    }

    const collectibles = collectiblesData?.entries || [];

    if (!collectibles.length) {
      return (
        <Alert severity="info">
          <AlertTitle>No collectibles found</AlertTitle>
          This wallet doesn't own any NFT collectibles.
        </Alert>
      );
    }

    return (
      <CollectibleGrid container spacing={2}>
        {collectibles.map((collectible) => {
          const openSeaUrl = `https://opensea.io/assets/${collectible.chain}/${collectible.contract_address}/${collectible.token_id}`;
          
          return (
            <Grid item xs={12} sm={6} md={4} key={`${collectible.contract_address}-${collectible.token_id}`}>
              <CollectibleCard>
                <a 
                  href={openSeaUrl} 
                  target="_blank" 
                  rel="noopener noreferrer"
                  style={{ textDecoration: 'none', color: 'inherit' }}
                >
                  {collectible.image_url ? (
                    <CollectibleImage src={collectible.image_url} alt={collectible.name || `NFT #${collectible.token_id}`} />
                  ) : (
                    <CollectiblePlaceholder>
                      NFT #{collectible.token_id.substring(0, 8)}...
                    </CollectiblePlaceholder>
                  )}
                  <CardContent>
                    <Typography variant="subtitle1" noWrap>
                      {collectible.name || `NFT #${collectible.token_id.substring(0, 8)}...`}
                    </Typography>
                    <Typography variant="body2" color="text.secondary" noWrap>
                      {collectible.collection_name || collectible.contract_address.substring(0, 8)}...
                    </Typography>
                    <Box sx={{ display: 'flex', alignItems: 'center', mt: 1 }}>
                      <Chip size="small" label={collectible.chain} sx={{ mr: 1 }} />
                      <Chip size="small" label={collectible.token_standard || 'NFT'} />
                      <Box sx={{ flexGrow: 1 }} />
                      <OpenInNewIcon fontSize="small" color="action" />
                    </Box>
                  </CardContent>
                </a>
              </CollectibleCard>
            </Grid>
          );
        })}
      </CollectibleGrid>
    );
  };

  const renderTokenInfoDrawer = () => {
    return (
      <Drawer
        anchor="right"
        open={drawerOpen}
        onClose={handleCloseDrawer}
      >
        <DrawerContent>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
            <Typography variant="h6">Token Details</Typography>
            <IconButton onClick={handleCloseDrawer}>
              <CloseIcon />
            </IconButton>
          </Box>
          
          {tokenInfoLoading && <CircularProgress size={24} />}
          
          {tokenInfoError && (
            <Alert severity="error">
              <AlertTitle>Error loading token details</AlertTitle>
              Failed to load token information. Please try again.
            </Alert>
          )}
          
          {selectedToken && (
            <>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
                {selectedToken.token_metadata?.logo ? (
                  <TokenIcon 
                    src={selectedToken.token_metadata.logo} 
                    alt={selectedToken.symbol} 
                    sx={{ width: 48, height: 48, mr: 2 }}
                  />
                ) : (
                  <TokenPlaceholder sx={{ width: 48, height: 48, mr: 2 }}>
                    {selectedToken.symbol?.substring(0, 2) || '?'}
                  </TokenPlaceholder>
                )}
                <Box>
                  <Typography variant="h6">
                    {selectedToken.symbol}
                    {selectedToken.low_liquidity && (
                      <Chip 
                        size="small" 
                        label="Low Liquidity" 
                        color="default" 
                        sx={{ ml: 1, opacity: 0.7 }} 
                      />
                    )}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    {selectedToken.name || 'Unknown Token'}
                  </Typography>
                </Box>
              </Box>
              
              <Divider sx={{ my: 2 }} />
              
              <Typography variant="subtitle1" gutterBottom>Balance</Typography>
              <InfoRow>
                <Typography variant="body2" color="text.secondary">Amount</Typography>
                <Typography variant="body2">
                  {formatAmount(selectedToken.amount, selectedToken.decimals)} {selectedToken.symbol}
                </Typography>
              </InfoRow>
              <InfoRow>
                <Typography variant="body2" color="text.secondary">Value (USD)</Typography>
                <Typography variant="body2">{formatUSD(token.value_usd)}</Typography>
              </InfoRow>
              <InfoRow>
                <Typography variant="body2" color="text.secondary">Chain</Typography>
                <Typography variant="body2">{selectedToken.chain}</Typography>
              </InfoRow>
              
              <Divider sx={{ my: 2 }} />
              
              <Typography variant="subtitle1" gutterBottom>Token Info</Typography>
              <InfoRow>
                <Typography variant="body2" color="text.secondary">Price (USD)</Typography>
                <Typography variant="body2">{formatUSD(selectedToken.price_usd)}</Typography>
              </InfoRow>
              <InfoRow>
                <Typography variant="body2" color="text.secondary">Decimals</Typography>
                <Typography variant="body2">{selectedToken.decimals}</Typography>
              </InfoRow>
              <InfoRow>
                <Typography variant="body2" color="text.secondary">Contract</Typography>
                <Typography variant="body2" sx={{ wordBreak: 'break-all' }}>
                  {selectedToken.address === 'native' ? 'Native Token' : formatAddress(selectedToken.address)}
                </Typography>
              </InfoRow>
              
              {tokenInfoData && tokenInfoData.entries && tokenInfoData.entries.length > 0 && (
                <>
                  <Divider sx={{ my: 2 }} />
                  <Typography variant="subtitle1" gutterBottom>Market Data</Typography>
                  <InfoRow>
                    <Typography variant="body2" color="text.secondary">Liquidity (USD)</Typography>
                    <Typography variant="body2">
                      {formatUSD(tokenInfoData.entries[0].pool_size_usd || 0)}
                    </Typography>
                  </InfoRow>
                  <InfoRow>
                    <Typography variant="body2" color="text.secondary">Pool Type</Typography>
                    <Typography variant="body2">
                      {tokenInfoData.entries[0].pool_type || 'Unknown'}
                    </Typography>
                  </InfoRow>
                  {tokenInfoData.entries[0].total_supply && (
                    <InfoRow>
                      <Typography variant="body2" color="text.secondary">Total Supply</Typography>
                      <Typography variant="body2">
                        {formatAmount(tokenInfoData.entries[0].total_supply, selectedToken.decimals)}
                      </Typography>
                    </InfoRow>
                  )}
                </>
              )}
              
              {selectedToken.token_metadata?.url && (
                <Button 
                  variant="outlined" 
                  startIcon={<OpenInNewIcon />}
                  href={selectedToken.token_metadata.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  sx={{ mt: 3 }}
                  fullWidth
                >
                  View Token Website
                </Button>
              )}
            </>
          )}
        </DrawerContent>
      </Drawer>
    );
  };

  return (
    <ErrorBoundary>
      <Card sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
        <Box sx={{ p: 2, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <Typography variant="h5" component="div">
                {formatAddress(walletAddress)}
            </Typography>
            {walletAddress && <RiskBadge walletAddress={walletAddress} />}
        </Box>
        <Divider />
        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <Tabs 
            value={tabValue} 
            onChange={handleTabChange} 
            aria-label="wallet analysis tabs"
            variant="fullWidth"
          >
            <Tab label="Tokens" {...a11yProps(0)} />
            <Tab label="Activity" {...a11yProps(1)} />
            <Tab label="Collectibles" {...a11yProps(2)} />
          </Tabs>
        </Box>
        
        <TabPanel value={tabValue} index={0}>
          {renderTokens()}
        </TabPanel>
        
        <TabPanel value={tabValue} index={1}>
          {renderActivity()}
        </TabPanel>
        
        <TabPanel value={tabValue} index={2}>
          {renderCollectibles()}
        </TabPanel>
        
        {renderTokenInfoDrawer()}
        <EvidenceDrawer
          evidenceBundleId={selectedEvidenceId}
          open={evidenceDrawerOpen}
          onClose={() => setEvidenceDrawerOpen(false)}
        />
      </Card>
    </ErrorBoundary>
  );
}
