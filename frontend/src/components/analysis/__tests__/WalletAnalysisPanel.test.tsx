import React from 'react';
import { render, screen, waitFor, act } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import '@testing-library/jest-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { useRouter } from 'next/navigation';

import WalletAnalysisPanel from '../WalletAnalysisPanel';
import * as api from '../../../lib/api';
import { useToast } from '../../../hooks/useToast';

// Mock the API functions
jest.mock('../../../lib/api', () => ({
  ...jest.requireActual('../../../lib/api'),
  getSimBalances: jest.fn(),
  getSimActivity: jest.fn(),
  getSimCollectibles: jest.fn(),
  getSimTokenInfo: jest.fn(),
  getSimRiskScore: jest.fn(),
}));

// Mock Next.js useRouter
jest.mock('next/navigation', () => ({
  useRouter: jest.fn(),
}));

// Mock useToast hook
jest.mock('../../../hooks/useToast', () => ({
  useToast: jest.fn(),
}));

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: false, // Disable retries for tests
    },
  },
});

const renderWithClient = (ui: React.ReactElement) => {
  return render(<QueryClientProvider client={queryClient}>{ui}</QueryClientProvider>);
};

describe('WalletAnalysisPanel', () => {
  const mockWalletAddress = '0x1234567890abcdef1234567890abcdef12345678';
  const mockToast = jest.fn();
  const mockPush = jest.fn();

  beforeEach(() => {
    jest.clearAllMocks();
    queryClient.clear();
    (useRouter as jest.Mock).mockReturnValue({ push: mockPush });
    (useToast as jest.Mock).mockReturnValue({ toast: mockToast });

    // Default successful mocks for API calls
    (api.getSimBalances as jest.Mock).mockResolvedValue({
      wallet_address: mockWalletAddress,
      balances: [
        {
          address: '0xToken1',
          amount: '1000000000000000000',
          chain: 'Ethereum',
          chain_id: 1,
          decimals: 18,
          symbol: 'ETH',
          price_usd: 3000,
          value_usd: 3000,
          token_metadata: { symbol: 'ETH', name: 'Ethereum', decimals: 18, logo: 'eth.png' },
          low_liquidity: false,
        },
        {
          address: '0xToken2',
          amount: '50000000',
          chain: 'Polygon',
          chain_id: 137,
          decimals: 6,
          symbol: 'USDC',
          price_usd: 1,
          value_usd: 50,
          token_metadata: { symbol: 'USDC', name: 'USD Coin', decimals: 6, logo: 'usdc.png' },
          low_liquidity: true,
        },
      ],
      count: 2,
      has_more: false,
    });

    (api.getSimActivity as jest.Mock).mockResolvedValue({
      wallet_address: mockWalletAddress,
      activity: [
        {
          id: 'tx1',
          type: 'send',
          chain: 'Ethereum',
          chain_id: 1,
          block_number: 12345,
          block_time: '1678886400', // March 15, 2023 12:00:00 PM UTC
          transaction_hash: '0xhash1',
          from_address: mockWalletAddress,
          to_address: '0xReceiver1',
          amount: '100000000000000000',
          value_usd: 300,
          token_metadata: { symbol: 'ETH' },
        },
      ],
      count: 1,
      has_more: false,
    });

    (api.getSimCollectibles as jest.Mock).mockResolvedValue({
      collectibles: [
        {
          contract_address: '0xNFT1',
          token_id: '1',
          chain: 'Ethereum',
          name: 'Cool NFT',
          image_url: 'nft1.png',
        },
      ],
      count: 1,
      has_more: false,
    });

    (api.getSimTokenInfo as jest.Mock).mockResolvedValue({
      token_info: [
        {
          address: '0xToken1',
          chain: 'Ethereum',
          symbol: 'ETH',
          name: 'Ethereum',
          decimals: 18,
          price_usd: 3000,
          pool_size_usd: 100000000,
          low_liquidity: false,
          total_supply: '100000000000000000000000000',
          pool_type: 'Uniswap V2',
        },
      ],
      count: 1,
      has_more: false,
    });

    (api.getSimRiskScore as jest.Mock).mockResolvedValue({
      wallet_address: mockWalletAddress,
      risk_score: 25,
      risk_level: 'LOW',
      risk_factors: ['No significant risks detected.'],
      summary: {},
    });
  });

  // 1. Component rendering with different wallet addresses
  test('renders WalletAnalysisPanel with provided wallet address', async () => {
    renderWithClient(<WalletAnalysisPanel walletAddress={mockWalletAddress} />);

    expect(screen.getByText('Tokens')).toBeInTheDocument();
    expect(screen.getByText('Activity')).toBeInTheDocument();
    expect(screen.getByText('Collectibles')).toBeInTheDocument();

    await waitFor(() => {
      expect(screen.getByText('ETH')).toBeInTheDocument();
      expect(screen.getByText('USDC')).toBeInTheDocument();
      expect(screen.getByText('Wallet Risk Score: 25/100')).toBeInTheDocument();
    });
  });

  test('renders with empty state when no wallet address is provided', async () => {
    renderWithClient(<WalletAnalysisPanel walletAddress="" />);

    await waitFor(() => {
      expect(screen.getByText('No wallet address provided')).toBeInTheDocument();
    });
  });

  // 2. Loading states for balances and activity
  test('shows loading spinners when data is being fetched', async () => {
    (api.getSimBalances as jest.Mock).mockReturnValue(new Promise(() => {})); // Never resolves
    (api.getSimActivity as jest.Mock).mockReturnValue(new Promise(() => {}));
    (api.getSimCollectibles as jest.Mock).mockReturnValue(new Promise(() => {}));
    (api.getSimRiskScore as jest.Mock).mockReturnValue(new Promise(() => {}));

    renderWithClient(<WalletAnalysisPanel walletAddress={mockWalletAddress} />);

    expect(screen.getAllByRole('progressbar')[0]).toBeInTheDocument(); // Balances loading
    // Activity tab is not active initially, so its spinner won't be visible
    // Collectibles tab is not active initially, so its spinner won't be visible
  });

  test('shows loading spinner for activity when switching to Activity tab', async () => {
    (api.getSimActivity as jest.Mock).mockReturnValue(new Promise(() => {})); // Never resolves

    renderWithClient(<WalletAnalysisPanel walletAddress={mockWalletAddress} />);

    // Switch to Activity tab
    await act(async () => {
      userEvent.click(screen.getByText('Activity'));
    });

    expect(screen.getByRole('progressbar')).toBeInTheDocument(); // Activity loading
  });

  test('shows loading spinner for collectibles when switching to Collectibles tab', async () => {
    (api.getSimCollectibles as jest.Mock).mockReturnValue(new Promise(() => {})); // Never resolves

    renderWithClient(<WalletAnalysisPanel walletAddress={mockWalletAddress} />);

    // Switch to Collectibles tab
    await act(async () => {
      userEvent.click(screen.getByText('Collectibles'));
    });

    expect(screen.getByRole('progressbar')).toBeInTheDocument(); // Collectibles loading
  });

  // 3. Error handling for API failures
  test('displays error message when balances API fails', async () => {
    (api.getSimBalances as jest.Mock).mockRejectedValue(new Error('Network error'));

    renderWithClient(<WalletAnalysisPanel walletAddress={mockWalletAddress} />);

    await waitFor(() => {
      expect(screen.getByText('Error loading balances')).toBeInTheDocument();
      expect(screen.getByText('Failed to load token balances. Please try again.')).toBeInTheDocument();
    });
  });

  test('displays error message when activity API fails', async () => {
    (api.getSimActivity as jest.Mock).mockRejectedValue(new Error('Server error'));

    renderWithClient(<WalletAnalysisPanel walletAddress={mockWalletAddress} />);

    // Switch to Activity tab
    await act(async () => {
      userEvent.click(screen.getByText('Activity'));
    });

    await waitFor(() => {
      expect(screen.getByText('Error loading activity')).toBeInTheDocument();
      expect(screen.getByText('Failed to load transaction activity. Please try again.')).toBeInTheDocument();
    });
  });

  test('displays error message when collectibles API fails', async () => {
    (api.getSimCollectibles as jest.Mock).mockRejectedValue(new Error('API limit exceeded'));

    renderWithClient(<WalletAnalysisPanel walletAddress={mockWalletAddress} />);

    // Switch to Collectibles tab
    await act(async () => {
      userEvent.click(screen.getByText('Collectibles'));
    });

    await waitFor(() => {
      expect(screen.getByText('Error loading collectibles')).toBeInTheDocument();
      expect(screen.getByText('Failed to load NFT collectibles. Please try again.')).toBeInTheDocument();
    });
  });

  test('displays error message when risk score API fails', async () => {
    (api.getSimRiskScore as jest.Mock).mockRejectedValue(new Error('Risk score calculation failed'));

    renderWithClient(<WalletAnalysisPanel walletAddress={mockWalletAddress} />);

    await waitFor(() => {
      expect(screen.getByText('Error loading risk score')).toBeInTheDocument();
      expect(screen.getByText('Failed to calculate wallet risk score.')).toBeInTheDocument();
    });
  });

  // 4. Risk score calculation and display
  test('displays correct risk score and level', async () => {
    (api.getSimRiskScore as jest.Mock).mockResolvedValue({
      wallet_address: mockWalletAddress,
      risk_score: 85,
      risk_level: 'HIGH',
      risk_factors: ['High value in low liquidity token.', 'Large value outflows detected.'],
      summary: {},
    });

    renderWithClient(<WalletAnalysisPanel walletAddress={mockWalletAddress} />);

    await waitFor(() => {
      expect(screen.getByText('Wallet Risk Score: 85/100')).toBeInTheDocument();
      expect(screen.getByText('HIGH')).toBeInTheDocument();
      expect(screen.getByText('High value in low liquidity token.')).toBeInTheDocument();
      expect(screen.getByText('Large value outflows detected.')).toBeInTheDocument();
    });
  });

  test('displays medium risk score correctly', async () => {
    (api.getSimRiskScore as jest.Mock).mockResolvedValue({
      wallet_address: mockWalletAddress,
      risk_score: 50,
      risk_level: 'MEDIUM',
      risk_factors: ['Some suspicious activity detected.'],
      summary: {},
    });

    renderWithClient(<WalletAnalysisPanel walletAddress={mockWalletAddress} />);

    await waitFor(() => {
      expect(screen.getByText('Wallet Risk Score: 50/100')).toBeInTheDocument();
      expect(screen.getByText('MEDIUM')).toBeInTheDocument();
      expect(screen.getByText('Some suspicious activity detected.')).toBeInTheDocument();
    });
  });

  // 5. Token balance formatting and display
  test('displays token balances with correct formatting and low liquidity chip', async () => {
    renderWithClient(<WalletAnalysisPanel walletAddress={mockWalletAddress} />);

    await waitFor(() => {
      expect(screen.getByText('ETH')).toBeInTheDocument();
      expect(screen.getByText('$3,000.00 per token')).toBeInTheDocument();
      expect(screen.getByText('$3,000.00')).toBeInTheDocument();

      expect(screen.getByText('USDC')).toBeInTheDocument();
      expect(screen.getByText('Low Liquidity')).toBeInTheDocument();
      expect(screen.getByText('$1.00 per token')).toBeInTheDocument();
      expect(screen.getByText('$50.00')).toBeInTheDocument();
    });
  });

  test('formats large token amounts with appropriate abbreviations', async () => {
    (api.getSimBalances as jest.Mock).mockResolvedValue({
      wallet_address: mockWalletAddress,
      balances: [
        {
          address: '0xToken3',
          amount: '1000000000000000000000',
          chain: 'Ethereum',
          chain_id: 1,
          decimals: 18,
          symbol: 'SHIB',
          price_usd: 0.00001,
          value_usd: 10000,
          token_metadata: { symbol: 'SHIB', name: 'Shiba Inu', decimals: 18, logo: 'shib.png' },
          low_liquidity: false,
        }
      ],
      count: 1,
      has_more: false,
    });

    renderWithClient(<WalletAnalysisPanel walletAddress={mockWalletAddress} />);

    await waitFor(() => {
      expect(screen.getByText('SHIB')).toBeInTheDocument();
      expect(screen.getByText('1,000.00K')).toBeInTheDocument(); // 1,000,000 abbreviated
      expect(screen.getByText('$10,000.00')).toBeInTheDocument();
    });
  });

  // 6. Activity feed rendering and filtering (and infinite scroll)
  test('displays activity feed and handles infinite scroll', async () => {
    (api.getSimActivity as jest.Mock)
      .mockResolvedValueOnce({
        wallet_address: mockWalletAddress,
        activity: [
          {
            id: 'tx1',
            type: 'send',
            chain: 'Ethereum',
            chain_id: 1,
            block_number: 1,
            block_time: '1678886400',
            transaction_hash: '0xhash1',
            from_address: mockWalletAddress,
            to_address: '0xReceiver1',
            amount: '1000000000000000000',
            value_usd: 300,
            token_metadata: { symbol: 'ETH', decimals: 18 },
          },
        ],
        count: 1,
        next_offset: 'offset1',
        has_more: true,
      })
      .mockResolvedValueOnce({
        wallet_address: mockWalletAddress,
        activity: [
          {
            id: 'tx2',
            type: 'receive',
            chain: 'Polygon',
            chain_id: 137,
            block_number: 2,
            block_time: '1678886500',
            transaction_hash: '0xhash2',
            from_address: '0xSender2',
            to_address: mockWalletAddress,
            amount: '50000000',
            value_usd: 50,
            token_metadata: { symbol: 'USDC', decimals: 6 },
          },
        ],
        count: 1,
        next_offset: null,
        has_more: false,
      });

    renderWithClient(<WalletAnalysisPanel walletAddress={mockWalletAddress} />);

    // Switch to Activity tab
    await act(async () => {
      userEvent.click(screen.getByText('Activity'));
    });

    await waitFor(() => {
      expect(screen.getByText('Send')).toBeInTheDocument();
      expect(screen.getByText(/To: 0xReceiver1/i)).toBeInTheDocument();
    });

    // Simulate scrolling to trigger next page load
    const activityContainer = screen.getByRole('tabpanel', { name: 'Activity' });
    Object.defineProperty(activityContainer, 'scrollTop', { value: 1000 });
    Object.defineProperty(activityContainer, 'scrollHeight', { value: 1100 });
    Object.defineProperty(activityContainer, 'clientHeight', { value: 100 });

    await act(async () => {
      activityContainer.dispatchEvent(new Event('scroll'));
    });

    await waitFor(() => {
      expect(api.getSimActivity).toHaveBeenCalledWith(mockWalletAddress, 'offset1');
      expect(screen.getByText('Receive')).toBeInTheDocument();
      expect(screen.getByText(/From: 0xSender2/i)).toBeInTheDocument();
    });
  });

  test('displays different activity types correctly', async () => {
    (api.getSimActivity as jest.Mock).mockResolvedValue({
      wallet_address: mockWalletAddress,
      activity: [
        {
          id: 'tx1',
          type: 'send',
          chain: 'Ethereum',
          chain_id: 1,
          block_time: '1678886400',
          transaction_hash: '0xhash1',
          from_address: mockWalletAddress,
          to_address: '0xReceiver1',
          token_metadata: { symbol: 'ETH' },
        },
        {
          id: 'tx2',
          type: 'receive',
          chain: 'Ethereum',
          chain_id: 1,
          block_time: '1678886500',
          transaction_hash: '0xhash2',
          from_address: '0xSender2',
          to_address: mockWalletAddress,
          token_metadata: { symbol: 'ETH' },
        },
        {
          id: 'tx3',
          type: 'swap',
          chain: 'Ethereum',
          chain_id: 1,
          block_time: '1678886600',
          transaction_hash: '0xhash3',
          from_address: mockWalletAddress,
          to_address: '0xDex1',
          token_metadata: { symbol: 'ETH' },
        },
        {
          id: 'tx4',
          type: 'approve',
          chain: 'Ethereum',
          chain_id: 1,
          block_time: '1678886700',
          transaction_hash: '0xhash4',
          from_address: mockWalletAddress,
          to_address: '0xToken1',
          token_metadata: { symbol: 'ETH' },
        },
      ],
      count: 4,
      has_more: false,
    });

    renderWithClient(<WalletAnalysisPanel walletAddress={mockWalletAddress} />);

    // Switch to Activity tab
    await act(async () => {
      userEvent.click(screen.getByText('Activity'));
    });

    await waitFor(() => {
      expect(screen.getByText('Send')).toBeInTheDocument();
      expect(screen.getByText('Receive')).toBeInTheDocument();
      expect(screen.getByText('Swap')).toBeInTheDocument();
      expect(screen.getByText('Approve')).toBeInTheDocument();
    });
  });

  // 7. Wallet address validation
  test('handles invalid wallet address gracefully', async () => {
    const invalidWalletAddress = '0xinvalid';
    (api.getSimBalances as jest.Mock).mockRejectedValue(new Error('Invalid wallet address'));

    renderWithClient(<WalletAnalysisPanel walletAddress={invalidWalletAddress} />);

    await waitFor(() => {
      expect(screen.getByText('Error loading balances')).toBeInTheDocument();
      expect(mockToast).toHaveBeenCalledWith({
        title: 'Error',
        description: expect.stringContaining('Failed to load token balances'),
        status: 'error',
      });
    });
  });

  // 8. Integration with the API client
  test('calls API with correct parameters', async () => {
    renderWithClient(<WalletAnalysisPanel walletAddress={mockWalletAddress} />);

    await waitFor(() => {
      expect(api.getSimBalances).toHaveBeenCalledWith(mockWalletAddress);
      expect(api.getSimRiskScore).toHaveBeenCalledWith(mockWalletAddress);
    });

    // Switch to Activity tab
    await act(async () => {
      userEvent.click(screen.getByText('Activity'));
    });

    await waitFor(() => {
      expect(api.getSimActivity).toHaveBeenCalledWith(mockWalletAddress, undefined);
    });

    // Switch to Collectibles tab
    await act(async () => {
      userEvent.click(screen.getByText('Collectibles'));
    });

    await waitFor(() => {
      expect(api.getSimCollectibles).toHaveBeenCalledWith(mockWalletAddress, undefined);
    });
  });

  // 9. User interactions (tab changes, drawer open/close)
  test('switches tabs correctly', async () => {
    renderWithClient(<WalletAnalysisPanel walletAddress={mockWalletAddress} />);

    // Initial tab is Tokens
    expect(screen.getByRole('tab', { name: 'Tokens' })).toHaveAttribute('aria-selected', 'true');
    expect(screen.getByRole('tab', { name: 'Activity' })).toHaveAttribute('aria-selected', 'false');
    expect(screen.getByRole('tab', { name: 'Collectibles' })).toHaveAttribute('aria-selected', 'false');

    // Switch to Activity tab
    await act(async () => {
      userEvent.click(screen.getByText('Activity'));
    });

    expect(screen.getByRole('tab', { name: 'Tokens' })).toHaveAttribute('aria-selected', 'false');
    expect(screen.getByRole('tab', { name: 'Activity' })).toHaveAttribute('aria-selected', 'true');
    expect(screen.getByRole('tab', { name: 'Collectibles' })).toHaveAttribute('aria-selected', 'false');

    // Switch to Collectibles tab
    await act(async () => {
      userEvent.click(screen.getByText('Collectibles'));
    });

    expect(screen.getByRole('tab', { name: 'Tokens' })).toHaveAttribute('aria-selected', 'false');
    expect(screen.getByRole('tab', { name: 'Activity' })).toHaveAttribute('aria-selected', 'false');
    expect(screen.getByRole('tab', { name: 'Collectibles' })).toHaveAttribute('aria-selected', 'true');
  });

  test('opens token detail drawer when clicking on a token', async () => {
    renderWithClient(<WalletAnalysisPanel walletAddress={mockWalletAddress} />);

    await waitFor(() => {
      expect(screen.getByText('ETH')).toBeInTheDocument();
    });

    // Click on the ETH token
    await act(async () => {
      userEvent.click(screen.getByText('ETH'));
    });

    // Check that the drawer is open with token details
    await waitFor(() => {
      expect(screen.getByText('Token Details')).toBeInTheDocument();
      expect(screen.getByText('Ethereum (ETH)')).toBeInTheDocument();
      expect(screen.getByText('Total Supply:')).toBeInTheDocument();
      expect(screen.getByText('Pool Size:')).toBeInTheDocument();
    });

    // Close the drawer
    const closeButton = screen.getByRole('button', { name: /close/i });
    await act(async () => {
      userEvent.click(closeButton);
    });

    // Check that the drawer is closed
    await waitFor(() => {
      expect(screen.queryByText('Token Details')).not.toBeInTheDocument();
    });
  });

  test('refreshes data when refresh button is clicked', async () => {
    renderWithClient(<WalletAnalysisPanel walletAddress={mockWalletAddress} />);

    await waitFor(() => {
      expect(screen.getByText('ETH')).toBeInTheDocument();
    });

    // Clear the mock calls to track new calls
    (api.getSimBalances as jest.Mock).mockClear();
    (api.getSimRiskScore as jest.Mock).mockClear();

    // Click the refresh button
    const refreshButton = screen.getByRole('button', { name: /refresh/i });
    await act(async () => {
      userEvent.click(refreshButton);
    });

    // Check that the APIs were called again
    await waitFor(() => {
      expect(api.getSimBalances).toHaveBeenCalledWith(mockWalletAddress);
      expect(api.getSimRiskScore).toHaveBeenCalledWith(mockWalletAddress);
    });
  });
});
