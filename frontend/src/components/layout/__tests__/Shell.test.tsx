import React from 'react';
import { render, screen, waitFor, act } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import '@testing-library/jest-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { useRouter } from 'next/navigation';

import Shell from '../Shell';
import { AuthProvider, useAuth } from '../../../hooks/useAuth';
import { useToast } from '../../../hooks/useToast';
import * as api from '../../../lib/api';

// Mock Next.js useRouter
jest.mock('next/navigation', () => ({
  useRouter: jest.fn(),
  usePathname: jest.fn(() => '/'),
}));

// Mock useAuth hook
jest.mock('../../../hooks/useAuth', () => ({
  useAuth: jest.fn(),
  AuthProvider: ({ children }: { children: React.ReactNode }) => <div>{children}</div>, // Simple mock for provider
}));

// Mock useToast hook
jest.mock('../../../hooks/useToast', () => ({
  useToast: jest.fn(),
}));

// Mock apiClient for health checks
jest.mock('../../../lib/api', () => ({
  ...jest.requireActual('../../../lib/api'),
  apiClient: {
    get: jest.fn(),
  },
}));

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: false, // Disable retries for tests
    },
  },
});

const renderWithClient = (ui: React.ReactElement) => {
  return render(
    <QueryClientProvider client={queryClient}>
      <AuthProvider>{ui}</AuthProvider>
    </QueryClientProvider>
  );
};

describe('Shell', () => {
  const mockUser = {
    id: 'user-123',
    username: 'testuser',
    email: 'test@example.com',
    is_active: true,
    is_superuser: false,
  };
  const mockToast = jest.fn();
  const mockPush = jest.fn();
  const mockLogout = jest.fn();

  beforeEach(() => {
    jest.clearAllMocks();
    queryClient.clear();
    (useRouter as jest.Mock).mockReturnValue({ push: mockPush, pathname: '/' });
    (useToast as jest.Mock).mockReturnValue({ toast: mockToast });
    (useAuth as jest.Mock).mockReturnValue({
      user: mockUser,
      isAuthenticated: true,
      isLoading: false,
      logout: mockLogout,
      login: jest.fn(),
      register: jest.fn(),
      refreshToken: jest.fn(),
      fetchCurrentUser: jest.fn(),
      isSessionValid: jest.fn(() => true),
    });

    // Default mock for health check
    (api.apiClient.get as jest.Mock).mockImplementation((url: string) => {
      if (url.includes('/health')) {
        return Promise.resolve({ data: { status: 'healthy' } });
      }
      return Promise.reject(new Error('Not mocked'));
    });
  });

  // 1. Basic rendering with sidebar, header, and main content area
  test('renders basic layout elements', async () => {
    renderWithClient(<Shell />);

    expect(screen.getByRole('banner')).toBeInTheDocument(); // Header
    expect(screen.getByRole('navigation')).toBeInTheDocument(); // Sidebar
    expect(screen.getByRole('main')).toBeInTheDocument(); // Main content area
    expect(screen.getByText(mockUser.username)).toBeInTheDocument(); // User info
  });

  // 2. Sidebar toggle functionality (open/close)
  test('toggles sidebar visibility when toggle button is clicked', async () => {
    const user = userEvent.setup();
    renderWithClient(<Shell />);

    const sidebar = screen.getByRole('navigation');
    const toggleButton = screen.getByLabelText('Toggle sidebar');

    // Sidebar should be visible initially
    expect(sidebar).toHaveClass('w-64'); // Assuming this class indicates visible sidebar

    // Click to close
    await user.click(toggleButton);
    
    // Sidebar should be collapsed
    expect(sidebar).toHaveClass('w-16'); // Assuming this class indicates collapsed sidebar
    
    // Click to open again
    await user.click(toggleButton);
    
    // Sidebar should be visible again
    expect(sidebar).toHaveClass('w-64');
  });

  // 3. Navigation link rendering and active state
  test('renders navigation links with correct active state', async () => {
    (useRouter as jest.Mock).mockReturnValue({ push: mockPush, pathname: '/dashboard' });
    renderWithClient(<Shell />);

    const dashboardLink = screen.getByRole('link', { name: /dashboard/i });
    const analysisLink = screen.getByRole('link', { name: /analysis/i });
    
    // Dashboard link should be active
    expect(dashboardLink).toHaveClass('bg-primary-700');
    // Analysis link should not be active
    expect(analysisLink).not.toHaveClass('bg-primary-700');
  });

  test('navigates when links are clicked', async () => {
    const user = userEvent.setup();
    renderWithClient(<Shell />);

    const analysisLink = screen.getByRole('link', { name: /analysis/i });
    
    // Click analysis link
    await user.click(analysisLink);
    
    // Should navigate to analysis page
    expect(mockPush).toHaveBeenCalledWith('/analysis');
  });

  // 4. User menu interactions (open, logout)
  test('opens user menu and logs out when logout is clicked', async () => {
    const user = userEvent.setup();
    renderWithClient(<Shell />);

    // Open user menu
    const userButton = screen.getByText(mockUser.username);
    await user.click(userButton);
    
    // Logout button should be visible
    const logoutButton = await screen.findByRole('menuitem', { name: /logout/i });
    expect(logoutButton).toBeInTheDocument();
    
    // Click logout
    await user.click(logoutButton);
    
    // Should call logout function
    expect(mockLogout).toHaveBeenCalledTimes(1);
    expect(mockToast).toHaveBeenCalledWith(expect.objectContaining({
      title: 'Success',
      description: 'Logged out successfully',
    }));
  });

  // 5. Search modal functionality (open, close, search input)
  test('opens search modal, allows input, and closes', async () => {
    const user = userEvent.setup();
    renderWithClient(<Shell />);

    // Open search modal
    const searchButton = screen.getByLabelText('Search');
    await user.click(searchButton);
    
    // Search modal should be visible
    const searchInput = await screen.findByPlaceholderText('Search...');
    expect(searchInput).toBeInTheDocument();
    
    // Type in search input
    await user.type(searchInput, 'test query');
    expect(searchInput).toHaveValue('test query');
    
    // Close search modal
    const closeButton = screen.getByLabelText('Close search');
    await user.click(closeButton);
    
    // Search modal should be closed
    await waitFor(() => {
      expect(screen.queryByPlaceholderText('Search...')).not.toBeInTheDocument();
    });
  });

  // 6. Keyboard shortcuts for sidebar and search
  test('opens search modal with keyboard shortcut', async () => {
    renderWithClient(<Shell />);
    
    // Press Ctrl+K
    await act(async () => {
      fireEvent.keyDown(document.body, { key: 'k', ctrlKey: true });
    });
    
    // Search modal should be visible
    const searchInput = await screen.findByPlaceholderText('Search...');
    expect(searchInput).toBeInTheDocument();
  });

  test('closes search modal with Escape key', async () => {
    renderWithClient(<Shell />);
    
    // Open search modal
    await act(async () => {
      fireEvent.keyDown(document.body, { key: 'k', ctrlKey: true });
    });
    
    // Search modal should be visible
    const searchInput = await screen.findByPlaceholderText('Search...');
    expect(searchInput).toBeInTheDocument();
    
    // Press Escape
    await act(async () => {
      fireEvent.keyDown(searchInput, { key: 'Escape' });
    });
    
    // Search modal should be closed
    await waitFor(() => {
      expect(screen.queryByPlaceholderText('Search...')).not.toBeInTheDocument();
    });
  });

  test('toggles sidebar with keyboard shortcut', async () => {
    renderWithClient(<Shell />);
    
    const sidebar = screen.getByRole('navigation');
    
    // Initially sidebar is expanded
    expect(sidebar).toHaveClass('w-64');
    
    // Press Ctrl+B
    await act(async () => {
      fireEvent.keyDown(document.body, { key: 'b', ctrlKey: true });
    });
    
    // Sidebar should be collapsed
    expect(sidebar).toHaveClass('w-16');
    
    // Press Ctrl+B again
    await act(async () => {
      fireEvent.keyDown(document.body, { key: 'b', ctrlKey: true });
    });
    
    // Sidebar should be expanded again
    expect(sidebar).toHaveClass('w-64');
  });

  // 7. Responsive behavior
  test('collapses sidebar automatically on small screens', async () => {
    // Mock window.innerWidth to simulate small screen
    Object.defineProperty(window, 'innerWidth', {
      writable: true,
      configurable: true,
      value: 500, // Mobile width
    });

    // Trigger resize event
    global.dispatchEvent(new Event('resize'));
    
    renderWithClient(<Shell />);
    
    const sidebar = screen.getByRole('navigation');
    
    // Sidebar should be collapsed on small screens
    expect(sidebar).toHaveClass('w-16');
    
    // Reset window.innerWidth
    Object.defineProperty(window, 'innerWidth', {
      writable: true,
      configurable: true,
      value: 1024, // Desktop width
    });
  });

  // 8. Integration with auth context for user info and logout
  test('displays user information from auth context', async () => {
    renderWithClient(<Shell />);
    
    // User info should be displayed
    expect(screen.getByText(mockUser.username)).toBeInTheDocument();
    
    // User role should be displayed
    expect(screen.getByText(/analyst/i)).toBeInTheDocument();
  });

  test('redirects to login page when not authenticated', async () => {
    // Mock unauthenticated state
    (useAuth as jest.Mock).mockReturnValue({
      user: null,
      isAuthenticated: false,
      isLoading: false,
      logout: mockLogout,
    });
    
    renderWithClient(<Shell />);
    
    // Should redirect to login
    expect(mockPush).toHaveBeenCalledWith('/login');
  });

  // 9. Health indicator display and updates
  test('displays health indicator with status', async () => {
    renderWithClient(<Shell />);
    
    // Health indicator should show healthy status
    await waitFor(() => {
      expect(screen.getByText(/API: Healthy/i)).toBeInTheDocument();
    });
  });

  test('updates health indicator when status changes', async () => {
    // First render with healthy status
    (api.apiClient.get as jest.Mock).mockImplementation((url: string) => {
      if (url.includes('/health')) {
        return Promise.resolve({ data: { status: 'healthy' } });
      }
      return Promise.reject(new Error('Not mocked'));
    });
    
    renderWithClient(<Shell />);
    
    // Health indicator should show healthy status
    await waitFor(() => {
      expect(screen.getByText(/API: Healthy/i)).toBeInTheDocument();
    });
    
    // Update mock to return unhealthy status
    (api.apiClient.get as jest.Mock).mockImplementation((url: string) => {
      if (url.includes('/health')) {
        return Promise.resolve({ data: { status: 'degraded' } });
      }
      return Promise.reject(new Error('Not mocked'));
    });
    
    // Manually trigger health check interval
    await act(async () => {
      jest.advanceTimersByTime(60000); // Advance 1 minute
    });
    
    // Health indicator should show degraded status
    await waitFor(() => {
      expect(screen.getByText(/API: Degraded/i)).toBeInTheDocument();
    });
  });

  test('handles health check API errors', async () => {
    // Mock API error
    (api.apiClient.get as jest.Mock).mockRejectedValue(new Error('API error'));
    
    renderWithClient(<Shell />);
    
    // Health indicator should show error status
    await waitFor(() => {
      expect(screen.getByText(/API: Error/i)).toBeInTheDocument();
    });
  });

  // 10. Context ribbon rendering and interactions
  test('renders context ribbon with default text', async () => {
    renderWithClient(<Shell />);
    
    // Context ribbon should be visible with default text
    expect(screen.getByText(/Context: Global/i)).toBeInTheDocument();
  });

  test('renders context ribbon with custom text', async () => {
    renderWithClient(<Shell contextText="Investigating wallet 0x123..." />);
    
    // Context ribbon should show custom text
    expect(screen.getByText(/Investigating wallet 0x123.../i)).toBeInTheDocument();
  });

  test('clears context when clear button is clicked', async () => {
    const user = userEvent.setup();
    renderWithClient(<Shell contextText="Investigating wallet 0x123..." />);
    
    // Context ribbon should show custom text
    expect(screen.getByText(/Investigating wallet 0x123.../i)).toBeInTheDocument();
    
    // Click clear button
    const clearButton = screen.getByLabelText('Clear context');
    await user.click(clearButton);
    
    // Context should be cleared
    expect(screen.getByText(/Context: Global/i)).toBeInTheDocument();
    expect(screen.queryByText(/Investigating wallet 0x123.../i)).not.toBeInTheDocument();
  });
});
