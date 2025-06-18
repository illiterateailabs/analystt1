import React from 'react';
import { render, screen, waitFor, act } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import '@testing-library/jest-dom';
import { useRouter, usePathname } from 'next/navigation';

import Header from '../Header';
import { useAuth } from '../../../hooks/useAuth';
import { useToast } from '../../../hooks/useToast';
import * as api from '../../../lib/api';

// Mock Next.js useRouter and usePathname
jest.mock('next/navigation', () => ({
  useRouter: jest.fn(),
  usePathname: jest.fn(),
}));

// Mock useAuth hook
jest.mock('../../../hooks/useAuth', () => ({
  useAuth: jest.fn(),
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

describe('Header', () => {
  const mockUser = {
    id: 'user-123',
    username: 'testuser',
    email: 'test@example.com',
    is_active: true,
    is_superuser: false,
    role: 'user',
  };
  const mockToast = jest.fn();
  const mockPush = jest.fn();
  const mockLogout = jest.fn();
  const mockOnSidebarToggle = jest.fn();
  const mockOnSearchToggle = jest.fn();

  beforeEach(() => {
    jest.clearAllMocks();
    (useRouter as jest.Mock).mockReturnValue({ push: mockPush });
    (usePathname as jest.Mock).mockReturnValue('/'); // Default path
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
      if (url.includes('/health/neo4j')) {
        return Promise.resolve({ data: { status: 'connected', version: '5.0' } });
      }
      if (url.includes('/health')) {
        return Promise.resolve({ data: { status: 'healthy' } });
      }
      return Promise.reject(new Error('Not mocked'));
    });
  });

  // 1. Rendering the title based on current path.
  test('renders correct title for dashboard path', () => {
    (usePathname as jest.Mock).mockReturnValue('/dashboard');
    render(<Header onSidebarToggle={mockOnSidebarToggle} onSearchToggle={mockOnSearchToggle} />);
    expect(screen.getByText('Dashboard')).toBeInTheDocument();
  });

  test('renders correct title for analysis path', () => {
    (usePathname as jest.Mock).mockReturnValue('/analysis');
    render(<Header onSidebarToggle={mockOnSidebarToggle} onSearchToggle={mockOnSearchToggle} />);
    expect(screen.getByText('Analysis')).toBeInTheDocument();
  });

  test('renders correct title for chat path', () => {
    (usePathname as jest.Mock).mockReturnValue('/chat');
    render(<Header onSidebarToggle={mockOnSidebarToggle} onSearchToggle={mockOnSearchToggle} />);
    expect(screen.getByText('Chat')).toBeInTheDocument();
  });

  test('renders correct title for prompts path', () => {
    (usePathname as jest.Mock).mockReturnValue('/prompts');
    render(<Header onSidebarToggle={mockOnSidebarToggle} onSearchToggle={mockOnSearchToggle} />);
    expect(screen.getByText('Prompts')).toBeInTheDocument();
  });

  test('renders correct title for templates path', () => {
    (usePathname as jest.Mock).mockReturnValue('/templates');
    render(<Header onSidebarToggle={mockOnSidebarToggle} onSearchToggle={mockOnSearchToggle} />);
    expect(screen.getByText('Templates')).toBeInTheDocument();
  });

  test('renders correct title for settings path', () => {
    (usePathname as jest.Mock).mockReturnValue('/settings');
    render(<Header onSidebarToggle={mockOnSidebarToggle} onSearchToggle={mockOnSearchToggle} />);
    expect(screen.getByText('Settings')).toBeInTheDocument();
  });

  test('renders default title for unknown path', () => {
    (usePathname as jest.Mock).mockReturnValue('/unknown');
    render(<Header onSidebarToggle={mockOnSidebarToggle} onSearchToggle={mockOnSearchToggle} />);
    expect(screen.getByText('Analyst Agent')).toBeInTheDocument();
  });

  // 2. Displaying health indicators (API, Neo4j, etc.) and their status.
  test('displays API and Neo4j health indicators when healthy', async () => {
    render(<Header onSidebarToggle={mockOnSidebarToggle} onSearchToggle={mockOnSearchToggle} />);

    await waitFor(() => {
      expect(screen.getByText(/API: Healthy/i)).toBeInTheDocument();
      expect(screen.getByText(/Neo4j: Connected/i)).toBeInTheDocument();
    });
  });

  test('displays API health indicator as degraded', async () => {
    (api.apiClient.get as jest.Mock).mockImplementation((url: string) => {
      if (url.includes('/health')) {
        return Promise.resolve({ data: { status: 'degraded' } });
      }
      return Promise.resolve({ data: { status: 'connected' } }); // Neo4j still connected
    });
    render(<Header onSidebarToggle={mockOnSidebarToggle} onSearchToggle={mockOnSearchToggle} />);

    await waitFor(() => {
      expect(screen.getByText(/API: Degraded/i)).toBeInTheDocument();
    });
  });

  test('displays Neo4j health indicator as error', async () => {
    (api.apiClient.get as jest.Mock).mockImplementation((url: string) => {
      if (url.includes('/health/neo4j')) {
        return Promise.resolve({ data: { status: 'error', message: 'Connection refused' } });
      }
      return Promise.resolve({ data: { status: 'healthy' } }); // API still healthy
    });
    render(<Header onSidebarToggle={mockOnSidebarToggle} onSearchToggle={mockOnSearchToggle} />);

    await waitFor(() => {
      expect(screen.getByText(/Neo4j: Error/i)).toBeInTheDocument();
    });
  });

  test('handles health check API errors gracefully', async () => {
    (api.apiClient.get as jest.Mock).mockRejectedValue(new Error('Network error'));
    render(<Header onSidebarToggle={mockOnSidebarToggle} onSearchToggle={mockOnSearchToggle} />);

    await waitFor(() => {
      expect(screen.getByText(/API: Error/i)).toBeInTheDocument();
      expect(screen.getByText(/Neo4j: Error/i)).toBeInTheDocument();
    });
  });

  // 3. User menu rendering and interactions (displaying username, logout action).
  test('displays username in user menu', async () => {
    render(<Header onSidebarToggle={mockOnSidebarToggle} onSearchToggle={mockOnSearchToggle} />);
    expect(screen.getByText(mockUser.username)).toBeInTheDocument();
  });

  test('opens user menu and logs out when logout is clicked', async () => {
    const user = userEvent.setup();
    render(<Header onSidebarToggle={mockOnSidebarToggle} onSearchToggle={mockOnSearchToggle} />);

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
      description: 'Logged out successfully',
      variant: 'info',
    }));
  });

  // 4. Search button functionality (opening search modal - can be a mock interaction).
  test('calls onSearchToggle when search button is clicked', async () => {
    const user = userEvent.setup();
    render(<Header onSidebarToggle={mockOnSidebarToggle} onSearchToggle={mockOnSearchToggle} />);

    const searchButton = screen.getByLabelText('Search');
    await user.click(searchButton);

    expect(mockOnSearchToggle).toHaveBeenCalledTimes(1);
  });

  // 5. Sidebar toggle button functionality (mock interaction).
  test('calls onSidebarToggle when sidebar toggle button is clicked', async () => {
    const user = userEvent.setup();
    render(<Header onSidebarToggle={mockOnSidebarToggle} onSearchToggle={mockOnSearchToggle} />);

    const toggleButton = screen.getByLabelText('Toggle sidebar');
    await user.click(toggleButton);

    expect(mockOnSidebarToggle).toHaveBeenCalledTimes(1);
  });

  // 6. Integration with useAuth for user information.
  test('displays guest info when not authenticated', () => {
    (useAuth as jest.Mock).mockReturnValue({
      user: null,
      isAuthenticated: false,
      isLoading: false,
    });
    render(<Header onSidebarToggle={mockOnSidebarToggle} onSearchToggle={mockOnSearchToggle} />);
    expect(screen.getByText('Guest')).toBeInTheDocument();
    expect(screen.getByText('Not Authenticated')).toBeInTheDocument();
  });

  test('displays loading state for user info when auth is loading', () => {
    (useAuth as jest.Mock).mockReturnValue({
      user: null,
      isAuthenticated: false,
      isLoading: true,
    });
    render(<Header onSidebarToggle={mockOnSidebarToggle} onSearchToggle={mockOnSearchToggle} />);
    expect(screen.getByRole('progressbar')).toBeInTheDocument(); // Spinner for user info
    expect(screen.queryByText(mockUser.username)).not.toBeInTheDocument();
  });
});
