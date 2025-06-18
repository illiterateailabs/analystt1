import { renderHook, act, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { useRouter } from 'next/navigation';
import Cookies from 'js-cookie';

import { AuthProvider, useAuth } from '../useAuth';
import * as api from '../../lib/api';
import { useToast } from '../useToast';

// Mock the API functions
jest.mock('../../lib/api', () => ({
  ...jest.requireActual('../../lib/api'),
  authAPI: {
    login: jest.fn(),
    register: jest.fn(),
    logout: jest.fn(),
    getCurrentUser: jest.fn(),
  },
  apiClient: {
    post: jest.fn(),
    get: jest.fn(),
  },
}));

// Mock Next.js useRouter
jest.mock('next/navigation', () => ({
  useRouter: jest.fn(),
}));

// Mock useToast hook
jest.mock('../useToast', () => ({
  useToast: jest.fn(),
}));

// Mock js-cookie
jest.mock('js-cookie', () => ({
  get: jest.fn(),
  set: jest.fn(),
  remove: jest.fn(),
}));

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: false, // Disable retries for tests
    },
  },
});

const wrapper = ({ children }: { children: React.ReactNode }) => (
  <QueryClientProvider client={queryClient}>
    <AuthProvider>{children}</AuthProvider>
  </QueryClientProvider>
);

describe('useAuth', () => {
  const mockToast = jest.fn();
  const mockPush = jest.fn();

  beforeEach(() => {
    jest.clearAllMocks();
    queryClient.clear();
    (useRouter as jest.Mock).mockReturnValue({ push: mockPush });
    (useToast as jest.Mock).mockReturnValue({ toast: mockToast });
    (Cookies.get as jest.Mock).mockReturnValue(undefined); // Default no CSRF token
    (api.authAPI.getCurrentUser as jest.Mock).mockRejectedValue(new Error('Not authenticated')); // Default unauthenticated
  });

  // Helper to set document.cookie for CSRF token simulation
  const setCsrfCookie = (token: string) => {
    Object.defineProperty(document, 'cookie', {
      get: jest.fn(() => `csrf_token=${token}`),
      configurable: true,
    });
  };

  // Helper to clear document.cookie
  const clearCsrfCookie = () => {
    Object.defineProperty(document, 'cookie', {
      get: jest.fn(() => ''),
      configurable: true,
    });
  };

  // 1. Initial authentication state
  test('should return initial unauthenticated state', async () => {
    const { result } = renderHook(() => useAuth(), { wrapper });

    expect(result.current.user).toBeNull();
    expect(result.current.isAuthenticated).toBe(false);
    expect(result.current.isLoading).toBe(true); // Initial loading state

    await waitFor(() => expect(result.current.isLoading).toBe(false)); // After initial fetchCurrentUser
    expect(api.authAPI.getCurrentUser).toHaveBeenCalledTimes(1);
  });

  test('should set authenticated state if user is already logged in', async () => {
    const mockUser = { id: '1', username: 'test', email: 'test@example.com', is_active: true, is_superuser: false };
    (api.authAPI.getCurrentUser as jest.Mock).mockResolvedValue(mockUser);

    const { result } = renderHook(() => useAuth(), { wrapper });

    await waitFor(() => expect(result.current.isLoading).toBe(false));
    expect(result.current.user).toEqual(mockUser);
    expect(result.current.isAuthenticated).toBe(true);
  });

  // 2. Login functionality and state updates
  test('login should update user state and redirect on success', async () => {
    const mockLoginData = { username: 'test', password: 'password' };
    const mockUser = { id: '1', username: 'test', email: 'test@example.com', is_active: true, is_superuser: false };
    (api.authAPI.login as jest.Mock).mockResolvedValue({ user: mockUser });
    setCsrfCookie('mock-csrf-token'); // Simulate CSRF token presence

    const { result } = renderHook(() => useAuth(), { wrapper });
    await waitFor(() => expect(result.current.isLoading).toBe(false)); // Wait for initial load

    await act(async () => {
      await result.current.login(mockLoginData.username, mockLoginData.password);
    });

    expect(api.authAPI.login).toHaveBeenCalledWith(mockLoginData.username, mockLoginData.password, false);
    expect(result.current.user).toEqual(mockUser);
    expect(result.current.isAuthenticated).toBe(true);
    expect(mockToast).toHaveBeenCalledWith(expect.objectContaining({ description: 'Login successful' }));
    expect(mockPush).toHaveBeenCalledWith('/dashboard');
    expect(result.current.isLoading).toBe(false);
  });

  test('login should handle rememberMe option', async () => {
    const mockLoginData = { username: 'test', password: 'password' };
    const mockUser = { id: '1', username: 'test', email: 'test@example.com', is_active: true, is_superuser: false };
    (api.authAPI.login as jest.Mock).mockResolvedValue({ user: mockUser });
    setCsrfCookie('mock-csrf-token');

    const { result } = renderHook(() => useAuth(), { wrapper });
    await waitFor(() => expect(result.current.isLoading).toBe(false));

    await act(async () => {
      await result.current.login(mockLoginData.username, mockLoginData.password, true);
    });

    expect(api.authAPI.login).toHaveBeenCalledWith(mockLoginData.username, mockLoginData.password, true);
  });

  // 3. Logout functionality and cleanup
  test('logout should clear user state and redirect', async () => {
    const mockUser = { id: '1', username: 'test', email: 'test@example.com', is_active: true, is_superuser: false };
    (api.authAPI.getCurrentUser as jest.Mock).mockResolvedValue(mockUser);
    (api.authAPI.logout as jest.Mock).mockResolvedValue(undefined);
    setCsrfCookie('mock-csrf-token');

    const { result } = renderHook(() => useAuth(), { wrapper });
    await waitFor(() => expect(result.current.isLoading).toBe(false));
    expect(result.current.isAuthenticated).toBe(true);

    await act(async () => {
      await result.current.logout();
    });

    expect(api.authAPI.logout).toHaveBeenCalledTimes(1);
    expect(result.current.user).toBeNull();
    expect(result.current.isAuthenticated).toBe(false);
    expect(mockToast).toHaveBeenCalledWith(expect.objectContaining({ description: 'Logged out successfully' }));
    expect(mockPush).toHaveBeenCalledWith('/login');
  });

  // 4. Registration functionality
  test('register should call API and redirect on success', async () => {
    const mockRegisterData = { username: 'newuser', email: 'new@example.com', password: 'password' };
    (api.authAPI.register as jest.Mock).mockResolvedValue({ user: { id: '2', ...mockRegisterData, is_active: true, is_superuser: false } });
    setCsrfCookie('mock-csrf-token');

    const { result } = renderHook(() => useAuth(), { wrapper });
    await waitFor(() => expect(result.current.isLoading).toBe(false));

    await act(async () => {
      await result.current.register(mockRegisterData.username, mockRegisterData.email, mockRegisterData.password);
    });

    expect(api.authAPI.register).toHaveBeenCalledWith(mockRegisterData.username, mockRegisterData.email, mockRegisterData.password);
    expect(mockToast).toHaveBeenCalledWith(expect.objectContaining({ description: 'Registration successful. Please log in.' }));
    expect(mockPush).toHaveBeenCalledWith('/login');
  });

  // 5. Token refresh mechanism
  test('refreshToken should call API and return true on success', async () => {
    const mockUser = { id: '1', username: 'test', email: 'test@example.com', is_active: true, is_superuser: false };
    (api.authAPI.getCurrentUser as jest.Mock).mockResolvedValue(mockUser);
    (api.apiClient.post as jest.Mock).mockResolvedValue({}); // Mock refresh endpoint
    setCsrfCookie('mock-csrf-token');

    const { result } = renderHook(() => useAuth(), { wrapper });
    await waitFor(() => expect(result.current.isLoading).toBe(false));

    let refreshResult;
    await act(async () => {
      refreshResult = await result.current.refreshToken();
    });

    expect(api.apiClient.post).toHaveBeenCalledWith('/auth/refresh', {}, { headers: { 'X-CSRF-Token': 'mock-csrf-token' } });
    expect(refreshResult).toBe(true);
    expect(result.current.user).toEqual(mockUser); // User state should be preserved
  });

  // 6. Error handling for auth failures
  test('login should show error toast on failure', async () => {
    const mockLoginData = { username: 'test', password: 'password' };
    (api.authAPI.login as jest.Mock).mockRejectedValue({ response: { data: { detail: 'Invalid credentials' } } });
    setCsrfCookie('mock-csrf-token');

    const { result } = renderHook(() => useAuth(), { wrapper });
    await waitFor(() => expect(result.current.isLoading).toBe(false));

    await act(async () => {
      await expect(result.current.login(mockLoginData.username, mockLoginData.password)).rejects.toThrow();
    });

    expect(mockToast).toHaveBeenCalledWith(expect.objectContaining({ description: 'Invalid credentials', variant: 'destructive' }));
    expect(result.current.user).toBeNull();
    expect(result.current.isAuthenticated).toBe(false);
  });

  test('register should show error toast on failure', async () => {
    const mockRegisterData = { username: 'newuser', email: 'new@example.com', password: 'password' };
    (api.authAPI.register as jest.Mock).mockRejectedValue({ response: { data: { detail: 'Username already exists' } } });
    setCsrfCookie('mock-csrf-token');

    const { result } = renderHook(() => useAuth(), { wrapper });
    await waitFor(() => expect(result.current.isLoading).toBe(false));

    await act(async () => {
      await expect(result.current.register(mockRegisterData.username, mockRegisterData.email, mockRegisterData.password)).rejects.toThrow();
    });

    expect(mockToast).toHaveBeenCalledWith(expect.objectContaining({ description: 'Username already exists', variant: 'destructive' }));
  });

  // 7. Automatic logout on token expiration
  test('refreshToken should clear state and redirect on failure', async () => {
    const mockUser = { id: '1', username: 'test', email: 'test@example.com', is_active: true, is_superuser: false };
    (api.authAPI.getCurrentUser as jest.Mock).mockResolvedValue(mockUser);
    (api.apiClient.post as jest.Mock).mockRejectedValue(new Error('Token expired')); // Simulate refresh failure
    setCsrfCookie('mock-csrf-token');

    const { result } = renderHook(() => useAuth(), { wrapper });
    await waitFor(() => expect(result.current.isLoading).toBe(false));

    let refreshResult;
    await act(async () => {
      refreshResult = await result.current.refreshToken();
    });

    expect(refreshResult).toBe(false);
    expect(result.current.user).toBeNull();
    expect(result.current.isAuthenticated).toBe(false);
    expect(mockToast).toHaveBeenCalledWith(expect.objectContaining({ description: 'Session expired, please log in again.', variant: 'destructive' }));
    expect(mockPush).toHaveBeenCalledWith('/login');
  });

  test('should set up refresh interval when authenticated', async () => {
    jest.useFakeTimers();
    const mockUser = { id: '1', username: 'test', email: 'test@example.com', is_active: true, is_superuser: false };
    (api.authAPI.getCurrentUser as jest.Mock).mockResolvedValue(mockUser);
    (api.apiClient.post as jest.Mock).mockResolvedValue({}); // Mock successful refresh
    setCsrfCookie('mock-csrf-token');

    const { result } = renderHook(() => useAuth(), { wrapper });
    await waitFor(() => expect(result.current.isLoading).toBe(false));
    expect(result.current.isAuthenticated).toBe(true);

    // Fast-forward past the refresh interval
    jest.advanceTimersByTime(14 * 60 * 1000); // 14 minutes

    expect(api.apiClient.post).toHaveBeenCalledWith('/auth/refresh', {}, { headers: { 'X-CSRF-Token': 'mock-csrf-token' } });
    
    jest.useRealTimers();
  });

  // 8. User data persistence and retrieval
  test('user data is retrieved on mount', async () => {
    const mockUser = { id: '1', username: 'test', email: 'test@example.com', is_active: true, is_superuser: false };
    (api.authAPI.getCurrentUser as jest.Mock).mockResolvedValue(mockUser);

    const { result } = renderHook(() => useAuth(), { wrapper });

    await waitFor(() => expect(result.current.isLoading).toBe(false));
    expect(api.authAPI.getCurrentUser).toHaveBeenCalledTimes(1);
    expect(result.current.user).toEqual(mockUser);
    expect(result.current.isAuthenticated).toBe(true);
  });

  test('fetchCurrentUser should update user state', async () => {
    const mockUser = { id: '1', username: 'test', email: 'test@example.com', is_active: true, is_superuser: false };
    (api.authAPI.getCurrentUser as jest.Mock).mockResolvedValue(mockUser);

    const { result } = renderHook(() => useAuth(), { wrapper });
    await waitFor(() => expect(result.current.isLoading).toBe(false));

    // Clear mock to track new calls
    (api.authAPI.getCurrentUser as jest.Mock).mockClear();
    
    // Call fetchCurrentUser manually
    await act(async () => {
      await result.current.fetchCurrentUser();
    });

    expect(api.authAPI.getCurrentUser).toHaveBeenCalledTimes(1);
    expect(result.current.user).toEqual(mockUser);
    expect(result.current.isAuthenticated).toBe(true);
  });

  // 9. CSRF token handling
  test('login sends CSRF token in header', async () => {
    const mockLoginData = { username: 'test', password: 'password' };
    const mockUser = { id: '1', username: 'test', email: 'test@example.com', is_active: true, is_superuser: false };
    (api.authAPI.login as jest.Mock).mockResolvedValue({ user: mockUser });
    setCsrfCookie('test-csrf-token-123');

    const { result } = renderHook(() => useAuth(), { wrapper });
    await waitFor(() => expect(result.current.isLoading).toBe(false));

    await act(async () => {
      await result.current.login(mockLoginData.username, mockLoginData.password);
    });

    expect(api.authAPI.login).toHaveBeenCalledWith(mockLoginData.username, mockLoginData.password, false);
    // Verify that the apiClient.post was called with the CSRF token in headers
    expect(api.apiClient.post).toHaveBeenCalledWith(
      '/auth/login',
      expect.any(Object),
      { headers: { 'X-CSRF-Token': 'test-csrf-token-123' } }
    );
  });

  test('logout sends CSRF token in header', async () => {
    const mockUser = { id: '1', username: 'test', email: 'test@example.com', is_active: true, is_superuser: false };
    (api.authAPI.getCurrentUser as jest.Mock).mockResolvedValue(mockUser);
    (api.authAPI.logout as jest.Mock).mockResolvedValue(undefined);
    setCsrfCookie('test-csrf-token-456');

    const { result } = renderHook(() => useAuth(), { wrapper });
    await waitFor(() => expect(result.current.isLoading).toBe(false));

    await act(async () => {
      await result.current.logout();
    });

    expect(api.authAPI.logout).toHaveBeenCalledTimes(1);
    expect(api.apiClient.post).toHaveBeenCalledWith(
      '/auth/logout',
      expect.any(Object),
      { headers: { 'X-CSRF-Token': 'test-csrf-token-456' } }
    );
  });

  test('register sends CSRF token in header', async () => {
    const mockRegisterData = { username: 'newuser', email: 'new@example.com', password: 'password' };
    (api.authAPI.register as jest.Mock).mockResolvedValue({ user: { id: '2', ...mockRegisterData, is_active: true, is_superuser: false } });
    setCsrfCookie('test-csrf-token-789');

    const { result } = renderHook(() => useAuth(), { wrapper });
    await waitFor(() => expect(result.current.isLoading).toBe(false));

    await act(async () => {
      await result.current.register(mockRegisterData.username, mockRegisterData.email, mockRegisterData.password);
    });

    expect(api.apiClient.post).toHaveBeenCalledWith(
      '/auth/register',
      expect.any(Object),
      { headers: { 'X-CSRF-Token': 'test-csrf-token-789' } }
    );
  });

  // 10. Session validation
  test('validates session with getCurrentUser on initialization', async () => {
    const mockUser = { id: '1', username: 'test', email: 'test@example.com', is_active: true, is_superuser: false };
    (api.authAPI.getCurrentUser as jest.Mock).mockResolvedValue(mockUser);

    renderHook(() => useAuth(), { wrapper });

    await waitFor(() => {
      expect(api.authAPI.getCurrentUser).toHaveBeenCalledTimes(1);
    });
  });

  test('handles session invalidation when getCurrentUser fails', async () => {
    (api.authAPI.getCurrentUser as jest.Mock).mockRejectedValue({ response: { status: 401, data: { detail: 'Invalid or expired token' } } });

    const { result } = renderHook(() => useAuth(), { wrapper });

    await waitFor(() => expect(result.current.isLoading).toBe(false));
    expect(result.current.user).toBeNull();
    expect(result.current.isAuthenticated).toBe(false);
  });

  test('isSessionValid returns true when authenticated', async () => {
    const mockUser = { id: '1', username: 'test', email: 'test@example.com', is_active: true, is_superuser: false };
    (api.authAPI.getCurrentUser as jest.Mock).mockResolvedValue(mockUser);

    const { result } = renderHook(() => useAuth(), { wrapper });
    await waitFor(() => expect(result.current.isLoading).toBe(false));

    expect(result.current.isSessionValid()).toBe(true);
  });

  test('isSessionValid returns false when not authenticated', async () => {
    (api.authAPI.getCurrentUser as jest.Mock).mockRejectedValue(new Error('Not authenticated'));

    const { result } = renderHook(() => useAuth(), { wrapper });
    await waitFor(() => expect(result.current.isLoading).toBe(false));

    expect(result.current.isSessionValid()).toBe(false);
  });
});
