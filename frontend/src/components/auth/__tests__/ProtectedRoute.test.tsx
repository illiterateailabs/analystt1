import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import { useRouter } from 'next/navigation';

import ProtectedRoute from '../ProtectedRoute';
import { useAuth } from '../../../hooks/useAuth';

// Mock Next.js useRouter
jest.mock('next/navigation', () => ({
  useRouter: jest.fn(),
  usePathname: jest.fn(() => '/protected'), // Mock current path
}));

// Mock useAuth hook
jest.mock('../../../hooks/useAuth', () => ({
  useAuth: jest.fn(),
}));

describe('ProtectedRoute', () => {
  const mockPush = jest.fn();
  const mockPathname = '/protected';

  beforeEach(() => {
    jest.clearAllMocks();
    (useRouter as jest.Mock).mockReturnValue({ push: mockPush, pathname: mockPathname });
    (useAuth as jest.Mock).mockReturnValue({
      user: null,
      isAuthenticated: false,
      isLoading: true, // Default to loading
      isSessionValid: jest.fn(() => false),
    });
  });

  // 1. Rendering children when user is authenticated
  test('renders children when user is authenticated', async () => {
    (useAuth as jest.Mock).mockReturnValue({
      user: { id: '1', username: 'testuser', is_superuser: false, role: 'user' },
      isAuthenticated: true,
      isLoading: false,
      isSessionValid: jest.fn(() => true),
    });

    render(
      <ProtectedRoute>
        <div data-testid="child-content">Protected Content</div>
      </ProtectedRoute>
    );

    await waitFor(() => {
      expect(screen.getByTestId('child-content')).toBeInTheDocument();
    });
    expect(screen.queryByRole('progressbar')).not.toBeInTheDocument(); // No loading spinner
    expect(mockPush).not.toHaveBeenCalled(); // No redirect
  });

  // 2. Redirecting to login when user is not authenticated
  test('redirects to login when user is not authenticated and not loading', async () => {
    (useAuth as jest.Mock).mockReturnValue({
      user: null,
      isAuthenticated: false,
      isLoading: false, // Finished loading, still unauthenticated
      isSessionValid: jest.fn(() => false),
    });

    render(
      <ProtectedRoute>
        <div data-testid="child-content">Protected Content</div>
      </ProtectedRoute>
    );

    await waitFor(() => {
      expect(mockPush).toHaveBeenCalledWith(`/login?returnTo=${encodeURIComponent(mockPathname)}`);
    });
    expect(screen.queryByTestId('child-content')).not.toBeInTheDocument();
    expect(screen.queryByRole('progressbar')).not.toBeInTheDocument();
  });

  // 3. Showing loading state while checking authentication
  test('shows loading spinner while checking authentication', async () => {
    render(
      <ProtectedRoute>
        <div data-testid="child-content">Protected Content</div>
      </ProtectedRoute>
    );

    expect(screen.getByRole('progressbar')).toBeInTheDocument(); // Loading spinner
    expect(screen.queryByTestId('child-content')).not.toBeInTheDocument();
    expect(mockPush).not.toHaveBeenCalled(); // No redirect yet
  });

  // 4. Role-based access control (if implemented)
  // 5. Handling different user roles (admin, user)
  test('renders children for admin user when admin role is required', async () => {
    (useAuth as jest.Mock).mockReturnValue({
      user: { id: '1', username: 'adminuser', is_superuser: true, role: 'admin' },
      isAuthenticated: true,
      isLoading: false,
      isSessionValid: jest.fn(() => true),
    });

    render(
      <ProtectedRoute requiredRoles={['admin']}>
        <div data-testid="admin-content">Admin Content</div>
      </ProtectedRoute>
    );

    await waitFor(() => {
      expect(screen.getByTestId('admin-content')).toBeInTheDocument();
    });
    expect(mockPush).not.toHaveBeenCalled();
  });

  test('redirects non-admin user when admin role is required', async () => {
    (useAuth as jest.Mock).mockReturnValue({
      user: { id: '1', username: 'testuser', is_superuser: false, role: 'user' },
      isAuthenticated: true,
      isLoading: false,
      isSessionValid: jest.fn(() => true),
    });

    render(
      <ProtectedRoute requiredRoles={['admin']}>
        <div data-testid="admin-content">Admin Content</div>
      </ProtectedRoute>
    );

    await waitFor(() => {
      expect(mockPush).toHaveBeenCalledWith('/unauthorized'); // Or a specific unauthorized page
    });
    expect(screen.queryByTestId('admin-content')).not.toBeInTheDocument();
  });

  test('renders children for user when user role is required', async () => {
    (useAuth as jest.Mock).mockReturnValue({
      user: { id: '1', username: 'testuser', is_superuser: false, role: 'user' },
      isAuthenticated: true,
      isLoading: false,
      isSessionValid: jest.fn(() => true),
    });

    render(
      <ProtectedRoute requiredRoles={['user']}>
        <div data-testid="user-content">User Content</div>
      </ProtectedRoute>
    );

    await waitFor(() => {
      expect(screen.getByTestId('user-content')).toBeInTheDocument();
    });
    expect(mockPush).not.toHaveBeenCalled();
  });

  test('renders children for admin user when user role is required (admin is also a user)', async () => {
    (useAuth as jest.Mock).mockReturnValue({
      user: { id: '1', username: 'adminuser', is_superuser: true, role: 'admin' },
      isAuthenticated: true,
      isLoading: false,
      isSessionValid: jest.fn(() => true),
    });

    render(
      <ProtectedRoute requiredRoles={['user']}>
        <div data-testid="user-content">User Content</div>
      </ProtectedRoute>
    );

    await waitFor(() => {
      expect(screen.getByTestId('user-content')).toBeInTheDocument();
    });
    expect(mockPush).not.toHaveBeenCalled();
  });

  // 6. Proper redirect after login (returnTo parameter)
  test('redirects to login with returnTo parameter', async () => {
    (useAuth as jest.Mock).mockReturnValue({
      user: null,
      isAuthenticated: false,
      isLoading: false,
      isSessionValid: jest.fn(() => false),
    });

    render(
      <ProtectedRoute>
        <div data-testid="child-content">Protected Content</div>
      </ProtectedRoute>
    );

    await waitFor(() => {
      expect(mockPush).toHaveBeenCalledWith(`/login?returnTo=${encodeURIComponent(mockPathname)}`);
    });
  });

  test('does not redirect if already on login page', async () => {
    (useRouter as jest.Mock).mockReturnValue({ push: mockPush, pathname: '/login' });
    (useAuth as jest.Mock).mockReturnValue({
      user: null,
      isAuthenticated: false,
      isLoading: false,
      isSessionValid: jest.fn(() => false),
    });

    render(
      <ProtectedRoute>
        <div data-testid="child-content">Protected Content</div>
      </ProtectedRoute>
    );

    await waitFor(() => {
      expect(mockPush).not.toHaveBeenCalled();
    });
    expect(screen.queryByTestId('child-content')).not.toBeInTheDocument();
  });
});
