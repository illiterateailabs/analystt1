import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import { useRouter, usePathname } from 'next/navigation';

import Sidebar from '../Sidebar';
import { useAuth } from '../../../hooks/useAuth';

// Mock Next.js useRouter and usePathname
jest.mock('next/navigation', () => ({
  useRouter: jest.fn(),
  usePathname: jest.fn(),
}));

// Mock useAuth hook
jest.mock('../../../hooks/useAuth', () => ({
  useAuth: jest.fn(),
}));

describe('Sidebar', () => {
  const mockPush = jest.fn();
  const mockUser = {
    id: 'user-123',
    username: 'testuser',
    email: 'test@example.com',
    is_active: true,
    is_superuser: false,
    role: 'user', // Add role for testing
  };

  beforeEach(() => {
    jest.clearAllMocks();
    (useRouter as jest.Mock).mockReturnValue({ push: mockPush });
    (usePathname as jest.Mock).mockReturnValue('/'); // Default path
    (useAuth as jest.Mock).mockReturnValue({
      user: mockUser,
      isAuthenticated: true,
      isLoading: false,
    });
  });

  // 1. Rendering of navigation links.
  test('renders all navigation links', () => {
    render(<Sidebar isOpen={true} />);

    expect(screen.getByRole('link', { name: /dashboard/i })).toBeInTheDocument();
    expect(screen.getByRole('link', { name: /analysis/i })).toBeInTheDocument();
    expect(screen.getByRole('link', { name: /chat/i })).toBeInTheDocument();
    expect(screen.getByRole('link', { name: /prompts/i })).toBeInTheDocument();
    expect(screen.getByRole('link', { name: /templates/i })).toBeInTheDocument();
    expect(screen.getByRole('link', { name: /settings/i })).toBeInTheDocument();
  });

  // 2. Correct active link highlighting based on current path.
  test('highlights the active link based on current pathname', () => {
    (usePathname as jest.Mock).mockReturnValue('/analysis');
    render(<Sidebar isOpen={true} />);

    expect(screen.getByRole('link', { name: /dashboard/i })).not.toHaveClass('bg-primary-700');
    expect(screen.getByRole('link', { name: /analysis/i })).toHaveClass('bg-primary-700');
    expect(screen.getByRole('link', { name: /chat/i })).not.toHaveClass('bg-primary-700');
  });

  test('highlights dashboard as active for root path', () => {
    (usePathname as jest.Mock).mockReturnValue('/');
    render(<Sidebar isOpen={true} />);

    expect(screen.getByRole('link', { name: /dashboard/i })).toHaveClass('bg-primary-700');
  });

  // 3. Handling of sidebar open/closed state (props).
  test('applies correct classes when sidebar is open', () => {
    render(<Sidebar isOpen={true} />);
    const sidebar = screen.getByRole('navigation');
    expect(sidebar).toHaveClass('w-64');
    expect(sidebar).not.toHaveClass('w-16');
    expect(screen.getByText(mockUser.username)).toBeInTheDocument(); // User info visible
  });

  test('applies correct classes when sidebar is closed', () => {
    render(<Sidebar isOpen={false} />);
    const sidebar = screen.getByRole('navigation');
    expect(sidebar).toHaveClass('w-16');
    expect(sidebar).not.toHaveClass('w-64');
    expect(screen.queryByText(mockUser.username)).not.toBeInTheDocument(); // User info hidden
  });

  // 4. Rendering of user profile section (if part of Sidebar).
  test('renders user profile section when sidebar is open', () => {
    render(<Sidebar isOpen={true} />);
    expect(screen.getByText(mockUser.username)).toBeInTheDocument();
    expect(screen.getByText(mockUser.role)).toBeInTheDocument();
  });

  test('does not render user profile section when sidebar is closed', () => {
    render(<Sidebar isOpen={false} />);
    expect(screen.queryByText(mockUser.username)).not.toBeInTheDocument();
    expect(screen.queryByText(mockUser.role)).not.toBeInTheDocument();
  });

  test('renders placeholder for user profile if no user is authenticated', () => {
    (useAuth as jest.Mock).mockReturnValue({
      user: null,
      isAuthenticated: false,
      isLoading: false,
    });
    render(<Sidebar isOpen={true} />);
    expect(screen.getByText('Guest')).toBeInTheDocument();
    expect(screen.getByText('Not Authenticated')).toBeInTheDocument();
  });

  // 5. Accessibility attributes for navigation items.
  test('navigation links have correct accessibility attributes', () => {
    render(<Sidebar isOpen={true} />);
    const dashboardLink = screen.getByRole('link', { name: /dashboard/i });
    expect(dashboardLink).toHaveAttribute('aria-current', 'page'); // For active link
    expect(dashboardLink).toHaveAttribute('href', '/dashboard');

    const analysisLink = screen.getByRole('link', { name: /analysis/i });
    expect(analysisLink).not.toHaveAttribute('aria-current'); // For inactive link
    expect(analysisLink).toHaveAttribute('href', '/analysis');
  });

  test('sidebar has appropriate ARIA role', () => {
    render(<Sidebar isOpen={true} />);
    expect(screen.getByRole('navigation')).toBeInTheDocument();
  });
});
