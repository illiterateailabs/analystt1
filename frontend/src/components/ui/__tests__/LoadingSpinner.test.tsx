import React from 'react';
import { render, screen } from '@testing-library/react';
import '@testing-library/jest-dom';
import LoadingSpinner from '../LoadingSpinner';

describe('LoadingSpinner', () => {
  // 1. Correct rendering with default props.
  test('renders with default props (medium size, primary color)', () => {
    render(<LoadingSpinner />);
    const spinner = screen.getByRole('progressbar');
    expect(spinner).toBeInTheDocument();
    expect(spinner).toHaveClass('MuiCircularProgress-medium'); // Default size
    expect(spinner).toHaveClass('MuiCircularProgress-colorPrimary'); // Default color
    expect(spinner).toHaveAttribute('aria-label', 'Loading...');
  });

  // 2. Rendering with different sizes (small, medium, large).
  test('renders with small size', () => {
    render(<LoadingSpinner size="small" />);
    const spinner = screen.getByRole('progressbar');
    expect(spinner).toHaveClass('MuiCircularProgress-small');
  });

  test('renders with large size', () => {
    render(<LoadingSpinner size="large" />);
    const spinner = screen.getByRole('progressbar');
    expect(spinner).toHaveClass('MuiCircularProgress-large');
  });

  // 3. Rendering with different color variants (primary, secondary, inherit).
  test('renders with secondary color', () => {
    render(<LoadingSpinner color="secondary" />);
    const spinner = screen.getByRole('progressbar');
    expect(spinner).toHaveClass('MuiCircularProgress-colorSecondary');
  });

  test('renders with inherit color', () => {
    render(<LoadingSpinner color="inherit" />);
    const spinner = screen.getByRole('progressbar');
    expect(spinner).toHaveClass('MuiCircularProgress-colorInherit');
  });

  // 4. Accessibility attributes (aria-label, role).
  test('has correct accessibility attributes', () => {
    render(<LoadingSpinner />);
    const spinner = screen.getByRole('progressbar');
    expect(spinner).toHaveAttribute('aria-label', 'Loading...');
  });

  test('allows custom aria-label', () => {
    render(<LoadingSpinner ariaLabel="Fetching data..." />);
    const spinner = screen.getByRole('progressbar');
    expect(spinner).toHaveAttribute('aria-label', 'Fetching data...');
  });

  // 5. Custom className passthrough.
  test('applies custom className', () => {
    render(<LoadingSpinner className="custom-spinner-class" />);
    const spinner = screen.getByRole('progressbar');
    expect(spinner).toHaveClass('custom-spinner-class');
  });
});
