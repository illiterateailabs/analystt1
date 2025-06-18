import React from 'react';
import { render, screen, act, waitFor, fireEvent } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import '@testing-library/jest-dom';
import ErrorBoundary from '../ErrorBoundary';

// Mock console.error to prevent test logs from cluttering the console
const mockConsoleError = jest.spyOn(console, 'error').mockImplementation(() => {});

// Component that throws an error during render
const ErrorThrowingComponent = ({ shouldThrow = true }) => {
  if (shouldThrow) {
    throw new Error('Test Error: Render error');
  }
  return <div>No error thrown</div>;
};

// Component that throws an error in an event handler
const EventErrorComponent = ({ shouldThrow = false }) => {
  const handleClick = () => {
    if (shouldThrow) {
      throw new Error('Test Error: Event handler error');
    }
  };

  return (
    <button onClick={handleClick} data-testid="error-button">
      Click to throw error
    </button>
  );
};

// Component that throws an error in useEffect
const EffectErrorComponent = ({ shouldThrow = false }) => {
  React.useEffect(() => {
    if (shouldThrow) {
      throw new Error('Test Error: Effect error');
    }
  }, [shouldThrow]);

  return <div>Effect component</div>;
};

// Custom fallback component for testing
const CustomFallback = ({ error, resetErrorBoundary }) => (
  <div data-testid="custom-fallback">
    <h2>Custom Error UI</h2>
    <p data-testid="error-message">{error.message}</p>
    <button onClick={resetErrorBoundary} data-testid="reset-button">
      Reset
    </button>
  </div>
);

describe('ErrorBoundary', () => {
  beforeEach(() => {
    mockConsoleError.mockClear();
  });

  afterAll(() => {
    mockConsoleError.mockRestore();
  });

  // 1. Normal rendering when no errors occur
  test('renders children normally when no errors occur', () => {
    render(
      <ErrorBoundary>
        <div data-testid="child-content">Child Content</div>
      </ErrorBoundary>
    );

    expect(screen.getByTestId('child-content')).toBeInTheDocument();
    expect(screen.getByText('Child Content')).toBeInTheDocument();
    expect(mockConsoleError).not.toHaveBeenCalled();
  });

  // 2. Error catching and fallback UI display
  test('renders fallback UI when child component throws', () => {
    // Using act because error boundaries use lifecycle methods that require it
    act(() => {
      render(
        <ErrorBoundary>
          <ErrorThrowingComponent />
        </ErrorBoundary>
      );
    });

    // Default fallback UI should be shown
    expect(screen.getByText('Something went wrong.')).toBeInTheDocument();
    expect(screen.getByText('Test Error: Render error')).toBeInTheDocument();
    expect(screen.queryByText('No error thrown')).not.toBeInTheDocument();
  });

  // 3. Error logging to console
  test('logs errors to console.error', () => {
    act(() => {
      render(
        <ErrorBoundary>
          <ErrorThrowingComponent />
        </ErrorBoundary>
      );
    });

    // Error should be logged to console
    expect(mockConsoleError).toHaveBeenCalled();
    // React 18 calls console.error twice during development - once for the error itself and once for the component stack
    expect(mockConsoleError.mock.calls[0][0]).toBeInstanceOf(Error);
    expect(mockConsoleError.mock.calls[0][0].message).toBe('Test Error: Render error');
  });

  // 4. Reset functionality
  test('resets error state when reset function is called', async () => {
    const onReset = jest.fn();
    const user = userEvent.setup();

    // Render with a component that will throw
    act(() => {
      render(
        <ErrorBoundary onReset={onReset} fallback={CustomFallback}>
          <ErrorThrowingComponent />
        </ErrorBoundary>
      );
    });

    // Verify error state
    expect(screen.getByTestId('custom-fallback')).toBeInTheDocument();
    expect(screen.getByTestId('error-message')).toHaveTextContent('Test Error: Render error');

    // Click reset button
    await user.click(screen.getByTestId('reset-button'));

    // Verify onReset was called
    expect(onReset).toHaveBeenCalledTimes(1);
  });

  test('renders children again after reset with resetKeys', async () => {
    const TestComponent = ({ shouldThrow }) => {
      return shouldThrow ? <ErrorThrowingComponent /> : <div>No error now</div>;
    };

    const { rerender } = render(
      <ErrorBoundary fallback={CustomFallback} resetKeys={[true]}>
        <TestComponent shouldThrow={true} />
      </ErrorBoundary>
    );

    // Verify error state
    expect(screen.getByTestId('custom-fallback')).toBeInTheDocument();

    // Rerender with different resetKeys to trigger reset
    rerender(
      <ErrorBoundary fallback={CustomFallback} resetKeys={[false]}>
        <TestComponent shouldThrow={false} />
      </ErrorBoundary>
    );

    // Verify component renders normally after reset
    await waitFor(() => {
      expect(screen.queryByTestId('custom-fallback')).not.toBeInTheDocument();
      expect(screen.getByText('No error now')).toBeInTheDocument();
    });
  });

  // 5. Different error types (render errors, async errors)
  test('catches errors thrown in event handlers', async () => {
    const user = userEvent.setup();

    render(
      <ErrorBoundary>
        <EventErrorComponent shouldThrow={true} />
      </ErrorBoundary>
    );

    // Trigger the error
    await user.click(screen.getByTestId('error-button'));

    // Verify error is caught and fallback is shown
    await waitFor(() => {
      expect(screen.getByText('Something went wrong.')).toBeInTheDocument();
      expect(screen.getByText('Test Error: Event handler error')).toBeInTheDocument();
    });
  });

  test('catches errors thrown in useEffect', async () => {
    act(() => {
      render(
        <ErrorBoundary>
          <EffectErrorComponent shouldThrow={true} />
        </ErrorBoundary>
      );
    });

    // Verify error is caught and fallback is shown
    await waitFor(() => {
      expect(screen.getByText('Something went wrong.')).toBeInTheDocument();
      expect(screen.getByText('Test Error: Effect error')).toBeInTheDocument();
    });
  });

  // 6. Custom fallback UI
  test('renders custom fallback UI when provided', () => {
    act(() => {
      render(
        <ErrorBoundary fallback={CustomFallback}>
          <ErrorThrowingComponent />
        </ErrorBoundary>
      );
    });

    // Verify custom fallback is used
    expect(screen.getByTestId('custom-fallback')).toBeInTheDocument();
    expect(screen.getByText('Custom Error UI')).toBeInTheDocument();
    expect(screen.getByTestId('error-message')).toHaveTextContent('Test Error: Render error');
    expect(screen.getByTestId('reset-button')).toBeInTheDocument();
  });

  // 7. Error boundary nesting
  test('inner error boundary catches errors without affecting outer boundary', () => {
    act(() => {
      render(
        <ErrorBoundary fallback={({ error }) => <div data-testid="outer-fallback">{error.message}</div>}>
          <div data-testid="outer-content">Outer content</div>
          <ErrorBoundary fallback={({ error }) => <div data-testid="inner-fallback">{error.message}</div>}>
            <ErrorThrowingComponent />
          </ErrorBoundary>
        </ErrorBoundary>
      );
    });

    // Inner error boundary should catch the error
    expect(screen.getByTestId('inner-fallback')).toBeInTheDocument();
    expect(screen.getByTestId('inner-fallback')).toHaveTextContent('Test Error: Render error');
    
    // Outer content should still be visible
    expect(screen.getByTestId('outer-content')).toBeInTheDocument();
    
    // Outer fallback should not be rendered
    expect(screen.queryByTestId('outer-fallback')).not.toBeInTheDocument();
  });

  test('outer error boundary catches errors when inner boundary is absent', () => {
    act(() => {
      render(
        <ErrorBoundary fallback={({ error }) => <div data-testid="outer-fallback">{error.message}</div>}>
          <ErrorThrowingComponent />
        </ErrorBoundary>
      );
    });

    // Outer error boundary should catch the error
    expect(screen.getByTestId('outer-fallback')).toBeInTheDocument();
    expect(screen.getByTestId('outer-fallback')).toHaveTextContent('Test Error: Render error');
  });

  // 8. Props handling during error states
  test('passes error and resetErrorBoundary to fallback component', () => {
    const onReset = jest.fn();
    
    act(() => {
      render(
        <ErrorBoundary 
          fallback={({ error, resetErrorBoundary }) => (
            <div>
              <span data-testid="error-type">{error.name}</span>
              <span data-testid="error-message">{error.message}</span>
              <button onClick={resetErrorBoundary} data-testid="reset-fn-button">Reset</button>
            </div>
          )}
          onReset={onReset}
        >
          <ErrorThrowingComponent />
        </ErrorBoundary>
      );
    });

    // Verify error props are passed correctly
    expect(screen.getByTestId('error-type')).toHaveTextContent('Error');
    expect(screen.getByTestId('error-message')).toHaveTextContent('Test Error: Render error');
    
    // Verify resetErrorBoundary function is passed and works
    fireEvent.click(screen.getByTestId('reset-fn-button'));
    expect(onReset).toHaveBeenCalledTimes(1);
  });

  test('onError callback is called when an error occurs', () => {
    const onError = jest.fn();
    
    act(() => {
      render(
        <ErrorBoundary onError={onError}>
          <ErrorThrowingComponent />
        </ErrorBoundary>
      );
    });

    // Verify onError was called with the error and component stack
    expect(onError).toHaveBeenCalledTimes(1);
    expect(onError.mock.calls[0][0]).toBeInstanceOf(Error);
    expect(onError.mock.calls[0][0].message).toBe('Test Error: Render error');
    expect(onError.mock.calls[0][1]).toHaveProperty('componentStack');
  });

  test('handles additional props passed to ErrorBoundary', () => {
    const testId = 'custom-error-boundary';
    const className = 'error-container';
    
    act(() => {
      render(
        <ErrorBoundary 
          data-testid={testId} 
          className={className}
          fallback={({ error }) => <div data-testid="fallback-with-props">{error.message}</div>}
        >
          <ErrorThrowingComponent />
        </ErrorBoundary>
      );
    });

    // Verify additional props are applied
    const fallback = screen.getByTestId('fallback-with-props');
    expect(fallback).toBeInTheDocument();
    expect(fallback.parentElement).toHaveAttribute('data-testid', testId);
    expect(fallback.parentElement).toHaveClass(className);
  });
});
