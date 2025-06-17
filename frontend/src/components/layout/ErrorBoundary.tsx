import React, { Component, ErrorInfo, ReactNode } from 'react';
import { AlertTriangle, RefreshCw } from 'lucide-react';

interface ErrorBoundaryProps {
  children: ReactNode;
  fallback?: ReactNode;
  onError?: (error: Error, errorInfo: ErrorInfo) => void;
  resetKey?: any; // When this prop changes, the error boundary will reset
}

interface ErrorBoundaryState {
  hasError: boolean;
  error: Error | null;
  errorInfo: ErrorInfo | null;
}

/**
 * ErrorBoundary component catches JavaScript errors anywhere in its child component tree,
 * logs those errors, and displays a fallback UI instead of the component tree that crashed.
 * 
 * Usage:
 * ```tsx
 * <ErrorBoundary>
 *   <ComponentThatMightError />
 * </ErrorBoundary>
 * ```
 */
class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = {
      hasError: false,
      error: null,
      errorInfo: null
    };
  }

  static getDerivedStateFromError(error: Error): Partial<ErrorBoundaryState> {
    // Update state so the next render will show the fallback UI
    return {
      hasError: true,
      error
    };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo): void {
    // Log the error to an error reporting service
    console.error('Error caught by ErrorBoundary:', error, errorInfo);
    
    // If Sentry or other monitoring is available
    if (typeof window !== 'undefined' && window.Sentry) {
      window.Sentry.captureException(error);
    }
    
    // Call onError callback if provided
    if (this.props.onError) {
      this.props.onError(error, errorInfo);
    }
    
    this.setState({
      errorInfo
    });
  }

  componentDidUpdate(prevProps: ErrorBoundaryProps): void {
    // Reset the error boundary when resetKey changes
    if (this.props.resetKey !== prevProps.resetKey && this.state.hasError) {
      this.reset();
    }
  }

  reset = (): void => {
    this.setState({
      hasError: false,
      error: null,
      errorInfo: null
    });
  };

  render(): ReactNode {
    if (this.state.hasError) {
      // If a custom fallback is provided, use it
      if (this.props.fallback) {
        return this.props.fallback;
      }

      // Default fallback UI
      const isDevelopment = process.env.NODE_ENV === 'development';
      
      return (
        <div className="flex flex-col items-center justify-center p-6 rounded-lg border border-red-200 bg-red-50 text-red-800 my-4 max-w-3xl mx-auto">
          <div className="flex items-center mb-4">
            <AlertTriangle className="h-8 w-8 mr-2 text-red-600" />
            <h2 className="text-xl font-semibold">Something went wrong</h2>
          </div>
          
          <p className="mb-4 text-center">
            We encountered an error while rendering this component. Our team has been notified.
          </p>
          
          <button
            onClick={this.reset}
            className="flex items-center px-4 py-2 bg-red-600 text-white rounded-md hover:bg-red-700 transition-colors"
          >
            <RefreshCw className="h-4 w-4 mr-2" />
            Try again
          </button>
          
          {/* Show error details in development mode */}
          {isDevelopment && this.state.error && (
            <div className="mt-6 p-4 bg-gray-800 text-white rounded-md w-full overflow-auto max-h-64">
              <p className="font-bold mb-2">Error Details (visible in development only):</p>
              <p className="mb-2">{this.state.error.toString()}</p>
              
              {this.state.errorInfo && (
                <details className="mt-2">
                  <summary className="cursor-pointer mb-2">Component Stack</summary>
                  <pre className="whitespace-pre-wrap text-xs">
                    {this.state.errorInfo.componentStack}
                  </pre>
                </details>
              )}
            </div>
          )}
        </div>
      );
    }

    return this.props.children;
  }
}

// Add a type declaration for Sentry to avoid TypeScript errors
declare global {
  interface Window {
    Sentry?: {
      captureException: (error: Error) => void;
    };
  }
}

export default ErrorBoundary;
