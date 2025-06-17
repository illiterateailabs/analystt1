import React from 'react';
import { cn } from '@/lib/utils';

export type SpinnerSize = 'xs' | 'sm' | 'md' | 'lg' | 'xl';
export type SpinnerVariant = 'primary' | 'secondary' | 'success' | 'danger' | 'warning' | 'info';

export interface LoadingSpinnerProps {
  /**
   * Size of the spinner
   * @default 'md'
   */
  size?: SpinnerSize;
  
  /**
   * Color variant of the spinner
   * @default 'primary'
   */
  variant?: SpinnerVariant;
  
  /**
   * Whether to show a label next to the spinner
   * @default false
   */
  showLabel?: boolean;
  
  /**
   * Custom label text
   * @default 'Loading...'
   */
  label?: string;
  
  /**
   * Additional CSS classes
   */
  className?: string;
  
  /**
   * Whether the spinner is visible
   * @default true
   */
  isVisible?: boolean;
  
  /**
   * Accessibility label for screen readers
   * @default 'Loading content'
   */
  ariaLabel?: string;
}

/**
 * LoadingSpinner component
 * 
 * A reusable spinner component with different sizes and variants.
 * Includes accessibility features and Tailwind CSS animations.
 */
const LoadingSpinner: React.FC<LoadingSpinnerProps> = ({
  size = 'md',
  variant = 'primary',
  showLabel = false,
  label = 'Loading...',
  className,
  isVisible = true,
  ariaLabel = 'Loading content',
}) => {
  if (!isVisible) return null;
  
  // Size mappings
  const sizeClasses = {
    xs: 'h-3 w-3 border-[1.5px]',
    sm: 'h-4 w-4 border-2',
    md: 'h-6 w-6 border-2',
    lg: 'h-8 w-8 border-3',
    xl: 'h-12 w-12 border-4',
  };
  
  // Variant (color) mappings
  const variantClasses = {
    primary: 'border-primary border-t-transparent text-primary',
    secondary: 'border-gray-300 border-t-transparent text-gray-300',
    success: 'border-green-500 border-t-transparent text-green-500',
    danger: 'border-red-500 border-t-transparent text-red-500',
    warning: 'border-yellow-500 border-t-transparent text-yellow-500',
    info: 'border-blue-500 border-t-transparent text-blue-500',
  };
  
  return (
    <div 
      className={cn(
        'inline-flex items-center justify-center',
        showLabel ? 'gap-2' : '',
        className
      )}
      role="status"
      aria-live="polite"
      aria-label={ariaLabel}
    >
      <div
        className={cn(
          'animate-spin rounded-full',
          sizeClasses[size],
          variantClasses[variant]
        )}
      />
      
      {showLabel && (
        <span className={cn(
          'text-sm font-medium',
          {
            'text-xs': size === 'xs' || size === 'sm',
            'text-sm': size === 'md',
            'text-base': size === 'lg',
            'text-lg': size === 'xl',
          },
          `text-${variant === 'primary' ? 'primary' : variant}-700`
        )}>
          {label}
        </span>
      )}
      
      {/* Visually hidden text for screen readers */}
      <span className="sr-only">{ariaLabel}</span>
    </div>
  );
};

export default LoadingSpinner;
