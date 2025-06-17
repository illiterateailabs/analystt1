import { useState, useCallback } from 'react';
import toast, { Toast, ToastOptions } from 'react-hot-toast';

// Toast variant types
export type ToastVariant = 'default' | 'success' | 'error' | 'warning' | 'info' | 'destructive';

// Toast position options
export type ToastPosition = 
  | 'top-left'
  | 'top-center'
  | 'top-right'
  | 'bottom-left'
  | 'bottom-center'
  | 'bottom-right';

// Toast notification props
export interface ToastProps {
  /**
   * The title of the toast
   */
  title?: string;
  
  /**
   * The description/message of the toast
   */
  description: string;
  
  /**
   * The visual style variant of the toast
   * @default 'default'
   */
  variant?: ToastVariant;
  
  /**
   * Duration in milliseconds the toast should remain visible
   * @default 3000
   */
  duration?: number;
  
  /**
   * Position of the toast on the screen
   * @default 'bottom-right'
   */
  position?: ToastPosition;
  
  /**
   * Custom icon to display in the toast
   */
  icon?: React.ReactNode;
  
  /**
   * Whether the toast should be dismissible with a close button
   * @default true
   */
  dismissible?: boolean;
  
  /**
   * Additional options to pass to react-hot-toast
   */
  options?: ToastOptions;
}

/**
 * Custom hook for displaying toast notifications
 * 
 * @example
 * ```tsx
 * const { toast } = useToast();
 * 
 * // Show a success toast
 * toast({
 *   title: "Success!",
 *   description: "Your changes have been saved.",
 *   variant: "success"
 * });
 * 
 * // Show an error toast
 * toast({
 *   title: "Error",
 *   description: "Failed to save changes.",
 *   variant: "error"
 * });
 * ```
 */
export const useToast = () => {
  // Keep track of displayed toasts to prevent duplicates
  const [toasts, setToasts] = useState<string[]>([]);

  // Map variant to toast styling
  const getToastStyle = (variant: ToastVariant): React.CSSProperties => {
    const baseStyle: React.CSSProperties = {
      padding: '12px 16px',
      borderRadius: '6px',
      display: 'flex',
      alignItems: 'flex-start',
      gap: '8px',
      maxWidth: '420px',
      boxShadow: '0 4px 12px rgba(0, 0, 0, 0.1)',
    };

    switch (variant) {
      case 'success':
        return {
          ...baseStyle,
          backgroundColor: '#ecfdf5',
          border: '1px solid #d1fae5',
          color: '#065f46',
        };
      case 'error':
      case 'destructive':
        return {
          ...baseStyle,
          backgroundColor: '#fef2f2',
          border: '1px solid #fee2e2',
          color: '#b91c1c',
        };
      case 'warning':
        return {
          ...baseStyle,
          backgroundColor: '#fffbeb',
          border: '1px solid #fef3c7',
          color: '#92400e',
        };
      case 'info':
        return {
          ...baseStyle,
          backgroundColor: '#eff6ff',
          border: '1px solid #dbeafe',
          color: '#1e40af',
        };
      default:
        return {
          ...baseStyle,
          backgroundColor: '#f9fafb',
          border: '1px solid #f3f4f6',
          color: '#1f2937',
        };
    }
  };

  // Convert position string to react-hot-toast position
  const getToastPosition = (position: ToastPosition): ToastOptions['position'] => {
    return position as ToastOptions['position'];
  };

  // Main toast function
  const showToast = useCallback(
    ({
      title,
      description,
      variant = 'default',
      duration = 3000,
      position = 'bottom-right',
      icon,
      dismissible = true,
      options = {},
    }: ToastProps) => {
      // Create a unique ID for this toast based on content
      const id = `${variant}:${title}:${description}`;

      // Prevent duplicate toasts
      if (toasts.includes(id)) {
        return;
      }

      // Add to tracked toasts
      setToasts((prev) => [...prev, id]);

      // Create the toast content
      const content = (
        <div style={{ display: 'flex', flexDirection: 'column' }}>
          {title && <div style={{ fontWeight: 600, marginBottom: '4px' }}>{title}</div>}
          <div>{description}</div>
        </div>
      );

      // Show the toast
      const toastId = toast(content, {
        duration,
        position: getToastPosition(position),
        style: getToastStyle(variant),
        icon: icon,
        dismissible,
        ...options,
        // When toast is dismissed, remove from tracked toasts
        onClose: () => {
          setToasts((prev) => prev.filter((t) => t !== id));
          options.onClose?.(toastId);
        },
      });

      return toastId;
    },
    [toasts]
  );

  // Convenience methods for different toast types
  const success = useCallback(
    (props: Omit<ToastProps, 'variant'>) => showToast({ ...props, variant: 'success' }),
    [showToast]
  );

  const error = useCallback(
    (props: Omit<ToastProps, 'variant'>) => showToast({ ...props, variant: 'error' }),
    [showToast]
  );

  const warning = useCallback(
    (props: Omit<ToastProps, 'variant'>) => showToast({ ...props, variant: 'warning' }),
    [showToast]
  );

  const info = useCallback(
    (props: Omit<ToastProps, 'variant'>) => showToast({ ...props, variant: 'info' }),
    [showToast]
  );

  const destructive = useCallback(
    (props: Omit<ToastProps, 'variant'>) => showToast({ ...props, variant: 'destructive' }),
    [showToast]
  );

  return {
    toast: showToast,
    success,
    error,
    warning,
    info,
    destructive,
    dismiss: toast.dismiss,
    // Export the original toast function for advanced use cases
    native: toast,
  };
};

export default useToast;
