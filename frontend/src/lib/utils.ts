import { type ClassValue, clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

/**
 * Combines multiple class names into a single string, with Tailwind CSS optimization.
 * Uses clsx for conditional class joining and twMerge to properly merge Tailwind classes.
 * 
 * @example
 * // Basic usage
 * cn('text-red-500', 'bg-blue-500')
 * // => 'text-red-500 bg-blue-500'
 * 
 * @example
 * // With conditionals
 * cn('text-lg', isActive && 'font-bold', isBig ? 'p-4' : 'p-2')
 * // => 'text-lg font-bold p-4' (if both isActive and isBig are true)
 * 
 * @example
 * // Merging conflicting Tailwind classes (last one wins)
 * cn('text-red-500', 'text-blue-500')
 * // => 'text-blue-500'
 */
export function cn(...inputs: ClassValue[]): string {
  return twMerge(clsx(inputs));
}

/**
 * Formats a date to a readable string
 * 
 * @param date - The date to format
 * @param options - Intl.DateTimeFormat options
 * @returns Formatted date string
 */
export function formatDate(
  date: Date | string | number,
  options: Intl.DateTimeFormatOptions = {
    year: 'numeric',
    month: 'long',
    day: 'numeric',
  }
): string {
  const d = typeof date === 'string' || typeof date === 'number'
    ? new Date(date)
    : date;
  
  return new Intl.DateTimeFormat('en-US', {
    ...options,
  }).format(d);
}

/**
 * Truncates a string to a specified length
 * 
 * @param str - The string to truncate
 * @param length - Maximum length before truncation
 * @param ending - String to append after truncation (default: '...')
 * @returns Truncated string
 */
export function truncate(str: string, length: number, ending: string = '...'): string {
  if (str.length <= length) return str;
  return str.substring(0, length - ending.length) + ending;
}

/**
 * Debounces a function call
 * 
 * @param fn - The function to debounce
 * @param delay - Delay in milliseconds
 * @returns Debounced function
 */
export function debounce<T extends (...args: any[]) => any>(
  fn: T,
  delay: number
): (...args: Parameters<T>) => void {
  let timeoutId: ReturnType<typeof setTimeout> | null = null;
  
  return function(...args: Parameters<T>): void {
    if (timeoutId) {
      clearTimeout(timeoutId);
    }
    
    timeoutId = setTimeout(() => {
      fn(...args);
    }, delay);
  };
}

/**
 * Generates a random string ID
 * 
 * @param length - Length of the ID (default: 8)
 * @returns Random string ID
 */
export function generateId(length: number = 8): string {
  return Math.random()
    .toString(36)
    .substring(2, 2 + length);
}

/**
 * Safely access nested object properties without throwing errors
 * 
 * @param obj - The object to access
 * @param path - Path to the property, using dot notation
 * @param defaultValue - Default value if path doesn't exist
 * @returns The value at the path or the default value
 */
export function getNestedValue<T = any>(
  obj: Record<string, any>,
  path: string,
  defaultValue: T | null = null
): T | null {
  const keys = path.split('.');
  let result = obj;
  
  for (const key of keys) {
    if (result === undefined || result === null) {
      return defaultValue;
    }
    result = result[key];
  }
  
  return (result as T) ?? defaultValue;
}

/**
 * Formats a number as currency
 * 
 * @param amount - The amount to format
 * @param currency - Currency code (default: 'USD')
 * @param locale - Locale for formatting (default: 'en-US')
 * @returns Formatted currency string
 */
export function formatCurrency(
  amount: number,
  currency: string = 'USD',
  locale: string = 'en-US'
): string {
  return new Intl.NumberFormat(locale, {
    style: 'currency',
    currency,
  }).format(amount);
}

/**
 * Capitalizes the first letter of a string
 * 
 * @param str - The string to capitalize
 * @returns Capitalized string
 */
export function capitalize(str: string): string {
  if (!str) return '';
  return str.charAt(0).toUpperCase() + str.slice(1);
}

/**
 * Checks if the code is running on the client-side
 */
export const isClient = typeof window !== 'undefined';

/**
 * Checks if the code is running on the server-side
 */
export const isServer = typeof window === 'undefined';
