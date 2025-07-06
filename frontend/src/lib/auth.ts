'use client';

import { jwtDecode } from 'jwt-decode';
import { API_BASE_URL } from './constants';

interface DecodedToken {
  sub: string; // Subject (e.g., user ID or email)
  role: string;
  exp: number; // Expiration time (Unix timestamp)
  iat: number; // Issued at time (Unix timestamp)
  // Add other fields your JWT might contain
}

interface UserInfo {
  user_id: string;
  username: string;
  is_superuser: boolean;
}

// Check if user is authenticated by verifying with the server
export const isAuthenticated = async (): Promise<boolean> => {
  try {
    const response = await fetch(`${API_BASE_URL}/auth/verify`, {
      method: 'GET',
      credentials: 'include', // Important: include cookies in the request
    });
    
    return response.ok;
  } catch (error) {
    console.error('Authentication verification error:', error);
    return false;
  }
};

// Get user info from server
export const getUserInfo = async (): Promise<UserInfo | null> => {
  try {
    const response = await fetch(`${API_BASE_URL}/auth/me`, {
      method: 'GET',
      credentials: 'include', // Important: include cookies in the request
    });
    
    if (!response.ok) {
      return null;
    }
    
    return await response.json();
  } catch (error) {
    console.error('Error fetching user info:', error);
    return null;
  }
};

// Logout function - calls server to clear cookies
export const logout = async (): Promise<void> => {
  try {
    await fetch(`${API_BASE_URL}/auth/logout`, {
      method: 'POST',
      credentials: 'include', // Important: include cookies in the request
    });
  } catch (error) {
    console.error('Logout error:', error);
  } finally {
    // Always redirect to login page, even if the server request fails
    window.location.href = '/login';
  }
};

// Refresh token logic - server handles token rotation via cookies
export const refreshAccessToken = async (): Promise<boolean> => {
  try {
    const response = await fetch(`${API_BASE_URL}/auth/refresh`, {
      method: 'POST',
      credentials: 'include', // Important: include cookies in the request
      headers: {
        'Content-Type': 'application/json',
      },
      // No need to send tokens in body - they're in the cookies
    });

    if (!response.ok) {
      console.error('Failed to refresh token:', response.statusText);
      await logout(); // Logout if refresh fails
      return false;
    }

    return true;
  } catch (error) {
    console.error('Network error during token refresh:', error);
    await logout(); // Logout on network errors during refresh
    return false;
  }
};

// Auth header helper for API calls - now just includes CSRF token if needed
export const getAuthHeaders = async (): Promise<HeadersInit> => {
  // With cookie auth, we only need Content-Type and CSRF token (if applicable)
  // The CSRF token would typically be read from a cookie that is not httpOnly
  const headers: HeadersInit = {
    'Content-Type': 'application/json',
  };
  
  // If your app uses CSRF protection, get the token from the cookie
  const csrfToken = document.cookie
    .split('; ')
    .find(row => row.startsWith('csrf_token='))
    ?.split('=')[1];
    
  if (csrfToken) {
    headers['X-CSRF-Token'] = csrfToken;
  }
  
  return headers;
};

// Helper to check if user session needs refresh (for proactive refreshing)
export const checkSessionStatus = async (): Promise<void> => {
  try {
    const response = await fetch(`${API_BASE_URL}/auth/status`, {
      method: 'GET',
      credentials: 'include',
    });
    
    if (response.status === 401) {
      // Session expired, try to refresh
      await refreshAccessToken();
    }
  } catch (error) {
    console.error('Session check error:', error);
  }
};
