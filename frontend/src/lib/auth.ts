'use client';

import { jwtDecode } from 'jwt-decode'; // You might need to install this: npm install jwt-decode
import { API_BASE_URL } from './constants'; // Assuming you have a constants file for API base URL

interface DecodedToken {
  sub: string; // Subject (e.g., user ID or email)
  role: string;
  exp: number; // Expiration time (Unix timestamp)
  iat: number; // Issued at time (Unix timestamp)
  // Add other fields your JWT might contain
}

const ACCESS_TOKEN_KEY = 'access_token';
const REFRESH_TOKEN_KEY = 'refresh_token';

// 1. Store/retrieve JWT tokens
export const setTokens = (accessToken: string, refreshToken: string) => {
  localStorage.setItem(ACCESS_TOKEN_KEY, accessToken);
  localStorage.setItem(REFRESH_TOKEN_KEY, refreshToken);
};

export const getAccessToken = (): string | null => {
  return localStorage.getItem(ACCESS_TOKEN_KEY);
};

export const getRefreshToken = (): string | null => {
  return localStorage.getItem(REFRESH_TOKEN_KEY);
};

// 2. Check if user is authenticated
export const isAuthenticated = (): boolean => {
  const token = getAccessToken();
  if (!token) {
    return false;
  }
  try {
    const decoded: DecodedToken = jwtDecode(token);
    const currentTime = Date.now() / 1000; // Convert to seconds
    return decoded.exp > currentTime; // Check if token is not expired
  } catch (error) {
    console.error('Error decoding access token:', error);
    return false;
  }
};

// 3. Get user info from token
export const getUserInfo = (): DecodedToken | null => {
  const token = getAccessToken();
  if (!token) {
    return null;
  }
  try {
    const decoded: DecodedToken = jwtDecode(token);
    return decoded;
  } catch (error) {
    console.error('Error decoding access token:', error);
    return null;
  }
};

// 4. Logout function
export const logout = () => {
  localStorage.removeItem(ACCESS_TOKEN_KEY);
  localStorage.removeItem(REFRESH_TOKEN_KEY);
  // Optionally redirect to login page
  window.location.href = '/login';
};

// 5. Refresh token logic
export const refreshAccessToken = async (): Promise<string | null> => {
  const refreshToken = getRefreshToken();
  if (!refreshToken) {
    console.warn('No refresh token available. User needs to re-authenticate.');
    logout(); // Force re-login if no refresh token
    return null;
  }

  try {
    const response = await fetch(`${API_BASE_URL}/auth/refresh`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ refresh_token: refreshToken }),
    });

    if (!response.ok) {
      const errorData = await response.json();
      console.error('Failed to refresh token:', errorData.error || response.statusText);
      logout(); // Logout if refresh fails
      return null;
    }

    const data = await response.json();
    setTokens(data.access_token, data.refresh_token);
    return data.access_token;
  } catch (error) {
    console.error('Network error during token refresh:', error);
    logout(); // Logout on network errors during refresh
    return null;
  }
};

// 6. Auth header helper for API calls
export const getAuthHeaders = async (): Promise<HeadersInit> => {
  let token = getAccessToken();

  // If token is expired or missing, try to refresh
  if (!token || !isAuthenticated()) {
    token = await refreshAccessToken();
  }

  if (token) {
    return {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${token}`,
    };
  } else {
    // If no token after refresh, return basic headers or throw error
    return {
      'Content-Type': 'application/json',
    };
  }
};

// Helper to check if token is about to expire (e.g., within 5 minutes)
export const isTokenAboutToExpire = (token: string, minutesThreshold: number = 5): boolean => {
  try {
    const decoded: DecodedToken = jwtDecode(token);
    const currentTime = Date.now() / 1000;
    return (decoded.exp - currentTime) < (minutesThreshold * 60);
  } catch (error) {
    return true; // Assume it's about to expire if decoding fails
  }
};
