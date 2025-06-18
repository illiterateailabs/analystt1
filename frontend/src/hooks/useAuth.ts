import { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import { useRouter } from 'next/router';
import { api } from '@/lib/api';
import { useToast } from '@/hooks/useToast';

// User type definition
export interface User {
  id: string;
  username: string;
  email: string;
  role: string;
  createdAt?: string;
}

// Auth context interface
interface AuthContextType {
  user: User | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  login: (email: string, password: string, rememberMe?: boolean) => Promise<void>;
  register: (username: string, email: string, password: string) => Promise<void>;
  logout: () => Promise<void>;
  refreshToken: () => Promise<boolean>;
}

// Create the auth context
const AuthContext = createContext<AuthContextType | undefined>(undefined);

// Provider component
interface AuthProviderProps {
  children: ReactNode;
}

export const AuthProvider = ({ children }: AuthProviderProps) => {
  const [user, setUser] = useState<User | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const router = useRouter();
  const { toast } = useToast();

  // Check if user is authenticated
  const isAuthenticated = !!user;

  // Fetch current user data
  const fetchCurrentUser = async (): Promise<void> => {
    try {
      setIsLoading(true);
      const response = await api.get('/auth/me');
      setUser(response.data);
    } catch (error) {
      // Silent fail on initial load
      setUser(null);
    } finally {
      setIsLoading(false);
    }
  };

  // Refresh token
  const refreshToken = async (): Promise<boolean> => {
    try {
      await api.post('/auth/refresh');
      return true;
    } catch (error) {
      return false;
    }
  };

  // Login function
  const login = async (email: string, password: string, rememberMe = false): Promise<void> => {
    try {
      setIsLoading(true);
      const response = await api.post('/auth/login', { email, password, rememberMe });
      setUser(response.data.user);
      
      toast({
        description: 'Login successful',
        variant: 'success',
      });
      
      router.push('/dashboard');
    } catch (error: any) {
      const message = error.response?.data?.message || 'Login failed. Please check your credentials.';
      toast({
        description: message,
        variant: 'destructive',
      });
      throw error;
    } finally {
      setIsLoading(false);
    }
  };

  // Register function
  const register = async (username: string, email: string, password: string): Promise<void> => {
    try {
      setIsLoading(true);
      await api.post('/auth/register', { username, email, password });
      
      toast({
        description: 'Registration successful. Please log in.',
        variant: 'success',
      });
      
      router.push('/login');
    } catch (error: any) {
      const message = error.response?.data?.message || 'Registration failed. Please try again.';
      toast({
        description: message,
        variant: 'destructive',
      });
      throw error;
    } finally {
      setIsLoading(false);
    }
  };

  // Logout function
  const logout = async (): Promise<void> => {
    try {
      await api.post('/auth/logout');
      setUser(null);
      
      toast({
        description: 'Logged out successfully',
        variant: 'info',
      });
      
      router.push('/login');
    } catch (error) {
      toast({
        description: 'Error logging out',
        variant: 'destructive',
      });
    }
  };

  // Check authentication status on mount and setup refresh interval
  useEffect(() => {
    // Initial auth check
    fetchCurrentUser();

    // Set up token refresh interval (every 14 minutes)
    // This helps keep the session alive before the typical 15-minute JWT expiry
    const refreshInterval = setInterval(() => {
      if (isAuthenticated) {
        refreshToken();
      }
    }, 14 * 60 * 1000);

    return () => clearInterval(refreshInterval);
  }, []);

  // Context value
  const value = {
    user,
    isAuthenticated,
    isLoading,
    login,
    register,
    logout,
    refreshToken,
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
};

// Hook for using the auth context
export const useAuth = (): AuthContextType => {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};

export default useAuth;
