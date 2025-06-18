import React, {
  createContext,
  useContext,
  useState,
  useEffect,
  useCallback,
  ReactNode,
} from 'react';
import { useRouter } from 'next/router';
import { authAPI, LoginRequest, RegisterRequest, UserResponse } from '@/lib/api';
import { useToast } from '@/hooks/useToast';

interface AuthContextType {
  user: UserResponse | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  login: (credentials: LoginRequest) => Promise<void>;
  register: (userData: RegisterRequest) => Promise<void>;
  logout: () => Promise<void>;
  refreshUser: () => Promise<void>;
  checkAuthStatus: () => Promise<boolean>;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

interface AuthProviderProps {
  children: ReactNode;
}

export const AuthProvider: React.FC<AuthProviderProps> = ({ children }) => {
  const [user, setUser] = useState<UserResponse | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const router = useRouter();
  const { toast } = useToast();

  // Function to fetch current user data
  const fetchCurrentUser = useCallback(async () => {
    try {
      const currentUser = await authAPI.getCurrentUser();
      setUser(currentUser);
      return currentUser;
    } catch (error) {
      // If getCurrentUser fails, it means no valid session/token
      setUser(null);
      return null;
    } finally {
      setIsLoading(false);
    }
  }, []);

  // Check authentication status
  const checkAuthStatus = useCallback(async (): Promise<boolean> => {
    try {
      setIsLoading(true);
      const currentUser = await fetchCurrentUser();
      return !!currentUser;
    } catch (error) {
      return false;
    } finally {
      setIsLoading(false);
    }
  }, [fetchCurrentUser]);

  // On mount, try to fetch the current user
  useEffect(() => {
    const initAuth = async () => {
      try {
        // Try to get current user (uses httpOnly cookies automatically)
        await fetchCurrentUser();
      } catch (error) {
        // If that fails, try to refresh the token
        try {
          await authAPI.refresh();
          // If refresh succeeds, try getting user again
          await fetchCurrentUser();
        } catch (refreshError) {
          // Both getCurrentUser and refresh failed - user is not authenticated
          setUser(null);
        }
      } finally {
        setIsLoading(false);
      }
    };

    initAuth();
  }, [fetchCurrentUser]);

  // Login function
  const login = useCallback(
    async (credentials: LoginRequest) => {
      setIsLoading(true);
      try {
        const response = await authAPI.login(credentials);
        setUser(response.user);
        
        toast({
          title: 'Login Successful',
          description: `Welcome back, ${response.user.username}!`,
          variant: 'success',
        });
        
        router.push('/dashboard');
      } catch (error: any) {
        const errorMessage = error.response?.data?.detail || 'Login failed. Please check your credentials.';
        
        toast({
          title: 'Login Error',
          description: errorMessage,
          variant: 'destructive',
        });
        
        setUser(null);
        throw new Error(errorMessage);
      } finally {
        setIsLoading(false);
      }
    },
    [router, toast]
  );

  // Register function
  const register = useCallback(
    async (userData: RegisterRequest) => {
      setIsLoading(true);
      try {
        const response = await authAPI.register(userData);
        setUser(response.user);
        
        toast({
          title: 'Registration Successful',
          description: `Welcome, ${response.user.username}!`,
          variant: 'success',
        });
        
        router.push('/dashboard');
      } catch (error: any) {
        const errorMessage = error.response?.data?.detail || 'Registration failed. Please try again.';
        
        toast({
          title: 'Registration Error',
          description: errorMessage,
          variant: 'destructive',
        });
        
        setUser(null);
        throw new Error(errorMessage);
      } finally {
        setIsLoading(false);
      }
    },
    [router, toast]
  );

  // Logout function
  const logout = useCallback(async () => {
    setIsLoading(true);
    try {
      await authAPI.logout();
      setUser(null);
      
      toast({
        title: 'Logged Out',
        description: 'You have been successfully logged out.',
        variant: 'info',
      });
      
      router.push('/login');
    } catch (error: any) {
      const errorMessage = error.response?.data?.detail || 'Logout failed. Please try again.';
      
      toast({
        title: 'Logout Error',
        description: errorMessage,
        variant: 'destructive',
      });
      
      // Even if logout fails on backend, clear local state for UX
      setUser(null);
      router.push('/login');
    } finally {
      setIsLoading(false);
    }
  }, [router, toast]);

  // Refresh user data
  const refreshUser = useCallback(async () => {
    setIsLoading(true);
    try {
      const refreshedUser = await fetchCurrentUser();
      if (!refreshedUser) {
        // If refresh fails, it means the session is truly gone
        // Try to refresh the token
        try {
          const refreshResponse = await authAPI.refresh();
          setUser(refreshResponse.user);
        } catch (refreshError) {
          // Refresh token is invalid or expired
          router.push('/login');
        }
      }
    } finally {
      setIsLoading(false);
    }
  }, [fetchCurrentUser, router]);

  // Context value
  const value = React.useMemo(
    () => ({
      user,
      isAuthenticated: !!user,
      isLoading,
      login,
      register,
      logout,
      refreshUser,
      checkAuthStatus,
    }),
    [user, isLoading, login, register, logout, refreshUser, checkAuthStatus]
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
};

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};

export default useAuth;
