'use client';

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import { isAuthenticated, getUserInfo } from '@/lib/auth';

interface ProtectedRouteProps {
  children: React.ReactNode;
  roles?: string[]; // Optional array of roles allowed to access this route
}

export default function ProtectedRoute({ children, roles }: ProtectedRouteProps) {
  const router = useRouter();
  const [loading, setLoading] = useState(true);
  const [isAuthorized, setIsAuthorized] = useState(false);

  useEffect(() => {
    const checkAuth = async () => {
      setLoading(true);
      
      // Check if user is authenticated
      const authenticated = isAuthenticated();
      
      if (!authenticated) {
        router.push('/login');
        return;
      }

      // If roles are specified, check if user has required role
      if (roles && roles.length > 0) {
        const userInfo = getUserInfo();
        
        if (!userInfo || !userInfo.role) {
          router.push('/unauthorized');
          return;
        }
        
        const hasRequiredRole = roles.includes(userInfo.role);
        
        if (!hasRequiredRole) {
          router.push('/unauthorized');
          return;
        }
      }
      
      // User is authenticated and has required role (if specified)
      setIsAuthorized(true);
      setLoading(false);
    };

    checkAuth();
  }, [router, roles]);

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-gray-100">
        <div className="text-center">
          <svg className="animate-spin h-10 w-10 text-blue-500 mx-auto" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
          </svg>
          <p className="mt-3 text-gray-700">Verifying authentication...</p>
        </div>
      </div>
    );
  }

  return isAuthorized ? <>{children}</> : null;
}
