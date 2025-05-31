export const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8000/api/v1';

// Frontend Route Paths
export const ROUTES = {
  HOME: '/',
  LOGIN: '/login',
  REGISTER: '/register',
  DASHBOARD: '/dashboard',
  REVIEWS: '/reviews',
  REVIEW_DETAIL: (taskId: string) => `/reviews/${taskId}`,
  UNAUTHORIZED: '/unauthorized',
  CHAT: '/chat',
  GRAPH: '/graph',
  PROMPTS: '/prompts',
  ANALYSIS: '/analysis',
};

// User Role Constants (mirroring backend roles)
export const USER_ROLES = {
  ADMIN: 'admin',
  ANALYST: 'analyst',
  COMPLIANCE: 'compliance',
  USER: 'user',
  AUDITOR: 'auditor',
};

// Task Status Constants
export const TASK_STATUS = {
  PENDING: 'pending',
  RUNNING: 'running',
  PAUSED: 'paused',
  COMPLETED: 'completed',
  FAILED: 'failed',
};

// Review Status Constants
export const REVIEW_STATUS = {
  PENDING: 'pending',
  APPROVED: 'approved',
  REJECTED: 'rejected',
};

// Other App Constants
export const APP_NAME = 'Analyst Agent';
export const APP_VERSION = '1.0.0';
export const JWT_EXPIRATION_THRESHOLD_MINUTES = 5; // How many minutes before expiry to try refreshing token
