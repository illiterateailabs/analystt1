// API client for the Analyst's Augmentation Agent
import axios from 'axios';

// Base API URL - configurable via environment
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api/v1';

// Axios instance with default config
const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add request interceptor for auth token
apiClient.interceptors.request.use(
  (config) => {
    // Get token from local storage if available
    const token = localStorage.getItem('auth_token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);

// Auth Types
export interface LoginRequest {
  username: string;
  password: string;
}

export interface RegisterRequest {
  username: string;
  email: string;
  password: string;
  full_name?: string;
}

export interface AuthResponse {
  access_token: string;
  token_type: string;
  user: {
    id: string;
    username: string;
    email: string;
    full_name?: string;
    is_active: boolean;
    is_superuser: boolean;
  };
}

// Chat Types
export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: string;
}

export interface ChatSession {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
  messages: ChatMessage[];
}

// Graph Types
export interface GraphNode {
  id: string;
  labels: string[];
  properties: Record<string, any>;
}

export interface GraphRelationship {
  id: string;
  type: string;
  startNode: string;
  endNode: string;
  properties: Record<string, any>;
}

export interface GraphData {
  nodes: GraphNode[];
  relationships: GraphRelationship[];
}

export interface GraphQueryRequest {
  query: string;
  parameters?: Record<string, any>;
}

export interface GraphQueryResponse {
  data: GraphData;
  query: string;
  parameters?: Record<string, any>;
  execution_time_ms: number;
}

// Crew Types
export interface CrewRunRequest {
  crew_name: string;
  inputs?: Record<string, any>;
  options?: Record<string, any>;
}

export interface CrewRunResponse {
  success: boolean;
  result?: any;
  error?: string;
  task_id?: string;
  execution_time_ms?: number;
}

// Analysis Types
export interface AnalysisRequest {
  text: string;
  options?: Record<string, any>;
}

export interface AnalysisResponse {
  analysis: any;
  execution_time_ms: number;
}

// Prompt Management Types
export interface AgentListItem {
  agent_id: string;
  description?: string;
  has_custom_prompt: boolean;
}

export interface PromptResponse {
  agent_id: string;
  system_prompt: string;
  description?: string;
  metadata?: Record<string, any>;
  is_default: boolean;
}

export interface PromptUpdate {
  system_prompt: string;
  description?: string;
  metadata?: Record<string, any>;
}

// Auth API
export const login = async (data: LoginRequest): Promise<AuthResponse> => {
  const response = await apiClient.post('/auth/login', data);
  return response.data;
};

export const register = async (data: RegisterRequest): Promise<AuthResponse> => {
  const response = await apiClient.post('/auth/register', data);
  return response.data;
};

export const logout = async (): Promise<void> => {
  localStorage.removeItem('auth_token');
};

export const getCurrentUser = async (): Promise<AuthResponse['user']> => {
  const response = await apiClient.get('/auth/me');
  return response.data;
};

// Chat API
export const getChatSessions = async (): Promise<ChatSession[]> => {
  const response = await apiClient.get('/chat/sessions');
  return response.data;
};

export const getChatSession = async (sessionId: string): Promise<ChatSession> => {
  const response = await apiClient.get(`/chat/sessions/${sessionId}`);
  return response.data;
};

export const createChatSession = async (title: string): Promise<ChatSession> => {
  const response = await apiClient.post('/chat/sessions', { title });
  return response.data;
};

export const sendChatMessage = async (
  sessionId: string,
  content: string
): Promise<ChatMessage> => {
  const response = await apiClient.post(`/chat/sessions/${sessionId}/messages`, {
    content,
  });
  return response.data;
};

// Graph API
export const executeGraphQuery = async (
  data: GraphQueryRequest
): Promise<GraphQueryResponse> => {
  const response = await apiClient.post('/graph/query', data);
  return response.data;
};

export const getGraphSchema = async (): Promise<any> => {
  const response = await apiClient.get('/graph/schema');
  return response.data;
};

// Crew API
export const runCrew = async (data: CrewRunRequest): Promise<CrewRunResponse> => {
  const response = await apiClient.post('/crew/run', data);
  return response.data;
};

export const getCrewStatus = async (taskId: string): Promise<any> => {
  const response = await apiClient.get(`/crew/status/${taskId}`);
  return response.data;
};

export const getAvailableCrews = async (): Promise<string[]> => {
  const response = await apiClient.get('/crew/available');
  return response.data.crews;
};

// Analysis API
export const analyzeText = async (data: AnalysisRequest): Promise<AnalysisResponse> => {
  const response = await apiClient.post('/analysis/text', data);
  return response.data;
};

export const analyzeImage = async (
  image: File,
  options?: Record<string, any>
): Promise<AnalysisResponse> => {
  const formData = new FormData();
  formData.append('image', image);
  
  if (options) {
    formData.append('options', JSON.stringify(options));
  }
  
  const response = await apiClient.post('/analysis/image', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
  
  return response.data;
};

// Prompt Management API
export const listAgents = async (): Promise<{ agents: AgentListItem[] }> => {
  const response = await apiClient.get('/prompts');
  return response.data;
};

export const getAgentPrompt = async (agentId: string): Promise<PromptResponse> => {
  const response = await apiClient.get(`/prompts/${agentId}`);
  return response.data;
};

export const updateAgentPrompt = async (
  agentId: string,
  data: PromptUpdate
): Promise<PromptResponse> => {
  const response = await apiClient.put(`/prompts/${agentId}`, data);
  return response.data;
};

export const resetAgentPrompt = async (agentId: string): Promise<PromptResponse> => {
  const response = await apiClient.post(`/prompts/${agentId}/reset`);
  return response.data;
};

export default apiClient;
