// API client for the Analyst's Augmentation Agent
import axios from 'axios';

// Base API URL - configurable via environment
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api/v1';

// -------------------------------------------------------------
// Cookie-based Axios instance (httpOnly auth, CSRF protection)
// -------------------------------------------------------------

/**
 * Extract CSRF token that backend sets as a cookie (e.g. `csrf_token=<uuid>`).
 */
const getCsrfToken = (): string | null => {
  if (typeof document === 'undefined') return null;
  const match = document.cookie.match(/(?:^|;\s*)csrf_token=([^;]+)/);
  return match ? decodeURIComponent(match[1]) : null;
};

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: { 'Content-Type': 'application/json' },
  withCredentials: true, // always send cookies
});

// Attach CSRF token & ensure credentials on every request
apiClient.interceptors.request.use(
  (config) => {
    // Ensure cookies are sent
    config.withCredentials = true;

    // Include CSRF token on state-changing requests
    const method = (config.method || 'get').toUpperCase();
    if (['POST', 'PUT', 'PATCH', 'DELETE'].includes(method)) {
      const csrf = getCsrfToken();
      if (csrf) {
        config.headers['X-CSRF-Token'] = csrf;
      }
    }
    return config;
  },
  (error) => Promise.reject(error)
);

// Automatically try to refresh token on 401 responses
let isRefreshing = false;
let failedQueue: Array<{
  resolve: (value?: unknown) => void;
  reject: (reason?: any) => void;
}> = [];

const processQueue = (error: any, tokenRefreshed: boolean) => {
  failedQueue.forEach((prom) => {
    if (tokenRefreshed) prom.resolve();
    else prom.reject(error);
  });
  failedQueue = [];
};

apiClient.interceptors.response.use(
  (res) => res,
  async (error) => {
    const originalRequest = error.config;

    // Avoid infinite loop
    if (error.response?.status === 401 && !originalRequest._retry) {
      if (isRefreshing) {
        return new Promise((resolve, reject) => {
          failedQueue.push({ resolve, reject });
        })
          .then(() => apiClient(originalRequest))
          .catch((err) => Promise.reject(err));
      }

      originalRequest._retry = true;
      isRefreshing = true;

      try {
        await apiClient.post(
          '/auth/refresh',
          {},
          { headers: { 'X-CSRF-Token': getCsrfToken() || '' } }
        );
        processQueue(null, true);
        return apiClient(originalRequest);
      } catch (refreshErr) {
        processQueue(refreshErr, false);
        return Promise.reject(refreshErr);
      } finally {
        isRefreshing = false;
      }
    }

    return Promise.reject(error);
  }
);

// Error handling helper
export const handleAPIError = (error: any) => {
  let message = 'An unexpected error occurred';
  let status = 500;

  if (error.response) {
    // Server responded with error
    status = error.response.status;
    message = error.response.data?.detail || error.response.statusText;
  } else if (error.request) {
    // Request made but no response
    message = 'No response from server. Please check your connection.';
  } else {
    // Request setup error
    message = error.message;
  }

  return { status, message };
};

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
export interface CrewRequest {
  crew_name: string;
  inputs: Record<string, any>;
  async_execution?: boolean;
}

export interface CrewResponse {
  success: boolean;
  crew_name: string;
  task_id?: string;
  result?: any;
  error?: string;
}

export interface AgentInfo {
  id: string;
  role: string;
  goal: string;
  tools: string[];
  backstory?: string;
}

export interface CrewInfo {
  name: string;
  process_type: string;
  manager?: string;
  agents: string[];
  description?: string;
}

export enum TaskState {
  PENDING = "pending",
  RUNNING = "running",
  PAUSED = "paused",
  COMPLETED = "completed",
  FAILED = "failed"
}

export enum ReviewStatus {
  PENDING = "pending",
  APPROVED = "approved",
  REJECTED = "rejected"
}

export interface ReviewRequest {
  findings: string;
  risk_level: string;
  regulatory_implications: string[];
  details?: Record<string, any>;
}

export interface ReviewResponse {
  status: ReviewStatus;
  reviewer: string;
  comments?: string;
}

export interface ResumeRequest {
  status: ReviewStatus;
  reviewer: string;
  comments?: string;
}

// Analysis Types
export interface AnalysisRequest { // For POST /analysis/text or /analysis/image
  text: string;
  options?: Record<string, any>;
}

export interface AnalysisResponse { // For POST /analysis/text or /analysis/image
  analysis: any;
  execution_time_ms: number;
}

// Specific Node/Edge types for Analysis Results (GET /analysis/{taskId})
export interface AnalysisResultNode {
  id: string;
  label: string;
  type?: string;
  properties?: Record<string, any>;
  risk_score?: number;
  size?: number;
  color?: string;
  [key: string]: any; // Allow other properties
}

export interface AnalysisResultEdge {
  from: string;
  to: string;
  label?: string;
  properties?: Record<string, any>;
  weight?: number;
  color?: string;
  [key: string]: any; // Allow other properties
}

export interface AnalysisResultGraphData {
  nodes: AnalysisResultNode[];
  edges: AnalysisResultEdge[];
}

export interface GeneratedVisualization {
  filename: string;
  content: string; // base64 encoded image data
  type: 'image/png' | 'image/jpeg' | 'image/svg+xml' | 'text/html';
}

export interface AnalysisResultsResponse { // For GET /analysis/{taskId}
  task_id: string;
  status: string;
  title?: string;
  executive_summary?: string;
  risk_score?: number;
  confidence?: number;
  detailed_findings?: string;
  graph_data?: AnalysisResultGraphData;
  visualizations?: GeneratedVisualization[];
  recommendations?: string[];
  code_generated?: string;
  execution_details?: any;
  error?: string;
  crew_name?: string;
  crew_inputs?: Record<string, any>;
  crew_result?: any;
}

// ------- Sim API Types (Balances & Activity) -------

export interface SimTokenMetadata {
  symbol: string;
  name?: string;
  decimals: number;
  logo?: string;
  url?: string;
}

export interface SimTokenBalance {
  address: string;
  amount: string;
  chain: string;
  chain_id: number;
  decimals: number;
  symbol: string;
  price_usd?: number;
  value_usd?: number;
  token_metadata?: SimTokenMetadata;
  low_liquidity?: boolean;
  pool_size?: number;
}

export interface SimBalancesResponse {
  wallet_address: string;
  balances: SimTokenBalance[];
  next_offset?: string;
  request_time?: string;
  response_time?: string;
}

export interface SimFunctionParameter {
  name: string;
  type: string;
  value: any;
}

export interface SimFunctionInfo {
  name: string;
  signature?: string;
  parameters?: SimFunctionParameter[];
}

export interface SimActivityItem {
  id?: string;
  type: 'send' | 'receive' | 'mint' | 'burn' | 'swap' | 'approve' | 'call';
  chain: string;
  chain_id: number;
  block_number: number;
  block_time: string;
  transaction_hash: string;
  from_address?: string;
  to_address?: string;
  asset_type?: string;
  amount?: string;
  value_usd?: number;
  token_address?: string;
  token_id?: string;
  token_metadata?: SimTokenMetadata;
  function?: SimFunctionInfo;
}

export interface SimActivityResponse {
  wallet_address: string;
  activity: SimActivityItem[];
  next_offset?: string;
  request_time?: string;
  response_time?: string;
}

// ------- New Sim API Types (Collectibles / TokenInfo / Holders / Risk) -------

export interface SimCollectible {
  contract_address: string;
  token_id: string;
  token_standard: 'ERC721' | 'ERC1155' | 'UNKNOWN';
  chain: string;
  chain_id: number;
  balance?: string;
  name?: string;
  collection_name?: string;
  image_url?: string;
}

export interface SimCollectiblesResponse {
  wallet_address: string;
  entries: SimCollectible[];
  next_offset?: string;
  request_time?: string;
  response_time?: string;
}

export interface SimTokenInfo {
  token_address: string;
  chain_id: number;
  chain: string;
  symbol: string;
  decimals: number;
  price_usd?: number;
  pool_size_usd?: number;
  pool_type?: string;
  total_supply?: string;
}

export interface SimTokenInfoResponse {
  entries: SimTokenInfo[];
  request_time?: string;
  response_time?: string;
}

export interface SimTokenHolder {
  wallet_address: string;
  balance: string;
  balance_percentage: number;
  value_usd?: number;
}

export interface SimTokenHoldersResponse {
  token_address: string;
  chain_id: string;
  holders: SimTokenHolder[];
  next_offset?: string;
  request_time?: string;
  response_time?: string;
}

export interface SimRiskScore {
  wallet_address: string;
  risk_score: number;
  risk_level: 'LOW' | 'MEDIUM' | 'HIGH';
  risk_factors: string[];
  summary: Record<string, any>;
}

// ------- Transaction-Flow & Cross-Chain Types -------

export interface TransactionFlowMetrics {
  total_transactions: number;
  total_value_usd: number;
  unique_addresses: number;
  unique_chains: number;
  average_transaction_value_usd: number;
  max_transaction_value_usd: number;
  time_span_hours: number;
  transaction_density: number;
  value_density_usd: number;
  graph_density: number;
}

export interface TransactionFlowAnalysis {
  nodes: any[];
  edges: any[];
  patterns: any[];
  metrics: TransactionFlowMetrics;
  risk_score: number;
  risk_factors: string[];
}

export interface CrossChainIdentityAnalysis {
  wallets: any[];
  cross_chain_movements: any[];
  identity_clusters: any[];
  bridge_usage: any[];
  overall_risk_score: number;
  overall_risk_factors: string[];
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
export const authAPI = {
  login: async (data: LoginRequest): Promise<AuthResponse> => {
    const response = await apiClient.post('/auth/login', data);
    return response.data;
  },
  
  register: async (data: RegisterRequest): Promise<AuthResponse> => {
    const response = await apiClient.post('/auth/register', data);
    return response.data;
  },
  
  logout: async (): Promise<void> => {
    localStorage.removeItem('auth_token');
  },
  
  getCurrentUser: async (): Promise<AuthResponse['user']> => {
    const response = await apiClient.get('/auth/me');
    return response.data;
  }
};

// Chat API
export const chatAPI = {
  getChatSessions: async (): Promise<ChatSession[]> => {
    const response = await apiClient.get('/chat/sessions');
    return response.data;
  },
  
  getChatSession: async (sessionId: string): Promise<ChatSession> => {
    const response = await apiClient.get(`/chat/sessions/${sessionId}`);
    return response.data;
  },
  
  createChatSession: async (title: string): Promise<ChatSession> => {
    const response = await apiClient.post('/chat/sessions', { title });
    return response.data;
  },
  
  sendChatMessage: async (
    sessionId: string,
    content: string
  ): Promise<ChatMessage> => {
    const response = await apiClient.post(`/chat/sessions/${sessionId}/messages`, {
      content,
    });
    return response.data;
  },
  
  sendMessage: async (message: string, includeGraphData: boolean = false): Promise<any> => {
    const response = await apiClient.post('/chat/message', {
      message,
      include_graph_data: includeGraphData
    });
    return response;
  },
  
  analyzeImage: async (file: File, prompt?: string): Promise<any> => {
    const formData = new FormData();
    formData.append('file', file);
    
    if (prompt) {
      formData.append('request', JSON.stringify({ prompt }));
    }
    
    const response = await apiClient.post('/chat/analyze-image', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    
    return response;
  }
};

// Graph API
export const graphAPI = {
  executeCypher: async (query: string, parameters?: Record<string, any>): Promise<any> => {
    const response = await apiClient.post('/graph/query', {
      query,
      parameters
    });
    return response;
  },
  
  getSchema: async (): Promise<any> => {
    const response = await apiClient.get('/graph/schema');
    return response;
  },
  
  naturalLanguageQuery: async (question: string): Promise<any> => {
    const response = await apiClient.post('/graph/nlq', {
      question
    });
    return response;
  },
  
  calculateCentrality: async (algorithm: string): Promise<any> => {
    const response = await apiClient.post('/graph/centrality', {
      algorithm
    });
    return response;
  }
};

// Crew API
export const crewAPI = {
  runCrew: async (request: CrewRequest): Promise<any> => {
    const response = await apiClient.post('/crew/run', request);
    return response;
  },
  
  getStatus: async (taskId: string): Promise<any> => {
    const response = await apiClient.get(`/crew/status/${taskId}`);
    return response;
  },
  
  listCrews: async (): Promise<any> => {
    const response = await apiClient.get('/crew/crews');
    return response;
  },
  
  getCrewDetails: async (crewName: string): Promise<any> => {
    const response = await apiClient.get(`/crew/crews/${crewName}`);
    return response;
  },
  
  listAgents: async (crewName?: string): Promise<any> => {
    const url = crewName ? `/crew/agents?crew_name=${crewName}` : '/crew/agents';
    const response = await apiClient.get(url);
    return response;
  },
  
  getAgentDetails: async (agentId: string): Promise<any> => {
    const response = await apiClient.get(`/crew/agents/${agentId}`);
    return response;
  },
  
  pauseTask: async (taskId: string, reviewRequest: ReviewRequest): Promise<any> => {
    const response = await apiClient.post(`/crew/pause/${taskId}`, reviewRequest);
    return response;
  },
  
  resumeTask: async (taskId: string, resumeRequest: ResumeRequest): Promise<any> => {
    const response = await apiClient.post(`/crew/resume/${taskId}`, resumeRequest);
    return response;
  },
  
  getReviewDetails: async (taskId: string): Promise<any> => {
    const response = await apiClient.get(`/crew/review/${taskId}`);
    return response;
  }
};

// Analysis API
export const analysisAPI = {
  analyzeText: async (data: AnalysisRequest): Promise<AnalysisResponse> => {
    const response = await apiClient.post('/analysis/text', data);
    return response.data;
  },
  
  analyzeImage: async (
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
  },

  fetchAnalysisResults: async (taskId: string): Promise<AnalysisResultsResponse> => {
    const response = await apiClient.get(`/analysis/${taskId}`);
    return response.data;
  }

  // ---------- Sim API Integration ----------
  ,
  /**
   * Fetch token balances for a wallet via Sim API proxy
   */
  getSimBalances: async (
    wallet: string,
    limit: number = 100,
    chainIds: string = 'all',
    metadata: string = 'url,logo'
  ): Promise<SimBalancesResponse> => {
    const response = await apiClient.get(`/analysis/sim/balances/${wallet}`, {
      params: { limit, chain_ids: chainIds, metadata },
    });
    return response.data;
  },

  /**
   * Fetch transaction activity for a wallet via Sim API proxy
   */
  getSimActivity: async (
    wallet: string,
    limit: number = 25,
    offset?: string
  ): Promise<SimActivityResponse> => {
    const response = await apiClient.get(`/analysis/sim/activity/${wallet}`, {
      params: { limit, offset },
    });
    return response.data;
  }

  ,
  /**
   * Fetch NFT collectibles owned by a wallet
   */
  getSimCollectibles: async (
    wallet: string,
    limit: number = 50,
    offset?: string,
    chainIds?: string
  ): Promise<SimCollectiblesResponse> => {
    const response = await apiClient.get(`/analysis/sim/collectibles/${wallet}`, {
      params: { limit, offset, chain_ids: chainIds },
    });
    return response.data;
  },

  /**
   * Fetch token information (price, liquidity) for a token address
   */
  getSimTokenInfo: async (
    tokenAddress: string,
    chainIds: string
  ): Promise<SimTokenInfoResponse> => {
    const response = await apiClient.get(
      `/analysis/sim/token-info/${tokenAddress}`,
      { params: { chain_ids: chainIds } }
    );
    return response.data;
  },

  /**
   * Fetch token holder distribution
   */
  getSimTokenHolders: async (
    chainId: string,
    tokenAddress: string,
    limit: number = 100,
    offset?: string
  ): Promise<SimTokenHoldersResponse> => {
    const response = await apiClient.get(
      `/analysis/sim/token-holders/${chainId}/${tokenAddress}`,
      { params: { limit, offset } }
    );
    return response.data;
  },

  /**
   * Fetch wallet risk score (Sim heuristic)
   */
  getSimRiskScore: async (wallet: string): Promise<SimRiskScore> => {
    const response = await apiClient.get(`/analysis/sim/risk-score/${wallet}`);
    return response.data;
  },

  /**
   * Transaction-flow analysis for one or more wallets
   */
  analyzeTransactionFlow: async (
    request: Record<string, any>
  ): Promise<TransactionFlowAnalysis> => {
    const response = await apiClient.post(
      '/analysis/transaction-flow/analyze',
      request
    );
    return response.data;
  },

  /**
   * Retrieve flow metrics for a wallet
   */
  getTransactionFlowMetrics: async (
    wallet: string,
    timeWindowHours: number = 24
  ): Promise<TransactionFlowMetrics> => {
    const response = await apiClient.get(
      `/analysis/transaction-flow/metrics/${wallet}`,
      { params: { time_window_hours: timeWindowHours } }
    );
    return response.data.metrics;
  },

  /**
   * Cross-chain identity analysis
   */
  analyzeCrossChainIdentity: async (
    request: Record<string, any>
  ): Promise<CrossChainIdentityAnalysis> => {
    const response = await apiClient.post('/analysis/cross-chain/analyze', request);
    return response.data;
  },

  /**
   * Retrieve identity cluster details
   */
  getCrossChainClusters: async (
    clusterId: string,
    walletAddresses: string[]
  ): Promise<any> => {
    const response = await apiClient.get(
      `/analysis/cross-chain/clusters/${clusterId}`,
      { params: { wallet_addresses: walletAddresses } }
    );
    return response.data;
  },
};

// Prompt Management API
export const promptsAPI = {
  listAgents: async (): Promise<{ agents: AgentListItem[] }> => {
    const response = await apiClient.get('/prompts');
    return response.data;
  },
  
  getAgentPrompt: async (agentId: string): Promise<PromptResponse> => {
    const response = await apiClient.get(`/prompts/${agentId}`);
    return response.data;
  },
  
  updateAgentPrompt: async (
    agentId: string,
    data: PromptUpdate
  ): Promise<PromptResponse> => {
    const response = await apiClient.put(`/prompts/${agentId}`, data);
    return response.data;
  },
  
  resetAgentPrompt: async (agentId: string): Promise<PromptResponse> => {
    const response = await apiClient.post(`/prompts/${agentId}/reset`);
    return response.data;
  }
};

// Legacy exports for backward compatibility
export const login = authAPI.login;
export const register = authAPI.register;
export const logout = authAPI.logout;
export const getCurrentUser = authAPI.getCurrentUser;
export const getChatSessions = chatAPI.getChatSessions;
export const getChatSession = chatAPI.getChatSession;
export const createChatSession = chatAPI.createChatSession;
export const sendChatMessage = chatAPI.sendChatMessage;
export const executeGraphQuery = graphAPI.executeCypher;
export const getGraphSchema = graphAPI.getSchema;
export const runCrew = crewAPI.runCrew;
export const getCrewStatus = crewAPI.getStatus;
export const getAvailableCrews = crewAPI.listCrews;
export const analyzeText = analysisAPI.analyzeText;
export const analyzeImage = analysisAPI.analyzeImage;
export const fetchAnalysisResults = analysisAPI.fetchAnalysisResults; // Added legacy export
export const getSimBalances = analysisAPI.getSimBalances;
export const getSimActivity = analysisAPI.getSimActivity;
export const listAgents = promptsAPI.listAgents;
export const getAgentPrompt = promptsAPI.getAgentPrompt;
export const updateAgentPrompt = promptsAPI.updateAgentPrompt;
export const resetAgentPrompt = promptsAPI.resetAgentPrompt;

export default apiClient;
