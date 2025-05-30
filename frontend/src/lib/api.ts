import axios from 'axios'

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

// Create axios instance with default config
export const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
})

// Request interceptor
api.interceptors.request.use(
  (config) => {
    // Add auth token if available
    const token = localStorage.getItem('auth_token')
    if (token) {
      config.headers.Authorization = `Bearer ${token}`
    }
    return config
  },
  (error) => {
    return Promise.reject(error)
  }
)

// Response interceptor
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      // Handle unauthorized access
      localStorage.removeItem('auth_token')
      // Redirect to login if needed
    }
    return Promise.reject(error)
  }
)

// API endpoints
export const chatAPI = {
  sendMessage: (data: {
    message: string
    context?: string
    conversation_id?: string
    include_graph_data?: boolean
  }) => api.post('/api/v1/chat/message', data),

  analyzeImage: (file: File, prompt?: string, extractEntities?: boolean) => {
    const formData = new FormData()
    formData.append('file', file)
    if (prompt) formData.append('prompt', prompt)
    if (extractEntities) formData.append('extract_entities', 'true')
    
    return api.post('/api/v1/chat/analyze-image', formData, {
      headers: { 'Content-Type': 'multipart/form-data' }
    })
  },

  getConversation: (conversationId: string) =>
    api.get(`/api/v1/chat/conversation/${conversationId}`),

  deleteConversation: (conversationId: string) =>
    api.delete(`/api/v1/chat/conversation/${conversationId}`),
}

export const graphAPI = {
  getSchema: () => api.get('/api/v1/graph/schema'),

  executeCypher: (query: string, parameters?: Record<string, any>) =>
    api.post('/api/v1/graph/query/cypher', { query, parameters }),

  naturalLanguageQuery: (question: string, context?: string) =>
    api.post('/api/v1/graph/query/natural', { question, context }),

  createNode: (labels: string | string[], properties: Record<string, any>) =>
    api.post('/api/v1/graph/nodes', { labels, properties }),

  createRelationship: (
    from_node_id: number,
    to_node_id: number,
    relationship_type: string,
    properties?: Record<string, any>
  ) =>
    api.post('/api/v1/graph/relationships', {
      from_node_id,
      to_node_id,
      relationship_type,
      properties,
    }),

  searchNodes: (
    labels?: string | string[],
    properties?: Record<string, any>,
    limit?: number
  ) =>
    api.post('/api/v1/graph/search', { labels, properties, limit }),

  calculateCentrality: (algorithm: string = 'pagerank', limit: number = 20) =>
    api.get('/api/v1/graph/analytics/centrality', {
      params: { algorithm, limit },
    }),

  detectCommunities: (algorithm: string = 'louvain') =>
    api.get('/api/v1/graph/analytics/communities', {
      params: { algorithm },
    }),
}

export const analysisAPI = {
  executeCode: (
    code: string,
    libraries?: string[],
    timeout?: number,
    sandbox_id?: string
  ) =>
    api.post('/api/v1/analysis/execute-code', {
      code,
      libraries,
      timeout,
      sandbox_id,
    }),

  performAnalysis: (
    task_description: string,
    data_source?: string,
    parameters?: Record<string, any>,
    output_format?: string
  ) =>
    api.post('/api/v1/analysis/analyze', {
      task_description,
      data_source,
      parameters,
      output_format,
    }),

  detectFraudPatterns: (
    pattern_type: string = 'money_laundering',
    limit: number = 100
  ) =>
    api.get('/api/v1/analysis/fraud-detection/patterns', {
      params: { pattern_type, limit },
    }),

  listSandboxFiles: (sandbox_id: string, directory: string = '.') =>
    api.get(`/api/v1/analysis/sandbox/${sandbox_id}/files`, {
      params: { directory },
    }),

  downloadSandboxFile: (sandbox_id: string, file_path: string) =>
    api.get(`/api/v1/analysis/sandbox/${sandbox_id}/download/${file_path}`),
}

export const systemAPI = {
  healthCheck: () => api.get('/health'),
  getRoot: () => api.get('/'),
}

// Utility functions
export const handleAPIError = (error: any) => {
  if (error.response) {
    // Server responded with error status
    return {
      message: error.response.data?.detail || 'An error occurred',
      status: error.response.status,
      data: error.response.data,
    }
  } else if (error.request) {
    // Request was made but no response received
    return {
      message: 'Network error - please check your connection',
      status: 0,
      data: null,
    }
  } else {
    // Something else happened
    return {
      message: error.message || 'An unexpected error occurred',
      status: -1,
      data: null,
    }
  }
}

export default api
