import axios from 'axios';
import { 
  authAPI, 
  chatAPI, 
  graphAPI, 
  crewAPI, 
  analysisAPI, 
  handleAPIError,
  LoginRequest,
  RegisterRequest
} from '../api';

// Mock axios
jest.mock('axios');
const mockedAxios = axios as jest.Mocked<typeof axios>;

// Mock localStorage
const localStorageMock = (() => {
  let store: Record<string, string> = {};
  return {
    getItem: jest.fn((key: string) => store[key] || null),
    setItem: jest.fn((key: string, value: string) => {
      store[key] = value;
    }),
    removeItem: jest.fn((key: string) => {
      delete store[key];
    }),
    clear: jest.fn(() => {
      store = {};
    }),
  };
})();

Object.defineProperty(window, 'localStorage', {
  value: localStorageMock,
});

describe('API Client', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    localStorageMock.clear();
  });

  describe('Auth API', () => {
    test('login should make POST request and store token', async () => {
      // Arrange
      const loginData: LoginRequest = {
        username: 'testuser',
        password: 'password123'
      };
      
      const mockResponse = {
        data: {
          access_token: 'mock-token',
          token_type: 'bearer',
          user: {
            id: 'user-123',
            username: 'testuser',
            email: 'test@example.com',
            is_active: true,
            is_superuser: false
          }
        }
      };
      
      mockedAxios.post.mockResolvedValueOnce(mockResponse);
      
      // Act
      const result = await authAPI.login(loginData);
      
      // Assert
      expect(mockedAxios.post).toHaveBeenCalledWith('/auth/login', loginData);
      expect(localStorageMock.setItem).toHaveBeenCalledWith('auth_token', 'mock-token');
      expect(result).toEqual(mockResponse.data);
    });
    
    test('register should make POST request', async () => {
      // Arrange
      const registerData: RegisterRequest = {
        username: 'newuser',
        email: 'new@example.com',
        password: 'password123',
        full_name: 'New User'
      };
      
      const mockResponse = {
        data: {
          access_token: 'new-token',
          token_type: 'bearer',
          user: {
            id: 'user-456',
            username: 'newuser',
            email: 'new@example.com',
            full_name: 'New User',
            is_active: true,
            is_superuser: false
          }
        }
      };
      
      mockedAxios.post.mockResolvedValueOnce(mockResponse);
      
      // Act
      const result = await authAPI.register(registerData);
      
      // Assert
      expect(mockedAxios.post).toHaveBeenCalledWith('/auth/register', registerData);
      expect(result).toEqual(mockResponse.data);
    });
    
    test('logout should remove token from localStorage', async () => {
      // Arrange
      localStorageMock.setItem('auth_token', 'some-token');
      
      // Act
      await authAPI.logout();
      
      // Assert
      expect(localStorageMock.removeItem).toHaveBeenCalledWith('auth_token');
    });
    
    test('getCurrentUser should make GET request with auth header', async () => {
      // Arrange
      localStorageMock.setItem('auth_token', 'existing-token');
      
      const mockResponse = {
        data: {
          id: 'user-123',
          username: 'testuser',
          email: 'test@example.com',
          is_active: true,
          is_superuser: false
        }
      };
      
      mockedAxios.get.mockResolvedValueOnce(mockResponse);
      
      // Act
      const result = await authAPI.getCurrentUser();
      
      // Assert
      expect(mockedAxios.get).toHaveBeenCalledWith('/auth/me');
      expect(result).toEqual(mockResponse.data);
    });
  });

  describe('Chat API', () => {
    test('sendMessage should make POST request with message', async () => {
      // Arrange
      const message = 'Hello, AI assistant!';
      const includeGraphData = true;
      
      const mockResponse = {
        data: {
          conversation_id: 'conv-123',
          response: 'Hello, human! How can I help you today?',
          cypher_query: null,
          graph_results: null
        }
      };
      
      mockedAxios.post.mockResolvedValueOnce(mockResponse);
      
      // Act
      const result = await chatAPI.sendMessage(message, includeGraphData);
      
      // Assert
      expect(mockedAxios.post).toHaveBeenCalledWith('/chat/message', {
        message,
        include_graph_data: includeGraphData
      });
      expect(result).toEqual(mockResponse);
    });
    
    test('getChatSessions should make GET request', async () => {
      // Arrange
      const mockResponse = {
        data: [
          {
            id: 'session-1',
            title: 'First conversation',
            created_at: '2025-01-01T12:00:00Z',
            updated_at: '2025-01-01T12:30:00Z',
            messages: []
          }
        ]
      };
      
      mockedAxios.get.mockResolvedValueOnce(mockResponse);
      
      // Act
      const result = await chatAPI.getChatSessions();
      
      // Assert
      expect(mockedAxios.get).toHaveBeenCalledWith('/chat/sessions');
      expect(result).toEqual(mockResponse.data);
    });
    
    test('analyzeImage should make POST request with FormData', async () => {
      // Arrange
      const file = new File(['dummy content'], 'test-image.jpg', { type: 'image/jpeg' });
      const prompt = 'Describe this image';
      
      const mockResponse = {
        data: {
          analysis: 'This image shows a landscape with mountains.'
        }
      };
      
      mockedAxios.post.mockResolvedValueOnce(mockResponse);
      
      // Act
      const result = await chatAPI.analyzeImage(file, prompt);
      
      // Assert
      expect(mockedAxios.post).toHaveBeenCalled();
      // Check that first arg is the correct endpoint
      expect(mockedAxios.post.mock.calls[0][0]).toBe('/chat/analyze-image');
      // Check that second arg is FormData (can't easily check content)
      expect(mockedAxios.post.mock.calls[0][1] instanceof FormData).toBe(true);
      // Check headers
      expect(mockedAxios.post.mock.calls[0][2]).toEqual({
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      expect(result).toEqual(mockResponse);
    });
  });

  describe('Graph API', () => {
    test('executeCypher should make POST request with query', async () => {
      // Arrange
      const query = 'MATCH (n) RETURN n LIMIT 10';
      const parameters = { param1: 'value1' };
      
      const mockResponse = {
        data: {
          results: [{ n: { id: 1, labels: ['Person'], properties: { name: 'Alice' } } }]
        }
      };
      
      mockedAxios.post.mockResolvedValueOnce(mockResponse);
      
      // Act
      const result = await graphAPI.executeCypher(query, parameters);
      
      // Assert
      expect(mockedAxios.post).toHaveBeenCalledWith('/graph/query', {
        query,
        parameters
      });
      expect(result).toEqual(mockResponse);
    });
    
    test('getSchema should make GET request', async () => {
      // Arrange
      const mockResponse = {
        data: {
          labels: ['Person', 'Movie'],
          relationships: ['ACTED_IN', 'DIRECTED'],
          properties: ['name', 'title', 'year']
        }
      };
      
      mockedAxios.get.mockResolvedValueOnce(mockResponse);
      
      // Act
      const result = await graphAPI.getSchema();
      
      // Assert
      expect(mockedAxios.get).toHaveBeenCalledWith('/graph/schema');
      expect(result).toEqual(mockResponse);
    });
    
    test('naturalLanguageQuery should make POST request with question', async () => {
      // Arrange
      const question = 'Who acted in The Matrix?';
      
      const mockResponse = {
        data: {
          cypher: 'MATCH (p:Person)-[:ACTED_IN]->(m:Movie {title: "The Matrix"}) RETURN p.name',
          results: [{ 'p.name': 'Keanu Reeves' }, { 'p.name': 'Carrie-Anne Moss' }]
        }
      };
      
      mockedAxios.post.mockResolvedValueOnce(mockResponse);
      
      // Act
      const result = await graphAPI.naturalLanguageQuery(question);
      
      // Assert
      expect(mockedAxios.post).toHaveBeenCalledWith('/graph/nlq', {
        question
      });
      expect(result).toEqual(mockResponse);
    });
  });

  describe('Analysis API', () => {
    test('analyzeText should make POST request with text data', async () => {
      // Arrange
      const analysisRequest = {
        text: 'Analyze this suspicious transaction pattern',
        options: { detailed: true }
      };
      
      const mockResponse = {
        data: {
          analysis: 'This appears to be a structuring pattern...',
          execution_time_ms: 1500
        }
      };
      
      mockedAxios.post.mockResolvedValueOnce(mockResponse);
      
      // Act
      const result = await analysisAPI.analyzeText(analysisRequest);
      
      // Assert
      expect(mockedAxios.post).toHaveBeenCalledWith('/analysis/text', analysisRequest);
      expect(result).toEqual(mockResponse.data);
    });
    
    test('fetchAnalysisResults should make GET request with taskId', async () => {
      // Arrange
      const taskId = 'task-123';
      
      const mockResponse = {
        data: {
          task_id: taskId,
          status: 'completed',
          title: 'Transaction Analysis',
          executive_summary: 'Detected potential structuring pattern',
          risk_score: 0.85,
          confidence: 0.92,
          detailed_findings: 'Multiple transactions just below reporting threshold...'
        }
      };
      
      mockedAxios.get.mockResolvedValueOnce(mockResponse);
      
      // Act
      const result = await analysisAPI.fetchAnalysisResults(taskId);
      
      // Assert
      expect(mockedAxios.get).toHaveBeenCalledWith(`/analysis/${taskId}`);
      expect(result).toEqual(mockResponse.data);
    });
  });

  describe('Crew API', () => {
    test('runCrew should make POST request with crew request', async () => {
      // Arrange
      const crewRequest = {
        crew_name: 'fraud_detection',
        inputs: { 
          transaction_data: 'csv_data',
          threshold: 10000
        },
        async_execution: true
      };
      
      const mockResponse = {
        data: {
          success: true,
          crew_name: 'fraud_detection',
          task_id: 'task-456',
          result: null
        }
      };
      
      mockedAxios.post.mockResolvedValueOnce(mockResponse);
      
      // Act
      const result = await crewAPI.runCrew(crewRequest);
      
      // Assert
      expect(mockedAxios.post).toHaveBeenCalledWith('/crew/run', crewRequest);
      expect(result).toEqual(mockResponse);
    });
    
    test('getStatus should make GET request with taskId', async () => {
      // Arrange
      const taskId = 'task-456';
      
      const mockResponse = {
        data: {
          task_id: taskId,
          status: 'running',
          progress: 0.65,
          message: 'Analyzing transaction patterns'
        }
      };
      
      mockedAxios.get.mockResolvedValueOnce(mockResponse);
      
      // Act
      const result = await crewAPI.getStatus(taskId);
      
      // Assert
      expect(mockedAxios.get).toHaveBeenCalledWith(`/crew/status/${taskId}`);
      expect(result).toEqual(mockResponse);
    });
  });

  describe('Error Handling', () => {
    test('handleAPIError should process response errors', () => {
      // Arrange
      const responseError = {
        response: {
          status: 400,
          data: { detail: 'Invalid input' },
          statusText: 'Bad Request'
        }
      };
      
      // Act
      const result = handleAPIError(responseError);
      
      // Assert
      expect(result).toEqual({
        status: 400,
        message: 'Invalid input'
      });
    });
    
    test('handleAPIError should process request errors', () => {
      // Arrange
      const requestError = {
        request: {},
        message: 'Network Error'
      };
      
      // Act
      const result = handleAPIError(requestError);
      
      // Assert
      expect(result).toEqual({
        status: 500,
        message: 'No response from server. Please check your connection.'
      });
    });
    
    test('handleAPIError should process other errors', () => {
      // Arrange
      const otherError = {
        message: 'Something went wrong'
      };
      
      // Act
      const result = handleAPIError(otherError);
      
      // Assert
      expect(result).toEqual({
        status: 500,
        message: 'Something went wrong'
      });
    });
    
    test('API calls should handle errors properly', async () => {
      // Arrange
      const error = {
        response: {
          status: 401,
          data: { detail: 'Authentication failed' },
          statusText: 'Unauthorized'
        }
      };
      
      mockedAxios.get.mockRejectedValueOnce(error);
      
      // Act & Assert
      await expect(chatAPI.getChatSessions()).rejects.toThrow();
    });
  });

  describe('Auth Token Handling', () => {
    test('API requests should include auth token from localStorage', async () => {
      // Arrange
      localStorageMock.setItem('auth_token', 'test-token');
      
      const mockResponse = { data: {} };
      mockedAxios.get.mockResolvedValueOnce(mockResponse);
      
      // Act
      await chatAPI.getChatSessions();
      
      // Assert - Check that the interceptor added the Authorization header
      const requestConfig = mockedAxios.get.mock.calls[0][1];
      expect(requestConfig).toBeDefined();
      expect(requestConfig?.headers?.Authorization).toBe('Bearer test-token');
    });
    
    test('API requests should not include auth token when not present', async () => {
      // Arrange - No token in localStorage
      const mockResponse = { data: {} };
      mockedAxios.get.mockResolvedValueOnce(mockResponse);
      
      // Act
      await chatAPI.getChatSessions();
      
      // Assert - Check that no Authorization header was added
      const requestConfig = mockedAxios.get.mock.calls[0][1];
      expect(requestConfig?.headers?.Authorization).toBeUndefined();
    });
  });
});
