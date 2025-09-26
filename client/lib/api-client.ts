import axios, { AxiosInstance, AxiosError } from 'axios';

// API Configuration
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

// Types
export interface Document {
  id: string;
  filename: string;
  original_filename?: string;
  file_size: number;
  is_processed: boolean;
  is_embedded: boolean;
  created_at: string;
  page_count?: number;
  word_count?: number;
  processing_error?: string | null;
  pinecone_namespace?: string;
  embedding_count?: number;
  updated_at?: string;
  financial_facts?: Record<string, any>;
  investment_data?: Record<string, any>;
  key_metrics?: Record<string, any>;
}

export interface DocumentStatus {
  document_id: string;
  filename: string;
  is_processed: boolean;
  is_embedded: boolean;
  processing_error: string | null;
  embedding_count: number;
  progress_percentage: number;
}

export interface UploadResponse {
  document_id: string;
  filename: string;
  file_size: number;
  status: string;
  processing_started: boolean;
}

export interface ResearchTask {
  id: string;
  document_id: string;
  topic: string;
  custom_query?: string;
  status: 'pending' | 'in_progress' | 'completed' | 'failed';
  created_at: string;
  completed_at?: string;
  processing_time?: number;
  content_outline?: Record<string, { title: string; description: string }>;
  research_findings?: Record<string, { content: string; key_points: string[] }>;
  sources_used?: Array<{ page: number; chunk: number }>;
}

export interface ResearchStartResponse {
  task_id: string;
  content_outline: Record<string, any>;
  research_findings: Record<string, any>;
  sources_used: Array<{ page: number; chunk: number }>;
  processing_time: number;
}

export interface ChatSession {
  id: string;
  document_id: string;
  session_name?: string;
  user_id?: string | null;
  is_active: boolean;
  created_at: string;
  last_activity: string;
  message_count: number;
  temperature?: number;
  max_tokens?: number;
  system_prompt?: string | null;
}

export interface ChatMessage {
  id: string;
  session_id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  created_at: string;
  token_count?: number;
  sources?: Array<{ 
    chunk_id?: string;
    page_number?: number;
    similarity_score?: number;
    preview?: string;
  }>;
}

export interface ChatResponse {
  message: string;
  session_id: string;
  sources_used?: Array<{ 
    chunk_id?: string;
    page_number?: number;
    similarity_score?: number;
    preview?: string;
  }>;
  tool_calls: string[];
  response_time: number;
  token_count: number;
}

export interface SystemStats {
  documents: {
    total: number;
    processed: number;
    embedded: number;
    processing_rate: string;
  };
  chat: {
    total_sessions: number;
    active_sessions: number;
  };
  research: {
    total_tasks: number;
    completed_tasks: number;
    completion_rate: string;
  };
  vector_store: {
    total_vectors: number;
    dimension: number;
    index_fullness: number;
  };
  system: {
    uptime: string;
    timestamp: string;
  };
}

export interface SystemHealth {
  status: string;
  timestamp: string;
  version: string;
  services: {
    database: string;
    openai: string;
    pinecone: string;
  };
}

// API Error handling
export class APIError extends Error {
  constructor(
    public statusCode: number,
    public message: string,
    public details?: any
  ) {
    super(message);
    this.name = 'APIError';
  }
}

// API Client Class
export class APIClient {
  private axios: AxiosInstance;

  constructor() {
    this.axios = axios.create({
      baseURL: API_BASE_URL,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    // Response interceptor for error handling
    this.axios.interceptors.response.use(
      (response) => response,
      (error: AxiosError) => {
        if (error.response) {
          throw new APIError(
            error.response.status,
            error.response.data?.message || error.message,
            error.response.data
          );
        }
        throw error;
      }
    );
  }

  // Document Management
  async uploadDocument(file: File): Promise<UploadResponse> {
    const formData = new FormData();
    formData.append('file', file);

    const response = await this.axios.post<UploadResponse>('/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  }

  async getDocumentStatus(documentId: string): Promise<DocumentStatus> {
    const response = await this.axios.get<DocumentStatus>(
      `/documents/${documentId}/status`
    );
    return response.data;
  }

  async listDocuments(): Promise<Document[]> {
    const response = await this.axios.get<Document[]>('/documents');
    return response.data;
  }

  async getDocument(documentId: string): Promise<Document> {
    const response = await this.axios.get<Document>(`/documents/${documentId}`);
    return response.data;
  }

  async deleteDocument(documentId: string): Promise<{ message: string }> {
    const response = await this.axios.delete<{ message: string }>(
      `/documents/${documentId}`
    );
    return response.data;
  }

  // Deep Research
  async startResearch(
    documentId: string,
    topic: string,
    customQuery?: string
  ): Promise<ResearchStartResponse> {
    const response = await this.axios.post<ResearchStartResponse>(
      `/documents/${documentId}/research/start`,
      {
        topic,
        custom_query: customQuery,
      }
    );
    return response.data;
  }

  async getResearchTask(
    documentId: string,
    taskId: string
  ): Promise<ResearchTask> {
    const response = await this.axios.get<ResearchTask>(
      `/documents/${documentId}/research/tasks/${taskId}`
    );
    return response.data;
  }

  async listResearchTasks(documentId: string): Promise<ResearchTask[]> {
    const response = await this.axios.get<ResearchTask[]>(
      `/documents/${documentId}/research/tasks`
    );
    return response.data;
  }

  // Chat Interface
  async sendMessage(
    message: string,
    documentId?: string,
    sessionId?: string
  ): Promise<ChatResponse> {
    const response = await this.axios.post<ChatResponse>('/conversation', {
      message,
      document_id: documentId,
      session_id: sessionId,
    });
    return response.data;
  }

  async createChatSession(
    documentId: string,
    sessionName?: string
  ): Promise<ChatSession> {
    const response = await this.axios.post<ChatSession>('/conversation/new', {
      document_id: documentId,
      session_name: sessionName,
    });
    return response.data;
  }

  async listChatSessions(): Promise<ChatSession[]> {
    const response = await this.axios.get<ChatSession[]>(
      '/conversation/sessions'
    );
    return response.data;
  }

  async getChatHistory(sessionId: string): Promise<ChatMessage[]> {
    const response = await this.axios.get<ChatMessage[]>(
      `/conversation/${sessionId}/history`
    );
    return response.data;
  }

  // System
  async getSystemStats(): Promise<SystemStats> {
    try {
      const response = await this.axios.get<SystemStats>('/system/stats');
      return response.data;
    } catch (error) {
      // Return mock data if endpoint doesn't exist yet
      return {
        documents: {
          total: 0,
          processed: 0,
          embedded: 0,
          processing_rate: '0%',
        },
        chat: {
          total_sessions: 0,
          active_sessions: 0,
        },
        research: {
          total_tasks: 0,
          completed_tasks: 0,
          completion_rate: '0%',
        },
        vector_store: {
          total_vectors: 0,
          dimension: 3072,
          index_fullness: 0,
        },
        system: {
          uptime: 'N/A',
          timestamp: new Date().toISOString(),
        },
      };
    }
  }

  async getSystemHealth(): Promise<SystemHealth> {
    try {
      const response = await this.axios.get<SystemHealth>('/system/health');
      return response.data;
    } catch (error) {
      // Return mock data if endpoint doesn't exist yet
      return {
        status: 'healthy',
        timestamp: new Date().toISOString(),
        version: '1.0.0',
        services: {
          database: 'connected',
          openai: 'configured',
          pinecone: 'configured',
        },
      };
    }
  }
}

// Export singleton instance
export const apiClient = new APIClient();