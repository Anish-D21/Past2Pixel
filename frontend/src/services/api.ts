import {
  AuthResponse,
  DamageAnalysisResult,
  OCRResult,
  OCRStatusInfo,
  RestorationResponse,
  RestorationSession,
  SessionDetailResponse,
  SystemStatus,
  TextPredictionResponse,
  UploadResponse,
  User,
} from '../types/api';

const API_BASE = '/api';

/**
 * Standard Token Storage in Memory and localStorage with safe retrieval
 */
export const getToken = (): string | null => {
  try {
    return localStorage.getItem('past2pixel_token');
  } catch {
    return null;
  }
};

export const setToken = (token: string): void => {
  try {
    localStorage.setItem('past2pixel_token', token);
  } catch {
    // ignore
  }
};

export const removeToken = (): void => {
  try {
    localStorage.removeItem('past2pixel_token');
  } catch {
    // ignore
  }
};

/**
 * Common fetch wrapper with Bearer token injection and centralized error handling
 */
async function request<T>(
  endpoint: string,
  options: RequestInit = {}
): Promise<T> {
  const token = getToken();
  const headers = new Headers(options.headers || {});

  if (token && !headers.has('Authorization')) {
    headers.set('Authorization', `Bearer ${token}`);
  }

  // Do not set Content-Type for FormData (browser sets boundary automatically)
  if (!(options.body instanceof FormData) && !headers.has('Content-Type')) {
    headers.set('Content-Type', 'application/json');
  }

  const url = endpoint.startsWith('http') ? endpoint : `${API_BASE}${endpoint}`;

  const response = await fetch(url, {
    ...options,
    headers,
  });

  if (response.status === 401) {
    // Unauthorized: token might be expired
    removeToken();
  }

  if (!response.ok) {
    let errorDetail = `Request failed with status ${response.status}`;
    try {
      const errorJson = await response.json();
      errorDetail = errorJson.detail || errorJson.message || errorDetail;
    } catch {
      // response wasn't JSON
    }
    throw new Error(errorDetail);
  }

  return response.json() as Promise<T>;
}

export const api = {
  // Authentication & Users
  auth: {
    register: (data: { email: string; username: string; password: string }): Promise<AuthResponse> =>
      request<AuthResponse>('/auth/register', {
        method: 'POST',
        body: JSON.stringify(data),
      }),

    login: (data: { username_or_email: string; password: string }): Promise<AuthResponse> =>
      request<AuthResponse>('/auth/login', {
        method: 'POST',
        body: JSON.stringify(data),
      }),

    getMe: (): Promise<User> => request<User>('/auth/me'),
  },

  // System Diagnostics & Hardware
  system: {
    getStatus: (): Promise<SystemStatus> => request<SystemStatus>('/health'),
    getReadiness: (): Promise<any> => request<any>('/readiness'),
  },

  // Upload & Sessions
  upload: {
    uploadManuscript: (file: File): Promise<UploadResponse> => {
      const formData = new FormData();
      formData.append('file', file);
      return request<UploadResponse>('/upload', {
        method: 'POST',
        body: formData,
      });
    },
  },

  // Restoration & Damage Localization
  restore: {
    processSession: (data: {
      session_id: string;
      device_preference?: string;
      enable_enhancement?: boolean;
    }): Promise<RestorationResponse> =>
      request<RestorationResponse>('/restore', {
        method: 'POST',
        body: JSON.stringify(data),
      }),
  },

  // Optical Character Recognition (OCR)
  ocr: {
    getStatus: (): Promise<OCRStatusInfo> => request<OCRStatusInfo>('/ocr/status'),

    extractText: (
      sessionId: string,
      language: string = 'san+hin',
      targetImage: 'restored' | 'original' | 'baseline' | 'model' | 'compare_all' = 'restored',
      preprocessing: string = 'none',
      mode: 'multi-pass' | 'single-pass' = 'multi-pass',
      psm?: number
    ): Promise<OCRResult> =>
      request<OCRResult>('/ocr', {
        method: 'POST',
        body: JSON.stringify({
          session_id: sessionId,
          language: language,
          target_image: targetImage,
          preprocessing: preprocessing,
          mode: mode,
          psm: psm,
        }),
      }),

    extractDirect: (
      file: File,
      language: string = 'san+hin',
      preprocessing: string = 'none',
      mode: 'multi-pass' | 'single-pass' = 'multi-pass',
      psm?: number
    ): Promise<OCRResult> => {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('language', language);
      formData.append('preprocessing', preprocessing);
      formData.append('mode', mode);
      if (psm !== undefined) {
        formData.append('psm', String(psm));
      }
      return request<OCRResult>('/ocr/direct', {
        method: 'POST',
        body: formData,
      });
    },
  },

  // Context-Aware Linguistic Text Prediction
  prediction: {
    predictText: (data: {
      left_context?: string;
      right_context?: string;
      uncertain_span?: string;
      full_text?: string;
      span_index?: number;
      max_candidates?: number;
    }): Promise<TextPredictionResponse> =>
      request<TextPredictionResponse>('/predict-text', {
        method: 'POST',
        body: JSON.stringify(data),
      }),
  },

  // Session History Management
  sessions: {
    listMySessions: (): Promise<RestorationSession[]> =>
      request<RestorationSession[]>('/sessions'),

    getSession: (sessionId: string): Promise<SessionDetailResponse> =>
      request<SessionDetailResponse>(`/sessions/${sessionId}`),

    deleteSession: (sessionId: string): Promise<{ deleted: boolean; session_id: string }> =>
      request<{ deleted: boolean; session_id: string }>(`/sessions/${sessionId}`, {
        method: 'DELETE',
      }),
  },
};
