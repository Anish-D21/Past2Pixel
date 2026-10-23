export interface User {
  id: string;
  username: string;
  email: string;
  created_at?: string;
}

export interface AuthResponse {
  access_token: string;
  token_type: string;
  expires_in_minutes: number;
  user: User;
}

export interface SystemStatus {
  status: string;
  version: string;
  timestamp: string;
  service: string;
}

export interface ReadinessResponse {
  status: 'READY' | 'DEGRADED';
  compute: {
    device_type: string;
    device_name: string;
    cuda_available: boolean;
    device_count: number;
    platform: string;
    processor: string;
    logical_cpus: number;
  };
  models: Record<string, ModelInfo>;
  ocr: {
    status: string;
    tesseract_available: boolean;
    tesseract_version?: string;
    available_languages: string[];
    has_devanagari_support: boolean;
    missing_languages: string[];
  };
  database: {
    status: string;
    storage_mode: string;
    uri_configured: boolean;
    database_name: string;
    is_persistent: boolean;
    last_error?: string | null;
  };
}

export interface ModelInfo {
  name: string;
  checkpoint_file: string;
  status: 'READY_FOR_INFERENCE' | 'UNTRAINED' | 'ARCHITECTURE_DEFINED' | 'CHECKPOINT_SAVED' | 'NOT_INITIALIZED';
  checkpoint_exists: boolean;
  checkpoint_path?: string | null;
  training_metadata?: {
    epoch?: number;
    val_loss?: number;
    metrics?: Record<string, number>;
    trained_device?: string;
  } | null;
  error?: string;
}

export interface UploadResponse {
  session_id: string;
  filename: string;
  original_filename: string;
  width: number;
  height: number;
  file_size_bytes: number;
  image_url: string;
  created_at: string;
}

export interface RestorationResponse {
  session_id: string;
  timestamp: string;
  device_info: {
    device_type: string;
    device_name: string;
    cuda_available: boolean;
    device_count: number;
  };
  timings_ms: Record<string, number>;
  model_readiness: Record<string, string>;
  damage_detection: {
    damaged_pixels: number;
    total_pixels: number;
    damaged_pixel_percentage: number;
    damage_severity: 'MINIMAL' | 'MODERATE' | 'SEVERE';
    model_readiness: string;
    segmentation_metrics?: {
      status: string;
      reason?: string;
      iou?: number | null;
      dice?: number | null;
      precision?: number | null;
      recall?: number | null;
    };
  };
  restoration: {
    model_name: string;
    model_readiness: string;
    evaluation_metrics: {
      status: string;
      reason?: string;
      psnr?: number | null;
      ssim?: number | null;
    };
  };
  artifacts: {
    original: string;
    restored?: string | null;
    damage_overlay?: string | null;
    damage_mask?: string | null;
  };
}

export interface OCRResult {
  session_id?: string;
  status: 'SUCCESS' | 'UNAVAILABLE' | 'FAILED';
  extracted_text: string;
  processing_time_ms: number;
  language_requested: string;
  language_used?: string | null;
  character_count: number;
  word_count: number;
  mean_confidence?: number | null;
  is_authentic_ocr: boolean;
  error?: string | null;
  setup_instructions?: Record<string, any> | null;
}

export interface TextPredictionCandidate {
  word: string;
  score: number;
}

export interface TextPredictionResponse {
  status: string;
  prediction?: string | null;
  candidates: TextPredictionCandidate[];
  is_speculative: boolean;
  uncertainty_label: string;
  disclaimer: string;
  surrounding_context: {
    left_context: string;
    right_context: string;
  };
}

export interface SessionSummary {
  session_id: string;
  created_at: string;
  status: string;
  filename?: string;
  has_ocr: boolean;
  has_restoration: boolean;
  damaged_percentage?: number | null;
}

export interface SessionDetailResponse {
  session_id: string;
  user_id: string;
  status: string;
  created_at: string;
  original_metadata: Record<string, any>;
  artifacts: {
    original: string;
    restored?: string | null;
    damage_overlay?: string | null;
    damage_mask?: string | null;
  };
  ocr_result?: OCRResult | null;
  restoration_result?: {
    timings_ms: Record<string, number>;
    model_readiness: Record<string, string>;
    damage_detection: Record<string, any>;
    restoration: Record<string, any>;
  } | null;
  text_predictions: any[];
}

export type RestorationSession = SessionSummary;
export type DamageAnalysisResult = Record<string, any>;

