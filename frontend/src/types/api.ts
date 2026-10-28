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
    is_installed?: boolean;
    tesseract_available: boolean;
    tesseract_version?: string;
    version?: string;
    tesseract_cmd?: string | null;
    tessdata_dir?: string | null;
    available_languages: string[];
    supported_configurations?: string[];
    has_devanagari_support: boolean;
    devanagari_support?: boolean;
    has_sanskrit_pack?: boolean;
    has_hindi_pack?: boolean;
    has_english_pack?: boolean;
    missing_languages?: string[];
    reason?: string;
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
    baseline?: string | null;
    model?: string | null;
    text_mask?: string | null;
    damage_overlay?: string | null;
    damage_mask?: string | null;
  };
  candidate_selection?: {
    selected_candidate: string;
    selection_reason: string;
    candidates_evaluated: Record<
      string,
      {
        composite_score: number;
        text_contrast: number;
        stroke_sharpness: number;
        stroke_preservation_ratio: number;
        background_noise_std: number;
      }
    >;
  };
  text_protection?: {
    ink_pixel_ratio: number;
    ink_pixel_count: number;
  };
}

export interface OCRWordDetail {
  word: string;
  original_word: string;
  confidence: number;
  status: 'IMAGE_VERIFIED' | 'IMAGE_SUPPORTED_CORRECTION' | 'UNCERTAIN_REVIEW_NEEDED' | 'DAMAGED_UNREADABLE';
  is_uncertain: boolean;
  evidence?: string;
  alternatives?: string[];
  line_num?: number;
  token_index?: number;
  patch_metrics?: {
    stroke_density?: number;
    contrast_ratio?: number;
    edge_energy?: number;
    has_visible_ink?: boolean;
    patch_data_uri?: string | null;
    bbox?: [number, number, number, number];
  };
}

export interface ImageGroundedCorrection {
  line_num: number;
  original: string;
  corrected: string;
  confidence: number;
  reason: string;
}

export interface OCRLineDetail {
  line_num: number;
  text: string;
  mean_confidence?: number | null;
  pass_name: string;
  bbox?: [number, number, number, number] | null;
  is_uncertain?: boolean;
  words?: OCRWordDetail[] | null;
  alternate_readings?: Array<{
    text: string;
    pass_name: string;
    mean_conf: number;
  }>;
}

export interface OCRResult {
  session_id?: string | null;
  status: 'SUCCESS' | 'UNAVAILABLE' | 'FAILED' | 'IMAGE_NOT_FOUND' | 'NO_LANGUAGE_DATA' | 'MISSING_LANGUAGE_PACK';
  extracted_text: string;
  raw_extracted_text?: string | null;
  corrected_text?: string | null;
  processing_time_ms: number;
  language_requested: string;
  language_used?: string | null;
  character_count: number;
  word_count: number;
  lines_count?: number;
  mean_confidence?: number | null;
  is_authentic_ocr: boolean;
  is_multi_pass?: boolean;
  image_source?: string | null;
  preprocessing_applied?: string | null;
  lines_detail?: OCRLineDetail[] | null;
  words_detail?: OCRWordDetail[] | null;
  image_grounded_corrections?: ImageGroundedCorrection[] | null;
  corrections_count?: number;
  uncertain_lines_count?: number;
  uncertain_lines?: OCRLineDetail[] | null;
  uncertain_spans_count?: number;
  image_quality?: {
    blur_score?: number;
    contrast_ratio?: number;
    skew_angle_deg?: number;
    background_std?: number;
    recommended_mode?: string;
  } | null;
  secondary_model_status?: {
    name: string;
    status: string;
    neural_model_installed?: boolean;
    notes?: string;
  } | null;
  deskew_applied?: boolean;
  pass_summaries?: Record<
    string,
    {
      name: string;
      lines_count: number;
      character_count: number;
      mean_confidence?: number | null;
      processing_time_ms: number;
    }
  > | null;
  provenance_breakdown?: Record<string, number> | null;
  original_dimensions?: [number, number] | null;
  processed_dimensions?: [number, number] | null;
  super_resolution_applied?: boolean;
  upscale_factor?: number;
  comparison_results?: Record<
    string,
    {
      extracted_text: string;
      character_count: number;
      word_count: number;
      mean_confidence?: number | null;
      status: string;
      image_file: string;
      preprocessing_applied?: string | null;
    }
  > | null;
  error?: string | null;
  setup_instructions?: Record<string, any> | null;
}

export interface OCRStatusInfo {
  status: 'READY' | 'LIMITED_LANGUAGES' | 'UNAVAILABLE';
  is_installed: boolean;
  tesseract_cmd: string | null;
  tessdata_dir?: string | null;
  version: string | null;
  available_languages: string[];
  supported_configurations?: string[];
  devanagari_support: boolean;
  has_sanskrit_pack: boolean;
  has_hindi_pack: boolean;
  has_english_pack?: boolean;
  secondary_recognition?: {
    name: string;
    status: string;
    available: boolean;
    notes?: string;
  } | null;
  reason: string;
  setup_instructions?: Record<string, any> | null;
}

export interface TextPredictionCandidate {
  word: string;
  score: number;
  evidence?: string | null;
  reconstructed_sentence?: string | null;
}

export interface TextPredictionResponse {
  status: string;
  prediction?: string | null;
  confidence_score?: number | null;
  original_sentence?: string | null;
  reconstructed_sentence?: string | null;
  candidates: TextPredictionCandidate[];
  is_speculative: boolean;
  uncertainty_label: string;
  disclaimer: string;
  surrounding_context: {
    left_context: string;
    right_context: string;
    uncertain_span?: string;
  };
  detected_script?: string | null;
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
    baseline?: string | null;
    model?: string | null;
    text_mask?: string | null;
    damage_overlay?: string | null;
    damage_mask?: string | null;
  };
  ocr_result?: OCRResult | null;
  restoration_result?: {
    timings_ms: Record<string, number>;
    model_readiness: Record<string, string>;
    damage_detection: Record<string, any>;
    restoration: Record<string, any>;
    device_info?: Record<string, any>;
    candidate_selection?: Record<string, any>;
    text_protection?: Record<string, any>;
  } | null;
  text_predictions: any[];
}

export type RestorationSession = SessionSummary;
export type DamageAnalysisResult = Record<string, any>;

