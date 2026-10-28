"""
Pydantic Request & Response Schemas for Past2Pixel FastAPI Backend.
Enforces typed inputs, data validation, and consistent API contracts.
"""
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, EmailStr


# -------------------------------------------------------------
# Health & Readiness Schemas
# -------------------------------------------------------------
class HealthResponse(BaseModel):
    status: str = Field(..., description="Overall service status (OK)")
    version: str = Field("1.0.0", description="API Version")
    timestamp: str = Field(..., description="Current server UTC timestamp")
    service: str = Field("Past2Pixel AI API", description="Service identifier")


class ReadinessResponse(BaseModel):
    status: str = Field(..., description="Overall readiness: READY or DEGRADED")
    compute: Dict[str, Any] = Field(..., description="Compute hardware and device detection")
    models: Dict[str, Any] = Field(..., description="Checkpoints and model lifecycle status")
    ocr: Dict[str, Any] = Field(..., description="Tesseract OCR availability and language packs")
    database: Dict[str, Any] = Field(..., description="MongoDB Atlas / local persistence status")


# -------------------------------------------------------------
# Authentication Schemas
# -------------------------------------------------------------
class UserRegisterRequest(BaseModel):
    email: str = Field(..., pattern=r"^[\w\.-]+@[\w\.-]+\.\w+$", description="User email address")
    username: str = Field(..., min_length=3, max_length=50, description="Unique username")
    password: str = Field(..., min_length=6, description="Password (at least 6 characters)")


class UserLoginRequest(BaseModel):
    username_or_email: str = Field(..., description="Username or email address")
    password: str = Field(..., description="Account password")


class UserResponse(BaseModel):
    id: str = Field(..., description="Unique User ID")
    email: str = Field(..., description="User email")
    username: str = Field(..., description="Username")
    created_at: str = Field(..., description="Registration timestamp")


class TokenResponse(BaseModel):
    access_token: str = Field(..., description="Signed JWT Bearer token")
    token_type: str = Field("bearer", description="Token type")
    expires_in_minutes: int = Field(..., description="Token validity period in minutes")
    user: UserResponse = Field(..., description="Authenticated user profile")


# -------------------------------------------------------------
# Upload & Session Schemas
# -------------------------------------------------------------
class UploadResponse(BaseModel):
    session_id: str = Field(..., description="Unique processing session ID")
    filename: str = Field(..., description="Safe sanitized server-side filename")
    original_filename: str = Field(..., description="User uploaded filename")
    width: int = Field(..., description="Image width in pixels")
    height: int = Field(..., description="Image height in pixels")
    file_size_bytes: int = Field(..., description="Image file size in bytes")
    image_url: str = Field(..., description="Static accessible URL to the original image")
    created_at: str = Field(..., description="Creation timestamp")


# -------------------------------------------------------------
# OCR Schemas
# -------------------------------------------------------------
class OCRRequest(BaseModel):
    session_id: Optional[str] = Field(None, description="Existing session ID to run OCR on")
    language: str = Field("san+hin", description="Language pack codes for Tesseract")
    target_image: Optional[str] = Field("restored", description="Target image: 'restored', 'original', 'baseline', 'model', or 'compare_all'")
    preprocessing: Optional[str] = Field("none", description="Controlled OCR preprocessing: 'none', 'auto', 'contrast', 'illumination', 'binarize', 'upscale'")
    mode: Optional[str] = Field("multi-pass", description="Extraction mode: 'multi-pass' (maximum text recovery) or 'single-pass'")
    psm: Optional[int] = Field(None, description="Optional explicit Page Segmentation Mode for single-pass")


class OCRResponse(BaseModel):
    session_id: Optional[str] = Field(None, description="Associated session ID")
    status: str = Field(..., description="SUCCESS, UNAVAILABLE, FAILED, or MISSING_LANGUAGE_PACK")
    extracted_text: str = Field("", description="Raw Unicode extracted text from visible characters")
    processing_time_ms: float = Field(0.0, description="Measured OCR duration in milliseconds")
    language_requested: str = Field("san+hin", description="Requested language string")
    language_used: Optional[str] = Field(None, description="Actual language pack used")
    character_count: int = Field(0, description="Total characters extracted")
    word_count: int = Field(0, description="Total words extracted")
    lines_count: Optional[int] = Field(0, description="Total reconstructed line count")
    mean_confidence: Optional[float] = Field(None, description="Mean character confidence percentage")
    is_authentic_ocr: bool = Field(False, description="True if extracted from real engine, False if unavailable")
    is_multi_pass: Optional[bool] = Field(False, description="True if extracted via multi-pass reconciliation")
    image_source: Optional[str] = Field(None, description="Image source used for OCR ('restored_unet.png', 'original.png', etc.)")
    preprocessing_applied: Optional[str] = Field("none", description="Image preprocessing mode applied")
    lines_detail: Optional[List[Dict[str, Any]]] = Field(None, description="Detailed line breakdown with bounding boxes, confidence, and provenance")
    uncertain_lines_count: Optional[int] = Field(0, description="Count of low-confidence or noise-flagged lines")
    uncertain_lines: Optional[List[Dict[str, Any]]] = Field(None, description="List of uncertain lines for paleographic review")
    raw_extracted_text: Optional[str] = Field(None, description="Immutable original raw OCR text before image-grounded correction")
    corrected_text: Optional[str] = Field(None, description="Final image-supported transcription verified against manuscript stroke evidence")
    words_detail: Optional[List[Dict[str, Any]]] = Field(None, description="Detailed word-level tokens with bounding boxes, stroke density, and patch thumbnails")
    image_grounded_corrections: Optional[List[Dict[str, Any]]] = Field(None, description="Audit log of paleographic corrections made using image evidence")
    corrections_count: Optional[int] = Field(0, description="Total image-grounded corrections applied")
    uncertain_spans_count: Optional[int] = Field(0, description="Count of low-confidence or unreadable words flagged for scholar review")
    image_quality: Optional[Dict[str, Any]] = Field(None, description="Physical image quality metrics: contrast ratio, blur score, skew angle")
    secondary_model_status: Optional[Dict[str, Any]] = Field(None, description="Diagnostic status of secondary recognition pass or neural model")
    deskew_applied: Optional[bool] = Field(False, description="Whether automatic skew correction was applied")
    pass_summaries: Optional[Dict[str, Any]] = Field(None, description="Per-pass statistics and processing times")
    provenance_breakdown: Optional[Dict[str, int]] = Field(None, description="Lines contributed by each pass")
    original_dimensions: Optional[List[int]] = Field(None, description="[width, height] of the input image")
    processed_dimensions: Optional[List[int]] = Field(None, description="[width, height] of the image supplied to OCR")
    super_resolution_applied: Optional[bool] = Field(False, description="Whether 2x super resolution upscaling was executed")
    upscale_factor: Optional[float] = Field(1.0, description="Image scale factor before recognition")
    comparison_results: Optional[Dict[str, Any]] = Field(None, description="Candidate-by-candidate OCR extraction comparison when compare_all is used")
    error: Optional[str] = Field(None, description="Error or reason if unavailable")
    setup_instructions: Optional[Dict[str, Any]] = Field(None, description="Setup instructions if OCR unavailable")


# -------------------------------------------------------------
# Restoration & Damage Detection Schemas
# -------------------------------------------------------------
class RestorationRequest(BaseModel):
    session_id: str = Field(..., description="Session ID containing uploaded manuscript image")
    device_preference: str = Field("auto", description="Compute device: auto, cpu, or cuda")


class RestorationResponse(BaseModel):
    session_id: str = Field(..., description="Processing session ID")
    timestamp: str = Field(..., description="Execution timestamp")
    device_info: Dict[str, Any] = Field(..., description="Hardware device information")
    timings_ms: Dict[str, float] = Field(..., description="Per-module execution latencies in ms")
    model_readiness: Dict[str, str] = Field(..., description="Status of models (READY vs UNTRAINED)")
    damage_detection: Dict[str, Any] = Field(..., description="Damage localization metrics and percentage")
    restoration: Dict[str, Any] = Field(..., description="Restoration metrics and output details")
    candidate_selection: Optional[Dict[str, Any]] = Field(None, description="Evidence-based candidate selection decision and comparative metrics")
    text_protection: Optional[Dict[str, Any]] = Field(None, description="High-resolution ink stroke protection metrics")
    artifacts: Dict[str, Optional[str]] = Field(..., description="Relative or absolute URLs to output images")


# -------------------------------------------------------------
# Damaged-Text Prediction Schemas
# -------------------------------------------------------------
class TextPredictionRequest(BaseModel):
    sentence: Optional[str] = Field(None, description="Full sentence containing placeholder or uncertain word (e.g. 'The king [MISSING] the temple.')")
    left_context: Optional[str] = Field("", description="Readable text preceding the damaged span")
    right_context: Optional[str] = Field("", description="Readable text following the damaged span")
    uncertain_span: Optional[str] = Field(None, description="The specific damaged or uncertain word/phrase (e.g. [MISSING], [DAMAGED], visited, or partial word)")
    full_text: Optional[str] = Field(None, description="Current full OCR-extracted or edited text")
    span_index: Optional[int] = Field(None, description="Zero-based token index of the uncertain span in full_text")
    max_candidates: int = Field(4, ge=1, le=10, description="Maximum candidate predictions to return")
    allow_multi_word: bool = Field(True, description="Whether to consider short multi-word completions for damaged regions")


class TextPredictionCandidate(BaseModel):
    word: str = Field(..., description="Predicted candidate word or phrase")
    score: float = Field(..., description="Relative n-gram likelihood / frequency score")
    evidence: Optional[str] = Field(None, description="Model explanation / probability breakdown")
    reconstructed_sentence: Optional[str] = Field(None, description="Complete sentence with this candidate applied")


class TextPredictionResponse(BaseModel):
    status: str = Field(..., description="PREDICTED, INSUFFICIENT_CONTEXT, or UNAVAILABLE")
    prediction: Optional[str] = Field(None, description="Top predicted word or phrase")
    confidence_score: Optional[float] = Field(None, description="Normalized confidence score for top prediction (0.0 to 1.0)")
    original_sentence: Optional[str] = Field(None, description="Original input sentence before prediction")
    reconstructed_sentence: Optional[str] = Field(None, description="Complete reconstructed sentence with top prediction applied")
    candidates: List[TextPredictionCandidate] = Field([], description="Ranked alternative candidates")
    is_speculative: bool = Field(True, description="Always True: predictions are never verified historical facts")
    uncertainty_label: str = Field("SPECULATIVE_CONTEXTUAL_COMPLETION", description="Scientific uncertainty indicator")
    disclaimer: str = Field(
        "Predicted text is statistically estimated from linguistic context; it must never be presented as verified historical content without independent manuscript proof.",
        description="Mandatory scientific disclaimer",
    )
    surrounding_context: Dict[str, str] = Field(..., description="Context strings used for prediction")
    detected_script: Optional[str] = Field(None, description="Detected script: devanagari, latin, or all")


# -------------------------------------------------------------
# History & Detail Schemas
# -------------------------------------------------------------
class SessionSummary(BaseModel):
    session_id: str
    created_at: str
    status: str
    filename: Optional[str] = None
    has_ocr: bool = False
    has_restoration: bool = False
    damaged_percentage: Optional[float] = None


class SessionDetailResponse(BaseModel):
    session_id: str
    user_id: Optional[str] = None
    created_at: str
    updated_at: str
    status: str
    original_metadata: Dict[str, Any]
    artifacts: Dict[str, Optional[str]]
    ocr_result: Optional[Dict[str, Any]] = None
    restoration_result: Optional[Dict[str, Any]] = None
    text_predictions: Optional[List[Dict[str, Any]]] = None
    persistence_info: Dict[str, Any]
