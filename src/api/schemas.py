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
    language: str = Field("san+hin+eng", description="Language pack codes for Tesseract")
    target_image: Optional[str] = Field("restored", description="Target image: 'restored' (uses restored_unet.png if present) or 'original'")


class OCRResponse(BaseModel):
    session_id: Optional[str] = Field(None, description="Associated session ID")
    status: str = Field(..., description="SUCCESS, UNAVAILABLE, or FAILED")
    extracted_text: str = Field("", description="Raw Unicode extracted text from visible characters")
    processing_time_ms: float = Field(0.0, description="Measured OCR duration in milliseconds")
    language_requested: str = Field("san+hin+eng", description="Requested language string")
    language_used: Optional[str] = Field(None, description="Actual language pack used")
    character_count: int = Field(0, description="Total characters extracted")
    word_count: int = Field(0, description="Total words extracted")
    mean_confidence: Optional[float] = Field(None, description="Mean character confidence percentage")
    is_authentic_ocr: bool = Field(False, description="True if extracted from real engine, False if unavailable")
    image_source: Optional[str] = Field(None, description="Image source used for OCR ('restored_unet.png' or 'original.png')")
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
    artifacts: Dict[str, Optional[str]] = Field(..., description="Relative or absolute URLs to output images")


# -------------------------------------------------------------
# Damaged-Text Prediction Schemas
# -------------------------------------------------------------
class TextPredictionRequest(BaseModel):
    left_context: str = Field(..., description="Readable text preceding the damaged span")
    right_context: str = Field(..., description="Readable text following the damaged span")
    max_candidates: int = Field(3, ge=1, le=10, description="Maximum candidate predictions to return")


class TextPredictionCandidate(BaseModel):
    word: str = Field(..., description="Predicted candidate word")
    score: float = Field(..., description="Relative n-gram likelihood / frequency score")


class TextPredictionResponse(BaseModel):
    status: str = Field(..., description="PREDICTED, INSUFFICIENT_CONTEXT, or UNAVAILABLE")
    prediction: Optional[str] = Field(None, description="Top predicted word/token")
    candidates: List[TextPredictionCandidate] = Field([], description="Ranked alternative candidates")
    is_speculative: bool = Field(True, description="Always True: predictions are never verified historical facts")
    uncertainty_label: str = Field("SPECULATIVE_CONTEXTUAL_COMPLETION", description="Scientific uncertainty indicator")
    disclaimer: str = Field(
        "Predicted text is statistically estimated from linguistic context; it must never be presented as verified historical content without independent manuscript proof.",
        description="Mandatory scientific disclaimer",
    )
    surrounding_context: Dict[str, str] = Field(..., description="Context strings used for prediction")


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
