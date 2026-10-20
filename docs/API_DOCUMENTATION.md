# Past2Pixel — REST API Specification & Endpoint Documentation

The Past2Pixel backend is built with **FastAPI** and provides high-performance, asynchronous endpoints for manuscript image upload, deep-learning restoration, damage localization, OCR text recognition, and context-aware text prediction.

Interactive Swagger/OpenAPI documentation is available at:
- **Swagger UI:** `http://127.0.0.1:8000/docs`
- **ReDoc:** `http://127.0.0.1:8000/redoc`
- **OpenAPI Schema:** `http://127.0.0.1:8000/openapi.json`

---

## 1. Authentication & Security

Protected endpoints require a signed JSON Web Token (JWT) provided in the HTTP `Authorization` header:
```http
Authorization: Bearer <access_token>
```

Tokens are issued upon successful registration or login and expire according to `ACCESS_TOKEN_EXPIRE_MINUTES` (default: 1440 minutes / 24 hours).

---

## 2. Endpoints Overview

| Category | Method | Endpoint | Auth Required | Description |
| :--- | :--- | :--- | :--- | :--- |
| **Diagnostics** | `GET` | `/api/health` | No | Basic health check ping |
| **Diagnostics** | `GET` | `/api/readiness` | No | Comprehensive hardware, model, OCR, and DB readiness audit |
| **Auth** | `POST` | `/api/auth/register` | No | Register a new user account with bcrypt hashing |
| **Auth** | `POST` | `/api/auth/login` | No | Authenticate user credentials and receive JWT Bearer token |
| **Auth** | `GET` | `/api/auth/me` | **Yes** | Retrieve authenticated user profile |
| **Upload** | `POST` | `/api/upload` | Optional | Validate and upload manuscript image; creates session |
| **OCR** | `POST` | `/api/ocr` | Optional | Execute Tesseract OCR extraction on a session manuscript |
| **OCR** | `POST` | `/api/ocr/direct` | No | Direct multipart image upload for instant OCR text extraction |
| **Restoration** | `POST` | `/api/restore` | Optional | Run U-Net restoration & damage segmentation on session image |
| **Prediction** | `POST` | `/api/predict-text` | Optional | Statistical context-aware missing-text prediction |
| **Sessions** | `GET` | `/api/sessions` | **Yes** | List all processing sessions belonging to the authenticated user |
| **Sessions** | `GET` | `/api/sessions/{session_id}` | Optional* | Retrieve detailed processing session report and artifact links |
| **Sessions** | `DELETE` | `/api/sessions/{session_id}` | **Yes** | Permanently delete session and associated disk artifacts |

*\*Unauthenticated users can view anonymous sessions; user-owned sessions are restricted to their owner.*

---

## 3. Detailed Endpoint Specifications

### 3.1 `GET /api/health`
Returns service availability status.
```json
{
  "status": "OK",
  "version": "1.0.0",
  "timestamp": "2026-10-20T16:44:07.393761+00:00",
  "service": "Past2Pixel AI API"
}
```

---

### 3.2 `GET /api/readiness`
Deep diagnostic check of all underlying components.
- **Compute:** CPU logical cores, CUDA GPU device name and memory.
- **Models:** Individual lifecycle statuses (`UNTRAINED`, `CHECKPOINT_SAVED`, `READY_FOR_INFERENCE`).
- **OCR:** Tesseract detection, binary path, and available language packs.
- **Database:** MongoDB Atlas connection status and persistence mode.

---

### 3.3 `POST /api/auth/register`
Creates a new user profile.
- **Request Body:**
```json
{
  "email": "student@university.edu",
  "username": "aiml_student",
  "password": "StrongPassword123!"
}
```
- **Response (201 Created):**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in_minutes": 1440,
  "user": {
    "id": "usr_0001_1792515025",
    "email": "student@university.edu",
    "username": "aiml_student",
    "created_at": "2026-10-20T16:44:07.000000+00:00"
  }
}
```

---

### 3.4 `POST /api/upload`
Uploads a manuscript image (`multipart/form-data`).
- **Validations:** Format (`.png`, `.jpg`, `.jpeg`, `.tiff`, `.bmp`), decodability, size limit (15 MB).
- **Response (201 Created):**
```json
{
  "session_id": "p2p_79ce119fea00_1792515025",
  "filename": "original.png",
  "original_filename": "vedic_folio_01.png",
  "width": 1606,
  "height": 146,
  "file_size_bytes": 142080,
  "image_url": "/static/sessions/p2p_79ce119fea00_1792515025/original.png",
  "created_at": "2026-10-20T16:45:00.000000+00:00"
}
```

---

### 3.5 `POST /api/ocr`
Extracts text from the session's manuscript image.
- **Request Body:**
```json
{
  "session_id": "p2p_79ce119fea00_1792515025",
  "language": "san+hin+eng"
}
```
- **Response (200 OK — When Tesseract Installed):**
```json
{
  "session_id": "p2p_79ce119fea00_1792515025",
  "status": "SUCCESS",
  "extracted_text": "अग्निमीळे पुरोहितं यज्ञस्य देवमृत्विजम्",
  "processing_time_ms": 112.4,
  "language_requested": "san+hin+eng",
  "language_used": "san",
  "character_count": 38,
  "word_count": 5,
  "mean_confidence": 92.5,
  "is_authentic_ocr": true,
  "error": null,
  "setup_instructions": null
}
```
- **Response (200 OK — When Tesseract Unavailable):**
Honest reporting with diagnostic setup instructions. Never fabricates fake text.

---

### 3.6 `POST /api/restore`
Executes deep learning restoration and damage localization.
- **Request Body:**
```json
{
  "session_id": "p2p_79ce119fea00_1792515025",
  "device_preference": "auto"
}
```
- **Response (200 OK):**
```json
{
  "session_id": "p2p_79ce119fea00_1792515025",
  "timestamp": "2026-10-20 16:50:00 UTC",
  "device_info": { "device_type": "cuda", "device_name": "NVIDIA GeForce RTX 5060 Laptop GPU" },
  "timings_ms": {
    "preprocessing": 4.2,
    "damage_detection": 18.5,
    "unet_restoration": 22.1,
    "autoencoder_baseline": 12.0,
    "total_pipeline": 56.8
  },
  "model_readiness": {
    "unet_restoration": "UNTRAINED_RANDOM_WEIGHTS",
    "damage_detector": "UNTRAINED_RANDOM_WEIGHTS",
    "autoencoder_baseline": "UNTRAINED_RANDOM_WEIGHTS"
  },
  "damage_detection": {
    "damaged_pixel_percentage": 24.5,
    "evaluation": { "status": "UNAVAILABLE", "reason": "No ground-truth reference mask provided for upload." }
  },
  "restoration": {
    "unet_restoration_metrics": { "status": "UNAVAILABLE", "reason": "Reference-Free evaluation: no clean reference provided." },
    "autoencoder_baseline_metrics": { "status": "UNAVAILABLE", "reason": "Reference-Free evaluation: no clean reference provided." }
  },
  "artifacts": {
    "original": "/static/sessions/p2p_79ce119fea00_1792515025/original.png",
    "restored": "/static/sessions/p2p_79ce119fea00_1792515025/restored_unet.png",
    "damage_overlay": "/static/sessions/p2p_79ce119fea00_1792515025/damage_overlay.png",
    "damage_mask": "/static/sessions/p2p_79ce119fea00_1792515025/damage_mask.png"
  }
}
```

---

### 3.7 `POST /api/predict-text`
Predicts plausible text completions for obscured or damaged manuscript regions using surrounding context.
- **Request Body:**
```json
{
  "left_context": "धर्मक्षेत्रे",
  "right_context": "समवेता",
  "max_candidates": 3
}
```
- **Response (200 OK):**
```json
{
  "status": "PREDICTED",
  "prediction": "कुरुक्षेत्रे",
  "candidates": [
    { "word": "कुरुक्षेत्रे", "score": 0.852 },
    { "word": "तीर्थे", "score": 0.104 }
  ],
  "is_speculative": true,
  "uncertainty_label": "SPECULATIVE_CONTEXTUAL_COMPLETION",
  "disclaimer": "Contextual prediction is linguistically estimated from surrounding text and corpus statistics; it must never be presented as verified historical content without independent manuscript confirmation.",
  "surrounding_context": {
    "left_context": "धर्मक्षेत्रे",
    "right_context": "समवेता"
  }
}
```

---

### 3.8 `GET /api/sessions`
Returns history of restoration sessions owned by the authenticated user.
- **Headers:** `Authorization: Bearer <access_token>`
- **Response:** Array of session summaries with timestamps, damage percentages, and status flags.

---

### 3.9 `DELETE /api/sessions/{session_id}`
Permanently deletes session metadata from database and cleans up image files on disk. Enforces user ownership.
