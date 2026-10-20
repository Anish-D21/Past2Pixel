"""
Manuscript Image Upload Endpoint for Past2Pixel API.
Validates file formats, sizes, and pixel decodability, generating secure processing sessions.
"""
from pathlib import Path
from datetime import datetime, timezone
import uuid
import time
import io
from PIL import Image
from fastapi import APIRouter, UploadFile, File, HTTPException, status, Depends

from ...core.config import settings
from ...auth.dependencies import get_optional_user
from ...db.mongodb import db_manager
from ..schemas import UploadResponse

router = APIRouter(tags=["Manuscript Upload & Processing"])


@router.post("/upload", response_model=UploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_manuscript(
    file: UploadFile = File(..., description="Manuscript document image (PNG, JPG, TIFF, BMP)"),
    current_user: dict = Depends(get_optional_user),
):
    """
    Validates uploaded document image, creates an isolated processing session,
    and stores the original image securely on disk with unique sanitized filenames.
    """
    settings.ensure_directories()

    # 1. Validate file extension
    original_filename = file.filename or "uploaded_image.png"
    suffix = Path(original_filename).suffix.lower()
    if suffix not in settings.ALLOWED_IMAGE_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file extension '{suffix}'. Allowed formats: {', '.join(settings.ALLOWED_IMAGE_EXTENSIONS)}",
        )

    # 2. Read contents and check file size limit
    max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    contents = await file.read()
    file_size = len(contents)

    if file_size == 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty (0 bytes).")
    if file_size > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"File size exceeds maximum allowed limit of {settings.MAX_UPLOAD_SIZE_MB} MB.",
        )

    # 3. Validate image decodability with PIL
    try:
        pil_img = Image.open(io.BytesIO(contents))
        pil_img.verify()  # verify integrity
        # Re-open after verify() closes the file
        pil_img = Image.open(io.BytesIO(contents))
        width, height = pil_img.size
        img_format = pil_img.format or "PNG"
        img_mode = pil_img.mode
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Uploaded file is corrupted or not a valid image: {str(e)}",
        )

    # 4. Generate unique session ID and directory
    session_id = f"p2p_{uuid.uuid4().hex[:12]}_{int(time.time())}"
    session_dir = settings.SESSIONS_STORAGE_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    # Save original image as original.png
    safe_filename = "original.png"
    original_dest = session_dir / safe_filename
    pil_img.convert("RGB").save(original_dest, format="PNG")

    # 5. Persist initial session record
    user_id = str(current_user["_id"]) if current_user else "anonymous"
    created_at = datetime.now(timezone.utc).isoformat()

    session_doc = {
        "session_id": session_id,
        "user_id": user_id,
        "status": "UPLOADED",
        "created_at": created_at,
        "original_metadata": {
            "original_filename": original_filename,
            "width": width,
            "height": height,
            "format": img_format,
            "mode": img_mode,
            "file_size_bytes": file_size,
        },
        "artifacts": {
            "original": f"/static/sessions/{session_id}/{safe_filename}",
            "restored": None,
            "damage_overlay": None,
            "damage_mask": None,
        },
        "ocr_result": None,
        "restoration_result": None,
        "text_predictions": [],
    }

    db_manager.create_session(session_doc)

    return UploadResponse(
        session_id=session_id,
        filename=safe_filename,
        original_filename=original_filename,
        width=width,
        height=height,
        file_size_bytes=file_size,
        image_url=f"/static/sessions/{session_id}/{safe_filename}",
        created_at=created_at,
    )
