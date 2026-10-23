"""
Optical Character Recognition (OCR) Endpoints for Past2Pixel API.
Performs real Tesseract extraction on uploaded or restored manuscript folios.
"""
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, HTTPException, status, Depends, UploadFile, File, Form
from PIL import Image
import io

from ...core.config import settings
from ...ocr.ocr_engine import ocr_engine
from ...auth.dependencies import get_optional_user
from ...db.mongodb import db_manager
from ..schemas import OCRResponse, OCRRequest

router = APIRouter(prefix="/ocr", tags=["OCR & Text Extraction"])


@router.post("", response_model=OCRResponse)
def run_ocr_on_session(
    req: OCRRequest,
    current_user: dict = Depends(get_optional_user),
):
    """
    Executes OCR text extraction on the manuscript image of an existing session.
    Honoring scientific integrity: if Tesseract is unavailable or missing language packs,
    reports the diagnostic status and setup instructions without fabricating text.
    """
    if not req.session_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="session_id is required to run OCR on an existing session.",
        )

    session = db_manager.get_session(req.session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{req.session_id}' not found.",
        )

    # Ownership enforcement: private sessions can only be accessed by their owner
    owner_id = session.get("user_id")
    if owner_id and owner_id != "anonymous":
        if not current_user or str(current_user["_id"]) != owner_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied: You do not have permission to access OCR for this session.",
            )

    session_dir = settings.SESSIONS_STORAGE_DIR / req.session_id
    img_path = session_dir / "original.png"
    if not img_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Original image for session '{req.session_id}' not found on server.",
        )

    # Run OCR extraction
    ocr_result = ocr_engine.extract_text(img_path, lang=req.language)
    ocr_result["session_id"] = req.session_id

    # Update session in DB
    db_manager.update_session(req.session_id, {"ocr_result": ocr_result, "status": "OCR_PROCESSED"})

    return OCRResponse(
        session_id=req.session_id,
        status=ocr_result["status"],
        extracted_text=ocr_result["extracted_text"],
        processing_time_ms=ocr_result["processing_time_ms"],
        language_requested=ocr_result["language_requested"],
        language_used=ocr_result.get("language_used"),
        character_count=ocr_result["character_count"],
        word_count=ocr_result["word_count"],
        mean_confidence=ocr_result.get("mean_confidence"),
        is_authentic_ocr=ocr_result["is_authentic_ocr"],
        error=ocr_result.get("error"),
        setup_instructions=ocr_result.get("setup_instructions"),
    )


@router.post("/direct", response_model=OCRResponse)
async def run_direct_ocr(
    file: UploadFile = File(..., description="Document image file for instant OCR extraction"),
    language: str = Form("san+hin+eng", description="Tesseract language code"),
):
    """
    Directly extracts machine-readable text from an uploaded image without requiring a session.
    """
    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty.")

    try:
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid image file: {e}")

    ocr_result = ocr_engine.extract_text(pil_img, lang=language)

    return OCRResponse(
        session_id=None,
        status=ocr_result["status"],
        extracted_text=ocr_result["extracted_text"],
        processing_time_ms=ocr_result["processing_time_ms"],
        language_requested=ocr_result["language_requested"],
        language_used=ocr_result.get("language_used"),
        character_count=ocr_result["character_count"],
        word_count=ocr_result["word_count"],
        mean_confidence=ocr_result.get("mean_confidence"),
        is_authentic_ocr=ocr_result["is_authentic_ocr"],
        error=ocr_result.get("error"),
        setup_instructions=ocr_result.get("setup_instructions"),
    )
