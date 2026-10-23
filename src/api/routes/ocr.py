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


@router.get("", response_model=dict)
@router.get("/status", response_model=dict)
def get_ocr_status():
    """
    Returns deep diagnostic report of OCR engine availability,
    executable path, version, and language capabilities.
    Enables frontend to audit engine readiness prior to execution.
    """
    return ocr_engine.get_status(force_refresh=True)


@router.post("", response_model=OCRResponse)
def run_ocr_on_session(
    req: OCRRequest,
    current_user: dict = Depends(get_optional_user),
):
    """
    Executes OCR text extraction on the manuscript image of an existing session.
    Prioritizes restored folio (restored_unet.png) over original degraded image when available,
    maximizing character recognition quality.
    Honoring scientific integrity: if Tesseract is unavailable or missing language packs,
    reports the diagnostic status and setup instructions without fabricating text.
    """
    if not req.session_id or not req.session_id.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="session_id is required to run OCR on an existing session. If you have an image file, use direct file upload.",
        )

    clean_session_id = req.session_id.strip()
    session = db_manager.get_session(clean_session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{clean_session_id}' not found. Please provide a valid restored folio session ID or use direct file upload.",
        )

    # Ownership enforcement: private sessions can only be accessed by their owner
    owner_id = session.get("user_id")
    if owner_id and owner_id != "anonymous":
        if not current_user or str(current_user["_id"]) != owner_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied: You do not have permission to access OCR for this session.",
            )

    session_dir = settings.SESSIONS_STORAGE_DIR / clean_session_id
    if not session_dir.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session storage directory for '{clean_session_id}' not found on server.",
        )

    # Preferred image selection: restored image preferred if present, otherwise original
    restored_path = session_dir / "restored_unet.png"
    baseline_path = session_dir / "restored_baseline.png"
    model_path = session_dir / "restored_model.png"
    original_path = session_dir / "original.png"

    target_choice = (getattr(req, "target_image", "restored") or "restored").lower().strip()

    # 1. Multi-candidate comparative OCR evaluation
    comparison_results = None
    preprocessing_mode = getattr(req, "preprocessing", "none") or "none"

    if target_choice == "compare_all":
        comparison_results = {}
        candidate_paths = {
            "original": original_path,
            "baseline": baseline_path if baseline_path.exists() else original_path,
            "model": model_path if model_path.exists() else original_path,
            "selected_restoration": restored_path if restored_path.exists() else original_path,
        }
        for c_name, c_path in candidate_paths.items():
            if c_path.exists():
                c_res = ocr_engine.extract_text(c_path, lang=req.language, preprocessing=preprocessing_mode)
                comparison_results[c_name] = {
                    "extracted_text": c_res["extracted_text"],
                    "character_count": c_res["character_count"],
                    "word_count": c_res["word_count"],
                    "mean_confidence": c_res.get("mean_confidence"),
                    "status": c_res["status"],
                    "image_file": c_path.name,
                    "preprocessing_applied": c_res.get("preprocessing_applied", "none"),
                }

        # Main result defaults to selected restoration
        img_path = restored_path if restored_path.exists() else original_path
        image_source = img_path.name
    elif target_choice in ("baseline", "traditional_baseline") and baseline_path.exists():
        img_path = baseline_path
        image_source = "restored_baseline.png"
    elif target_choice in ("model", "model_guided") and model_path.exists():
        img_path = model_path
        image_source = "restored_model.png"
    elif target_choice == "original" and original_path.exists():
        img_path = original_path
        image_source = "original.png"
    elif restored_path.exists():
        img_path = restored_path
        image_source = "restored_unet.png"
    elif original_path.exists():
        img_path = original_path
        image_source = "original.png"
    else:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Neither restored folio nor original image found on server for session '{clean_session_id}'.",
        )

    # Run OCR extraction
    ocr_result = ocr_engine.extract_text(img_path, lang=req.language, preprocessing=preprocessing_mode)
    ocr_result["session_id"] = clean_session_id
    ocr_result["image_source"] = image_source
    if comparison_results:
        ocr_result["comparison_results"] = comparison_results

    # Update session in DB
    db_manager.update_session(clean_session_id, {"ocr_result": ocr_result, "status": "OCR_PROCESSED"})

    return OCRResponse(
        session_id=clean_session_id,
        status=ocr_result["status"],
        extracted_text=ocr_result["extracted_text"],
        processing_time_ms=ocr_result["processing_time_ms"],
        language_requested=ocr_result["language_requested"],
        language_used=ocr_result.get("language_used"),
        character_count=ocr_result["character_count"],
        word_count=ocr_result["word_count"],
        mean_confidence=ocr_result.get("mean_confidence"),
        is_authentic_ocr=ocr_result["is_authentic_ocr"],
        image_source=image_source,
        preprocessing_applied=ocr_result.get("preprocessing_applied", "none"),
        comparison_results=comparison_results,
        error=ocr_result.get("error"),
        setup_instructions=ocr_result.get("setup_instructions"),
    )


@router.post("/direct", response_model=OCRResponse)
async def run_direct_ocr(
    file: UploadFile = File(..., description="Document image file for instant OCR extraction"),
    language: str = Form("san+hin", description="Tesseract language code"),
    preprocessing: str = Form("none", description="Preprocessing variant: 'none', 'auto', 'contrast', 'illumination', 'binarize', 'upscale'"),
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

    ocr_result = ocr_engine.extract_text(pil_img, lang=language, preprocessing=preprocessing)

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
        image_source="direct_upload",
        preprocessing_applied=ocr_result.get("preprocessing_applied", "none"),
        error=ocr_result.get("error"),
        setup_instructions=ocr_result.get("setup_instructions"),
    )
