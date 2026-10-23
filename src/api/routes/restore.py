"""
Restoration & Damage Detection Endpoints for Past2Pixel API.
Executes the deep learning pipeline (U-Net Restoration, Damage Detection, Baseline Autoencoder).
"""
from pathlib import Path
from fastapi import APIRouter, HTTPException, status, Depends
from ...core.config import settings
from ...pipeline.restoration_engine import RestorationEngine
from ...auth.dependencies import get_optional_user
from ...db.mongodb import db_manager
from ..schemas import RestorationRequest, RestorationResponse

router = APIRouter(prefix="/restore", tags=["Restoration & Damage Detection"])

# Shared engine instance for API handling
engine_instance = None


def get_restoration_engine(device_pref: str = "auto") -> RestorationEngine:
    global engine_instance
    if engine_instance is None:
        engine_instance = RestorationEngine(device_preference=device_pref)
    return engine_instance


@router.post("", response_model=RestorationResponse)
def run_restoration_pipeline(
    req: RestorationRequest,
    current_user: dict = Depends(get_optional_user),
):
    """
    Executes damage localization and restoration on the session's manuscript image.
    Generates pixel-level damage masks, visual heatmaps, and enhanced restored folios.
    Explicitly reports whether models are running on trained checkpoints or untrained weights.
    """
    session = db_manager.get_session(req.session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{req.session_id}' not found.",
        )

    # Ownership enforcement: private sessions can only be restored by their owner
    owner_id = session.get("user_id")
    if owner_id and owner_id != "anonymous":
        if not current_user or str(current_user["_id"]) != owner_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied: You do not have permission to restore this session.",
            )

    session_dir = settings.SESSIONS_STORAGE_DIR / req.session_id
    img_path = session_dir / "original.png"
    if not img_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Original image for session '{req.session_id}' was not found on disk.",
        )

    # Initialize engine and run processing
    engine = get_restoration_engine(req.device_preference)

    try:
        report = engine.process_image(
            input_image=img_path,
            output_dir=session_dir,
            session_id=req.session_id,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Restoration engine pipeline failed during execution: {str(e)}",
        )

    # Construct static URLs for generated artifacts
    sid = req.session_id
    artifacts = {
        "original": f"/static/sessions/{sid}/original.png",
        "restored": f"/static/sessions/{sid}/restored_unet.png",
        "baseline": f"/static/sessions/{sid}/restored_baseline.png",
        "model": f"/static/sessions/{sid}/restored_model.png",
        "text_mask": f"/static/sessions/{sid}/text_mask.png",
        "damage_overlay": f"/static/sessions/{sid}/damage_overlay.png",
        "damage_mask": f"/static/sessions/{sid}/damage_mask.png",
    }

    # Save update to database
    db_update = {
        "status": "RESTORATION_COMPLETED",
        "artifacts": artifacts,
        "restoration_result": {
            "timings_ms": report["timings_ms"],
            "model_readiness": report["model_readiness"],
            "damage_detection": report["damage_detection_result"],
            "restoration": report["restoration_results"],
            "candidate_selection": report.get("candidate_selection"),
            "text_protection": report.get("text_protection_metrics"),
        },
    }
    db_manager.update_session(req.session_id, db_update)

    return RestorationResponse(
        session_id=req.session_id,
        timestamp=report["timestamp"],
        device_info=report["device_info"],
        timings_ms=report["timings_ms"],
        model_readiness=report["model_readiness"],
        damage_detection=report["damage_detection_result"],
        restoration=report["restoration_results"],
        candidate_selection=report.get("candidate_selection"),
        text_protection=report.get("text_protection_metrics"),
        artifacts=artifacts,
    )
