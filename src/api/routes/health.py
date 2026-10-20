"""
Health and Readiness Diagnostic Endpoints for Past2Pixel API.
"""
from datetime import datetime, timezone
from fastapi import APIRouter
import torch

from ...core.config import settings
from ...core.device import get_device, get_device_info
from ...models.model_registry import model_registry
from ...ocr.ocr_engine import ocr_engine
from ...db.mongodb import db_manager
from ..schemas import HealthResponse, ReadinessResponse

router = APIRouter(tags=["Health & Diagnostics"])


@router.get("/health", response_model=HealthResponse)
def get_health():
    """Simple ping endpoint for load balancers and container orchestrators."""
    return HealthResponse(
        status="OK",
        version="1.0.0",
        timestamp=datetime.now(timezone.utc).isoformat(),
        service="Past2Pixel AI API",
    )


@router.get("/readiness", response_model=ReadinessResponse)
def get_readiness():
    """
    Comprehensive diagnostic endpoint inspecting:
    - Compute hardware (CPU logical cores, CUDA GPU device)
    - PyTorch model checkpoint readiness states
    - Tesseract OCR engine detection and available language packs
    - MongoDB Atlas / local persistence storage status
    """
    device = get_device("auto")
    device_info = get_device_info(device)
    models_status = model_registry.get_all_statuses()
    ocr_status = ocr_engine.get_status()
    db_status = db_manager.get_status()

    # Determine overall readiness
    # If DB is functioning (Atlas or local fallback) and compute is verified, service is functional
    is_ready = db_status["status"] in ["CONNECTED", "LOCAL_PERSISTENCE"]
    overall = "READY" if is_ready else "DEGRADED"

    return ReadinessResponse(
        status=overall,
        compute=device_info,
        models=models_status,
        ocr=ocr_status,
        database=db_status,
    )
