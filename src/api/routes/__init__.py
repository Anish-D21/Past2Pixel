"""API Routes Package for Past2Pixel."""
from .health import router as health_router
from .auth import router as auth_router
from .upload import router as upload_router
from .ocr import router as ocr_router
from .restore import router as restore_router
from .predict import router as predict_router
from .sessions import router as sessions_router

__all__ = [
    "health_router",
    "auth_router",
    "upload_router",
    "ocr_router",
    "restore_router",
    "predict_router",
    "sessions_router",
]
