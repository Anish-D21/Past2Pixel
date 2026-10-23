"""
FastAPI Application Entrypoint for Past2Pixel.
Assembles routers, middleware, static file mounting, and custom error handlers.
"""
from pathlib import Path
from contextlib import asynccontextmanager
import logging
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from ..core.config import settings
from .routes import (
    health_router,
    auth_router,
    upload_router,
    ocr_router,
    restore_router,
    predict_router,
    sessions_router,
)

logger = logging.getLogger("past2pixel.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle startup and shutdown handler."""
    settings.ensure_directories()
    logger.info("Past2Pixel API started. Runtime directories ensured.")
    yield
    logger.info("Past2Pixel API shutting down.")


def create_app() -> FastAPI:
    """Factory creating and configuring the FastAPI application."""
    settings.ensure_directories()

    app = FastAPI(
        title="Past2Pixel — Ancient Manuscript AI API",
        description=(
            "Deep Learning & Computer Vision Backend for Ancient Manuscript Damage Detection, "
            "Restoration, OCR Text Extraction, and Context-Aware Linguistic Prediction."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # 1. CORS Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 2. Mount Static Files for Session Artifacts
    sessions_dir = settings.SESSIONS_STORAGE_DIR
    sessions_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/static/sessions", StaticFiles(directory=str(sessions_dir)), name="sessions")

    # 3. Include Routers
    api_prefix = "/api"
    app.include_router(health_router)  # /health and /readiness
    app.include_router(health_router, prefix=api_prefix)  # /api/health and /api/readiness
    app.include_router(auth_router, prefix=api_prefix)
    app.include_router(upload_router, prefix=api_prefix)
    app.include_router(ocr_router, prefix=api_prefix)
    app.include_router(restore_router, prefix=api_prefix)
    app.include_router(predict_router, prefix=api_prefix)
    app.include_router(sessions_router, prefix=api_prefix)

    @app.get("/", tags=["Root"])
    def root():
        """Root welcome endpoint providing service status and quick links."""
        return {
            "service": "Past2Pixel — Ancient Manuscript AI API",
            "status": "ONLINE",
            "version": "1.0.0",
            "docs_url": "/docs",
            "redoc_url": "/redoc",
            "health_url": "/health",
            "readiness_url": "/readiness",
            "frontend_instructions": "To launch the user interface, run 'cd frontend && npm run dev' and navigate to http://localhost:5173",
        }

    # 4. Exception Handlers (Security: Do not leak internal stack traces)
    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": True,
                "status_code": exc.status_code,
                "detail": exc.detail,
            },
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error": True,
                "status_code": 422,
                "detail": "Input validation error",
                "validation_errors": exc.errors(),
            },
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        logger.error("Unhandled server exception on %s: %s", request.url.path, exc, exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": True,
                "status_code": 500,
                "detail": "An internal server error occurred. Please consult backend logs.",
            },
        )

    return app


# Default application instance for uvicorn
app = create_app()
