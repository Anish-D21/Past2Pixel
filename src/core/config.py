"""
Centralized Configuration Module for Past2Pixel.
Uses Python dataclasses with environment variable overrides.
Zero third-party dependency for maximum portability across CPU and GPU systems.
"""
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import os

# Project root directory derived strictly from this file's canonical location
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def load_project_env(env_path: Optional[Path] = None, override: bool = False) -> bool:
    """
    Loads environment variables from the project root .env file.
    
    Features:
    - Independent of current working directory (always anchors to PROJECT_ROOT).
    - Preserves externally supplied environment variables by default (override=False).
    - Leverages python-dotenv when installed, with a robust fallback parser if missing.
    """
    target = Path(env_path) if env_path else (PROJECT_ROOT / ".env")
    if not target.is_file():
        return False

    try:
        from dotenv import load_dotenv
        return bool(load_dotenv(dotenv_path=target, override=override))
    except ImportError:
        pass

    # Built-in fallback parser if python-dotenv is not installed
    loaded_any = False
    try:
        with open(target, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip().strip("'\"")
                if key and (override or key not in os.environ):
                    os.environ[key] = val
                    loaded_any = True
        return loaded_any
    except Exception:
        return False


# Automatically load .env from project root at module import time
load_project_env(override=False)


@dataclass
class Settings:
    # Base paths
    BASE_DIR: Path = field(default_factory=lambda: PROJECT_ROOT)
    
    @property
    def DATA_DIR(self) -> Path:
        p = os.getenv("DATA_DIR")
        return Path(p) if p else self.BASE_DIR / "data"

    @property
    def RAW_DATA_DIR(self) -> Path:
        return self.DATA_DIR / "raw"

    @property
    def VEDO_DATASET_DIR(self) -> Path:
        p = os.getenv("VEDO_DATASET_DIR")
        if p:
            return Path(p)
        for candidate in ["VedoOCR", "VedOCR", "vedo_ocr"]:
            cand = self.RAW_DATA_DIR / candidate
            if cand.exists():
                return cand
        return self.RAW_DATA_DIR / "VedoOCR"

    @property
    def VEDO_CONTENT_DIR(self) -> Path:
        """Resolves to the directory containing line_crops, master.tsv, etc."""
        base = self.VEDO_DATASET_DIR
        nested = base / "VedOCR_kaggle_release"
        return nested if nested.exists() else base

    @property
    def SANSKRIT_HINDI_DATASET_DIR(self) -> Path:
        p = os.getenv("SANSKRIT_HINDI_DATASET_DIR")
        if p:
            return Path(p)
        for candidate in ["sanskrit_hindi_ocr_vqa", "Sanskrit & Hindi OCR + VQA", "sanskrit_hindi_vqa"]:
            cand = self.RAW_DATA_DIR / candidate
            if cand.exists():
                return cand
        return self.RAW_DATA_DIR / "sanskrit_hindi_ocr_vqa"

    @property
    def SANSKRIT_HINDI_CONTENT_DIR(self) -> Path:
        """Resolves to the directory containing train/val/test splits."""
        base = self.SANSKRIT_HINDI_DATASET_DIR
        nested = base / "Dataset"
        return nested if nested.exists() else base

    @property
    def SYNTHETIC_DATA_DIR(self) -> Path:
        p = os.getenv("SYNTHETIC_DATA_DIR")
        return Path(p) if p else self.DATA_DIR / "synthetic"

    @property
    def SPLITS_DIR(self) -> Path:
        return self.DATA_DIR / "splits"

    @property
    def SAMPLES_DIR(self) -> Path:
        return self.DATA_DIR / "samples"

    @property
    def CHECKPOINT_DIR(self) -> Path:
        p = os.getenv("CHECKPOINT_DIR")
        return Path(p) if p else self.BASE_DIR / "checkpoints"

    @property
    def REPORTS_DIR(self) -> Path:
        p = os.getenv("REPORTS_DIR")
        return Path(p) if p else self.BASE_DIR / "reports"

    @property
    def SESSIONS_STORAGE_DIR(self) -> Path:
        p = os.getenv("SESSIONS_STORAGE_DIR")
        return Path(p) if p else self.DATA_DIR / "sessions"

    # Backend & Authentication Configuration
    MONGODB_URI: str = field(default_factory=lambda: os.getenv("MONGODB_URI", "").strip().strip("'\""))
    MONGODB_DB_NAME: str = field(default_factory=lambda: os.getenv("MONGODB_DB_NAME", "past2pixel").strip().strip("'\""))
    JWT_SECRET_KEY: str = field(default_factory=lambda: os.getenv("JWT_SECRET_KEY", "past2pixel-insecure-secret-key-change-in-production"))
    JWT_ALGORITHM: str = field(default_factory=lambda: os.getenv("JWT_ALGORITHM", "HS256"))
    ACCESS_TOKEN_EXPIRE_MINUTES: int = field(default_factory=lambda: int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "1440")))

    # Uploads & Security
    MAX_UPLOAD_SIZE_MB: int = field(default_factory=lambda: int(os.getenv("MAX_UPLOAD_SIZE_MB", "15")))
    ALLOWED_IMAGE_EXTENSIONS: tuple = (".png", ".jpg", ".jpeg", ".tiff", ".bmp")

    # OCR Configuration
    TESSERACT_CMD: str = field(default_factory=lambda: os.getenv("TESSERACT_CMD", ""))

    # Compute & Device
    DEVICE: str = field(default_factory=lambda: os.getenv("DEVICE", "auto"))
    NUM_WORKERS: int = field(default_factory=lambda: int(os.getenv("NUM_WORKERS", "0")))
    SEED: int = field(default_factory=lambda: int(os.getenv("SEED", "42")))

    # Image Dimensions
    INPUT_IMAGE_SIZE: int = field(default_factory=lambda: int(os.getenv("INPUT_IMAGE_SIZE", "256")))
    IMAGE_CHANNELS: int = field(default_factory=lambda: int(os.getenv("IMAGE_CHANNELS", "3")))

    # Training Hyperparameters
    DEFAULT_BATCH_SIZE: int = field(default_factory=lambda: int(os.getenv("DEFAULT_BATCH_SIZE", "8")))
    DEFAULT_EPOCHS: int = field(default_factory=lambda: int(os.getenv("DEFAULT_EPOCHS", "15")))
    LEARNING_RATE: float = field(default_factory=lambda: float(os.getenv("LEARNING_RATE", "0.0002")))
    WEIGHT_DECAY: float = field(default_factory=lambda: float(os.getenv("WEIGHT_DECAY", "1e-5")))

    # Checkpoint filenames
    UNET_RESTORATION_CHECKPOINT: str = "unet_restoration_best.pt"
    AUTOENCODER_CHECKPOINT: str = "autoencoder_baseline_best.pt"
    DAMAGE_DETECTOR_CHECKPOINT: str = "damage_detector_best.pt"

    def ensure_directories(self) -> None:
        """Create required runtime directories if they do not exist."""
        for path in [
            self.DATA_DIR,
            self.RAW_DATA_DIR,
            self.VEDO_DATASET_DIR,
            self.SANSKRIT_HINDI_DATASET_DIR,
            self.SYNTHETIC_DATA_DIR,
            self.SPLITS_DIR,
            self.SAMPLES_DIR,
            self.CHECKPOINT_DIR,
            self.REPORTS_DIR,
            self.SESSIONS_STORAGE_DIR,
        ]:
            path.mkdir(parents=True, exist_ok=True)


# Global settings singleton
settings = Settings()


def get_settings() -> Settings:
    """Returns the active global settings instance."""
    return settings


def reload_settings(env_path: Optional[Path] = None, override: bool = True) -> Settings:
    """
    Reloads environment variables and refreshes the global settings singleton.
    Useful for testing or dynamic environment reconfiguration.
    """
    global settings
    load_project_env(env_path=env_path, override=override)
    settings = Settings()
    return settings
