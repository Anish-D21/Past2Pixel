"""
Centralized Configuration Module for Past2Pixel.
Uses Python dataclasses with environment variable overrides.
Zero third-party dependency for maximum portability across CPU and GPU systems.
"""
from pathlib import Path
from dataclasses import dataclass, field
import os


@dataclass
class Settings:
    # Base paths
    BASE_DIR: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent.parent)
    
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
        return Path(p) if p else self.RAW_DATA_DIR / "vedo_ocr"

    @property
    def SANSKRIT_HINDI_DATASET_DIR(self) -> Path:
        p = os.getenv("SANSKRIT_HINDI_DATASET_DIR")
        return Path(p) if p else self.RAW_DATA_DIR / "sanskrit_hindi_vqa"

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
        ]:
            path.mkdir(parents=True, exist_ok=True)


# Global settings singleton
settings = Settings()
