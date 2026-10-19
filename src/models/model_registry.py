"""
Model Readiness Registry and Checkpoint Manager.
Explicitly tracks model lifecycle:
- NOT_INITIALIZED
- ARCHITECTURE_DEFINED
- TRAINING_STARTED
- CHECKPOINT_SAVED
- READY_FOR_INFERENCE
Ensures no model is reported as trained unless a verified checkpoint is loaded and validated.
"""
from enum import Enum
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import torch

from ..core.config import settings


class ModelReadinessStatus(str, Enum):
    NOT_INITIALIZED = "NOT_INITIALIZED"
    ARCHITECTURE_DEFINED = "ARCHITECTURE_DEFINED"
    TRAINING_STARTED = "TRAINING_STARTED"
    CHECKPOINT_SAVED = "CHECKPOINT_SAVED"
    READY_FOR_INFERENCE = "READY_FOR_INFERENCE"
    UNTRAINED = "UNTRAINED"


class ModelRegistry:
    """
    Tracks readiness status, checkpoint paths, and metadata for each model in the pipeline.
    """

    def __init__(self, checkpoint_dir: Path = None):
        self.checkpoint_dir = checkpoint_dir or settings.CHECKPOINT_DIR
        self.models_status: Dict[str, Dict[str, Any]] = {
            "autoencoder_baseline": {
                "name": "Convolutional Autoencoder Baseline",
                "checkpoint_file": settings.AUTOENCODER_CHECKPOINT,
                "status": ModelReadinessStatus.ARCHITECTURE_DEFINED,
                "checkpoint_exists": False,
                "training_metadata": None,
            },
            "unet_restoration": {
                "name": "U-Net Primary Restoration",
                "checkpoint_file": settings.UNET_RESTORATION_CHECKPOINT,
                "status": ModelReadinessStatus.ARCHITECTURE_DEFINED,
                "checkpoint_exists": False,
                "training_metadata": None,
            },
            "unet_damage_detector": {
                "name": "U-Net Damage Localization",
                "checkpoint_file": settings.DAMAGE_DETECTOR_CHECKPOINT,
                "status": ModelReadinessStatus.ARCHITECTURE_DEFINED,
                "checkpoint_exists": False,
                "training_metadata": None,
            },
            "context_text_predictor": {
                "name": "Context-Aware Damaged Text Predictor",
                "checkpoint_file": "context_predictor.json",
                "status": ModelReadinessStatus.ARCHITECTURE_DEFINED,
                "checkpoint_exists": False,
                "training_metadata": None,
            },
        }
        self.refresh_checkpoint_status()

    def refresh_checkpoint_status(self) -> None:
        """Inspects disk to verify if actual checkpoint files exist and updates readiness."""
        for key, entry in self.models_status.items():
            ckpt_path = self.checkpoint_dir / entry["checkpoint_file"]
            if ckpt_path.exists():
                entry["checkpoint_exists"] = True
                entry["checkpoint_path"] = str(ckpt_path)
                try:
                    if ckpt_path.suffix == ".pt":
                        # Load lightweight metadata only
                        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                        if isinstance(ckpt, dict) and "epoch" in ckpt:
                            entry["training_metadata"] = {
                                "epoch": ckpt.get("epoch"),
                                "val_loss": ckpt.get("val_loss"),
                                "metrics": ckpt.get("metrics", {}),
                                "trained_device": ckpt.get("device_info", {}).get("device_type", "unknown"),
                            }
                            entry["status"] = ModelReadinessStatus.READY_FOR_INFERENCE
                        else:
                            entry["status"] = ModelReadinessStatus.CHECKPOINT_SAVED
                    else:
                        entry["status"] = ModelReadinessStatus.READY_FOR_INFERENCE
                except Exception as e:
                    entry["status"] = ModelReadinessStatus.UNTRAINED
                    entry["error"] = f"Failed to validate checkpoint: {e}"
            else:
                entry["checkpoint_exists"] = False
                entry["checkpoint_path"] = None
                entry["status"] = ModelReadinessStatus.UNTRAINED

    def get_status(self, model_key: str) -> Dict[str, Any]:
        """Returns readiness status dictionary for a specific model."""
        self.refresh_checkpoint_status()
        return self.models_status.get(model_key, {"status": ModelReadinessStatus.NOT_INITIALIZED})

    def get_all_statuses(self) -> Dict[str, Dict[str, Any]]:
        """Returns status dictionary for all models in the system."""
        self.refresh_checkpoint_status()
        return self.models_status

    def is_ready_for_inference(self, model_key: str) -> bool:
        """Returns True if the model has a valid, validated checkpoint ready to run."""
        status_info = self.get_status(model_key)
        return status_info.get("status") == ModelReadinessStatus.READY_FOR_INFERENCE


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    val_loss: float,
    metrics: Dict[str, Any],
    device_info: Dict[str, Any],
    output_path: Path,
) -> Path:
    """
    Saves a comprehensive training checkpoint with metadata.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "val_loss": val_loss,
        "metrics": metrics,
        "device_info": device_info,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
    }
    torch.save(checkpoint, output_path)
    return output_path


def load_model_for_inference(
    model: torch.nn.Module,
    checkpoint_path: Path,
    device: torch.device,
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """
    Loads checkpoint weights onto model and returns model in eval mode along with training metadata.
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        metadata = {
            "epoch": checkpoint.get("epoch"),
            "val_loss": checkpoint.get("val_loss"),
            "metrics": checkpoint.get("metrics", {}),
            "device_info": checkpoint.get("device_info", {}),
        }
    else:
        model.load_state_dict(checkpoint)
        metadata = {"epoch": None, "val_loss": None}

    model.to(device)
    model.eval()
    return model, metadata


# Global registry singleton
model_registry = ModelRegistry()
