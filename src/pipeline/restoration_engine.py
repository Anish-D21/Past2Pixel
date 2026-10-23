"""
Phase 1 Integrated Restoration & Damage Detection Engine.
Executes preprocessing, damage localization, restoration, and objective metric computation.
Captures genuine processing time, device info, and intermediate outputs for image-specific reports.
"""
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union
import uuid
import time
from PIL import Image
import numpy as np
import torch
import torchvision.transforms.functional as TF

from ..core.config import settings
from ..core.device import get_device, get_device_info, Timer
from ..models.unet_restoration import UNetRestoration
from ..models.unet_damage_detector import UNetDamageDetector, generate_damage_overlay
from ..models.autoencoder import ConvAutoencoder
from ..models.model_registry import model_registry, load_model_for_inference, ModelReadinessStatus
from ..metrics.restoration_metrics import evaluate_restoration_pair
from ..metrics.damage_metrics import calculate_segmentation_metrics


class RestorationEngine:
    """
    Core AI engine coordinating damage detection, image restoration, and metric calculations.
    """

    def __init__(self, device_preference: str = "auto"):
        self.device = get_device(device_preference)
        self.device_info = get_device_info(self.device)

        # Initialize network architectures
        self.unet_restoration = UNetRestoration(in_channels=3, out_channels=3).to(self.device)
        self.damage_detector = UNetDamageDetector(in_channels=3).to(self.device)
        self.autoencoder_baseline = ConvAutoencoder(in_channels=3).to(self.device)

        # Attempt checkpoint loading
        self._load_checkpoints()

    def _load_checkpoints(self) -> None:
        """Loads available weights for each model independently and updates readiness."""
        ckpt_dir = settings.CHECKPOINT_DIR
        unet_ckpt = ckpt_dir / settings.UNET_RESTORATION_CHECKPOINT
        detector_ckpt = ckpt_dir / settings.DAMAGE_DETECTOR_CHECKPOINT
        ae_ckpt = ckpt_dir / settings.AUTOENCODER_CHECKPOINT

        if unet_ckpt.exists():
            try:
                self.unet_restoration, _ = load_model_for_inference(
                    self.unet_restoration, unet_ckpt, self.device
                )
                print(f"[RestorationEngine] Loaded U-Net restoration checkpoint from {unet_ckpt}")
            except Exception as e:
                print(f"[RestorationEngine Warning] Could not load U-Net checkpoint: {e}")

        if detector_ckpt.exists():
            try:
                self.damage_detector, _ = load_model_for_inference(
                    self.damage_detector, detector_ckpt, self.device
                )
                print(f"[RestorationEngine] Loaded Damage Detector checkpoint from {detector_ckpt}")
            except Exception as e:
                print(f"[RestorationEngine Warning] Could not load Damage Detector checkpoint: {e}")

        if ae_ckpt.exists():
            try:
                self.autoencoder_baseline, _ = load_model_for_inference(
                    self.autoencoder_baseline, ae_ckpt, self.device
                )
                print(f"[RestorationEngine] Loaded Autoencoder baseline checkpoint from {ae_ckpt}")
            except Exception as e:
                print(f"[RestorationEngine Warning] Could not load Autoencoder checkpoint: {e}")

    def process_image(
        self,
        input_image: Union[Path, str, Image.Image],
        reference_clean_image: Optional[Union[Path, str, Image.Image]] = None,
        reference_damage_mask: Optional[Union[Path, str, np.ndarray]] = None,
        output_dir: Optional[Path] = None,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Executes complete restoration & damage detection pipeline on a single image.
        Captures genuine timing, hardware specs, intermediate visual arrays, and metrics.
        """
        session_id = session_id or f"p2p_{uuid.uuid4().hex[:12]}_{int(time.time())}"
        overall_timer = Timer("overall_pipeline")

        with overall_timer:
            # 1. Load and validate input image
            if isinstance(input_image, (str, Path)):
                img_path = Path(input_image)
                filename = img_path.name
                original_pil = Image.open(img_path).convert("RGB")
            else:
                filename = "uploaded_image.png"
                original_pil = input_image.convert("RGB")

            orig_w, orig_h = original_pil.size
            orig_metadata = {
                "filename": filename,
                "original_width": orig_w,
                "original_height": orig_h,
                "original_mode": original_pil.mode,
            }

            # 2. Preprocess: resize to standardized input resolution
            target_size = (settings.INPUT_IMAGE_SIZE, settings.INPUT_IMAGE_SIZE)
            with Timer("preprocessing") as prep_timer:
                resized_pil = original_pil.resize(target_size, Image.Resampling.BILINEAR)
                input_tensor = TF.to_tensor(resized_pil).unsqueeze(0).to(self.device)

            # 3. Damage Localization
            detector_ready = model_registry.is_ready_for_inference("unet_damage_detector")
            with Timer("damage_detection") as det_timer:
                self.damage_detector.eval()
                with torch.no_grad():
                    damage_prob_tensor = self.damage_detector(input_tensor)
                    damage_mask_256 = damage_prob_tensor.squeeze().cpu().numpy()

                # Scale damage mask back to original resolution
                damage_mask_pil = Image.fromarray((damage_mask_256 * 255).astype(np.uint8))
                damage_mask_orig_res = damage_mask_pil.resize((orig_w, orig_h), Image.Resampling.BILINEAR)
                damage_mask_np = np.array(damage_mask_orig_res, dtype=np.float32) / 255.0

                # Generate visual overlay
                damage_overlay_pil = generate_damage_overlay(
                    original_pil, damage_mask_np, threshold=0.35
                )

            # 4. Image Restoration (U-Net)
            unet_ready = model_registry.is_ready_for_inference("unet_restoration")
            with Timer("unet_restoration") as unet_timer:
                self.unet_restoration.eval()
                with torch.no_grad():
                    restored_tensor = self.unet_restoration(input_tensor)
                    restored_256_arr = restored_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
                    restored_256_pil = Image.fromarray(
                        np.clip(restored_256_arr * 255.0, 0, 255).astype(np.uint8)
                    )
                    # Upsample restored output back to original dimensions
                    restored_pil = restored_256_pil.resize((orig_w, orig_h), Image.Resampling.BILINEAR)

            # 5. Optional Autoencoder Baseline
            ae_ready = model_registry.is_ready_for_inference("autoencoder_baseline")
            with Timer("autoencoder_baseline") as ae_timer:
                self.autoencoder_baseline.eval()
                with torch.no_grad():
                    ae_tensor = self.autoencoder_baseline(input_tensor)
                    ae_256_arr = ae_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
                    ae_pil = Image.fromarray(
                        np.clip(ae_256_arr * 255.0, 0, 255).astype(np.uint8)
                    ).resize((orig_w, orig_h), Image.Resampling.BILINEAR)

            # 6. Evaluation Metrics
            # Restoration metrics (PSNR & SSIM)
            clean_ref_pil = None
            if reference_clean_image is not None:
                if isinstance(reference_clean_image, (str, Path)):
                    clean_ref_pil = Image.open(reference_clean_image).convert("RGB").resize((orig_w, orig_h))
                else:
                    clean_ref_pil = reference_clean_image.convert("RGB").resize((orig_w, orig_h))

            unet_metrics = evaluate_restoration_pair(restored_pil, clean_ref_pil)
            ae_metrics = evaluate_restoration_pair(ae_pil, clean_ref_pil)

            # Damage detection metrics (IoU & Dice)
            true_mask_np = None
            if reference_damage_mask is not None:
                if isinstance(reference_damage_mask, (str, Path)):
                    with Image.open(reference_damage_mask) as mf:
                        true_mask_np = np.array(mf.resize((orig_w, orig_h)), dtype=np.float32) / 255.0
                else:
                    true_mask_np = reference_damage_mask

            damage_eval = calculate_segmentation_metrics(damage_mask_np, true_mask_np)

            # 7. Save Session Artifacts if output directory provided
            saved_paths = {}
            if output_dir:
                out_base = Path(output_dir)
                if out_base.name == session_id:
                    out_p = out_base
                else:
                    out_p = out_base / session_id
                out_p.mkdir(parents=True, exist_ok=True)
                orig_path = out_p / "original.png"
                restored_path = out_p / "restored_unet.png"
                overlay_path = out_p / "damage_overlay.png"
                mask_path = out_p / "damage_mask.png"

                original_pil.save(orig_path)
                restored_pil.save(restored_path)
                damage_overlay_pil.save(overlay_path)
                damage_mask_orig_res.save(mask_path)

                saved_paths = {
                    "original_image": str(orig_path),
                    "restored_image": str(restored_path),
                    "damage_overlay": str(overlay_path),
                    "damage_mask": str(mask_path),
                }

        report = {
            "session_id": session_id,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "device_info": self.device_info,
            "input_metadata": orig_metadata,
            "timings_ms": {
                "preprocessing": round(prep_timer.elapsed_ms, 2),
                "damage_detection": round(det_timer.elapsed_ms, 2),
                "unet_restoration": round(unet_timer.elapsed_ms, 2),
                "autoencoder_baseline": round(ae_timer.elapsed_ms, 2),
                "total_pipeline": round(overall_timer.elapsed_ms, 2),
            },
            "model_readiness": {
                "unet_restoration": "READY_FOR_INFERENCE" if unet_ready else "UNTRAINED_RANDOM_WEIGHTS",
                "damage_detector": "READY_FOR_INFERENCE" if detector_ready else "UNTRAINED_RANDOM_WEIGHTS",
                "autoencoder_baseline": "READY_FOR_INFERENCE" if ae_ready else "UNTRAINED_RANDOM_WEIGHTS",
            },
            "damage_detection_result": {
                "damaged_pixel_percentage": round(float((damage_mask_np > 0.35).mean() * 100), 2),
                "evaluation": damage_eval,
            },
            "restoration_results": {
                "unet_restoration_metrics": unet_metrics,
                "autoencoder_baseline_metrics": ae_metrics,
            },
            "saved_artifact_paths": saved_paths,
            "images": {
                "original": original_pil,
                "restored": restored_pil,
                "damage_mask": damage_mask_np,
                "damage_overlay": damage_overlay_pil,
                "autoencoder_output": ae_pil,
            },
        }

        return report
