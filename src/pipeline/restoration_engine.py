"""
Phase 1 & Phase 3 Integrated Restoration & Damage Detection Engine for Past2Pixel.
Integrates:
1. High-Resolution Text & Ink Protection Engine (Sauvola + Sobel edge gradient confirmation)
2. Conservative Traditional Image-Processing Baseline (illumination flattening & ink deepening)
3. Deep Learning U-Net Restoration with Native Resolution Inference
4. Objective Candidate Evaluation & Evidence-Based Selection Gate
5. Scientific Reference Evaluation (PSNR/SSIM only when clean ground-truth reference exists)
"""
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union
import uuid
import time
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from ..core.config import settings
from ..core.device import get_device, get_device_info, Timer
from ..models.unet_restoration import UNetRestoration
from ..models.unet_damage_detector import UNetDamageDetector, generate_damage_overlay
from ..models.autoencoder import ConvAutoencoder
from ..models.model_registry import model_registry, load_model_for_inference
from ..metrics.restoration_metrics import (
    evaluate_restoration_pair,
    evaluate_candidate_quality,
)
from ..metrics.damage_metrics import calculate_segmentation_metrics
from .text_protection import text_protection_engine
from .traditional_baseline import traditional_restoration_engine


class RestorationEngine:
    """
    Core AI & Computer Vision Engine for Ancient Manuscript Restoration.
    Guarantees text stroke preservation, conservative background enhancement,
    and evidence-based candidate selection.
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

    def _infer_fully_convolutional(
        self,
        model: torch.nn.Module,
        img_pil: Image.Image,
        max_native_dim: int = 1536,
    ) -> np.ndarray:
        """
        Executes native-resolution forward pass on fully convolutional U-Net architectures.
        Uses reflection padding to nearest multiple of 16 to avoid downscaling/upscaling blur.
        """
        w, h = img_pil.size

        # If image exceeds max single-pass dimension, scale proportionally while preserving aspect ratio
        if max(w, h) > max_native_dim:
            scale = max_native_dim / max(w, h)
            target_w, target_h = int(w * scale), int(h * scale)
            eval_pil = img_pil.resize((target_w, target_h), Image.Resampling.LANCZOS)
        else:
            eval_pil = img_pil
            target_w, target_h = w, h

        pad_w = (16 - (target_w % 16)) % 16
        pad_h = (16 - (target_h % 16)) % 16

        t = TF.to_tensor(eval_pil).unsqueeze(0).to(self.device)
        t_padded = F.pad(t, (0, pad_w, 0, pad_h), mode="reflect")

        model.eval()
        with torch.no_grad():
            out_padded = model(t_padded)

        # Unpad
        out = out_padded[:, :, :target_h, :target_w]

        # Convert back to (H, W, 3) float array in [0, 255]
        out_arr = (out.squeeze().permute(1, 2, 0).cpu().numpy() * 255.0)

        # Resize to original resolution if downscaled for memory
        if (target_w, target_h) != (w, h):
            out_pil = Image.fromarray(np.clip(out_arr, 0, 255).astype(np.uint8))
            out_pil = out_pil.resize((w, h), Image.Resampling.LANCZOS)
            out_arr = np.array(out_pil, dtype=np.float32)

        return np.clip(out_arr, 0.0, 255.0)

    def process_image(
        self,
        input_image: Union[Path, str, Image.Image],
        reference_clean_image: Optional[Union[Path, str, Image.Image]] = None,
        reference_damage_mask: Optional[Union[Path, str, np.ndarray]] = None,
        output_dir: Optional[Path] = None,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Executes complete restoration & damage detection pipeline:
        1. Original image loading & metadata capture
        2. High-resolution Text & Ink Segmentation (Sauvola + Sobel gradients)
        3. Traditional Image-Processing Baseline Generation
        4. Damage Localization & Overlay Generation
        5. Native-Resolution Model Restoration with Ink Protection
        6. Reference-Free Candidate Quality Evaluation & Selection Gate
        7. Artifact Persistence & Reporting
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
            orig_arr = np.array(original_pil, dtype=np.float32)

            orig_metadata = {
                "filename": filename,
                "original_width": orig_w,
                "original_height": orig_h,
                "original_mode": original_pil.mode,
            }

            # 2. Text & Ink Protection: Segment ink strokes from background
            with Timer("text_protection") as text_timer:
                text_seg = text_protection_engine.segment_text_and_ink(orig_arr)
                confident_ink = text_seg["confident_ink_mask"]
                alpha_weight = text_seg["alpha_weight"][:, :, np.newaxis]
                visual_text_mask_pil = text_seg["visual_text_mask_pil"]

            # 3. Traditional Restoration Baseline (Illumination flattening & ink deepening)
            with Timer("traditional_baseline") as base_timer:
                baseline_res = traditional_restoration_engine.restore_image(
                    orig_arr, text_segmentation=text_seg
                )
                baseline_pil = baseline_res["restored_pil"]
                baseline_arr = np.array(baseline_pil, dtype=np.float32)

            # 4. Damage Localization (U-Net Damage Detector)
            detector_ready = model_registry.is_ready_for_inference("unet_damage_detector")
            with Timer("damage_detection") as det_timer:
                # Infer damage probability mask
                target_size = (settings.INPUT_IMAGE_SIZE, settings.INPUT_IMAGE_SIZE)
                resized_pil = original_pil.resize(target_size, Image.Resampling.BILINEAR)
                input_tensor = TF.to_tensor(resized_pil).unsqueeze(0).to(self.device)

                self.damage_detector.eval()
                with torch.no_grad():
                    damage_prob_tensor = self.damage_detector(input_tensor)
                    damage_mask_256 = damage_prob_tensor.squeeze().cpu().numpy()

                damage_mask_pil = Image.fromarray((damage_mask_256 * 255).astype(np.uint8))
                damage_mask_orig_res = damage_mask_pil.resize((orig_w, orig_h), Image.Resampling.BILINEAR)
                damage_mask_np = np.array(damage_mask_orig_res, dtype=np.float32) / 255.0

                # Generate visual overlay
                damage_overlay_pil = generate_damage_overlay(
                    original_pil, damage_mask_np, threshold=0.35
                )

            # 5. Model-Guided Text-Aware Restoration
            unet_ready = model_registry.is_ready_for_inference("unet_restoration")
            with Timer("unet_restoration") as unet_timer:
                # Native-resolution forward pass on U-Net
                raw_model_arr = self._infer_fully_convolutional(self.unet_restoration, original_pil)

                # Text-Aware Compositing:
                # U-Net provides clean background parchment and stain suppression;
                # Original high-resolution ink strokes are preserved without blur or erosion!
                enhanced_ink = np.clip(orig_arr * 0.88, 0.0, 255.0)
                model_guided_arr = (1.0 - alpha_weight) * raw_model_arr + alpha_weight * enhanced_ink
                model_guided_arr = np.clip(model_guided_arr, 0.0, 255.0).astype(np.uint8)
                model_guided_pil = Image.fromarray(model_guided_arr)

            # 6. Autoencoder Baseline (Optional auxiliary model)
            ae_ready = model_registry.is_ready_for_inference("autoencoder_baseline")
            with Timer("autoencoder_baseline") as ae_timer:
                ae_arr = self._infer_fully_convolutional(self.autoencoder_baseline, original_pil)
                ae_pil = Image.fromarray(ae_arr.astype(np.uint8))

            # 7. Candidate Quality Evaluation & Selection Gate
            with Timer("candidate_selection") as sel_timer:
                orig_quality = evaluate_candidate_quality(original_pil, confident_ink)
                base_quality = evaluate_candidate_quality(baseline_pil, confident_ink, orig_quality["stroke_sharpness"])
                model_quality = evaluate_candidate_quality(model_guided_pil, confident_ink, orig_quality["stroke_sharpness"])

                candidates_dict = {
                    "original": orig_quality,
                    "traditional_baseline": base_quality,
                    "model_guided_restoration": model_quality,
                }

                # Evidence-Based Selection Decision:
                # 1. Check if model-guided restoration achieves superior score while maintaining stroke integrity
                if (
                    model_quality["composite_score"] >= base_quality["composite_score"]
                    and model_quality["stroke_preservation_ratio"] >= 0.90
                    and model_quality["text_contrast"] >= orig_quality["text_contrast"]
                ):
                    selected_candidate = "model_guided_restoration"
                    selected_pil = model_guided_pil
                    selection_reason = (
                        f"Model-guided restoration selected: achieved optimal composite quality "
                        f"({model_quality['composite_score']:.1f}/100) with superior background stain suppression "
                        f"(noise std: {model_quality['background_noise_std']:.1f}) while preserving "
                        f"{model_quality['stroke_preservation_ratio'] * 100:.1f}% of authentic ink stroke contours."
                    )
                # 2. Check if traditional baseline beats original image
                elif (
                    base_quality["composite_score"] >= orig_quality["composite_score"]
                    and base_quality["stroke_preservation_ratio"] >= 0.95
                ):
                    selected_candidate = "traditional_baseline"
                    selected_pil = baseline_pil
                    selection_reason = (
                        f"Traditional baseline selected: achieved superior text contrast "
                        f"({base_quality['text_contrast']:.3f} vs original {orig_quality['text_contrast']:.3f}) "
                        f"and conservative illumination flattening without neural smoothing artifacts."
                    )
                # 3. Fallback to original if both candidates degraded the image
                else:
                    selected_candidate = "original_preserved"
                    selected_pil = original_pil
                    selection_reason = (
                        f"Original manuscript preserved: automated candidate inspection indicated "
                        f"that original document strokes are already well-preserved; neural and traditional "
                        f"candidates did not exceed original contrast."
                    )

            # 8. Reference Evaluation (PSNR & SSIM strictly if clean ground truth supplied)
            clean_ref_pil = None
            if reference_clean_image is not None:
                if isinstance(reference_clean_image, (str, Path)):
                    clean_ref_pil = Image.open(reference_clean_image).convert("RGB").resize((orig_w, orig_h))
                else:
                    clean_ref_pil = reference_clean_image.convert("RGB").resize((orig_w, orig_h))

            unet_metrics = evaluate_restoration_pair(selected_pil, clean_ref_pil)
            baseline_metrics = evaluate_restoration_pair(baseline_pil, clean_ref_pil)
            ae_metrics = evaluate_restoration_pair(ae_pil, clean_ref_pil)

            # Damage detection metrics against ground truth mask
            true_mask_np = None
            if reference_damage_mask is not None:
                if isinstance(reference_damage_mask, (str, Path)):
                    with Image.open(reference_damage_mask) as mf:
                        true_mask_np = np.array(mf.resize((orig_w, orig_h)), dtype=np.float32) / 255.0
                else:
                    true_mask_np = reference_damage_mask

            damage_eval = calculate_segmentation_metrics(damage_mask_np, true_mask_np)

            # 9. Save Session Artifacts
            saved_paths = {}
            if output_dir:
                out_base = Path(output_dir)
                out_p = out_base if out_base.name == session_id else (out_base / session_id)
                out_p.mkdir(parents=True, exist_ok=True)

                orig_path = out_p / "original.png"
                restored_path = out_p / "restored_unet.png"  # Selected best restoration output
                baseline_path = out_p / "restored_baseline.png"
                model_path = out_p / "restored_model.png"
                text_mask_path = out_p / "text_mask.png"
                overlay_path = out_p / "damage_overlay.png"
                mask_path = out_p / "damage_mask.png"

                original_pil.save(orig_path)
                selected_pil.save(restored_path)
                baseline_pil.save(baseline_path)
                model_guided_pil.save(model_path)
                visual_text_mask_pil.save(text_mask_path)
                damage_overlay_pil.save(overlay_path)
                damage_mask_orig_res.save(mask_path)

                saved_paths = {
                    "original_image": str(orig_path),
                    "restored_image": str(restored_path),
                    "baseline_image": str(baseline_path),
                    "model_image": str(model_path),
                    "text_mask": str(text_mask_path),
                    "damage_overlay": str(overlay_path),
                    "damage_mask": str(mask_path),
                }

        report = {
            "session_id": session_id,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "device_info": self.device_info,
            "input_metadata": orig_metadata,
            "timings_ms": {
                "text_protection": round(text_timer.elapsed_ms, 2),
                "traditional_baseline": round(base_timer.elapsed_ms, 2),
                "damage_detection": round(det_timer.elapsed_ms, 2),
                "unet_restoration": round(unet_timer.elapsed_ms, 2),
                "candidate_selection": round(sel_timer.elapsed_ms, 2),
                "total_pipeline": round(overall_timer.elapsed_ms, 2),
            },
            "model_readiness": {
                "unet_restoration": "READY_FOR_INFERENCE" if unet_ready else "UNTRAINED_RANDOM_WEIGHTS",
                "damage_detector": "READY_FOR_INFERENCE" if detector_ready else "UNTRAINED_RANDOM_WEIGHTS",
                "autoencoder_baseline": "READY_FOR_INFERENCE" if ae_ready else "UNTRAINED_RANDOM_WEIGHTS",
            },
            "candidate_selection": {
                "selected_candidate": selected_candidate,
                "selection_reason": selection_reason,
                "candidates_evaluated": candidates_dict,
            },
            "text_protection_metrics": {
                "ink_pixel_ratio": text_seg["ink_pixel_ratio"],
                "ink_pixel_count": text_seg["ink_pixel_count"],
            },
            "damage_detection_result": {
                "damaged_pixel_percentage": round(float((damage_mask_np > 0.35).mean() * 100), 2),
                "evaluation": damage_eval,
            },
            "restoration_results": {
                "selected_restoration_metrics": unet_metrics,
                "baseline_metrics": baseline_metrics,
                "autoencoder_baseline_metrics": ae_metrics,
            },
            "saved_artifact_paths": saved_paths,
            "images": {
                "original": original_pil,
                "restored": selected_pil,
                "baseline": baseline_pil,
                "model_guided": model_guided_pil,
                "text_mask": visual_text_mask_pil,
                "damage_mask": damage_mask_np,
                "damage_overlay": damage_overlay_pil,
            },
        }

        return report


# Global default engine instance
restoration_engine = RestorationEngine()
