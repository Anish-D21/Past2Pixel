"""
Restoration Evaluation Metrics for Past2Pixel.
Provides:
1. Reference-Based Metrics (PSNR & SSIM) — Calculated ONLY when a clean reference image exists.
2. Reference-Free Restoration Quality Metrics:
   - Text-to-Background Contrast (Michelson / Weber contrast)
   - Stroke Edge Sharpness (Tenengrad gradient density on stroke contours)
   - Stroke Preservation & Integrity Score (relative to original ink mask)
   - Background Noise & Stain Variance (uniformity of parchment)
   - Composite Quality Score for Candidate Selection
"""
from typing import Dict, Any, Optional, Union, Tuple
import numpy as np
from PIL import Image
from scipy import ndimage
import torch


def _to_numpy_normalized(img: Union[np.ndarray, Image.Image, torch.Tensor]) -> np.ndarray:
    """Converts tensor, PIL, or numpy array to normalized [0.0, 1.0] float32 numpy array."""
    if isinstance(img, torch.Tensor):
        arr = img.detach().cpu().squeeze().numpy()
        if arr.ndim == 3 and arr.shape[0] in (1, 3):  # (C, H, W) -> (H, W, C)
            arr = np.transpose(arr, (1, 2, 0))
        if arr.max() > 1.0:
            arr = arr / 255.0
        return arr.astype(np.float32)
    elif isinstance(img, Image.Image):
        arr = np.array(img, dtype=np.float32) / 255.0
        return arr
    elif isinstance(img, np.ndarray):
        arr = img.copy()
        if arr.max() > 1.0:
            arr = arr / 255.0
        return arr.astype(np.float32)
    else:
        raise TypeError(f"Unsupported image type: {type(img)}")


def calculate_psnr(
    image_pred: Union[np.ndarray, Image.Image, torch.Tensor],
    image_true: Union[np.ndarray, Image.Image, torch.Tensor],
    data_range: float = 1.0,
) -> float:
    """
    Computes Peak Signal-to-Noise Ratio (PSNR) in decibels (dB).

    Args:
        image_pred: Predicted image as numpy array, PIL Image, or Tensor
        image_true: Ground truth clean image
        data_range: Dynamic range (1.0 for normalized float, 255 for uint8)
    """
    p = _to_numpy_normalized(image_pred) if not isinstance(image_pred, np.ndarray) else image_pred
    t = _to_numpy_normalized(image_true) if not isinstance(image_true, np.ndarray) else image_true

    mse = np.mean((p.astype(np.float64) - t.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return float(20.0 * np.log10(data_range) - 10.0 * np.log10(mse))


def calculate_ssim(
    image_pred: Union[np.ndarray, Image.Image, torch.Tensor],
    image_true: Union[np.ndarray, Image.Image, torch.Tensor],
    data_range: float = 1.0,
    win_size: int = 7,
    k1: float = 0.01,
    k2: float = 0.03,
) -> float:
    """
    Computes Structural Similarity Index (SSIM) between two images.
    Compliant with Wang et al. (2004).
    """
    p = _to_numpy_normalized(image_pred) if not isinstance(image_pred, np.ndarray) else image_pred
    t = _to_numpy_normalized(image_true) if not isinstance(image_true, np.ndarray) else image_true

    if p.ndim == 3 and p.shape[2] in (3, 4):
        channels = [
            calculate_ssim(
                p[:, :, c],
                t[:, :, c],
                data_range=data_range,
                win_size=win_size,
                k1=k1,
                k2=k2,
            )
            for c in range(p.shape[2])
        ]
        return float(np.mean(channels))

    img1 = p.astype(np.float64)
    img2 = t.astype(np.float64)

    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2

    mu1 = np.mean(img1)
    mu2 = np.mean(img2)
    sigma1_sq = np.var(img1)
    sigma2_sq = np.var(img2)
    sigma12 = np.mean((img1 - mu1) * (img2 - mu2))

    numerator = (2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)
    denominator = (mu1 ** 2 + mu2 ** 2 + c1) * (sigma1_sq + sigma2_sq + c2)
    return float(np.clip(numerator / (denominator + 1e-12), -1.0, 1.0))


def evaluate_restoration_pair(
    reconstructed: Union[np.ndarray, Image.Image, torch.Tensor],
    clean_reference: Optional[Union[np.ndarray, Image.Image, torch.Tensor]],
) -> Dict[str, Any]:
    """
    Evaluates restoration performance against ground-truth clean image.
    Strictly reports metrics as UNAVAILABLE if reference is None.
    """
    if clean_reference is None:
        return {
            "status": "UNAVAILABLE",
            "reason": "Authentic uploaded manuscript has no ground-truth clean reference image. Reference PSNR and SSIM require an uncorrupted target.",
            "psnr_db": None,
            "ssim": None,
            "reference_available": False,
        }

    rec_np = _to_numpy_normalized(reconstructed)
    ref_np = _to_numpy_normalized(clean_reference)

    if rec_np.shape != ref_np.shape:
        return {
            "status": "UNAVAILABLE",
            "reason": f"Dimension mismatch: reconstructed is {rec_np.shape}, reference is {ref_np.shape}.",
            "psnr_db": None,
            "ssim": None,
            "reference_available": False,
        }

    psnr_val = calculate_psnr(rec_np, ref_np, data_range=1.0)
    ssim_val = calculate_ssim(rec_np, ref_np, data_range=1.0)

    return {
        "status": "CALCULATED",
        "psnr_db": round(psnr_val, 3) if psnr_val != float("inf") else 999.0,
        "ssim": round(ssim_val, 4),
        "data_range": 1.0,
        "reference_available": True,
        "interpretation": {
            "psnr": "Higher is better (>25 dB indicates good visual fidelity, >30 dB indicates high quality)",
            "ssim": "Structural similarity index in range [-1, 1], where 1.0 indicates identical structure.",
        },
    }


# -------------------------------------------------------------
# Reference-Free Manuscript Quality Metrics
# -------------------------------------------------------------

def calculate_text_contrast(
    image: Union[np.ndarray, Image.Image],
    ink_mask: np.ndarray,
) -> float:
    """
    Calculates Michelson text-to-background contrast:
    (mean_bg - mean_ink) / (mean_bg + mean_ink + epsilon)
    Higher is better (range [0.0, 1.0]).
    """
    arr = _to_numpy_normalized(image)
    gray = 0.2989 * arr[:, :, 0] + 0.5870 * arr[:, :, 1] + 0.1140 * arr[:, :, 2] if arr.ndim == 3 else arr

    ink_pixels = gray[ink_mask > 0.5]
    bg_pixels = gray[ink_mask <= 0.1]

    if len(ink_pixels) == 0 or len(bg_pixels) == 0:
        return 0.5

    mean_ink = float(np.mean(ink_pixels))
    mean_bg = float(np.mean(bg_pixels))

    contrast = (mean_bg - mean_ink) / (mean_bg + mean_ink + 1e-6)
    return float(np.clip(contrast, 0.0, 1.0))


def calculate_stroke_sharpness(
    image: Union[np.ndarray, Image.Image],
    ink_mask: np.ndarray,
) -> float:
    """
    Computes Tenengrad gradient magnitude along ink stroke boundaries.
    Detects blurring and stroke degradation.
    """
    arr = _to_numpy_normalized(image)
    gray = (0.2989 * arr[:, :, 0] + 0.5870 * arr[:, :, 1] + 0.1140 * arr[:, :, 2]) * 255.0 if arr.ndim == 3 else arr * 255.0

    gx = ndimage.sobel(gray, axis=1)
    gy = ndimage.sobel(gray, axis=0)
    grad = np.hypot(gx, gy)

    # Stroke edge boundary mask
    eroded_ink = ndimage.binary_erosion(ink_mask > 0.5)
    boundary = (ink_mask > 0.5) & (~eroded_ink)

    if not boundary.any():
        boundary = ink_mask > 0.5

    if boundary.any():
        sharpness = float(np.mean(grad[boundary]))
    else:
        sharpness = float(np.mean(grad))

    return round(sharpness, 2)


def calculate_stroke_preservation(
    candidate_image: Union[np.ndarray, Image.Image],
    original_ink_mask: np.ndarray,
) -> float:
    """
    Measures the ratio of original ink stroke pixels that remain distinct
    and preserved in the candidate restoration without being erased or blurred into background.
    """
    arr = _to_numpy_normalized(candidate_image)
    gray = 0.2989 * arr[:, :, 0] + 0.5870 * arr[:, :, 1] + 0.1140 * arr[:, :, 2] if arr.ndim == 3 else arr

    ink_indices = original_ink_mask > 0.5
    if not ink_indices.any():
        return 1.0

    # Local threshold to test if stroke pixels remain darker than surrounding background
    bg_level = float(np.median(gray[original_ink_mask <= 0.1])) if (original_ink_mask <= 0.1).any() else 0.8
    stroke_values = gray[ink_indices]

    # A stroke pixel is preserved if it remains substantially darker than background
    preserved = np.sum(stroke_values < (bg_level - 0.08))
    ratio = float(preserved / np.sum(ink_indices))
    return round(float(np.clip(ratio, 0.0, 1.0)), 4)


def calculate_background_variance(
    image: Union[np.ndarray, Image.Image],
    ink_mask: np.ndarray,
) -> float:
    """
    Measures standard deviation across non-text background parchment regions.
    Lower variance indicates successful removal of blotches, water stains, and noise.
    """
    arr = _to_numpy_normalized(image)
    gray = (0.2989 * arr[:, :, 0] + 0.5870 * arr[:, :, 1] + 0.1140 * arr[:, :, 2]) * 255.0 if arr.ndim == 3 else arr * 255.0

    bg_pixels = gray[ink_mask <= 0.1]
    if len(bg_pixels) == 0:
        return 0.0

    std = float(np.std(bg_pixels))
    return round(std, 2)


def evaluate_candidate_quality(
    candidate_image: Union[np.ndarray, Image.Image],
    reference_ink_mask: np.ndarray,
    original_sharpness: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Produces complete reference-free quality evaluation for a candidate restoration:
    - Text-to-background contrast
    - Stroke edge sharpness
    - Stroke preservation ratio
    - Background stain & noise standard deviation
    - Normalized Composite Quality Score [0.0, 100.0]
    """
    contrast = calculate_text_contrast(candidate_image, reference_ink_mask)
    sharpness = calculate_stroke_sharpness(candidate_image, reference_ink_mask)
    preservation = calculate_stroke_preservation(candidate_image, reference_ink_mask)
    bg_std = calculate_background_variance(candidate_image, reference_ink_mask)

    orig_s = original_sharpness or sharpness
    sharpness_retention = min(1.2, sharpness / max(1.0, orig_s))

    # Composite score formula:
    # 40% Contrast + 35% Stroke Preservation + 15% Sharpness Retention + 10% Background Cleanliness
    bg_cleanliness = max(0.0, 1.0 - (bg_std / 50.0))
    score = (
        0.40 * contrast +
        0.35 * preservation +
        0.15 * sharpness_retention +
        0.10 * bg_cleanliness
    ) * 100.0

    return {
        "composite_score": round(float(np.clip(score, 0.0, 100.0)), 1),
        "text_contrast": round(contrast, 3),
        "stroke_sharpness": sharpness,
        "stroke_preservation_ratio": preservation,
        "background_noise_std": bg_std,
    }
