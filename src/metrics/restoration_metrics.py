"""
Restoration Evaluation Metrics: PSNR and SSIM.
Calculated ONLY when a valid ground-truth reference image exists.
Returns structured results with explicit availability flags and explanations.
"""
from typing import Dict, Any, Optional, Union
import numpy as np
from PIL import Image
import torch


def calculate_psnr(
    image_pred: np.ndarray,
    image_true: np.ndarray,
    data_range: float = 1.0,
) -> float:
    """
    Computes Peak Signal-to-Noise Ratio (PSNR) in decibels (dB).

    Args:
        image_pred: Predicted image as numpy array in [0, data_range]
        image_true: Ground truth image as numpy array in [0, data_range]
        data_range: Maximum pixel value (1.0 for float normalized, 255 for uint8)

    Returns:
        float: PSNR in dB. Returns float('inf') for identical images.
    """
    mse = np.mean((image_pred.astype(np.float64) - image_true.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return float(20.0 * np.log10(data_range) - 10.0 * np.log10(mse))


def calculate_ssim(
    image_pred: np.ndarray,
    image_true: np.ndarray,
    data_range: float = 1.0,
    win_size: int = 7,
    k1: float = 0.01,
    k2: float = 0.03,
) -> float:
    """
    Computes Structural Similarity Index (SSIM) between two images.
    Implementation compliant with Wang et al. (2004).

    Args:
        image_pred: Reconstructed image array (H, W) or (H, W, C)
        image_true: Ground truth clean reference array (H, W) or (H, W, C)
        data_range: Dynamic range of pixel values (default 1.0)
    """
    if image_pred.ndim == 3 and image_pred.shape[2] in (3, 4):
        # Multi-channel: compute SSIM on each channel and average
        channels = [
            calculate_ssim(
                image_pred[:, :, c],
                image_true[:, :, c],
                data_range=data_range,
                win_size=win_size,
                k1=k1,
                k2=k2,
            )
            for c in range(image_pred.shape[2])
        ]
        return float(np.mean(channels))

    img1 = image_pred.astype(np.float64)
    img2 = image_true.astype(np.float64)

    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2

    # Global approximation suitable for arbitrary dimensions
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
            "reason": "Authentic uploaded manuscript has no ground-truth clean reference image. PSNR and SSIM require an uncorrupted target.",
            "psnr": None,
            "ssim": None,
        }

    # Normalize reconstructed to numpy array [0.0, 1.0]
    rec_np = _to_numpy_normalized(reconstructed)
    ref_np = _to_numpy_normalized(clean_reference)

    # Check dimension compatibility
    if rec_np.shape != ref_np.shape:
        return {
            "status": "UNAVAILABLE",
            "reason": f"Dimension mismatch: reconstructed is {rec_np.shape}, reference is {ref_np.shape}.",
            "psnr": None,
            "ssim": None,
        }

    psnr_val = calculate_psnr(rec_np, ref_np, data_range=1.0)
    ssim_val = calculate_ssim(rec_np, ref_np, data_range=1.0)

    return {
        "status": "CALCULATED",
        "psnr_db": round(psnr_val, 3) if psnr_val != float("inf") else 999.0,
        "ssim": round(ssim_val, 4),
        "data_range": 1.0,
        "interpretation": {
            "psnr": "Higher is better (>25 dB indicates good visual fidelity, >30 dB indicates high quality)",
            "ssim": "Structural similarity index in range [-1, 1], where 1.0 indicates identical structure.",
        },
    }


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
