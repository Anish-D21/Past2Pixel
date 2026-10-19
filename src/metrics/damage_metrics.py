"""
Damage Localization & Segmentation Evaluation Metrics.
Calculates IoU, Dice Coefficient, Precision, and Recall when ground-truth masks exist.
Reports UNAVAILABLE for real unannotated images.
"""
from typing import Dict, Any, Optional, Union
import numpy as np
import torch


def calculate_segmentation_metrics(
    pred_mask: Union[np.ndarray, torch.Tensor],
    true_mask: Optional[Union[np.ndarray, torch.Tensor]],
    threshold: float = 0.5,
    smooth: float = 1e-6,
) -> Dict[str, Any]:
    """
    Evaluates damage detection output against ground-truth damage mask.

    Args:
        pred_mask: Continuous probability map or binary prediction [0.0, 1.0]
        true_mask: Ground-truth binary damage mask [0.0, 1.0]
        threshold: Binarization cutoff
        smooth: Epsilon to avoid division by zero
    """
    if true_mask is None:
        return {
            "status": "UNAVAILABLE",
            "reason": "Authentic manuscript image lacks ground-truth damage annotations. Segmentation metrics (IoU, Dice) require reference masks.",
            "iou": None,
            "dice": None,
            "precision": None,
            "recall": None,
        }

    # Format pred_mask as binary numpy array (H, W)
    if isinstance(pred_mask, torch.Tensor):
        p_arr = pred_mask.detach().cpu().squeeze().numpy()
    else:
        p_arr = pred_mask.squeeze()
    pred_bin = (p_arr >= threshold).astype(bool)

    # Format true_mask as binary numpy array (H, W)
    if isinstance(true_mask, torch.Tensor):
        t_arr = true_mask.detach().cpu().squeeze().numpy()
    else:
        t_arr = true_mask.squeeze()
    true_bin = (t_arr >= threshold).astype(bool)

    intersection = np.logical_and(pred_bin, true_bin).sum()
    union = np.logical_or(pred_bin, true_bin).sum()

    iou = (intersection + smooth) / (union + smooth)
    dice = (2.0 * intersection + smooth) / (pred_bin.sum() + true_bin.sum() + smooth)

    true_positives = intersection
    false_positives = np.logical_and(pred_bin, ~true_bin).sum()
    false_negatives = np.logical_and(~pred_bin, true_bin).sum()

    precision = (true_positives + smooth) / (true_positives + false_positives + smooth)
    recall = (true_positives + smooth) / (true_positives + false_negatives + smooth)

    return {
        "status": "CALCULATED",
        "threshold": threshold,
        "iou": round(float(iou), 4),
        "dice": round(float(dice), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "damaged_area_predicted_ratio": round(float(pred_bin.mean()), 4),
        "damaged_area_true_ratio": round(float(true_bin.mean()), 4),
        "interpretation": {
            "iou": "Intersection-over-Union (Jaccard Index) ranges [0, 1]. >0.5 indicates good localization.",
            "dice": "Dice Similarity Coefficient (F1-score) harmonic mean of precision and recall.",
        },
    }
