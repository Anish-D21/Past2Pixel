"""
Dataset A Loader: VedoOCR — Vedic Sanskrit Manuscript OCR Dataset.
Handles loading manuscript images, applying controlled synthetic degradation pairs,
and formatting PyTorch tensors for restoration and damage localization.
"""
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

from .synthetic_degradation import SyntheticDegrader
from ..core.config import settings


class ManuscriptPairDataset(Dataset):
    """
    PyTorch Dataset returning triplets of:
    - Degraded Image Tensor: (C, H, W) in [0, 1]
    - Clean Ground-Truth Image Tensor: (C, H, W) in [0, 1]
    - Ground-Truth Damage Mask Tensor: (1, H, W) in [0, 1]
    - Metadata dict (filename, document ID, applied degradation history)
    """

    def __init__(
        self,
        image_paths: List[Path],
        target_size: Tuple[int, int] = (256, 256),
        is_training: bool = True,
        seed: int = 42,
    ):
        self.image_paths = image_paths
        self.target_size = target_size
        self.is_training = is_training
        self.degrader = SyntheticDegrader(seed=seed)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_path = self.image_paths[idx]
        with Image.open(img_path) as img:
            clean_pil = img.convert("RGB")

        # Resize to standardized target size
        clean_pil = clean_pil.resize(self.target_size, Image.Resampling.BILINEAR)

        # Apply synthetic degradation and capture ground-truth damage mask
        degraded_pil, damage_mask_np, deg_meta = self.degrader.apply_pipeline(
            clean_pil
        )

        # Convert to PyTorch tensors [C, H, W] in range [0.0, 1.0]
        clean_tensor = TF.to_tensor(clean_pil)
        degraded_tensor = TF.to_tensor(degraded_pil)

        # Mask tensor [1, H, W]
        mask_tensor = torch.from_numpy(damage_mask_np).unsqueeze(0).float()

        doc_id = img_path.stem.split("_")[0]

        return {
            "degraded": degraded_tensor,
            "clean": clean_tensor,
            "damage_mask": mask_tensor,
            "filename": img_path.name,
            "document_id": doc_id,
            "metadata": deg_meta,
        }


def generate_synthetic_manuscript_sample(
    output_path: Path,
    width: int = 512,
    height: int = 512,
    seed: int = 42,
) -> Path:
    """
    Generates a synthetic parchment-style clean manuscript page with Devanagari text
    to allow immediate pipeline development and CPU smoke-testing before full dataset download.
    """
    rng = np.random.default_rng(seed)
    # Warm golden/sepia parchment base
    base_color = np.ones((height, width, 3), dtype=np.uint8)
    base_color[:, :, 0] = rng.integers(235, 245)  # R
    base_color[:, :, 1] = rng.integers(215, 230)  # G
    base_color[:, :, 2] = rng.integers(180, 205)  # B

    # Subtle texture grain
    grain = rng.integers(-8, 8, (height, width, 3))
    parchment = np.clip(base_color + grain, 0, 255).astype(np.uint8)
    img = Image.fromarray(parchment)
    draw = ImageDraw.Draw(img)

    # Sanskrit / Devanagari sample text lines
    sample_lines = [
        "॥ ॐ तत्सद्ब्रह्मणे नमः ॥",
        "अग्निमीळे पुरोहितं यज्ञस्य देवमृत्विजम् ।",
        "होतारं रत्नधातमम् ॥ १ ॥",
        "अग्निः पूर्वेभिरृषिभिरीड्यो नूतनैरुत ।",
        "स देवाँ एह वक्षति ॥ २ ॥",
        "इषे त्वोर्जे त्वा वायव स्थ देवो वः सविता ।",
        "प्रार्पयतु श्रेष्ठतमाय कर्मणे ॥ ३ ॥",
    ]

    margin_x = 40
    start_y = 50
    line_spacing = 55
    ink_color = (40, 25, 20)

    for i, line in enumerate(sample_lines):
        y = start_y + (i * line_spacing)
        draw.text((margin_x, y), line, fill=ink_color)

    # Red rubrication marks at margin
    draw.rectangle([margin_x - 15, start_y, margin_x - 10, start_y + 350], fill=(160, 40, 30))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)
    return output_path
