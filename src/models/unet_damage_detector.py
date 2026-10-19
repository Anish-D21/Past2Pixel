"""
U-Net Damage Localization & Segmentation Network.
Takes an uploaded manuscript image and outputs a pixel-level damage probability mask
indicating tears, stains, fading, scratches, and missing regions.
"""
from typing import Dict, Any, Tuple
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from .unet_restoration import DoubleConv, Down, Up


class UNetDamageDetector(nn.Module):
    """
    U-Net Semantic Segmentation Architecture for Damage Localization.
    
    Input shape:  (B, 3, H, W) RGB manuscript image
    Output shape: (B, 1, H, W) Per-pixel damage probability in range [0.0, 1.0]
    """

    def __init__(self, in_channels: int = 3, base_filters: int = 32, bilinear: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.base_filters = base_filters
        self.bilinear = bilinear

        self.inc = DoubleConv(in_channels, base_filters)
        self.down1 = Down(base_filters, base_filters * 2)
        self.down2 = Down(base_filters * 2, base_filters * 4)
        self.down3 = Down(base_filters * 4, base_filters * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_filters * 8, (base_filters * 16) // factor)

        self.up1 = Up(base_filters * 16, (base_filters * 8) // factor, bilinear)
        self.up2 = Up(base_filters * 8, (base_filters * 4) // factor, bilinear)
        self.up3 = Up(base_filters * 4, (base_filters * 2) // factor, bilinear)
        self.up4 = Up(base_filters * 2, base_filters, bilinear)

        # Output single-channel binary segmentation map
        self.outc = nn.Sequential(
            nn.Conv2d(base_filters, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        probability_map = self.outc(x)
        return probability_map

    def predict_mask(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Runs forward pass and applies threshold to generate a binary mask."""
        self.eval()
        with torch.no_grad():
            prob = self.forward(x)
            binary_mask = (prob >= threshold).float()
            return binary_mask

    def get_model_info(self) -> Dict[str, Any]:
        """Returns architecture metadata and parameter counts."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "model_name": "UNetDamageDetector",
            "type": "Damage Localization / Segmentation",
            "in_channels": self.in_channels,
            "out_channels": 1,
            "base_filters": self.base_filters,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "has_skip_connections": True,
            "description": "U-Net segmentation network providing pixel-level probability masks for damage and surface degradation.",
        }


def generate_damage_overlay(
    original_image: Image.Image,
    damage_mask: np.ndarray,
    overlay_color: Tuple[int, int, int] = (230, 50, 40),
    alpha: float = 0.45,
    threshold: float = 0.3,
) -> Image.Image:
    """
    Renders an overlay highlighting damaged regions in translucent red/crimson.

    Args:
        original_image: PIL Image of the original manuscript
        damage_mask: np.ndarray (H, W) with damage probabilities in [0.0, 1.0]
        overlay_color: RGB tuple for highlight color
        alpha: Maximum transparency of the highlight
        threshold: Minimum damage score to apply overlay

    Returns:
        PIL.Image of original with damage highlight overlay
    """
    img_rgb = original_image.convert("RGB")
    w, h = img_rgb.size

    # Ensure mask matches image dimensions
    if damage_mask.shape != (h, w):
        mask_pil = Image.fromarray((damage_mask * 255).astype(np.uint8))
        mask_pil = mask_pil.resize((w, h), Image.Resampling.BILINEAR)
        mask_np = np.array(mask_pil, dtype=np.float32) / 255.0
    else:
        mask_np = damage_mask

    img_arr = np.array(img_rgb, dtype=np.float32)
    overlay_arr = img_arr.copy()

    # Apply overlay where mask exceeds threshold
    active = mask_np > threshold
    for c in range(3):
        color_val = overlay_color[c]
        effective_alpha = np.clip((mask_np - threshold) / (1.0 - threshold), 0.0, 1.0) * alpha
        overlay_arr[:, :, c] = np.where(
            active,
            (1.0 - effective_alpha) * img_arr[:, :, c] + effective_alpha * color_val,
            img_arr[:, :, c],
        )

    return Image.fromarray(np.clip(overlay_arr, 0, 255).astype(np.uint8))
