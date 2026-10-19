"""
Baseline Convolutional Autoencoder for Manuscript Restoration.
Serves as an empirical baseline against the primary U-Net architecture.
"""
from typing import Dict, Any
import torch
import torch.nn as nn


class ConvAutoencoder(nn.Module):
    """
    Convolutional Autoencoder baseline without skip connections.
    Encodes degraded manuscript into a compact latent bottleneck and reconstructs clean image.

    Input shape:  (B, in_channels, H, W)
    Output shape: (B, in_channels, H, W) in range [0.0, 1.0]
    """

    def __init__(self, in_channels: int = 3, base_filters: int = 32):
        super().__init__()
        self.in_channels = in_channels
        self.base_filters = base_filters

        # Encoder Path
        self.encoder = nn.Sequential(
            # Block 1: H -> H/2
            nn.Conv2d(in_channels, base_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters, base_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block 2: H/2 -> H/4
            nn.Conv2d(base_filters, base_filters * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block 3: H/4 -> H/8
            nn.Conv2d(base_filters * 2, base_filters * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters * 4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_filters * 4, base_filters * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters * 8, base_filters * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters * 4),
            nn.ReLU(inplace=True),
        )

        # Decoder Path
        self.decoder = nn.Sequential(
            # UpBlock 1: H/8 -> H/4
            nn.ConvTranspose2d(base_filters * 4, base_filters * 2, kernel_size=2, stride=2),
            nn.BatchNorm2d(base_filters * 2),
            nn.ReLU(inplace=True),
            # UpBlock 2: H/4 -> H/2
            nn.ConvTranspose2d(base_filters * 2, base_filters, kernel_size=2, stride=2),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(inplace=True),
            # UpBlock 3: H/2 -> H
            nn.ConvTranspose2d(base_filters, base_filters, kernel_size=2, stride=2),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(inplace=True),
            # Output projection to [0, 1]
            nn.Conv2d(base_filters, in_channels, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x)
        bottleneck = self.bottleneck(latent)
        reconstruction = self.decoder(bottleneck)
        return reconstruction

    def get_model_info(self) -> Dict[str, Any]:
        """Returns architecture metadata and parameter counts."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "model_name": "ConvAutoencoder",
            "type": "Baseline Image Restoration",
            "in_channels": self.in_channels,
            "base_filters": self.base_filters,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "has_skip_connections": False,
            "limitations": "Lacks high-resolution skip connections; prone to over-smoothing fine Sanskrit ink strokes and diacritics.",
        }
