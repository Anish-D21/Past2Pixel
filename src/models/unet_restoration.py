"""
Primary U-Net Architecture for Ancient Manuscript Image Restoration.
Employs encoder-decoder paths with cross-level skip connections to preserve
fine historical ink contours, Devanagari ligatures, and parchment texturing.
"""
from typing import Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(Convolution => [BatchNorm] => LeakyReLU) * 2"""

    def __init__(self, in_channels: int, out_channels: int, mid_channels: int = None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv with skip connection concatenation"""

    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # Input shape handling for padding if dimensions are odd
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        # Skip connection: concatenate along channel dimension
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNetRestoration(nn.Module):
    """
    Primary U-Net Restoration Network.
    
    Architecture:
    - Contracting Path: 4 downsampling blocks with feature maps: 32 -> 64 -> 128 -> 256
    - Bottleneck: 512 feature maps with dropout to mitigate overfitting
    - Expanding Path: 4 upsampling blocks with concatenation skip connections: 512 -> 256 -> 128 -> 64 -> 32
    - Output Head: 1x1 convolution mapping to output channels, activated by Sigmoid [0.0, 1.0]
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 3, base_filters: int = 32, bilinear: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
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
        self.outc = nn.Sequential(
            nn.Conv2d(base_filters, out_channels, kernel_size=1),
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
        logits = self.outc(x)
        return logits

    def get_model_info(self) -> Dict[str, Any]:
        """Returns architecture metadata and parameter counts."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "model_name": "UNetRestoration",
            "type": "Primary Image Restoration",
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "base_filters": self.base_filters,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "has_skip_connections": True,
            "description": "Encoder-decoder with direct skip connections that concatenate low-level edge features into upsampled reconstructions.",
        }
