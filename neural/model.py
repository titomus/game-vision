"""Small, original convolutional denoiser for GameVision feasibility experiments.

This is deliberately NOT NVIDIA DLSS, a diffusion model, or a photorealistic
material relighting model. Train it on permitted images before judging quality.
"""
from __future__ import annotations
import torch
from torch import nn


class DepthwiseBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class TinyDenoiser(nn.Module):
    """Residual denoiser, input and output NCHW float RGB in [0, 1].

    Only convolutions and pointwise ops; no spatial resize in the network.
    About a thousand learnable parameters at default width=8.
    """

    def __init__(self, width: int = 8, blocks: int = 2):
        super().__init__()
        if width < 4 or width > 64 or not 1 <= blocks <= 8:
            raise ValueError("width must be 4..64 and blocks 1..8")
        self.width = width
        self.blocks = blocks
        self.features = nn.Sequential(
            nn.Conv2d(3, width, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            *(DepthwiseBlock(width) for _ in range(blocks)),
            nn.Conv2d(width, 3, kernel_size=3, padding=1),
        )
        # Start at exact identity. Only training creates meaningful denoising.
        nn.init.zeros_(self.features[-1].weight)
        nn.init.zeros_(self.features[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x - self.features(x)


def pipeline(model: nn.Module, image: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """GPU-resident resize -> network -> resize, no host readback or IO."""
    import torch.nn.functional as F
    if not 0 < scale <= 1:
        raise ValueError("scale must be within (0, 1]")
    if scale == 1:
        work = image
    else:
        h, w = image.shape[-2:]
        work = F.interpolate(image, size=(max(16, int(h * scale)), max(16, int(w * scale))),
                             mode="bilinear", align_corners=False)
    result = model(work)
    if scale != 1:
        result = F.interpolate(result, size=image.shape[-2:],
                               mode="bilinear", align_corners=False)
    return result.clamp(0, 1)
