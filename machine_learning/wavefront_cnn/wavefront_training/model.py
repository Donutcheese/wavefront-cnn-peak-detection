"""Residual 1-D U-Net for sample-level wavefront heatmap prediction."""

from __future__ import annotations

import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, stride: int = 1, dilation: int = 1) -> None:
        super().__init__()
        padding = dilation
        self.main = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 3, stride=stride, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, 3, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm1d(out_channels),
        )
        self.skip = (
            nn.Identity()
            if in_channels == out_channels and stride == 1
            else nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        )
        self.activation = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.main(x) + self.skip(x))


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.fuse = ResidualBlock(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if not torch.jit.is_tracing() and x.shape[-1] != skip.shape[-1]:
            raise ValueError("decoder and skip lengths differ; input length must be divisible by 8")
        return self.fuse(torch.cat((x, skip), dim=1))


class WavefrontResUNet(nn.Module):
    """Three-level residual encoder-decoder with dilated temporal context."""

    def __init__(self, base_channels: int = 32, bottleneck_dilations: tuple[int, ...] = (1, 2, 4, 8)) -> None:
        super().__init__()
        c1, c2, c3, c4 = base_channels, base_channels * 2, base_channels * 4, base_channels * 8
        self.stem = nn.Sequential(
            nn.Conv1d(1, c1, 7, padding=3, bias=False),
            nn.BatchNorm1d(c1),
            nn.SiLU(inplace=True),
        )
        self.encoder1 = ResidualBlock(c1, c2, stride=2)
        self.encoder2 = ResidualBlock(c2, c3, stride=2)
        self.encoder3 = ResidualBlock(c3, c4, stride=2)
        self.bottleneck = nn.Sequential(
            *(ResidualBlock(c4, c4, dilation=dilation) for dilation in bottleneck_dilations)
        )
        self.decoder3 = DecoderBlock(c4, c3, c3)
        self.decoder2 = DecoderBlock(c3, c2, c2)
        self.decoder1 = DecoderBlock(c2, c1, c1)
        self.head = nn.Conv1d(c1, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not torch.jit.is_tracing():
            if x.ndim != 3 or x.shape[1] != 1:
                raise ValueError(f"expected [batch, 1, samples], got {tuple(x.shape)}")
            if x.shape[-1] % 8 != 0:
                raise ValueError("sample length must be divisible by 8")
        stem = self.stem(x)
        enc1 = self.encoder1(stem)
        enc2 = self.encoder2(enc1)
        encoded = self.encoder3(enc2)
        encoded = self.bottleneck(encoded)
        decoded = self.decoder3(encoded, enc2)
        decoded = self.decoder2(decoded, enc1)
        decoded = self.decoder1(decoded, stem)
        return self.head(decoded)


class CoarseFineWavefrontNet(nn.Module):
    """Classify a coarse temporal bin, then regress an offset inside that bin."""

    def __init__(
        self,
        base_channels: int = 32,
        num_bins: int = 256,
        window_samples: int = 8192,
        bottleneck_dilations: tuple[int, ...] = (1, 2, 4, 8),
    ) -> None:
        super().__init__()
        if window_samples % num_bins != 0:
            raise ValueError("window_samples must be divisible by num_bins")
        c1, c2, c3, c4 = base_channels, base_channels * 2, base_channels * 4, base_channels * 8
        self.num_bins = num_bins
        self.window_samples = window_samples
        self.offset_limit = (window_samples / num_bins - 1.0) / 2.0
        self.stem = nn.Sequential(
            nn.Conv1d(1, c1, 7, padding=3, bias=False),
            nn.BatchNorm1d(c1),
            nn.SiLU(inplace=True),
        )
        self.encoder1 = ResidualBlock(c1, c2, stride=2)
        self.encoder2 = ResidualBlock(c2, c3, stride=2)
        self.encoder3 = ResidualBlock(c3, c4, stride=2)
        self.bottleneck = nn.Sequential(
            *(ResidualBlock(c4, c4, dilation=dilation) for dilation in bottleneck_dilations)
        )
        self.bin_pool = nn.AdaptiveAvgPool1d(num_bins)
        self.coarse_head = nn.Conv1d(c4, 1, kernel_size=1)
        self.offset_head = nn.Conv1d(c4, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 3 or x.shape[1] != 1:
            raise ValueError(f"expected [batch, 1, samples], got {tuple(x.shape)}")
        features = self.stem(x)
        features = self.encoder1(features)
        features = self.encoder2(features)
        features = self.encoder3(features)
        features = self.bottleneck(features)
        features = self.bin_pool(features)
        coarse_logits = self.coarse_head(features).squeeze(1)
        raw_offsets = self.offset_head(features).squeeze(1)
        offsets = torch.tanh(raw_offsets / self.offset_limit) * self.offset_limit
        return coarse_logits, offsets
