"""Compound dense and coordinate loss for wavefront localization."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class WavefrontLoss(nn.Module):
    def __init__(
        self,
        *,
        focal_weight: float = 1.0,
        dice_weight: float = 0.5,
        coordinate_weight: float = 0.05,
        focal_alpha: float = 0.75,
        focal_gamma: float = 2.0,
        softargmax_temperature: float = 0.15,
        coordinate_scale: float = 64.0,
    ) -> None:
        super().__init__()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.coordinate_weight = coordinate_weight
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.softargmax_temperature = softargmax_temperature
        self.coordinate_scale = coordinate_scale

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        coordinate: torch.Tensor,
        sample_weight: torch.Tensor,
    ) -> torch.Tensor:
        probability = torch.sigmoid(logits)
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        pt = probability * target + (1.0 - probability) * (1.0 - target)
        alpha = self.focal_alpha * target + (1.0 - self.focal_alpha) * (1.0 - target)
        focal = (alpha * (1.0 - pt).pow(self.focal_gamma) * bce).mean(dim=(1, 2))

        intersection = (probability * target).sum(dim=(1, 2))
        denominator = probability.sum(dim=(1, 2)) + target.sum(dim=(1, 2))
        dice = 1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)

        positions = torch.arange(logits.shape[-1], device=logits.device, dtype=logits.dtype)
        distribution = torch.softmax(logits.squeeze(1) / self.softargmax_temperature, dim=-1)
        predicted_coordinate = (distribution * positions).sum(dim=-1)
        coordinate_loss = F.smooth_l1_loss(
            predicted_coordinate, coordinate, reduction="none", beta=4.0
        ) / self.coordinate_scale

        per_sample = (
            self.focal_weight * focal
            + self.dice_weight * dice
            + self.coordinate_weight * coordinate_loss
        )
        return (per_sample * sample_weight).sum() / sample_weight.sum().clamp_min(1e-6)


class CoarseFineLoss(nn.Module):
    """Soft bin classification plus true-bin offset regression."""

    def __init__(
        self,
        *,
        window_samples: int = 8192,
        num_bins: int = 256,
        bin_sigma: float = 0.35,
        offset_weight: float = 0.75,
        coordinate_weight: float = 0.05,
    ) -> None:
        super().__init__()
        if window_samples % num_bins != 0:
            raise ValueError("window_samples must be divisible by num_bins")
        self.window_samples = window_samples
        self.num_bins = num_bins
        self.bin_width = window_samples / num_bins
        self.bin_sigma = bin_sigma
        self.offset_weight = offset_weight
        self.coordinate_weight = coordinate_weight

    def forward(
        self,
        coarse_logits: torch.Tensor,
        offsets: torch.Tensor,
        coordinate: torch.Tensor,
        sample_weight: torch.Tensor,
    ) -> torch.Tensor:
        bin_positions = torch.arange(
            self.num_bins, device=coarse_logits.device, dtype=coarse_logits.dtype
        )
        continuous_bin = coordinate / self.bin_width - 0.5
        target_distribution = torch.exp(
            -0.5 * ((bin_positions[None, :] - continuous_bin[:, None]) / self.bin_sigma) ** 2
        )
        target_distribution /= target_distribution.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        classification_loss = -(
            target_distribution * torch.log_softmax(coarse_logits, dim=-1)
        ).sum(dim=-1)

        true_bin = torch.floor(coordinate / self.bin_width).long().clamp(0, self.num_bins - 1)
        batch = torch.arange(coarse_logits.shape[0], device=coarse_logits.device)
        selected_offset = offsets[batch, true_bin]
        bin_center = (true_bin.to(coordinate.dtype) + 0.5) * self.bin_width - 0.5
        true_offset = coordinate - bin_center
        offset_loss = F.smooth_l1_loss(
            selected_offset, true_offset, reduction="none", beta=1.0
        ) / (self.bin_width / 2.0)

        probabilities = torch.softmax(coarse_logits, dim=-1)
        centers = (bin_positions + 0.5) * self.bin_width - 0.5
        expected_coordinate = (probabilities * (centers[None, :] + offsets)).sum(dim=-1)
        coordinate_loss = F.smooth_l1_loss(
            expected_coordinate, coordinate, reduction="none", beta=self.bin_width
        ) / self.bin_width

        per_sample = (
            classification_loss
            + self.offset_weight * offset_loss
            + self.coordinate_weight * coordinate_loss
        )
        return (per_sample * sample_weight).sum() / sample_weight.sum().clamp_min(1e-6)
