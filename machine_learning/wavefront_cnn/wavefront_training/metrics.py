"""Coordinate extraction and regression metrics."""

from __future__ import annotations

import math

import torch


def decode_coarse_fine(
    coarse_logits: torch.Tensor,
    offsets: torch.Tensor,
    *,
    window_samples: int,
) -> torch.Tensor:
    """Decode the argmax bin and its learned local offset into a sample coordinate."""
    num_bins = coarse_logits.shape[-1]
    if window_samples % num_bins != 0:
        raise ValueError("window_samples must be divisible by the number of bins")
    bin_width = window_samples / num_bins
    selected_bin = coarse_logits.argmax(dim=-1)
    batch = torch.arange(coarse_logits.shape[0], device=coarse_logits.device)
    selected_offset = offsets[batch, selected_bin]
    center = (selected_bin.to(coarse_logits.dtype) + 0.5) * bin_width - 0.5
    return center + selected_offset


def subsample_peak(probability: torch.Tensor) -> torch.Tensor:
    """Refine the discrete maximum with a three-point parabolic fit."""
    values = probability.squeeze(1)
    peak = values.argmax(dim=-1)
    length = values.shape[-1]
    left_index = (peak - 1).clamp(0, length - 1)
    right_index = (peak + 1).clamp(0, length - 1)
    batch = torch.arange(values.shape[0], device=values.device)
    left = values[batch, left_index]
    center = values[batch, peak]
    right = values[batch, right_index]
    denominator = left - 2.0 * center + right
    valid = (peak > 0) & (peak < length - 1) & (denominator < -1e-12)
    safe_denominator = torch.where(valid, denominator, -torch.ones_like(denominator))
    offset = 0.5 * (left - right) / safe_denominator
    offset = torch.where(valid, offset.clamp(-0.5, 0.5), torch.zeros_like(offset))
    return peak.to(values.dtype) + offset


def regression_metrics(
    predicted: torch.Tensor,
    target: torch.Tensor,
    sampling_rate_hz: float,
) -> dict[str, float | int]:
    error = (predicted.detach().float().cpu() - target.detach().float().cpu()).abs()
    if error.numel() == 0:
        raise ValueError("metrics require at least one observation")
    sorted_error = error.sort().values
    percentile_index = min(math.ceil(0.95 * error.numel()) - 1, error.numel() - 1)
    result: dict[str, float | int] = {
        "count": int(error.numel()),
        "mae_samples": float(error.mean()),
        "rmse_samples": float(torch.sqrt((error**2).mean())),
        "p95_samples": float(sorted_error[percentile_index]),
    }
    result["mae_us"] = result["mae_samples"] * 1e6 / sampling_rate_hz
    result["p95_us"] = result["p95_samples"] * 1e6 / sampling_rate_hz
    for tolerance in (1, 2, 4, 8, 16):
        result[f"within_{tolerance}_samples"] = float((error <= tolerance).float().mean())
    return result
