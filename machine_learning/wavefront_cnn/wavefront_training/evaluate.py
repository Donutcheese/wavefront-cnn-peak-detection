"""Evaluate a trained hard-label checkpoint on validation or test data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from .data import create_dataloaders
from .engine import evaluate_model, load_checkpoint, resolve_device
from .losses import WavefrontLoss
from .model import WavefrontResUNet


def evaluate_checkpoint(
    checkpoint: str | Path,
    dataset_dir: str | Path,
    *,
    split: str,
    device_name: str,
    output_dir: str | Path,
) -> dict:
    device = resolve_device(device_name)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = state["config"]
    model = WavefrontResUNet(
        base_channels=int(config["base_channels"]),
        bottleneck_dilations=tuple(config["bottleneck_dilations"]),
    ).to(device)
    load_checkpoint(checkpoint, model, optimizer=None, device=device)
    loaders = create_dataloaders(
        dataset_dir,
        statuses=("hard",),
        batch_size=int(config.get("batch_size", 32)),
        num_workers=0,
        heatmap_sigma=float(config.get("heatmap_sigma", 6.0)),
        seed=int(config.get("seed", 42)),
    )
    metrics, rows = evaluate_model(
        model,
        loaders[split],
        WavefrontLoss(),
        device=device,
        sampling_rate_hz=float(config.get("sampling_rate_hz", 1_250_000.0)),
        return_rows=True,
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_dir / f"{split}_predictions.csv", index=False)
    (output_dir / f"{split}_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-dir", default="data/derived/wavefront_dataset_v1")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--device", choices=("auto", "mps", "cpu", "cuda"), default="mps")
    parser.add_argument("--output-dir", default="data/derived/wavefront_cnn_run/evaluation")
    args = parser.parse_args()
    print(json.dumps(evaluate_checkpoint(
        args.checkpoint,
        args.dataset_dir,
        split=args.split,
        device_name=args.device,
        output_dir=args.output_dir,
    ), indent=2))


if __name__ == "__main__":
    main()

