"""Two-stage hard+soft to hard-only transfer training entry point."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import TrainingConfig
from .data import create_dataloaders
from .engine import (
    evaluate_model,
    freeze_encoder,
    load_checkpoint,
    resolve_device,
    save_checkpoint,
    train_one_epoch,
)
from .losses import WavefrontLoss
from .model import WavefrontResUNet


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def _fit_stage(
    *,
    stage: str,
    model: WavefrontResUNet,
    loaders: dict,
    config: TrainingConfig,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    output_dir: Path,
    freeze_epochs: int,
) -> Path:
    criterion = WavefrontLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    checkpoint = output_dir / f"best_{stage}.pt"
    best_mae = float("inf")
    stale_epochs = 0
    history: list[dict] = []

    for epoch in range(1, epochs + 1):
        if freeze_epochs:
            freeze_encoder(model, frozen=epoch <= freeze_epochs)
        train_metrics = train_one_epoch(
            model,
            loaders["train"],
            criterion,
            optimizer,
            device=device,
            scaler=None,
            sampling_rate_hz=config.sampling_rate_hz,
            amp=config.amp,
            max_batches=config.max_batches,
        )
        val_metrics, _ = evaluate_model(
            model,
            loaders["val"],
            criterion,
            device=device,
            sampling_rate_hz=config.sampling_rate_hz,
            amp=config.amp,
            max_batches=config.max_batches,
        )
        val_mae = float(val_metrics["mae_samples"])
        scheduler.step(val_mae)
        row = {
            "stage": stage,
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        history.append(row)
        pd.DataFrame(history).to_csv(output_dir / f"history_{stage}.csv", index=False)
        print(
            f"[{stage}] epoch={epoch:03d} train_loss={train_metrics['loss']:.5f} "
            f"val_mae={val_mae:.3f} samples val_p95={val_metrics['p95_samples']:.3f}"
        )

        if val_mae < best_mae:
            best_mae = val_mae
            stale_epochs = 0
            save_checkpoint(
                checkpoint,
                model,
                optimizer,
                epoch=epoch,
                best_metric=best_mae,
                config=config.to_dict(),
            )
        else:
            stale_epochs += 1
            if stale_epochs >= config.patience:
                print(f"[{stage}] early stopping after {epoch} epochs")
                break
    return checkpoint


def run_transfer_training(config: TrainingConfig) -> dict:
    set_reproducible_seed(config.seed)
    device = resolve_device(config.device)
    output_dir = config.ensure_output_dir()
    print(f"execution_device={device}")
    if device.type == "mps":
        print(f"mps_allocated_bytes={torch.mps.current_allocated_memory()}")

    model = WavefrontResUNet(
        base_channels=config.base_channels,
        bottleneck_dilations=config.bottleneck_dilations,
    ).to(device)

    pretrain_loaders = create_dataloaders(
        config.dataset_dir,
        statuses=("hard", "soft"),
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        heatmap_sigma=config.heatmap_sigma,
        seed=config.seed,
    )
    pretrain_checkpoint = _fit_stage(
        stage="pretrain",
        model=model,
        loaders=pretrain_loaders,
        config=config,
        device=device,
        epochs=config.pretrain_epochs,
        learning_rate=config.pretrain_lr,
        output_dir=output_dir,
        freeze_epochs=0,
    )

    load_checkpoint(pretrain_checkpoint, model, optimizer=None, device=device)
    hard_loaders = create_dataloaders(
        config.dataset_dir,
        statuses=("hard",),
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        heatmap_sigma=config.heatmap_sigma,
        seed=config.seed + 1,
    )
    finetune_checkpoint = _fit_stage(
        stage="finetune",
        model=model,
        loaders=hard_loaders,
        config=config,
        device=device,
        epochs=config.finetune_epochs,
        learning_rate=config.finetune_lr,
        output_dir=output_dir,
        freeze_epochs=config.freeze_epochs,
    )

    load_checkpoint(finetune_checkpoint, model, optimizer=None, device=device)
    criterion = WavefrontLoss()
    test_metrics, rows = evaluate_model(
        model,
        hard_loaders["test"],
        criterion,
        device=device,
        sampling_rate_hz=config.sampling_rate_hz,
        amp=config.amp,
        return_rows=True,
        max_batches=config.max_batches,
    )
    pd.DataFrame(rows).to_csv(output_dir / "test_predictions.csv", index=False)
    report = {
        "device": str(device),
        "pretrain_checkpoint": str(pretrain_checkpoint.resolve()),
        "finetune_checkpoint": str(finetune_checkpoint.resolve()),
        "test_metrics": test_metrics,
        "config": config.to_dict(),
    }
    (output_dir / "test_metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def _parse_args() -> TrainingConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default="data/derived/wavefront_dataset_v1")
    parser.add_argument("--output-dir", default="data/derived/wavefront_cnn_run")
    parser.add_argument("--device", choices=("auto", "mps", "cpu", "cuda"), default="mps")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--pretrain-epochs", type=int, default=40)
    parser.add_argument("--finetune-epochs", type=int, default=60)
    parser.add_argument("--freeze-epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="run one batch and one epoch per stage")
    args = parser.parse_args()
    if args.smoke:
        args.base_channels = 8
        args.batch_size = min(args.batch_size, 4)
        args.pretrain_epochs = 1
        args.finetune_epochs = 1
        args.freeze_epochs = 1
    return TrainingConfig(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        device=args.device,
        seed=args.seed,
        base_channels=args.base_channels,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pretrain_epochs=args.pretrain_epochs,
        finetune_epochs=args.finetune_epochs,
        freeze_epochs=args.freeze_epochs,
        patience=args.patience,
        amp=args.amp,
        max_batches=1 if args.smoke else None,
    )


if __name__ == "__main__":
    run_transfer_training(_parse_args())

