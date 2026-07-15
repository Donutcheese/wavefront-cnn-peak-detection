"""Training primitives shared by transfer-learning stages and tests."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Iterable

import torch

from .metrics import regression_metrics, subsample_peak


def resolve_device(requested: str) -> torch.device:
    requested = requested.lower()
    if requested == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available in this Python environment")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if requested not in {"cpu", "mps", "cuda"}:
        raise ValueError("device must be one of auto, cpu, mps, cuda")
    return torch.device(requested)


def freeze_encoder(model: torch.nn.Module, *, frozen: bool) -> None:
    for name in ("stem", "encoder1", "encoder2", "encoder3"):
        for parameter in getattr(model, name).parameters():
            parameter.requires_grad = not frozen


def _keep_frozen_encoder_in_eval_mode(model: torch.nn.Module) -> None:
    """Prevent BatchNorm statistics from drifting while an encoder is frozen."""
    for name in ("stem", "encoder1", "encoder2", "encoder3"):
        module = getattr(model, name)
        if not any(parameter.requires_grad for parameter in module.parameters()):
            module.eval()


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    *,
    epoch: int,
    best_metric: float,
    config: dict,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "epoch": epoch,
            "best_metric": best_metric,
            "config": config,
        },
        path,
    )


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    *,
    device: torch.device,
) -> dict:
    state = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    if optimizer is not None and state.get("optimizer") is not None:
        optimizer.load_state_dict(state["optimizer"])
    return state


def _autocast(device: torch.device, enabled: bool):
    if not enabled or device.type == "cpu":
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=torch.float16)


def train_one_epoch(
    model: torch.nn.Module,
    loader: Iterable[dict],
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    scaler,
    sampling_rate_hz: float,
    amp: bool = False,
    grad_clip: float = 5.0,
    max_batches: int | None = None,
) -> dict[str, float | int]:
    model.train()
    _keep_frozen_encoder_in_eval_mode(model)
    losses: list[float] = []
    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        signal = batch["signal"].to(device)
        target = batch["target"].to(device)
        coordinate = batch["coordinate"].to(device)
        sample_weight = batch["sample_weight"].to(device)
        optimizer.zero_grad(set_to_none=True)
        with _autocast(device, amp):
            logits = model(signal)
            loss = criterion(logits, target, coordinate, sample_weight)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        losses.append(float(loss.detach().cpu()))
        predictions.append(subsample_peak(torch.sigmoid(logits.detach())).cpu())
        targets.append(coordinate.detach().cpu())

    metrics = regression_metrics(torch.cat(predictions), torch.cat(targets), sampling_rate_hz)
    metrics["loss"] = sum(losses) / len(losses)
    return metrics


@torch.inference_mode()
def evaluate_model(
    model: torch.nn.Module,
    loader: Iterable[dict],
    criterion: torch.nn.Module,
    *,
    device: torch.device,
    sampling_rate_hz: float,
    amp: bool = False,
    return_rows: bool = False,
    max_batches: int | None = None,
) -> tuple[dict[str, float | int], list[dict]]:
    model.eval()
    losses: list[float] = []
    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    rows: list[dict] = []
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        signal = batch["signal"].to(device)
        target = batch["target"].to(device)
        coordinate = batch["coordinate"].to(device)
        sample_weight = batch["sample_weight"].to(device)
        with _autocast(device, amp):
            logits = model(signal)
            loss = criterion(logits, target, coordinate, sample_weight)
        predicted = subsample_peak(torch.sigmoid(logits.float()))
        losses.append(float(loss.cpu()))
        predictions.append(predicted.cpu())
        targets.append(coordinate.cpu())
        if return_rows:
            for sample_id, pred, truth, status in zip(
                batch["sample_id"], predicted.cpu().tolist(), coordinate.cpu().tolist(), batch["label_status"], strict=True
            ):
                rows.append(
                    {
                        "sample_id": sample_id,
                        "predicted_index": pred,
                        "target_index": truth,
                        "absolute_error_samples": abs(pred - truth),
                        "label_status": status,
                    }
                )
    metrics = regression_metrics(torch.cat(predictions), torch.cat(targets), sampling_rate_hz)
    metrics["loss"] = sum(losses) / len(losses)
    return metrics, rows
