"""HDF5-backed phase dataset and label-preserving waveform augmentation."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

PHASE_TO_INDEX = {"A": 0, "B": 1, "C": 2}
STATUS_WEIGHT = {"hard": 1.0, "soft": 0.5}


def build_phase_index(
    labels_csv: str | Path,
    *,
    split: str,
    statuses: Iterable[str],
) -> pd.DataFrame:
    """Return valid phase rows for one precomputed event-level split."""
    allowed_splits = {"train", "val", "test"}
    if split not in allowed_splits:
        raise ValueError(f"split must be one of {sorted(allowed_splits)}, got {split!r}")
    statuses = tuple(statuses)
    unknown = set(statuses) - set(STATUS_WEIGHT)
    if unknown:
        raise ValueError(f"unsupported label statuses: {sorted(unknown)}")

    table = pd.read_csv(labels_csv)
    required = {
        "sample_index",
        "sample_id",
        "phase",
        "window_wavefront_index",
        "confidence",
        "label_status",
        "split_event",
    }
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"phase label table is missing columns: {sorted(missing)}")

    selected = table.loc[
        (table["split_event"] == split)
        & table["label_status"].isin(statuses)
        & (table["window_wavefront_index"] >= 0)
        & table["phase"].isin(PHASE_TO_INDEX)
    ].copy()
    selected["sample_index"] = selected["sample_index"].astype(np.int64)
    selected["window_wavefront_index"] = selected["window_wavefront_index"].astype(np.float32)
    return selected.reset_index(drop=True)


def gaussian_heatmap(length: int, center: float, sigma: float) -> np.ndarray:
    """Create a unit-height Gaussian target centered on a sample coordinate."""
    if length <= 0 or sigma <= 0:
        raise ValueError("length and sigma must be positive")
    positions = np.arange(length, dtype=np.float32)
    target = np.exp(-0.5 * ((positions - np.float32(center)) / np.float32(sigma)) ** 2)
    return target.astype(np.float32, copy=False)


def shift_waveform_and_index(
    signal: np.ndarray,
    coordinate: float,
    shift: int,
) -> tuple[np.ndarray, float]:
    """Zero-pad a temporal shift and move its coordinate by the same amount."""
    output = np.zeros_like(signal)
    if shift > 0:
        output[shift:] = signal[:-shift]
    elif shift < 0:
        output[:shift] = signal[-shift:]
    else:
        output[...] = signal
    return output, float(coordinate + shift)


class WavefrontDataset(Dataset):
    """Lazy per-phase view over the waveform HDF5 dataset."""

    def __init__(
        self,
        h5_path: str | Path,
        index: pd.DataFrame,
        *,
        augment: bool,
        heatmap_sigma: float = 6.0,
        max_shift: int = 192,
        noise_std: float = 0.025,
        baseline_std: float = 0.02,
        polarity_probability: float = 0.5,
    ) -> None:
        self.h5_path = str(Path(h5_path).resolve())
        self.index = index.reset_index(drop=True)
        self.augment = augment
        self.heatmap_sigma = heatmap_sigma
        self.max_shift = max_shift
        self.noise_std = noise_std
        self.baseline_std = baseline_std
        self.polarity_probability = polarity_probability
        self._h5: h5py.File | None = None
        self._h5_pid: int | None = None

        with h5py.File(self.h5_path, "r") as handle:
            if "signals" not in handle:
                raise ValueError("HDF5 dataset does not contain /signals")
            shape = handle["signals"].shape
            if len(shape) != 3 or shape[1] != 3:
                raise ValueError(f"signals must have shape [N, 3, L], got {shape}")
            self.signal_length = int(shape[2])
            self.sampling_rate_hz = float(handle.attrs["target_sampling_rate_hz"])

    def __len__(self) -> int:
        return len(self.index)

    def _handle(self) -> h5py.File:
        pid = os.getpid()
        if self._h5 is None or self._h5_pid != pid:
            if self._h5 is not None:
                self._h5.close()
            self._h5 = h5py.File(self.h5_path, "r", swmr=True)
            self._h5_pid = pid
        return self._h5

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_h5"] = None
        state["_h5_pid"] = None
        return state

    def __del__(self) -> None:
        handle = getattr(self, "_h5", None)
        try:
            if handle is not None and handle.id.valid:
                handle.close()
        except (AttributeError, TypeError, ValueError):
            pass

    def _augment(self, signal: np.ndarray, coordinate: float) -> tuple[np.ndarray, float]:
        lower = max(-self.max_shift, int(np.ceil(-coordinate)))
        upper = min(self.max_shift, int(np.floor(self.signal_length - 1 - coordinate)))
        shift = int(np.random.randint(lower, upper + 1)) if upper >= lower else 0
        signal, coordinate = shift_waveform_and_index(signal, coordinate, shift)

        gain = np.random.uniform(0.75, 1.25)
        polarity = -1.0 if np.random.random() < self.polarity_probability else 1.0
        signal = signal * np.float32(gain * polarity)
        signal += np.random.normal(0.0, self.noise_std, self.signal_length).astype(np.float32)
        drift_slope = np.random.normal(0.0, self.baseline_std)
        signal += np.linspace(-drift_slope, drift_slope, self.signal_length, dtype=np.float32)
        return signal, coordinate

    def __getitem__(self, item: int) -> dict[str, torch.Tensor | str]:
        row = self.index.iloc[item]
        phase_index = PHASE_TO_INDEX[str(row.phase)]
        signal = np.asarray(
            self._handle()["signals"][int(row.sample_index), phase_index, :],
            dtype=np.float32,
        ).copy()
        coordinate = float(row.window_wavefront_index)
        if self.augment:
            signal, coordinate = self._augment(signal, coordinate)

        target = gaussian_heatmap(self.signal_length, coordinate, self.heatmap_sigma)
        return {
            "signal": torch.from_numpy(signal[None, :]),
            "target": torch.from_numpy(target[None, :]),
            "coordinate": torch.tensor(coordinate, dtype=torch.float32),
            "sample_weight": torch.tensor(STATUS_WEIGHT[str(row.label_status)], dtype=torch.float32),
            "sample_id": f"{row.sample_id}:{row.phase}",
            "label_status": str(row.label_status),
        }


def create_dataloaders(
    dataset_dir: str | Path,
    *,
    statuses: tuple[str, ...],
    batch_size: int,
    num_workers: int,
    heatmap_sigma: float,
    seed: int,
) -> dict[str, DataLoader]:
    """Create loaders from immutable event-level split assignments."""
    dataset_dir = Path(dataset_dir)
    labels_csv = dataset_dir / "phase_labels.csv"
    h5_path = dataset_dir / "waveforms.h5"
    generator = torch.Generator().manual_seed(seed)
    loaders: dict[str, DataLoader] = {}
    for split in ("train", "val", "test"):
        index = build_phase_index(labels_csv, split=split, statuses=statuses)
        dataset = WavefrontDataset(
            h5_path,
            index,
            augment=split == "train",
            heatmap_sigma=heatmap_sigma,
        )
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=split == "train",
            num_workers=num_workers,
            pin_memory=False,
            persistent_workers=num_workers > 0,
            generator=generator if split == "train" else None,
        )
    return loaders
