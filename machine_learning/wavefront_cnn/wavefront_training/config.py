"""Configuration shared by training, evaluation and ONNX export."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class TrainingConfig:
    dataset_dir: str
    output_dir: str
    device: str = "mps"
    seed: int = 42
    base_channels: int = 32
    bottleneck_dilations: tuple[int, ...] = (1, 2, 4, 8)
    batch_size: int = 32
    num_workers: int = 0
    heatmap_sigma: float = 6.0
    pretrain_epochs: int = 40
    finetune_epochs: int = 60
    freeze_epochs: int = 5
    pretrain_lr: float = 2e-3
    finetune_lr: float = 5e-4
    weight_decay: float = 1e-4
    patience: int = 10
    amp: bool = False
    max_batches: int | None = None
    window_samples: int = 8192
    sampling_rate_hz: float = 1_250_000.0

    def to_dict(self) -> dict:
        result = asdict(self)
        result["bottleneck_dilations"] = list(self.bottleneck_dilations)
        return result

    def ensure_output_dir(self) -> Path:
        path = Path(self.output_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path

