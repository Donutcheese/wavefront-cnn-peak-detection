"""自动伪标签与复核优先级加载。

可选加载数据集 v1 的三类 CSV 作为标注参考（均不修改）：
- phase_labels.csv：自动标注器输出，提供每相 raw_wavefront_index 与各检测器候选；
- review_queue.csv：自动流程判定需要人工复核的样本；
- stage0_worst30.csv：阶段 0 诊断给出的最差样本（最高优先级）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


@dataclass
class AutoPhaseLabel:
    raw_wavefront_index: float | None
    confidence: float | None
    label_status: str
    spread_us: float | None
    detector_indices: dict[str, float] = field(default_factory=dict)


@dataclass
class AutoFileInfo:
    phases: dict[str, AutoPhaseLabel] = field(default_factory=dict)
    in_review_queue: bool = False
    in_worst30: bool = False

    @property
    def priority(self) -> int:
        """数值越小优先级越高：最差样本 0，复核队列 1，其余 2。"""
        if self.in_worst30:
            return 0
        if self.in_review_queue:
            return 1
        return 2


def _parse_detector_indices(cell: object) -> dict[str, float]:
    if not isinstance(cell, str) or not cell.strip():
        return {}
    try:
        data = json.loads(cell)
    except json.JSONDecodeError:
        return {}
    return {
        str(name): float(value)
        for name, value in data.items()
        if value is not None
    }


class AutoLabelIndex:
    """按 file_name 索引的自动标签与优先级信息。"""

    def __init__(self) -> None:
        self._files: dict[str, AutoFileInfo] = {}

    def _entry(self, file_name: str) -> AutoFileInfo:
        return self._files.setdefault(file_name, AutoFileInfo())

    def load_phase_labels(self, csv_path: str | Path) -> int:
        table = pd.read_csv(csv_path)
        count = 0
        for _, row in table.iterrows():
            file_name = str(row.get("file_name", ""))
            phase = str(row.get("phase", ""))
            if not file_name or phase not in ("A", "B", "C"):
                continue
            raw_index = row.get("raw_wavefront_index")
            spread = row.get("spread_us")
            confidence = row.get("confidence")
            self._entry(file_name).phases[phase] = AutoPhaseLabel(
                raw_wavefront_index=float(raw_index) if pd.notna(raw_index) else None,
                confidence=float(confidence) if pd.notna(confidence) else None,
                label_status=str(row.get("label_status", "")),
                spread_us=float(spread) if pd.notna(spread) and spread != float("inf") else None,
                detector_indices=_parse_detector_indices(row.get("detector_indices")),
            )
            count += 1
        return count

    def load_review_queue(self, csv_path: str | Path) -> int:
        table = pd.read_csv(csv_path)
        names = table.get("file_name")
        if names is None:
            return 0
        count = 0
        for name in names.dropna().astype(str).unique():
            self._entry(name).in_review_queue = True
            count += 1
        return count

    def load_worst30(self, csv_path: str | Path) -> int:
        table = pd.read_csv(csv_path)
        names = table.get("file_name")
        if names is None:
            return 0
        count = 0
        for name in names.dropna().astype(str).unique():
            self._entry(name).in_worst30 = True
            count += 1
        return count

    def info_for(self, file_name: str) -> AutoFileInfo | None:
        return self._files.get(file_name)

    def priority_for(self, file_name: str) -> int:
        info = self._files.get(file_name)
        return info.priority if info is not None else 2
