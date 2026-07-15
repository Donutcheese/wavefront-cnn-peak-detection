"""gold_labels.csv 同步存储。

设计要点：
- 主键 (file_name, phase) upsert，不覆盖自动伪标签表 phase_labels.csv；
- 每次提交后立即原子写盘（临时文件 + os.replace），进程崩溃不丢已保存记录；
- gold 坐标使用原始录波采样点坐标 raw 语义，与数据集 v2 重建对接。
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path

import pandas as pd

GOLD_COLUMNS = [
    "file_name",
    "file_path",
    "phase",
    "gold_wavefront_index",
    "gold_time_us",
    "region_start_index",
    "region_end_index",
    "status",
    "auto_wavefront_index",
    "sampling_rate_hz",
    "note",
    "annotator",
    "updated_at",
]

# gold=人工确认；unsure=存疑；reject=波形无法标注
VALID_STATUSES = ("gold", "unsure", "reject")


@dataclass
class GoldLabel:
    file_name: str
    file_path: str
    phase: str
    gold_wavefront_index: float
    gold_time_us: float
    status: str
    region_start_index: float | None = None
    region_end_index: float | None = None
    auto_wavefront_index: float | None = None
    sampling_rate_hz: float | None = None
    note: str = ""
    annotator: str = ""
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))


class GoldLabelStore:
    """以 (file_name, phase) 为主键的 gold 标签表，落盘为单个 CSV。"""

    def __init__(self, csv_path: str | Path) -> None:
        self.csv_path = Path(csv_path)
        self._records: dict[tuple[str, str], dict] = {}
        if self.csv_path.exists():
            self._load()

    def _load(self) -> None:
        table = pd.read_csv(self.csv_path, dtype={"file_name": str, "phase": str})
        for column in GOLD_COLUMNS:
            if column not in table.columns:
                table[column] = None
        for _, row in table.iterrows():
            key = (str(row["file_name"]), str(row["phase"]))
            self._records[key] = {c: row[c] for c in GOLD_COLUMNS}

    def upsert(self, label: GoldLabel) -> None:
        if label.status not in VALID_STATUSES:
            raise ValueError(f"非法标注状态: {label.status}")
        self._records[(label.file_name, label.phase)] = asdict(label)
        self.flush()

    def remove(self, file_name: str, phase: str) -> None:
        if self._records.pop((file_name, phase), None) is not None:
            self.flush()

    def get(self, file_name: str, phase: str) -> dict | None:
        return self._records.get((file_name, phase))

    def phases_of(self, file_name: str) -> dict[str, dict]:
        return {
            phase: record
            for (name, phase), record in self._records.items()
            if name == file_name
        }

    def count(self) -> int:
        return len(self._records)

    def flush(self) -> None:
        """原子写盘：先写同目录临时文件，再 os.replace 覆盖。"""
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        table = pd.DataFrame(list(self._records.values()), columns=GOLD_COLUMNS)
        table = table.sort_values(["file_name", "phase"]).reset_index(drop=True)
        fd, tmp_path = tempfile.mkstemp(
            dir=str(self.csv_path.parent), prefix=".gold_", suffix=".csv"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8-sig", newline="") as handle:
                table.to_csv(handle, index=False)
            os.replace(tmp_path, self.csv_path)
        except BaseException:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
