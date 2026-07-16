"""标注工具云标签同步（写死绑定 ningxia_core 环境）。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import SyncConfig, load_sync_config
from .factory import create_backend
from .runtime_paths import resolve_core_dir, resolve_writable_label_dir

# 本仓库唯一联立目标：拓扑核心集
HARDCODED_DATASET = "ningxia_core"
HARDCODED_ANNOTATOR = "wavefront_operator"
HARDCODED_ENV_ID = "wavefrontdataset-d0e13om229bd53d"


def get_default_core_dir() -> Path:
    return resolve_core_dir()


# 兼容旧引用名
DEFAULT_CORE_DIR = resolve_core_dir()


@dataclass
class CloudPhaseLabel:
    sample_id: str
    sample_index: int
    file_name: str
    phase: str
    label_status: str
    raw_wavefront_index: float | None
    window_wavefront_index: int
    region_start_index: float | None
    region_end_index: float | None
    annotator: str
    note: str
    rev: int
    updated_at: str


class CloudLabelSync:
    """pull / upsert 云端 wf_phase_labels，与本地 gold_labels 双向对齐。"""

    def __init__(self, config: SyncConfig | None = None) -> None:
        self.config = config or load_sync_config()
        from dataclasses import replace

        self.config = replace(
            self.config,
            dataset=HARDCODED_DATASET,
            env_id=self.config.env_id or HARDCODED_ENV_ID,
            annotator=HARDCODED_ANNOTATOR,
        )
        self.backend = create_backend(self.config)
        self.by_file_phase: dict[tuple[str, str], CloudPhaseLabel] = {}
        self.sample_id_by_file: dict[str, str] = {}
        self.sample_index_by_file: dict[str, int] = {}

    def load_file_index(self, core_dir: Path | None = None) -> Path:
        root = Path(core_dir or resolve_core_dir())
        index_path = root / "core_file_index.csv"
        if not index_path.is_file():
            raise FileNotFoundError(f"缺少核心文件索引: {index_path}")
        import pandas as pd

        table = pd.read_csv(index_path)
        for _, row in table.iterrows():
            name = str(row["file_name"])
            self.sample_id_by_file[name] = str(row["sample_id"])
            self.sample_index_by_file[name] = int(row["sample_index"])
        return index_path

    def pull_labels(self) -> int:
        docs = self.backend.list_documents(
            self.config.labels_collection,
            query={"dataset": HARDCODED_DATASET},
        )
        self.by_file_phase.clear()
        for doc in docs:
            label = CloudPhaseLabel(
                sample_id=str(_unwrap_ejson(doc.get("sample_id", ""))),
                sample_index=_as_int(doc.get("sample_index"), -1),
                file_name=str(_unwrap_ejson(doc.get("file_name", ""))),
                phase=str(_unwrap_ejson(doc.get("phase", ""))),
                label_status=str(_unwrap_ejson(doc.get("label_status", "unlabeled"))),
                raw_wavefront_index=_as_float(doc.get("raw_wavefront_index")),
                window_wavefront_index=_as_int(doc.get("window_wavefront_index"), -1),
                region_start_index=_as_float(doc.get("region_start_index")),
                region_end_index=_as_float(doc.get("region_end_index")),
                annotator=str(_unwrap_ejson(doc.get("annotator", "")) or ""),
                note=str(_unwrap_ejson(doc.get("note", "")) or ""),
                rev=_as_int(doc.get("rev"), 1),
                updated_at=str(_unwrap_ejson(doc.get("updated_at", ""))),
            )
            if label.file_name and label.phase:
                self.by_file_phase[(label.file_name, label.phase)] = label
            if label.file_name and label.sample_id:
                self.sample_id_by_file[label.file_name] = label.sample_id
                self.sample_index_by_file[label.file_name] = label.sample_index
        return len(self.by_file_phase)

    def get(self, file_name: str, phase: str) -> CloudPhaseLabel | None:
        return self.by_file_phase.get((file_name, phase))

    def upsert_annotation(
        self,
        *,
        file_name: str,
        phase: str,
        status: str,
        raw_wavefront_index: float,
        region_start: float | None,
        region_end: float | None,
        sampling_rate_hz: float | None,
        note: str = "",
        annotator: str | None = None,
    ) -> CloudPhaseLabel:
        sample_id = self.sample_id_by_file.get(file_name)
        sample_index = self.sample_index_by_file.get(file_name, -1)
        if not sample_id:
            raise KeyError(f"文件不在核心集索引中: {file_name}")
        prev = self.by_file_phase.get((file_name, phase))
        rev = (prev.rev + 1) if prev else 1
        now = datetime.now().isoformat(timespec="seconds")
        who = (annotator or self.config.annotator or HARDCODED_ANNOTATOR).strip()
        doc: dict[str, Any] = {
            "_id": f"{sample_id}:{phase}",
            "dataset": HARDCODED_DATASET,
            "sample_id": sample_id,
            "sample_index": sample_index,
            "phase": phase,
            "file_name": file_name,
            "window_wavefront_index": prev.window_wavefront_index if prev else -1,
            "raw_wavefront_index": float(raw_wavefront_index),
            "region_start_index": region_start,
            "region_end_index": region_end,
            "label_status": status,
            "confidence": 1.0 if status == "gold" else 0.5,
            "split_event": "",
            "sampling_rate_hz_src": sampling_rate_hz,
            "annotator": who,
            "note": note,
            "rev": rev,
            "updated_at": now,
        }
        self.backend.upsert_document(self.config.labels_collection, doc["_id"], doc)
        label = CloudPhaseLabel(
            sample_id=sample_id,
            sample_index=sample_index,
            file_name=file_name,
            phase=phase,
            label_status=status,
            raw_wavefront_index=float(raw_wavefront_index),
            window_wavefront_index=int(doc["window_wavefront_index"]),
            region_start_index=region_start,
            region_end_index=region_end,
            annotator=who,
            note=note,
            rev=rev,
            updated_at=now,
        )
        self.by_file_phase[(file_name, phase)] = label
        return label


def _unwrap_ejson(value: Any) -> Any:
    """展开 CloudBase/EJSON 包装的标量。"""
    if isinstance(value, dict):
        for key in ("$numberInt", "$numberLong", "$numberDouble", "$numberDecimal"):
            if key in value:
                return value[key]
        if "$oid" in value:
            return value["$oid"]
        if "$date" in value:
            return value["$date"]
    return value


def _as_int(value: Any, default: int = -1) -> int:
    value = _unwrap_ejson(value)
    if value is None or value == "":
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _as_float(value: Any) -> float | None:
    value = _unwrap_ejson(value)
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:  # NaN
        return None
    return number
