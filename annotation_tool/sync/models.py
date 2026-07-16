"""本地 A 格式 ↔ 云文档字段映射。"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _to_float_or_none(value: Any) -> float | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    if text == "" or text.lower() == "nan":
        return None
    return float(value)


def build_sample_doc(
    row: pd.Series,
    *,
    dataset: str,
    storage_key: str,
    window_samples: int = 8192,
    target_fs_hz: float = 1_250_000.0,
) -> dict[str, Any]:
    sample_id = str(row["sample_id"])
    return {
        "_id": sample_id,
        "dataset": dataset,
        "file_name": str(row.get("file_name", "")),
        "sample_index": int(row["sample_index"]),
        "window_samples": int(window_samples),
        "target_fs_hz": float(target_fs_hz),
        "source_fs_hz": float(row.get("sampling_rate_hz", row.get("sampling_rate_hz_src", 0.0)) or 0.0),
        "storage_key": storage_key,
        "split_event": str(row.get("split_event", "")),
        "created_at": _now(),
        "updated_at": _now(),
    }


def build_label_doc(row: pd.Series, *, dataset: str) -> dict[str, Any]:
    sample_id = str(row["sample_id"])
    phase = str(row["phase"])
    raw = _to_float_or_none(row.get("raw_wavefront_index"))

    def _clean_str(value: Any) -> str:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return ""
        text = str(value).strip()
        return "" if text.lower() == "nan" else text

    return {
        "_id": f"{sample_id}:{phase}",
        "dataset": dataset,
        "sample_id": sample_id,
        "sample_index": int(row["sample_index"]),
        "phase": phase,
        "file_name": _clean_str(row.get("file_name", "")),
        "window_wavefront_index": int(row.get("window_wavefront_index", -1)),
        "raw_wavefront_index": raw,
        "region_start_index": _to_float_or_none(row.get("region_start_index")),
        "region_end_index": _to_float_or_none(row.get("region_end_index")),
        "label_status": _clean_str(row.get("label_status", "unlabeled")) or "unlabeled",
        "confidence": float(row.get("confidence", 0.0) or 0.0),
        "split_event": _clean_str(row.get("split_event", "")),
        "sampling_rate_hz_src": _to_float_or_none(row.get("sampling_rate_hz_src")),
        "annotator": _clean_str(row.get("annotator", "")) or "wavefront_operator",
        "note": _clean_str(row.get("note", "")),
        "auto_window_wavefront_index": _to_float_or_none(row.get("auto_window_wavefront_index")),
        "auto_label_status": _clean_str(row.get("auto_label_status", "")),
        "rev": int(row.get("rev", 1) or 1) if not (isinstance(row.get("rev"), float) and pd.isna(row.get("rev"))) else 1,
        "updated_at": _clean_str(row.get("updated_at")) or _now(),
    }


def labels_to_phase_dataframe(docs: list[dict[str, Any]], *, map_gold_to_hard: bool) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for doc in docs:
        status = str(doc.get("label_status", "unlabeled"))
        window_index = int(doc.get("window_wavefront_index", -1))
        confidence = float(doc.get("confidence", 0.0) or 0.0)
        if map_gold_to_hard and status == "gold":
            status = "hard"
            if window_index < 0:
                raise ValueError(f"gold 标签缺少窗内坐标: {doc.get('_id')}")
            confidence = max(confidence, 1.0)
        rows.append(
            {
                "sample_index": int(doc["sample_index"]),
                "sample_id": str(doc["sample_id"]),
                "phase": str(doc["phase"]),
                "window_wavefront_index": window_index,
                "confidence": confidence,
                "label_status": status,
                "split_event": str(doc.get("split_event", "")),
                "file_name": str(doc.get("file_name", "")),
                "raw_wavefront_index": doc.get("raw_wavefront_index", ""),
                "sampling_rate_hz_src": doc.get("sampling_rate_hz_src", ""),
                "cloud_object_key": str(doc.get("storage_key", "")),
                "updated_at": str(doc.get("updated_at", "")),
                "auto_window_wavefront_index": doc.get("auto_window_wavefront_index", ""),
                "auto_label_status": doc.get("auto_label_status", ""),
                "auto_confidence": "",
                "auto_raw_wavefront_index": "",
                "annotator": doc.get("annotator", ""),
                "note": doc.get("note", ""),
                "rev": int(doc.get("rev", 1) or 1),
            }
        )
    return pd.DataFrame(rows)
