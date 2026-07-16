#!/usr/bin/env python3
"""P2：CloudBase（或本地镜像）→ 本地格式 A，供训练读取。"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import h5py
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = REPO_ROOT / "data" / "derived" / "wavefront_dataset_ningxia_a"
sys.path.insert(0, str(REPO_ROOT / "annotation_tool"))

from sync import create_backend, load_sync_config  # noqa: E402
from sync.models import labels_to_phase_dataframe  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="从 CloudBase / 本地镜像拉取到本地 A")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--skip-h5", action="store_true", help="不下载波形（本地已有 h5 时）")
    parser.add_argument(
        "--map-gold-to-hard",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="将云端 gold 映射为训练用 hard（默认开启）",
    )
    parser.add_argument(
        "--labels-only-status",
        type=str,
        default="",
        help="仅导出指定状态，逗号分隔；空=全部",
    )
    parser.add_argument("--dataset-name", type=str, default=None)
    return parser


def main() -> int:
    args = _parser().parse_args()
    dataset_dir = args.dataset_dir.resolve()
    dataset_dir.mkdir(parents=True, exist_ok=True)

    config = load_sync_config(args.config)
    if args.dataset_name:
        from dataclasses import replace

        config = replace(config, dataset=args.dataset_name)
    backend = create_backend(config)

    samples = backend.list_documents(config.samples_collection, query={"dataset": config.dataset})
    labels = backend.list_documents(config.labels_collection, query={"dataset": config.dataset})
    if not samples:
        raise SystemExit(f"远端无样本: dataset={config.dataset}")

    status_filter = {item.strip() for item in args.labels_only_status.split(",") if item.strip()}
    if status_filter:
        labels = [doc for doc in labels if str(doc.get("label_status")) in status_filter]

    h5_path = dataset_dir / "waveforms.h5"
    if not args.skip_h5:
        storage_key = str(samples[0].get("storage_key") or config.storage_key)
        file_id = samples[0].get("file_id")
        print(f"[pull] 下载波形 {storage_key} -> {h5_path}", flush=True)
        backend.download_file(storage_key, str(h5_path), file_id=file_id)
    elif not h5_path.is_file():
        raise SystemExit(f"--skip-h5 但本地不存在: {h5_path}")

    with h5py.File(h5_path, "r") as handle:
        shape = tuple(handle["signals"].shape)

    phase_df = labels_to_phase_dataframe(labels, map_gold_to_hard=args.map_gold_to_hard)
    # 保证列齐全并按 sample_index/phase 排序
    phase_df = phase_df.sort_values(["sample_index", "phase"]).reset_index(drop=True)
    labels_path = dataset_dir / "phase_labels.csv"
    phase_df.to_csv(labels_path, index=False, encoding="utf-8")

    manifest_rows = []
    for doc in sorted(samples, key=lambda item: int(item.get("sample_index", 0))):
        manifest_rows.append(
            {
                "sample_index": int(doc["sample_index"]),
                "sample_id": str(doc["_id"]),
                "file_name": str(doc.get("file_name", "")),
                "sampling_rate_hz": doc.get("source_fs_hz", ""),
                "split_event": doc.get("split_event", ""),
                "cloud_object_key": doc.get("storage_key", config.storage_key),
                "file_id": doc.get("file_id", ""),
            }
        )
    pd.DataFrame(manifest_rows).to_csv(dataset_dir / "manifest.csv", index=False, encoding="utf-8")

    status_counts = phase_df["label_status"].value_counts().to_dict() if len(phase_df) else {}
    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "backend": config.backend,
        "dataset": config.dataset,
        "h5_shape": list(shape),
        "pulled_samples": len(samples),
        "pulled_labels": len(phase_df),
        "label_status_counts": status_counts,
        "map_gold_to_hard": args.map_gold_to_hard,
        "output_dir": str(dataset_dir),
    }
    (dataset_dir / "pull_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0 if len(samples) > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
