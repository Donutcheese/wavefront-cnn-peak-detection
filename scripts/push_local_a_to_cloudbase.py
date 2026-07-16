#!/usr/bin/env python3
"""P2：本地格式 A → CloudBase（或本地镜像）。支持并发 upsert。"""

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
from sync.models import build_label_doc, build_sample_doc  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="推送本地 A 数据集到 CloudBase / 本地镜像")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--config", type=Path, default=None, help="cloudbase.local.json 路径")
    parser.add_argument("--skip-h5", action="store_true", help="跳过波形文件上传")
    parser.add_argument("--max-docs", type=int, default=None, help="仅推送前 N 条标签（试推）")
    parser.add_argument("--dataset-name", type=str, default=None, help="覆盖配置中的 dataset 名")
    parser.add_argument("--workers", type=int, default=16, help="并发 upsert 线程数")
    parser.add_argument("--file-id", type=str, default="", help="已上传波形的 fileID（可选）")
    return parser


def main() -> int:
    args = _parser().parse_args()
    dataset_dir = args.dataset_dir.resolve()
    h5_path = dataset_dir / "waveforms.h5"
    labels_path = dataset_dir / "phase_labels.csv"
    manifest_path = dataset_dir / "manifest.csv"
    for path in (h5_path, labels_path, manifest_path):
        if not path.is_file():
            raise SystemExit(f"缺少文件: {path}")

    config = load_sync_config(args.config)
    if args.dataset_name:
        from dataclasses import replace

        config = replace(config, dataset=args.dataset_name)
    # 与核心集 storage_key 对齐
    if (dataset_dir / "core_file_index.csv").exists():
        from dataclasses import replace

        config = replace(
            config,
            storage_key="wavefront/ningxia_core/signals/waveforms.h5",
            dataset=args.dataset_name or "ningxia_core",
        )

    backend = create_backend(config)
    backend.ensure_collections()

    with h5py.File(h5_path, "r") as handle:
        window_samples = int(handle.attrs["window_samples"])
        target_fs = float(handle.attrs["target_sampling_rate_hz"])
        n_events = int(handle["signals"].shape[0])

    labels = pd.read_csv(labels_path)
    manifest = pd.read_csv(manifest_path)
    if args.max_docs is not None:
        labels = labels.head(args.max_docs).copy()
        keep_ids = set(labels["sample_id"].astype(str))
        manifest = manifest[manifest["sample_id"].astype(str).isin(keep_ids)].copy()

    file_id = args.file_id or None
    if not args.skip_h5:
        print(f"[push] 上传波形 {h5_path} -> {config.storage_key}", flush=True)
        file_id = backend.upload_file(str(h5_path), config.storage_key)
        print(f"[push] file_id={file_id}", flush=True)

    sample_docs = []
    for _, row in manifest.iterrows():
        doc = build_sample_doc(
            row,
            dataset=config.dataset,
            storage_key=config.storage_key,
            window_samples=window_samples,
            target_fs_hz=target_fs,
        )
        if file_id:
            doc["file_id"] = file_id
        # 拓扑字段（核心集有则带上）
        for key in (
            "line_name",
            "terminal",
            "canonical_line_name",
            "line_length_km",
            "m_rtu",
            "n_rtu",
            "m_station",
            "n_station",
            "topology_match",
            "source_path",
        ):
            if key in row and pd.notna(row[key]):
                doc[key] = row[key]
        sample_docs.append(doc)

    label_docs = [build_label_doc(row, dataset=config.dataset) for _, row in labels.iterrows()]

    print(
        f"[push] upsert samples={len(sample_docs)} labels={len(label_docs)} "
        f"backend={config.backend} workers={args.workers}",
        flush=True,
    )
    upsert = getattr(backend, "upsert_documents_parallel", None)
    if upsert is None:
        for doc in sample_docs:
            backend.upsert_document(config.samples_collection, doc["_id"], doc)
        for index, doc in enumerate(label_docs, start=1):
            backend.upsert_document(config.labels_collection, doc["_id"], doc)
            if index % 500 == 0:
                print(f"[push] labels progress {index}/{len(label_docs)}", flush=True)
        pushed_samples = len(sample_docs)
        pushed_labels = len(label_docs)
    else:
        pushed_samples = upsert(config.samples_collection, sample_docs, workers=args.workers)
        print(f"[push] samples done: {pushed_samples}", flush=True)
        # 分块并发，便于进度
        chunk = 500
        pushed_labels = 0
        for start in range(0, len(label_docs), chunk):
            part = label_docs[start : start + chunk]
            pushed_labels += upsert(config.labels_collection, part, workers=args.workers)
            print(f"[push] labels progress {pushed_labels}/{len(label_docs)}", flush=True)

    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "backend": config.backend,
        "dataset": config.dataset,
        "storage_key": config.storage_key,
        "file_id": file_id,
        "local_events": n_events,
        "pushed_samples": pushed_samples,
        "pushed_labels": pushed_labels,
        "remote_sample_count": pushed_samples,
        "remote_label_count": pushed_labels,
        "mirror_root": config.mirror_root if config.backend == "local_mirror" else "",
    }
    out = dataset_dir / "push_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0 if pushed_samples == len(sample_docs) and pushed_labels == len(label_docs) else 2


if __name__ == "__main__":
    raise SystemExit(main())
