#!/usr/bin/env python3
"""P1：宁夏 .all → 本地格式 A（waveforms.h5 + phase_labels.csv）。

裁窗/重采样复用 FaultLocation_demo 的 v1 worker（label_phase + extract_training_window），
但跳过双端测距配对（该步骤对云标注入库非必需且极慢）。
默认将标签写为 unlabeled，并保留 auto_* 伪标签列。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import h5py
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEMO_ROOT = Path(r"C:\Users\Administrator\Documents\FaultLocation_demo")
DEFAULT_DATA_ROOT = DEFAULT_DEMO_ROOT / "data" / "宁夏数据"
DEFAULT_TOPOLOGY = DEFAULT_DEMO_ROOT / "docs" / "现场终端线路拓扑参数.xlsx"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "derived" / "wavefront_dataset_ningxia_a"

PHASES = ("A", "B", "C")
PHASE_COLUMNS = [
    "sample_index",
    "sample_id",
    "phase",
    "window_wavefront_index",
    "confidence",
    "label_status",
    "split_event",
    "file_name",
    "raw_wavefront_index",
    "sampling_rate_hz_src",
    "cloud_object_key",
    "updated_at",
    "auto_window_wavefront_index",
    "auto_label_status",
    "auto_confidence",
    "auto_raw_wavefront_index",
]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="宁夏录波入库为本地格式 A（h5+csv）")
    parser.add_argument("--demo-root", type=Path, default=DEFAULT_DEMO_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--topology-workbook", type=Path, default=DEFAULT_TOPOLOGY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--target-fs", type=float, default=1_250_000.0)
    parser.add_argument("--window-samples", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument(
        "--keep-auto-labels",
        action="store_true",
        help="保留 builder 的 hard/soft/review（默认改为 unlabeled）",
    )
    parser.add_argument(
        "--storage-key",
        type=str,
        default="wavefront/ningxia/signals/waveforms.h5",
    )
    return parser


def _ensure_demo_import(demo_root: Path) -> None:
    project = (demo_root / "ningxia_python_project").resolve()
    if not project.is_dir():
        raise SystemExit(f"未找到 demo 工程目录: {project}")
    text = str(project)
    if text not in sys.path:
        sys.path.insert(0, text)


def _build_windows(
    *,
    data_root: Path,
    topology_workbook: Path,
    output_dir: Path,
    target_fs: float,
    window_samples: int,
    workers: int,
    max_files: int | None,
    seed: int,
    force_unlabeled: bool,
    storage_key: str,
) -> dict:
    from net_faultlocation.dataset.builder import (
        PHASES as BUILDER_PHASES,
        BuildConfig,
        _ProcessFailure,
        _ProcessedSample,
        _WorkerConfig,
        _create_hdf5,
        _empty_topology,
        _iter_results,
        assign_split,
        nominal_sampling_rate_hz,
    )
    from net_faultlocation.dataset.topology import TopologyResolver, parse_all_record

    assert BUILDER_PHASES == PHASES
    config = BuildConfig(
        data_root=data_root,
        topology_workbook=topology_workbook,
        output_dir=output_dir,
        target_sampling_rate_hz=target_fs,
        window_samples=window_samples,
        workers=workers,
        max_files=max_files,
        seed=seed,
    )
    paths = sorted(data_root.rglob("*.all"), key=lambda path: str(path))
    if max_files is not None:
        paths = paths[: max(0, int(max_files))]
    resolver = TopologyResolver.from_workbook(topology_workbook)

    h5_path = output_dir / "waveforms.h5"
    if h5_path.exists():
        h5_path.unlink()
    handle, arrays = _create_hdf5(h5_path, len(paths), config)

    manifest_rows: list[dict] = []
    phase_rows: list[dict] = []
    error_rows: list[dict] = []
    auto_counts: dict[str, int] = {}
    decoded = 0
    now = datetime.now().isoformat(timespec="seconds")

    try:
        for sequence, result in enumerate(_iter_results(paths, config), start=1):
            if isinstance(result, _ProcessFailure):
                error_rows.append({"path": result.path, "error": result.error})
                continue
            assert isinstance(result, _ProcessedSample)
            path = Path(result.path)
            parsed = parse_all_record(path)
            topology = (
                resolver.resolve(parsed.line_name)
                if parsed is not None
                else _empty_topology(topology_workbook)
            )
            sample_index = decoded
            arrays["signals"][sample_index] = result.signals
            arrays["wavefront_index"][sample_index] = result.window_wavefront_indices
            arrays["peak_index"][sample_index] = result.window_peak_indices
            arrays["raw_wavefront_index"][sample_index] = result.raw_wavefront_indices
            arrays["raw_peak_index"][sample_index] = result.raw_peak_indices
            arrays["confidence"][sample_index] = result.confidences
            arrays["label_status"][sample_index] = result.status_codes
            arrays["normalization_center"][sample_index] = result.centers
            arrays["normalization_scale"][sample_index] = result.scales

            line_name = parsed.line_name if parsed is not None else ""
            event_key = parsed.event_key if parsed is not None else result.sample_id
            split_event = assign_split(event_key, seed)

            manifest_rows.append(
                {
                    "sample_index": sample_index,
                    "sample_id": result.sample_id,
                    "path": str(path),
                    "file_name": path.name,
                    "event_key": event_key,
                    "line_name": line_name,
                    "data_length": result.data_length,
                    "sampling_rate_hz": result.sampling_rate_hz,
                    "nominal_sampling_rate_hz": nominal_sampling_rate_hz(result.sampling_rate_hz),
                    "window_start_time_s": result.window_start_time_s,
                    "target_sampling_rate_hz": target_fs,
                    "topology_match": topology.match_status,
                    "split_event": split_event,
                    "cloud_object_key": storage_key,
                }
            )

            for phase_index, (phase_name, label) in enumerate(zip(PHASES, result.labels)):
                auto_status = label.label_status
                auto_window = int(result.window_wavefront_indices[phase_index])
                auto_conf = float(label.confidence)
                auto_raw = int(label.wavefront_index)
                auto_counts[auto_status] = auto_counts.get(auto_status, 0) + 1
                if force_unlabeled:
                    status = "unlabeled"
                    window_index = -1
                    confidence = 0.0
                    raw_value: int | str = ""
                else:
                    status = auto_status
                    window_index = auto_window
                    confidence = auto_conf
                    raw_value = auto_raw
                phase_rows.append(
                    {
                        "sample_index": sample_index,
                        "sample_id": result.sample_id,
                        "phase": phase_name,
                        "window_wavefront_index": window_index,
                        "confidence": confidence,
                        "label_status": status,
                        "split_event": split_event,
                        "file_name": path.name,
                        "raw_wavefront_index": raw_value,
                        "sampling_rate_hz_src": result.sampling_rate_hz,
                        "cloud_object_key": storage_key,
                        "updated_at": now,
                        "auto_window_wavefront_index": auto_window,
                        "auto_label_status": auto_status,
                        "auto_confidence": auto_conf,
                        "auto_raw_wavefront_index": auto_raw,
                    }
                )

            decoded += 1
            if sequence % 100 == 0 or sequence == len(paths):
                print(
                    f"dataset progress: {sequence}/{len(paths)}, decoded={decoded}, errors={len(error_rows)}",
                    flush=True,
                )
    finally:
        for dataset in arrays.values():
            dataset.resize((decoded,) + dataset.shape[1:])
        handle.close()

    h5_temp = h5_path.with_suffix(h5_path.suffix + ".tmp")
    if h5_temp.exists():
        # _create_hdf5 写入 *.h5.tmp，需原子替换
        if h5_path.exists():
            h5_path.unlink()
        h5_temp.replace(h5_path)

    pd.DataFrame(manifest_rows).to_csv(output_dir / "manifest.csv", index=False, encoding="utf-8")
    pd.DataFrame(phase_rows, columns=PHASE_COLUMNS).to_csv(
        output_dir / "phase_labels.csv", index=False, encoding="utf-8"
    )
    pd.DataFrame(error_rows).to_csv(output_dir / "errors.csv", index=False, encoding="utf-8")

    return {
        "total_files": len(paths),
        "decoded_files": decoded,
        "decode_errors": len(error_rows),
        "phase_labels": len(phase_rows),
        "auto_label_status_counts": auto_counts,
    }


def _self_check(output_dir: Path, expected_files: int | None) -> dict:
    h5_path = output_dir / "waveforms.h5"
    csv_path = output_dir / "phase_labels.csv"
    with h5py.File(h5_path, "r") as handle:
        shape = tuple(handle["signals"].shape)
        target_fs = float(handle.attrs["target_sampling_rate_hz"])
        window = int(handle.attrs["window_samples"])
    labels = pd.read_csv(csv_path)
    n_events = int(shape[0])
    n_rows = int(len(labels))
    status_counts = labels["label_status"].value_counts().to_dict()
    report = {
        "h5_shape": list(shape),
        "target_sampling_rate_hz": target_fs,
        "window_samples": window,
        "phase_rows": n_rows,
        "events": n_events,
        "label_status_counts": status_counts,
        "rows_per_event": float(n_rows / n_events) if n_events else 0.0,
        "ok_shape": shape[1:] == (3, window) and window == 8192,
        "ok_rows": n_rows == n_events * 3,
        "ok_expected_files": expected_files is None or n_events == expected_files,
        "h5_size_mb": round(h5_path.stat().st_size / (1024 * 1024), 2),
    }
    report["passed"] = bool(report["ok_shape"] and report["ok_rows"] and report["ok_expected_files"])
    return report


def main() -> int:
    args = _parser().parse_args()
    if not args.data_root.is_dir():
        raise SystemExit(f"数据目录不存在: {args.data_root}")
    if not args.topology_workbook.is_file():
        raise SystemExit(f"拓扑表不存在: {args.topology_workbook}")

    _ensure_demo_import(args.demo_root)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    all_files = sorted(args.data_root.rglob("*.all"))
    expected = min(len(all_files), args.max_files) if args.max_files else len(all_files)
    print(f"[ingest] 扫描到 .all: {len(all_files)}, 将处理: {expected}", flush=True)

    builder = _build_windows(
        data_root=args.data_root.resolve(),
        topology_workbook=args.topology_workbook.resolve(),
        output_dir=output_dir,
        target_fs=args.target_fs,
        window_samples=args.window_samples,
        workers=args.workers,
        max_files=args.max_files,
        seed=args.seed,
        force_unlabeled=not args.keep_auto_labels,
        storage_key=args.storage_key,
    )
    check = _self_check(output_dir, expected_files=builder["decoded_files"])
    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "output_dir": str(output_dir),
        "force_unlabeled": not args.keep_auto_labels,
        "builder": builder,
        "self_check": check,
    }
    (output_dir / "ingest_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0 if check["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
