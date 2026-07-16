#!/usr/bin/env python3
"""按「更新后 SVG ∩ Excel 精确匹配」筛选宁夏核心拓扑数据集。"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEMO_ROOT = Path(r"C:\Users\Administrator\Documents\FaultLocation_demo")
DEFAULT_SVG = DEMO_ROOT / "docs" / "现场终端线路拓扑图_更新后.svg"
DEFAULT_XLSX = DEMO_ROOT / "docs" / "现场终端线路拓扑参数.xlsx"
DEFAULT_ALL_ROOT = DEMO_ROOT / "data" / "宁夏数据" / "hisdata"
DEFAULT_SRC = REPO_ROOT / "data" / "derived" / "wavefront_dataset_ningxia_a"
DEFAULT_OUT = REPO_ROOT / "data" / "derived" / "wavefront_dataset_ningxia_core"


def parse_svg_line_names(svg_path: Path) -> set[str]:
    text = svg_path.read_text(encoding="utf-8")
    names: set[str] = set()
    for label in re.findall(r'aria-label="([^"]+)"', text):
        match = re.match(r"^(.+线)", label)
        if match:
            names.add(match.group(1))
    return names


def main() -> int:
    parser = argparse.ArgumentParser(description="构建宁夏拓扑核心数据集")
    parser.add_argument("--svg", type=Path, default=DEFAULT_SVG)
    parser.add_argument("--xlsx", type=Path, default=DEFAULT_XLSX)
    parser.add_argument("--all-root", type=Path, default=DEFAULT_ALL_ROOT)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SRC)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--require-both-ends",
        action="store_true",
        help="仅保留同时有 M/N 端的事件（更严）",
    )
    args = parser.parse_args()

    sys.path.insert(0, str(DEMO_ROOT / "ningxia_python_project"))
    from net_faultlocation.dataset.topology import TopologyResolver, parse_all_record

    svg_lines = parse_svg_line_names(args.svg)
    resolver = TopologyResolver.from_workbook(args.xlsx)
    excel_lines = set(resolver._rows.keys())
    authorized = svg_lines & excel_lines
    if not authorized:
        raise SystemExit("SVG 与 Excel 无线路交集")

    records = []
    for path in sorted(args.all_root.rglob("*.all")):
        parsed = parse_all_record(path)
        if parsed is None:
            continue
        match = resolver.resolve(parsed.line_name)
        if match.match_status != "exact" or match.canonical_line_name not in authorized:
            continue
        records.append((path, parsed, match))

    if args.require_both_ends:
        from collections import defaultdict

        groups: dict[str, dict[str, list]] = defaultdict(lambda: {"M": [], "N": []})
        for path, parsed, match in records:
            groups[f"{parsed.timestamp}|{parsed.line_name}"][parsed.terminal].append(
                (path, parsed, match)
            )
        records = []
        for ends in groups.values():
            if ends["M"] and ends["N"]:
                records.extend(ends["M"])
                records.extend(ends["N"])

    exact_names = {path.name for path, _, _ in records}
    src_manifest = pd.read_csv(args.source_dir / "manifest.csv")
    src_labels = pd.read_csv(args.source_dir / "phase_labels.csv")
    keep_manifest = src_manifest[src_manifest["file_name"].isin(exact_names)].copy()
    if keep_manifest.empty:
        raise SystemExit("现有 h5/manifest 中无匹配文件，请先跑 ingest")

    old_to_new = {
        int(old): new_idx
        for new_idx, old in enumerate(keep_manifest["sample_index"].astype(int).tolist())
    }
    keep_ids = set(keep_manifest["sample_id"].astype(str))
    keep_labels = src_labels[src_labels["sample_id"].astype(str).isin(keep_ids)].copy()
    keep_labels["sample_index"] = keep_labels["sample_index"].astype(int).map(old_to_new)
    keep_manifest = keep_manifest.copy()
    keep_manifest["sample_index"] = keep_manifest["sample_index"].astype(int).map(old_to_new)
    keep_manifest = keep_manifest.sort_values("sample_index").reset_index(drop=True)
    keep_labels = keep_labels.sort_values(["sample_index", "phase"]).reset_index(drop=True)

    # 拓扑字段写入
    topo_by_file = {
        path.name: {
            "line_name": parsed.line_name,
            "terminal": parsed.terminal,
            "canonical_line_name": match.canonical_line_name,
            "line_length_km": match.line_length_km,
            "m_rtu": match.m_rtu,
            "n_rtu": match.n_rtu,
            "m_station": match.m_station,
            "n_station": match.n_station,
            "topology_match": match.match_status,
            "source_path": str(path.resolve()),
        }
        for path, parsed, match in records
    }
    for col in (
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
        keep_manifest[col] = keep_manifest["file_name"].map(
            lambda name, c=col: topo_by_file.get(name, {}).get(c, "")
        )

    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    src_h5 = args.source_dir / "waveforms.h5"
    dst_h5 = out / "waveforms.h5"
    old_indices = [int(v) for v in sorted(old_to_new.keys(), key=lambda x: old_to_new[x])]
    with h5py.File(src_h5, "r") as src, h5py.File(dst_h5, "w") as dst:
        signals = src["signals"]
        n = len(old_indices)
        window = int(src.attrs["window_samples"])
        dset = dst.create_dataset(
            "signals",
            shape=(n, 3, window),
            dtype="f4",
            chunks=(1, 3, window),
            compression="lzf",
        )
        for new_i, old_i in enumerate(old_indices):
            dset[new_i] = signals[old_i]
        for key, value in src.attrs.items():
            dst.attrs[key] = value
        dst.attrs["dataset"] = "ningxia_core"
        dst.attrs["topology_filter"] = "svg_exact_excel"

    keep_manifest["cloud_object_key"] = "wavefront/ningxia_core/signals/waveforms.h5"
    keep_labels["cloud_object_key"] = "wavefront/ningxia_core/signals/waveforms.h5"
    keep_labels["dataset"] = "ningxia_core"
    keep_manifest["dataset"] = "ningxia_core"

    keep_manifest.to_csv(out / "manifest.csv", index=False, encoding="utf-8")
    keep_labels.to_csv(out / "phase_labels.csv", index=False, encoding="utf-8")

    # 标注工具文件索引（绝对路径，打开即用）
    index_rows = []
    for _, row in keep_manifest.iterrows():
        meta = topo_by_file.get(str(row["file_name"]), {})
        index_rows.append(
            {
                "sample_index": int(row["sample_index"]),
                "sample_id": row["sample_id"],
                "file_name": row["file_name"],
                "source_path": meta.get("source_path", ""),
                "line_name": meta.get("line_name", ""),
                "terminal": meta.get("terminal", ""),
                "line_length_km": meta.get("line_length_km", ""),
                "split_event": row.get("split_event", ""),
            }
        )
    pd.DataFrame(index_rows).to_csv(out / "core_file_index.csv", index=False, encoding="utf-8")

    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "svg": str(args.svg),
        "xlsx": str(args.xlsx),
        "authorized_lines": sorted(authorized),
        "core_files": len(exact_names),
        "core_events": int(keep_manifest.shape[0]),
        "core_phase_rows": int(keep_labels.shape[0]),
        "require_both_ends": bool(args.require_both_ends),
        "h5_shape": [int(keep_manifest.shape[0]), 3, int(window)],
        "h5_size_mb": round(dst_h5.stat().st_size / (1024 * 1024), 2),
        "output_dir": str(out),
    }
    (out / "core_build_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
