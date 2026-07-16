#!/usr/bin/env python3
"""构建标注组便携 Kit（方案 A）：exe + 配置 + 相对路径索引 + 核心 .all。

产物目录结构:
  WavefrontAnnotatorKit/
    WavefrontGoldAnnotator.exe
    cloudbase.local.json
    operator.txt
    使用说明.txt
    wavefront_dataset_ningxia_core/
      core_file_index.csv   # source_path 为相对路径 hisdata/...
      phase_labels.csv
    hisdata/                # 仅核心集 .all
    annotator_data/         # 运行时金标缓存
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORE = REPO_ROOT / "data" / "derived" / "wavefront_dataset_ningxia_core"
DEFAULT_HISDATA = Path(
    r"C:\Users\Administrator\Documents\FaultLocation_demo\data\宁夏数据\hisdata"
)
DEFAULT_EXE = REPO_ROOT / "annotation_tool" / "dist" / "WavefrontGoldAnnotator.exe"
DEFAULT_CFG = REPO_ROOT / "annotation_tool" / "sync" / "cloudbase.local.json"
DEFAULT_OUT = REPO_ROOT / "dist" / "WavefrontAnnotatorKit"


def _rel_under_hisdata(abs_path: Path, hisdata_root: Path) -> str:
    """将绝对路径转为 Kit 内相对路径 hisdata/...（正斜杠）。"""
    resolved = abs_path.resolve()
    root = hisdata_root.resolve()
    try:
        rel = resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"文件不在 hisdata 根下: {resolved}") from exc
    return "hisdata/" + rel.as_posix()


def main() -> int:
    parser = argparse.ArgumentParser(description="构建标注组便携 Kit")
    parser.add_argument("--core-dir", type=Path, default=DEFAULT_CORE)
    parser.add_argument("--hisdata-root", type=Path, default=DEFAULT_HISDATA)
    parser.add_argument("--exe", type=Path, default=DEFAULT_EXE)
    parser.add_argument("--config", type=Path, default=DEFAULT_CFG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--skip-copy-all",
        action="store_true",
        help="只重写索引/拷贝元数据，不复制 .all（调试用）",
    )
    args = parser.parse_args()

    index_src = args.core_dir / "core_file_index.csv"
    phase_src = args.core_dir / "phase_labels.csv"
    if not index_src.is_file():
        raise SystemExit(f"缺少索引: {index_src}")
    if not args.exe.is_file():
        raise SystemExit(f"缺少 exe，请先打包: {args.exe}")
    if not args.config.is_file():
        raise SystemExit(f"缺少云配置: {args.config}")
    if not args.hisdata_root.is_dir():
        raise SystemExit(f"hisdata 根不存在: {args.hisdata_root}")

    out = args.output_dir
    core_out = out / "wavefront_dataset_ningxia_core"
    his_out = out / "hisdata"
    data_out = out / "annotator_data"
    out.mkdir(parents=True, exist_ok=True)
    core_out.mkdir(parents=True, exist_ok=True)
    data_out.mkdir(parents=True, exist_ok=True)

    table = pd.read_csv(index_src)
    if "source_path" not in table.columns:
        raise SystemExit("core_file_index.csv 缺少 source_path 列")

    rel_paths: list[str] = []
    missing: list[str] = []
    copied = 0
    total = len(table)
    for i, row in table.iterrows():
        src = Path(str(row["source_path"]))
        if not src.is_file():
            missing.append(str(src))
            rel_paths.append("")
            continue
        rel = _rel_under_hisdata(src, args.hisdata_root)
        rel_paths.append(rel)
        if not args.skip_copy_all:
            dst = out / Path(rel)
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.is_file() or dst.stat().st_size != src.stat().st_size:
                shutil.copy2(src, dst)
                copied += 1
        if (i + 1) % 200 == 0 or (i + 1) == total:
            print(f"[进度] 索引 {i + 1}/{total}，已复制 {copied} 个 .all", flush=True)

    out_table = table.copy()
    out_table["source_path_orig"] = table["source_path"]
    out_table["source_path"] = rel_paths
    if missing:
        print(f"[警告] 源端缺失 {len(missing)} 个文件，已写入空路径", file=sys.stderr)
        out_table = out_table[out_table["source_path"].astype(str).str.len() > 0]

    out_table.to_csv(core_out / "core_file_index.csv", index=False, encoding="utf-8-sig")
    if phase_src.is_file():
        shutil.copy2(phase_src, core_out / "phase_labels.csv")
    for name in ("manifest.csv", "core_build_report.json"):
        src = args.core_dir / name
        if src.is_file():
            shutil.copy2(src, core_out / name)

    shutil.copy2(args.exe, out / "WavefrontGoldAnnotator.exe")
    shutil.copy2(args.config, out / "cloudbase.local.json")

    (out / "operator.txt").write_text(
        "# 请改成你的标注员名（一行），保存后重启 exe\n"
        "wavefront_operator\n",
        encoding="utf-8",
    )
    (out / "使用说明.txt").write_text(
        "行波波头 Gold 标注 — 便携 Kit\n"
        "================================\n"
        "1. 保持本文件夹结构不变（exe、hisdata、wavefront_dataset_ningxia_core 同级）。\n"
        "2. 编辑 operator.txt，改成你的名字（多人并行时便于云端追溯）。\n"
        "3. 双击 WavefrontGoldAnnotator.exe；启动后自动加载核心集并拉取云标签。\n"
        "4. 标注保存后：本地写入 annotator_data/gold_labels.csv，同时推送到云数据库。\n"
        "5. 若云同步失败：检查网络，或向管理员索取新的 cloudbase.local.json 覆盖本目录同名文件。\n"
        "6. 不要单独移动 exe；不要删除 hisdata。\n",
        encoding="utf-8",
    )

    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "output_dir": str(out),
        "core_files": int(len(out_table)),
        "copied_all": int(copied),
        "missing_source": len(missing),
        "skip_copy_all": bool(args.skip_copy_all),
        "exe": str(args.exe),
        "hisdata_root": str(args.hisdata_root),
    }
    (out / "kit_build_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nKit 已生成: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
