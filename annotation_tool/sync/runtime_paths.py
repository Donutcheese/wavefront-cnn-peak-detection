"""运行时路径：开发态与 PyInstaller 冻结态统一解析。"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def app_dir() -> Path:
    """可写目录：exe 旁或 annotation_tool 根。"""
    if is_frozen():
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[1]


def kit_root() -> Path:
    """标注分发包根目录（与 exe 同级：含 hisdata/ 与核心索引）。"""
    return app_dir()


def bundle_dir() -> Path:
    """只读资源目录：_MEIPASS 或 annotation_tool 根。"""
    if is_frozen() and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parents[1]


def repo_data_dir() -> Path:
    """仓库 data/derived（仅开发态可靠）。"""
    return Path(__file__).resolve().parents[2] / "data" / "derived"


def resolve_config_candidates() -> list[Path]:
    """cloudbase.local.json 查找顺序：exe 旁 > 捆绑包 > 开发路径。"""
    candidates = [
        app_dir() / "cloudbase.local.json",
        bundle_dir() / "sync" / "cloudbase.local.json",
        bundle_dir() / "cloudbase.local.json",
    ]
    if not is_frozen():
        sync_dir = Path(__file__).resolve().parent
        candidates.extend(
            [
                sync_dir / "cloudbase.local.json",
                sync_dir.parents[1] / "scripts" / "cloudbase.local.json",
            ]
        )
    return candidates


def resolve_core_dir() -> Path:
    """核心集目录：优先 exe/Kit 旁，其次捆绑包，再次仓库 derived。"""
    beside = app_dir() / "wavefront_dataset_ningxia_core"
    if (beside / "core_file_index.csv").is_file():
        return beside
    bundled = bundle_dir() / "wavefront_dataset_ningxia_core"
    if (bundled / "core_file_index.csv").is_file():
        return bundled
    derived = repo_data_dir() / "wavefront_dataset_ningxia_core"
    if (derived / "core_file_index.csv").is_file():
        return derived
    return beside


def resolve_writable_label_dir() -> Path:
    """金标 CSV 可写目录（永远在 exe/工程旁，不写进 _MEIPASS）。"""
    path = app_dir() / "annotator_data"
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_all_source_path(source_path: str | Path) -> Path | None:
    """解析索引中的 source_path：绝对路径优先，否则相对 Kit 根。"""
    raw = str(source_path).strip().replace("\\", "/")
    if not raw:
        return None
    candidates: list[Path] = []
    as_path = Path(raw)
    if as_path.is_absolute():
        candidates.append(as_path)
    # Windows 盘符路径在非本机上无效，仍尝试相对 Kit 根拼接
    roots = [kit_root(), resolve_core_dir().parent, app_dir()]
    for root in roots:
        candidates.append((root / raw).resolve())
        # 兼容索引只写了 hisdata 之后的相对段
        if not raw.startswith("hisdata/"):
            candidates.append((root / "hisdata" / raw).resolve())
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.is_file():
            return candidate
    return None


def resolve_operator_name(default: str = "wavefront_operator") -> str:
    """标注员：环境变量 > operator.txt > 默认值。"""
    env = (os.environ.get("WAVEFRONT_ANNOTATOR") or "").strip()
    if env:
        return env
    for path in (
        app_dir() / "operator.txt",
        resolve_writable_label_dir() / "operator.txt",
    ):
        if path.is_file():
            text = path.read_text(encoding="utf-8").strip().splitlines()
            for line in text:
                name = line.strip()
                if name and not name.startswith("#"):
                    return name
    return default
