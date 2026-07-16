"""入口：默认联立 ningxia_core 云库并加载拓扑核心集。

用法:
  python -m wavefront_annotator
  python -m wavefront_annotator --no-core
  python -m wavefront_annotator [录波目录] [--auto-labels CSV ...]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

# PyInstaller 直接冻结本文件时 __package__ 为空，需回退到绝对导入
if __package__:
    from .main_window import MainWindow
    from .resources import load_app_icon
else:
    from wavefront_annotator.main_window import MainWindow
    from wavefront_annotator.resources import load_app_icon


def main() -> int:
    parser = argparse.ArgumentParser(description="行波波头 Gold 标注工具（联立 CloudBase 核心集）")
    parser.add_argument("directory", nargs="?", help="包含 .all/.vall 的录波目录（可选）")
    parser.add_argument(
        "--auto-labels",
        action="append",
        default=[],
        help="phase_labels.csv / review_queue.csv / stage0_worst30.csv，可多次指定",
    )
    parser.add_argument(
        "--no-core",
        action="store_true",
        help="不自动加载拓扑核心集",
    )
    args = parser.parse_args()

    app = QApplication(sys.argv)
    app.setApplicationName("WavefrontGoldAnnotator")
    app.setOrganizationName("WavefrontCNN")
    app.setWindowIcon(load_app_icon())
    window = MainWindow()

    for csv_path in args.auto_labels:
        path = Path(csv_path)
        if not path.exists():
            print(f"[警告] 自动标签文件不存在，已跳过: {path}", file=sys.stderr)
            continue
        if "review_queue" in path.name:
            window.auto_index.load_review_queue(path)
        elif "worst" in path.name:
            window.auto_index.load_worst30(path)
        else:
            window.auto_index.load_phase_labels(path)

    if args.directory:
        directory = Path(args.directory)
        if directory.is_dir():
            window.load_directory(directory)
        else:
            print(f"[警告] 目录不存在: {directory}", file=sys.stderr)
    elif not args.no_core:
        window.load_core_dataset()

    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
