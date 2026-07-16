"""入口：python -m wavefront_annotator [录波目录] [--auto-labels CSV ...]"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

from .main_window import MainWindow
from .resources import load_app_icon


def main() -> int:
    parser = argparse.ArgumentParser(description="行波波头 Gold 标注工具")
    parser.add_argument("directory", nargs="?", help="包含 .all/.vall 的录波目录")
    parser.add_argument(
        "--auto-labels",
        action="append",
        default=[],
        help="phase_labels.csv / review_queue.csv / stage0_worst30.csv，可多次指定",
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

    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
