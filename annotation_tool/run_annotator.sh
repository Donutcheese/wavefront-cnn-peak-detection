#!/usr/bin/env bash
# 行波波头 Gold 标注工具 - macOS/Linux 启动脚本（conda pyqt 环境）
# 用法: ./run_annotator.sh [录波目录] [--auto-labels CSV ...]
set -euo pipefail
cd "$(dirname "$0")"
exec conda run -n pyqt --no-capture-output python -m wavefront_annotator "$@"
