#!/usr/bin/env bash
# 打包行波波头 Gold 标注工具（单文件可执行，嵌入 app_icon）
# 用法: ./build_annotator.sh
set -euo pipefail
cd "$(dirname "$0")"

if command -v conda >/dev/null 2>&1; then
  PY=(conda run -n pyqt --no-capture-output python)
else
  PY=(python)
fi

"${PY[@]}" -m pip install -q "pyinstaller>=6.0"
"${PY[@]}" -m PyInstaller --noconfirm --clean wavefront_annotator.spec
echo "打包完成: dist/WavefrontGoldAnnotator"
