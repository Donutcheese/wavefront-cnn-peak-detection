"""应用资源路径：logo / 窗口图标 / 打包图标。"""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtGui import QIcon


def assets_dir() -> Path:
    """开发运行与 PyInstaller 冻结环境统一解析 assets 目录。"""
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS) / "wavefront_annotator" / "assets"
    return Path(__file__).resolve().parent / "assets"


def icon_png_path() -> Path:
    return assets_dir() / "app_icon.png"


def icon_ico_path() -> Path:
    return assets_dir() / "app_icon.ico"


def load_app_icon() -> QIcon:
    """加载多分辨率应用图标（开发运行与打包后共用同一资源）。"""
    icon = QIcon()
    root = assets_dir()
    ico = root / "app_icon.ico"
    if ico.exists():
        icon.addFile(str(ico))
    for size in (16, 32, 48, 64, 128, 256, 512):
        png = root / f"app_icon_{size}.png"
        if png.exists():
            icon.addFile(str(png))
    fallback = root / "app_icon.png"
    if icon.isNull() and fallback.exists():
        icon.addFile(str(fallback))
    return icon
