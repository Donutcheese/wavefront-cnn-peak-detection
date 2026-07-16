# -*- mode: python ; coding: utf-8 -*-
# PyInstaller 规格：打包后任务栏 / Dock / 可执行文件均使用同一 app_icon
# 用法见 build_annotator.sh / build_annotator.bat

from pathlib import Path

from PyInstaller.utils.hooks import collect_all

block_cipher = None
root = Path(SPECPATH)
assets = root / "wavefront_annotator" / "assets"
icon_ico = assets / "app_icon.ico"
icon_png = assets / "app_icon.png"

datas = [(str(assets), "wavefront_annotator/assets")]
binaries = []
hiddenimports = []

for package in ("PySide6", "pyqtgraph"):
    pkg_datas, pkg_binaries, pkg_hidden = collect_all(package)
    datas += pkg_datas
    binaries += pkg_binaries
    hiddenimports += pkg_hidden

a = Analysis(
    ["wavefront_annotator/__main__.py"],
    pathex=[str(root)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports + ["wavefront_annotator"],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# Windows 用 .ico；macOS/Linux 用高清 PNG（PyInstaller 会映射为应用图标）
exe_icon = str(icon_ico if icon_ico.exists() else icon_png)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="WavefrontGoldAnnotator",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=exe_icon,
)
