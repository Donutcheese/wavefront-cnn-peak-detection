# -*- mode: python ; coding: utf-8 -*-
# 完全体：本地标注 + CloudBase 云同步 + 拓扑核心集索引
# 用法: build_annotator.bat  或  python -m PyInstaller --noconfirm --clean wavefront_annotator.spec

from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_submodules

block_cipher = None
root = Path(SPECPATH)
repo = root.parent
assets = root / "wavefront_annotator" / "assets"
icon_ico = assets / "app_icon.ico"
icon_png = assets / "app_icon.png"
core_src = repo / "data" / "derived" / "wavefront_dataset_ningxia_core"
sync_cfg = root / "sync" / "cloudbase.local.json"
scripts_cfg = repo / "scripts" / "cloudbase.local.json"

datas = [(str(assets), "wavefront_annotator/assets")]
binaries = []
hiddenimports = []

# Qt / 绘图
for package in ("PySide6", "pyqtgraph"):
    pkg_datas, pkg_binaries, pkg_hidden = collect_all(package)
    datas += pkg_datas
    binaries += pkg_binaries
    hiddenimports += pkg_hidden

# 云同步依赖
for package in ("requests", "certifi", "charset_normalizer", "idna", "urllib3", "pandas", "numpy"):
    try:
        pkg_datas, pkg_binaries, pkg_hidden = collect_all(package)
        datas += pkg_datas
        binaries += pkg_binaries
        hiddenimports += pkg_hidden
    except Exception:
        hiddenimports += collect_submodules(package)

# 云配置（优先 sync 下）
cfg_path = sync_cfg if sync_cfg.is_file() else scripts_cfg
if cfg_path.is_file():
    datas.append((str(cfg_path), "sync"))
    datas.append((str(cfg_path), "."))  # 亦放到根，便于查找

# 核心集索引与自动标签（不含巨大 h5；.all 仍读本机绝对路径）
if core_src.is_dir():
    for name in ("core_file_index.csv", "phase_labels.csv", "manifest.csv", "core_build_report.json"):
        path = core_src / name
        if path.is_file():
            datas.append((str(path), "wavefront_dataset_ningxia_core"))

a = Analysis(
    ["wavefront_annotator_entry.py"],
    pathex=[str(root)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports
    + [
        "wavefront_annotator",
        "wavefront_annotator.__main__",
        "wavefront_annotator.main_window",
        "wavefront_annotator.resources",
        "wavefront_annotator.all_decoder",
        "wavefront_annotator.auto_labels",
        "wavefront_annotator.label_store",
        "wavefront_annotator.waveform_view",
        "wavefront_annotator.theme",
        "wavefront_annotator.glass_segmented",
        "wavefront_annotator.win_mica",
        "sync",
        "sync.config",
        "sync.factory",
        "sync.models",
        "sync.local_mirror",
        "sync.cloudbase_client",
        "sync.cloud_label_sync",
        "sync.runtime_paths",
        "sync.backend_base",
        "requests",
        "pandas",
        "numpy",
        "h5py",
    ],
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
