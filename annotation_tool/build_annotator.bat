@echo off
rem 打包完全体：本地标注 + CloudBase 云同步 + 拓扑核心集索引
cd /d "%~dp0"
if exist .venv\Scripts\python.exe (
    .venv\Scripts\python.exe -m pip install -q "pyinstaller>=6.0" requests pandas h5py
    .venv\Scripts\python.exe -m PyInstaller --noconfirm --clean wavefront_annotator.spec
) else (
    python -m pip install -q "pyinstaller>=6.0" requests pandas h5py
    python -m PyInstaller --noconfirm --clean wavefront_annotator.spec
)
if exist dist\WavefrontGoldAnnotator.exe (
    if exist sync\cloudbase.local.json copy /Y sync\cloudbase.local.json dist\cloudbase.local.json >nul
    if exist ..\scripts\cloudbase.local.json copy /Y ..\scripts\cloudbase.local.json dist\cloudbase.local.json >nul
    echo 打包完成: dist\WavefrontGoldAnnotator.exe
    echo 已复制 cloudbase.local.json 到 dist\ （可覆盖更新临时密钥）
) else (
    echo 打包失败：未生成 exe
    exit /b 1
)
pause
