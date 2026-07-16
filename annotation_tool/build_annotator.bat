@echo off
rem 打包行波波头 Gold 标注工具（单文件 exe，嵌入 app_icon.ico）
cd /d "%~dp0"
where conda >nul 2>nul
if %errorlevel%==0 (
    conda run -n pyqt --no-capture-output python -m pip install -q "pyinstaller>=6.0"
    conda run -n pyqt --no-capture-output python -m PyInstaller --noconfirm --clean wavefront_annotator.spec
) else if exist .venv\Scripts\python.exe (
    .venv\Scripts\python.exe -m pip install -q "pyinstaller>=6.0"
    .venv\Scripts\python.exe -m PyInstaller --noconfirm --clean wavefront_annotator.spec
) else (
    python -m pip install -q "pyinstaller>=6.0"
    python -m PyInstaller --noconfirm --clean wavefront_annotator.spec
)
echo 打包完成: dist\WavefrontGoldAnnotator.exe
pause
