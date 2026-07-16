@echo off
rem 1) 打包含云功能的 exe  2) 组装标注组便携 Kit（exe+配置+相对索引+核心.all）
cd /d "%~dp0"
if exist .venv\Scripts\python.exe (
    set PY=.venv\Scripts\python.exe
) else (
    set PY=python
)

echo === [1/2] PyInstaller 完全体 exe ===
%PY% -m pip install -q "pyinstaller>=6.0" requests pandas h5py
%PY% -m PyInstaller --noconfirm --clean wavefront_annotator.spec
if errorlevel 1 exit /b 1
if exist sync\cloudbase.local.json copy /Y sync\cloudbase.local.json dist\cloudbase.local.json >nul

echo === [2/2] 组装 WavefrontAnnotatorKit ===
%PY% ..\scripts\build_annotator_kit.py
if errorlevel 1 exit /b 1
echo.
echo 完成。分发目录: ..\dist\WavefrontAnnotatorKit\
pause
