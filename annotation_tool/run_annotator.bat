@echo off
rem 行波波头 Gold 标注工具 - Windows 启动脚本
rem 方式一(conda): conda env create -f environment.yml
rem 方式二(venv):  python -m venv .venv && .venv\Scripts\pip install -r requirements.txt
cd /d "%~dp0"
where conda >nul 2>nul
if %errorlevel%==0 (
    conda run -n pyqt --no-capture-output python -m wavefront_annotator %*
    goto :eof
)
if exist .venv\Scripts\python.exe (
    .venv\Scripts\python.exe -m wavefront_annotator %*
) else (
    python -m wavefront_annotator %*
)
pause
