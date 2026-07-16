@echo off
rem 行波波头 Gold 标注工具 - 默认联立 CloudBase ningxia_core 拓扑核心集
rem 启动即: pull 云标签 + 加载 core_file_index.csv 对应 .all
cd /d "%~dp0"
if exist .venv\Scripts\python.exe (
    .venv\Scripts\python.exe -m wavefront_annotator %*
) else (
    python -m wavefront_annotator %*
)
pause
