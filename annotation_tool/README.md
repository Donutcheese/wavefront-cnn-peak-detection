# 行波波头 Gold 标注工具

面向本仓库阶段 1（数据集 v2 重建）的人工复核客户端，用于建立 gold 标签集。
技术栈：`PySide6` + `pyqtgraph`，Windows / macOS / Linux 通用。

![界面截图](../docs/assets/annotation_tool_gold_client.png)

## 功能

| 功能 | 说明 |
|---|---|
| 目录选择 | 递归扫描所选目录下全部 `.all` / `.vall` 录波文件 |
| 波形可视化 | **ABC 单窗切换**：主图同一时刻只显示活动相；下方 `A 相 / B 相 / C 相` 按钮或 `1/2/3` 换相；各相卡尺与区间状态独立保留 |
| 卡尺标注 | 每相一条可拖动红色卡尺，双击落点，←/→ 逐点微调（Shift 步长 10） |
| 框线区间 | `R` 键在卡尺附近开/关 `LinearRegionItem`，用于圈定搜索窗或存疑范围 |
| 导数辅助 | 可叠加 &#124;di/dt&#124; 曲线，辅助定位波头突变沿 |
| 自动标签对照 | 可加载 `phase_labels.csv`（紫色点划线 + 各检测器候选虚线）、`review_queue.csv`、`stage0_worst30.csv`，文件列表按复核优先级排序 |
| CSV 同步 | 每次保存立即原子写入录波目录下的 `gold_labels.csv`（临时文件 + 替换，崩溃不丢数据），主键 `(file_name, phase)` upsert |
| 应用图标 | 开发运行与打包产物共用 `wavefront_annotator/assets/app_icon.*`（窗口 / 任务栏 / exe 图标） |

## 安装与启动

推荐使用 conda（macOS / Windows 通用，环境名 `pyqt`）：

```bash
cd annotation_tool
conda env create -f environment.yml
```

macOS 启动：

```bash
./run_annotator.sh /path/to/录波目录
```

Windows 启动：双击 `run_annotator.bat`（自动优先使用 conda `pyqt` 环境，无 conda 时回退 `.venv`）。

不用 conda 时的 venv 方式（Windows）：

```bat
cd annotation_tool
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
run_annotator.bat
```

## 打包（嵌入应用图标）

```bash
# macOS / Linux
./build_annotator.sh

# Windows
build_annotator.bat
```

产物位于 `dist/WavefrontGoldAnnotator`（Windows 为 `.exe`）。PyInstaller 规格 `wavefront_annotator.spec` 已将 `assets/app_icon.ico`（或 PNG）写入可执行文件图标，并打包 `assets/` 供运行时 `QApplication` / 主窗口加载。

命令行直接指定目录与参考 CSV：

```bat
.venv\Scripts\python -m wavefront_annotator D:\录波数据 ^
  --auto-labels ..\data\derived\wavefront_dataset_v1\phase_labels.csv ^
  --auto-labels ..\data\derived\wavefront_dataset_v1\review_queue.csv ^
  --auto-labels ..\wavefront_stage0_analysis\stage0_worst30.csv
```

## 快捷键

| 按键 | 动作 |
|---|---|
| `1` / `2` / `3` 或点击相按钮 | 切换主图显示的活动相 A / B / C |
| 双击波形 | 在该相放置卡尺 |
| `←` / `→`（`Shift` 加速） | 卡尺逐点微调 |
| `G` / `U` / `X` | 活动相状态设为 gold / unsure / reject |
| `R` | 框线区间开/关 |
| `Z` | 缩放到活动相卡尺附近 |
| `Home` | 复位视图 |
| `Space` | 保存当前文件标注 |
| `Enter` | 保存并跳到下一个文件 |
| `PgUp` / `PgDn` | 上/下一个文件 |

## gold_labels.csv 字段

| 列 | 含义 |
|---|---|
| `file_name` / `file_path` / `phase` | 主键与来源 |
| `gold_wavefront_index` | 人工确认的波头位置（**原始录波采样点坐标**，非窗口坐标） |
| `gold_time_us` | 对应时刻（µs），按 GPS 频率字段换算 |
| `region_start_index` / `region_end_index` | 框线区间（可空） |
| `status` | `gold` / `unsure` / `reject` |
| `auto_wavefront_index` | 自动伪标签位置，供审计对比 |
| `sampling_rate_hz` / `annotator` / `updated_at` | 换算依据与审计信息 |

## 标注流程建议（对接优化方案阶段 1）

1. 加载三个参考 CSV，列表自动置顶“最差样本”和“复核队列”。
2. 优先标注红色高亮文件；打开导数辅助，双击首个陡峭突变沿，微调卡尺。
3. 波头清晰 → `G`；多候选难以判定 → `U` 并用 `R` 圈定候选区间；坏波形/无波头 → `X`。
4. `Enter` 保存并进入下一个文件；`gold_labels.csv` 实时落盘。
5. 后续数据集 v2 重建脚本直接消费 `gold_labels.csv` 的 `raw` 坐标，与固定物理搜索窗方案对接。

## 测试

```bash
python tests/test_smoke.py
```

覆盖：`gold_labels.csv` upsert 往返；如果本地存在 `tests/sample_data/` 和数据集标签表，则额外执行解码器数值校验、主窗口离屏加载与保存。
