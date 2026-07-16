# 宁夏拓扑核心集 + 标注工具联立

## 筛选规则

1. SVG：`FaultLocation_demo/docs/现场终端线路拓扑图_更新后.svg`（41 条线路）
2. Excel：`现场终端线路拓扑参数.xlsx`（线路参数精确匹配）
3. 宁夏 `.all`：文件名可解析且 `TopologyResolver.match_status == exact`，且线路 ∈ SVG∩Excel

结果：**2103** 个录波 / **6309** 相标签 / h5 `[2103,3,8192]` ≈ 185 MB

## 本地产物

`data/derived/wavefront_dataset_ningxia_core/`

- `waveforms.h5`
- `phase_labels.csv`（默认 unlabeled）
- `manifest.csv`（含 line_length / M-N RTU / source_path）
- `core_file_index.csv`（标注工具用绝对路径）

重建：

```bat
annotation_tool\.venv\Scripts\python.exe scripts\build_ningxia_core_topology_dataset.py
```

## 云端（环境 `wavefrontdataset-d0e13om229bd53d`）

| 资源 | 内容 |
|---|---|
| 云存储 | `wavefront/ningxia_core/signals/waveforms.h5` |
| `wf_samples` | dataset=`ningxia_core`，2103 条 + 拓扑字段 |
| `wf_phase_labels` | 6309 条，初始 unlabeled |

## 标注工具联立（写死）

| 项 | 值 |
|---|---|
| 数据集 | `ningxia_core` |
| 用户 | `wavefront_operator` |
| 环境 | `wavefrontdataset-d0e13om229bd53d` |

启动：`annotation_tool\run_annotator.bat`

流程：

1. 启动 → pull 云标签 → 加载核心集 `.all` 索引  
2. 标注保存 → 本地 `gold_labels.csv` + upsert 云 `wf_phase_labels`  
3. 工具栏「拉取云标签」可手动再同步  

临时密钥在 `scripts/cloudbase.local.json`（gitignore）。过期后在对话执行「登录云开发」并重新导出临时密钥即可。
