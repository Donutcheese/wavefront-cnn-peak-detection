# CloudBase 开通与 P2 同步说明

## 1. 控制台准备（真云）

1. 打开 [腾讯云 CloudBase 控制台](https://console.cloud.tencent.com/tcb)，创建**免费体验版**环境（上海/广州）。
2. 记下 **环境 ID**（`envId`）。
3. 在「数据库」中新建集合（也可首次写入时自动创建）：
  - `wf_samples`
  - `wf_phase_labels`
4. 开通「云存储」。
5. 获取腾讯云 API 密钥：访问管理 → API 密钥 → `SecretId` / `SecretKey`
  （仅用于本机 `push`/`pull` 脚本，**禁止打进 exe**）。

安全规则建议（开发期可先放宽，上线再收紧）：

- 认证用户可读 `wf_samples` / `wf_phase_labels`
- 认证用户可写 `wf_phase_labels` 的标签字段
- `sample_index` / `storage_key` 仅管理员脚本写入

## 2. 本地配置

```bat
copy scripts\cloudbase.local.json.example scripts\cloudbase.local.json
```

编辑 `scripts/cloudbase.local.json`：


| 字段                                    | 说明                                          |
| ------------------------------------- | ------------------------------------------- |
| `backend`                             | `local_mirror`（无账号自测）或 `cloudbase`（真云）      |
| `env_id` / `secret_id` / `secret_key` | 真云必填                                        |
| `dataset`                             | 默认 `ningxia`                                |
| `storage_key`                         | 默认 `wavefront/ningxia/signals/waveforms.h5` |


也可用环境变量：`TCB_BACKEND`、`TCB_ENV`、`TENCENTCLOUD_SECRETID`、`TENCENTCLOUD_SECRETKEY`。

`cloudbase.local.json` 已加入 `.gitignore`，勿提交密钥。

## 3. 集合字段（与大纲一致）

### `wf_samples`

`_id(=sample_id)`, `dataset`, `file_name`, `sample_index`, `window_samples`, `target_fs_hz`, `source_fs_hz`, `storage_key`, `file_id`, `split_event`, `created_at`, `updated_at`

### `wf_phase_labels`

`_id(=sample_id:phase)`, `dataset`, `sample_id`, `sample_index`, `phase`, `window_wavefront_index`, `raw_wavefront_index`, `region_*`, `label_status`, `annotator`, `note`, `rev`, `updated_at`, `auto_*`

## 4. 命令

依赖：`annotation_tool/.venv`（含 `requests` `h5py` `pandas`）。

### 本地镜像往返（无需腾讯云账号）

```bat
cd annotation_tool
.venv\Scripts\python.exe ..\scripts\push_local_a_to_cloudbase.py
.venv\Scripts\python.exe ..\scripts\pull_cloudbase_to_local_a.py --skip-h5
```

镜像目录默认：`data/derived/cloudbase_mirror/`。

### 推真云

把配置里 `backend` 改为 `cloudbase` 并填密钥后：

```bat
.venv\Scripts\python.exe ..\scripts\push_local_a_to_cloudbase.py
```

宁夏全量约 210 MB h5 + 2394 samples + 7182 labels，注意免费点额度。

### 拉回训练

```bat
.venv\Scripts\python.exe ..\scripts\pull_cloudbase_to_local_a.py
```

默认将云端 `gold` → 本地 `hard`，可被现有 `build_phase_index` 消费。仅标签更新、本地已有 h5 时加 `--skip-h5`。

## 5. 验收标准（P2）


| 项    | 标准                                                                                |
| ---- | --------------------------------------------------------------------------------- |
| push | `push_report.json` 中 `remote_sample_count>=2394` 且 `remote_label_count>=7182`（全量） |
| pull | 还原 `phase_labels.csv`；`gold→hard` 后可被训练索引过滤                                       |
| 密钥   | 不出现在仓库与 exe 中                                                                     |


