# 波头 CNN：PyTorch + Apple M5 GPU

## 推荐入口：全代码 Notebook

如果希望在一个文件中查看和修改全部代码，请直接打开：

`machine_learning/wavefront_cnn/wavefront_cnn_m5_all_in_one.ipynb`

该 Notebook 不导入项目内训练模块，数据集类、增强、粗区间分类网络、区间偏移回归、训练循环、实时训练面板、验证测试和模型导出全部位于 Notebook 内。训练时每个 epoch 会刷新：

- 训练集与验证集损失。
- 验证集 MAE 和 P95 误差。
- 当前学习率。
- 固定验证波形的标签位置与预测位置。
- 训练结束后的误差分布、预测散点和最差波形。

## 技术路线

网络将每个 A/B/C 相作为独立样本，输入 8192 点波形。模型先从 256 个区间中选择波头所在区间，每个区间覆盖 32 个采样点，再回归区间内偏移得到浮点采样坐标。该结构使用 softmax 分类，训练目标与最终 argmax 推理一致，并移除了会产生边界伪峰的转置卷积。训练分两阶段：

1. `hard + soft` 预训练，学习不同线路和噪声条件下的通用行波特征。
2. 加载最佳预训练权重，先冻结编码器，再只使用 `hard` 标签精调。

数据划分直接读取 `phase_labels.csv` 的 `split_event`，不会把同一事件拆到不同集合。线路长度不进入波头 CNN。

完整训练前会自动运行 32 个 hard 样本无增强过拟合自检。只有同时达到 `MAE < 2` 点和 `within_4 = 100%`，才允许启动全量训练。

## 启动 Notebook

```bash
cd /path/to/基于cnn的波峰寻找算法
conda activate m5
python -m ipykernel install --user --name m5 --display-name "Python (m5)"
jupyter lab machine_learning/wavefront_cnn/wavefront_cnn_m5_training.ipynb
```

建议改为直接启动全代码版本：

```bash
jupyter lab machine_learning/wavefront_cnn/wavefront_cnn_m5_all_in_one.ipynb
```

依次运行所有单元。确认 MPS 自检通过后，在配置单元修改：

```python
RUN_FULL_TRAINING = True
```

然后重新运行“两阶段迁移训练”单元。默认配置为准确率与稳定性优先：FP32、batch size 32、预训练 40 epoch、精调 60 epoch、早停 10 epoch。

## 当前训练入口

粗到细版本以 `wavefront_cnn_m5_all_in_one.ipynb` 为唯一推荐训练入口。旧的热图 CLI 和旧检查点不再用于训练或故障定位。

## 输出文件

- `best_pretrain.pt`：hard+soft 最佳预训练权重。
- `best_finetune.pt`：hard-only 最佳精调权重，最终使用此文件。
- `history_pretrain.csv`、`history_finetune.csv`：每轮训练指标。
- `test_predictions.csv`：每个测试样本的预测和误差。
- `test_metrics.json`：MAE、RMSE、P95、微秒误差和命中率。

新结果目录为 `data/derived/wavefront_cnn_coarse_fine_run/`，不会覆盖旧热图模型结果。

## M5 参数调整

- 内存不足：先将 `batch_size` 从 32 降为 16，再降为 8。
- HDF5 多进程不稳定：保持 `num_workers=0`；确认稳定后可尝试 2。
- 准确率优先：保持 `amp=False`，不要启用 MPS 半精度。
- 训练中断：保留输出目录中的最佳检查点；可先使用 `best_pretrain.pt` 进行评估或二次开发。
- 不要使用 `review` 标签训练第一版模型；先检查 `review_queue.csv`。

## 独立评估

```bash
conda run -n m5 env PYTHONPATH=. python -m \
  machine_learning.wavefront_cnn.wavefront_training.evaluate \
  --checkpoint data/derived/wavefront_cnn_run/best_finetune.pt \
  --dataset-dir data/derived/wavefront_dataset_v1 \
  --split test --device mps \
  --output-dir data/derived/wavefront_cnn_run/evaluation
```

在 1.25 MHz 下，1 个采样点对应 0.8 μs。接入故障定位前，应重点检查测试集 `p95_samples` 和 `within_4_samples`，不能只看训练损失。
