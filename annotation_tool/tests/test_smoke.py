"""冒烟测试：解码器数值校验 + 标签存储读写往返 + 主窗口离屏实例化。"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wavefront_annotator.all_decoder import decode_all_file
from wavefront_annotator.label_store import GoldLabel, GoldLabelStore

SAMPLE_DIR = Path(__file__).resolve().parent / "sample_data"
DEMO_FILE = SAMPLE_DIR / "140423231753左昌线M0170.all"
PHASE_LABELS = Path(__file__).resolve().parents[2] / "data/derived/wavefront_dataset_v1/phase_labels.csv"


def _has_local_sample_data() -> bool:
    return DEMO_FILE.exists()


def test_decoder() -> None:
    if not _has_local_sample_data():
        print("decoder skipped: tests/sample_data is local and not committed")
        return
    record = decode_all_file(DEMO_FILE)
    # 对照 manifest.csv: data_length=16384, sampling_rate_hz=624985.81
    assert record.data_length == 16384, record.data_length
    assert abs(record.sampling_rate_hz - 624985.81) < 0.01, record.sampling_rate_hz
    assert record.signals.shape == (3, 16384)
    assert np.isfinite(record.signals).all()
    # 对照 phase_labels.csv: A 相 raw_wavefront_index=14042 附近应有明显突变
    deriv = np.abs(np.diff(record.signals[0]))
    window_peak = deriv[14000:14100].max()
    assert window_peak > np.percentile(deriv, 99), "波头附近缺少突变，解码可能错位"
    print(f"decoder ok: length={record.data_length}, fs={record.sampling_rate_hz:.2f} Hz")


def test_label_store() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = Path(tmp) / "gold_labels.csv"
        store = GoldLabelStore(csv_path)
        store.upsert(
            GoldLabel(
                file_name="a.all",
                file_path="/x/a.all",
                phase="A",
                gold_wavefront_index=14042.0,
                gold_time_us=22467.71,
                status="gold",
                region_start_index=13800.0,
                region_end_index=14300.0,
                auto_wavefront_index=14042.0,
                sampling_rate_hz=624985.81,
                annotator="tester",
            )
        )
        # 覆盖更新同一主键
        store.upsert(
            GoldLabel(
                file_name="a.all",
                file_path="/x/a.all",
                phase="A",
                gold_wavefront_index=14045.0,
                gold_time_us=22472.51,
                status="unsure",
            )
        )
        reloaded = GoldLabelStore(csv_path)
        assert reloaded.count() == 1
        entry = reloaded.get("a.all", "A")
        assert entry is not None
        assert float(entry["gold_wavefront_index"]) == 14045.0
        assert entry["status"] == "unsure"
    print("label store ok: upsert 覆盖与原子落盘正常")


def test_main_window_offscreen() -> None:
    if not _has_local_sample_data():
        print("main window skipped: tests/sample_data is local and not committed")
        return
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    from wavefront_annotator.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    (SAMPLE_DIR / "gold_labels.csv").unlink(missing_ok=True)
    window = MainWindow()
    if PHASE_LABELS.exists():
        window.auto_index.load_phase_labels(PHASE_LABELS)
    window.load_directory(SAMPLE_DIR)
    assert window.current_record is not None
    # 模拟标注：A 相设为 gold 并保存
    window.phase_panels["A"].set_status_key("gold")
    window.save_current()
    assert window.store is not None and window.store.count() >= 1
    saved = window.store.phases_of(window.current_record.file_name)
    assert "A" in saved
    print(f"main window ok: 加载 {len(window.files)} 个文件，保存 gold 记录 {window.store.count()} 条")
    window.close()


if __name__ == "__main__":
    test_decoder()
    test_label_store()
    test_main_window_offscreen()
    print("all smoke tests passed")
