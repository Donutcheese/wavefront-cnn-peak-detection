"""主窗口：目录扫描、文件列表、三相标注面板、CSV 同步。"""

from __future__ import annotations

import getpass
import traceback
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QColor, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from .all_decoder import WaveRecord, decode_all_file
from .auto_labels import AutoLabelIndex
from .label_store import GoldLabel, GoldLabelStore
from .resources import load_app_icon
from .waveform_view import PHASES, PHASE_COLORS, WaveformView

# 相切换按钮：活动相高亮，其余弱化（贴近截图中的单窗切换交互）
_PHASE_BTN_STYLE = """
QPushButton {{
    background-color: #2a3036;
    color: #c5ccd3;
    border: 1px solid #3a424a;
    border-radius: 4px;
    padding: 10px 0;
    font-size: 14px;
    font-weight: 600;
}}
QPushButton:hover {{
    background-color: #343b42;
    border-color: {accent};
}}
QPushButton:checked {{
    background-color: #6b2d3c;
    color: #ffffff;
    border-color: {accent};
}}
"""

PRIORITY_TEXT = {0: "最差样本", 1: "复核队列", 2: ""}
STATUS_TEXT = {"gold": "✔ gold", "unsure": "? 存疑", "reject": "✘ 拒绝"}
DONE_COLOR = QColor("#2e7d32")
PARTIAL_COLOR = QColor("#f9a825")
PRIORITY_COLOR = QColor("#c62828")


class PhaseAnnotationPanel(QGroupBox):
    """单相标注控制：卡尺读数、状态选择、区间显示。"""

    def __init__(self, phase: str, parent: QWidget | None = None) -> None:
        super().__init__(f"{phase} 相", parent)
        self.phase = phase
        layout = QGridLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)

        self.position_label = QLabel("卡尺: —")
        self.time_label = QLabel("时间: —")
        self.auto_label = QLabel("auto: —")
        self.status_combo = QComboBox()
        self.status_combo.addItems(["未标注", "gold（确认）", "unsure（存疑）", "reject（拒绝）"])
        self.region_label = QLabel("区间: —")

        layout.addWidget(self.position_label, 0, 0)
        layout.addWidget(self.time_label, 0, 1)
        layout.addWidget(self.auto_label, 1, 0)
        layout.addWidget(self.region_label, 1, 1)
        layout.addWidget(self.status_combo, 2, 0, 1, 2)

    def status_key(self) -> str | None:
        return {1: "gold", 2: "unsure", 3: "reject"}.get(self.status_combo.currentIndex())

    def set_status_key(self, key: str | None) -> None:
        index = {"gold": 1, "unsure": 2, "reject": 3}.get(key or "", 0)
        self.status_combo.setCurrentIndex(index)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("行波波头 Gold 标注工具")
        self.resize(1500, 900)
        self.setWindowIcon(load_app_icon())

        self.auto_index = AutoLabelIndex()
        self.store: GoldLabelStore | None = None
        self.files: list[Path] = []
        self.current_record: WaveRecord | None = None
        self.current_row = -1
        self.annotator = getpass.getuser()

        self._build_ui()
        self._build_shortcuts()
        self._sync_phase_buttons("A")

    # ---------- UI 构建 ----------

    def _build_ui(self) -> None:
        toolbar = self.addToolBar("main")
        toolbar.setMovable(False)

        open_action = QAction("打开录波目录…", self)
        open_action.triggered.connect(self.choose_directory)
        toolbar.addAction(open_action)

        load_auto_action = QAction("加载自动标签CSV…", self)
        load_auto_action.triggered.connect(self.choose_auto_labels)
        toolbar.addAction(load_auto_action)

        toolbar.addSeparator()
        self.deriv_checkbox = QCheckBox("导数辅助 |di/dt|")
        self.deriv_checkbox.toggled.connect(lambda on: self.waveform_view.set_derivative_visible(on))
        toolbar.addWidget(self.deriv_checkbox)

        reset_action = QAction("复位视图 (Home)", self)
        reset_action.triggered.connect(lambda: self.waveform_view.reset_view())
        toolbar.addAction(reset_action)

        # 左侧：文件列表
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(4, 4, 4, 4)
        self.filter_edit = QLineEdit()
        self.filter_edit.setPlaceholderText("过滤文件名…")
        self.filter_edit.textChanged.connect(self._apply_filter)
        self.file_list = QListWidget()
        self.file_list.currentRowChanged.connect(self._on_file_selected)
        left_layout.addWidget(self.filter_edit)
        left_layout.addWidget(self.file_list)

        # 中间：单相波形视图（A/B/C 切换显示）
        self.waveform_view = WaveformView()
        self.waveform_view.goldChanged.connect(self._on_gold_changed)
        self.waveform_view.phaseFocused.connect(self._on_phase_focused)

        # 相切换条：同一时刻只展开一相，避免三联图挤占标注精度
        phase_switch = QWidget()
        phase_switch_layout = QHBoxLayout(phase_switch)
        phase_switch_layout.setContentsMargins(4, 4, 4, 2)
        phase_switch_layout.setSpacing(8)
        self.phase_button_group = QButtonGroup(self)
        self.phase_button_group.setExclusive(True)
        self.phase_buttons: dict[str, QPushButton] = {}
        for phase in PHASES:
            button = QPushButton(f"{phase} 相")
            button.setCheckable(True)
            button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            button.setMinimumHeight(40)
            button.setStyleSheet(_PHASE_BTN_STYLE.format(accent=PHASE_COLORS[phase]))
            button.clicked.connect(lambda _checked=False, p=phase: self._focus_phase(p))
            self.phase_button_group.addButton(button)
            self.phase_buttons[phase] = button
            phase_switch_layout.addWidget(button)

        # 底部：三相标注面板 + 操作按钮（面板始终可见，便于跨相对照）
        bottom = QWidget()
        bottom_layout = QHBoxLayout(bottom)
        bottom_layout.setContentsMargins(4, 2, 4, 2)
        self.phase_panels: dict[str, PhaseAnnotationPanel] = {}
        for phase in PHASES:
            panel = PhaseAnnotationPanel(phase)
            panel.status_combo.currentIndexChanged.connect(self._mark_dirty)
            self.phase_panels[phase] = panel
            bottom_layout.addWidget(panel, stretch=1)

        button_column = QVBoxLayout()
        self.save_button = QPushButton("保存 (Space)")
        self.save_button.clicked.connect(self.save_current)
        self.save_next_button = QPushButton("保存并下一个 (Enter)")
        self.save_next_button.clicked.connect(self.save_and_next)
        self.region_button = QPushButton("框线区间开/关 (R)")
        self.region_button.clicked.connect(self._toggle_region)
        button_column.addWidget(self.save_button)
        button_column.addWidget(self.save_next_button)
        button_column.addWidget(self.region_button)
        bottom_layout.addLayout(button_column)

        center = QWidget()
        center_layout = QVBoxLayout(center)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.addWidget(self.waveform_view, stretch=1)
        center_layout.addWidget(phase_switch)
        center_layout.addWidget(bottom)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(center)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([340, 1160])
        self.setCentralWidget(splitter)

        self.setStatusBar(QStatusBar())
        self._update_status("打开录波目录开始标注。双击波形放置卡尺，拖动微调，←/→ 逐点移动。")

    def _build_shortcuts(self) -> None:
        bindings = [
            ("1", lambda: self._focus_phase("A")),
            ("2", lambda: self._focus_phase("B")),
            ("3", lambda: self._focus_phase("C")),
            ("Left", lambda: self.waveform_view.nudge_active_gold(-1)),
            ("Right", lambda: self.waveform_view.nudge_active_gold(+1)),
            ("Shift+Left", lambda: self.waveform_view.nudge_active_gold(-10)),
            ("Shift+Right", lambda: self.waveform_view.nudge_active_gold(+10)),
            ("Space", self.save_current),
            ("Return", self.save_and_next),
            ("R", self._toggle_region),
            ("Z", lambda: self.waveform_view.zoom_active_to_gold()),
            ("Home", lambda: self.waveform_view.reset_view()),
            ("PgDown", lambda: self._navigate(+1)),
            ("PgUp", lambda: self._navigate(-1)),
            ("G", lambda: self._set_active_status("gold")),
            ("U", lambda: self._set_active_status("unsure")),
            ("X", lambda: self._set_active_status("reject")),
        ]
        for key, handler in bindings:
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.setContext(Qt.ShortcutContext.WindowShortcut)
            shortcut.activated.connect(handler)

    # ---------- 目录与文件加载 ----------

    def choose_directory(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "选择包含 .all/.vall 的目录")
        if not directory:
            return
        self.load_directory(Path(directory))

    def load_directory(self, directory: Path) -> None:
        files = sorted(
            [p for p in directory.rglob("*") if p.suffix.lower() in (".all", ".vall")],
            key=lambda p: (self.auto_index.priority_for(p.name), p.name),
        )
        if not files:
            QMessageBox.warning(self, "未找到文件", f"目录中没有 .all/.vall 文件:\n{directory}")
            return
        self.files = files
        self.store = GoldLabelStore(directory / "gold_labels.csv")
        self._rebuild_file_list()
        self.file_list.setCurrentRow(0)
        self._update_status(
            f"已加载 {len(files)} 个录波文件，gold 标签写入 {directory / 'gold_labels.csv'}"
        )

    def choose_auto_labels(self) -> None:
        path_str, _ = QFileDialog.getOpenFileName(
            self, "选择 phase_labels.csv / review_queue.csv / stage0_worst30.csv", "", "CSV (*.csv)"
        )
        if not path_str:
            return
        path = Path(path_str)
        try:
            if "review_queue" in path.name:
                count = self.auto_index.load_review_queue(path)
                message = f"复核队列 {count} 个文件"
            elif "worst" in path.name:
                count = self.auto_index.load_worst30(path)
                message = f"最差样本 {count} 个文件"
            else:
                count = self.auto_index.load_phase_labels(path)
                message = f"自动相标签 {count} 条"
        except Exception as exc:  # noqa: BLE001 - 需要把任意解析失败反馈给用户
            QMessageBox.critical(self, "加载失败", f"{path.name}\n{exc}")
            return
        if self.files:
            current = self.files[self.current_row].name if self.current_row >= 0 else None
            self.files.sort(key=lambda p: (self.auto_index.priority_for(p.name), p.name))
            self._rebuild_file_list()
            if current is not None:
                for row, p in enumerate(self.files):
                    if p.name == current:
                        self.file_list.setCurrentRow(row)
                        break
        if self.current_record is not None:
            self._apply_auto_labels(self.current_record)
        self._update_status(f"已加载 {message}（文件列表已按优先级重排）")

    def _rebuild_file_list(self) -> None:
        self.file_list.blockSignals(True)
        self.file_list.clear()
        for path in self.files:
            item = QListWidgetItem(self._file_item_text(path))
            item.setData(Qt.ItemDataRole.UserRole, str(path))
            self._style_item(item, path)
            self.file_list.addItem(item)
        self.file_list.blockSignals(False)
        self._apply_filter(self.filter_edit.text())

    def _file_item_text(self, path: Path) -> str:
        tags = []
        priority_text = PRIORITY_TEXT[self.auto_index.priority_for(path.name)]
        if priority_text:
            tags.append(priority_text)
        if self.store is not None:
            done = len(self.store.phases_of(path.name))
            if done:
                tags.append(f"{done}/3")
        suffix = f"  [{' | '.join(tags)}]" if tags else ""
        return f"{path.name}{suffix}"

    def _style_item(self, item: QListWidgetItem, path: Path) -> None:
        done = len(self.store.phases_of(path.name)) if self.store else 0
        if done >= 3:
            item.setForeground(DONE_COLOR)
        elif done > 0:
            item.setForeground(PARTIAL_COLOR)
        elif self.auto_index.priority_for(path.name) < 2:
            item.setForeground(PRIORITY_COLOR)

    def _apply_filter(self, text: str) -> None:
        needle = text.strip().lower()
        for row in range(self.file_list.count()):
            item = self.file_list.item(row)
            item.setHidden(bool(needle) and needle not in item.text().lower())

    # ---------- 文件切换与渲染 ----------

    def _on_file_selected(self, row: int) -> None:
        if row < 0 or row >= len(self.files):
            return
        self.current_row = row
        path = self.files[row]
        try:
            record = decode_all_file(path)
        except Exception as exc:  # noqa: BLE001 - 解码失败不应中断标注流程
            traceback.print_exc()
            QMessageBox.critical(self, "解码失败", f"{path.name}\n{exc}")
            return
        self.current_record = record
        self.waveform_view.load_record(record.signals)
        self._apply_auto_labels(record)
        self._restore_saved_labels(record)
        self._refresh_phase_panels()
        fs_text = f"{record.sampling_rate_hz / 1e6:.4f} MHz" + ("" if record.sampling_rate_valid else "（名义值）")
        self._update_status(
            f"[{row + 1}/{len(self.files)}] {record.file_name}  点数={record.data_length}  采样率={fs_text}  {record.timestamp_text}"
        )

    def _apply_auto_labels(self, record: WaveRecord) -> None:
        info = self.auto_index.info_for(record.file_name)
        for phase in PHASES:
            phase_plot = self.waveform_view.phase_plots[phase]
            auto = info.phases.get(phase) if info else None
            if auto is not None:
                phase_plot.set_auto_label(auto.raw_wavefront_index)
                phase_plot.set_candidates(auto.detector_indices)
                text = "auto: —"
                if auto.raw_wavefront_index is not None:
                    text = f"auto: {auto.raw_wavefront_index:.0f} ({auto.label_status}, conf={auto.confidence or 0:.2f})"
                self.phase_panels[phase].auto_label.setText(text)
            else:
                phase_plot.set_auto_label(None)
                phase_plot.set_candidates({})
                self.phase_panels[phase].auto_label.setText("auto: —")

    def _restore_saved_labels(self, record: WaveRecord) -> None:
        saved = self.store.phases_of(record.file_name) if self.store else {}
        info = self.auto_index.info_for(record.file_name)
        for i, phase in enumerate(PHASES):
            phase_plot = self.waveform_view.phase_plots[phase]
            entry = saved.get(phase)
            if entry is not None:
                phase_plot.set_gold(float(entry["gold_wavefront_index"]))
                start, end = entry.get("region_start_index"), entry.get("region_end_index")
                has_region = start is not None and end is not None and not (
                    isinstance(start, float) and np.isnan(start)
                )
                phase_plot.set_region((float(start), float(end)) if has_region else None)
                self.phase_panels[phase].set_status_key(str(entry["status"]))
                continue
            # 无已保存标签时给出初始卡尺：优先自动标签，其次导数极大点
            auto = info.phases.get(phase) if info else None
            if auto is not None and auto.raw_wavefront_index is not None:
                initial = auto.raw_wavefront_index
            else:
                signal = record.signals[i]
                initial = float(np.argmax(np.abs(np.diff(signal)))) if signal.size > 1 else 0.0
            phase_plot.set_gold(initial)
            phase_plot.set_region(None)
            self.phase_panels[phase].set_status_key(None)

    # ---------- 标注交互 ----------

    def _on_gold_changed(self, phase: str, index: float) -> None:
        self._refresh_phase_panel(phase)

    def _on_phase_focused(self, phase: str) -> None:
        self._sync_phase_buttons(phase)
        for name, panel in self.phase_panels.items():
            panel.setStyleSheet("QGroupBox { border: 1px solid #ff5252; }" if name == phase else "")

    def _sync_phase_buttons(self, phase: str) -> None:
        button = self.phase_buttons.get(phase)
        if button is not None and not button.isChecked():
            button.setChecked(True)

    def _focus_phase(self, phase: str) -> None:
        self.waveform_view.set_active_phase(phase, emit_focus=False)
        self._on_phase_focused(phase)

    def _toggle_region(self) -> None:
        phase = self.waveform_view.active_phase
        phase_plot = self.waveform_view.phase_plots[phase]
        center = phase_plot.gold_position()
        if center is not None:
            phase_plot.toggle_region_around(center)
            self._refresh_phase_panel(phase)

    def _set_active_status(self, key: str) -> None:
        self.phase_panels[self.waveform_view.active_phase].set_status_key(key)

    def _mark_dirty(self) -> None:
        pass  # 状态改变即时体现在面板；落盘统一由保存动作触发

    def _refresh_phase_panels(self) -> None:
        for phase in PHASES:
            self._refresh_phase_panel(phase)

    def _refresh_phase_panel(self, phase: str) -> None:
        if self.current_record is None:
            return
        panel = self.phase_panels[phase]
        phase_plot = self.waveform_view.phase_plots[phase]
        position = phase_plot.gold_position()
        if position is None:
            panel.position_label.setText("卡尺: —")
            panel.time_label.setText("时间: —")
        else:
            panel.position_label.setText(f"卡尺: {position:.1f}")
            time_us = position / self.current_record.sampling_rate_hz * 1e6
            panel.time_label.setText(f"时间: {time_us:.2f} µs")
        bounds = phase_plot.region_bounds()
        panel.region_label.setText(
            f"区间: [{bounds[0]:.0f}, {bounds[1]:.0f}]" if bounds else "区间: —"
        )

    # ---------- 保存与导航 ----------

    def save_current(self) -> None:
        if self.current_record is None or self.store is None:
            return
        record = self.current_record
        info = self.auto_index.info_for(record.file_name)
        saved_phases = []
        for phase in PHASES:
            panel = self.phase_panels[phase]
            status = panel.status_key()
            if status is None:
                continue  # 未标注的相不写入
            phase_plot = self.waveform_view.phase_plots[phase]
            position = phase_plot.gold_position()
            if position is None:
                continue
            bounds = phase_plot.region_bounds()
            auto = info.phases.get(phase) if info else None
            self.store.upsert(
                GoldLabel(
                    file_name=record.file_name,
                    file_path=record.file_path,
                    phase=phase,
                    gold_wavefront_index=round(position, 1),
                    gold_time_us=round(position / record.sampling_rate_hz * 1e6, 3),
                    status=status,
                    region_start_index=round(bounds[0], 1) if bounds else None,
                    region_end_index=round(bounds[1], 1) if bounds else None,
                    auto_wavefront_index=auto.raw_wavefront_index if auto else None,
                    sampling_rate_hz=record.sampling_rate_hz,
                    annotator=self.annotator,
                )
            )
            saved_phases.append(f"{phase}:{STATUS_TEXT[status]}")
        if saved_phases:
            item = self.file_list.item(self.current_row)
            if item is not None:
                item.setText(self._file_item_text(self.files[self.current_row]))
                self._style_item(item, self.files[self.current_row])
            self._update_status(
                f"已保存 {record.file_name}  {'  '.join(saved_phases)}  （gold 总数 {self.store.count()}）"
            )
        else:
            self._update_status("没有已设定状态的相，未写入。先用 G/U/X 或下拉框设定每相状态。")

    def save_and_next(self) -> None:
        self.save_current()
        self._navigate(+1)

    def _navigate(self, delta: int) -> None:
        if not self.files:
            return
        row = self.current_row + delta
        while 0 <= row < self.file_list.count() and self.file_list.item(row).isHidden():
            row += delta
        if 0 <= row < len(self.files):
            self.file_list.setCurrentRow(row)

    def _update_status(self, text: str) -> None:
        self.statusBar().showMessage(text)

    def closeEvent(self, event) -> None:
        if self.store is not None:
            self.store.flush()
        super().closeEvent(event)
