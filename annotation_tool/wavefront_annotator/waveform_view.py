"""三相波形视图：pyqtgraph 三联图 + 卡尺 + 框线区间 + 导数辅助。

性能策略：
- curve.setDownsampling(auto=True, method="peak")：视口外自动峰值降采样，
  保证缩放/平移时不丢波头尖峰且帧率稳定；
- setClipToView(True)：仅渲染视口内数据；
- 三相 X 轴联动，双击任一相即落卡尺。
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QWidget, QVBoxLayout

PHASES = ("A", "B", "C")
PHASE_COLORS = {"A": "#ffd54f", "B": "#4fc3f7", "C": "#81c784"}
GOLD_COLOR = "#ff5252"
AUTO_COLOR = "#b39ddb"
CANDIDATE_COLOR = "#607d8b"
REGION_BRUSH = (255, 82, 82, 30)

pg.setConfigOptions(antialias=False, useOpenGL=False, background="#101418", foreground="#d0d6db")


class PhasePlot:
    """单相波形图：波形曲线、导数曲线、自动标签参考线、gold 卡尺、框线区间。"""

    def __init__(self, layout: pg.GraphicsLayoutWidget, phase: str, row: int) -> None:
        self.phase = phase
        self.plot: pg.PlotItem = layout.addPlot(row=row, col=0)
        self.plot.setLabel("left", f"{phase} 相")
        self.plot.showGrid(x=True, y=True, alpha=0.15)
        self.plot.setClipToView(True)
        self.plot.setDownsampling(auto=True, mode="peak")

        self.curve = self.plot.plot(pen=pg.mkPen(PHASE_COLORS[phase], width=1))
        self.deriv_curve = self.plot.plot(
            pen=pg.mkPen("#ef9a9a", width=1, style=Qt.PenStyle.DashLine)
        )
        self.deriv_curve.setVisible(False)

        # 自动伪标签参考线（不可拖动，紫色点划线）
        self.auto_line = pg.InfiniteLine(
            angle=90,
            movable=False,
            pen=pg.mkPen(AUTO_COLOR, width=1, style=Qt.PenStyle.DashDotLine),
            label="auto",
            labelOpts={"color": AUTO_COLOR, "position": 0.92, "fill": (0, 0, 0, 120)},
        )
        self.auto_line.setVisible(False)
        self.plot.addItem(self.auto_line)

        # 各检测器候选（灰色细虚线）
        self.candidate_lines: list[pg.InfiniteLine] = []

        # gold 卡尺（红色，可拖动）
        self.gold_line = pg.InfiniteLine(
            angle=90,
            movable=True,
            pen=pg.mkPen(GOLD_COLOR, width=2),
            hoverPen=pg.mkPen("#ff8a80", width=3),
            label="gold {value:.0f}",
            labelOpts={"color": GOLD_COLOR, "position": 0.05, "fill": (0, 0, 0, 150)},
        )
        self.gold_line.setVisible(False)
        self.plot.addItem(self.gold_line)

        # 框线区间（可选，用于圈定搜索窗/存疑范围）
        self.region = pg.LinearRegionItem(brush=REGION_BRUSH, movable=True)
        self.region.setZValue(-5)
        self.region.setVisible(False)
        self.plot.addItem(self.region)

        self.n_samples = 0

    def set_data(self, samples: np.ndarray) -> None:
        self.n_samples = int(samples.shape[0])
        x = np.arange(self.n_samples, dtype=np.float64)
        self.curve.setData(x, samples)
        deriv = np.abs(np.diff(samples, prepend=samples[0]))
        peak = float(deriv.max())
        span = float(samples.max() - samples.min()) or 1.0
        scale = (0.5 * span / peak) if peak > 0 else 0.0
        # 导数幅值缩放到波形量级下方，便于同图叠加观察突变点
        self.deriv_curve.setData(x, deriv * scale + float(samples.min()))
        self.plot.setLimits(xMin=-64, xMax=self.n_samples + 64)
        self.plot.autoRange()

    def set_auto_label(self, index: float | None) -> None:
        if index is None:
            self.auto_line.setVisible(False)
        else:
            self.auto_line.setPos(index)
            self.auto_line.setVisible(True)

    def set_candidates(self, indices: dict[str, float]) -> None:
        for line in self.candidate_lines:
            self.plot.removeItem(line)
        self.candidate_lines.clear()
        for name, index in indices.items():
            line = pg.InfiniteLine(
                angle=90,
                movable=False,
                pen=pg.mkPen(CANDIDATE_COLOR, width=1, style=Qt.PenStyle.DotLine),
                label=name.replace("_", "\n"),
                labelOpts={"color": CANDIDATE_COLOR, "position": 0.75, "fill": (0, 0, 0, 100)},
            )
            line.setPos(index)
            self.plot.addItem(line)
            self.candidate_lines.append(line)

    def gold_position(self) -> float | None:
        return float(self.gold_line.value()) if self.gold_line.isVisible() else None

    def set_gold(self, index: float | None) -> None:
        if index is None:
            self.gold_line.setVisible(False)
        else:
            # 先设可见再设位置，保证 InfLineLabel 文本随位置刷新
            self.gold_line.setVisible(True)
            self.gold_line.setPos(float(index))
            if self.gold_line.label is not None:
                self.gold_line.label.valueChanged()

    def region_bounds(self) -> tuple[float, float] | None:
        if not self.region.isVisible():
            return None
        low, high = self.region.getRegion()
        return float(low), float(high)

    def set_region(self, bounds: tuple[float, float] | None) -> None:
        if bounds is None:
            self.region.setVisible(False)
        else:
            self.region.setRegion(bounds)
            self.region.setVisible(True)

    def toggle_region_around(self, center: float, half_width: float = 256.0) -> None:
        if self.region.isVisible():
            self.region.setVisible(False)
        else:
            low = max(0.0, center - half_width)
            high = min(float(self.n_samples - 1), center + half_width)
            self.region.setRegion((low, high))
            self.region.setVisible(True)

    def set_derivative_visible(self, visible: bool) -> None:
        self.deriv_curve.setVisible(visible)


class WaveformView(QWidget):
    """三相联动波形视图。"""

    goldChanged = Signal(str, float)  # (phase, index)：卡尺被拖动或双击落点
    phaseFocused = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._layout_widget = pg.GraphicsLayoutWidget()
        container = QVBoxLayout(self)
        container.setContentsMargins(0, 0, 0, 0)
        container.addWidget(self._layout_widget)

        self.phase_plots: dict[str, PhasePlot] = {}
        for row, phase in enumerate(PHASES):
            phase_plot = PhasePlot(self._layout_widget, phase, row)
            self.phase_plots[phase] = phase_plot
            if row > 0:
                phase_plot.plot.setXLink(self.phase_plots[PHASES[0]].plot)
            phase_plot.gold_line.sigPositionChanged.connect(
                lambda line=phase_plot.gold_line, p=phase: self.goldChanged.emit(p, float(line.value()))
            )
        self.phase_plots[PHASES[-1]].plot.setLabel("bottom", "采样点（原始坐标）")

        self.active_phase = "A"
        self._layout_widget.scene().sigMouseClicked.connect(self._on_mouse_clicked)

    def _on_mouse_clicked(self, event) -> None:
        for phase, phase_plot in self.phase_plots.items():
            vb = phase_plot.plot.vb
            if phase_plot.plot.sceneBoundingRect().contains(event.scenePos()):
                self.active_phase = phase
                self.phaseFocused.emit(phase)
                if event.double():
                    x = float(vb.mapSceneToView(event.scenePos()).x())
                    x = max(0.0, min(x, float(phase_plot.n_samples - 1)))
                    phase_plot.set_gold(x)
                    self.goldChanged.emit(phase, x)
                    event.accept()
                break

    def load_record(self, signals: np.ndarray) -> None:
        for i, phase in enumerate(PHASES):
            self.phase_plots[phase].set_data(signals[i])

    def set_derivative_visible(self, visible: bool) -> None:
        for phase_plot in self.phase_plots.values():
            phase_plot.set_derivative_visible(visible)

    def nudge_active_gold(self, delta: float) -> None:
        phase_plot = self.phase_plots[self.active_phase]
        position = phase_plot.gold_position()
        if position is None:
            return
        new_pos = max(0.0, min(position + delta, float(phase_plot.n_samples - 1)))
        phase_plot.set_gold(new_pos)
        self.goldChanged.emit(self.active_phase, new_pos)

    def zoom_active_to_gold(self, half_width: float = 512.0) -> None:
        phase_plot = self.phase_plots[self.active_phase]
        position = phase_plot.gold_position()
        if position is None:
            return
        self.phase_plots[PHASES[0]].plot.setXRange(position - half_width, position + half_width)

    def reset_view(self) -> None:
        for phase_plot in self.phase_plots.values():
            phase_plot.plot.autoRange()
