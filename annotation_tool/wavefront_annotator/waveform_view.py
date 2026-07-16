"""三相波形视图：单窗切换显示 + 卡尺 + 框线区间 + 导数辅助。

显示策略：
- 同一时刻仅渲染当前活动相，通过 A/B/C 切换按钮或快捷键 1/2/3 换相；
- 各相标注状态（卡尺、区间、自动标签）独立保留，切换不丢失；
- curve.setDownsampling(auto=True, method="peak") + setClipToView(True)
  保证缩放/平移时保留波头尖峰且帧率稳定。
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
        self.plot.setLabel("bottom", "采样点（原始坐标）")
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
        # 用独立标志记录启用态：父 PlotItem 隐藏时 Qt isVisible() 会变 False，不能据此丢状态
        self._gold_enabled = False
        self._region_enabled = False
        self._auto_enabled = False
        self._deriv_enabled = False

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
            self._auto_enabled = False
            self.auto_line.setVisible(False)
        else:
            self._auto_enabled = True
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
        return float(self.gold_line.value()) if self._gold_enabled else None

    def set_gold(self, index: float | None) -> None:
        if index is None:
            self._gold_enabled = False
            self.gold_line.setVisible(False)
        else:
            # 先设可见再设位置，保证 InfLineLabel 文本随位置刷新
            self._gold_enabled = True
            self.gold_line.setVisible(True)
            self.gold_line.setPos(float(index))
            if self.gold_line.label is not None:
                self.gold_line.label.valueChanged()

    def region_bounds(self) -> tuple[float, float] | None:
        if not self._region_enabled:
            return None
        low, high = self.region.getRegion()
        return float(low), float(high)

    def set_region(self, bounds: tuple[float, float] | None) -> None:
        if bounds is None:
            self._region_enabled = False
            self.region.setVisible(False)
        else:
            self._region_enabled = True
            self.region.setRegion(bounds)
            self.region.setVisible(True)

    def toggle_region_around(self, center: float, half_width: float = 256.0) -> None:
        if self._region_enabled:
            self.set_region(None)
        else:
            low = max(0.0, center - half_width)
            high = min(float(self.n_samples - 1), center + half_width)
            self.set_region((low, high))

    def set_derivative_visible(self, visible: bool) -> None:
        self._deriv_enabled = bool(visible)
        self.deriv_curve.setVisible(visible)

    def restore_overlay_visibility(self) -> None:
        """父图重新显示后，按启用标志恢复覆盖层可见性。"""
        self.gold_line.setVisible(self._gold_enabled)
        self.region.setVisible(self._region_enabled)
        self.auto_line.setVisible(self._auto_enabled)
        self.deriv_curve.setVisible(self._deriv_enabled)


class WaveformView(QWidget):
    """单窗切换的三相波形视图：同一时刻只显示活动相。"""

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
            phase_plot.gold_line.sigPositionChanged.connect(
                lambda line=phase_plot.gold_line, p=phase: self.goldChanged.emit(p, float(line.value()))
            )

        self.active_phase = "A"
        self._layout_widget.scene().sigMouseClicked.connect(self._on_mouse_clicked)
        self.set_active_phase("A", emit_focus=False)

    def set_active_phase(self, phase: str, emit_focus: bool = True) -> None:
        """切换到指定相：隐藏其余相图，仅展开当前相。"""
        if phase not in self.phase_plots:
            raise ValueError(f"未知相别: {phase}")
        self.active_phase = phase
        for name, phase_plot in self.phase_plots.items():
            visible = name == phase
            phase_plot.plot.setVisible(visible)
            # 隐藏行不占布局高度，保证活动相铺满主图区
            try:
                phase_plot.plot.setMaximumHeight(16777215 if visible else 0)
                phase_plot.plot.setMinimumHeight(0 if not visible else 120)
            except Exception:  # noqa: BLE001 - 兼容不同 pyqtgraph 版本
                pass
            if visible:
                phase_plot.restore_overlay_visibility()
        if emit_focus:
            self.phaseFocused.emit(phase)

    def _on_mouse_clicked(self, event) -> None:
        phase_plot = self.phase_plots[self.active_phase]
        if not phase_plot.plot.sceneBoundingRect().contains(event.scenePos()):
            return
        self.phaseFocused.emit(self.active_phase)
        if event.double():
            vb = phase_plot.plot.vb
            x = float(vb.mapSceneToView(event.scenePos()).x())
            x = max(0.0, min(x, float(max(phase_plot.n_samples - 1, 0))))
            phase_plot.set_gold(x)
            self.goldChanged.emit(self.active_phase, x)
            event.accept()

    def load_record(self, signals: np.ndarray) -> None:
        for i, phase in enumerate(PHASES):
            self.phase_plots[phase].set_data(signals[i])
        self.set_active_phase(self.active_phase, emit_focus=False)

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
        phase_plot.plot.setXRange(position - half_width, position + half_width)

    def reset_view(self) -> None:
        self.phase_plots[self.active_phase].plot.autoRange()
