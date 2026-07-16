"""伪毛玻璃分段控件：外层玻璃底座 + 白色滑动选中块 + 透明点击区。

对齐参考对话的效果优先级（Windows 通用，不依赖 Win11 真模糊）：
1. 内层白色滑块
2. 外层浅灰半透明底
3. 圆角与描边
4. 多层弱阴影
5. 真背景模糊省略（纯色深色底上视觉贡献极小，且利于 Win10 性能）
"""

from __future__ import annotations

from PySide6.QtCore import (
    Property,
    QEasingCurve,
    QPropertyAnimation,
    QRectF,
    Qt,
    Signal,
)
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

from .theme import TOKENS


class GlassSegmentedControl(QWidget):
    """水平分段切换器（如 A/B/C 相）。"""

    currentChanged = Signal(str)

    def __init__(
        self,
        labels: list[str],
        values: list[str] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        if not labels:
            raise ValueError("labels 不能为空")
        self._labels = list(labels)
        self._values = list(values) if values is not None else list(labels)
        if len(self._labels) != len(self._values):
            raise ValueError("labels 与 values 长度必须一致")

        self._index = 0
        self._accent = QColor(TOKENS.accent)
        self._thumb_progress = 0.0  # 0 .. n-1，驱动滑块水平位置
        self._hover_index = -1

        self.setMinimumHeight(44)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self._anim = QPropertyAnimation(self, b"thumbProgress", self)
        self._anim.setDuration(220)
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)

    # ----- 公共 API -----

    def count(self) -> int:
        return len(self._values)

    def currentValue(self) -> str:
        return self._values[self._index]

    def currentIndex(self) -> int:
        return self._index

    def setCurrentValue(self, value: str, *, animate: bool = True) -> None:
        try:
            index = self._values.index(value)
        except ValueError:
            return
        self.setCurrentIndex(index, animate=animate)

    def setCurrentIndex(self, index: int, *, animate: bool = True) -> None:
        if index < 0 or index >= self.count():
            return
        if index == self._index:
            self._set_thumb_progress(float(index))
            return
        self._index = index
        target = float(index)
        if animate and self.isVisible():
            self._anim.stop()
            self._anim.setStartValue(self._thumb_progress)
            self._anim.setEndValue(target)
            self._anim.start()
        else:
            self._set_thumb_progress(target)
        self.currentChanged.emit(self._values[index])
        self.update()

    def setAccentColor(self, color: QColor | str) -> None:
        self._accent = QColor(color)
        self.update()

    # ----- 动画属性 -----

    def get_thumb_progress(self) -> float:
        return self._thumb_progress

    def _set_thumb_progress(self, value: float) -> None:
        self._thumb_progress = float(value)
        self.update()

    thumbProgress = Property(float, get_thumb_progress, _set_thumb_progress)

    # ----- 几何 -----

    def _segment_rect(self, index: int) -> QRectF:
        margin = 4.0
        inner = QRectF(self.rect()).adjusted(margin, margin, -margin, -margin)
        width = inner.width() / self.count()
        return QRectF(inner.left() + index * width, inner.top(), width, inner.height())

    def _thumb_rect(self) -> QRectF:
        # 在相邻分段之间插值
        i0 = int(self._thumb_progress)
        i1 = min(i0 + 1, self.count() - 1)
        t = self._thumb_progress - i0
        r0 = self._segment_rect(i0)
        r1 = self._segment_rect(i1)
        x = r0.left() + (r1.left() - r0.left()) * t
        return QRectF(x, r0.top(), r0.width(), r0.height()).adjusted(2, 2, -2, -2)

    def _index_at(self, x: float, y: float) -> int:
        if not self.rect().contains(int(x), int(y)):
            return -1
        for i in range(self.count()):
            if self._segment_rect(i).contains(x, y):
                return i
        return -1

    # ----- 事件 -----

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            index = self._index_at(event.position().x(), event.position().y())
            if index >= 0:
                self.setCurrentIndex(index)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        index = self._index_at(event.position().x(), event.position().y())
        if index != self._hover_index:
            self._hover_index = index
            self.update()
        super().mouseMoveEvent(event)

    def leaveEvent(self, event) -> None:
        self._hover_index = -1
        self.update()
        super().leaveEvent(event)

    def keyPressEvent(self, event) -> None:
        if event.key() in (Qt.Key.Key_Left, Qt.Key.Key_Up):
            self.setCurrentIndex(max(0, self._index - 1))
            event.accept()
            return
        if event.key() in (Qt.Key.Key_Right, Qt.Key.Key_Down):
            self.setCurrentIndex(min(self.count() - 1, self._index + 1))
            event.accept()
            return
        super().keyPressEvent(event)

    # ----- 绘制 -----

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        track = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        radius = float(TOKENS.radius_pill)

        # 外层玻璃底座（伪毛玻璃：半透明渐变 + 描边 + 弱阴影）
        shadow = QPainterPath()
        shadow.addRoundedRect(track.adjusted(1, 2, -1, 1), radius, radius)
        painter.fillPath(shadow, QColor(0, 0, 0, 55))

        track_path = QPainterPath()
        track_path.addRoundedRect(track, radius, radius)
        painter.fillPath(track_path, QColor(255, 255, 255, 22))
        # 顶部高光条
        highlight = QRectF(track.left() + 8, track.top() + 1, track.width() - 16, 1.5)
        painter.fillRect(highlight, QColor(255, 255, 255, 40))
        painter.setPen(QPen(QColor(255, 255, 255, 48), 1.0))
        painter.drawPath(track_path)

        # 白色滑动选中块
        thumb = self._thumb_rect()
        thumb_path = QPainterPath()
        thumb_radius = max(8.0, radius - 6.0)
        thumb_path.addRoundedRect(thumb, thumb_radius, thumb_radius)
        painter.fillPath(thumb_path, QColor(255, 255, 255, 235))
        # 相色细描边，增强选中态辨识
        accent = QColor(self._accent)
        accent.setAlpha(180)
        painter.setPen(QPen(accent, 1.5))
        painter.drawPath(thumb_path)

        # 文字层：雅黑加粗，避免深色底上发细
        font = QFont(self.font())
        if not font.family() or "YaHei" not in font.family() and "雅黑" not in font.family():
            font.setFamily("Microsoft YaHei")
        font.setWeight(QFont.Weight.Bold)
        font.setPointSize(max(11, font.pointSize()))
        painter.setFont(font)
        for i, label in enumerate(self._labels):
            rect = self._segment_rect(i)
            selected = abs(self._thumb_progress - i) < 0.35
            if selected:
                color = QColor(17, 24, 39)
            elif i == self._hover_index:
                color = QColor(TOKENS.text_primary)
            else:
                color = QColor(TOKENS.text_secondary)
            painter.setPen(color)
            painter.drawText(rect, int(Qt.AlignmentFlag.AlignCenter), label)

        # 键盘焦点环
        if self.hasFocus():
            focus = track.adjusted(2, 2, -2, -2)
            pen = QPen(QColor(TOKENS.accent), 1.5, Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.drawRoundedRect(focus, radius - 2, radius - 2)
