"""暗色 OLED + 局部毛玻璃设计系统（Windows 客户端）。

设计依据：UI UX Pro Max（Dark Mode OLED + Glassmorphism）与
参考对话中的「伪毛玻璃优先、真模糊仅限关键控件」策略。
"""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtWidgets import QApplication


@dataclass(frozen=True)
class ThemeTokens:
    """界面色板与几何常量。"""

    bg_app: str = "#0B0F14"
    bg_panel: str = "#121820"
    bg_elevated: str = "#1A222C"
    bg_input: str = "#151C24"
    bg_hover: str = "#243040"
    border: str = "rgba(255, 255, 255, 28)"
    border_strong: str = "rgba(255, 255, 255, 48)"
    text_primary: str = "#E8EEF4"
    text_secondary: str = "#9AA7B5"
    text_muted: str = "#6B7785"
    accent: str = "#3B82F6"
    cta: str = "#F59E0B"
    danger: str = "#FF5252"
    success: str = "#34D399"
    warning: str = "#FBBF24"
    glass_fill: str = "rgba(255, 255, 255, 18)"
    glass_fill_strong: str = "rgba(255, 255, 255, 28)"
    glass_highlight: str = "rgba(255, 255, 255, 55)"
    radius_sm: int = 8
    radius_md: int = 12
    radius_lg: int = 16
    radius_pill: int = 20


TOKENS = ThemeTokens()

# 列表状态色（略提亮，适配 OLED 深底对比度）
DONE_HEX = "#34D399"
PARTIAL_HEX = "#FBBF24"
PRIORITY_HEX = "#F87171"


def _global_qss(t: ThemeTokens = TOKENS, font_family: str = "Microsoft YaHei") -> str:
    # 字重统一 600：雅黑无 Medium 时 500 会落到 Regular 发细
    return f"""
QWidget {{
    background-color: {t.bg_app};
    color: {t.text_primary};
    font-family: "{font_family}";
    font-size: 13px;
    font-weight: 600;
}}
QMainWindow, QDialog {{
    background-color: {t.bg_app};
}}
QToolBar {{
    background-color: rgba(18, 24, 32, 230);
    border: none;
    border-bottom: 1px solid {t.border};
    spacing: 8px;
    padding: 6px 10px;
    font-weight: 600;
}}
QToolBar QToolButton {{
    background-color: transparent;
    color: {t.text_primary};
    border: 1px solid transparent;
    border-radius: {t.radius_sm}px;
    padding: 6px 12px;
    font-weight: 600;
}}
QToolBar QToolButton:hover {{
    background-color: {t.bg_hover};
    border-color: {t.border};
}}
QToolBar QToolButton:pressed {{
    background-color: {t.bg_elevated};
}}
QStatusBar {{
    background-color: rgba(18, 24, 32, 230);
    color: {t.text_secondary};
    border-top: 1px solid {t.border};
    padding: 2px 8px;
}}
QSplitter::handle {{
    background-color: {t.border};
    width: 1px;
}}
QLineEdit, QComboBox {{
    background-color: {t.bg_input};
    color: {t.text_primary};
    border: 1px solid {t.border};
    border-radius: {t.radius_sm}px;
    padding: 7px 10px;
    selection-background-color: {t.accent};
    font-family: "{font_family}";
    font-weight: 600;
}}
QLineEdit:focus, QComboBox:focus {{
    border: 1px solid {t.accent};
}}
QComboBox::drop-down {{
    border: none;
    width: 22px;
}}
QComboBox QAbstractItemView {{
    background-color: {t.bg_elevated};
    color: {t.text_primary};
    border: 1px solid {t.border_strong};
    selection-background-color: {t.accent};
}}
QListWidget {{
    background-color: rgba(18, 24, 32, 210);
    border: 1px solid {t.border};
    border-radius: {t.radius_md}px;
    padding: 4px;
    outline: none;
}}
QListWidget::item {{
    padding: 8px 10px;
    border-radius: {t.radius_sm}px;
    margin: 1px 2px;
    font-weight: 600;
}}
QListWidget::item:hover {{
    background-color: {t.bg_hover};
}}
QListWidget::item:selected {{
    background-color: rgba(59, 130, 246, 55);
    color: {t.text_primary};
}}
QGroupBox {{
    background-color: rgba(26, 34, 44, 200);
    border: 1px solid {t.border};
    border-radius: {t.radius_md}px;
    margin-top: 12px;
    padding: 12px 10px 10px 10px;
    font-weight: 600;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
    color: {t.text_secondary};
}}
QGroupBox[active="true"] {{
    border: 1px solid {t.danger};
    background-color: rgba(255, 82, 82, 18);
}}
QGroupBox[active="true"]::title {{
    color: {t.danger};
}}
QPushButton {{
    background-color: {t.bg_elevated};
    color: {t.text_primary};
    border: 1px solid {t.border};
    border-radius: {t.radius_sm}px;
    padding: 9px 14px;
    font-family: "{font_family}";
    font-weight: 600;
}}
QPushButton:hover {{
    background-color: {t.bg_hover};
    border-color: {t.border_strong};
}}
QPushButton:pressed {{
    background-color: #0F1620;
}}
QPushButton#primaryButton {{
    background-color: {t.accent};
    border-color: {t.accent};
    color: #FFFFFF;
}}
QPushButton#primaryButton:hover {{
    background-color: #2563EB;
}}
QPushButton#ctaButton {{
    background-color: {t.cta};
    border-color: {t.cta};
    color: #111827;
}}
QPushButton#ctaButton:hover {{
    background-color: #D97706;
}}
QCheckBox {{
    spacing: 8px;
    color: {t.text_primary};
}}
QCheckBox::indicator {{
    width: 16px;
    height: 16px;
    border-radius: 4px;
    border: 1px solid {t.border_strong};
    background-color: {t.bg_input};
}}
QCheckBox::indicator:checked {{
    background-color: {t.accent};
    border-color: {t.accent};
}}
QScrollBar:vertical {{
    background: transparent;
    width: 10px;
    margin: 4px 2px;
}}
QScrollBar::handle:vertical {{
    background: rgba(255, 255, 255, 40);
    border-radius: 4px;
    min-height: 24px;
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}
QLabel {{
    background: transparent;
    color: {t.text_primary};
    font-weight: 600;
}}
QMessageBox {{
    background-color: {t.bg_panel};
    font-family: "{font_family}";
}}
"""


def resolve_ui_font_family() -> str:
    """优先微软雅黑正体，避开 Light / Variable 细体族。"""
    preferred = [
        "Microsoft YaHei",
        "微软雅黑",
        "Microsoft YaHei UI",
        "微软雅黑 UI",
    ]
    families = set(QFontDatabase.families())
    for name in preferred:
        if name in families and "Light" not in name and "Thin" not in name:
            return name
    return "Microsoft YaHei"


def build_ui_font(point_size: int = 11) -> QFont:
    """构建界面正文字体：微软雅黑 + DemiBold，禁止 Thin/Light。

    微软雅黑通常仅有 Regular/Bold；请求 Medium(500) 常落到 400 而发细，
    故正文使用 DemiBold(600)，映射到 Bold 字形以保证笔画足够。
    """
    family = resolve_ui_font_family()
    font = QFont(family)
    font.setPointSize(point_size)
    font.setStyleHint(QFont.StyleHint.SansSerif)
    font.setHintingPreference(QFont.HintingPreference.PreferFullHinting)
    font.setWeight(QFont.Weight.DemiBold)
    return font


def apply_application_theme(app: QApplication) -> None:
    """应用全局微软雅黑字体与 QSS。"""
    family = resolve_ui_font_family()
    app.setFont(build_ui_font(11))
    app.setStyle("Fusion")
    app.setStyleSheet(_global_qss(font_family=family))


def set_active_property(widget, active: bool) -> None:
    """刷新 Qt 动态属性样式（用于活动相面板高亮）。"""
    widget.setProperty("active", active)
    style = widget.style()
    style.unpolish(widget)
    style.polish(widget)
    widget.update()
