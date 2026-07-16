"""Windows 窗口材质：Win11 Mica / Win10 安全降级。

策略：
- Win11 22H2+：尝试 SYSTEMBACKDROP_TYPE = Mica + 沉浸式深色标题栏
- Win10 / 旧版 Win11：仅尝试沉浸式深色标题栏，窗口内容走纯色/伪毛玻璃 QSS
- 任意 API 失败静默回退，不影响标注主流程
"""

from __future__ import annotations

import ctypes
import sys
from ctypes import wintypes
from typing import Any


# DWM 窗口属性（部分仅 Win11 可用）
DWMWA_USE_IMMERSIVE_DARK_MODE = 20
DWMWA_SYSTEMBACKDROP_TYPE = 38
DWMSBT_AUTO = 0
DWMSBT_NONE = 1
DWMSBT_MAINWINDOW = 2  # Mica
DWMSBT_TRANSIENTWINDOW = 3  # Acrylic
DWMSBT_TABBEDWINDOW = 4


def _windows_build() -> int:
    if sys.platform != "win32":
        return 0
    try:
        return sys.getwindowsversion().build
    except Exception:  # noqa: BLE001
        return 0


def is_windows_11_or_newer() -> bool:
    """Windows 11 起始 build 约为 22000。"""
    return _windows_build() >= 22000


def supports_system_backdrop() -> bool:
    """DWMWA_SYSTEMBACKDROP_TYPE 自 Win11 22H2 (22621) 起较稳定。"""
    return _windows_build() >= 22621


def _hwnd_from_widget(widget: Any) -> int:
    handle = int(widget.winId())
    return handle


def _dwm_set_attr(hwnd: int, attr: int, value: int) -> bool:
    try:
        dwmapi = ctypes.windll.dwmapi
        val = wintypes.DWORD(value)
        hr = dwmapi.DwmSetWindowAttribute(
            wintypes.HWND(hwnd),
            wintypes.DWORD(attr),
            ctypes.byref(val),
            ctypes.sizeof(val),
        )
        return hr == 0
    except Exception:  # noqa: BLE001
        return False


def apply_immersive_dark_titlebar(widget: Any) -> bool:
    """Win10 1809+ / Win11：深色标题栏。失败返回 False。"""
    if sys.platform != "win32":
        return False
    try:
        hwnd = _hwnd_from_widget(widget)
    except Exception:  # noqa: BLE001
        return False
    # 部分 Win10 构建使用属性 19，优先 20，失败再试 19
    if _dwm_set_attr(hwnd, DWMWA_USE_IMMERSIVE_DARK_MODE, 1):
        return True
    return _dwm_set_attr(hwnd, 19, 1)


def apply_mica_if_available(widget: Any) -> str:
    """为顶层窗口应用材质。

    返回值：
    - ``mica``：已启用 Win11 Mica
    - ``dark_title``：仅深色标题栏（Win10 典型路径）
    - ``none``：未应用任何系统材质
    """
    if sys.platform != "win32":
        return "none"

    dark_ok = apply_immersive_dark_titlebar(widget)

    if supports_system_backdrop():
        try:
            hwnd = _hwnd_from_widget(widget)
        except Exception:  # noqa: BLE001
            return "dark_title" if dark_ok else "none"
        if _dwm_set_attr(hwnd, DWMWA_SYSTEMBACKDROP_TYPE, DWMSBT_MAINWINDOW):
            return "mica"

    return "dark_title" if dark_ok else "none"
