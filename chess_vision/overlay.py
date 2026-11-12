"""Click-through transparent Qt overlay that draws the engine's best move."""

from __future__ import annotations

import ctypes
import logging
import sys
from ctypes import wintypes

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import (
    QColor,
    QFont,
    QFontMetricsF,
    QGuiApplication,
    QPainter,
    QPen,
    QPolygonF,
)
from PyQt6.QtWidgets import QWidget

logger = logging.getLogger(__name__)

WDA_EXCLUDEFROMCAPTURE = 0x00000011
GWL_EXSTYLE = -20
WS_EX_LAYERED = 0x00080000
WS_EX_TRANSPARENT = 0x00000020
WS_EX_NOACTIVATE = 0x08000000
WS_EX_TOOLWINDOW = 0x00000080


class SuggestionOverlay(QWidget):
    """Full-screen, borderless, always-on-top arrow drawing surface.

    The window is transparent for mouse and keyboard input. On Windows it is
    also excluded from screen captures so the recognition loop never sees its
    own arrow. If that exclusion is unavailable, the application temporarily
    hides the overlay while taking a screenshot.
    """

    def __init__(self) -> None:
        super().__init__(None)
        flags = (
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowTransparentForInput
            | Qt.WindowType.WindowDoesNotAcceptFocus
            | Qt.WindowType.NoDropShadowWindowHint
        )
        self.setWindowFlags(flags)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)
        self.setWindowTitle("Chess Vision — flèche")
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        self._capture_size: tuple[int, int] = (1920, 1080)
        self._start: tuple[float, float] | None = None
        self._end: tuple[float, float] | None = None
        self._label = ""
        self._arrow_color = QColor("#FF8A00")
        self._arrow_alpha = 205
        self._show_label = True
        self.capture_excluded = False
        self._capture_affinity_checked = False

    # ------------------------------------------------------------------ setup
    def set_capture_size(self, width: int, height: int) -> None:
        if width > 0 and height > 0:
            self._capture_size = (int(width), int(height))

    def set_visual_style(self, color: str, opacity: int, show_label: bool) -> None:
        self._arrow_color = QColor(color)
        if not self._arrow_color.isValid():
            self._arrow_color = QColor("#FF8A00")
        self._arrow_alpha = max(40, min(255, int(opacity)))
        self._show_label = bool(show_label)

    # ------------------------------------------------------------------ public
    def show_move(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
        label: str = "",
    ) -> None:
        self._start = (float(start[0]), float(start[1]))
        self._end = (float(end[0]), float(end[1]))
        self._label = label
        self._place_on_primary_screen()
        self.show()
        self.raise_()
        self._configure_native_window()
        self.update()

    def clear(self) -> None:
        self._start = None
        self._end = None
        self._label = ""
        self.hide()

    def hide_for_capture(self) -> None:
        if self.isVisible():
            self.hide()

    def restore_after_capture(self) -> None:
        if self._start is not None and self._end is not None:
            self.show()
            self.raise_()
            self.update()

    # ----------------------------------------------------------------- helpers
    def _place_on_primary_screen(self) -> None:
        screen = QGuiApplication.primaryScreen()
        if screen is None:
            return
        geometry = screen.geometry()
        if self.geometry() != geometry:
            self.setGeometry(geometry)

    def _configure_native_window(self) -> None:
        if sys.platform != "win32":
            self.capture_excluded = False
            return
        try:
            hwnd = wintypes.HWND(int(self.winId()))
            user32 = ctypes.windll.user32
            user32.GetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int]
            user32.GetWindowLongW.restype = ctypes.c_long
            user32.SetWindowLongW.argtypes = [
                wintypes.HWND,
                ctypes.c_int,
                ctypes.c_long,
            ]
            user32.SetWindowLongW.restype = ctypes.c_long
            style = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
            style |= WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW
            user32.SetWindowLongW(hwnd, GWL_EXSTYLE, style)

            if not self._capture_affinity_checked:
                user32.SetWindowDisplayAffinity.argtypes = [wintypes.HWND, wintypes.DWORD]
                user32.SetWindowDisplayAffinity.restype = wintypes.BOOL
                ok = bool(user32.SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE))
                self.capture_excluded = ok
                self._capture_affinity_checked = True
                if not ok:
                    logger.warning(
                        "Exclusion de l'overlay de la capture indisponible; "
                        "l'overlay sera masqué brièvement pour lire l'écran."
                    )
        except Exception:
            logger.warning("Configuration native de l'overlay impossible", exc_info=True)
            self.capture_excluded = False

    def _to_local(self, point: tuple[float, float]) -> QPointF:
        cw, ch = self._capture_size
        if cw <= 0 or ch <= 0:
            return QPointF(0, 0)
        return QPointF(
            point[0] * self.width() / cw,
            point[1] * self.height() / ch,
        )

    # ------------------------------------------------------------------ paint
    def paintEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        del event
        if self._start is None or self._end is None:
            return

        start = self._to_local(self._start)
        end = self._to_local(self._end)
        dx = end.x() - start.x()
        dy = end.y() - start.y()
        length = (dx * dx + dy * dy) ** 0.5
        if length < 2:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)

        screen_min = min(self.width(), self.height())
        line_width = max(10.0, min(26.0, screen_min * 0.012))
        head_width = line_width * 2.55
        head_length = line_width * 3.05
        ux, uy = dx / length, dy / length
        px, py = -uy, ux

        tip = end
        base = QPointF(end.x() - ux * head_length, end.y() - uy * head_length)
        left = QPointF(base.x() + px * head_width / 2, base.y() + py * head_width / 2)
        right = QPointF(base.x() - px * head_width / 2, base.y() - py * head_width / 2)

        color = QColor(self._arrow_color)
        color.setAlpha(self._arrow_alpha)
        halo = QColor(10, 10, 10, max(65, self._arrow_alpha // 3))

        # Dark halo improves contrast on very light and very dark themes.
        halo_pen = QPen(halo, line_width + 5)
        halo_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        halo_pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(halo_pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawLine(start, base)

        line_pen = QPen(color, line_width)
        line_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        line_pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(line_pen)
        painter.drawLine(start, base)

        head = QPolygonF([tip, left, right])
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(color)
        painter.drawPolygon(head)

        if self._show_label and self._label:
            midpoint = QPointF((start.x() + end.x()) / 2, (start.y() + end.y()) / 2)
            offset = line_width * 2.0 + 6
            label_pos = QPointF(midpoint.x() + px * offset, midpoint.y() + py * offset)
            self._draw_label(painter, label_pos)
        painter.end()

    def _draw_label(self, painter: QPainter, center: QPointF) -> None:
        font = QFont("Segoe UI", 10)
        font.setWeight(QFont.Weight.DemiBold)
        painter.setFont(font)
        metrics = QFontMetricsF(font)
        text_width = metrics.horizontalAdvance(self._label)
        text_height = metrics.height()
        pad_x, pad_y = 10.0, 5.0
        width = text_width + 2 * pad_x
        height = text_height + 2 * pad_y
        left = center.x() - width / 2
        top = center.y() - height / 2
        left = max(8.0, min(self.width() - width - 8.0, left))
        top = max(8.0, min(self.height() - height - 8.0, top))
        rect = QRectF(left, top, width, height)

        painter.setPen(QPen(QColor(255, 255, 255, 38), 1))
        painter.setBrush(QColor(15, 15, 15, 190))
        painter.drawRoundedRect(rect, 7, 7)
        painter.setPen(QColor(255, 255, 255, 240))
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, self._label)
