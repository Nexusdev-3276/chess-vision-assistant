"""One-time calibration dialogs: board corners and visual validation."""

from __future__ import annotations

import logging

import chess
import cv2
import numpy as np
from PyQt6.QtCore import QPointF, QRectF, QSize, Qt, QTimer
from PyQt6.QtGui import (
    QColor,
    QFont,
    QGuiApplication,
    QImage,
    QPainter,
    QPen,
    QPixmap,
)
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from .geometry import (
    CANVAS_SIZE,
    ORIENTATION_BLACK,
    ORIENTATION_WHITE,
    BoardGeometry,
)

logger = logging.getLogger(__name__)


def bgr_to_qpixmap(image_bgr: np.ndarray) -> QPixmap:
    """Convert an OpenCV BGR image to a QPixmap with a safe copy."""
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Image vide.")
    if image_bgr.ndim == 2:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2RGB)
    else:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    rgb = np.ascontiguousarray(rgb)
    height, width = rgb.shape[:2]
    qimage = QImage(
        rgb.data,
        width,
        height,
        rgb.strides[0],
        QImage.Format.Format_RGB888,
    ).copy()
    return QPixmap.fromImage(qimage)


class BoardSelectionDialog(QDialog):
    """Full-screen screenshot on which the user clicks the four board corners."""

    TOP_BAR = 118
    IMAGE_MARGIN = 14

    def __init__(self, screenshot_bgr: np.ndarray, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Calibration — coins du plateau")
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Dialog
        )
        self.setModal(True)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(False)

        screen = QGuiApplication.primaryScreen()
        if screen is not None:
            self.setGeometry(screen.geometry())

        self._screenshot = screenshot_bgr
        self._pixmap = bgr_to_qpixmap(screenshot_bgr)
        self._image_size = (self._pixmap.width(), self._pixmap.height())
        self._points: list[tuple[float, float]] = []
        self._closing = False

    @property
    def points(self) -> tuple[tuple[float, float], ...]:
        return tuple(self._points)

    @property
    def screenshot(self) -> np.ndarray:
        return self._screenshot

    # ------------------------------------------------------------------ paint
    def _image_target(self) -> QRectF:
        area = QRectF(
            0,
            self.TOP_BAR,
            max(1, self.width()),
            max(1, self.height() - self.TOP_BAR),
        )
        area = area.adjusted(
            self.IMAGE_MARGIN,
            self.IMAGE_MARGIN,
            -self.IMAGE_MARGIN,
            -self.IMAGE_MARGIN,
        )
        img_w, img_h = self._image_size
        if img_w <= 0 or img_h <= 0:
            return area
        scale = min(area.width() / img_w, area.height() / img_h)
        w = img_w * scale
        h = img_h * scale
        return QRectF(
            area.left() + (area.width() - w) / 2,
            area.top() + (area.height() - h) / 2,
            w,
            h,
        )

    def _widget_to_image(self, pos: QPointF) -> tuple[float, float] | None:
        target = self._image_target()
        if not target.contains(pos):
            return None
        img_w, img_h = self._image_size
        if target.width() <= 0 or target.height() <= 0:
            return None
        x = (pos.x() - target.left()) / target.width() * img_w
        y = (pos.y() - target.top()) / target.height() * img_h
        return (
            max(0.0, min(img_w - 1.0, x)),
            max(0.0, min(img_h - 1.0, y)),
        )

    def _image_to_widget(self, point: tuple[float, float]) -> QPointF:
        target = self._image_target()
        img_w, img_h = self._image_size
        return QPointF(
            target.left() + point[0] / img_w * target.width(),
            target.top() + point[1] / img_h * target.height(),
        )

    def paintEvent(self, event) -> None:  # noqa: N802
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), QColor("#0E1116"))

        self._paint_top_bar(painter)

        target = self._image_target()
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        painter.drawPixmap(target, self._pixmap, QRectF(self._pixmap.rect()))

        if not self._points:
            self._paint_corner_hint(painter, target)
            painter.end()
            return

        widget_points = [self._image_to_widget(p) for p in self._points]
        pen = QPen(QColor(0, 230, 255, 230), 2.0)
        painter.setPen(pen)
        if len(widget_points) > 1:
            for a, b in zip(widget_points, widget_points[1:]):
                painter.drawLine(a, b)
        if len(widget_points) == 4:
            painter.drawLine(widget_points[-1], widget_points[0])

        marker_font = QFont("Segoe UI", 10)
        marker_font.setWeight(QFont.Weight.Bold)
        painter.setFont(marker_font)
        for index, point in enumerate(widget_points, start=1):
            painter.setPen(QPen(QColor(0, 0, 0, 160), 2))
            painter.setBrush(QColor(0, 230, 255, 235))
            painter.drawEllipse(point, 13, 13)
            painter.setPen(QColor(10, 15, 20))
            painter.drawText(
                QRectF(point.x() - 13, point.y() - 13, 26, 26),
                Qt.AlignmentFlag.AlignCenter,
                str(index),
            )
        painter.end()

    def _paint_top_bar(self, painter: QPainter) -> None:
        bar = QRectF(0, 0, self.width(), self.TOP_BAR)
        painter.fillRect(bar, QColor("#171C24"))
        painter.setPen(QPen(QColor("#2C3441"), 1))
        painter.drawLine(0, self.TOP_BAR, self.width(), self.TOP_BAR)

        title_font = QFont("Segoe UI", 13)
        title_font.setWeight(QFont.Weight.DemiBold)
        painter.setFont(title_font)
        painter.setPen(QColor("#F5F7FA"))
        painter.drawText(
            QRectF(22, 12, self.width() - 44, 28),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            "Calibration du plateau",
        )

        body_font = QFont("Segoe UI", 10)
        painter.setFont(body_font)
        painter.setPen(QColor("#C5CBD4"))
        text = (
            "Cliquez les 4 coins de la zone de jeu, dans cet ordre : "
            "haut-gauche → haut-droit → bas-droit → bas-gauche.\n"
            "Clic droit ou Retour arrière : annuler un point   •   Échap : annuler"
        )
        painter.drawText(
            QRectF(22, 43, self.width() - 44, 62),
            Qt.AlignmentFlag.AlignLeft
            | Qt.AlignmentFlag.AlignTop
            | Qt.TextFlag.TextWordWrap,
            text,
        )

    def _paint_corner_hint(self, painter: QPainter, target: QRectF) -> None:
        painter.setPen(QPen(QColor(255, 255, 255, 80), 1, Qt.PenStyle.DashLine))
        painter.drawLine(
            target.left(), target.top(), target.left() + 30, target.top()
        )
        painter.drawLine(
            target.left(), target.top(), target.left(), target.top() + 30
        )
        painter.drawLine(
            target.right(), target.bottom(), target.right() - 30, target.bottom()
        )
        painter.drawLine(
            target.right(), target.bottom(), target.right(), target.bottom() - 30
        )

    # ------------------------------------------------------------------ input
    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.RightButton:
            if self._points:
                self._points.pop()
                self.update()
            return
        if event.button() != Qt.MouseButton.LeftButton:
            return
        if self._closing or len(self._points) >= 4:
            return
        mapped = self._widget_to_image(event.position())
        if mapped is None:
            return
        self._points.append(mapped)
        self.update()
        if len(self._points) == 4:
            self._closing = True
            QTimer.singleShot(120, self.accept)

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if event.key() == Qt.Key.Key_Escape:
            self.reject()
            return
        if event.key() in (Qt.Key.Key_Backspace, Qt.Key.Key_Delete):
            if self._points:
                self._points.pop()
                self.update()
            return
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and len(self._points) == 4:
            self.accept()
            return
        super().keyPressEvent(event)


class BoardPreview(QWidget):
    """Square board preview that scales with the space available in the dialog."""

    MIN_SIDE = 220

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._pixmap: QPixmap | None = None
        self.setMinimumSize(self.MIN_SIDE, self.MIN_SIDE)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )

    def set_pixmap(self, pixmap: QPixmap) -> None:
        self._pixmap = pixmap
        self.update()

    def sizeHint(self) -> QSize:  # noqa: N802
        return QSize(480, 480)

    def paintEvent(self, event) -> None:  # noqa: N802
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        background = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#0E1116"))
        painter.drawRoundedRect(background, 6, 6)

        if self._pixmap is None or self._pixmap.isNull():
            return
        side = max(0, min(self.width(), self.height()) - 2)
        if side <= 0:
            return
        target = QRectF(
            (self.width() - side) / 2,
            (self.height() - side) / 2,
            side,
            side,
        )
        painter.drawPixmap(target, self._pixmap, QRectF(self._pixmap.rect()))


class CalibrationPreviewDialog(QDialog):
    """Shows the rectified board and lets the user choose the orientation."""

    def __init__(
        self,
        warped_board: np.ndarray,
        initial_orientation: str = ORIENTATION_WHITE,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Calibration — vérification")
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
        self.setModal(True)
        self._warped = warped_board

        screen = QGuiApplication.primaryScreen()
        available = screen.availableGeometry() if screen is not None else None
        available_height = available.height() if available is not None else 900
        available_width = available.width() if available is not None else 1280

        width = min(680, max(600, available_width - 80))
        height = min(840, max(520, available_height - 100))
        self.resize(width, height)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(12)

        title = QLabel("Vérifiez le cadrage")
        font = title.font()
        font.setPointSize(font.pointSize() + 5)
        font.setBold(True)
        title.setFont(font)
        layout.addWidget(title)

        info = QLabel(
            "Le plateau doit être en position initiale complète, sans animation "
            "ni fenêtre superposée. Les lignes cyan doivent suivre les cases."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        self.image_label = BoardPreview()
        layout.addWidget(self.image_label, 1)

        form = QFormLayout()
        form.setHorizontalSpacing(16)
        self.orientation_combo = QComboBox()
        self.orientation_combo.addItem(
            "Blancs en bas (je joue les Blancs)", ORIENTATION_WHITE
        )
        self.orientation_combo.addItem(
            "Noirs en bas (je joue les Noirs)", ORIENTATION_BLACK
        )
        index = self.orientation_combo.findData(initial_orientation)
        self.orientation_combo.setCurrentIndex(max(0, index))
        form.addRow("Orientation du plateau :", self.orientation_combo)
        layout.addLayout(form)

        hint = QLabel(
            "Astuce : si le cadrage ou l'orientation est faux, cliquez sur "
            "« Recommencer » et sélectionnez à nouveau les quatre coins."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#6B7280;")
        layout.addWidget(hint)

        self._retry_requested = False
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        self.cancel_button = QPushButton("Annuler")
        self.retry_button = QPushButton("Recommencer")
        self.save_button = QPushButton("Enregistrer la calibration")
        self.save_button.setDefault(True)
        self.save_button.setStyleSheet(
            "QPushButton { background:#166534; color:white; padding:8px 16px; "
            "border-radius:6px; font-weight:600; }"
            "QPushButton:hover { background:#15803D; }"
        )
        self.cancel_button.clicked.connect(self.reject)
        self.retry_button.clicked.connect(self._request_retry)
        self.save_button.clicked.connect(self.accept)
        buttons.addWidget(self.cancel_button)
        buttons.addWidget(self.retry_button)
        buttons.addWidget(self.save_button)
        layout.addLayout(buttons)

        self.orientation_combo.currentIndexChanged.connect(self._refresh_preview)
        self._refresh_preview()

    def orientation(self) -> str:
        data = self.orientation_combo.currentData()
        return str(data) if data in (ORIENTATION_WHITE, ORIENTATION_BLACK) else ORIENTATION_WHITE

    def _request_retry(self) -> None:
        self._retry_requested = True
        self.reject()

    @property
    def retry_requested(self) -> bool:
        return self._retry_requested

    def _refresh_preview(self) -> None:
        preview = self._warped.copy()
        overlay = preview.copy()
        for index in range(1, 8):
            position = index * (CANVAS_SIZE // 8)
            cv2.line(overlay, (position, 0), (position, CANVAS_SIZE), (255, 220, 0), 1)
            cv2.line(overlay, (0, position), (CANVAS_SIZE, position), (255, 220, 0), 1)
        cv2.addWeighted(overlay, 0.55, preview, 0.45, 0, preview)
        cv2.rectangle(preview, (0, 0), (CANVAS_SIZE - 1, CANVAS_SIZE - 1), (0, 220, 255), 2)

        top_left_square = BoardGeometry.cell_to_square(0, 0, self.orientation())
        label = chess.square_name(top_left_square)
        cv2.rectangle(preview, (8, 8), (92, 38), (15, 15, 15), -1)
        cv2.putText(
            preview,
            f"Coin = {label}",
            (14, 31),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        pixmap = bgr_to_qpixmap(preview)
        self.image_label.set_pixmap(pixmap)


def ask_calibration_warning(parent: QWidget | None = None) -> bool:
    """Minimal confirmation helper kept here to keep the calibration flow together."""
    from PyQt6.QtWidgets import QMessageBox

    result = QMessageBox.warning(
        parent,
        "Calibration",
        "La calibration apprend l'apparence des pièces. Elle doit être faite "
        "avec le plateau en position initiale complète, sans animation, et "
        "sur l'écran principal.\n\n"
        "La fenêtre de l'assistant va se masquer, puis vous cliquerez les "
        "quatre coins du plateau.\n\n"
        "Continuer ?",
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
        QMessageBox.StandardButton.Yes,
    )
    return result == QMessageBox.StandardButton.Yes
