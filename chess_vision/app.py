"""Main control window and application orchestration."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import chess
import cv2
import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QAction, QColor, QFont, QIcon, QPainter, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QSystemTrayIcon,
    QVBoxLayout,
    QWidget,
)

from .calibration import (
    BoardSelectionDialog,
    CalibrationPreviewDialog,
    ask_calibration_warning,
)
from .config import (
    APP_NAME,
    CALIBRATION_PATH,
    PREVIEW_PATH,
    TEMPLATES_PATH,
    AppConfig,
    ensure_data_dir,
    find_stockfish,
)
from .engine import EngineWorker, Suggestion
from .geometry import ORIENTATION_BLACK, ORIENTATION_WHITE, BoardGeometry
from .overlay import SuggestionOverlay
from .recognition import RecognitionResult, TemplateLibrary
from .screen import ScreenCapture
from .settings import SettingsDialog
from .tracker import GameTracker

logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.setMinimumWidth(660)
        self.resize(760, 560)

        self.config = AppConfig.load()
        self.capture: ScreenCapture | None = None
        self.library: TemplateLibrary | None = None
        self.geometry: BoardGeometry | None = None
        self.overlay = SuggestionOverlay()
        self.tracker = GameTracker()
        self.engine_worker: EngineWorker | None = None
        self._resolved_engine = ""
        self._engine_failure_handled = False

        self.timer = QTimer(self)
        self.timer.setTimerType(Qt.TimerType.CoarseTimer)
        self.timer.timeout.connect(self._poll)

        self.running = False
        self._polling = False
        self._quitting = False
        self._tray_hint_shown = False
        self._tray_icon: QSystemTrayIcon | None = None
        self._tray_menu: QMenu | None = None
        self._tray_show_action: QAction | None = None
        self._tray_toggle_action: QAction | None = None

        self._last_stable_placement: str | None = None
        self._candidate_placement: str | None = None
        self._candidate_count = 0
        self._recognition_failures = 0
        self._board_change_pending = False
        self._pending_fen: str | None = None
        self._last_requested_fen: str | None = None
        self._last_suggestion: Suggestion | None = None
        self._frame_size: tuple[int, int] = (0, 0)
        self._calibration_loaded = False
        self._calibration_error = ""
        self._suppress_auto_hide = False

        self._build_ui()
        self._set_app_icon()
        self._init_capture()
        self._load_calibration()
        self._resolve_engine()
        self._setup_tray()
        self._update_visual_style()
        self._refresh_detail_label()
        self._update_run_controls()
        self._update_turn_label()

        QTimer.singleShot(350, self._startup)

    # ================================================================== UI setup
    def _build_ui(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget#root { background: #0F1115; }
            QWidget { color: #E5E7EB; font-family: 'Segoe UI'; font-size: 10pt; }
            QLabel#title { font-size: 20pt; font-weight: 700; color: #F8FAFC; }
            QLabel#subtitle { color: #8B949E; font-size: 9pt; }
            QLabel#turn { color: #AEB6C2; font-size: 9pt; font-weight: 600; }
            QLabel#move { color: #FFB347; font-size: 16pt; font-weight: 700; }
            QLabel#status { color: #9CA3AF; }
            QLabel#detail { color: #6B7280; font-size: 8.5pt; }
            QFrame#card { background: #161B22; border: 1px solid #28303B; border-radius: 10px; }
            QPushButton {
                background: #1C2430; color: #E5E7EB; border: 1px solid #33404F;
                border-radius: 6px; padding: 7px 12px; font-weight: 600;
            }
            QPushButton:hover { background: #263140; border-color: #475569; }
            QPushButton:disabled { color: #64748B; background: #151A21; border-color: #232B35; }
            QPushButton#primary {
                background: #166534; border-color: #22C55E; color: #F0FDF4;
                padding: 10px 16px; font-size: 10.5pt;
            }
            QPushButton#primary:hover { background: #15803D; }
            """
        )

        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(20, 18, 20, 16)
        outer.setSpacing(12)

        title = QLabel(APP_NAME)
        title.setObjectName("title")
        subtitle = QLabel(
            "Lecture visuelle du plateau, analyse Stockfish et flèche indicative. "
            "Aucun contrôle de la souris ou du clavier."
        )
        subtitle.setObjectName("subtitle")
        subtitle.setWordWrap(True)
        outer.addWidget(title)
        outer.addWidget(subtitle)

        card = QFrame()
        card.setObjectName("card")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(16, 14, 16, 14)
        card_layout.setSpacing(6)
        self.turn_label = QLabel("Trait : —")
        self.turn_label.setObjectName("turn")
        self.move_label = QLabel("Aucune suggestion")
        self.move_label.setObjectName("move")
        self.move_label.setWordWrap(True)
        self.status_label = QLabel("Prêt.")
        self.status_label.setObjectName("status")
        self.status_label.setWordWrap(True)
        card_layout.addWidget(self.turn_label)
        card_layout.addWidget(self.move_label)
        card_layout.addWidget(self.status_label)
        outer.addWidget(card)

        primary_row = QHBoxLayout()
        self.btn_toggle = QPushButton("Démarrer l'analyse")
        self.btn_toggle.setObjectName("primary")
        self.btn_toggle.clicked.connect(self._toggle_running)
        self.btn_calibrate = QPushButton("Calibrer le plateau")
        self.btn_calibrate.clicked.connect(self.start_calibration)
        self.btn_settings = QPushButton("Réglages")
        self.btn_settings.clicked.connect(self._open_settings)
        primary_row.addWidget(self.btn_toggle, 1)
        primary_row.addWidget(self.btn_calibrate)
        primary_row.addWidget(self.btn_settings)
        outer.addLayout(primary_row)

        secondary_row = QHBoxLayout()
        self.btn_refresh = QPushButton("Actualiser")
        self.btn_refresh.setToolTip("Relire immédiatement l'écran et relancer l'analyse.")
        self.btn_refresh.clicked.connect(self._force_refresh)
        self.btn_flip = QPushButton("Inverser le trait")
        self.btn_flip.setToolTip(
            "Corrige le trait si l'assistant a démarré au milieu d'une partie "
            "ou s'est désynchronisé."
        )
        self.btn_flip.clicked.connect(self._flip_turn)
        secondary_row.addWidget(self.btn_refresh)
        secondary_row.addWidget(self.btn_flip)
        secondary_row.addStretch(1)
        outer.addLayout(secondary_row)

        self.detail_label = QLabel("")
        self.detail_label.setObjectName("detail")
        self.detail_label.setWordWrap(True)
        outer.addWidget(self.detail_label)

        footer = QLabel(
            "Placez cette fenêtre hors du plateau, ou laissez-la se réduire dans "
            "la zone de notification. Fermer la fenêtre ne quitte pas l'application."
        )
        footer.setObjectName("detail")
        footer.setWordWrap(True)
        outer.addWidget(footer)
        outer.addStretch(1)

    def _set_app_icon(self) -> None:
        size = 64
        pixmap = QPixmap(size, size)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setBrush(QColor("#FF8A00"))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(2, 2, size - 4, size - 4)
        painter.setPen(QColor("#111827"))
        font = QFont("Segoe UI Symbol", 38)
        painter.setFont(font)
        painter.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter, "♞")
        painter.end()
        self._app_icon = QIcon(pixmap)
        self.setWindowIcon(self._app_icon)

    def _setup_tray(self) -> None:
        if not QSystemTrayIcon.isSystemTrayAvailable():
            logger.info("Zone de notification indisponible.")
            return
        self._tray_icon = QSystemTrayIcon(self._app_icon, self)
        self._tray_icon.setToolTip(f"{APP_NAME} — en pause")
        menu = QMenu()
        self._tray_show_action = QAction("Afficher la fenêtre", self)
        self._tray_show_action.triggered.connect(self._show_main_window)
        self._tray_toggle_action = QAction("Démarrer l'analyse", self)
        self._tray_toggle_action.triggered.connect(self._toggle_running)
        calibrate_action = QAction("Calibrer le plateau", self)
        calibrate_action.triggered.connect(self.start_calibration)
        quit_action = QAction("Quitter", self)
        quit_action.triggered.connect(self._quit)
        menu.addAction(self._tray_show_action)
        menu.addAction(self._tray_toggle_action)
        menu.addAction(calibrate_action)
        menu.addSeparator()
        menu.addAction(quit_action)
        self._tray_menu = menu
        self._tray_icon.setContextMenu(menu)
        self._tray_icon.activated.connect(self._on_tray_activated)
        self._tray_icon.show()

    # ================================================================== startup
    def _init_capture(self) -> None:
        try:
            self.capture = ScreenCapture(monitor_index=1)
        except Exception as exc:
            logger.exception("Capture d'écran indisponible")
            self.capture = None
            message = str(exc)
            QTimer.singleShot(
                250,
                lambda: QMessageBox.critical(
                    self,
                    "Capture d'écran",
                    "Impossible d'accéder à l'écran principal.\n\n"
                    f"{message}\n\nFermez l'application puis relancez-la.",
                ),
            )

    def _load_calibration(self) -> None:
        if not CALIBRATION_PATH.is_file() or not TEMPLATES_PATH.is_file():
            self._calibration_loaded = False
            return
        try:
            library = TemplateLibrary.load(CALIBRATION_PATH, TEMPLATES_PATH)
            geometry = BoardGeometry(
                tuple((x, y) for x, y in library.meta.quad),
                library.meta.orientation,
            )
            if self.capture is not None:
                width, height = self.capture.size
                for x, y in geometry.quad:
                    if x < -2 or y < -2 or x > width + 2 or y > height + 2:
                        raise ValueError(
                            "La zone calibrée se trouve hors de l'écran actuel. "
                            "Une nouvelle calibration est nécessaire."
                        )
            self.library = library
            self.geometry = geometry
            self._calibration_loaded = True
            self.tracker.reset_to_start()
            self.overlay.set_visual_style(
                self.config.arrow_color,
                self.config.arrow_opacity,
                self.config.show_label,
            )
            logger.info("Calibration chargée (%s)", library.meta.created_at)
        except Exception as exc:
            logger.exception("Calibration invalide")
            self.library = None
            self.geometry = None
            self._calibration_loaded = False
            self._calibration_error = str(exc)

    def _resolve_engine(self) -> None:
        found = find_stockfish(self.config.stockfish_path)
        self._resolved_engine = str(found) if found else ""
        self._refresh_detail_label()

    def _refresh_detail_label(self) -> None:
        if not hasattr(self, "detail_label"):
            return
        if not self._calibration_loaded:
            calibration = "aucune calibration"
            if self._calibration_error:
                calibration = "calibration invalide"
        else:
            calibration = "plateau calibré"
        engine = Path(self._resolved_engine).name if self._resolved_engine else "introuvable"
        self.detail_label.setText(
            f"Moteur : {engine}   •   {calibration}   •   "
            f"écran principal uniquement"
        )

    def _startup(self) -> None:
        if self.capture is None:
            self._set_status("Capture d'écran indisponible.", "error")
            return
        if not self._calibration_loaded:
            if self._calibration_error:
                self._set_status(
                    "Calibration invalide: " + self._calibration_error, "error"
                )
            else:
                self._set_status(
                    "Calibration requise. Placez le plateau en position initiale, "
                    "puis cliquez sur « Calibrer le plateau ».",
                    "warn",
                )
            self._show_first_run_dialog()
            return
        if self._resolved_engine:
            self._start()
        else:
            self._set_status(
                "Calibration prête, mais Stockfish est introuvable. Ouvrez "
                "« Réglages » pour indiquer son exécutable.",
                "warn",
            )

    def _show_first_run_dialog(self) -> None:
        message = QMessageBox(self)
        message.setWindowTitle("Première utilisation")
        message.setIcon(QMessageBox.Icon.Information)
        message.setText("Bienvenue dans Chess Vision.")
        message.setInformativeText(
            "Avant de commencer :\n\n"
            "1. Ouvrez votre échiquier sur l'écran principal.\n"
            "2. Mettez-le en position initiale complète et sans animation.\n"
            "3. L'assistant va masquer cette fenêtre, puis vous cliquerez les "
            "quatre coins de la zone de jeu.\n\n"
            "L'outil ne fait que lire l'écran et dessiner une flèche. Il ne "
            "contrôle ni la souris ni le clavier."
        )
        start_button = message.addButton(
            "Commencer la calibration", QMessageBox.ButtonRole.AcceptRole
        )
        message.addButton("Plus tard", QMessageBox.ButtonRole.RejectRole)
        message.exec()
        if message.clickedButton() is start_button:
            self.start_calibration(confirm=False)
        else:
            self._set_status(
                "Calibration à effectuer quand vous êtes prêt, via « Calibrer le plateau ».",
                "warn",
            )

    # ================================================================== actions
    def _toggle_running(self) -> None:
        if self.running:
            self._pause()
        else:
            self._start()

    def _start(self) -> None:
        if self.running:
            return
        if not self._calibration_loaded:
            self._set_status("Calibration requise avant de démarrer.", "warn")
            self.start_calibration(confirm=False)
            return
        if self.capture is None:
            self._set_status("Capture d'écran indisponible.", "error")
            return

        self._resolve_engine()
        if not self._resolved_engine:
            self._set_status("Stockfish introuvable. Choisissez son exécutable.", "error")
            if not self._open_settings():
                return
            self._resolve_engine()
            if not self._resolved_engine:
                return

        if not self._ensure_engine():
            return

        self._reset_tracking_state()
        self.overlay.clear()
        self._set_move_text("En attente de la première lecture…")
        self.running = True
        self.timer.start(max(250, self.config.poll_ms))
        self._update_run_controls()
        self._update_turn_label()
        self._set_status("Lecture de l'écran et analyse Stockfish…", "info")
        QTimer.singleShot(50, self._poll)
        auto_hide = (
            self.config.hide_on_start
            and self._tray_icon is not None
            and not self._suppress_auto_hide
        )
        self._suppress_auto_hide = False
        if auto_hide:
            QTimer.singleShot(450, self._hide_to_tray)

    def _pause(self, message: str = "Analyse en pause.") -> None:
        if not self.running:
            self._update_run_controls()
            return
        self.running = False
        self.timer.stop()
        self.overlay.clear()
        self._last_suggestion = None
        self._update_run_controls()
        self._set_status(message, "info")
        if self._tray_icon is not None:
            self._tray_icon.setToolTip(f"{APP_NAME} — en pause")
        self._set_move_text("Aucune suggestion")

    def _reset_tracking_state(self) -> None:
        self._last_stable_placement = None
        self._candidate_placement = None
        self._candidate_count = 0
        self._recognition_failures = 0
        self._board_change_pending = False
        self._pending_fen = None
        self._last_requested_fen = None
        self._last_suggestion = None

    def _ensure_engine(self) -> bool:
        if (
            self.engine_worker is not None
            and self.engine_worker.isRunning()
            and self.engine_worker.engine_path == self._resolved_engine
        ):
            return True
        self._stop_engine()
        self._engine_failure_handled = False
        worker = EngineWorker(
            self._resolved_engine,
            threads=self.config.engine_threads,
            hash_mb=self.config.engine_hash_mb,
            parent=self,
        )
        worker.suggestion_ready.connect(self._on_suggestion)
        worker.engine_failed.connect(self._on_engine_failed)
        worker.engine_ready.connect(self._on_engine_ready)
        self.engine_worker = worker
        worker.start()
        return True

    def _stop_engine(self, wait_ms: int = 3500) -> None:
        worker = self.engine_worker
        self.engine_worker = None
        if worker is None:
            return
        worker.stop()
        if not worker.wait(wait_ms):
            logger.warning("Le thread Stockfish ne s'arrête pas à temps")

    def _open_settings(self) -> bool:
        dialog = SettingsDialog(self.config, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return False
        old_path = self._resolved_engine
        self.config = dialog.result_config()
        self.config.save()
        self._resolve_engine()
        self._update_visual_style()
        self._update_turn_label()
        self._refresh_detail_label()
        if self.running:
            self.timer.setInterval(max(250, self.config.poll_ms))
        if old_path != self._resolved_engine:
            if self.running:
                self._pause("Chemin du moteur modifié. Cliquez sur « Démarrer ».")
            self._stop_engine()
        elif self.running:
            self._refresh_advice()
        self._set_status("Réglages enregistrés.", "ok")
        return True

    def _refresh_advice(self) -> None:
        if not self.running:
            return
        if not self._should_advise(self.tracker.board):
            self.overlay.clear()
            self._last_suggestion = None
            self._set_move_text("En attente de votre tour.")
        elif self._last_suggestion is not None:
            self._display_suggestion(self._last_suggestion)
        else:
            self._force_refresh()

    def _update_visual_style(self) -> None:
        self.overlay.set_visual_style(
            self.config.arrow_color,
            self.config.arrow_opacity,
            self.config.show_label,
        )

    def _force_refresh(self) -> None:
        if not self.running:
            self._set_status("Démarrez l'analyse pour actualiser la flèche.", "warn")
            return
        self._poll(force=True)

    def _flip_turn(self) -> None:
        if not self._calibration_loaded:
            self._set_status("Calibration requise.", "warn")
            return
        self.tracker.flip_turn()
        self.overlay.clear()
        self._last_suggestion = None
        self._candidate_placement = None
        self._candidate_count = 0
        self._update_turn_label()
        if self.tracker.board.is_game_over():
            self._set_move_text("Partie terminée.")
            return
        if self._should_advise(self.tracker.board):
            if self.running:
                self._request_analysis(self.tracker.board, force=True)
            else:
                self._set_status("Trait inversé. Démarrez l'analyse pour actualiser.", "info")
        else:
            self._set_move_text("En attente de votre tour.")
            self._set_status("Trait inversé : conseils masqués.", "info")

    # ================================================================== calibration
    def start_calibration(self, confirm: bool = True) -> None:
        if self.capture is None:
            self._set_status("Capture d'écran indisponible.", "error")
            return
        was_running = self.running
        if confirm and not ask_calibration_warning(self):
            return
        if self.running:
            self._pause("Calibration en cours…")
        self.overlay.clear()
        self.hide()
        QTimer.singleShot(500, lambda: self._run_calibration_flow(was_running))

    def _run_calibration_flow(self, was_running: bool) -> None:
        try:
            if self.capture is None:
                raise RuntimeError("Capture d'écran indisponible.")
            frame = self.capture.grab()

            while True:
                selector = BoardSelectionDialog(frame, self)
                if selector.exec() != QDialog.DialogCode.Accepted:
                    logger.info("Sélection des coins annulée")
                    self._restore_after_calibration(was_running)
                    return
                points = selector.points
                if len(points) != 4:
                    raise ValueError("Quatre coins sont nécessaires.")

                try:
                    temporary = BoardGeometry(points, ORIENTATION_WHITE)
                except ValueError as exc:
                    QMessageBox.warning(
                        None,
                        "Coins invalides",
                        f"{exc}\n\nCliquez à nouveau les quatre coins, dans "
                        "l'ordre indiqué.",
                    )
                    continue
                warped = temporary.warp(frame)
                default_orientation = (
                    ORIENTATION_BLACK
                    if self.config.human_color == "black"
                    else ORIENTATION_WHITE
                )
                preview = CalibrationPreviewDialog(
                    warped, default_orientation, self
                )
                if preview.exec() == QDialog.DialogCode.Accepted:
                    orientation = preview.orientation()
                    break
                if preview.retry_requested:
                    logger.info("Nouvelle sélection des coins demandée")
                    continue
                logger.info("Validation de la calibration annulée")
                self._restore_after_calibration(was_running)
                return

            geometry = BoardGeometry(points, orientation)
            library = TemplateLibrary.calibrate(warped, geometry)

            ensure_data_dir()
            library.save(CALIBRATION_PATH, TEMPLATES_PATH)
            try:
                cv2.imwrite(str(PREVIEW_PATH), warped)
            except Exception:
                logger.warning("Impossible de sauvegarder l'aperçu", exc_info=True)

            self.library = library
            self.geometry = geometry
            self._calibration_loaded = True
            self._calibration_error = ""
            if self.config.advice_mode == "human":
                self.config.human_color = (
                    "white" if orientation == ORIENTATION_WHITE else "black"
                )
            self.config.save()
            self.tracker.reset_to_start()
            self._reset_tracking_state()
            self.overlay.clear()
            self._update_visual_style()
            self._update_turn_label()
            self._refresh_detail_label()
            self._set_move_text("Calibration enregistrée.")
            self._set_status(
                "Calibration réussie. Gardez le plateau fixe et visible.", "ok"
            )
            self.show()
            self.raise_()
            self.activateWindow()
            if self._tray_icon is not None:
                self._tray_icon.showMessage(
                    APP_NAME,
                    "Calibration enregistrée. L'analyse visuelle démarre.",
                    QSystemTrayIcon.MessageIcon.Information,
                    3500,
                )
            if self._resolved_engine:
                self._suppress_auto_hide = True
                QTimer.singleShot(200, self._start)
            else:
                self._set_status(
                    "Calibration réussie, mais Stockfish est introuvable. "
                    "Ouvrez « Réglages ».",
                    "warn",
                )
        except Exception as exc:
            logger.exception("Échec de la calibration")
            QMessageBox.critical(
                None,
                "Calibration impossible",
                "La calibration n'a pas pu être enregistrée.\n\n"
                f"{exc}\n\n"
                "Vérifiez que le plateau est entièrement visible, en position "
                "initiale, puis réessayez.",
            )
            self._restore_after_calibration(was_running)

    def _restore_after_calibration(self, was_running: bool) -> None:
        if self.isMinimized():
            self.showNormal()
        self.show()
        self.raise_()
        self.activateWindow()
        if self.library is None:
            self._set_status("Calibration annulée ou invalide.", "warn")
        if was_running and self._calibration_loaded:
            self._start()

    # ================================================================== polling
    def _grab_frame(self) -> np.ndarray:
        assert self.capture is not None
        hidden = False
        if self.overlay.isVisible() and not self.overlay.capture_excluded:
            self.overlay.hide_for_capture()
            QApplication.processEvents()
            hidden = True
            time.sleep(0.04)
        try:
            return self.capture.grab()
        finally:
            if hidden:
                self.overlay.restore_after_capture()

    def _poll(self, force: bool = False) -> None:
        if not self.running or self._polling:
            return
        if self.library is None or self.geometry is None or self.capture is None:
            return
        self._polling = True
        try:
            frame = self._grab_frame()
            height, width = frame.shape[:2]
            self._frame_size = (int(width), int(height))
            self.overlay.set_capture_size(width, height)
            warped = self.geometry.warp(frame)
            result = self.library.recognize(warped, self.config.min_score)
        except Exception:
            logger.exception("Échec de la lecture d'écran")
            self._recognition_failures += 1
            self._set_status(
                "Lecture impossible. Vérifiez que le plateau est visible.", "error"
            )
            if self._recognition_failures >= 3:
                self.overlay.clear()
                self._last_stable_placement = None
                self._last_suggestion = None
            return
        finally:
            self._polling = False

        if not result.ok:
            self._on_recognition_failure(result)
            return
        self._recognition_failures = 0
        self._handle_placement(result.placement, result.confidence, force=force)

    def _on_recognition_failure(self, result: RecognitionResult) -> None:
        self._recognition_failures += 1
        if self._recognition_failures == 1:
            self._set_status(
                "Lecture incertaine. Attendez la fin d'une animation ou d'un "
                "survol de pièce, et gardez le plateau visible.",
                "warn",
            )
        if self._recognition_failures >= 3:
            self.overlay.clear()
            self._last_suggestion = None
            self._last_stable_placement = None
            self._candidate_placement = None
            self._candidate_count = 0
            self._board_change_pending = False
            self._set_move_text("Flèche masquée (position illisible).")
            self._set_status(
                "Impossible de lire la position de façon stable. "
                f"{result.error or ''} Recalibrez si la fenêtre a bougé.",
                "warn",
            )

    def _handle_placement(
        self, placement: str, confidence: float, force: bool = False
    ) -> None:
        if self._last_stable_placement is None:
            self._accept_placement(placement, confidence)
            return

        if placement == self._last_stable_placement:
            was_pending = self._board_change_pending
            self._candidate_placement = None
            self._candidate_count = 0
            self._board_change_pending = False
            if force and self._should_advise(self.tracker.board):
                self._request_analysis(self.tracker.board, force=True)
            elif was_pending:
                self._maybe_restore_suggestion()
            return

        # The board is no longer the one for which the arrow was computed.
        self._board_change_pending = True
        self.overlay.clear()
        self._last_suggestion = None
        self._set_move_text("Nouvelle position détectée…")
        if placement == self._candidate_placement:
            self._candidate_count += 1
        else:
            self._candidate_placement = placement
            self._candidate_count = 1
        required = 1 if force else max(1, self.config.confirmations)
        if self._candidate_count < required:
            self._set_status("Vérification de la nouvelle position…", "info")
            return
        self._accept_placement(placement, confidence)

    def _accept_placement(self, placement: str, confidence: float) -> None:
        tracker_result = self.tracker.observe(placement)
        self._last_stable_placement = placement
        self._candidate_placement = None
        self._candidate_count = 0
        self._board_change_pending = False
        self._last_suggestion = None
        self._pending_fen = None
        self._update_turn_label()
        logger.debug(
            "Position acceptée (confiance %.3f, statut %s)",
            confidence,
            tracker_result.status,
        )

        board = self.tracker.board
        if tracker_result.status == "resync" or tracker_result.status == "new_game":
            self._set_status(
                tracker_result.message
                or "Position resynchronisée. Vérifiez le trait si nécessaire.",
                "warn",
            )
        if board.is_game_over():
            self.overlay.clear()
            self._set_move_text("Partie terminée.")
            self._set_status("La partie est terminée (mat, pat ou nulle).", "info")
            return
        if not self._should_advise(board):
            self.overlay.clear()
            self._set_move_text("En attente de votre tour.")
            self._set_status(
                "Trait adverse : les suggestions sont masquées. "
                "Elles reviendront à votre tour.",
                "info",
            )
            return
        if not self.running:
            return
        self._request_analysis(board, force=True)

    def _request_analysis(self, board: chess.Board, force: bool = False) -> None:
        if not self.running or self.engine_worker is None:
            return
        if not self.engine_worker.isRunning():
            self._set_status("Stockfish n'est pas disponible.", "error")
            return
        fen = board.fen()
        if not force and fen == self._last_requested_fen:
            return
        self._last_requested_fen = fen
        self._pending_fen = fen
        self.overlay.clear()
        self._set_move_text("Analyse en cours…")
        self._set_status(
            f"Trait aux {self._turn_name(board)} — Stockfish réfléchit…", "info"
        )
        self.engine_worker.request(board, self.config.movetime_ms)

    def _maybe_restore_suggestion(self) -> None:
        if (
            self.running
            and self._last_suggestion is not None
            and self._last_suggestion.fen == self.tracker.board.fen()
            and self._should_advise(self.tracker.board)
        ):
            self._display_suggestion(self._last_suggestion)

    def _on_suggestion(self, suggestion: Suggestion) -> None:
        if not self.running or self._board_change_pending:
            return
        if self._pending_fen is not None and suggestion.fen != self._pending_fen:
            return
        if self.tracker.board.fen() != suggestion.fen:
            return
        if not self._should_advise(self.tracker.board):
            return
        try:
            move = chess.Move.from_uci(suggestion.uci)
        except ValueError:
            logger.warning("Coup UCI invalide reçu: %s", suggestion.uci)
            return
        if move not in self.tracker.board.legal_moves:
            logger.warning("Coup proposé illégal: %s", suggestion.uci)
            return
        self._last_suggestion = suggestion
        self._pending_fen = None
        self._display_suggestion(suggestion)

    def _display_suggestion(self, suggestion: Suggestion) -> None:
        if self.geometry is None or not self.running:
            return
        try:
            move = chess.Move.from_uci(suggestion.uci)
            start_frame = self.geometry.square_center_in_frame(move.from_square)
            end_frame = self.geometry.square_center_in_frame(move.to_square)
        except Exception:
            logger.exception("Impossible de positionner la flèche")
            return
        if self._frame_size == (0, 0) and self.capture is not None:
            self._frame_size = self.capture.size
        self.overlay.set_capture_size(*self._frame_size)
        label = (
            f"{suggestion.san}   {suggestion.score_text}".strip()
            if self.config.show_label
            else ""
        )
        self.overlay.show_move(start_frame, end_frame, label)
        board = self.tracker.board
        self._set_move_text(f"Coup conseillé : {suggestion.san}")
        eval_text = suggestion.score_text or "—"
        self._set_status(
            f"Trait aux {self._turn_name(board)}. Évaluation : {eval_text} "
            "(du point de vue des Blancs).",
            "ok",
        )
        if self._tray_icon is not None:
            self._tray_icon.setToolTip(
                f"{APP_NAME} — {suggestion.san} ({suggestion.score_text})"
            )
        self._update_turn_label()

    def _on_engine_ready(self) -> None:
        logger.info("Moteur Stockfish prêt")
        if self.running and self._last_stable_placement is None:
            self._set_status("Stockfish prêt. Lecture de l'écran…", "ok")

    def _on_engine_failed(self, message: str) -> None:
        logger.error("Échec du moteur: %s", message)
        if self._engine_failure_handled:
            return
        self._engine_failure_handled = True
        self.timer.stop()
        self.running = False
        self.overlay.clear()
        self._show_main_window()
        self._update_run_controls()
        self._set_status("Stockfish s'est arrêté. Vérifiez le chemin du moteur.", "error")
        self._stop_engine(1200)
        self._pending_fen = None
        self._last_requested_fen = None
        self._last_suggestion = None
        QMessageBox.critical(self, "Stockfish", message)
        self._update_turn_label()

    # ================================================================== UI helpers
    def _set_status(self, text: str, level: str = "info") -> None:
        colors = {
            "info": "#9CA3AF",
            "ok": "#86EFAC",
            "warn": "#FBBF24",
            "error": "#F87171",
        }
        self.status_label.setStyleSheet(f"color: {colors.get(level, colors['info'])};")
        self.status_label.setText(text)

    def _set_move_text(self, text: str) -> None:
        self.move_label.setText(text)

    def _update_turn_label(self) -> None:
        if not self._calibration_loaded:
            self.turn_label.setText("Trait : —")
            return
        board = self.tracker.board
        name = "Blancs" if board.turn == chess.WHITE else "Noirs"
        suffix = ""
        if self.config.advice_mode == "human":
            human = chess.WHITE if self.config.human_color == "white" else chess.BLACK
            suffix = " · votre trait" if board.turn == human else " · trait adverse"
        self.turn_label.setText(f"Trait : {name}{suffix}")

    @staticmethod
    def _turn_name(board: chess.Board) -> str:
        return "Blancs" if board.turn == chess.WHITE else "Noirs"

    def _should_advise(self, board: chess.Board) -> bool:
        if self.config.advice_mode == "always":
            return True
        human = chess.WHITE if self.config.human_color == "white" else chess.BLACK
        return board.turn == human

    def _update_run_controls(self) -> None:
        self.btn_toggle.setText(
            "Mettre en pause" if self.running else "Démarrer l'analyse"
        )
        self.btn_refresh.setEnabled(self.running)
        self.btn_flip.setEnabled(self._calibration_loaded)
        if self._tray_toggle_action is not None:
            self._tray_toggle_action.setText(
                "Mettre en pause" if self.running else "Démarrer l'analyse"
            )
        if self._tray_icon is not None:
            self._tray_icon.setToolTip(
                f"{APP_NAME} — {'analyse en cours' if self.running else 'en pause'}"
            )

    def _show_main_window(self) -> None:
        if self.isMinimized():
            self.showNormal()
        self.show()
        self.raise_()
        self.activateWindow()

    def _hide_to_tray(self) -> None:
        if self._tray_icon is None or not self._tray_icon.isVisible():
            return
        if not self.running:
            return
        self.hide()
        if not self._tray_hint_shown:
            self._tray_icon.showMessage(
                APP_NAME,
                "Analyse en cours. Cliquez sur l'icône pour rouvrir les commandes "
                "ou mettre en pause.",
                QSystemTrayIcon.MessageIcon.Information,
                4000,
            )
            self._tray_hint_shown = True

    def _on_tray_activated(self, reason: QSystemTrayIcon.ActivationReason) -> None:
        if reason in (
            QSystemTrayIcon.ActivationReason.Trigger,
            QSystemTrayIcon.ActivationReason.DoubleClick,
        ):
            if self.isVisible():
                self.hide()
            else:
                self._show_main_window()

    # ================================================================== shutdown
    def closeEvent(self, event) -> None:  # noqa: N802
        if self._quitting:
            event.accept()
            return
        if self._tray_icon is not None and self._tray_icon.isVisible():
            event.ignore()
            self.hide()
            if not self._tray_hint_shown:
                self._tray_icon.showMessage(
                    APP_NAME,
                    "L'application continue dans la zone de notification. "
                    "Utilisez son menu pour quitter.",
                    QSystemTrayIcon.MessageIcon.Information,
                    4000,
                )
                self._tray_hint_shown = True
            return
        self._quit()
        event.accept()

    def _quit(self) -> None:
        self._quitting = True
        self.timer.stop()
        self.running = False
        self.overlay.clear()
        try:
            self.overlay.close()
        except Exception:
            logger.debug("Fermeture de l'overlay ignorée", exc_info=True)
        self.shutdown_engine()
        if self.capture is not None:
            self.capture.close()
            self.capture = None
        if self._tray_icon is not None:
            self._tray_icon.hide()
        QApplication.quit()

    def shutdown_engine(self) -> None:
        self._stop_engine(4000)

    def shutdown(self) -> None:
        """Idempotent cleanup for QApplication.aboutToQuit."""
        self._quitting = True
        self.timer.stop()
        self.running = False
        self.overlay.clear()
        try:
            self.overlay.close()
        except Exception:
            logger.debug("Fermeture de l'overlay ignorée", exc_info=True)
        self.shutdown_engine()
        if self.capture is not None:
            self.capture.close()
            self.capture = None
        if self._tray_icon is not None:
            self._tray_icon.hide()
