"""Application entry point."""

from __future__ import annotations

import ctypes
import logging
import sys
from logging.handlers import RotatingFileHandler

from PyQt6.QtWidgets import QApplication, QMessageBox

from chess_vision.app import MainWindow
from chess_vision.config import (
    APP_ID,
    APP_NAME,
    DATA_DIR,
    VERSION,
    ensure_data_dir,
)

logger = logging.getLogger(__name__)


def _configure_logging() -> None:
    ensure_data_dir()
    formatter = logging.Formatter(
        "%(asctime)s  %(levelname)-7s  %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    if sys.stderr is not None:
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        root.addHandler(console)

    try:
        log_path = DATA_DIR / "chess_vision.log"
        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=1_000_000,
            backupCount=3,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
    except OSError:
        logger.warning("Journalisation sur fichier indisponible", exc_info=True)


def _install_exception_hook() -> None:
    def hook(exc_type, exc, traceback) -> None:
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc, traceback)
            return
        logger.critical("Exception non gérée", exc_info=(exc_type, exc, traceback))
        try:
            QMessageBox.critical(
                None,
                APP_NAME,
                "Une erreur inattendue s'est produite.\n\n"
                f"{exc}\n\n"
                "Les détails ont été écrits dans le journal de l'application.",
            )
        except Exception:
            pass

    sys.excepthook = hook


def _set_windows_app_id() -> None:
    if sys.platform != "win32":
        return
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            f"{APP_ID}.{VERSION}"
        )
    except Exception:
        logger.debug("Identifiant applicatif Windows non défini", exc_info=True)


def main() -> int:
    _configure_logging()
    _set_windows_app_id()

    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setApplicationDisplayName(APP_NAME)
    app.setApplicationVersion(VERSION)
    app.setOrganizationName(APP_ID)
    app.setQuitOnLastWindowClosed(False)

    _install_exception_hook()

    window = MainWindow()
    app.aboutToQuit.connect(window.shutdown)
    window.show()
    exit_code = app.exec()
    window.shutdown()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
