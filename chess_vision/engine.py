"""Stockfish integration in a background Qt thread."""

from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass

import chess
import chess.engine
from PyQt6.QtCore import QThread, pyqtSignal

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Suggestion:
    fen: str
    uci: str
    san: str
    score_text: str
    score_cp: int | None
    mate: int | None
    depth: int
    pv: tuple[str, ...]


def _format_eval(score: chess.engine.PovScore | None) -> tuple[str, int | None, int | None]:
    """Format an evaluation from White's point of view."""
    if score is None:
        return "", None, None
    white = score.pov(chess.WHITE)
    if white.is_mate():
        mate = white.mate()
        if mate is None:
            return "", None, None
        return (f"#{abs(mate)}" if mate > 0 else f"-#{abs(mate)}"), None, mate
    cp = white.score()
    if cp is None:
        return "", None, None
    return f"{cp / 100:+.2f}", cp, None


class EngineWorker(QThread):
    """A single persistent engine, fed with the latest position only."""

    suggestion_ready = pyqtSignal(object)
    engine_failed = pyqtSignal(str)
    engine_ready = pyqtSignal()

    def __init__(
        self,
        engine_path: str,
        threads: int = 2,
        hash_mb: int = 128,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._engine_path = engine_path
        self._threads = max(1, int(threads))
        self._hash_mb = max(16, int(hash_mb))
        self._requests: queue.Queue[tuple[chess.Board, int] | None] = queue.Queue(
            maxsize=1
        )
        self._lock = threading.Lock()
        self._stop_requested = False

    @property
    def engine_path(self) -> str:
        return self._engine_path

    def request(self, board: chess.Board, movetime_ms: int) -> None:
        """Replace any queued request with this new position."""
        if self._stop_requested:
            return
        item = (board.copy(stack=False), max(100, int(movetime_ms)))
        with self._lock:
            try:
                while True:
                    self._requests.get_nowait()
            except queue.Empty:
                pass
            try:
                self._requests.put_nowait(item)
            except queue.Full:
                pass

    def stop(self) -> None:
        self._stop_requested = True
        with self._lock:
            try:
                while True:
                    self._requests.get_nowait()
            except queue.Empty:
                pass
            try:
                self._requests.put_nowait(None)
            except queue.Full:
                pass

    def run(self) -> None:
        engine: chess.engine.SimpleEngine | None = None
        try:
            logger.info("Démarrage de Stockfish: %s", self._engine_path)
            engine = chess.engine.SimpleEngine.popen_uci(
                [self._engine_path], timeout=15.0
            )
            try:
                engine.configure(
                    {"Threads": self._threads, "Hash": self._hash_mb}
                )
            except Exception:
                logger.warning("Options Threads/Hash non acceptées par le moteur", exc_info=True)
            self.engine_ready.emit()

            while not self._stop_requested:
                item = self._requests.get()
                if item is None:
                    break
                board, movetime_ms = item
                if self._stop_requested:
                    break

                info = engine.analyse(
                    board,
                    chess.engine.Limit(time=movetime_ms / 1000.0),
                )
                pv = info.get("pv") or []
                if not pv:
                    continue
                best = pv[0]
                if best not in board.legal_moves:
                    continue
                try:
                    san = board.san(best)
                except Exception:
                    san = best.uci()
                score_text, score_cp, mate = _format_eval(info.get("score"))
                suggestion = Suggestion(
                    fen=board.fen(),
                    uci=best.uci(),
                    san=san,
                    score_text=score_text,
                    score_cp=score_cp,
                    mate=mate,
                    depth=int(info.get("depth", 0) or 0),
                    pv=tuple(move.uci() for move in pv),
                )
                self.suggestion_ready.emit(suggestion)
        except Exception as exc:
            logger.exception("Erreur du moteur Stockfish")
            message = str(exc).strip() or exc.__class__.__name__
            self.engine_failed.emit(
                "Stockfish n'a pas pu démarrer ou s'est arrêté.\n\n"
                f"{message}\n\nVérifiez le chemin du moteur dans les réglages."
            )
        finally:
            if engine is not None:
                try:
                    engine.quit()
                except Exception:
                    logger.debug("Arrêt de Stockfish ignoré", exc_info=True)
            logger.info("Thread Stockfish terminé")
