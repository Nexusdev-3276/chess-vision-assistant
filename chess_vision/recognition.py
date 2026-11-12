"""Piece recognition learned from one screenshot of the initial position.

The tool deliberately avoids baked-in piece artwork: board themes vary a lot.
At calibration time it learns the 12 piece appearances of the theme currently
on screen, plus the visual activity of empty squares. Recognition is then a
small nearest-neighbour template matcher on background-normalised patches.

This is not a neural network and makes no attempt to understand arbitrary
themes. It is a robust, transparent recogniser for the exact theme and board
size captured at calibration time.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import chess
import cv2
import numpy as np

from .geometry import (
    CANVAS_SIZE,
    CELL_SIZE,
    VALID_ORIENTATIONS,
    BoardGeometry,
)

logger = logging.getLogger(__name__)

EMPTY_LABEL = "."
PIECE_LABELS = tuple("PNBRQKpnbrqk")
FEATURE_SIZE = 64
MARGIN_RATIO = 0.10
RAW_WEIGHT = 0.60
HIGH_PASS_WEIGHT = 0.40
HIGH_PASS_SIGMA = 1.2
ACTIVITY_SIGMA = 3.0
MIN_MARGIN = 0.02
BACKGROUND_PATCH = 5
CALIBRATION_VERSION = 2


@dataclass
class CalibrationMeta:
    quad: list[list[float]]
    orientation: str
    empty_threshold: float
    created_at: str
    version: int = CALIBRATION_VERSION
    canvas_size: int = CANVAS_SIZE
    start_fen: str = chess.STARTING_FEN

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "quad": self.quad,
            "orientation": self.orientation,
            "empty_threshold": self.empty_threshold,
            "created_at": self.created_at,
            "canvas_size": self.canvas_size,
            "start_fen": self.start_fen,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CalibrationMeta":
        quad = data["quad"]
        orientation = str(data["orientation"])
        if orientation not in VALID_ORIENTATIONS:
            raise ValueError("Orientation de calibration invalide.")
        return cls(
            quad=[[float(x), float(y)] for x, y in quad],
            orientation=orientation,
            empty_threshold=float(data.get("empty_threshold", 1.0)),
            created_at=str(data.get("created_at", "")),
            version=int(data.get("version", CALIBRATION_VERSION)),
            canvas_size=int(data.get("canvas_size", CANVAS_SIZE)),
            start_fen=str(data.get("start_fen", chess.STARTING_FEN)),
        )


@dataclass
class RecognitionResult:
    ok: bool
    placement: str = ""
    labels: tuple[str, ...] = ()
    confidence: float = 0.0
    unknown_cells: tuple[int, ...] = ()
    error: str = ""


def _cell_gray(cell: np.ndarray) -> np.ndarray:
    if cell is None or cell.size == 0:
        raise ValueError("Case vide.")
    h, w = cell.shape[:2]
    mx = max(2, int(round(w * MARGIN_RATIO)))
    my = max(2, int(round(h * MARGIN_RATIO)))
    inner = cell[my : h - my, mx : w - mx]
    if inner.size == 0:
        raise ValueError("Case trop petite.")
    if inner.ndim == 2:
        gray = inner.astype(np.float32)
    else:
        gray = cv2.cvtColor(inner, cv2.COLOR_BGR2GRAY).astype(np.float32)
    if gray.shape != (FEATURE_SIZE, FEATURE_SIZE):
        gray = cv2.resize(
            gray, (FEATURE_SIZE, FEATURE_SIZE), interpolation=cv2.INTER_AREA
        )
    return gray


def _background_level(gray: np.ndarray) -> float:
    """Estimate the square's background brightness from its four corners.

    Looking at the corners (not the centre) avoids a piece dominating the
    estimate.  The median across the four corners also rejects a coordinate
    label or a stray anti-aliased pixel in one corner.
    """
    h, w = gray.shape[:2]
    k = max(2, min(BACKGROUND_PATCH, h // 4, w // 4))
    patches = (
        gray[:k, :k],
        gray[:k, w - k :],
        gray[h - k :, :k],
        gray[h - k :, w - k :],
    )
    levels = [float(np.median(patch)) for patch in patches]
    return max(float(np.median(levels)), 1.0)


def _normalised(gray: np.ndarray) -> np.ndarray:
    """Bring a cell into a background-independent contrast space.

    A logarithm of the ratio is used instead of a plain subtraction: a black
    piece on a light square and the same piece on a dark square then produce
    closely matching patterns, which matters for queens and kings because the
    initial position only shows them on one square colour.
    """
    background = _background_level(gray)
    normalised = np.log((gray.astype(np.float32) + 1.0) / (background + 1.0))
    return np.clip(normalised, -4.0, 4.0)


def _activity_from(normalised: np.ndarray) -> float:
    hp = normalised - cv2.GaussianBlur(normalised, (0, 0), ACTIVITY_SIGMA)
    return float(np.mean(np.abs(hp)))


def _activity(gray: np.ndarray) -> float:
    return _activity_from(_normalised(gray))


def _features(gray: np.ndarray) -> tuple[np.ndarray, float]:
    """Return a unit-normalised raw+high-pass feature vector and activity."""
    normalised = _normalised(gray)
    raw = normalised - float(np.mean(normalised))
    raw_norm = float(np.linalg.norm(raw))
    if raw_norm > 1e-6:
        raw = raw / raw_norm
    else:
        raw = np.zeros_like(raw)

    hp = normalised - cv2.GaussianBlur(normalised, (0, 0), HIGH_PASS_SIGMA)
    hp = hp - float(np.mean(hp))
    hp_norm = float(np.linalg.norm(hp))
    if hp_norm > 1e-6:
        hp = hp / hp_norm
    else:
        hp = np.zeros_like(hp)

    vector = np.concatenate(
        [
            raw.ravel() * np.sqrt(RAW_WEIGHT),
            hp.ravel() * np.sqrt(HIGH_PASS_WEIGHT),
        ]
    ).astype(np.float32)
    norm = float(np.linalg.norm(vector))
    if norm > 1e-6:
        vector = vector / norm
    return vector, _activity_from(normalised)


def extract_visual_cells(canvas: np.ndarray) -> list[np.ndarray]:
    if canvas.shape[0] < CANVAS_SIZE or canvas.shape[1] < CANVAS_SIZE:
        raise ValueError("Image redressée trop petite.")
    cells: list[np.ndarray] = []
    for row in range(8):
        for col in range(8):
            y0, y1 = row * CELL_SIZE, (row + 1) * CELL_SIZE
            x0, x1 = col * CELL_SIZE, (col + 1) * CELL_SIZE
            cells.append(canvas[y0:y1, x0:x1])
    return cells


def _expected_start_labels(geometry: BoardGeometry) -> list[str]:
    board = chess.Board()
    labels: list[str] = []
    for row in range(8):
        for col in range(8):
            square = BoardGeometry.cell_to_square(row, col, geometry.orientation)
            piece = board.piece_at(square)
            labels.append(piece.symbol() if piece else EMPTY_LABEL)
    return labels


def _infer_empty_threshold(empty_activities: list[float], piece_activities: list[float]) -> float:
    if not empty_activities or not piece_activities:
        return 0.5
    empty_max = max(empty_activities)
    piece_min = min(piece_activities)
    if piece_min > empty_max:
        threshold = empty_max + 0.30 * (piece_min - empty_max)
    else:
        # Poorly separated theme (textured board). Stay conservative so that
        # real pieces are not mistaken for empty squares.
        threshold = max(0.0, piece_min * 0.45)
    return float(max(0.05, threshold))


def _validate_position(board: chess.Board) -> str | None:
    white_kings = len(board.pieces(chess.KING, chess.WHITE))
    black_kings = len(board.pieces(chess.KING, chess.BLACK))
    if white_kings != 1 or black_kings != 1:
        return "Il faut exactement un roi de chaque couleur."

    wk = board.king(chess.WHITE)
    bk = board.king(chess.BLACK)
    if wk is None or bk is None:
        return "Roi manquant."
    if chess.square_distance(wk, bk) <= 1:
        return "Les deux rois sont adjacents."

    for color in (chess.WHITE, chess.BLACK):
        pieces = list(board.piece_map().values())
        own = [p for p in pieces if p.color == color]
        if len(own) > 16:
            return "Trop de pièces pour une seule couleur."
        if len(board.pieces(chess.PAWN, color)) > 8:
            return "Trop de pions."
        for pawn_square in board.pieces(chess.PAWN, color):
            rank = chess.square_rank(pawn_square)
            if rank in (0, 7):
                return "Un pion se trouve sur la dernière rangée."
    return None


class TemplateLibrary:
    """Learned piece templates plus calibration metadata."""

    def __init__(
        self,
        templates: np.ndarray,
        labels: np.ndarray,
        meta: CalibrationMeta,
    ) -> None:
        if templates.ndim != 2:
            raise ValueError("Matrice de modèles invalide.")
        labels = np.asarray([str(label) for label in labels], dtype=str)
        if len(labels) != templates.shape[0]:
            raise ValueError("Nombre de modèles incohérent.")
        if templates.shape[0] == 0:
            raise ValueError("Aucun modèle de pièce.")
        if labels.size == 0:
            raise ValueError("Aucun label de pièce.")

        self.meta = meta
        self._labels = labels
        self._templates = np.asarray(templates, dtype=np.float32)
        # Defensive renormalisation when loading older files.
        norms = np.linalg.norm(self._templates, axis=1, keepdims=True)
        self._templates = self._templates / np.maximum(norms, 1e-9)

        self._class_indices: dict[str, np.ndarray] = {}
        for label in PIECE_LABELS:
            indices = np.flatnonzero(self._labels == label)
            if indices.size == 0:
                raise ValueError(f"Aucune apparence apprise pour la pièce '{label}'.")
            self._class_indices[label] = indices
        if set(np.unique(self._labels)) - set(PIECE_LABELS):
            raise ValueError("Labels de pièces inconnus dans les modèles.")

    @classmethod
    def calibrate(
        cls,
        warped_board: np.ndarray,
        geometry: BoardGeometry,
    ) -> "TemplateLibrary":
        """Learn pieces from a screenshot showing the standard initial set-up."""
        if warped_board.shape[0] != CANVAS_SIZE or warped_board.shape[1] != CANVAS_SIZE:
            raise ValueError(
                f"L'image du plateau doit faire {CANVAS_SIZE}x{CANVAS_SIZE}."
            )

        cells = extract_visual_cells(warped_board)
        expected = _expected_start_labels(geometry)
        features: list[np.ndarray] = []
        feature_labels: list[str] = []
        empty_activities: list[float] = []
        piece_activities: list[float] = []

        for cell, label in zip(cells, expected):
            gray = _cell_gray(cell)
            vector, activity = _features(gray)
            if label == EMPTY_LABEL:
                empty_activities.append(activity)
            else:
                if float(np.linalg.norm(vector)) < 1e-6:
                    logger.warning("Apparence quasi vide pour la pièce %s", label)
                features.append(vector)
                feature_labels.append(label)
                piece_activities.append(activity)

        if not features:
            raise ValueError("Aucune pièce détectée sur le plateau de calibration.")
        missing = set(PIECE_LABELS) - set(feature_labels)
        if missing:
            raise ValueError(
                "La position initiale complète est requise pour apprendre toutes "
                "les pièces. Apparences manquantes: " + ", ".join(sorted(missing))
            )

        empty_threshold = _infer_empty_threshold(empty_activities, piece_activities)
        meta = CalibrationMeta(
            quad=[[float(x), float(y)] for x, y in geometry.quad],
            orientation=geometry.orientation,
            empty_threshold=empty_threshold,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        logger.info(
            "Calibration: seuil de case vide=%.3f (vides %.3f..%.3f, pièces %.3f..%.3f)",
            empty_threshold,
            min(empty_activities) if empty_activities else 0.0,
            max(empty_activities) if empty_activities else 0.0,
            min(piece_activities),
            max(piece_activities),
        )
        return cls(
            templates=np.stack(features),
            labels=np.asarray(feature_labels, dtype=str),
            meta=meta,
        )

    def recognize(
        self,
        warped_board: np.ndarray,
        min_score: float = 0.55,
    ) -> RecognitionResult:
        """Read a rectified board and return its piece placement (no turn info)."""
        if warped_board.shape[0] != CANVAS_SIZE or warped_board.shape[1] != CANVAS_SIZE:
            return RecognitionResult(False, error="Taille d'image inattendue.")
        if warped_board.ndim == 2:
            color_canvas = cv2.cvtColor(warped_board, cv2.COLOR_GRAY2BGR)
        else:
            color_canvas = warped_board

        labels: list[str] = []
        unknown: list[int] = []
        scores: list[float] = []
        cells = extract_visual_cells(color_canvas)

        for index, cell in enumerate(cells):
            try:
                gray = _cell_gray(cell)
                vector, activity = _features(gray)
            except Exception:
                labels.append(EMPTY_LABEL)
                unknown.append(index)
                scores.append(0.0)
                continue

            if activity <= self.meta.empty_threshold:
                labels.append(EMPTY_LABEL)
                scores.append(1.0)
                continue

            if float(np.linalg.norm(vector)) < 1e-6:
                labels.append(EMPTY_LABEL)
                unknown.append(index)
                scores.append(0.0)
                continue

            sims = self._templates @ vector
            class_scores: dict[str, float] = {}
            for label, indices in self._class_indices.items():
                class_scores[label] = float(np.max(sims[indices]))
            ranked = sorted(class_scores.items(), key=lambda item: item[1], reverse=True)
            best_label, best_score = ranked[0]
            second_score = ranked[1][1] if len(ranked) > 1 else -1.0
            scores.append(best_score)

            if best_score < min_score or (best_score - second_score) < MIN_MARGIN:
                labels.append(EMPTY_LABEL)
                unknown.append(index)
            else:
                labels.append(best_label)

        if unknown:
            return RecognitionResult(
                ok=False,
                labels=tuple(labels),
                confidence=float(min(scores)) if scores else 0.0,
                unknown_cells=tuple(unknown),
                error=f"{len(unknown)} case(s) incertaine(s).",
            )

        board = chess.Board(None)
        try:
            for row in range(8):
                for col in range(8):
                    label = labels[row * 8 + col]
                    if label == EMPTY_LABEL:
                        continue
                    square = BoardGeometry.cell_to_square(row, col, self.meta.orientation)
                    board.set_piece_at(square, chess.Piece.from_symbol(label))
        except Exception as exc:
            return RecognitionResult(False, error=f"Placement illégal: {exc}")

        error = _validate_position(board)
        if error:
            return RecognitionResult(
                ok=False,
                labels=tuple(labels),
                confidence=float(min(scores)) if scores else 0.0,
                error=error,
            )
        return RecognitionResult(
            ok=True,
            placement=board.board_fen(),
            labels=tuple(labels),
            confidence=float(min(scores)) if scores else 0.0,
        )

    def save(self, meta_path: Path, templates_path: Path) -> None:
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_json = meta_path.with_name(meta_path.name + ".tmp")
        tmp_json.write_text(
            json.dumps(self.meta.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        os.replace(tmp_json, meta_path)

        tmp_npz = templates_path.with_name(templates_path.name + ".tmp")
        with tmp_npz.open("wb") as handle:
            np.savez_compressed(
                handle,
                features=self._templates,
                labels=self._labels,
            )
        os.replace(tmp_npz, templates_path)
        logger.info("Calibration sauvegardée dans %s", meta_path.parent)

    @classmethod
    def load(cls, meta_path: Path, templates_path: Path) -> "TemplateLibrary":
        raw = json.loads(meta_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("Fichier de calibration invalide.")
        meta = CalibrationMeta.from_dict(raw)
        if meta.version != CALIBRATION_VERSION:
            raise ValueError("Version de calibration non prise en charge.")
        with np.load(templates_path, allow_pickle=False) as archive:
            features = np.asarray(archive["features"], dtype=np.float32)
            labels = np.asarray(archive["labels"]).astype(str)
        geometry = BoardGeometry(tuple((x, y) for x, y in meta.quad), meta.orientation)
        if geometry.orientation != meta.orientation:
            raise ValueError("Orientation incohérente.")
        return cls(features, labels, meta)
