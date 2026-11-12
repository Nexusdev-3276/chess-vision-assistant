"""Board geometry: four-corner calibration, perspective warp and square mapping."""

from __future__ import annotations

from dataclasses import dataclass

import chess
import cv2
import numpy as np

CANVAS_SIZE = 800
CELL_SIZE = CANVAS_SIZE // 8
ORIENTATION_WHITE = "white_bottom"
ORIENTATION_BLACK = "black_bottom"
VALID_ORIENTATIONS = (ORIENTATION_WHITE, ORIENTATION_BLACK)


@dataclass
class BoardGeometry:
    """Maps the screen board quad to a flat 800x800 chessboard image.

    Quad order is always: top-left, top-right, bottom-right, bottom-left in
    screen/view coordinates. Orientation describes which colour is at the
    bottom of the screen.
    """

    quad: tuple[tuple[float, float], ...]
    orientation: str = ORIENTATION_WHITE

    def __post_init__(self) -> None:
        if len(self.quad) != 4:
            raise ValueError("Le plateau doit avoir exactement 4 coins.")
        points = np.asarray(self.quad, dtype=np.float32)
        if points.shape != (4, 2) or not np.all(np.isfinite(points)):
            raise ValueError("Coordonnées de plateau invalides.")
        if self.orientation not in VALID_ORIENTATIONS:
            raise ValueError("Orientation de plateau inconnue.")
        if not cv2.isContourConvex(points.reshape(-1, 1, 2)):
            raise ValueError(
                "Les 4 coins doivent être cliqués dans le sens horaire "
                "(haut-gauche, haut-droit, bas-droit, bas-gauche)."
            )
        if abs(cv2.contourArea(points)) < 10_000:
            raise ValueError("La zone du plateau est trop petite ou déformée.")
        # In screen coordinates (y grows downwards) a clockwise quad has a
        # positive shoelace area.  Rejecting a reversed quad prevents a
        # mirrored board that would silently swap files.
        x = points[:, 0]
        y = points[:, 1]
        signed_area = 0.5 * float(
            np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
        )
        if signed_area <= 0:
            raise ValueError(
                "Les 4 coins doivent être cliqués dans le sens horaire: "
                "haut-gauche, haut-droit, bas-droit, bas-gauche."
            )

        self.quad = tuple((float(x), float(y)) for x, y in points)
        dst = np.array(
            [[0, 0], [CANVAS_SIZE, 0], [CANVAS_SIZE, CANVAS_SIZE], [0, CANVAS_SIZE]],
            dtype=np.float32,
        )
        self._matrix = cv2.getPerspectiveTransform(points, dst)
        if not np.all(np.isfinite(self._matrix)):
            raise ValueError("Impossible de calculer la perspective du plateau.")
        try:
            self._inverse = np.linalg.inv(self._matrix)
        except np.linalg.LinAlgError as exc:  # pragma: no cover - degenerate quad
            raise ValueError("Impossible de calculer la perspective du plateau.") from exc

    def warp(self, frame: np.ndarray) -> np.ndarray:
        """Return the board rectified to CANVAS_SIZE x CANVAS_SIZE."""
        if frame is None or frame.size == 0:
            raise ValueError("Image d'écran vide.")
        warped = cv2.warpPerspective(
            frame,
            self._matrix,
            (CANVAS_SIZE, CANVAS_SIZE),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        return warped

    @staticmethod
    def square_to_cell(square: int, orientation: str) -> tuple[int, int]:
        """Return the visual (row, column) of a chess square."""
        file_index = chess.square_file(square)
        rank_index = chess.square_rank(square)
        if orientation == ORIENTATION_WHITE:
            return 7 - rank_index, file_index
        return rank_index, 7 - file_index

    @staticmethod
    def cell_to_square(row: int, col: int, orientation: str) -> int:
        """Return the chess square at a visual (row, column)."""
        if not (0 <= row < 8 and 0 <= col < 8):
            raise ValueError("Case visuelle hors plateau.")
        if orientation == ORIENTATION_WHITE:
            file_index, rank_index = col, 7 - row
        else:
            file_index, rank_index = 7 - col, row
        return chess.square(file_index, rank_index)

    def cell_center_in_frame(self, row: int, col: int) -> tuple[float, float]:
        """Map a visual cell centre back to screen/frame coordinates."""
        if not (0 <= row < 8 and 0 <= col < 8):
            raise ValueError("Case visuelle hors plateau.")
        point = np.array(
            [[[(col + 0.5) * CELL_SIZE, (row + 0.5) * CELL_SIZE]]], dtype=np.float32
        )
        mapped = cv2.perspectiveTransform(point, self._inverse)
        return float(mapped[0, 0, 0]), float(mapped[0, 0, 1])

    def square_center_in_frame(self, square: int) -> tuple[float, float]:
        row, col = self.square_to_cell(square, self.orientation)
        return self.cell_center_in_frame(row, col)

    def cell_rectangle_in_frame(self, row: int, col: int) -> np.ndarray:
        """Return the four screen points of a visual cell."""
        corners = np.array(
            [
                [[col * CELL_SIZE, row * CELL_SIZE]],
                [[(col + 1) * CELL_SIZE, row * CELL_SIZE]],
                [[(col + 1) * CELL_SIZE, (row + 1) * CELL_SIZE]],
                [[col * CELL_SIZE, (row + 1) * CELL_SIZE]],
            ],
            dtype=np.float32,
        )
        return cv2.perspectiveTransform(corners, self._inverse).reshape(4, 2)
