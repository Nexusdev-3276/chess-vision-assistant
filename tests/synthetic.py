"""Small deterministic board renderer used by the headless tests.

The real application learns the user's actual theme.  These tests only need a
theme that is visually unambiguous: a coloured board with coordinate labels,
plus token-like pieces whose letter and colour identify them.
"""

from __future__ import annotations

import chess
import cv2
import numpy as np

from chess_vision.geometry import CANVAS_SIZE, CELL_SIZE, BoardGeometry

LIGHT = (210, 238, 238)
DARK = (86, 150, 118)
GOLD = (0, 215, 255)


def _draw_piece(canvas: np.ndarray, piece: chess.Piece, row: int, col: int) -> None:
    cx = col * CELL_SIZE + CELL_SIZE // 2
    cy = row * CELL_SIZE + CELL_SIZE // 2
    white = piece.color == chess.WHITE
    fill = (245, 245, 245) if white else (28, 28, 28)
    ink = (20, 20, 20) if white else (245, 245, 245)

    cv2.circle(canvas, (cx, cy), 34, fill, -1, lineType=cv2.LINE_AA)
    cv2.circle(canvas, (cx, cy), 34, ink, 3, lineType=cv2.LINE_AA)

    letter = piece.symbol().upper()
    scale, thickness = 1.25, 3
    (text_w, text_h), _ = cv2.getTextSize(
        letter, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness
    )
    origin = (cx - text_w // 2, cy + text_h // 2)
    cv2.putText(
        canvas,
        letter,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        ink,
        thickness,
        cv2.LINE_AA,
    )


def _draw_labels(canvas: np.ndarray, orientation: str) -> None:
    for row in range(8):
        for col in range(8):
            square = BoardGeometry.cell_to_square(row, col, orientation)
            name = chess.square_name(square)
            ink = (70, 70, 70) if (row + col) % 2 == 0 else (225, 225, 225)
            cv2.putText(
                canvas,
                name[0].upper(),
                (col * CELL_SIZE + 3, row * CELL_SIZE + 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.28,
                ink,
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                name[1],
                (col * CELL_SIZE + CELL_SIZE - 9, row * CELL_SIZE + CELL_SIZE - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.28,
                ink,
                1,
                cv2.LINE_AA,
            )


def render_board(
    board: chess.Board,
    orientation: str,
    last_move: chess.Move | None = None,
) -> np.ndarray:
    """Render a board as an 800x800 BGR image."""
    canvas = np.empty((CANVAS_SIZE, CANVAS_SIZE, 3), dtype=np.uint8)
    for row in range(8):
        for col in range(8):
            color = LIGHT if (row + col) % 2 == 0 else DARK
            canvas[
                row * CELL_SIZE : (row + 1) * CELL_SIZE,
                col * CELL_SIZE : (col + 1) * CELL_SIZE,
            ] = color

    _draw_labels(canvas, orientation)

    if last_move is not None:
        for square in (last_move.from_square, last_move.to_square):
            row, col = BoardGeometry.square_to_cell(square, orientation)
            y = slice(row * CELL_SIZE, (row + 1) * CELL_SIZE)
            x = slice(col * CELL_SIZE, (col + 1) * CELL_SIZE)
            cell = canvas[y, x]
            tint = np.full_like(cell, GOLD)
            canvas[y, x] = cv2.addWeighted(cell, 0.62, tint, 0.38, 0)

    for square, piece in board.piece_map().items():
        row, col = BoardGeometry.square_to_cell(square, orientation)
        _draw_piece(canvas, piece, row, col)
    return canvas
