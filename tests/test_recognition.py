from __future__ import annotations

import unittest

import chess

from chess_vision.geometry import (
    ORIENTATION_BLACK,
    ORIENTATION_WHITE,
    BoardGeometry,
)
from chess_vision.recognition import TemplateLibrary
from tests.synthetic import render_board

SIMPLE_QUAD = ((0.0, 0.0), (800.0, 0.0), (800.0, 800.0), (0.0, 800.0))


def board_after(moves: str) -> chess.Board:
    board = chess.Board()
    for uci in moves.split():
        board.push_uci(uci)
    return board


def last_move(board: chess.Board) -> chess.Move | None:
    return board.peek() if board.move_stack else None


class RecognitionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.start = chess.Board()
        cls.libraries: dict[str, TemplateLibrary] = {}
        for orientation in (ORIENTATION_WHITE, ORIENTATION_BLACK):
            geometry = BoardGeometry(SIMPLE_QUAD, orientation)
            image = render_board(cls.start, orientation)
            cls.libraries[orientation] = TemplateLibrary.calibrate(image, geometry)

    def assert_board_detected(
        self, board: chess.Board, orientation: str = ORIENTATION_WHITE
    ) -> None:
        image = render_board(board, orientation, last_move(board))
        result = self.libraries[orientation].recognize(image)
        self.assertTrue(result.ok, result.error)
        self.assertEqual(result.placement, board.board_fen())
        self.assertGreater(result.confidence, 0.55)

    def test_starting_position(self) -> None:
        self.assert_board_detected(self.start)

    def test_opening_position_with_last_move_highlight(self) -> None:
        board = board_after(
            "e2e4 e7e5 g1f3 b8c6 f1b5 a7a6 b5a4 g8f6 e1g1"
        )
        self.assert_board_detected(board)

    def test_capture_and_empty_squares(self) -> None:
        board = board_after("e2e4 d7d5 e4d5 d8d5 b1c3")
        self.assert_board_detected(board, ORIENTATION_WHITE)

    def test_black_at_bottom(self) -> None:
        board = board_after("d2d4 d7d5 c2c4 e7e6 b1c3 g8f6")
        self.assert_board_detected(board, ORIENTATION_BLACK)

    def test_missing_king_is_rejected(self) -> None:
        board = self.start.copy(stack=False)
        board.remove_piece_at(chess.E8)
        image = render_board(board, ORIENTATION_WHITE)
        result = self.libraries[ORIENTATION_WHITE].recognize(image)
        self.assertFalse(result.ok)
        self.assertIn("roi", result.error.lower())


if __name__ == "__main__":
    unittest.main()
