from __future__ import annotations

import unittest

import chess

from chess_vision.tracker import START_PLACEMENT, GameTracker, board_from_placement


class TrackerTests(unittest.TestCase):
    def test_initial_position_is_recognised(self) -> None:
        tracker = GameTracker()
        result = tracker.observe(START_PLACEMENT)
        self.assertEqual(result.status, "same")
        self.assertEqual(tracker.board.fen(), chess.Board().fen())

    def test_legal_moves_are_followed(self) -> None:
        tracker = GameTracker()
        board = chess.Board()
        for uci in ("e2e4", "e7e5", "g1f3", "b8c6"):
            board.push_uci(uci)
            result = tracker.observe(board.board_fen())
            self.assertEqual(result.status, "moved")
            self.assertEqual(result.moves, (chess.Move.from_uci(uci),))
            self.assertEqual(tracker.board.board_fen(), board.board_fen())
        self.assertEqual(tracker.board.turn, chess.WHITE)

    def test_castling_is_followed(self) -> None:
        tracker = GameTracker()
        board = chess.Board()
        moves = (
            "e2e4 e7e5 g1f3 b8c6 f1b5 a7a6 b5a4 g8f6 e1g1"
        ).split()
        for uci in moves:
            board.push_uci(uci)
            result = tracker.observe(board.board_fen())
            self.assertEqual(result.status, "moved", uci)
        self.assertEqual(tracker.board.board_fen(), board.board_fen())
        self.assertTrue(tracker.board.has_castling_rights(chess.BLACK))
        self.assertFalse(tracker.board.has_castling_rights(chess.WHITE))

    def test_new_game_resets_the_tracker(self) -> None:
        tracker = GameTracker()
        board = chess.Board()
        board.push_uci("e2e4")
        tracker.observe(board.board_fen())
        result = tracker.observe(START_PLACEMENT)
        self.assertEqual(result.status, "new_game")
        self.assertEqual(tracker.board.fen(), chess.Board().fen())

    def test_resynchronisation_after_missed_moves(self) -> None:
        board = chess.Board()
        for uci in ("e2e4", "e7e5", "g1f3"):
            board.push_uci(uci)
        tracker = GameTracker()
        result = tracker.observe(board.board_fen())
        self.assertEqual(result.status, "resync")
        self.assertEqual(tracker.board.board_fen(), board.board_fen())
        self.assertIn("Position resynchronisée", result.message)

    def test_resync_infers_turn_from_legality(self) -> None:
        # With White to move, Black would be left in check, which is illegal.
        # The resynchronised board must therefore give the move to Black.
        placement = "4Q2k/8/8/8/8/8/8/K7"
        tracker = GameTracker()
        result = tracker.observe(placement)
        self.assertEqual(result.status, "resync")
        self.assertEqual(tracker.board.turn, chess.BLACK)
        self.assertTrue(tracker.board.is_valid(), "position should be legal")

    def test_flip_turn(self) -> None:
        tracker = GameTracker()
        tracker.flip_turn()
        self.assertEqual(tracker.board.turn, chess.BLACK)
        tracker.flip_turn()
        self.assertEqual(tracker.board.turn, chess.WHITE)

    def test_castling_rights_are_inferred_conservatively(self) -> None:
        board = board_from_placement(START_PLACEMENT, chess.WHITE)
        self.assertEqual(board.castling_xfen(), "KQkq")

        placement = board.board_fen().replace("R", "1", 1)
        without_a1_rook = board_from_placement(placement, chess.WHITE)
        self.assertNotIn("Q", without_a1_rook.castling_xfen())


if __name__ == "__main__":
    unittest.main()
