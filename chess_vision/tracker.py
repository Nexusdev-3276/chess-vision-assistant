"""Keep track of whose turn it is from changes of the visual position.

Screen reading only gives us the piece placement. To query Stockfish we also
need the side to move, castling rights and (for the move immediately after a
double pawn push) the en-passant square. This module maintains a real
python-chess board and advances it with legal moves whenever the observed
placement changes. If the game cannot be followed, it falls back to a visual
resynchronisation and the UI offers a manual turn flip.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import chess

logger = logging.getLogger(__name__)

START_PLACEMENT = chess.Board().board_fen()


def board_from_placement(placement: str, turn: chess.Color) -> chess.Board:
    """Build a playable board from a piece placement.

    Castling rights are inferred conservatively from king/rook home squares.
    En passant cannot be inferred from a single image and is therefore absent.
    """
    probe = chess.Board(None)
    probe.set_board_fen(placement)

    white_king = probe.piece_at(chess.E1)
    black_king = probe.piece_at(chess.E8)
    white_rook_a = probe.piece_at(chess.A1)
    white_rook_h = probe.piece_at(chess.H1)
    black_rook_a = probe.piece_at(chess.A8)
    black_rook_h = probe.piece_at(chess.H8)

    rights: list[str] = []
    if white_king == chess.Piece(chess.KING, chess.WHITE):
        if white_rook_h == chess.Piece(chess.ROOK, chess.WHITE):
            rights.append("K")
        if white_rook_a == chess.Piece(chess.ROOK, chess.WHITE):
            rights.append("Q")
    if black_king == chess.Piece(chess.KING, chess.BLACK):
        if black_rook_h == chess.Piece(chess.ROOK, chess.BLACK):
            rights.append("k")
        if black_rook_a == chess.Piece(chess.ROOK, chess.BLACK):
            rights.append("q")

    turn_letter = "w" if turn == chess.WHITE else "b"
    castling = "".join(rights) if rights else "-"
    fen = f"{placement} {turn_letter} {castling} - 0 1"
    return chess.Board(fen)


@dataclass
class TrackResult:
    board: chess.Board
    status: str  # "same" | "moved" | "new_game" | "resync"
    moves: tuple[chess.Move, ...] = ()
    message: str = ""


class GameTracker:
    def __init__(self, board: chess.Board | None = None) -> None:
        self.board = (board or chess.Board()).copy(stack=False)

    def reset(self, board: chess.Board | None = None) -> None:
        self.board = (board or chess.Board()).copy(stack=False)

    def reset_to_start(self) -> None:
        self.reset(chess.Board())

    def set_turn(self, color: chess.Color) -> None:
        self.board.turn = color
        self.board.ep_square = None

    def flip_turn(self) -> None:
        self.board.turn = not self.board.turn
        self.board.ep_square = None

    def _find_sequence(
        self,
        target_placement: str,
        max_depth: int = 2,
        time_budget: float = 0.60,
    ) -> tuple[chess.Move, ...] | None:
        deadline = time.monotonic() + time_budget

        def search(board: chess.Board, depth: int) -> tuple[chess.Move, ...] | None:
            if time.monotonic() > deadline:
                return None
            for move in board.legal_moves:
                child = board.copy(stack=False)
                child.push(move)
                if child.board_fen() == target_placement:
                    return (move,)
                if depth > 1:
                    sub = search(child, depth - 1)
                    if sub is not None:
                        return (move, *sub)
                if time.monotonic() > deadline:
                    return None
            return None

        return search(self.board.copy(stack=False), max_depth)

    def observe(self, placement: str) -> TrackResult:
        """Advance the tracked board to match an observed piece placement."""
        current_placement = self.board.board_fen()
        if placement == current_placement:
            return TrackResult(board=self.board, status="same")

        if placement == START_PLACEMENT:
            self.reset_to_start()
            return TrackResult(
                board=self.board,
                status="new_game",
                message="Position initiale détectée: nouvelle partie.",
            )

        sequence = self._find_sequence(placement)
        if sequence:
            for move in sequence:
                self.board.push(move)
            return TrackResult(
                board=self.board,
                status="moved",
                moves=sequence,
            )

        # A move was missed or the board was set up manually. Keep the last
        # known side to move when possible; if that makes the position
        # illegal (typically the opponent is left in check), try the other
        # side. The user can always correct it with "Inverser le trait".
        old_turn = self.board.turn
        inferred_turn = old_turn
        resynced = board_from_placement(placement, old_turn)
        if not resynced.is_valid():
            alternate = board_from_placement(placement, not old_turn)
            if alternate.is_valid():
                resynced = alternate
                inferred_turn = not old_turn
                logger.info("Trait déduit de la légalité de la position: %s", inferred_turn)
            else:
                logger.warning(
                    "La position resynchronisée n'est pas parfaitement légale; "
                    "le trait est laissé à vérifier."
                )

        self.board = resynced
        color_name = "Blancs" if inferred_turn == chess.WHITE else "Noirs"
        logger.info("Resynchronisation visuelle, trait supposé: %s", inferred_turn)
        return TrackResult(
            board=self.board,
            status="resync",
            message=(
                "Position resynchronisée sans historique. "
                f"Trait supposé : {color_name}. Vérifiez avec « Inverser le trait » "
                "si nécessaire."
            ),
        )
