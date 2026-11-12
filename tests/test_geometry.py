from __future__ import annotations

import unittest

import chess
import cv2
import numpy as np

from chess_vision.geometry import (
    CANVAS_SIZE,
    CELL_SIZE,
    ORIENTATION_BLACK,
    ORIENTATION_WHITE,
    BoardGeometry,
)

PERSPECTIVE_QUAD = (
    (120.0, 80.0),
    (1030.0, 145.0),
    (980.0, 790.0),
    (80.0, 710.0),
)


class OrientationTests(unittest.TestCase):
    def test_white_bottom_corners(self) -> None:
        self.assertEqual(
            BoardGeometry.cell_to_square(0, 0, ORIENTATION_WHITE), chess.A8
        )
        self.assertEqual(
            BoardGeometry.cell_to_square(0, 7, ORIENTATION_WHITE), chess.H8
        )
        self.assertEqual(
            BoardGeometry.cell_to_square(7, 0, ORIENTATION_WHITE), chess.A1
        )
        self.assertEqual(
            BoardGeometry.cell_to_square(7, 7, ORIENTATION_WHITE), chess.H1
        )

    def test_black_bottom_corners(self) -> None:
        self.assertEqual(
            BoardGeometry.cell_to_square(0, 0, ORIENTATION_BLACK), chess.H1
        )
        self.assertEqual(
            BoardGeometry.cell_to_square(0, 7, ORIENTATION_BLACK), chess.A1
        )
        self.assertEqual(
            BoardGeometry.cell_to_square(7, 0, ORIENTATION_BLACK), chess.H8
        )
        self.assertEqual(
            BoardGeometry.cell_to_square(7, 7, ORIENTATION_BLACK), chess.A8
        )

    def test_round_trip_every_square(self) -> None:
        for orientation in (ORIENTATION_WHITE, ORIENTATION_BLACK):
            for square in chess.SQUARES:
                row, col = BoardGeometry.square_to_cell(square, orientation)
                self.assertEqual(
                    BoardGeometry.cell_to_square(row, col, orientation), square
                )


class WarpTests(unittest.TestCase):
    def test_homography_round_trip(self) -> None:
        geometry = BoardGeometry(PERSPECTIVE_QUAD, ORIENTATION_WHITE)
        canvas = np.zeros((900, 1200, 3), dtype=np.uint8)
        expected: dict[tuple[int, int], tuple[int, int, int]] = {}
        for row in range(8):
            for col in range(8):
                color = (
                    30 + row * 25,
                    20 + col * 27,
                    30 + (row * 8 + col) * 3,
                )
                polygon = geometry.cell_rectangle_in_frame(row, col)
                cv2.fillConvexPoly(
                    canvas, np.round(polygon).astype(np.int32), color
                )
                expected[(row, col)] = color

        warped = geometry.warp(canvas)
        self.assertEqual(warped.shape, (CANVAS_SIZE, CANVAS_SIZE, 3))
        for (row, col), color in expected.items():
            center = (
                col * CELL_SIZE + CELL_SIZE // 2,
                row * CELL_SIZE + CELL_SIZE // 2,
            )
            actual = warped[center[1], center[0]]
            np.testing.assert_allclose(actual, color, atol=3)

    def test_square_centres_are_inside_the_quad(self) -> None:
        geometry = BoardGeometry(PERSPECTIVE_QUAD, ORIENTATION_WHITE)
        for square in (chess.A1, chess.H1, chess.A8, chess.H8, chess.E4):
            x, y = geometry.square_center_in_frame(square)
            self.assertGreater(x, 0)
            self.assertGreater(y, 0)
        e1 = geometry.square_center_in_frame(chess.E1)
        e8 = geometry.square_center_in_frame(chess.E8)
        self.assertGreater(e1[1], e8[1])


class ValidationTests(unittest.TestCase):
    def test_reversed_order_is_rejected(self) -> None:
        reversed_quad = tuple(reversed(PERSPECTIVE_QUAD))
        with self.assertRaises(ValueError):
            BoardGeometry(reversed_quad, ORIENTATION_WHITE)

    def test_tiny_area_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            BoardGeometry(((0, 0), (30, 0), (30, 30), (0, 30)))

    def test_non_convex_quad_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            BoardGeometry(((0, 0), (100, 0), (40, 40), (0, 100)))

    def test_unknown_orientation_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            BoardGeometry(PERSPECTIVE_QUAD, "sideways")


if __name__ == "__main__":
    unittest.main()
