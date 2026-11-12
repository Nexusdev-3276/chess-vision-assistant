from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from chess_vision.config import AppConfig, find_stockfish


class ConfigTests(unittest.TestCase):
    def test_values_are_clamped(self) -> None:
        config = AppConfig(
            movetime_ms=1,
            poll_ms=99,
            confirmations=50,
            arrow_opacity=2,
            engine_threads=0,
            engine_hash_mb=1,
            min_score=5.0,
            advice_mode="nonsense",
            human_color="purple",
        ).normalized()
        self.assertEqual(config.movetime_ms, 100)
        self.assertEqual(config.poll_ms, 250)
        self.assertEqual(config.confirmations, 5)
        self.assertEqual(config.arrow_opacity, 40)
        self.assertEqual(config.engine_threads, 1)
        self.assertEqual(config.engine_hash_mb, 16)
        self.assertEqual(config.min_score, 0.99)
        self.assertEqual(config.advice_mode, "human")
        self.assertEqual(config.human_color, "white")

    def test_save_and_load_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            original = AppConfig(
                stockfish_path="C:/Tools/stockfish.exe",
                movetime_ms=900,
                poll_ms=500,
                advice_mode="always",
                human_color="black",
                show_label=False,
            )
            original.save(path)
            loaded = AppConfig.load(path)
            self.assertEqual(loaded.stockfish_path, original.stockfish_path)
            self.assertEqual(loaded.movetime_ms, 900)
            self.assertEqual(loaded.poll_ms, 500)
            self.assertEqual(loaded.advice_mode, "always")
            self.assertEqual(loaded.human_color, "black")
            self.assertFalse(loaded.show_label)

    def test_explicit_stockfish_file_wins(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fake = Path(tmp) / "stockfish.exe"
            fake.write_bytes(b"")
            self.assertEqual(find_stockfish(str(fake)).resolve(), fake.resolve())


if __name__ == "__main__":
    unittest.main()
