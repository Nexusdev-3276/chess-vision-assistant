"""Application configuration and on-disk locations.

The tool stores its settings and the learned piece templates in the user's
local application-data directory, never next to the game.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
from dataclasses import asdict, dataclass, fields
from pathlib import Path

APP_ID = "ChessVisionAssistant"
APP_NAME = "Chess Vision"
VERSION = "1.0.0"

logger = logging.getLogger(__name__)


def _data_dir() -> Path:
    if sys.platform == "win32":
        root = os.environ.get("LOCALAPPDATA")
        base = Path(root) if root else Path.home() / "AppData" / "Local"
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        root = os.environ.get("XDG_DATA_HOME")
        base = Path(root) if root else Path.home() / ".local" / "share"
    return base / APP_ID


DATA_DIR = _data_dir()
CONFIG_PATH = DATA_DIR / "config.json"
CALIBRATION_PATH = DATA_DIR / "calibration.json"
TEMPLATES_PATH = DATA_DIR / "templates.npz"
PREVIEW_PATH = DATA_DIR / "calibration_preview.png"


def ensure_data_dir() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR


@dataclass
class AppConfig:
    """User-editable settings (persisted as JSON)."""

    stockfish_path: str = ""
    movetime_ms: int = 700
    poll_ms: int = 600
    confirmations: int = 2
    advice_mode: str = "human"  # "human" or "always"
    human_color: str = "white"  # "white" or "black"
    show_label: bool = True
    arrow_color: str = "#FF8A00"
    arrow_opacity: int = 205
    engine_threads: int = 2
    engine_hash_mb: int = 128
    min_score: float = 0.55
    hide_on_start: bool = True

    def normalized(self) -> "AppConfig":
        """Return a copy with values clamped to sane ranges."""
        cfg = AppConfig(**asdict(self))
        try:
            cfg.movetime_ms = max(100, min(10_000, int(cfg.movetime_ms)))
        except (TypeError, ValueError):
            cfg.movetime_ms = 700
        try:
            cfg.poll_ms = max(250, min(5_000, int(cfg.poll_ms)))
        except (TypeError, ValueError):
            cfg.poll_ms = 600
        try:
            cfg.confirmations = max(1, min(5, int(cfg.confirmations)))
        except (TypeError, ValueError):
            cfg.confirmations = 2
        try:
            cfg.arrow_opacity = max(40, min(255, int(cfg.arrow_opacity)))
        except (TypeError, ValueError):
            cfg.arrow_opacity = 205
        try:
            cfg.engine_threads = max(1, min(32, int(cfg.engine_threads)))
        except (TypeError, ValueError):
            cfg.engine_threads = 2
        try:
            cfg.engine_hash_mb = max(16, min(4096, int(cfg.engine_hash_mb)))
        except (TypeError, ValueError):
            cfg.engine_hash_mb = 128
        try:
            cfg.min_score = max(0.1, min(0.99, float(cfg.min_score)))
        except (TypeError, ValueError):
            cfg.min_score = 0.55

        if cfg.advice_mode not in {"human", "always"}:
            cfg.advice_mode = "human"
        if cfg.human_color not in {"white", "black"}:
            cfg.human_color = "white"
        if not cfg.stockfish_path:
            cfg.stockfish_path = ""
        if not cfg.arrow_color.startswith("#"):
            cfg.arrow_color = "#FF8A00"
        cfg.show_label = bool(cfg.show_label)
        cfg.hide_on_start = bool(cfg.hide_on_start)
        return cfg

    @classmethod
    def load(cls, path: Path | None = None) -> "AppConfig":
        path = path or CONFIG_PATH
        if not path.is_file():
            return cls()
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ValueError("config root must be an object")
            known = {f.name for f in fields(cls)}
            values = {k: v for k, v in raw.items() if k in known}
            return cls(**values).normalized()
        except Exception:
            logger.exception("Impossible de lire la configuration %s", path)
            return cls()

    def save(self, path: Path | None = None) -> None:
        path = path or CONFIG_PATH
        ensure_data_dir()
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(
            json.dumps(asdict(self.normalized()), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        os.replace(tmp, path)


def _add_candidate(candidates: list[Path], value: str | os.PathLike[str] | None) -> None:
    if not value:
        return
    raw = os.path.expandvars(os.path.expanduser(str(value))).strip().strip('"')
    if not raw:
        return
    candidate = Path(raw)
    if candidate.is_file():
        candidates.append(candidate)
        return
    located = shutil.which(raw)
    if located:
        candidates.append(Path(located))


def find_stockfish(preferred: str = "") -> Path | None:
    """Locate a Stockfish executable without requiring extra packages.

    The explicit path (or STOCKFISH_PATH) wins, then PATH, then common install
    locations and a few sensible locations next to the project.
    """
    candidates: list[Path] = []
    _add_candidate(candidates, os.environ.get("STOCKFISH_PATH"))
    _add_candidate(candidates, preferred)
    _add_candidate(candidates, "stockfish")

    if sys.platform == "win32":
        project_root = Path(__file__).resolve().parent.parent
        folders = [
            Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Stockfish",
            Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / "Stockfish",
            Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "Programs" / "Stockfish",
            Path.cwd(),
            Path.cwd() / "stockfish",
            project_root,
            project_root / "stockfish",
            project_root / "bin",
        ]
        for folder in folders:
            try:
                if folder.is_dir():
                    candidates.extend(sorted(folder.glob("stockfish*.exe")))
            except OSError:
                pass
    else:
        for raw in (
            "/usr/games/stockfish",
            "/usr/bin/stockfish",
            "/usr/local/bin/stockfish",
            "/opt/homebrew/bin/stockfish",
            "/Applications/Stockfish.app/Contents/MacOS/stockfish",
        ):
            _add_candidate(candidates, raw)

    seen: set[str] = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            resolved = candidate
        key = str(resolved).lower()
        if key in seen:
            continue
        seen.add(key)
        if resolved.is_file():
            return resolved
    return None
