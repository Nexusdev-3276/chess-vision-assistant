"""Fast access to the primary monitor through mss."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import mss
import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MonitorInfo:
    left: int
    top: int
    width: int
    height: int


class ScreenCapture:
    """Persistent mss capture object returning OpenCV BGR frames.

    Only the primary monitor is used. The calibration screenshot and the live
    frames therefore share the same coordinate system.
    """

    def __init__(self, monitor_index: int = 1) -> None:
        self._sct = mss.mss()
        monitors = self._sct.monitors
        if len(monitors) <= 1:
            raise RuntimeError("Aucun écran n'a été détecté.")
        if monitor_index < 1 or monitor_index >= len(monitors):
            monitor_index = 1 if len(monitors) > 1 else 0
        self.monitor_index = monitor_index
        m = monitors[monitor_index]
        self.monitor = MonitorInfo(
            left=int(m["left"]),
            top=int(m["top"]),
            width=int(m["width"]),
            height=int(m["height"]),
        )
        logger.info("Capture de l'écran principal: %s", self.monitor)

    @property
    def size(self) -> tuple[int, int]:
        return self.monitor.width, self.monitor.height

    def grab(self) -> np.ndarray:
        """Capture the monitor and return a writable BGR array."""
        raw = np.asarray(self._sct.grab(self._sct.monitors[self.monitor_index]))
        return cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)

    def close(self) -> None:
        try:
            self._sct.close()
        except Exception:
            logger.debug("Fermeture de mss ignorée", exc_info=True)
