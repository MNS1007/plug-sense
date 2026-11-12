"""Reads the power bank's display from the front camera: lit means the plug is latched and charging.

The display shows nothing until the plug clicks in, then glows red (white a few seconds later),
and stays red for 7 to 10 seconds after unplugging. The reading is the share of very bright
pixels inside a box around the display, so it works for red and for white.
"""
from __future__ import annotations

import numpy as np

BOX = (590, 480, 710, 715)   # x0, y0, x1, y1 in the 1280x720 front picture, around the display
BRIGHT, LIT_SHARE = 215, 0.04


def share(frame_bgr: np.ndarray, box=BOX) -> float:
    x0, y0, x1, y1 = box
    return float((frame_bgr[y0:y1, x0:x1].max(axis=2) >= BRIGHT).mean())


def lit(frame_bgr: np.ndarray, box=BOX, dark_share: float = 0.0) -> bool:
    """dark_share: the reading taken just before the push, to cancel glare or the battery light."""
    return share(frame_bgr, box) - dark_share >= LIT_SHARE
