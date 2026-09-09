"""Overlay coloring helpers.

Two overlay modes, both rendered as RGBA so that background/zero voxels are
*fully transparent* (alpha 0), not merely low-alpha:

* ``all_green``  -- every selected island drawn in a single semi-transparent
  green (like the old ``mito_cmap``).
* ``color_by_number`` -- each island id mapped to a distinct, randomized color
  so neighbouring islands are easy to tell apart.

The functions take a 2D label slice (0 = background) and a set of selected ids,
and return an ``(H, W, 4)`` float RGBA image in 0..1.
"""

from __future__ import annotations

from typing import Iterable, Set

import numpy as np
import colorsys

# Default overlay opacity for selected voxels.
_ALPHA = 0.5
_GREEN = (0.0, 1.0, 0.0)
_YELLOW = (1.0, 1.0, 0.0)


def _selected_mask(label_slice: np.ndarray, selected: Set[int]) -> np.ndarray:
    """Boolean mask of pixels whose label id is in ``selected`` (and non-zero)."""
    if not selected:
        return np.zeros(label_slice.shape, dtype=bool)
    return np.isin(label_slice, np.fromiter(selected, dtype=np.int64)) & (label_slice > 0)


def build_label_colors(n: int, seed: int = 0) -> np.ndarray:
    """Return an ``(n+1, 3)`` RGB table for label ids 0..n.

    Index 0 is black (background, never shown). Ids 1..n get evenly spaced,
    shuffled HSV hues so distinct islands look distinct. The shuffle is
    deterministic for a given ``(n, seed)`` so colors are stable within a run.
    """
    colors = np.zeros((n + 1, 3), dtype=float)
    if n <= 0:
        return colors
    rng = np.random.default_rng(seed)
    hues = (np.arange(n) / max(n, 1)) % 1.0
    rng.shuffle(hues)
    for i, h in enumerate(hues, start=1):
        # High saturation/value so overlays stay vivid over grayscale tomogram.
        colors[i] = colorsys.hsv_to_rgb(float(h), 0.85, 1.0)
    return colors


def green_overlay(label_slice: np.ndarray, selected: Set[int], alpha: float = _ALPHA) -> np.ndarray:
    """RGBA overlay: all selected islands in green, background transparent."""
    h, w = label_slice.shape
    rgba = np.zeros((h, w, 4), dtype=float)
    mask = _selected_mask(label_slice, selected)
    rgba[mask, 0] = _GREEN[0]
    rgba[mask, 1] = _GREEN[1]
    rgba[mask, 2] = _GREEN[2]
    rgba[mask, 3] = alpha
    return rgba


def numbered_overlay(
    label_slice: np.ndarray,
    selected: Set[int],
    label_colors: np.ndarray,
    alpha: float = _ALPHA,
) -> np.ndarray:
    """RGBA overlay: each selected island colored by ``label_colors[id]``."""
    h, w = label_slice.shape
    rgba = np.zeros((h, w, 4), dtype=float)
    mask = _selected_mask(label_slice, selected)
    if mask.any():
        ids = label_slice[mask]
        # Guard against ids beyond the current color table (after relabeling).
        ids = np.clip(ids, 0, len(label_colors) - 1)
        rgba[mask, :3] = label_colors[ids]
        rgba[mask, 3] = alpha
    return rgba


def highlight_overlay(label_slice: np.ndarray, highlight_id: int, alpha: float = 0.85) -> np.ndarray:
    """RGBA overlay drawing only ``highlight_id`` in yellow; rest transparent."""
    h, w = label_slice.shape
    rgba = np.zeros((h, w, 4), dtype=float)
    if highlight_id is None or highlight_id <= 0:
        return rgba
    mask = label_slice == highlight_id
    rgba[mask, 0] = _YELLOW[0]
    rgba[mask, 1] = _YELLOW[1]
    rgba[mask, 2] = _YELLOW[2]
    rgba[mask, 3] = alpha
    return rgba
