"""Pure volume logic for ``pick-granules``.

Granules are the voxels inside a binary mask (e.g. a mitochondrion
segmentation) whose tomogram value is below a threshold (dark, the default) or
above it (``bright=True``). Volumes are numpy arrays in ``(Z, Y, X)`` order, as
in ``curate``.

No matplotlib here: everything is importable and testable headlessly.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..segmentation_curator import labeling
from ..segmentation_eroder.core import erode


class GranuleError(ValueError):
    """Bad parameters or input for the granule picker."""


def prepare_mask(mask: np.ndarray, erode_px: int = 0, three_d: bool = False) -> np.ndarray:
    """Binarize ``mask`` (any value > 0) and optionally erode it.

    Eroding pulls the region away from the mask boundary, where membranes are
    as dark as granules and would otherwise be picked. XY by default (each Z
    slice in-plane); ``three_d=True`` also erodes the top and bottom in Z.
    """
    m = np.asarray(mask) > 0
    if erode_px and erode_px > 0:
        m = erode(m, iterations=int(erode_px), three_d=bool(three_d))
    return m


def blur(volume: np.ndarray, sigma: float) -> np.ndarray:
    """3D Gaussian blur (sigma in voxels); ``sigma <= 0`` only casts to float32."""
    vol = np.asarray(volume, dtype=np.float32)
    if sigma and sigma > 0:
        from scipy.ndimage import gaussian_filter

        vol = gaussian_filter(vol, sigma=float(sigma))
    return vol


def threshold_granules(tomo: np.ndarray, mask: np.ndarray, threshold: float,
                       bright: bool = False) -> np.ndarray:
    """Boolean volume: voxels inside ``mask`` darker than ``threshold``.

    With ``bright=True`` the voxels brighter than ``threshold`` are picked.
    Works on 2D slices too.
    """
    tomo = np.asarray(tomo)
    mask = np.asarray(mask, dtype=bool)
    if tomo.shape != mask.shape:
        raise GranuleError(f"tomogram and mask shapes differ: {tomo.shape} vs {mask.shape}")
    hit = tomo > threshold if bright else tomo < threshold
    return hit & mask


def remove_small(binary: np.ndarray, min_size: int, connectivity: int = 26) -> np.ndarray:
    """Drop connected islands with fewer than ``min_size`` voxels."""
    binary = np.asarray(binary, dtype=bool)
    if not min_size or min_size <= 1 or not binary.any():
        return binary
    labels, n = labeling.label_islands(binary, connectivity=connectivity)
    sizes = np.bincount(labels.ravel(), minlength=n + 1)
    keep = sizes >= min_size
    keep[0] = False
    return keep[labels]


def masked_values(tomo: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Sorted tomogram values inside ``mask`` (for percentiles and live counts)."""
    vals = np.asarray(tomo)[np.asarray(mask, dtype=bool)]
    return np.sort(vals.astype(np.float32, copy=False))


def threshold_from_percentile(sorted_vals: np.ndarray, percentile: float) -> float:
    """Value at ``percentile`` (0-100) of the masked values."""
    if sorted_vals.size == 0:
        raise GranuleError("the mask is empty, nothing to threshold")
    if not 0 <= percentile <= 100:
        raise GranuleError("percentile must be between 0 and 100")
    return float(np.percentile(sorted_vals, percentile))


def count_picked(sorted_vals: np.ndarray, threshold: float, bright: bool = False) -> int:
    """Number of masked voxels that ``threshold`` picks (O(log n) on sorted values)."""
    if bright:
        return int(sorted_vals.size - np.searchsorted(sorted_vals, threshold, side="right"))
    return int(np.searchsorted(sorted_vals, threshold, side="left"))


def percentile_of(sorted_vals: np.ndarray, threshold: float) -> float:
    """Percent of masked voxels below ``threshold``."""
    if sorted_vals.size == 0:
        return 0.0
    return 100.0 * np.searchsorted(sorted_vals, threshold, side="left") / sorted_vals.size


def pick(tomo: np.ndarray, mask: np.ndarray, threshold: float, bright: bool = False,
         min_size: int = 0, connectivity: int = 26) -> np.ndarray:
    """Threshold + small-island removal. Returns a uint8 0/1 volume."""
    out = threshold_granules(tomo, mask, threshold, bright=bright)
    out = remove_small(out, min_size, connectivity=connectivity)
    return out.astype(np.uint8)


# ---------------------------------------------------------------------------
# MRC I/O (voxel size and origin are carried over to the output).
# ---------------------------------------------------------------------------
def read_volume(path: str):
    """Return ``(data, voxel_size, origin)`` from an MRC file."""
    import mrcfile

    with mrcfile.open(path, permissive=True) as mrc:
        if mrc.data is None:
            raise GranuleError(f"MRC file {path!r} contains no data")
        data = np.asarray(mrc.data).copy()
        voxel_size = mrc.voxel_size.copy()
        origin = mrc.header.origin.copy()
    return data, voxel_size, origin


def write_mask(path: str, mask: np.ndarray, voxel_size=None, origin=None) -> None:
    """Write ``mask`` as a uint8 0/1 MRC (like ``curate``)."""
    import mrcfile

    data = np.ascontiguousarray(mask, dtype=np.uint8)
    with mrcfile.new(path, overwrite=True) as mrc:
        mrc.set_data(data)
        if voxel_size is not None:
            mrc.voxel_size = voxel_size
        if origin is not None:
            mrc.header.origin = origin
        mrc.update_header_from_data()
        mrc.update_header_stats()


def initial_threshold(sorted_vals: np.ndarray, threshold: Optional[float],
                      percentile: Optional[float], default_percentile: float = 5.0) -> float:
    """Resolve the starting threshold from ``--threshold`` / ``--percentile``."""
    if threshold is not None and percentile is not None:
        raise GranuleError("give --threshold or --percentile, not both")
    if threshold is not None:
        return float(threshold)
    return threshold_from_percentile(
        sorted_vals, default_percentile if percentile is None else percentile)
