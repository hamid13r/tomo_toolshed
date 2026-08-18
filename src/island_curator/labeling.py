"""Pure volume logic for island curation.

No matplotlib imports here — everything in this module is unit-testable
headlessly. Volumes are numpy arrays with axis order (Z, Y, X).
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np
import cc3d
from scipy import ndimage as ndi
from skimage.morphology import ball


def label_islands(binary: np.ndarray, connectivity: int = 26) -> Tuple[np.ndarray, int]:
    """Label connected components ("islands") in a binary volume.

    Parameters
    ----------
    binary : ndarray
        Boolean / integer volume; anything > 0 is treated as foreground.
    connectivity : {6, 18, 26}
        3D connectivity passed to cc3d.

    Returns
    -------
    (labels, n) : (ndarray[int], int)
        ``labels`` has background 0 and islands numbered 1..n.
    """
    if connectivity not in (6, 18, 26):
        raise ValueError("connectivity must be one of 6, 18, 26")
    binary = np.asarray(binary) > 0
    labels, n = cc3d.connected_components(
        binary.astype(np.uint8), connectivity=connectivity, return_N=True
    )
    return labels.astype(np.int32), int(n)


def apply_zrange(volume: np.ndarray, z_min: int, z_max: int) -> np.ndarray:
    """Zero every voxel whose Z index is outside ``[z_min, z_max]`` (inclusive).

    Works on either a binary mask or a labeled volume; returns a new array of
    the same dtype. Callers that pass a labeled volume should relabel afterward
    (see :func:`label_islands`) because a crop can split an island in two.
    """
    volume = np.asarray(volume)
    nz = volume.shape[0]
    z_min = max(0, int(z_min))
    z_max = min(nz - 1, int(z_max))
    out = np.zeros_like(volume)
    if z_min <= z_max:
        out[z_min : z_max + 1] = volume[z_min : z_max + 1]
    return out


def compute_bboxes_and_sizes(
    labels: np.ndarray, n: int
) -> Tuple[Dict[int, Tuple[slice, slice, slice]], np.ndarray]:
    """Compute per-island bounding boxes and voxel sizes in a single pass.

    Uses ``np.bincount`` for sizes and ``scipy.ndimage.find_objects`` for
    bounding boxes, avoiding an O(N x volume) per-island ``np.where`` loop.

    Returns
    -------
    (bboxes, sizes)
        ``bboxes`` maps island id -> (z_slice, y_slice, x_slice).
        ``sizes`` is an array indexed by label id (``sizes[i]`` = voxel count of
        island ``i``; ``sizes[0]`` is the background count).
    """
    labels = np.asarray(labels)
    sizes = np.bincount(labels.ravel(), minlength=n + 1)
    # find_objects returns a list indexed by (label - 1); entry is None if a
    # label id is absent from the volume.
    slices = ndi.find_objects(labels, max_label=n)
    bboxes: Dict[int, Tuple[slice, slice, slice]] = {}
    for idx, sl in enumerate(slices):
        if sl is not None:
            bboxes[idx + 1] = sl
    return bboxes, sizes


def filter_by_size(sizes: np.ndarray, min_size: int, ids: Iterable[int] = None) -> List[int]:
    """Return the island ids whose voxel size is ``>= min_size``.

    ``sizes`` is the array from :func:`compute_bboxes_and_sizes` (indexed by id).
    ``ids`` restricts the candidate set; defaults to all ids 1..len(sizes)-1.
    """
    sizes = np.asarray(sizes)
    if ids is None:
        ids = range(1, len(sizes))
    return [int(i) for i in ids if sizes[i] >= min_size]


def renumber(
    labels: np.ndarray, keep_ids: Iterable[int]
) -> Tuple[np.ndarray, Dict[int, int]]:
    """Remap ``keep_ids`` to contiguous 1..N, dropping everything else.

    Returns
    -------
    (new_labels, id_map)
        ``new_labels`` contains only the kept islands, renumbered 1..N in the
        order given by sorted ``keep_ids``. ``id_map`` maps old id -> new id.
        Voxel membership of each kept island is preserved exactly.
    """
    labels = np.asarray(labels)
    keep_ids = sorted({int(i) for i in keep_ids if i != 0})
    id_map = {old: new for new, old in enumerate(keep_ids, start=1)}

    # Build a lookup table indexed by old id for a fast vectorized remap.
    max_old = int(labels.max()) if labels.size else 0
    lut = np.zeros(max_old + 1, dtype=np.int32)
    for old, new in id_map.items():
        if old <= max_old:
            lut[old] = new
    new_labels = lut[labels]
    return new_labels, id_map


def binary_from_selection(labels: np.ndarray, selected_ids: Iterable[int]) -> np.ndarray:
    """Build a uint8 0/1 binary mask that is 1 wherever the label is selected."""
    labels = np.asarray(labels)
    selected = np.fromiter((int(i) for i in selected_ids), dtype=np.int64)
    if selected.size == 0:
        return np.zeros(labels.shape, dtype=np.uint8)
    mask = np.isin(labels, selected)
    return mask.astype(np.uint8)


def dilate_binary(binary: np.ndarray, radius: int = 1, iterations: int = 1) -> np.ndarray:
    """Morphologically dilate a binary volume with a spherical structuring element.

    Dilation can merge previously separate islands, so callers should relabel
    the result. Returns a uint8 0/1 array.
    """
    struct = ball(int(radius))
    out = ndi.binary_dilation(np.asarray(binary) > 0, structure=struct, iterations=int(iterations))
    return out.astype(np.uint8)


def erode_binary(binary: np.ndarray, radius: int = 1, iterations: int = 1) -> np.ndarray:
    """Morphologically erode a binary volume with a spherical structuring element.

    Erosion can split one island into several or remove small ones entirely, so
    callers should relabel the result. Returns a uint8 0/1 array.
    """
    struct = ball(int(radius))
    out = ndi.binary_erosion(np.asarray(binary) > 0, structure=struct, iterations=int(iterations))
    return out.astype(np.uint8)


def bbox_center(bbox: Tuple[slice, slice, slice]) -> Tuple[int, int, int]:
    """Return the integer (z, y, x) center of a (z, y, x) slice bounding box."""
    zc = (bbox[0].start + bbox[0].stop - 1) // 2
    yc = (bbox[1].start + bbox[1].stop - 1) // 2
    xc = (bbox[2].start + bbox[2].stop - 1) // 2
    return int(zc), int(yc), int(xc)
