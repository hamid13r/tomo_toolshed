"""Binary erosion of a segmentation volume, in the XY plane or in 3D.

Axis convention matches ``segmentation_curator``: volumes are numpy arrays in
``(Z, Y, X)`` order, as returned by ``mrcfile``. "XY" erosion therefore works
slice by slice along axis 0 (Z) and never moves anything between Z slices.

Erosion is done with plain NumPy shifts (a voxel survives an iteration only if
it and every neighbour in the structuring element are foreground), so this
module has no SciPy dependency and is import-clean of matplotlib. Voxels outside
the volume count as foreground, so objects touching the volume edge are not
eaten from the edge inward.
"""

from __future__ import annotations

from itertools import product
from typing import List, Optional, Tuple

import numpy as np

#: Valid connectivities per mode. 4/8 are in-plane (XY); 6/18/26 are 3D and use
#: the same names as ``curate --connectivity``.
CONNECTIVITY_XY = (4, 8)
CONNECTIVITY_3D = (6, 18, 26)
DEFAULT_CONNECTIVITY = {False: 4, True: 6}


class ErodeError(ValueError):
    """Bad parameters or input for the eroder."""


def neighbour_offsets(three_d: bool, connectivity: Optional[int] = None) -> List[Tuple[int, int, int]]:
    """``(dz, dy, dx)`` offsets of the structuring element, excluding the centre.

    XY mode: ``dz`` is always 0; connectivity 4 is the in-plane cross, 8 the
    3x3 square. 3D mode: 6 is the 3D cross, 18 adds the edge neighbours, 26 is
    the full 3x3x3 cube.
    """
    if connectivity is None:
        connectivity = DEFAULT_CONNECTIVITY[three_d]
    valid = CONNECTIVITY_3D if three_d else CONNECTIVITY_XY
    if connectivity not in valid:
        mode = "3D" if three_d else "XY"
        raise ErodeError(
            f"connectivity {connectivity} is not valid for {mode} erosion "
            f"(choose from {', '.join(map(str, valid))})")
    # Max Manhattan distance of a neighbour for each connectivity.
    reach = {4: 1, 8: 2, 6: 1, 18: 2, 26: 3}[connectivity]
    z_range = (-1, 0, 1) if three_d else (0,)
    return [(dz, dy, dx)
            for dz, dy, dx in product(z_range, (-1, 0, 1), (-1, 0, 1))
            if (dz, dy, dx) != (0, 0, 0) and abs(dz) + abs(dy) + abs(dx) <= reach]


def _erode_once(mask: np.ndarray, offsets, three_d: bool) -> np.ndarray:
    pad_z = 1 if three_d else 0
    padded = np.pad(mask, ((pad_z, pad_z), (1, 1), (1, 1)), constant_values=True)
    nz, ny, nx = mask.shape
    out = mask.copy()
    for dz, dy, dx in offsets:
        out &= padded[pad_z + dz: pad_z + dz + nz, 1 + dy: 1 + dy + ny, 1 + dx: 1 + dx + nx]
    return out


def erode(mask: np.ndarray, iterations: int = 1, three_d: bool = False,
          connectivity: Optional[int] = None) -> np.ndarray:
    """Erode a binary ``(Z, Y, X)`` (or 2D ``(Y, X)``) mask ``iterations`` times.

    Any non-zero voxel is foreground (same rule as ``curate``). Returns a bool
    array of the input's shape. ``three_d=False`` erodes each Z slice in the XY
    plane only; ``three_d=True`` erodes along Z as well.
    """
    if iterations < 1:
        raise ErodeError("iterations must be >= 1")
    arr = np.asarray(mask)
    if arr.ndim == 2:
        if three_d:
            raise ErodeError("3D erosion needs a 3D volume; this mask is 2D")
        arr = arr[np.newaxis]
        squeeze = True
    elif arr.ndim == 3:
        squeeze = False
    else:
        raise ErodeError(f"expected a 2D or 3D mask, got {arr.ndim}D")

    offsets = neighbour_offsets(three_d, connectivity)
    out = arr > 0
    for _ in range(iterations):
        if not out.any():
            break
        out = _erode_once(out, offsets, three_d)
    return out[0] if squeeze else out


# ---------------------------------------------------------------------------
# MRC I/O. Kept here (not in curate's io.py) so the origin is carried over too.
# ---------------------------------------------------------------------------
def read_mask(path: str):
    """Return ``(data, voxel_size, origin)`` from an MRC file."""
    import mrcfile

    with mrcfile.open(path, permissive=True) as mrc:
        if mrc.data is None:
            raise ErodeError(f"MRC file {path!r} contains no data")
        data = np.asarray(mrc.data).copy()
        voxel_size = mrc.voxel_size.copy()
        origin = mrc.header.origin.copy()
    return data, voxel_size, origin


def write_mask(path: str, mask: np.ndarray, voxel_size=None, origin=None) -> None:
    """Write ``mask`` as a uint8 0/1 MRC (like ``curate``), keeping voxel size and origin."""
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
