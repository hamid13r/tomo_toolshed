"""MRC read/write helpers built on ``mrcfile``.

Kept free of matplotlib so it can be imported and tested headlessly.
"""

from __future__ import annotations

import numpy as np
import mrcfile


def read_mrc(path: str) -> np.ndarray:
    """Read an MRC/REC volume and return its data as a numpy array.

    Uses ``permissive=True`` so slightly non-conformant headers (common with
    tomography software) still load. The returned array is a copy owned by the
    caller (the file handle is closed on return).
    """
    with mrcfile.open(path, permissive=True) as mrc:
        if mrc.data is None:
            raise ValueError(f"MRC file {path!r} contains no data.")
        return np.asarray(mrc.data).copy()


def write_mrc(path: str, data: np.ndarray, voxel_size=None, overwrite: bool = True) -> None:
    """Write ``data`` to ``path`` as an MRC file.

    ``voxel_size`` may be a scalar or a length-3 (x, y, z) sequence in angstroms;
    if given it is stamped into the header so downstream tools keep the scale.
    """
    data = np.ascontiguousarray(data)
    with mrcfile.new(path, overwrite=overwrite) as mrc:
        mrc.set_data(data)
        if voxel_size is not None:
            mrc.voxel_size = voxel_size
        mrc.update_header_from_data()
        mrc.update_header_stats()


def get_voxel_size(path: str):
    """Return the voxel size recorded in an MRC header (or ``None``)."""
    try:
        with mrcfile.open(path, permissive=True, header_only=True) as mrc:
            return mrc.voxel_size
    except Exception:
        return None
