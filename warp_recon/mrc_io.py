"""Thin MRC helpers built on the `mrcfile` package."""
from __future__ import annotations
import numpy as np

try:
    import mrcfile
except ImportError as e:      # pragma: no cover
    mrcfile = None
    _IMPORT_ERR = e


def _need_mrcfile():
    if mrcfile is None:
        raise ImportError(
            "The 'mrcfile' package is required. Install with: pip install mrcfile"
        ) from _IMPORT_ERR


def read_mrc(path):
    """Return (data float32 array [Z,Y,X] or [Y,X], voxel_size_angstrom float)."""
    _need_mrcfile()
    with mrcfile.open(path, permissive=True) as mrc:
        data = np.asarray(mrc.data, dtype=np.float32)
        vs = float(mrc.voxel_size.x) if mrc.voxel_size.x else 0.0
    return data, vs


def write_mrc(path, data, voxel_size_angstrom, as_float16=False):
    """Write a volume/image. Warp writes 16-bit MRC by default; we default to
    float32 (set as_float16=True to mimic Warp's disk format)."""
    _need_mrcfile()
    arr = np.asarray(data, dtype=np.float16 if as_float16 else np.float32)
    with mrcfile.new(path, overwrite=True) as mrc:
        mrc.set_data(arr)
        mrc.voxel_size = (voxel_size_angstrom,) * 3


def write_png_slice(path, vol):
    """Write a central-Z slice preview PNG with Warp's 3-sigma contrast stretch."""
    try:
        from PIL import Image as PILImage
    except ImportError:
        return
    sl = vol[vol.shape[0] // 2].astype(np.float32)
    q = sl[sl.shape[0] // 4: -sl.shape[0] // 4, sl.shape[1] // 4: -sl.shape[1] // 4]
    m, sd = float(q.mean()), float(q.std()) or 1.0
    lo, hi = m - 3 * sd, m + 3 * sd
    img = np.clip((sl - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
    PILImage.fromarray(img).save(path)
