"""Memory-light tomogram loading, binning and Gaussian filtering for the viewers.

This module deliberately imports no matplotlib, so it can be tested headless and
shared by both ``curate`` (``gui``) and ``preview``. The binning reads the MRC
through ``mrcfile.mmap`` in Z-slab chunks into a preallocated float32 output, so
peak memory for ``--bin 2`` on a multi-GB tomogram stays at a fraction of the
full float32 volume rather than loading it all first.
"""

import mmap

import mrcfile
import numpy as np


def binned_shape(shape, bin_factor):
    """Output (Z, Y, X) after mean-pooling by ``bin_factor``; each axis is cropped first."""
    return tuple(n // bin_factor for n in shape)


def _release(raw_mmap, start, length):
    """Hint the OS to drop a consumed byte range of the mmap (page-aligned), best effort."""
    if raw_mmap is None or not hasattr(raw_mmap, "madvise") or not hasattr(mmap, "MADV_DONTNEED"):
        return
    page = mmap.PAGESIZE
    aligned = (start + page - 1) // page * page          # round start up to a page
    end = (start + length) // page * page                # round end down to a page
    if end > aligned:
        try:
            raw_mmap.madvise(mmap.MADV_DONTNEED, aligned, end - aligned)
        except (OSError, ValueError):
            pass


def bin_volume(path, bin_factor=1):
    """Read an MRC and mean-pool it by ``bin_factor`` in 3D; return (Z, Y, X) float32.

    Each axis is first cropped to a multiple of ``bin_factor``. The volume is read
    through ``mrcfile.mmap`` one output-Z-slab at a time into a preallocated float32
    array, and each consumed slab of the mapping is released with ``MADV_DONTNEED``,
    so the whole float32 volume is never resident at once -- peak input memory is a
    few Z-planes rather than the full tomogram. ``bin_factor`` 1 returns the full
    volume as float32, identical to a plain float32 cast.
    """
    if bin_factor < 1:
        raise ValueError("bin_factor must be >= 1")
    b = int(bin_factor)
    with mrcfile.mmap(path, permissive=True, mode="r") as mrc:
        data = mrc.data
        nz, ny, nx = data.shape
        oz, oy, ox = nz // b, ny // b, nx // b
        cy, cx = oy * b, ox * b
        out = np.empty((oz, oy, ox), np.float32)
        raw = data.base if isinstance(getattr(data, "base", None), mmap.mmap) else None
        data_offset = int(getattr(data, "offset", 0))
        plane_bytes = int(data.strides[0])  # bytes per Z plane (data is C-contiguous)
        for k in range(oz):
            slab = np.asarray(data[k * b:(k + 1) * b, :cy, :cx], dtype=np.float32)
            if b == 1:
                out[k] = slab[0]
            else:
                out[k] = slab.reshape(b, oy, b, ox, b).mean(axis=(0, 2, 4))
            _release(raw, data_offset + k * b * plane_bytes, b * plane_bytes)
    return out


def gaussian_sigma_voxels(sigma_ang, apix):
    """Gaussian sigma in voxels from a sigma in Angstrom at pixel size ``apix`` (Angstrom)."""
    return float(sigma_ang) / float(apix) if sigma_ang and sigma_ang > 0 else 0.0


def stretch_to_uint8(vol, lo_pct=0.5, hi_pct=99.5):
    """Contrast-stretch a float volume to uint8 between its ``lo_pct``/``hi_pct`` percentiles."""
    lo, hi = np.percentile(vol[::4, ::4, ::4], [lo_pct, hi_pct])
    stretched = np.clip((vol - lo) / max(hi - lo, 1e-6) * 255.0, 0, 255)
    return stretched.astype(np.uint8)


def load_tomogram_uint8(path, bin_factor=1, gaussian_ang=0.0, apix=None):
    """Read a tomogram as (Z, Y, X) uint8 for display, binned and optionally 3D-blurred.

    The volume is mean-pooled by ``bin_factor`` (memory-light, see :func:`bin_volume`),
    then, if ``gaussian_ang`` > 0, filtered in 3D with a Gaussian whose sigma is
    ``gaussian_ang`` Angstrom converted to binned voxels with ``apix`` (which must be
    the *binned* pixel size). The 0.5-99.5 percentile contrast stretch to uint8 is
    done last, after binning and filtering. ``bin_factor`` 1 with no Gaussian is
    byte-identical to the previous plain loader.
    """
    vol = bin_volume(path, bin_factor)
    sigma = gaussian_sigma_voxels(gaussian_ang, apix) if apix else 0.0
    if sigma > 0:
        from scipy.ndimage import gaussian_filter
        vol = gaussian_filter(vol, sigma)
    return stretch_to_uint8(vol)
