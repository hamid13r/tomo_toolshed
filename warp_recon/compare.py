"""Compare two reconstructed tomograms (e.g. this engine's output vs Warp's
`ts_reconstruct`): real-space correlation, axis-flip detection, Fourier Shell
Correlation, and radially-averaged power spectra.

Kept separate from the reconstruction engine -- this only ever reads finished
volumes, it never touches tilt images or metadata.
"""
from __future__ import annotations
import numpy as np

try:
    from scipy.signal.windows import tukey
except ImportError:      # pragma: no cover
    tukey = None


def robust_stats(vol: np.ndarray) -> dict:
    """Mean/std plus percentile-based dispersion, which is far less sensitive
    than std to the handful of outlier voxels a CTF-weighted reconstruction
    can produce near zero-coverage Fourier shells."""
    v = np.asarray(vol, dtype=np.float64)
    p1, p25, p50, p75, p99 = np.percentile(v, [1, 25, 50, 75, 99])
    return {
        "mean": float(v.mean()),
        "std": float(v.std()),
        "min": float(v.min()),
        "max": float(v.max()),
        "p1": float(p1),
        "p50": float(p50),
        "p99": float(p99),
        "iqr": float(p75 - p25),
        "p1_p99_range": float(p99 - p1),
    }


def _flip_views(vol: np.ndarray):
    """Yield (flip_code, view) for all 8 combinations of flipping the three
    axes. flip_code is a 3-tuple of +1/-1 for (z, y, x)."""
    for sz in (1, -1):
        for sy in (1, -1):
            for sx in (1, -1):
                yield (sz, sy, sx), vol[::sz, ::sy, ::sx]


def find_best_orientation(reference: np.ndarray, other: np.ndarray,
                          xy_stride: int = 2, z_top_frac: float = 0.25):
    """Search the 8 axis-flip combinations x 2 signs of `other` for the one
    that best matches `reference` by Pearson correlation.

    Scored on the highest-variance Z-slices of `reference` only (i.e. the
    slab that actually contains the specimen): a coarse pass over the WHOLE
    volume is dominated by background/missing-wedge voxels that carry no
    orientation information and drowns out the real signal (in practice this
    made the naive whole-volume search pick a wrong flip). Also searches
    contrast sign, since Warp's own `--dont_invert` is a per-run choice, not
    a fixed convention.

    Returns (best_flip_code, best_sign, table) where table maps
    (flip_code, sign) -> correlation on the scored ROI.
    """
    nz = reference.shape[0]
    zvar = reference.reshape(nz, -1).var(axis=1)
    k = max(1, int(round(nz * z_top_frac)))
    top_z = np.argsort(zvar)[::-1][:k]

    ref_roi = reference[top_z][:, ::xy_stride, ::xy_stride].ravel()
    table = {}
    for code, view in _flip_views(other):
        roi = view[top_z][:, ::xy_stride, ::xy_stride].ravel()
        if roi.shape != ref_roi.shape:
            continue
        for sign in (1, -1):
            table[(code, sign)] = float(np.corrcoef(ref_roi, sign * roi)[0, 1])
    best_code, best_sign = max(table, key=table.get)
    return best_code, best_sign, table


def apply_flip(vol: np.ndarray, flip_code) -> np.ndarray:
    sz, sy, sx = flip_code
    return np.ascontiguousarray(vol[::sz, ::sy, ::sx])


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])


def specimen_band_pearson(reference: np.ndarray, other: np.ndarray, z_top_frac: float = 0.25) -> float:
    """Pearson r restricted to the highest-variance Z-slices of `reference`
    (i.e. the slab that actually contains the specimen). A whole-volume or
    even a uniformly-central-cropped correlation is heavily diluted by
    background/missing-wedge slices that carry no real signal; this is the
    metric that actually reflects reconstruction quality where it matters."""
    nz = reference.shape[0]
    zvar = reference.reshape(nz, -1).var(axis=1)
    k = max(1, int(round(nz * z_top_frac)))
    top_z = np.argsort(zvar)[::-1][:k]
    return pearson(reference[top_z], other[top_z])


def central_crop(vol: np.ndarray, frac: float) -> np.ndarray:
    """Crop `frac` off each side of every axis (frac=0.1 keeps the central
    80% along each axis), to keep missing-wedge edge shadows out of a
    correlation/statistics estimate."""
    slices = []
    for n in vol.shape:
        lo = int(round(n * frac))
        hi = n - lo
        slices.append(slice(lo, hi))
    return vol[tuple(slices)]


def _taper3d(shape, alpha=0.25):
    if tukey is None:
        return np.ones(shape, dtype=np.float32)
    wz = tukey(shape[0], alpha)
    wy = tukey(shape[1], alpha)
    wx = tukey(shape[2], alpha)
    return (wz[:, None, None] * wy[None, :, None] * wx[None, None, :]).astype(np.float32)


def _radius_bins(shape, angpix, nbins):
    fz = np.fft.fftfreq(shape[0], d=angpix)
    fy = np.fft.fftfreq(shape[1], d=angpix)
    fx = np.fft.fftfreq(shape[2], d=angpix)
    r = np.sqrt(fz[:, None, None] ** 2 + fy[None, :, None] ** 2 + fx[None, None, :] ** 2)
    nyquist = 0.5 / angpix
    dr = nyquist / nbins
    idx = np.minimum((r / dr).astype(np.int64), nbins - 1)
    freqs = (np.arange(nbins) + 0.5) * dr
    return idx.ravel(), freqs, dr


def fourier_shell_correlation(vol1: np.ndarray, vol2: np.ndarray, angpix: float,
                              nbins: int | None = None, taper_alpha: float = 0.25):
    """Standard FSC between two same-shape, same-voxel-size volumes.

    Returns dict with 'freq' (1/A), 'fsc', and 'resolution' (A at 0.5 and
    0.143 thresholds, by linear interpolation of the first downward crossing;
    None if the curve never crosses).
    """
    assert vol1.shape == vol2.shape, "FSC requires matching shapes"
    shape = vol1.shape
    if nbins is None:
        nbins = max(shape) // 2

    w = _taper3d(shape, taper_alpha)
    F1 = np.fft.fftn((vol1 * w).astype(np.float32))
    F2 = np.fft.fftn((vol2 * w).astype(np.float32))

    idx, freqs, _ = _radius_bins(shape, angpix, nbins)
    num = np.bincount(idx, weights=(F1 * np.conj(F2)).real.ravel(), minlength=nbins)
    den1 = np.bincount(idx, weights=(np.abs(F1) ** 2).ravel(), minlength=nbins)
    den2 = np.bincount(idx, weights=(np.abs(F2) ** 2).ravel(), minlength=nbins)
    with np.errstate(invalid="ignore", divide="ignore"):
        fsc = num / np.sqrt(den1 * den2)
    fsc = np.nan_to_num(fsc, nan=0.0)

    return {
        "freq": freqs,
        "fsc": fsc,
        "resolution_0.5": _crossing_resolution(freqs, fsc, 0.5),
        "resolution_0.143": _crossing_resolution(freqs, fsc, 0.143),
    }


def _crossing_resolution(freqs, fsc, threshold):
    # Skip the DC/first shell. It contains a single Fourier component (the
    # volume mean), so its "correlation" is exactly +-1 and depends only on the
    # relative sign of the two volumes -- a -1 there made the `idx == 0` guard
    # below report "no crossing" for every sign-flipped volume, even when the
    # rest of the curve was excellent. Crossings are never quoted off the DC
    # shell anyway.
    freqs = np.asarray(freqs)[1:]
    fsc = np.asarray(fsc)[1:]
    below = fsc < threshold
    if not below.any() or not (~below).any():
        return None
    # first index where the curve drops below threshold after having been above
    # it. The lowest shells can sit below threshold too (per-tilt high-pass
    # differences between engines), so the search starts at the first shell
    # that is above it -- not at shell 0, which reported "no crossing" for
    # curves that plainly cross (e.g. 0.95 plateau -> 0.5 at 31 A).
    first_above = int(np.argmax(~below))
    rest = below[first_above:]
    if not rest.any():
        return None
    idx = first_above + int(np.argmax(rest))
    f0, f1 = freqs[idx - 1], freqs[idx]
    c0, c1 = fsc[idx - 1], fsc[idx]
    if c1 == c0:
        f_cross = f0
    else:
        f_cross = f0 + (threshold - c0) * (f1 - f0) / (c1 - c0)
    return float(1.0 / f_cross) if f_cross > 0 else None


def radial_power_spectrum(vol: np.ndarray, angpix: float, nbins: int | None = None,
                           taper_alpha: float = 0.25):
    """Radially-averaged power spectrum (mean |F|^2 per shell)."""
    shape = vol.shape
    if nbins is None:
        nbins = max(shape) // 2
    w = _taper3d(shape, taper_alpha)
    F = np.fft.fftn((vol * w).astype(np.float32))
    power = (np.abs(F) ** 2)
    idx, freqs, _ = _radius_bins(shape, angpix, nbins)
    psum = np.bincount(idx, weights=power.ravel(), minlength=nbins)
    pcount = np.bincount(idx, minlength=nbins)
    with np.errstate(invalid="ignore", divide="ignore"):
        pmean = psum / pcount
    return freqs, np.nan_to_num(pmean, nan=0.0)
