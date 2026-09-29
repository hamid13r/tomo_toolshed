"""
Real-space novaCTF engine -- novaCTF's 3D-CTF weighted back-projection
(Turonova et al. 2017, doi:10.1016/j.jsb.2017.07.007), driven by *Warp's*
full per-voxel geometry, including the local-motion grids.

Why a third engine
------------------
Neither existing path can combine Warp's local alignment with a true 3D CTF:

  * the novaCTF binary takes an IMOD-style aligned stack + .tlt, so it only
    knows a global alignment -- Warp's GridMovementX/Y local shifts have no
    representation there -- and it silently forces XAXISTILT to 0
    (parameterSetup.cpp);
  * the Fourier engine's `--ctf3d_*` option (reconstruct.reconstruct_novactf)
    can only give each Z-strip one defocus per tilt, applied to that tilt's
    whole footprint, because Fourier-slice insertion populates every Z at once.

This engine back-projects voxel by voxel, so it can do both at once. For every
tilt t and every voxel v:

  1. project v through `geometry.positions_one_tilt` -- Warp's exact
     GetPositionInAllTilts, incl. LevelAngleX/Y, AxisOffset, volume-warp and
     (optionally) local-motion grids -- giving its image position and its
     defocus at *its own depth along tilt t's beam*;
  2. pick, out of copies of tilt t's image CTF-corrected at defocus steps of
     `defocus_step_nm`, the copy whose defocus is nearest to v's;
  3. add that copy's bilinearly-interpolated value at v's image position.

Step 2 is novaCTF's own algorithm (ctf3d.cpp: generateFocusGrid picks a
defocus strip per voxel per view, computeOneRow accumulates it). Differences
from the binary, all deliberate:

  * the per-tilt images are NOT resampled into an aligned stack first: each
    voxel samples the raw (preprocessed) tilt image directly, at the position
    Warp's geometry gives -- one interpolation instead of two, and the only
    way the local-motion grids can be honoured;
  * the radial (R-weighting) filter is therefore applied as a 2-D Fourier
    filter along the image direction perpendicular to the tilt axis, rather
    than along the rows of an axis-aligned stack. Same ramp, same cutoff /
    Gaussian falloff and same per-view tilt-increment attenuation as
    filterProjections.cpp:radialWeighting;
  * CTF correction and the radial filter share one forward FFT per tilt;
  * the defocus copies are generated on demand, per tilt, at the exact
    defocus steps each tilt needs (novaCTF writes N fixed stacks to disk);
  * X-axis tilt is applied (it is part of Warp's tilt matrix).

The CTF is `ctf.ctf_2d(weighted=False)`, which is identical to novaCTF's
ctfCorrection.cpp `ctfAmp = w*sin(phase) + A*cos(phase)` (same sign), so
correction="multiplication" multiplies by exactly what novaCTF multiplies by,
and "phaseflip" by its sign.

Optional `weighting="warp"` additionally applies the active weighting_fn's
per-tilt amplitude scale and dose exposure filter (cos(tilt) and B = -4*dose
for Warp's own model). Unlike the etomo engine's WeightFile, the dose filter
*can* travel here, because every tilt already goes through a 2-D Fourier
filter. The default "none" is pure novaCTF weighting, for A/B tests against
the binary.
"""
from __future__ import annotations
import copy
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import numpy as np
import scipy.fft as spfft

from . import geometry as geo
from .ctf import ctf_2d, weight_envelope
from .parallel import get_threads, apply_numba_threads

try:
    import numba
    from numba import njit, prange
    HAVE_NUMBA = True
except ImportError:                      # numpy fallback below, much slower
    HAVE_NUMBA = False

DEG = np.pi / 180.0


@dataclass
class NovaCTFOptions:
    """Knobs for the real-space novaCTF engine."""
    defocus_step_nm: float = 10.0      # novaCTF -DefocusStep; 0 = 2-D CTF (one defocus per tilt)
    radial: tuple = (0.3, 0.05)        # novaCTF -RADIAL cutoff,falloff in cycles/pixel
    correction: str = "multiplication" # "multiplication" | "phaseflip" | "none"
    correct_astigmatism: bool = True
    local_motion: bool = True          # use Warp's GridMovementX/Y
    weighting: str = "none"            # "none" = novaCTF's; "warp" = + weighting_fn scale & dose filter
    geometry_step: int = 8             # voxels between exact geometry evaluations


# --------------------------------------------------------------------------- #
# radial filter (filterProjections.cpp:radialWeighting)                       #
# --------------------------------------------------------------------------- #
def view_attenuation(angles_deg):
    """Per-view weight from the local tilt increment, exactly as novaCTF's
    radialWeighting (and IMOD tilt's -DENSWEIGHT): neighbours weighted 2 and
    1/1.5, normalized by the mean increment. Input in acquisition order is
    fine; the neighbours are taken in angle order, as novaCTF sees them in
    its (sorted) .tlt."""
    angles_deg = np.asarray(angles_deg, np.float64)
    order = np.argsort(angles_deg)
    a = -angles_deg[order] * DEG
    n = len(a)
    att = np.ones(n)
    if n > 1:
        avgint = (a[-1] - a[0]) / (n - 1)
        wincr = (2.0, 1.0 / 1.5)
        for iv in range(n):
            sumint = wsum = 0.0
            for iw in range(2):
                if iv - iw > 0:
                    wsum += wincr[iw]
                    sumint += wincr[iw] * (a[iv - iw] - a[iv - iw - 1])
                if iv + iw + 1 < n:
                    wsum += wincr[iw]
                    sumint += wincr[iw] * (a[iv + iw + 1] - a[iv + iw])
            att[iv] = (sumint / wsum) / avgint
    out = np.empty(n)
    out[order] = att
    return out


def _radial_filter(kx, ky, perp, line_len, cutoff, falloff):
    """novaCTF's ramp along `perp` (unit image vector perpendicular to the
    tilt axis): ramp = frequency index along the padded line (i = r*L), 0.2
    at DC (Kak & Slaney), Gaussian falloff of sigma `falloff` cycles/px
    beyond `cutoff` cycles/px."""
    r = np.abs(kx * perp[0] + ky * perp[1])            # cycles / pixel
    w = np.maximum(r * line_len, 0.2)
    beyond = r >= cutoff
    edge = max(cutoff * line_len - 1.0, 0.2)
    w[beyond] = edge * np.exp(-(((r[beyond] - cutoff) * line_len + 1.0)
                                / max(falloff * line_len, 1e-6)) ** 2)
    return w.astype(np.float32)


# --------------------------------------------------------------------------- #
# back-projection kernels                                                     #
# --------------------------------------------------------------------------- #
if HAVE_NUMBA:
    @njit(parallel=True, fastmath=True, cache=True, nogil=True)
    def _bp_numba(vol, planes, g, PX, PY, KF, kmin):
        # The geometry is trilinear in the coarse grid. Along a voxel row only
        # X varies, so the (Z, Y) part is done once per row at the coarse X
        # nodes, leaving one lerp per field per voxel instead of an 8-corner
        # interpolation -- the same values, ~4x less work in the hot loop.
        nz, ny, nx = vol.shape
        K, H, W = planes.shape
        cz, cy, cx = PX.shape
        for iz in prange(nz):
            fz = iz / g
            z0 = min(int(fz), cz - 2)
            tz = fz - z0
            rpx = np.empty(cx)
            rpy = np.empty(cx)
            rkf = np.empty(cx)
            for iy in range(ny):
                fy = iy / g
                y0 = min(int(fy), cy - 2)
                ty = fy - y0
                w00 = (1.0 - tz) * (1.0 - ty)
                w01 = (1.0 - tz) * ty
                w10 = tz * (1.0 - ty)
                w11 = tz * ty
                for c in range(cx):
                    rpx[c] = (w00 * PX[z0, y0, c] + w01 * PX[z0, y0 + 1, c]
                              + w10 * PX[z0 + 1, y0, c] + w11 * PX[z0 + 1, y0 + 1, c])
                    rpy[c] = (w00 * PY[z0, y0, c] + w01 * PY[z0, y0 + 1, c]
                              + w10 * PY[z0 + 1, y0, c] + w11 * PY[z0 + 1, y0 + 1, c])
                    rkf[c] = (w00 * KF[z0, y0, c] + w01 * KF[z0, y0 + 1, c]
                              + w10 * KF[z0 + 1, y0, c] + w11 * KF[z0 + 1, y0 + 1, c])
                for ix in range(nx):
                    fx = ix / g
                    x0 = min(int(fx), cx - 2)
                    tx = fx - x0
                    px = rpx[x0] + tx * (rpx[x0 + 1] - rpx[x0])
                    py = rpy[x0] + tx * (rpy[x0 + 1] - rpy[x0])
                    if px < 0.0 or py < 0.0 or px > W - 1 or py > H - 1:
                        continue                      # novaCTF edge fill: mean (~0 here)
                    kf = rkf[x0] + tx * (rkf[x0 + 1] - rkf[x0])
                    k = int(np.floor(kf + 0.5)) - kmin
                    if k < 0:
                        k = 0
                    elif k > K - 1:
                        k = K - 1
                    ix0 = min(int(px), W - 2)
                    iy0 = min(int(py), H - 2)
                    fxp = px - ix0
                    fyp = py - iy0
                    v = ((1.0 - fyp) * ((1.0 - fxp) * planes[k, iy0, ix0]
                                        + fxp * planes[k, iy0, ix0 + 1])
                         + fyp * ((1.0 - fxp) * planes[k, iy0 + 1, ix0]
                                  + fxp * planes[k, iy0 + 1, ix0 + 1]))
                    vol[iz, iy, ix] += v

def _bp_numpy(vol, planes, g, PX, PY, KF, kmin):
    """Same as _bp_numba, one Z-plane at a time (slow; for envs without numba)."""
    from scipy.ndimage import map_coordinates
    nz, ny, nx = vol.shape
    K, H, W = planes.shape
    yy, xx = np.meshgrid(np.arange(ny) / g, np.arange(nx) / g, indexing="ij")
    for iz in range(nz):
        c = [np.full_like(yy, iz / g), yy, xx]
        px = map_coordinates(PX, c, order=1, mode="nearest")
        py = map_coordinates(PY, c, order=1, mode="nearest")
        kf = map_coordinates(KF, c, order=1, mode="nearest")
        k = np.clip(np.floor(kf + 0.5).astype(int) - kmin, 0, K - 1)
        inside = (px >= 0) & (py >= 0) & (px <= W - 1) & (py <= H - 1)
        val = map_coordinates(planes, [k.astype(np.float64), py, px], order=1,
                              mode="nearest")
        vol[iz] += np.where(inside, val, 0.0).astype(np.float32)


# --------------------------------------------------------------------------- #
# CTF-corrected defocus copies (the hot loop: one per plane per tilt)         #
# --------------------------------------------------------------------------- #
def ctf_phase_terms(p, sx, sy):
    """Split ctf_2d's phase into a defocus-independent part and a slope:
    argument(defocus_um) = const + slope * defocus_um. Everything except the
    defocus itself (Cs, astigmatism, phase shift) is then evaluated once per
    tilt instead of once per plane. Same formula as ctf.ctf_2d, term by term."""
    K1, K2, K3, _K4 = p.ks()
    s2 = sx.astype(np.float64) ** 2 + sy.astype(np.float64) ** 2
    const = K2 * s2 * s2 - p.phase_shift * np.pi
    if p.defocus_delta != 0.0:
        phi = np.arctan2(sy, sx)
        delta_A = -(p.defocus_delta * 1e4)
        const = const + K1 * s2 * 0.5 * delta_A * np.cos(2.0 * (phi - np.deg2rad(p.defocus_angle)))
    slope = -1e4 * K1 * s2
    return const, slope, float(p.amplitude), float(K3)


if HAVE_NUMBA:
    @njit(parallel=True, cache=True, nogil=True)
    def _ctf_apply_numba(F, const, slope, defs, amp, K3, phaseflip, out):
        K = defs.shape[0]
        ny, nx = F.shape
        for idx in prange(K * ny):
            k = idx // ny
            y = idx - k * ny
            d = defs[k]
            for x in range(nx):
                arg = const[y, x] + slope[y, x] * d
                c = amp * np.cos(arg) - K3 * np.sin(arg)
                if phaseflip:
                    c = 1.0 if c > 0.0 else (-1.0 if c < 0.0 else 0.0)
                out[k, y, x] = F[y, x] * c


def _ctf_apply_numpy(F, const, slope, defs, amp, K3, phaseflip, out):
    for k, d in enumerate(defs):
        arg = const + slope * d
        c = amp * np.cos(arg) - K3 * np.sin(arg)
        out[k] = F * (np.sign(c) if phaseflip else c)


def _defocus_planes(F, defs, ctf_terms, correction, shape_pad, shape_out, batch=16):
    """All CTF-corrected copies of one tilt: G_k = F * CTF(defs[k]) (or its
    sign), inverse-FFT'd in batches -- one multi-threaded irfft2 per batch
    instead of one per plane, and the CTF evaluated in parallel by numba."""
    H, W = shape_out
    planes = np.empty((len(defs), H, W), np.float32)
    if correction == "none":
        planes[:] = spfft.irfft2(F, s=shape_pad, workers=get_threads())[:H, :W]
        return planes
    const, slope, amp, K3 = ctf_terms
    apply = _ctf_apply_numba if HAVE_NUMBA else _ctf_apply_numpy
    buf = np.empty((min(batch, len(defs)),) + F.shape, np.complex64)
    for b0 in range(0, len(defs), batch):
        d = np.asarray(defs[b0:b0 + batch], np.float64)
        g = buf[:len(d)]
        apply(F, const, slope, d, amp, K3, correction == "phaseflip", g)
        planes[b0:b0 + len(d)] = spfft.irfft2(g, s=shape_pad, axes=(-2, -1),
                                             workers=get_threads())[:, :H, :W]
    return planes


def _positions_threaded(pool, model, t, pts, size_rounding, local_motion, nchunks):
    """positions_one_tilt over chunks of points in a thread pool: its cost is
    numpy + scipy map_coordinates, both of which release the GIL."""
    if pool is None or len(pts) < 50_000:
        return geo.positions_one_tilt(model, t, pts, size_rounding, local_motion=local_motion)
    parts = list(pool.map(lambda c: geo.positions_one_tilt(model, t, c, size_rounding,
                                                           local_motion=local_motion),
                          np.array_split(pts, nchunks)))
    return tuple(np.concatenate([q[i] for q in parts]) for i in range(3))


# --------------------------------------------------------------------------- #
# engine                                                                      #
# --------------------------------------------------------------------------- #
def _coarse_axis(n, g):
    """Nodes 0, g, 2g, ... reaching at least n-1 (uniform, so the kernel can
    locate a voxel's cell by division)."""
    return np.arange(int(np.ceil((n - 1) / g)) + 1) * g


def reconstruct_novactf_rs(model, tilt_images, opts, progress=print):
    """3D-CTF real-space back-projection on Warp's geometry (see module doc)."""
    from .reconstruct import _compute_volume_grid, _preprocess_tilts

    no = getattr(opts, "novactf", None) or NovaCTFOptions()
    if no.correction not in ("multiplication", "phaseflip", "none"):
        raise ValueError(f"unknown novactf correction {no.correction!r}")
    if no.weighting not in ("none", "warp"):
        raise ValueError(f"unknown novactf weighting {no.weighting!r}")

    Vx, Vy, Vz, _S, _Nz = _compute_volume_grid(model, opts)
    scaled, size_rounding = _preprocess_tilts(model, tilt_images, opts)
    H, W = scaled[0].shape
    ap = float(opts.angpix)
    tilts = [t for t in range(model.n_tilts) if bool(model.use_tilt[t])]
    step_um = no.defocus_step_nm * 1e-3
    progress(f"volume voxels: {Vx}x{Vy}x{Vz}; engine: real-space novaCTF "
             f"({no.correction}, "
             + (f"defocus step {no.defocus_step_nm:g} nm" if step_um > 0
                else "2-D CTF: one defocus per tilt")
             + f", local motion {'on' if no.local_motion else 'off'}, "
             f"weighting {no.weighting}, {'numba' if HAVE_NUMBA else 'numpy'})")

    # voxel index -> Warp physical coordinate (Angstrom), same convention as
    # the Fourier engine: voxel Vx/2 sits on the volume centre
    def phys(i, n, dim):
        return (np.asarray(i, np.float64) - n / 2.0) * ap + dim / 2.0

    g = max(int(no.geometry_step), 1)
    cz, cy, cx = _coarse_axis(Vz, g), _coarse_axis(Vy, g), _coarse_axis(Vx, g)
    Z, Y, X = np.meshgrid(phys(cz, Vz, model.volume_dims_A[2]),
                          phys(cy, Vy, model.volume_dims_A[1]),
                          phys(cx, Vx, model.volume_dims_A[0]), indexing="ij")
    coarse_pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    cshape = Z.shape
    center = model.volume_dims_A / 2.0

    # shared padded FFT grid
    Py, Px = spfft.next_fast_len(int(H * 1.1) + 16), spfft.next_fast_len(int(W * 1.1) + 16)
    ky = spfft.fftfreq(Py)[:, None]
    kx = spfft.rfftfreq(Px)[None, :]
    kx, ky = np.broadcast_arrays(kx, ky)
    sx, sy = kx / ap, ky / ap                           # cycles / Angstrom

    atten = view_attenuation(model.angles[tilts])
    wparams = None
    if no.weighting == "warp":
        wfn = opts._weighting_fn
        _xy, dcen = geo.positions_in_all_tilts(model, center, size_rounding)
        wparams = wfn(model, center, dcen, opts.use_global_weights)

    nthreads = max(1, min(32, get_threads()))
    pool = ThreadPoolExecutor(nthreads) if nthreads > 1 else None

    def prepare(j):
        """Everything for tilt j except the back-projection: geometry of the
        coarse nodes, filter, and the CTF-corrected defocus copies."""
        t = tilts[j]
        apply_numba_threads()                 # per-thread in numba; prepare may run in a worker
        # --- geometry: image position + defocus of every coarse node ------
        x_A, y_A, d_um = _positions_threaded(pool, model, t, coarse_pts, size_rounding,
                                             no.local_motion, nthreads)
        _cx, _cy, d_c = geo.positions_one_tilt(model, t, center[None], size_rounding,
                                               local_motion=no.local_motion)
        d_c = float(d_c[0])
        PX = (x_A / ap).reshape(cshape).astype(np.float64)
        PY = (y_A / ap).reshape(cshape).astype(np.float64)
        if step_um > 0:
            KF = ((d_um - d_c) / step_um).reshape(cshape)
        else:
            KF = np.zeros(cshape)
        kmin = int(np.floor(KF.min() + 0.5))
        kmax = int(np.floor(KF.max() + 0.5))

        # tilt-axis direction in this image, straight from the geometry
        # (image displacement along volume Y); the ramp runs perpendicular
        yA = geo.positions_one_tilt(model, t, np.stack([center, center + [0, 1000.0, 0]]),
                                    size_rounding, local_motion=False)
        u = np.array([yA[0][1] - yA[0][0], yA[1][1] - yA[1][0]])
        u /= np.linalg.norm(u)
        perp = np.array([u[1], -u[0]])
        npad = min(50, 2 * max(8, int(W) // 20))        # filterProjections' padding
        line_len = abs(perp[0]) * W + abs(perp[1]) * H + npad
        filt = atten[j] * _radial_filter(kx, ky, perp, line_len, *no.radial)

        # --- CTF parameters for this tilt ---------------------------------
        base = copy.copy(model.ctf)
        tc = np.array([[0.5, 0.5, t / max(model.n_tilts - 1, 1)]])
        base.defocus_delta = (float(model.grid_ctf_defocus_delta.interpolate(tc)[0])
                              if no.correct_astigmatism else 0.0)
        base.defocus_angle = float(model.grid_ctf_defocus_angle.interpolate(tc)[0])
        base.phase_shift = float(model.grid_ctf_phase.interpolate(tc)[0])
        if wparams is not None:
            filt = filt * weight_envelope(wparams[t], sx, sy).astype(np.float32)

        pad = np.zeros((Py, Px), np.float32)
        pad[:H, :W] = scaled[t]
        F = spfft.rfft2(pad, workers=get_threads()) * filt

        defs = d_c + np.arange(kmin, kmax + 1) * step_um
        terms = ctf_phase_terms(base, sx, sy) if no.correction != "none" else None
        planes = _defocus_planes(F.astype(np.complex64), defs, terms, no.correction,
                                 (Py, Px), (H, W))
        info = (f"  tilt {j + 1}/{len(tilts)} ({model.angles[t]:+.1f} deg): "
                f"{kmax - kmin + 1} defocus plane(s), "
                f"{d_c + kmin * step_um:.3f}..{d_c + kmax * step_um:.3f} um")
        return planes, PX, PY, KF, kmin, info

    bp = _bp_numba if HAVE_NUMBA else _bp_numpy
    apply_numba_threads()
    vol = np.zeros((Vz, Vy, Vx), np.float32)

    # Pipeline: while tilt j back-projects (memory-bound gathers), tilt j+1's
    # geometry and FFT-bound CTF copies are prepared in a second thread.
    # Needs a numba threading layer that allows concurrent parallel regions
    # (tbb / omp); with 'workqueue' it falls back to one tilt at a time.
    first = prepare(0)
    bp(vol, first[0], float(g), *first[1:5])
    progress(first[5])
    overlap = (not HAVE_NUMBA) or numba.threading_layer() != "workqueue"
    prefetch = ThreadPoolExecutor(1) if overlap and len(tilts) > 1 else None
    nxt = prefetch.submit(prepare, 1) if prefetch and len(tilts) > 1 else None
    for j in range(1, len(tilts)):
        cur = nxt.result() if nxt is not None else prepare(j)
        nxt = prefetch.submit(prepare, j + 1) if prefetch and j + 1 < len(tilts) else None
        bp(vol, cur[0], float(g), *cur[1:5])
        progress(cur[5])
        del cur
    if prefetch is not None:
        prefetch.shutdown()
    if pool is not None:
        pool.shutdown()
    vol /= float(len(tilts))                            # novaCTF: scale = 1/nviews
    return {"reconstruction": vol}
