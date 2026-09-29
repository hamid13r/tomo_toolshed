"""
Tomogram reconstruction engine.

This reproduces the algorithm of TiltSeries.ReconstructFull:

  1. load per-tilt averages, Fourier-rescale to the target pixel size
  2. preprocess each tilt (dirt/mask, subtract mean, high-pass, normalize, invert)
  3. for the tomogram, insert each tilt as a CTF-weighted central slice into a
     3D Fourier volume, using
        - the weighted CTF  (phase-flip * dose/geom weighting)  as the data term
        - |unweighted CTF|                                      as the weight term
        - a coverage ("Samples") term, capped at 1
     then reconstruct = Data*min(cover,1) / max(Weight, floor), inverse FFT
  4. optional deconvolution, contrast handling, and MRC/PNG output

Two engines:
  * mode="global"     (default, CPU-practical): one Fourier volume for the whole
    tomogram, per-tilt CTF evaluated at the tomogram-center defocus. Fast; the
    documented "faithful & close" approximation (no per-subvolume local defocus
    or local warp).
  * mode="subvolume"  (experimental, slow): tiles the volume into padded sub-
    volumes exactly like Warp, evaluating the CTF/weighting per sub-volume. Most
    faithful; only practical on small volumes or a GPU port.

Optional novaCTF-style 3D-CTF correction (set `ctf3d_defocus_step_nm` or
`ctf3d_num_strips` on ReconOptions, mode="global" only): splits the tomogram
thickness into N Z-strips and reconstructs each with its own per-tilt defocus,
then stitches the correctly-focused Z-slab from each into the final volume --
see `reconstruct_novactf` docstring for how this adapts novaCTF's algorithm to
this engine's joint-Fourier-insertion architecture.

The weighting and filtering are injected as callables so new versions are a
one-line change (see weighting.py / filters.py).
"""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import numpy as np
import scipy.fft as spfft

from .ctf import ctf_2d
from . import geometry as geo
from . import weighting as wmod
from . import filters as fmod
from .parallel import get_threads


@dataclass
class ReconOptions:
    angpix: float                      # target pixel size (Angstrom) == --angpix
    raw_angpix: float                  # unbinned tilt-image pixel size (Angstrom)
    invert: bool = True
    normalize: bool = True
    renorm_variance: bool = False      # see filters.preprocess_tilt docstring
    local_motion: bool = True          # bake GridMovementX/Y into the tilts (bake_local_motion)
    highpass: bool = True              # False = no band-pass on the tilts (plain WBP-style)
    do_deconv: bool = False
    deconv_strength: float = 1.0
    deconv_falloff: float = 1.0
    deconv_highpass: float = 300.0
    subvolume_size: int = 64
    subvolume_padding: float = 3.0
    weight_floor: float = 0.01
    use_global_weights: bool = False
    pad_factor: float = 1.15           # in-plane padding (footprint under tilt)
    z_pad_factor: float = 1.0          # Z padding (keep ~= tomogram thickness)
    mode: str = "global"               # "global" | "etomo" | "novactf" (see reconstruct())
    ctf3d_defocus_step_nm: float = 0.0 # novaCTF-style 3D-CTF: Z-strip thickness (nm)
    ctf3d_num_strips: int = 0          # novaCTF-style 3D-CTF: Z-strip count (alternative to step)
    # mode="etomo" only: an etomo.EtomoOptions. Untyped to keep this module
    # free of an import back into etomo.py, which imports from here.
    etomo: object = None
    # mode="novactf" only: a novactf.NovaCTFOptions (untyped for the same reason)
    novactf: object = None


# --------------------------------------------------------------------------- #
# helpers                                                                     #
# --------------------------------------------------------------------------- #
def _even(n):
    return int(round(n / 2.0)) * 2


def fourier_rescale(img, new_shape):
    """Fourier crop/pad a 2D image to new_shape (matching Warp's scaling).
    Uses scipy.fft (multi-threaded, --threads) instead of numpy.fft: numpy's FFT is
    single-threaded, and this runs on the full native-resolution tilt image
    once per tilt -- profiling showed it as the single largest cost in the
    whole reconstruction (>50% of wall-clock on a 64-core machine using ~1
    core). scipy.fft is API-compatible; only the extra `workers` kwarg differs."""
    if img.shape == tuple(new_shape):
        return img.astype(np.float32)
    F = spfft.fftshift(spfft.fft2(img, workers=get_threads()))
    oy, ox = img.shape
    ny, nx = new_shape
    out = np.zeros((ny, nx), np.complex64)
    cy0, cx0 = oy // 2, ox // 2
    cy1, cx1 = ny // 2, nx // 2
    hy = min(oy, ny) // 2
    hx = min(ox, nx) // 2
    out[cy1 - hy:cy1 + hy, cx1 - hx:cx1 + hx] = \
        F[cy0 - hy:cy0 + hy, cx0 - hx:cx0 + hx]
    scaled = spfft.ifft2(spfft.ifftshift(out), workers=get_threads()) * (nx * ny) / (ox * oy)
    return np.real(scaled).astype(np.float32)


def _extract_patch(img, cx, cy, size):
    """Extract a size x size patch centered on (cx, cy) with sub-pixel shift.
    Missing area is zero-padded. Returns float32 patch centered so that (cx,cy)
    maps to the patch center."""
    ix, iy = int(np.floor(cx)), int(np.floor(cy))
    rx, ry = cx - ix, cy - iy
    half = size // 2
    y0, y1 = iy - half, iy - half + size
    x0, x1 = ix - half, ix - half + size
    patch = np.zeros((size, size), np.float32)
    sy0, sx0 = max(0, -y0), max(0, -x0)
    dy0, dx0 = max(0, y0), max(0, x0)
    dy1, dx1 = min(img.shape[0], y1), min(img.shape[1], x1)
    if dy1 > dy0 and dx1 > dx0:
        patch[sy0:sy0 + (dy1 - dy0), sx0:sx0 + (dx1 - dx0)] = img[dy0:dy1, dx0:dx1]
    # sub-pixel shift via Fourier phase. After the integer crop (cx, cy) sits
    # at patch index (half + rx, half + ry); moving it onto `half` is a shift
    # by (-rx, -ry), i.e. a ramp of exp(+2*pi*i*f*r). (This was exp(-...),
    # which moved it to half + 2r instead: a per-tilt error of up to 1 px that
    # blurred the tomogram and left ~1 voxel offsets vs Warp.)
    if abs(rx) > 1e-4 or abs(ry) > 1e-4:
        fy = np.fft.fftfreq(size)[:, None]
        fx = np.fft.fftfreq(size)[None, :]
        ramp = np.exp(2j * np.pi * (fx * rx + fy * ry))
        patch = np.real(spfft.ifft2(spfft.fft2(patch, workers=get_threads()) * ramp, workers=get_threads())).astype(np.float32)
    return patch


def _centered_slice_ft(patch):
    """Math-centered 2D FT (DC at center)."""
    return spfft.fftshift(spfft.fft2(spfft.ifftshift(patch), workers=get_threads())).astype(np.complex64)


def _trilinear_scatter(vol, weightvol, covervol, coords, values, weights):
    """Scatter-add complex `values` and real `weights`/coverage into 3D grids
    using trilinear interpolation. coords: (M,3) float indices (x, y, z), where
    x->axis2, y->axis1, z->axis0. Grid may be anisotropic.

    Accumulates via np.unique(..., return_inverse=True) + np.bincount rather
    than np.add.at: add.at can't vectorize over colliding indices (it's one
    of numpy's slowest primitives) and this is the dominant cost of every
    reconstruction pass. bincount alone would need an output the size of the
    *whole* volume on every call (its length is set by the largest index, not
    by how many voxels are actually touched) -- with this called once per
    tilt, that's a full-volume-sized allocation dozens of times over. Folding
    duplicate indices down to the unique set actually touched first keeps the
    temporary arrays bounded by this call's real footprint (O(len(coords)),
    not O(vol.size)), and both steps are vectorized in C."""
    nz, ny, nx = vol.shape
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    x0 = np.floor(x).astype(np.int64); y0 = np.floor(y).astype(np.int64); z0 = np.floor(z).astype(np.int64)
    fx = x - x0; fy = y - y0; fz = z - z0

    idx_parts, re_parts, im_parts, w_parts, cov_parts = [], [], [], [], []
    values_re = values.real
    values_im = values.imag
    for dz in (0, 1):
        wz = (fz if dz else 1 - fz)
        zz = z0 + dz
        for dy in (0, 1):
            wy = (fy if dy else 1 - fy)
            yy = y0 + dy
            for dx in (0, 1):
                wx = (fx if dx else 1 - fx)
                xx = x0 + dx
                m = (xx >= 0) & (xx < nx) & (yy >= 0) & (yy < ny) & (zz >= 0) & (zz < nz)
                w = (wx * wy * wz)[m]
                idx_parts.append((zz[m] * ny + yy[m]) * nx + xx[m])
                re_parts.append(values_re[m] * w)
                im_parts.append(values_im[m] * w)
                w_parts.append(weights[m] * w)
                cov_parts.append(w)

    flat_idx = np.concatenate(idx_parts)
    uniq_idx, inverse = np.unique(flat_idx, return_inverse=True)

    vol_flat = vol.reshape(-1)
    weight_flat = weightvol.reshape(-1)
    cover_flat = covervol.reshape(-1)

    vol_flat.real[uniq_idx] += np.bincount(inverse, weights=np.concatenate(re_parts)).astype(vol.real.dtype)
    vol_flat.imag[uniq_idx] += np.bincount(inverse, weights=np.concatenate(im_parts)).astype(vol.real.dtype)
    weight_flat[uniq_idx] += np.bincount(inverse, weights=np.concatenate(w_parts)).astype(weight_flat.dtype)
    cover_flat[uniq_idx] += np.bincount(inverse, weights=np.concatenate(cov_parts)).astype(cover_flat.dtype)


# --------------------------------------------------------------------------- #
# global engine                                                               #
# --------------------------------------------------------------------------- #
def _compute_volume_grid(model, opts: ReconOptions):
    """Output voxel size (Vx,Vy,Vz) and the (possibly padded) reconstruction
    grid (S,S,Nz) it's built in. Defocus-independent, shared by every engine."""
    Vx = _even(model.volume_dims_A[0] / opts.angpix)
    Vy = _even(model.volume_dims_A[1] / opts.angpix)
    Vz = _even(model.volume_dims_A[2] / opts.angpix)
    # anisotropic reconstruction grid sized to the actual box (keeps Z thin so
    # memory stays ~O(box), not O(max_dim^3)):
    #   in-plane S x S covers the tomogram footprint under tilt-axis rotation
    #   Nz stays at the (thin) tomogram thickness
    S = _even(max(Vx, Vy) * opts.pad_factor)
    Nz = _even(Vz * opts.z_pad_factor)
    return Vx, Vy, Vz, S, Nz


def _preprocess_tilts(model, tilt_images, opts: ReconOptions):
    """Rescale + filter every tilt to the target pixel size. Independent of
    defocus, so novaCTF's Z-strip passes share this instead of repeating it."""
    n = model.n_tilts
    down = opts.angpix / opts.raw_angpix

    # Warp's high-pass cutoff: 1/(SizeSub*padding/2) cycles/pixel -> Nyquist frac
    hp_frac = 1.0 / (opts.subvolume_size * opts.subvolume_padding / 2.0) * 2.0

    def one(t):
        raw = tilt_images[t].astype(np.float32)
        dsy = _even(raw.shape[0] / down)
        dsx = _even(raw.shape[1] / down)
        img = fourier_rescale(raw, (dsy, dsx))
        return fmod.preprocess_tilt(img, opts.angpix, hp_frac,
                                    normalize=opts.normalize, invert=opts.invert,
                                    renorm_variance=opts.renorm_variance,
                                    do_highpass=opts.highpass)

    # tilts are independent and every step is FFT / numpy work that releases
    # the GIL, so a thread pool over tilts beats one tilt at a time even though
    # each FFT is itself multi-threaded (small per-tilt FFTs scale poorly)
    with ThreadPoolExecutor(max_workers=max(1, min(n, 16, get_threads()))) as ex:
        scaled = list(ex.map(one, range(n)))

    # size rounding factor (usually ~1) from the first tilt
    sr_x = scaled[0].shape[1] / (tilt_images[0].shape[1] / down)
    sr_y = scaled[0].shape[0] / (tilt_images[0].shape[0] / down)
    size_rounding = (sr_x, sr_y, 1.0)

    # image dims in Angstrom for the geometry. Prefer the value from the .xml
    # (ImageDimensionsAngstrom = raw dims * raw pixel size); fall back to the
    # supplied tilt-image header dims if the xml did not carry it.
    if not (model.image_dims_A > 0).all():
        model.image_dims_A = np.array([tilt_images[0].shape[1] * opts.raw_angpix,
                                       tilt_images[0].shape[0] * opts.raw_angpix],
                                      np.float32)
    return scaled, size_rounding


def _grid_is_spatial(grid):
    """True if a movement grid varies across the image (more than one node in
    X or Y) -- only then is there local motion to bake in."""
    x, y, _z = grid.dims
    return x > 1 or y > 1


def bake_local_motion(model, t, img, angpix, size_rounding):
    """Resample tilt t so Warp's local-motion shift is already applied.

    In GetPositionInAllTilts the local motion is a 2-D shift subtracted from
    the *globally* projected image position T(v), looked up at T(v) itself
    (GridMovementX/Y are indexed by normalized image position and tilt only,
    not by depth):

        xy(v) = T(v) - mv(T(v))

    So the warped image I'(q) = I(q - mv(q)) satisfies I'(T(v)) = I(xy(v))
    for every voxel v at every depth: sampling I' with the *global* alignment
    reproduces Warp's local geometry exactly, up to one extra interpolation.
    That is what lets engines that only take one global alignment per tilt --
    IMOD's `tilt`, and this module's Fourier insertion, which extracts one
    patch per tilt at the tomogram centre -- use Warp's local motion.
    """
    from scipy.ndimage import map_coordinates
    H, W = img.shape
    srx, sry = float(size_rounding[0]), float(size_rounding[1])
    v, u = np.mgrid[0:H, 0:W].astype(np.float64)
    qx, qy = u * angpix / srx, v * angpix / sry           # Angstrom, pre-rounding
    gstep = 1.0 / max(model.n_tilts - 1, 1)
    coords = np.stack([(qx / model.image_dims_A[0]).ravel(),
                       (qy / model.image_dims_A[1]).ravel(),
                       np.full(qx.size, t * gstep)], axis=1)
    mvx = model.grid_movement_x.interpolate(coords).reshape(H, W)
    mvy = model.grid_movement_y.interpolate(coords).reshape(H, W)
    src = [v - sry * mvy / angpix, u - srx * mvx / angpix]
    return map_coordinates(img, src, order=3, mode="constant", cval=0.0).astype(np.float32)


def _apply_local_motion(model, scaled, size_rounding, xy_A, center, angpix,
                        enabled, progress):
    """Shared by the fourier and etomo engines. If the movement grids vary
    across the image and `enabled`, bake them into the tilts and return the
    *global* projected centre (no local shift) per tilt; otherwise return the
    inputs unchanged, whose xy_A then carries the local shift at the centre."""
    spatial = _grid_is_spatial(model.grid_movement_x) or _grid_is_spatial(model.grid_movement_y)
    if not enabled:
        progress("  local motion: OFF (only its value at the tomogram centre)")
        return scaled, xy_A
    if not spatial:
        return scaled, xy_A
    xy_A = np.array(xy_A, copy=True)
    for t in range(model.n_tilts):
        gx, gy, _d = geo.positions_one_tilt(model, t, center[None], size_rounding,
                                            local_motion=False)
        xy_A[t] = (gx[0], gy[0])
    scaled = [bake_local_motion(model, t, scaled[t], angpix, size_rounding)
              if bool(model.use_tilt[t]) else scaled[t] for t in range(model.n_tilts)]
    progress(f"  local motion: baked GridMovementX/Y "
             f"({'x'.join(map(str, model.grid_movement_x.dims))}) into the tilt images")
    return scaled, xy_A


def _insert_all_tilts(model, scaled, opts: ReconOptions, xy_A, defocus_um,
                      weighting_fn, S, Nz, center, progress):
    """CTF-weighted Fourier-slice insertion for one defocus assumption (the
    single global pass, or one novaCTF Z-strip). `xy_A` (patch center per
    tilt) stays fixed at the true tomogram center regardless of defocus --
    only the CTF term varies with `defocus_um`.
    Returns (data_vol, weight_vol, cover_vol, params)."""
    n = model.n_tilts
    params = weighting_fn(model, center, defocus_um, opts.use_global_weights)

    # centered frequency indices for the S x S slice
    idx = np.arange(S) - S // 2
    KX, KY = np.meshgrid(idx, idx)
    KXf = KX.ravel().astype(np.float64)
    KYf = KY.ravel().astype(np.float64)
    # physical frequencies (1/A) for CTF evaluation (in-plane grid size S)
    sx = (KX / (S * opts.angpix)).astype(np.float32)
    sy = (KY / (S * opts.angpix)).astype(np.float32)

    data_vol = np.zeros((Nz, S, S), np.complex64)
    weight_vol = np.zeros((Nz, S, S), np.float32)
    cover_vol = np.zeros((Nz, S, S), np.float32)

    # Z index scaling: in-plane grid is S wide, but Z has only Nz samples at the
    # same pixel size, so a physical fz maps to a Z index scaled by Nz/S.
    z_scale = Nz / float(S)

    for t in range(n):
        if not bool(model.use_tilt[t]):
            continue
        # projected center in pixels (at target pixel size)
        cx = xy_A[t, 0] / opts.angpix
        cy = xy_A[t, 1] / opts.angpix
        patch = _extract_patch(scaled[t], cx, cy, S)
        slice_ft = _centered_slice_ft(patch)

        cw = ctf_2d(params[t], sx, sy, weighted=True)          # weight + phase flip
        cu = np.abs(ctf_2d(params[t], sx, sy, weighted=False)) # reconstruction weight
        data_slice = (slice_ft * cw).ravel()
        weight_slice = cu.ravel().astype(np.float32)

        # central-slice insertion: f3d = R_t^T @ (kx, ky, 0)
        R = geo.rotation_for_tilt(model, t, center)
        fx = R[0, 0] * KXf + R[1, 0] * KYf
        fy = R[0, 1] * KXf + R[1, 1] * KYf
        fz = (R[0, 2] * KXf + R[1, 2] * KYf) * z_scale
        coords = np.stack([fx + S // 2, fy + S // 2, fz + Nz // 2], axis=1)
        _trilinear_scatter(data_vol, weight_vol, cover_vol,
                           coords, data_slice, weight_slice)
        progress(f"  inserted tilt {t + 1}/{n} (angle {model.angles[t]:+.1f}, "
                 f"defocus {defocus_um[t]:.2f} um)")

    return data_vol, weight_vol, cover_vol, params


def _combine_volume(data_vol, weight_vol, cover_vol, opts: ReconOptions, shape):
    """Combine + deapodize + crop, exactly like ReconstructFull."""
    cover = np.minimum(cover_vol, 1.0)
    data_vol = data_vol * cover
    weight_vol = np.maximum(weight_vol, opts.weight_floor)
    recon_ft = data_vol / weight_vol

    vol = spfft.fftshift(np.real(spfft.ifftn(spfft.ifftshift(recon_ft), workers=get_threads()))).astype(np.float32)
    vol = _deapodize(vol)
    return _crop_center(vol, shape)


def reconstruct_global(model, tilt_images, opts: ReconOptions, progress=print):
    """tilt_images: list of 2D float arrays at raw pixel size, in model order."""
    n = model.n_tilts
    assert len(tilt_images) == n, "need one image per tilt"

    Vx, Vy, Vz, S, Nz = _compute_volume_grid(model, opts)
    progress(f"volume voxels: {Vx}x{Vy}x{Vz}; reconstruction grid: "
             f"{S}x{S}x{Nz} (in-plane x Z)")

    scaled, size_rounding = _preprocess_tilts(model, tilt_images, opts)

    # tomogram center in physical Angstrom
    center = model.volume_dims_A / 2.0
    xy_A, defocus_um = geo.positions_in_all_tilts(model, center, size_rounding)
    scaled, xy_A = _apply_local_motion(model, scaled, size_rounding, xy_A, center,
                                       opts.angpix, opts.local_motion, progress)

    data_vol, weight_vol, cover_vol, params = _insert_all_tilts(
        model, scaled, opts, xy_A, defocus_um, opts._weighting_fn,
        S, Nz, center, progress)

    vol = _combine_volume(data_vol, weight_vol, cover_vol, opts, (Vz, Vy, Vx))

    outputs = {"reconstruction": vol}
    if opts.do_deconv:
        outputs["deconv"] = fmod.deconvolve(
            vol, opts.angpix, params[n // 2],
            strength=opts.deconv_strength, falloff=opts.deconv_falloff,
            highpass=opts.deconv_highpass)
    return outputs


# --------------------------------------------------------------------------- #
# novaCTF-style 3D-CTF correction (Z-strip defocus)                          #
# --------------------------------------------------------------------------- #
def _novactf_num_strips(volume_thickness_A, opts: ReconOptions):
    """Mirrors novaCTF's Geometry::computeNumberOfParts: from a step size in
    nm, floor(thickness/step), forced odd and >=1. Falls back to an explicit
    strip count if no step size was given."""
    if opts.ctf3d_defocus_step_nm and opts.ctf3d_defocus_step_nm > 0:
        step_A = opts.ctf3d_defocus_step_nm * 10.0
        n = max(int(np.floor(volume_thickness_A / step_A)), 1)
        if n % 2 == 0:
            n -= 1
        return max(n, 1)
    return max(int(opts.ctf3d_num_strips), 1)


def reconstruct_novactf(model, tilt_images, opts: ReconOptions, progress=print):
    """novaCTF-style 3D-CTF correction (Turonova et al. 2017): split the
    tomogram thickness into N Z-strips and reconstruct each with its own
    per-tilt defocus, then stitch the correctly-focused Z-slab from each
    strip's reconstruction into the final volume.

    novaCTF itself gets this precision from an explicit real-space
    weighted-back-projection loop: for every output voxel it picks, *per
    contributing tilt*, whichever of N pre-corrected projection copies has
    the defocus closest to that voxel's true depth along *that tilt's* beam
    (ctf3d.cpp:computeOneRow / generateFocusGrid) -- since the same 3D point
    sits at a different depth along the beam for each tilt angle. This
    engine instead does joint multi-tilt Fourier-slice insertion (the
    central-slice theorem: one rotated 2D FT populates every Z at once), so
    there is no per-voxel-per-tilt term left to select between after the
    fact -- a strip can only get ONE representative defocus, applied to a
    tilt's *entire* projected footprint, not varied further across X the way
    novaCTF's own per-tilt geometry does. This is the same category of
    approximation as mode="global" vs Warp's true per-subvolume local
    defocus (see README).

    What we *do* get exactly right: geometry.positions_in_all_tilts already
    computes the true per-tilt defocus at any 3D point via the full rotated
    ray (not novaCTF's own flat, angle-independent nm shift per strip), so
    evaluating it at each strip's Z-shifted center gives genuinely correct
    per-tilt, per-strip defocus values for free.

    Cost: ~N times a single reconstruction pass. Preprocessing (rescale +
    filter) is shared across strips; only the CTF-weighted insertion repeats.
    """
    n = model.n_tilts
    assert len(tilt_images) == n, "need one image per tilt"

    Vx, Vy, Vz, S, Nz = _compute_volume_grid(model, opts)
    N = _novactf_num_strips(model.volume_dims_A[2], opts)
    progress(f"volume voxels: {Vx}x{Vy}x{Vz}; reconstruction grid: "
             f"{S}x{S}x{Nz} (in-plane x Z); novaCTF 3D-CTF: {N} Z-strip(s)")

    scaled, size_rounding = _preprocess_tilts(model, tilt_images, opts)

    center = model.volume_dims_A / 2.0
    xy_A, _ = geo.positions_in_all_tilts(model, center, size_rounding)
    scaled, xy_A = _apply_local_motion(model, scaled, size_rounding, xy_A, center,
                                       opts.angpix, opts.local_motion, progress)

    strip_thickness_A = model.volume_dims_A[2] / N
    merged = np.zeros((Vz, Vy, Vx), np.float32)
    last_params = None

    for i in range(N):
        z_offset = (i - N // 2) * strip_thickness_A
        strip_center = center + np.array([0.0, 0.0, z_offset])
        _, defocus_um_i = geo.positions_in_all_tilts(model, strip_center, size_rounding)

        progress(f"novaCTF strip {i + 1}/{N} (Z offset {z_offset:+.1f} A):")
        data_vol, weight_vol, cover_vol, params = _insert_all_tilts(
            model, scaled, opts, xy_A, defocus_um_i, opts._weighting_fn,
            S, Nz, center, progress)
        vol_i = _combine_volume(data_vol, weight_vol, cover_vol, opts, (Vz, Vy, Vx))
        last_params = params

        z0 = int(round(i * Vz / N))
        z1 = Vz if i == N - 1 else int(round((i + 1) * Vz / N))
        merged[z0:z1] = vol_i[z0:z1]

    outputs = {"reconstruction": merged}
    if opts.do_deconv:
        outputs["deconv"] = fmod.deconvolve(
            merged, opts.angpix, last_params[n // 2],
            strength=opts.deconv_strength, falloff=opts.deconv_falloff,
            highpass=opts.deconv_highpass)
    return outputs


def _deapodize(vol):
    """Divide out the trilinear (triangle) interpolation kernel: sinc^2,
    per axis (grid may be anisotropic)."""
    nz, ny, nx = vol.shape
    kz = np.maximum(np.sinc((np.arange(nz) - nz // 2) / nz) ** 2, 1e-3)
    ky = np.maximum(np.sinc((np.arange(ny) - ny // 2) / ny) ** 2, 1e-3)
    kx = np.maximum(np.sinc((np.arange(nx) - nx // 2) / nx) ** 2, 1e-3)
    vol = vol / kz[:, None, None]
    vol = vol / ky[None, :, None]
    vol = vol / kx[None, None, :]
    return vol.astype(np.float32)


def _crop_center(vol, shape):
    sz, sy, sx = shape
    nz, ny, nx = vol.shape
    z0 = nz // 2 - sz // 2
    y0 = ny // 2 - sy // 2
    x0 = nx // 2 - sx // 2
    return vol[z0:z0 + sz, y0:y0 + sy, x0:x0 + sx].copy()


def reconstruct(model, tilt_images, opts: ReconOptions,
                weighting_fn=None, progress=print):
    """Top-level entry. weighting_fn defaults to the exact Warp weighting."""
    opts._weighting_fn = weighting_fn or wmod.warp_weighting
    if opts.mode == "etomo":
        # real-space weighted back-projection / SIRT via IMOD's `tilt`, driven
        # by Warp's alignment. Lazy import: etomo.py imports from this module.
        from .etomo import reconstruct_etomo
        return reconstruct_etomo(model, tilt_images, opts, progress=progress)
    if opts.mode == "novactf":
        # real-space 3D-CTF back-projection on Warp's per-voxel geometry
        from .novactf import reconstruct_novactf_rs
        return reconstruct_novactf_rs(model, tilt_images, opts, progress=progress)
    if opts.mode == "global":
        if opts.ctf3d_defocus_step_nm or opts.ctf3d_num_strips:
            return reconstruct_novactf(model, tilt_images, opts, progress=progress)
        return reconstruct_global(model, tilt_images, opts, progress=progress)
    raise NotImplementedError(
        "mode='subvolume' is provided as a documented extension point; the "
        "global engine is the CPU-practical default. See README.")
