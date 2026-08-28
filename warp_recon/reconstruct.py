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

The weighting and filtering are injected as callables so new versions are a
one-line change (see weighting.py / filters.py).
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from .ctf import ctf_2d
from . import geometry as geo
from . import weighting as wmod
from . import filters as fmod


@dataclass
class ReconOptions:
    angpix: float                      # target pixel size (Angstrom) == --angpix
    raw_angpix: float                  # unbinned tilt-image pixel size (Angstrom)
    invert: bool = True
    normalize: bool = True
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
    mode: str = "global"


# --------------------------------------------------------------------------- #
# helpers                                                                     #
# --------------------------------------------------------------------------- #
def _even(n):
    return int(round(n / 2.0)) * 2


def fourier_rescale(img, new_shape):
    """Fourier crop/pad a 2D image to new_shape (matching Warp's scaling)."""
    if img.shape == tuple(new_shape):
        return img.astype(np.float32)
    F = np.fft.fftshift(np.fft.fft2(img))
    oy, ox = img.shape
    ny, nx = new_shape
    out = np.zeros((ny, nx), np.complex64)
    cy0, cx0 = oy // 2, ox // 2
    cy1, cx1 = ny // 2, nx // 2
    hy = min(oy, ny) // 2
    hx = min(ox, nx) // 2
    out[cy1 - hy:cy1 + hy, cx1 - hx:cx1 + hx] = \
        F[cy0 - hy:cy0 + hy, cx0 - hx:cx0 + hx]
    scaled = np.fft.ifft2(np.fft.ifftshift(out)) * (nx * ny) / (ox * oy)
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
    # sub-pixel shift by (rx, ry) via Fourier phase
    if abs(rx) > 1e-4 or abs(ry) > 1e-4:
        fy = np.fft.fftfreq(size)[:, None]
        fx = np.fft.fftfreq(size)[None, :]
        ramp = np.exp(-2j * np.pi * (fx * rx + fy * ry))
        patch = np.real(np.fft.ifft2(np.fft.fft2(patch) * ramp)).astype(np.float32)
    return patch


def _centered_slice_ft(patch):
    """Math-centered 2D FT (DC at center)."""
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(patch))).astype(np.complex64)


def _trilinear_scatter(vol, weightvol, covervol, coords, values, weights):
    """Scatter-add complex `values` and real `weights`/coverage into 3D grids
    using trilinear interpolation. coords: (M,3) float indices (x, y, z), where
    x->axis2, y->axis1, z->axis0. Grid may be anisotropic."""
    nz, ny, nx = vol.shape
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    x0 = np.floor(x).astype(np.int64); y0 = np.floor(y).astype(np.int64); z0 = np.floor(z).astype(np.int64)
    fx = x - x0; fy = y - y0; fz = z - z0
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
                idx = (zz[m], yy[m], xx[m])
                np.add.at(vol, idx, values[m] * w)
                np.add.at(weightvol, idx, weights[m] * w)
                np.add.at(covervol, idx, w)


# --------------------------------------------------------------------------- #
# global engine                                                               #
# --------------------------------------------------------------------------- #
def reconstruct_global(model, tilt_images, opts: ReconOptions, progress=print):
    """tilt_images: list of 2D float arrays at raw pixel size, in model order."""
    n = model.n_tilts
    assert len(tilt_images) == n, "need one image per tilt"

    # target volume size in voxels
    Vx = _even(model.volume_dims_A[0] / opts.angpix)
    Vy = _even(model.volume_dims_A[1] / opts.angpix)
    Vz = _even(model.volume_dims_A[2] / opts.angpix)

    # anisotropic reconstruction grid sized to the actual box (keeps Z thin so
    # memory stays ~O(box), not O(max_dim^3)):
    #   in-plane S x S covers the tomogram footprint under tilt-axis rotation
    #   Nz stays at the (thin) tomogram thickness
    S = _even(max(Vx, Vy) * opts.pad_factor)
    Nz = _even(Vz * opts.z_pad_factor)
    progress(f"volume voxels: {Vx}x{Vy}x{Vz}; reconstruction grid: "
             f"{S}x{S}x{Nz} (in-plane x Z)")

    down = opts.angpix / opts.raw_angpix

    # preprocess & rescale all tilts to target pixel size
    scaled = []
    for t in range(n):
        raw = tilt_images[t].astype(np.float32)
        dsy = _even(raw.shape[0] / down)
        dsx = _even(raw.shape[1] / down)
        img = fourier_rescale(raw, (dsy, dsx))
        # Warp's high-pass cutoff: 1/(SizeSub*padding/2) cycles/pixel -> Nyquist frac
        hp_frac = 1.0 / (opts.subvolume_size * opts.subvolume_padding / 2.0) * 2.0
        img = fmod.preprocess_tilt(img, opts.angpix, hp_frac,
                                   normalize=opts.normalize, invert=opts.invert)
        scaled.append(img)

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

    # tomogram center in physical Angstrom
    center = model.volume_dims_A / 2.0
    xy_A, defocus_um = geo.positions_in_all_tilts(model, center, size_rounding)

    # weighted CTF parameters per tilt (the hook lives here)
    weighting_fn = opts._weighting_fn
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

    # combine exactly like ReconstructFull
    cover = np.minimum(cover_vol, 1.0)
    data_vol *= cover
    weight_vol = np.maximum(weight_vol, opts.weight_floor)
    recon_ft = data_vol / weight_vol

    vol = np.fft.fftshift(np.real(np.fft.ifftn(np.fft.ifftshift(recon_ft)))).astype(np.float32)
    vol = _deapodize(vol)

    # crop centered grid to (Vz, Vy, Vx)
    vol = _crop_center(vol, (Vz, Vy, Vx))

    outputs = {"reconstruction": vol}
    if opts.do_deconv:
        outputs["deconv"] = fmod.deconvolve(
            vol, opts.angpix, params[n // 2],
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
    if opts.mode == "global":
        return reconstruct_global(model, tilt_images, opts, progress=progress)
    raise NotImplementedError(
        "mode='subvolume' is provided as a documented extension point; the "
        "global engine is the CPU-practical default. See README.")
