#!/usr/bin/env python
"""
Self-consistency test (no Warp data needed).

Builds a phantom volume, forward-projects it into tilt images using Warp's exact
geometry (GetPositionInAllTilts), then runs the reconstruction engine and checks
that the phantom is recovered. This validates the geometry sign conventions and
the Fourier-insertion / weighting normalization end to end.

A high correlation (structure recovered, blurred only by the missing wedge)
means the pipeline is internally correct. Validating that it matches *Warp's*
output bit-for-bit additionally requires a real project + a ts_reconstruct
tomogram (see README, "Validating against Warp").
"""
from __future__ import annotations
import copy
import os
import numpy as np

from warp_recon.metadata import TiltSeriesModel
from warp_recon.ctf import CTFParams
from warp_recon import geometry as geo
from warp_recon.reconstruct import reconstruct, ReconOptions


def make_phantom(dz, dy, dx):
    vol = np.zeros((dz, dy, dx), np.float32)
    rng = np.random.default_rng(0)
    zz, yy, xx = np.mgrid[0:dz, 0:dy, 0:dx]
    for _ in range(12):
        cz, cy, cx = rng.uniform(0.25, 0.75, 3) * (dz, dy, dx)
        r = rng.uniform(2, 5)
        vol += np.exp(-(((zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2) / (2 * r * r))).astype(np.float32)
    return vol


def make_model(dz, dy, dx, angpix, img_y, img_x, axis_angle=85.0,
               level_x=0.0, level_y=0.0, axis_jitter=0.0, offset_scale=0.0):
    m = TiltSeriesModel(name="phantom")
    m.volume_dims_A = np.array([dx, dy, dz], np.float32) * angpix
    m.image_dims_A = np.array([img_x, img_y], np.float32) * angpix
    angles = np.arange(-60, 61, 4).astype(np.float32)
    n = len(angles)
    m.angles = angles
    m.dose = (np.arange(n) * 3.0).astype(np.float32)
    m.use_tilt = np.ones(n, bool)
    m.level_angle_x = level_x
    m.level_angle_y = level_y
    rng = np.random.default_rng(7)
    m.axis_angles = (axis_angle + rng.uniform(-axis_jitter, axis_jitter, n)).astype(np.float32)
    m.axis_offset_x = (rng.uniform(-offset_scale, offset_scale, n)).astype(np.float32)
    m.axis_offset_y = (rng.uniform(-offset_scale, offset_scale, n)).astype(np.float32)
    m.ctf = CTFParams(pixel_size=angpix, voltage=300, cs=2.7,
                      amplitude=1.0, defocus=0.0)   # amplitude=1 -> CTF≈1 for test
    return m


def forward_project(model, phantom, angpix):
    """Splat-forward-project the phantom into tilt images using the exact
    Warp geometry positions_in_all_tilts (bilinear splat)."""
    dz, dy, dx = phantom.shape
    img_x = int(round(model.image_dims_A[0] / angpix))
    img_y = int(round(model.image_dims_A[1] / angpix))
    n = model.n_tilts
    imgs = [np.zeros((img_y, img_x), np.float32) for _ in range(n)]

    # voxel physical coords (x,y,z) at voxel centers
    zz, yy, xx = np.mgrid[0:dz, 0:dy, 0:dx]
    coords = np.stack([xx.ravel() + 0.5, yy.ravel() + 0.5, zz.ravel() + 0.5], axis=1) * angpix
    vals = phantom.ravel()

    for t in range(n):
        # positions_in_all_tilts is per-coord; vectorize by looping matrix here
        R = geo.tilt_matrix(model, t)
        center = model.volume_dims_A / 2.0
        img_center = model.image_dims_A / 2.0
        centered = coords - center
        transformed = centered @ R.T   # (M,3); rows R applied: (R@c) = c@R.T
        px = (transformed[:, 0] + model.axis_offset_x[t] + img_center[0]) / angpix
        py = (transformed[:, 1] + model.axis_offset_y[t] + img_center[1]) / angpix
        _bilinear_splat(imgs[t], px, py, vals)
    return imgs


def _bilinear_splat(img, px, py, vals):
    ny, nx = img.shape
    x0 = np.floor(px).astype(np.int64); y0 = np.floor(py).astype(np.int64)
    fx = px - x0; fy = py - y0
    for dyi in (0, 1):
        wy = fy if dyi else 1 - fy
        yy = y0 + dyi
        for dxi in (0, 1):
            wx = fx if dxi else 1 - fx
            xx = x0 + dxi
            m = (xx >= 0) & (xx < nx) & (yy >= 0) & (yy < ny)
            np.add.at(img, (yy[m], xx[m]), (vals * wx * wy)[m])


def flat_ctf_weighting(model, coord_phys, defocus_um, use_global_weights=False):
    out = []
    for t in range(model.n_tilts):
        p = copy.copy(model.ctf)
        p.defocus = 0.0
        p.amplitude = 1.0     # -> CTF = cos(0) = 1
        p.scale = 1.0
        p.bfactor = 0.0
        out.append(p)
    return out


def _corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-9))


def _flip_table(phantom, rec):
    """Correlation of every axis-flip of `rec` against the phantom. Used to
    pin IMOD's output handedness rather than assume it (see
    warp_recon.etomo._reorient)."""
    out = {}
    for fz in (1, -1):
        for fy in (1, -1):
            for fx in (1, -1):
                out[(fz, fy, fx)] = _corr(phantom, rec[::fz, ::fy, ::fx])
    return out


def test_etomo(verbose=True):
    """Same phantom, same Warp geometry, but reconstructed by IMOD `tilt`.

    This is the end-to-end certificate for the Warp -> IMOD alignment
    conversion in warp_recon/etomo.py: the .xf / .tlt / XAXISTILT are consumed
    by the real `newstack` and `tilt` binaries, so a wrong sign or a wrong
    shift convention shows up as a collapsed correlation. The model
    deliberately carries a non-zero LevelAngleX/Y, per-tilt tilt-axis jitter
    and per-tilt axis offsets, since those are exactly the parts of the
    conversion that are not fixed by inspection.
    """
    import shutil
    from warp_recon.etomo import EtomoOptions, _imod_env

    _env, imod_dir = _imod_env()
    if not os.path.exists(os.path.join(imod_dir, "bin", "tilt")):
        print(f"SKIP etomo test: no IMOD at {imod_dir}")
        return None

    dz, dy, dx = 32, 48, 48
    angpix = 10.0
    phantom = make_phantom(dz, dy, dx)
    model = make_model(dz, dy, dx, angpix, img_y=80, img_x=80,
                       axis_angle=85.0, level_x=1.5, level_y=-3.0,
                       axis_jitter=0.4, offset_scale=25.0)
    imgs = forward_project(model, phantom, angpix)

    workdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "_selftest", "etomo")
    shutil.rmtree(workdir, ignore_errors=True)

    def opts_for(mode, **kw):
        return ReconOptions(angpix=angpix, raw_angpix=angpix,
                            invert=False, normalize=False, do_deconv=False,
                            pad_factor=1.8, mode=mode, **kw)

    quiet = (lambda *_: None) if not verbose else print
    ref = reconstruct(copy.deepcopy(model), imgs, opts_for("global"),
                      weighting_fn=flat_ctf_weighting,
                      progress=lambda *_: None)["reconstruction"]
    rec = reconstruct(copy.deepcopy(model), imgs,
                      opts_for("etomo", etomo=EtomoOptions(recon="wbp", workdir=workdir)),
                      weighting_fn=flat_ctf_weighting, progress=quiet)["reconstruction"]

    print(f"phantom shape {phantom.shape}, fourier {ref.shape}, etomo {rec.shape}")
    if rec.shape != phantom.shape:
        print(f"FAIL: etomo box {rec.shape} != phantom {phantom.shape}")
        return False

    table = _flip_table(phantom, rec)
    best = max(table, key=table.get)
    c_id, c_best = table[(1, 1, 1)], table[best]
    print(f"correlation  fourier engine : {_corr(phantom, ref):.3f}")
    print(f"correlation  etomo (tilt)   : {c_id:.3f}   [as returned]")
    print(f"best axis flip {best} -> {c_best:.3f}")
    if best != (1, 1, 1):
        print("  flip table (z,y,x): "
              + ", ".join(f"{k}={v:+.3f}" for k, v in sorted(table.items(),
                                                             key=lambda kv: -kv[1])[:4]))
        print("FAIL: etomo output orientation is wrong -- fix "
              "warp_recon.etomo._reorient to apply this flip")
        return False
    ok = c_id > 0.55
    print("PASS (etomo geometry)" if ok else
          "CHECK ETOMO GEOMETRY (low correlation)")
    return ok


def test_novactf(verbose=True):
    """Same jittered phantom through the real-space novaCTF engine
    (warp_recon/novactf.py). Two checks: the vectorized per-voxel geometry
    (positions_one_tilt) must equal positions_in_all_tilts exactly, and the
    back-projection must recover the phantom in the identity orientation --
    its geometry is Warp's own, so no axis flip may be needed."""
    from warp_recon.novactf import NovaCTFOptions

    dz, dy, dx = 32, 48, 48
    angpix = 10.0
    phantom = make_phantom(dz, dy, dx)
    model = make_model(dz, dy, dx, angpix, img_y=80, img_x=80,
                       axis_angle=85.0, level_x=1.5, level_y=-3.0,
                       axis_jitter=0.4, offset_scale=25.0)
    imgs = forward_project(model, phantom, angpix)

    pts = np.random.default_rng(3).uniform(0, 1, (16, 3)) * model.volume_dims_A
    err = 0.0
    for p in pts:
        xy, d = geo.positions_in_all_tilts(model, p)
        for t in range(model.n_tilts):
            x, y, dd = geo.positions_one_tilt(model, t, p[None])
            err = max(err, abs(x[0] - xy[t, 0]), abs(y[0] - xy[t, 1]), abs(dd[0] - d[t]))

    opts = ReconOptions(angpix=angpix, raw_angpix=angpix, invert=False,
                        normalize=False, do_deconv=False, mode="novactf",
                        novactf=NovaCTFOptions(correction="none", geometry_step=4))
    rec = reconstruct(copy.deepcopy(model), imgs, opts, weighting_fn=flat_ctf_weighting,
                      progress=print if verbose else (lambda *_: None))["reconstruction"]
    c_id = _corr(phantom, rec)
    best, c_best = max(_flip_table(phantom, rec).items(), key=lambda kv: kv[1])
    print(f"positions_one_tilt vs positions_in_all_tilts: max |diff| {err:.2e}")
    print(f"correlation  novactf engine : {c_id:.3f}   best axis flip {best} -> {c_best:.3f}")
    ok = err < 1e-6 and best == (1, 1, 1) and c_id > 0.55
    print("PASS (novactf geometry)" if ok else "CHECK NOVACTF ENGINE")
    return ok


def main():
    dz, dy, dx = 32, 48, 48
    angpix = 10.0
    phantom = make_phantom(dz, dy, dx)
    model = make_model(dz, dy, dx, angpix, img_y=80, img_x=80)
    imgs = forward_project(model, phantom, angpix)

    opts = ReconOptions(angpix=angpix, raw_angpix=angpix,
                        invert=False, normalize=False, do_deconv=False,
                        pad_factor=1.8, mode="global")
    out = reconstruct(model, imgs, opts, weighting_fn=flat_ctf_weighting,
                      progress=lambda *_: None)
    rec = out["reconstruction"]

    # compare (both are (dz,dy,dx))
    a = phantom - phantom.mean()
    b = rec - rec.mean()
    corr = float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-9))
    print(f"phantom shape {phantom.shape}, recon shape {rec.shape}")
    print(f"correlation (phantom vs reconstruction): {corr:.3f}")
    # missing-wedge blur limits this; internal consistency should be clearly high
    print("PASS" if corr > 0.55 else "CHECK GEOMETRY (low correlation)")

    print("\n--- IMOD/etomo engine ---")
    test_etomo(verbose=False)

    print("\n--- real-space novaCTF engine ---")
    test_novactf(verbose=False)


if __name__ == "__main__":
    main()
