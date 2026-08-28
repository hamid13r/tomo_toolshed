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


def make_model(dz, dy, dx, angpix, img_y, img_x, axis_angle=85.0):
    m = TiltSeriesModel(name="phantom")
    m.volume_dims_A = np.array([dx, dy, dz], np.float32) * angpix
    m.image_dims_A = np.array([img_x, img_y], np.float32) * angpix
    angles = np.arange(-60, 61, 4).astype(np.float32)
    n = len(angles)
    m.angles = angles
    m.dose = (np.arange(n) * 3.0).astype(np.float32)
    m.use_tilt = np.ones(n, bool)
    m.axis_angles = np.full(n, axis_angle, np.float32)
    m.axis_offset_x = np.zeros(n, np.float32)
    m.axis_offset_y = np.zeros(n, np.float32)
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
        px = (transformed[:, 0] + img_center[0]) / angpix
        py = (transformed[:, 1] + img_center[1]) / angpix
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


if __name__ == "__main__":
    main()
