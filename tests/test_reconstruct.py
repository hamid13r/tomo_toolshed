"""Reconstruction engine: self-consistency and a full end-to-end run."""
from __future__ import annotations

import copy

import numpy as np

from warp_recon import load_tiltseries_xml, reconstruct, ReconOptions
from warp_recon.mrc_io import read_mrc


def test_self_consistency_phantom_recovered():
    """Forward-project a phantom through the exact Warp geometry, reconstruct,
    and check the phantom is recovered (missing-wedge-limited)."""
    from warp_recon.metadata import TiltSeriesModel
    from warp_recon.ctf import CTFParams
    from warp_recon import geometry as geo

    dz, dy, dx = 32, 48, 48
    angpix = 10.0
    # phantom: a handful of gaussian blobs
    rng = np.random.default_rng(0)
    zz, yy, xx = np.mgrid[0:dz, 0:dy, 0:dx]
    phantom = np.zeros((dz, dy, dx), np.float32)
    for _ in range(12):
        cz, cy, cx = rng.uniform(0.25, 0.75, 3) * (dz, dy, dx)
        r = rng.uniform(2, 5)
        phantom += np.exp(-(((zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2)
                            / (2 * r * r))).astype(np.float32)

    m = TiltSeriesModel(name="phantom")
    m.volume_dims_A = np.array([dx, dy, dz], np.float32) * angpix
    m.image_dims_A = np.array([80, 80], np.float32) * angpix
    angles = np.arange(-60, 61, 4).astype(np.float32)
    n = len(angles)
    m.angles = angles
    m.dose = (np.arange(n) * 3.0).astype(np.float32)
    m.use_tilt = np.ones(n, bool)
    m.axis_angles = np.full(n, 85.0, np.float32)
    m.axis_offset_x = np.zeros(n, np.float32)
    m.axis_offset_y = np.zeros(n, np.float32)
    m.ctf = CTFParams(pixel_size=angpix, voltage=300, cs=2.7, amplitude=1.0, defocus=0.0)

    # forward project with the exact geometry
    imgs = [np.zeros((80, 80), np.float32) for _ in range(n)]
    coords = np.stack([xx.ravel() + 0.5, yy.ravel() + 0.5, zz.ravel() + 0.5], 1) * angpix
    vals = phantom.ravel()
    for t in range(n):
        R = geo.tilt_matrix(m, t)
        centered = coords - m.volume_dims_A / 2.0
        tr = centered @ R.T
        px = (tr[:, 0] + m.image_dims_A[0] / 2.0) / angpix
        py = (tr[:, 1] + m.image_dims_A[1] / 2.0) / angpix
        x0 = np.floor(px).astype(np.int64); y0 = np.floor(py).astype(np.int64)
        fx = px - x0; fy = py - y0
        for dyi in (0, 1):
            wy = fy if dyi else 1 - fy
            yidx = y0 + dyi
            for dxi in (0, 1):
                wx = fx if dxi else 1 - fx
                xidx = x0 + dxi
                mask = (xidx >= 0) & (xidx < 80) & (yidx >= 0) & (yidx < 80)
                np.add.at(imgs[t], (yidx[mask], xidx[mask]), (vals * wx * wy)[mask])

    def flat_weighting(model, coord_phys, defocus_um, use_global_weights=False):
        out = []
        for _ in range(model.n_tilts):
            p = copy.copy(model.ctf)
            p.defocus = 0.0; p.amplitude = 1.0; p.scale = 1.0; p.bfactor = 0.0
            out.append(p)
        return out

    opts = ReconOptions(angpix=angpix, raw_angpix=angpix, invert=False,
                        normalize=False, do_deconv=False, pad_factor=1.8, mode="global")
    out = reconstruct(m, imgs, opts, weighting_fn=flat_weighting, progress=lambda *_: None)
    rec = out["reconstruction"]

    assert rec.shape == (dz, dy, dx)
    a = phantom - phantom.mean()
    b = rec - rec.mean()
    corr = float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-9))
    assert corr > 0.7, f"geometry/insertion correlation too low: {corr:.3f}"


def test_end_to_end_cli_path(synthetic_series, tmp_path):
    """Drive the public API the CLI uses: parse XML, load the stack, reconstruct,
    and write an MRC. Asserts a finite, non-trivial volume at the right box."""
    xml_path, stack_path, n = synthetic_series
    model = load_tiltseries_xml(str(xml_path))
    stack, raw_px = read_mrc(str(stack_path))
    assert raw_px == 10.0
    assert stack.shape[0] == n
    tilt_images = [stack[i] for i in range(stack.shape[0])]

    angpix = 20.0
    opts = ReconOptions(angpix=angpix, raw_angpix=raw_px, do_deconv=True)
    out = reconstruct(model, tilt_images, opts, progress=lambda *_: None)
    rec = out["reconstruction"]

    # box: 1600/1600/400 A at 20 A -> 80 x 80 x 20 voxels
    assert rec.shape == (20, 80, 80)
    assert np.isfinite(rec).all()
    assert rec.std() > 0, "reconstruction is flat (all images masked to zero?)"
    assert "deconv" in out and np.isfinite(out["deconv"]).all()
