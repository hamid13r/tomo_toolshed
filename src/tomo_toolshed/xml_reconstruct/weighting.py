"""
Per-tilt weighting model -- THE place to experiment with "different weighting".

`warp_weighting` reproduces TiltSeries.GetCTFsForOneParticle exactly:

    Scale (amplitude weight):
        if no dose-weight grid:  Scale = cos(tilt_angle)
        else:                    Scale = GridDoseWeights(t) * GridLocationWeights(x,y,t)
        Scale *= (UseTilt ? 1 : 1e-4)
        (optionally *= GlobalWeight)

    Bfactor (dose exposure filter, Angstrom^2, negative attenuates high freq):
        if no dose-bfac grid:    Bfactor = -Dose[t] * 4
        else:                    Bfactor = min(GridDoseBfacs(t), -Dose[t]*3) + GridLocationBfacs(x,y,t)
        (optionally += GlobalBfactor)
        BfactorDelta / BfactorAngle from their grids.

Each function takes (model, coord_phys, defocus_um, use_global_weights) and
returns a list[CTFParams] (one per tilt) with `defocus`, `scale`, `bfactor`,
`bfactor_delta`, `bfactor_angle` filled in on top of the base CTF (Cs, voltage,
amplitude, pixel size, phase). Astigmatism (defocus_delta/angle) is pulled from
the model's per-tilt grids too.

To make a NEW weighting scheme, copy `warp_weighting`, change the Scale/Bfactor
math, and pass it to reconstruct(..., weighting_fn=my_weighting).
"""
from __future__ import annotations
import copy
import numpy as np
from .ctf import CTFParams

DEG = np.pi / 180.0


def _grid_is_trivial(grid):
    return int(np.prod(grid.dims)) <= 1


def warp_weighting(model, coord_phys, defocus_um, use_global_weights=False):
    """Exact reproduction of Warp's per-tilt weighted CTF parameters."""
    n = model.n_tilts
    Vx, Vy = model.volume_dims_A[0], model.volume_dims_A[1]
    gstep = 1.0 / max(n - 1, 1)

    # normalized coords for the location grids
    loc = np.stack([np.full(n, coord_phys[0] / Vx),
                    np.full(n, coord_phys[1] / Vy),
                    np.arange(n) * gstep], axis=1)
    tcoords = np.stack([np.full(n, 0.5), np.full(n, 0.5),
                        np.arange(n) * gstep], axis=1)

    # per-tilt astigmatism from grids
    ddelta = model.grid_ctf_defocus_delta.interpolate(tcoords)
    dangle = model.grid_ctf_defocus_angle.interpolate(tcoords)
    phase = model.grid_ctf_phase.interpolate(tcoords)

    # --- Scale ---
    if _grid_is_trivial(model.grid_dose_weights):
        scale = np.cos(model.angles * DEG)
    else:
        scale = (model.grid_dose_weights.interpolate(tcoords)
                 * model.grid_location_weights.interpolate(loc))
    scale = scale * np.where(model.use_tilt, 1.0, 1e-4)

    # --- Bfactor ---
    if _grid_is_trivial(model.grid_dose_bfacs):
        bfac = -model.dose * 4.0
    else:
        bfac = (np.minimum(model.grid_dose_bfacs.interpolate(tcoords), -model.dose * 3.0)
                + model.grid_location_bfacs.interpolate(loc))
    bdelta = model.grid_dose_bfacs_delta.interpolate(tcoords)
    bangle = model.grid_dose_bfacs_angle.interpolate(tcoords)

    if use_global_weights:
        bfac = bfac + model.global_bfactor
        scale = scale * model.global_weight

    out = []
    for t in range(n):
        p = copy.copy(model.ctf)                 # base Cs/V/amp/pixel
        p.defocus = float(defocus_um[t])
        p.defocus_delta = float(ddelta[t])
        p.defocus_angle = float(dangle[t])
        p.phase_shift = float(phase[t]) if not _grid_is_trivial(model.grid_ctf_phase) else p.phase_shift
        p.scale = float(scale[t])
        p.bfactor = float(bfac[t])
        p.bfactor_delta = float(bdelta[t])
        p.bfactor_angle = float(bangle[t])
        out.append(p)
    return out


# ------------------------------------------------------------------ #
# Example custom scheme: a Grant-&-Grigorieff-style exposure filter   #
# with a tunable dose-to-Bfactor factor (replaces the hard-coded 4).  #
# ------------------------------------------------------------------ #
def make_dose_bfactor_weighting(dose_bfactor_scale=4.0, cos_weighting=True):
    def fn(model, coord_phys, defocus_um, use_global_weights=False):
        params = warp_weighting(model, coord_phys, defocus_um, use_global_weights)
        for t, p in enumerate(params):
            p.bfactor = -model.dose[t] * dose_bfactor_scale
            if not cos_weighting:
                p.scale = 1.0 * (1.0 if model.use_tilt[t] else 1e-4)
        return params
    return fn


# ------------------------------------------------------------------ #
# MotionCor3-style dose weighting: the Grant & Grigorieff (2015)      #
# critical-exposure curve MotionCor3 applies per FRAME, applied here  #
# per TILT using each tilt's accumulated dose.                        #
# ------------------------------------------------------------------ #
def motioncor3_dose_weighting(model, coord_phys, defocus_um, use_global_weights=False):
    """Same Scale/geometry as `warp_weighting`, but replaces Warp's linear
    dose Bfactor (`Bfactor = -Dose*4`, a single Gaussian falloff) with
    MotionCor3's per-frame critical-exposure curve (Correct/GWeightFrame.cu),
    applied here per tilt via each tilt's accumulated dose
    (`model.dose[t]`, the same quantity Warp's own Bfactor model uses).

    The curve was fit directly to measured resolution-dependent radiation
    damage (Grant & Grigorieff, eLife 2015) rather than approximated by one
    Gaussian, and falls off more slowly at low dose / high resolution -- the
    aim is to let more high-resolution signal from low-dose (near-zero-tilt)
    images survive into the reconstruction. See ctf.motioncor3_dose_envelope
    for the actual curve.
    """
    params = warp_weighting(model, coord_phys, defocus_um, use_global_weights)
    for t, p in enumerate(params):
        p.dose_model = "motioncor3"
        p.dose_ea2 = float(model.dose[t])
    return params
