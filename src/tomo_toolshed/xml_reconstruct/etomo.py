"""
IMOD/etomo reconstruction engine -- real-space weighted back-projection (and
SIRT) via IMOD's `tilt`, driven by *Warp's* alignment rather than an IMOD
fiducial alignment of its own.

Why this belongs in this repo
-----------------------------
Everything else here reconstructs by CTF-weighted Fourier-slice insertion (the
central-slice theorem), which is what Warp does. `tilt` is a completely
different algorithm: real-space back-projection with a 1-D radial filter
applied to each line of each projection. Running it off the *same* alignment
and the *same* preprocessed tilt images makes the reconstruction algorithm the
only variable, so the result is directly comparable with the rest of the list.

Converting Warp's alignment to IMOD's
-------------------------------------
`geometry.tilt_matrix` builds, per tilt t,

    R_t = Euler(0, angle_t + LevelAngleY, -AxisAngle_t) @ RotateX(LevelAngleX)

and since `euler(0, b, g) == RotZ(-g) @ RotY(-b)` that is exactly

    R_t = RotZ(AxisAngle_t) @ RotY(-(angle_t + LevelAngleY)) @ RotX(LevelAngleX)

IMOD splits the same geometry into two pieces: an in-plane 2-D transform baked
into the *aligned stack* by `newstack`, and a pure tilt about the (now vertical)
Y axis performed by `tilt`. So peeling the leading RotZ off into a `.xf` leaves
precisely IMOD's model:

    .xf 2x2   = RotZ2D(-AxisAngle_t)
    .xf shift = -RotZ2D(-AxisAngle_t) @ (projected tomogram centre - image centre)
    .tlt      = angle_t + LevelAngleY
    XAXISTILT = LevelAngleX

This is not guesswork. For this project's data Warp's alignment was *imported
from* etomo (`warp_tiltseries/tiltstack/<series>/`), so the conversion can be --
and was -- checked against IMOD's own files for VLP3x3_p03_ts_002:

  * IMOD `.tlt` equals Warp's `<Angles>` exactly (-58.96, -56.07, -53.19, ...).
  * IMOD `.xf` 2x2 equals RotZ2D(-AxisAngle) to 5 decimals: the file holds
    `0.1643666 0.9863998 -0.9863998 0.1643666`; cos/sin(80.53956 deg) =
    0.164367 / 0.986400.
  * IMOD `.xf` shift equals -A @ AxisOffset / angpix: the direction matches
    exactly, both components sharing one scale factor of 2.363 = 15/6.348
    (that `.xf` belongs to the bin-6 stack; the `.st` on disk was later
    rebinned to 15 A/px).

The shift here is derived from `geometry.positions_in_all_tilts` rather than
from `AxisOffsetX/Y` directly, which is strictly more complete: it also picks up
the volume-warp and local-motion grids, and it guarantees the tomogram centre
lands at the aligned-stack centre exactly as `reconstruct._insert_all_tilts`
places it via `_extract_patch`.

What "weighted" means here, and what this engine does NOT do
------------------------------------------------------------
`tilt`'s "weighted" back-projection is weighted in two senses, neither of which
is cos(tilt):

  * the **radial R-weighting** ramp filter applied per projection line in
    frequency space (`RADIAL cutoff falloff`), and
  * **`-DENSWEIGHT`** (on by default), a per-view weight proportional to the
    local average *tilt increment* between views -- so for an equal-angle series
    it is essentially uniform.

(`-COSINTERP` is unrelated: it is cosine *stretching*, a back-projection speed
optimization that pre-stretches each input line by 1/cos(theta) so it registers
with the output planes. It is disabled on the GPU and is not a weighting.)

The Fourier engine, by contrast, applies `Scale = cos(theta)` per tilt
(`weighting.warp_weighting`, from Warp's `GetCTFsForOneParticle`). To close that
gap, `EtomoOptions.view_weight="warp"` feeds the active weighting_fn's per-tilt
amplitude scale to `tilt` via its `-WeightFile`. Only the *scalar* part of a
weighting scheme can travel that way: the dose exposure filter is a
frequency-dependent B-factor envelope and has no one-number-per-view
representation, so it is not transferred.

`tilt` still has no notion of a CTF. So unlike the Fourier engine, the etomo
path applies **no phase flip and no |CTF| weighting**, and no dose B-factor.
That is a real, deliberate difference between the two entries in the comparison
list, not an oversight. The per-tilt *preprocessing* is shared
(`reconstruct._preprocess_tilts`: Fourier-crop to the target pixel size,
background subtract, edge mask, high-pass, invert), so the tilt data going in is
bit-identical to what the Fourier engine inserts.
"""
from __future__ import annotations
from dataclasses import dataclass
import os
import subprocess
import numpy as np

from . import geometry as geo
from .mrc_io import read_mrc, write_mrc

DEG = np.pi / 180.0


@dataclass
class EtomoOptions:
    """Knobs for the IMOD `tilt` back-projection."""
    recon: str = "wbp"                     # "wbp" | "fakesirt" | "sirt"
    sirt_iters: int = 10                   # iterations for fakesirt / sirt
    radial: tuple = (0.35, 0.035)          # WBP radial filter (etomo's default)
    sirt_radial: tuple = (0.40, 0.035)     # radial filter for true SIRT (sirtsetup's)
    xaxistilt_sign: float = 1.0            # sign applied to LevelAngleX -> XAXISTILT
    view_weight: str = "none"              # "none" = pure IMOD; "warp" = feed the
                                           # weighting_fn's per-tilt amplitude scale
                                           # (cos(theta) by default) via tilt's WeightFile
    local_motion: bool = True              # bake Warp's GridMovementX/Y into the stack
                                           # (see bake_local_motion)
    gpu: int = -1                          # <0 = CPU; 0 = best available; N = GPU N
    workdir: str | None = None             # where the IMOD project is written
    imod_dir: str | None = None


# --------------------------------------------------------------------------- #
# IMOD plumbing                                                               #
# --------------------------------------------------------------------------- #
def _imod_env(imod_dir=None):
    env = dict(os.environ)
    d = imod_dir or env.get("IMOD_DIR") or "/usr/local/IMOD"
    env["IMOD_DIR"] = d
    env["PATH"] = os.path.join(d, "bin") + os.pathsep + env.get("PATH", "")
    env["LD_LIBRARY_PATH"] = (os.path.join(d, "lib") + os.pathsep
                              + env.get("LD_LIBRARY_PATH", ""))
    env["IMOD_OUTPUT_FORMAT"] = "MRC"      # keep tilt from emitting HDF
    # tilt / newstack are OpenMP; hold them to --threads like everything else
    from .parallel import get_threads
    env["OMP_NUM_THREADS"] = str(get_threads())
    return env, d


def _run(cmd, cwd, env, stdin_text=None, progress=print):
    progress(f"  $ {os.path.basename(cmd[0])} {' '.join(cmd[1:])}")
    p = subprocess.run(cmd, cwd=cwd, env=env, input=stdin_text, text=True,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if p.returncode != 0:
        raise RuntimeError(f"{os.path.basename(cmd[0])} failed (exit "
                           f"{p.returncode}):\n{p.stdout}")
    return p.stdout


def _reorient(full):
    """`tilt -PERPENDICULAR` writes one section per specimen Y, i.e. shape
    (Y, Z, X) as mrcfile reads it. Bring it to the (Z, Y, X) convention Warp
    writes -- the numpy equivalent of IMOD's `clip rotx` / `trimvol -rx`.

    The bare Y<->Z transpose is already the right handedness: no axis flip is
    needed. That is not obvious from IMOD's docs, so it is pinned by
    `validate.test_etomo`, which searches all 8 axis flips against a phantom of
    known 3-D structure and fails if anything other than the identity wins.
    """
    return np.ascontiguousarray(full.transpose(1, 0, 2))


# --------------------------------------------------------------------------- #
# Warp alignment -> IMOD .xf / .tlt                                           #
# --------------------------------------------------------------------------- #
def warp_alignment_to_imod(model, opts, xy_A, img_shape, tilt_idx):
    """Return (xf_rows, tlt_angles) for the tilts in `tilt_idx`.

    xy_A     : (NTilts, 2) projected tomogram centre per tilt, Angstrom, from
               geometry.positions_in_all_tilts.
    xf_rows  : (N,6) [a11 a12 a21 a22 dx dy], IMOD's input->output convention,
               in pixels of the binned stack.
    tlt_angles : (N,) degrees, for the .tlt file.
    """
    dsy, dsx = img_shape
    # Half-pixel conventions. Warp puts the centre of an N-pixel axis at index
    # N/2; IMOD (newstack's .xf, tilt's input and output) at (N-1)/2. So:
    #  * the .xf is applied about the raw image's IMOD centre, (N-1)/2;
    #  * the tomogram centre is sent to aligned index N/2 = IMOD centre + 0.5,
    #    which is what makes tilt's (N-1)/2-centred output voxels line up
    #    with Warp's N/2-centred ones.
    # Using Warp's N/2 for both left a (-0.54, +0.46) px error rotated by the
    # tilt axis, plus -0.5 voxel from the output centring: ~1.04 voxel in X on
    # a -94.5 deg axis series (HRR021_2_S02_L02_ts_003), enough to halve the
    # FSC vs Warp from ~60 A on. Measured against the novactf engine (which
    # uses Warp's geometry directly) the fix leaves <=0.2 voxel.
    img_center_px = np.array([(dsx - 1) / 2.0, (dsy - 1) / 2.0])

    xf_rows = np.zeros((len(tilt_idx), 6), np.float64)
    tlt = np.zeros(len(tilt_idx), np.float64)
    for i, t in enumerate(tilt_idx):
        ax = float(model.axis_angles[t]) * DEG
        c, s = np.cos(ax), np.sin(ax)
        A = np.array([[c, s], [-s, c]])              # RotZ2D(-AxisAngle)
        # the tomogram centre's projection must land on the aligned-stack centre
        d = -A @ (xy_A[t] / opts.angpix - img_center_px) + 0.5
        xf_rows[i] = (A[0, 0], A[0, 1], A[1, 0], A[1, 1], d[0], d[1])
        tlt[i] = float(model.angles[t]) + float(model.level_angle_y)
    return xf_rows, tlt


def aligned_stack_size(model, tilt_idx, dsx, dsy, Vx, Vy):
    """Size of the aligned stack newstack writes. It must hold the raw tilt
    *after* the .xf rotates its tilt axis to vertical -- not the raw size:
    for a tilt axis near +-90 deg (e.g. -94.5 on HRR021_2 data, 720x512
    tilts) the tomogram's 720-px length along the axis ends up vertical, and
    a 720x512 stack cut it to 104 rows. Bounding box of the rotated tilt,
    at least the tomogram box, rounded up to even (newstack centres the output
    on the input centre, so even sizes keep that exact)."""
    ax = np.deg2rad(np.asarray(model.axis_angles, np.float64)[tilt_idx])
    c, s = np.abs(np.cos(ax)).max(), np.abs(np.sin(ax)).max()
    bx = c * dsx + s * dsy
    by = s * dsx + c * dsy
    ev = lambda v: int(np.ceil(v / 2.0)) * 2
    return max(ev(bx), ev(Vx)), max(ev(by), ev(Vy))


def _write_xf(path, xf_rows):
    with open(path, "w") as f:
        for r in xf_rows:
            f.write(f"{r[0]:12.7f}{r[1]:12.7f}{r[2]:12.7f}{r[3]:12.7f}"
                    f"{r[4]:12.3f}{r[5]:12.3f}\n")


def _write_weights(path, weights):
    """tilt's -WeightFile: one scalar amplitude factor per view.

    NOTE this can only carry the per-tilt *scalar* part of a weighting scheme.
    The dose exposure filter is a frequency-dependent B-factor envelope, which
    has no representation in a one-number-per-view file and is therefore NOT
    transferred (see the module docstring).
    """
    with open(path, "w") as f:
        for w in weights:
            f.write(f"{w:.6f}\n")


def _write_tlt(path, angles):
    with open(path, "w") as f:
        for a in angles:
            f.write(f"{a:9.2f}\n")


# --------------------------------------------------------------------------- #
# tilt / newstack parameter blocks                                            #
# --------------------------------------------------------------------------- #
def _newstack_params(st, ali, xf, dsx, dsy):
    return [f"InputFile {st}", f"OutputFile {ali}", f"TransformFile {xf}",
            f"SizeToOutputInXandY {dsx},{dsy}", "TaperAtFill 1,0",
            "AdjustOrigin", "ImagesAreBinned 1.0"]


def _tilt_params(eo, ali, rec, tlt, dsx, dsy, Vx, Vy, Vz, xtilt, weightfile=None):
    y0 = max((dsy - Vy) // 2, 0)
    y1 = min(y0 + Vy - 1, dsy - 1)
    # True SIRT reprojects its own estimate and differences it against the input
    # projections, so `tilt` requires the output width to equal the input width
    # ("For SIRT, sizes of input projections, rec file, and width/thickness
    # entries must match"). Reconstruct full width and let the caller crop to
    # the tomogram box -- which also avoids truncating the X edges.
    width = dsx if eo.recon == "sirt" else min(Vx, dsx)
    par = [f"InputProjections {ali}", f"OutputFile {rec}", f"TILTFILE {tlt}",
           f"THICKNESS {Vz}", f"FULLIMAGE {dsx} {dsy}", "SUBSETSTART 0 0",
           "IMAGEBINNED 1", f"WIDTH {width}", f"SLICE {y0} {y1}",
           f"XAXISTILT {xtilt:.4f}", "OFFSET 0.0", "SHIFT 0.0 0.0",
           "PERPENDICULAR", "MODE 2", "AdjustOrigin"]
    if weightfile:
        par.append(f"WeightFile {weightfile}")
    if eo.recon == "sirt":
        r = eo.sirt_radial
        par += [f"RADIAL {r[0]} {r[1]}", "FalloffIsTrueSigma 1",
                f"SIRTIterations {int(eo.sirt_iters)}"]
    else:
        r = eo.radial
        par += [f"RADIAL {r[0]} {r[1]}", "FalloffIsTrueSigma 1"]
        if eo.recon == "fakesirt":
            par.append(f"FakeSIRTiterations {int(eo.sirt_iters)}")
    if eo.gpu is not None and eo.gpu >= 0:
        par += [f"UseGPU {int(eo.gpu)}", "ActionIfGPUFails 1,2"]
    return par


def _write_com(path, binary, params):
    """Write a real IMOD .com file so the run can be inspected, edited and
    re-run by hand (`submfg tilt.com`) or opened in etomo."""
    with open(path, "w") as f:
        f.write("# generated by tomo_toolshed.xml_reconstruct.etomo -- Warp alignment, IMOD tilt\n")
        f.write("$setenv IMOD_OUTPUT_FORMAT MRC\n")
        f.write(f"${binary} -StandardInput\n")
        for p in params:
            f.write(p + "\n")


# --------------------------------------------------------------------------- #
# engine                                                                      #
# --------------------------------------------------------------------------- #
def reconstruct_etomo(model, tilt_images, opts, progress=print):
    """Reconstruct with IMOD `tilt` using Warp's alignment.

    Shares `reconstruct._preprocess_tilts` with the Fourier engine, so the only
    difference between this entry and the others in the comparison list is the
    back-projection itself (see the module docstring for what `tilt` does not do).
    """
    # imported here, not at module scope: reconstruct.py dispatches into this
    # module, so a top-level import would be circular.
    from .reconstruct import (_compute_volume_grid, _preprocess_tilts, _crop_center,
                              _apply_local_motion)

    eo = getattr(opts, "etomo", None) or EtomoOptions()
    if eo.recon not in ("wbp", "fakesirt", "sirt"):
        raise ValueError(f"unknown etomo recon mode {eo.recon!r}")
    env, imod_dir = _imod_env(eo.imod_dir)
    bins = {n: os.path.join(imod_dir, "bin", n) for n in ("newstack", "tilt")}
    for n, b in bins.items():
        if not os.path.exists(b):
            raise RuntimeError(f"IMOD binary not found: {b}  (pass --imod_dir)")

    Vx, Vy, Vz, _S, _Nz = _compute_volume_grid(model, opts)
    progress(f"volume voxels: {Vx}x{Vy}x{Vz}; engine: IMOD tilt "
             f"({eo.recon}" + (f", {eo.sirt_iters} iters"
                               if eo.recon != "wbp" else "") + ")")

    scaled, size_rounding = _preprocess_tilts(model, tilt_images, opts)
    dsy, dsx = scaled[0].shape

    tilt_idx = [t for t in range(model.n_tilts) if bool(model.use_tilt[t])]
    if len(tilt_idx) < model.n_tilts:
        progress(f"  excluding {model.n_tilts - len(tilt_idx)} tilt(s) with "
                 f"UseTilt=False from the stack")

    center = model.volume_dims_A / 2.0
    xy_A, defocus_um = geo.positions_in_all_tilts(model, center, size_rounding)
    # the .xf carries the global alignment only when the local motion is baked
    # into the images -- else the centre's local shift would be applied twice
    scaled, xy_A = _apply_local_motion(model, scaled, size_rounding, xy_A, center,
                                       opts.angpix, eo.local_motion, progress)
    xf_rows, tlt_angles = warp_alignment_to_imod(
        model, opts, xy_A, (dsy, dsx), tilt_idx)
    adsx, adsy = aligned_stack_size(model, tilt_idx, dsx, dsy, Vx, Vy)
    if (adsx, adsy) != (dsx, dsy):
        progress(f"  aligned stack {adsx}x{adsy} (raw tilts {dsx}x{dsy}, rotated by "
                 f"the tilt-axis angle)")
    xtilt = eo.xaxistilt_sign * float(model.level_angle_x)

    # optional per-view amplitude weights via tilt's WeightFile
    weights = None
    if eo.view_weight == "warp":
        wfn = getattr(opts, "_weighting_fn", None)
        if wfn is None:
            from .weighting import warp_weighting as wfn
        params = wfn(model, center, defocus_um, opts.use_global_weights)
        weights = np.array([float(params[t].scale) for t in tilt_idx])
        progress(f"  per-view WeightFile from the weighting_fn's amplitude scale: "
                 f"{weights.min():.3f}..{weights.max():.3f} "
                 f"(highest tilt gets {100 * weights.min() / weights.max():.0f}% "
                 f"of the flattest)")
    elif eo.view_weight != "none":
        raise ValueError(f"unknown etomo view_weight {eo.view_weight!r}")

    work = eo.workdir or os.path.join(os.getcwd(), "etomo_" + model.name)
    os.makedirs(work, exist_ok=True)
    base = model.name
    st, ali = base + ".st", base + "_ali.mrc"
    xf, tltf = base + ".xf", base + ".tlt"
    rec = base + "_full_rec.mrc"
    wf = base + ".weights" if weights is not None else None

    progress(f"  IMOD project: {work}")
    write_mrc(os.path.join(work, st),
              np.stack([scaled[t] for t in tilt_idx]).astype(np.float32),
              opts.angpix)
    _write_xf(os.path.join(work, xf), xf_rows)
    _write_tlt(os.path.join(work, tltf), tlt_angles)
    if wf:
        _write_weights(os.path.join(work, wf), weights)

    ns_par = _newstack_params(st, ali, xf, adsx, adsy)
    _write_com(os.path.join(work, "newst.com"), "newstack", ns_par)
    _run([bins["newstack"], "-StandardInput"], work, env,
         "\n".join(ns_par) + "\n", progress)

    t_par = _tilt_params(eo, ali, rec, tltf, adsx, adsy, Vx, Vy, Vz, xtilt, wf)
    _write_com(os.path.join(work, "tilt.com"), "tilt", t_par)
    try:
        _run([bins["tilt"], "-StandardInput"], work, env,
             "\n".join(t_par) + "\n", progress)
    except RuntimeError as err:
        # true SIRT needs vertical slices when the specimen is X-tilted; if
        # this build refuses, fall back to a flat specimen and say so loudly
        # rather than silently reporting a reconstruction with wrong geometry.
        if eo.recon == "sirt" and abs(xtilt) > 1e-6:
            progress(f"  ! tilt failed with XAXISTILT {xtilt:+.4f}: {err}")
            progress("  ! retrying true SIRT with XAXISTILT 0 -- the "
                     "specimen X-tilt is NOT applied in this variant")
            t_par = _tilt_params(eo, ali, rec, tltf, adsx, adsy, Vx, Vy, Vz, 0.0, wf)
            _write_com(os.path.join(work, "tilt.com"), "tilt", t_par)
            _run([bins["tilt"], "-StandardInput"], work, env,
                 "\n".join(t_par) + "\n", progress)
        else:
            raise

    full, _vs = read_mrc(os.path.join(work, rec))
    vol = _reorient(full)
    if vol.shape != (Vz, Vy, Vx):
        vol = _crop_center(vol, (Vz, Vy, Vx))
    progress(f"  tilt output {full.shape} (Y,Z,X) -> {vol.shape} (Z,Y,X)")
    return {"reconstruction": vol}
