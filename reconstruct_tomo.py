#!/usr/bin/env python
"""
Command-line front-end, mirroring WarpTools `ts_reconstruct`.

Example
-------
    python reconstruct_tomo.py \
        --xml   warp_tiltseries/TS_01.xml \
        --tomostar warp_tiltseries/tomostar/TS_01.tomostar \
        --tilt_stack aligned_stacks/TS_01.st \
        --angpix 10 \
        --output reconstruction \
        --deconv

Tilt-image input (choose one):
    --tilt_stack FILE   a 3D MRC/ST stack, slices ordered like the model's tilts
    --tilt_dir  DIR     folder of per-tilt averages named <movie-root>.mrc
                        (matched to the .xml MoviePath entries)

If --raw_angpix is omitted it is read from the tilt-image MRC header.
"""
from __future__ import annotations
import argparse
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np

import numpy as np
from warp_recon import load_tiltseries_xml, reconstruct, ReconOptions
from warp_recon.etomo import EtomoOptions
from warp_recon.novactf import NovaCTFOptions
from warp_recon.parallel import get_threads, set_threads
from warp_recon import warp_weighting, make_dose_bfactor_weighting, motioncor3_dose_weighting
from warp_recon.mrc_io import read_mrc, write_mrc, write_png_slice
from warp_recon.metadata import read_settings_pixelsize, read_settings_tomo_dims


def _load_one_tilt(tilt_dir, movie_path):
    root = Path(movie_path).stem
    cand = list(tilt_dir.glob(root + ".mrc")) + list(tilt_dir.glob(root + ".*"))
    if not cand:
        raise SystemExit(f"could not find tilt image for '{root}' in {tilt_dir}")
    im, v = read_mrc(cand[0])
    if im.ndim == 3:
        im = im[0]
    return im, v


def load_tilt_images(model, args):
    if args.tilt_stack:
        data, vs = read_mrc(args.tilt_stack)
        if data.ndim != 3:
            raise SystemExit("--tilt_stack must be a 3D stack")
        if data.shape[0] != model.n_tilts:
            raise SystemExit(
                f"stack has {data.shape[0]} slices but model has {model.n_tilts} tilts")
        imgs = [data[i] for i in range(data.shape[0])]
        return imgs, vs
    if args.tilt_dir:
        d = Path(args.tilt_dir)
        # each tilt is an independent file read (disk I/O releases the GIL),
        # so a thread pool gives a near-linear speedup over reading serially
        with ThreadPoolExecutor(max_workers=max(1, min(16, len(model.movie_paths), get_threads()))) as ex:
            results = list(ex.map(lambda mp: _load_one_tilt(d, mp), model.movie_paths))
        imgs = [im for im, _ in results]
        vs = next((v for _, v in results if v is not None), None)
        return imgs, vs
    raise SystemExit("provide --tilt_stack or --tilt_dir")


def main():
    ap = argparse.ArgumentParser(description="Faithful ts_reconstruct re-implementation")
    ap.add_argument("--xml", required=True, help="per-tilt-series .xml (Warp metadata)")
    ap.add_argument("--tomostar", help=".tomostar (fallback for raw angles/dose)")
    ap.add_argument("--settings", help="Warp .settings (for raw pixel size)")
    ap.add_argument("--angpix", type=float, required=True,
                    help="OUTPUT tomogram pixel size (A) -- this is the knob that "
                         "sets the tomogram sampling, same as ts_reconstruct --angpix")
    ap.add_argument("--raw_angpix", type=float, help="unbinned tilt pixel size (A)")
    ap.add_argument("--dimensions", type=int, nargs=3, metavar=("X", "Y", "Z"),
                    help="tomogram box in UNBINNED pixels (like Warp's <Tomo> "
                         "Dimensions). Overrides the .settings / .xml box.")
    ap.add_argument("--tilt_stack", help="3D MRC/ST stack ordered like the tilts")
    ap.add_argument("--tilt_dir", help="folder of per-tilt averages <root>.mrc")
    ap.add_argument("--output", default="reconstruction", help="output folder")
    ap.add_argument("--threads", type=int, default=0,
                    help="CPU threads for everything parallel: FFTs, numba kernels, "
                         "thread pools, and IMOD tilt (OMP_NUM_THREADS). Default 0 = "
                         "the CPUs this process may run on (respects SLURM / taskset "
                         "limits, unlike the node's total core count)")

    ap.add_argument("--deconv", action="store_true")
    ap.add_argument("--deconv_strength", type=float, default=1.0)
    ap.add_argument("--deconv_falloff", type=float, default=1.0)
    ap.add_argument("--deconv_highpass", type=float, default=300.0)
    ap.add_argument("--dont_invert", action="store_true")
    ap.add_argument("--dont_normalize", action="store_true")
    ap.add_argument("--renorm_variance", action="store_true",
                    help="force each tilt to unit std after the high-pass (off by "
                         "default -- this over-boosts high-tilt/low-signal images "
                         "and was the dominant source of excess high-frequency "
                         "noise vs Warp's ts_reconstruct; see warp_recon/filters.py)")
    ap.add_argument("--no_highpass", action="store_true",
                    help="skip the per-tilt high-pass (band-pass) entirely while KEEPING "
                         "the rest of normalization (background subtraction, edge mask, "
                         "and --renorm_variance if given). Unlike --dont_normalize, which "
                         "disables all three. Gives a plain WBP-style preprocessing; "
                         "expect low-frequency cupping/ramp in the result, since per-tilt "
                         "illumination gradients now back-project instead of being removed.")
    ap.add_argument("--no_local_motion", action="store_true",
                    help="ignore the spatial variation of Warp's GridMovementX/Y "
                         "local-motion grids, in every engine (by default they are "
                         "applied: baked into the tilt images for fourier/etomo, per "
                         "voxel for novactf). Same as --etomo_no_local_motion / "
                         "--novactf_no_local_motion for those engines.")
    ap.add_argument("--subvolume_size", type=int, default=64)
    ap.add_argument("--subvolume_padding", type=float, default=3.0)
    ap.add_argument("--pad_factor", type=float, default=1.0,
                    help="reconstruction-cube padding (global engine)")
    ap.add_argument("--weight_floor", type=float, default=0.01,
                    help="minimum CTF-weight divisor before division (ReconstructFull's "
                         "floor); raising this damps noise amplification in low-coverage "
                         "Fourier shells at the cost of also damping real weak signal there")
    ap.add_argument("--float16", action="store_true", help="write 16-bit MRC like Warp")

    ap.add_argument("--ctf3d_defocus_step", type=float, default=0.0,
                    help="novaCTF-style 3D-CTF correction: split the tomogram thickness "
                         "into Z-strips of this thickness (nm) and reconstruct each with "
                         "its own per-tilt defocus, stitching the correctly-focused "
                         "Z-slab from each into the final volume (Turonova et al. 2017). "
                         "0 (default) disables it. Mutually exclusive with "
                         "--ctf3d_num_strips; costs ~N times a single reconstruction.")
    ap.add_argument("--ctf3d_num_strips", type=int, default=0,
                    help="novaCTF-style 3D-CTF correction: explicit Z-strip count instead "
                         "of --ctf3d_defocus_step. 0 (default) disables it.")

    # ---- engine selection ------------------------------------------------
    ap.add_argument("--engine", choices=["fourier", "etomo", "novactf"], default="fourier",
                    help="reconstruction algorithm. 'fourier' (default) is this "
                         "repo's CTF-weighted Fourier-slice insertion, i.e. what "
                         "Warp does. 'etomo' hands the same alignment and the same "
                         "preprocessed tilts to IMOD's `tilt` for real-space "
                         "weighted back-projection / SIRT -- note `tilt` applies "
                         "NO CTF phase flip and NO dose weighting, only its radial "
                         "filter (see warp_recon/etomo.py). 'novactf' is novaCTF's "
                         "3D-CTF real-space back-projection, but projecting every "
                         "voxel through Warp's full geometry incl. the local-motion "
                         "grids (see warp_recon/novactf.py).")
    ap.add_argument("--etomo_recon", choices=["wbp", "fakesirt", "sirt"], default="wbp",
                    help="--engine etomo only. 'wbp' = weighted back-projection with "
                         "IMOD's radial filter. 'fakesirt' = tilt's SIRT-like filter "
                         "(-FakeSIRTiterations), a radial filter analytically "
                         "equivalent to N SIRT iterations. 'sirt' = true iterative "
                         "SIRT (-SIRTIterations).")
    ap.add_argument("--etomo_sirt_iters", type=int, default=10,
                    help="iterations for --etomo_recon fakesirt/sirt (default 10)")
    ap.add_argument("--etomo_radial", type=float, nargs=2, default=[0.35, 0.035],
                    metavar=("CUTOFF", "FALLOFF"),
                    help="tilt RADIAL filter for wbp/fakesirt (etomo's own default)")
    ap.add_argument("--etomo_sirt_radial", type=float, nargs=2, default=[0.40, 0.035],
                    metavar=("CUTOFF", "FALLOFF"),
                    help="tilt RADIAL filter for true SIRT (sirtsetup's own default)")
    ap.add_argument("--etomo_xaxistilt_sign", type=int, choices=[1, -1], default=1,
                    help="sign mapping Warp's LevelAngleX onto tilt's XAXISTILT. The "
                         "magnitude is fixed by the .xml; only the sign convention is "
                         "IMOD's, so it is exposed for an A/B test.")
    ap.add_argument("--etomo_view_weight", choices=["none", "warp"], default="none",
                    help="per-view amplitude weighting for --engine etomo, via tilt's "
                         "WeightFile. 'none' (default) is pure IMOD: tilt weights only "
                         "by the radial ramp filter and by the local tilt increment "
                         "(-DENSWEIGHT), NOT by cos(tilt). 'warp' feeds the active "
                         "weighting scheme's per-tilt amplitude scale (cos(tilt) for "
                         "--dose_weighting warp) to tilt, matching what the fourier "
                         "engine applies. Only the scalar per-view factor transfers "
                         "this way -- the dose B-factor envelope is frequency-"
                         "dependent and cannot go through a WeightFile.")
    ap.add_argument("--etomo_no_local_motion", action="store_true",
                    help="--engine etomo: do NOT bake Warp's GridMovementX/Y local-motion "
                         "grids into the tilt images (default: baked in, so IMOD tilt "
                         "reproduces Warp's local alignment; see reconstruct.bake_local_motion)")
    ap.add_argument("--etomo_gpu", type=int, default=-1,
                    help="tilt UseGPU: <0 (default) CPU, 0 best available GPU, N GPU #N")
    ap.add_argument("--etomo_workdir", default=None,
                    help="where to write the IMOD project (default <output>/etomo)")
    ap.add_argument("--imod_dir", default=None,
                    help="IMOD installation (default $IMOD_DIR or /usr/local/IMOD)")

    # ---- --engine novactf ------------------------------------------------
    ap.add_argument("--novactf_step", type=float, default=10.0,
                    help="--engine novactf: defocus step (nm) between CTF-corrected "
                         "copies of each tilt, like novaCTF -DefocusStep. Each voxel "
                         "uses the copy nearest its own depth along each tilt's "
                         "beam. 0 = 2-D CTF (one defocus per tilt).")
    ap.add_argument("--novactf_correction", choices=["multiplication", "phaseflip", "none"],
                    default="multiplication",
                    help="--engine novactf: novaCTF -CorrectionType (default "
                         "multiplication); 'none' skips CTF correction entirely")
    ap.add_argument("--novactf_radial", type=float, nargs=2, default=[0.3, 0.05],
                    metavar=("CUTOFF", "FALLOFF"),
                    help="--engine novactf: novaCTF -RADIAL, cycles/pixel")
    ap.add_argument("--novactf_weighting", choices=["none", "warp"], default="none",
                    help="--engine novactf: 'none' (default) = novaCTF's own per-view "
                         "weighting only; 'warp' also applies the --dose_weighting "
                         "scheme's per-tilt amplitude scale and dose exposure filter")
    ap.add_argument("--novactf_no_local_motion", action="store_true",
                    help="--engine novactf: ignore Warp's GridMovementX/Y (global "
                         "alignment only, i.e. what the novaCTF binary can represent)")
    ap.add_argument("--novactf_no_astig", action="store_true",
                    help="--engine novactf: ignore per-tilt astigmatism "
                         "(novaCTF -CorrectAstigmatism 0)")
    ap.add_argument("--novactf_geometry_step", type=int, default=8,
                    help="--engine novactf: voxels between exact geometry "
                         "evaluations (trilinear in between; the mapping is affine "
                         "apart from the smooth local-motion grids)")

    # experimentation hooks
    ap.add_argument("--dose_bfactor_scale", type=float, default=None,
                    help="use a custom dose->Bfactor factor instead of Warp's 4 "
                         "(ignored if --dose_weighting=motioncor3)")
    ap.add_argument("--dose_weighting", choices=["warp", "motioncor3"], default="warp",
                    help="per-tilt dose exposure filter. 'warp' (default) is Warp's own "
                         "linear model (Bfactor = -dose*4, a single Gaussian falloff). "
                         "'motioncor3' uses the Grant & Grigorieff (2015) critical-"
                         "exposure curve MotionCor3 applies to frames "
                         "(Correct/GWeightFrame.cu), applied here per tilt via each "
                         "tilt's accumulated dose -- falls off more gently than a "
                         "Gaussian at low dose/high resolution, aiming to preserve more "
                         "high-resolution signal from low-dose tilts.")
    args = ap.parse_args()
    print(f"threads: {set_threads(args.threads)}")

    model = load_tiltseries_xml(args.xml, args.tomostar)
    tilt_images, header_angpix = load_tilt_images(model, args)

    raw_angpix = args.raw_angpix
    if raw_angpix is None and args.settings:
        raw_angpix = read_settings_pixelsize(args.settings)
    if raw_angpix is None:
        raw_angpix = header_angpix
    if not raw_angpix or raw_angpix <= 0:
        raise SystemExit("could not determine raw pixel size; pass --raw_angpix")
    print(f"raw pixel size: {raw_angpix:.3f} A  ->  output tomogram pixel size "
          f"{args.angpix:.3f} A")

    # --- tomogram box selection ---------------------------------------------
    # priority: --dimensions  >  settings <Tomo> Dimensions  >  .xml value
    box_src = "xml"
    if args.dimensions is not None:
        dx, dy, dz = args.dimensions
        model.volume_dims_A = np.array([dx, dy, dz], np.float32) * raw_angpix
        box_src = "--dimensions"
    elif args.settings:
        dims = read_settings_tomo_dims(args.settings)
        if dims is not None:
            model.volume_dims_A = np.array(dims, np.float32) * raw_angpix
            box_src = "settings <Tomo>"
    nv = (model.volume_dims_A / args.angpix)
    print(f"tomogram box: {model.volume_dims_A} A  (from {box_src})  ->  "
          f"~{int(round(nv[0]))}x{int(round(nv[1]))}x{int(round(nv[2]))} voxels "
          f"at {args.angpix} A")

    opts = ReconOptions(
        angpix=args.angpix, raw_angpix=raw_angpix,
        invert=not args.dont_invert, normalize=not args.dont_normalize,
        do_deconv=args.deconv, deconv_strength=args.deconv_strength,
        deconv_falloff=args.deconv_falloff, deconv_highpass=args.deconv_highpass,
        subvolume_size=args.subvolume_size, subvolume_padding=args.subvolume_padding,
        pad_factor=args.pad_factor, weight_floor=args.weight_floor,
        renorm_variance=args.renorm_variance, highpass=not args.no_highpass,
        local_motion=not args.no_local_motion,
        mode={"etomo": "etomo", "novactf": "novactf"}.get(args.engine, "global"),
        etomo=EtomoOptions(
            recon=args.etomo_recon, sirt_iters=args.etomo_sirt_iters,
            radial=tuple(args.etomo_radial), sirt_radial=tuple(args.etomo_sirt_radial),
            xaxistilt_sign=float(args.etomo_xaxistilt_sign),
            view_weight=args.etomo_view_weight, gpu=args.etomo_gpu,
            local_motion=not (args.etomo_no_local_motion or args.no_local_motion),
            workdir=args.etomo_workdir or os.path.join(args.output, "etomo"),
            imod_dir=args.imod_dir),
        ctf3d_defocus_step_nm=args.ctf3d_defocus_step, ctf3d_num_strips=args.ctf3d_num_strips,
        novactf=NovaCTFOptions(
            defocus_step_nm=args.novactf_step, radial=tuple(args.novactf_radial),
            correction=args.novactf_correction,
            correct_astigmatism=not args.novactf_no_astig,
            local_motion=not (args.novactf_no_local_motion or args.no_local_motion),
            weighting=args.novactf_weighting,
            geometry_step=args.novactf_geometry_step),
    )

    if args.engine == "novactf":
        ignored = [n for n, on in (("--deconv", args.deconv),
                                   ("--ctf3d_defocus_step/--ctf3d_num_strips",
                                    bool(args.ctf3d_defocus_step or args.ctf3d_num_strips)))
                   if on]
        if args.novactf_weighting == "none" and (args.dose_weighting != "warp"
                                                 or args.dose_bfactor_scale is not None):
            ignored.append("--dose_weighting/--dose_bfactor_scale (needs "
                           "--novactf_weighting warp)")
        if ignored:
            print("note: --engine novactf ignores " + ", ".join(ignored))

    if args.engine == "etomo":
        # be explicit rather than silently dropping them: `tilt` has no CTF or
        # dose model, and the deconvolution filter is a post-step of the
        # Fourier engine's output path.
        wt_used = args.etomo_view_weight == "warp"
        ignored = [n for n, on in (("--deconv", args.deconv),
                                   ("--dose_weighting motioncor3 (B-factor part; "
                                    "its per-tilt scale IS applied)" if wt_used
                                    else "--dose_weighting motioncor3",
                                    args.dose_weighting != "warp"),
                                   ("--dose_bfactor_scale",
                                    args.dose_bfactor_scale is not None),
                                   ("--ctf3d_defocus_step/--ctf3d_num_strips",
                                    bool(args.ctf3d_defocus_step or args.ctf3d_num_strips)))
                   if on]
        if ignored:
            print("note: --engine etomo ignores " + ", ".join(ignored)
                  + " (IMOD tilt has no CTF/dose model)")

    wfn = warp_weighting
    if args.dose_weighting == "motioncor3":
        wfn = motioncor3_dose_weighting
        print("using MotionCor3-style (Grant & Grigorieff) per-tilt dose weighting")
    elif args.dose_bfactor_scale is not None:
        wfn = make_dose_bfactor_weighting(args.dose_bfactor_scale)
        print(f"using custom dose->Bfactor scale = {args.dose_bfactor_scale}")

    outputs = reconstruct(model, tilt_images, opts, weighting_fn=wfn)

    os.makedirs(args.output, exist_ok=True)
    base = model.name
    rec_path = os.path.join(args.output, base + ".mrc")
    write_mrc(rec_path, outputs["reconstruction"], args.angpix, as_float16=args.float16)
    write_png_slice(os.path.join(args.output, base + ".png"), outputs["reconstruction"])
    print("wrote", rec_path)
    if "deconv" in outputs:
        dpath = os.path.join(args.output, base + "_deconv.mrc")
        write_mrc(dpath, outputs["deconv"], args.angpix, as_float16=args.float16)
        print("wrote", dpath)


if __name__ == "__main__":
    main()
