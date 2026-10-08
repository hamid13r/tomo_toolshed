"""Click front-end for the reconstruction engine, mirroring WarpTools
``ts_reconstruct``.

The pure logic lives in the :mod:`tomo_toolshed.xml_reconstruct` package (this is
the click layer only). User errors raise :class:`click.ClickException` rather
than tracebacks.

Every multi-word option accepts both a dash and an underscore spelling
(``--tilt-stack`` / ``--tilt_stack``) so older command lines keep working.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import click
import numpy as np

from . import load_tiltseries_xml, reconstruct, ReconOptions
from . import warp_weighting, make_dose_bfactor_weighting, motioncor3_dose_weighting
from .etomo import EtomoOptions
from .novactf import NovaCTFOptions
from .parallel import get_threads, set_threads
from .mrc_io import read_mrc, write_mrc, write_png_slice
from .metadata import read_settings_pixelsize, read_settings_tomo_dims


def _dual(name, **kw):
    """click.option with both the dash and the underscore spelling of ``name``."""
    dest = name.lstrip("-").replace("-", "_")
    return click.option(name, "--" + dest, dest, **kw)


def _load_one_tilt(tilt_dir, movie_path):
    root = Path(movie_path).stem
    cand = sorted(set(tilt_dir.glob(root + ".mrc")) | set(tilt_dir.glob(root + ".*")))
    if not cand:
        raise click.ClickException(
            f"could not find tilt image for '{root}' in {tilt_dir}")
    im, v = read_mrc(str(cand[0]))
    if im.ndim == 3:
        im = im[0]
    return im, v


def load_tilt_images(model, tilt_stack, tilt_dir):
    """Return (list_of_2D_arrays, header_pixel_size) for the tilt images.

    Exactly one of ``tilt_stack`` / ``tilt_dir`` must be given.
    """
    if tilt_stack:
        data, vs = read_mrc(tilt_stack)
        if data.ndim != 3:
            raise click.ClickException("--tilt_stack must be a 3D stack")
        if data.shape[0] != model.n_tilts:
            raise click.ClickException(
                f"stack has {data.shape[0]} slices but model has "
                f"{model.n_tilts} tilts")
        return [data[i] for i in range(data.shape[0])], vs
    if tilt_dir:
        d = Path(tilt_dir)
        # each tilt is an independent file read (disk I/O releases the GIL),
        # so a thread pool gives a near-linear speedup over reading serially
        n_workers = max(1, min(16, len(model.movie_paths), get_threads()))
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            results = list(ex.map(lambda mp: _load_one_tilt(d, mp),
                                  model.movie_paths))
        imgs = [im for im, _ in results]
        vs = next((v for _, v in results if v is not None), None)
        return imgs, vs
    raise click.ClickException("provide --tilt_stack or --tilt_dir")


@click.command(name="xml-reconstruct")
# ---- input / output -------------------------------------------------------
@click.option("--xml", "xml_path", required=True,
              help="per-tilt-series .xml (Warp metadata)")
@click.option("--tomostar", help=".tomostar (fallback for raw angles/dose)")
@click.option("--settings", help="Warp .settings (for raw pixel size)")
@click.option("--angpix", type=float, required=True,
              help="OUTPUT tomogram pixel size (A) -- sets the tomogram "
                   "sampling, same as ts_reconstruct --angpix")
@click.option("--raw-angpix", "--raw_angpix", "raw_angpix", type=float,
              help="unbinned tilt pixel size (A)")
@click.option("--dimensions", type=int, nargs=3, default=None,
              metavar="X Y Z",
              help="tomogram box in UNBINNED pixels (like Warp's <Tomo> "
                   "Dimensions). Overrides the .settings / .xml box.")
@click.option("--tilt-stack", "--tilt_stack", "tilt_stack",
              help="3D MRC/ST stack ordered like the tilts")
@click.option("--tilt-dir", "--tilt_dir", "tilt_dir",
              help="folder of per-tilt averages <root>.mrc")
@click.option("--output", default="reconstruction", help="output folder")
@click.option("--threads", type=int, default=0, show_default=True,
              help="CPU threads for everything parallel: FFTs, numba kernels, "
                   "thread pools, and IMOD tilt (OMP_NUM_THREADS). 0 = the CPUs "
                   "this process may run on (respects SLURM / taskset limits, "
                   "unlike the node's total core count)")
# ---- engine ---------------------------------------------------------------
@click.option("--engine", type=click.Choice(["fourier", "etomo", "novactf"]),
              default="fourier", show_default=True,
              help="reconstruction algorithm. 'fourier' is CTF-weighted "
                   "Fourier-slice insertion, i.e. what Warp does. 'etomo' hands "
                   "the same alignment and preprocessed tilts to IMOD's `tilt` "
                   "(WBP / SIRT; no CTF phase flip, no dose weighting). "
                   "'novactf' is novaCTF's 3D-CTF real-space back-projection "
                   "through Warp's full geometry incl. local-motion grids.")
# ---- preprocessing / fourier engine ---------------------------------------
@click.option("--deconv", is_flag=True, help="also write a deconvolved volume")
@click.option("--deconv-strength", "--deconv_strength", "deconv_strength",
              type=float, default=1.0, show_default=True)
@click.option("--deconv-falloff", "--deconv_falloff", "deconv_falloff",
              type=float, default=1.0, show_default=True)
@click.option("--deconv-highpass", "--deconv_highpass", "deconv_highpass",
              type=float, default=300.0, show_default=True)
@click.option("--dont-invert", "--dont_invert", "dont_invert", is_flag=True)
@click.option("--dont-normalize", "--dont_normalize", "dont_normalize",
              is_flag=True)
@_dual("--renorm-variance", is_flag=True,
       help="force each tilt to unit std after the high-pass (off by default: "
            "it over-boosts high-tilt/low-signal images and was the dominant "
            "source of excess high-frequency noise vs Warp's ts_reconstruct)")
@_dual("--no-highpass", is_flag=True,
       help="skip the per-tilt high-pass entirely while keeping background "
            "subtraction, edge mask and --renorm-variance (unlike "
            "--dont-normalize, which disables all three). Plain WBP-style "
            "preprocessing; expect low-frequency cupping in the result.")
@_dual("--no-local-motion", is_flag=True,
       help="ignore the spatial variation of Warp's GridMovementX/Y local-motion "
            "grids in every engine (applied by default). Same as "
            "--etomo-no-local-motion / --novactf-no-local-motion for those engines.")
@click.option("--subvolume-size", "--subvolume_size", "subvolume_size",
              type=int, default=64, show_default=True)
@click.option("--subvolume-padding", "--subvolume_padding", "subvolume_padding",
              type=float, default=3.0, show_default=True)
@click.option("--pad-factor", "--pad_factor", "pad_factor", type=float,
              default=1.15, show_default=True,
              help="in-plane grid padding to hold the tilted footprint; "
                   "raise if edges look clipped at high tilt")
@_dual("--weight-floor", type=float, default=0.01, show_default=True,
       help="minimum CTF-weight divisor (ReconstructFull's floor); raising it "
            "damps noise in low-coverage Fourier shells, and weak real signal too")
@click.option("--float16", is_flag=True, help="write 16-bit MRC like Warp")
@_dual("--ctf3d-defocus-step", type=float, default=0.0, show_default=True,
       help="fourier engine: novaCTF-style 3D-CTF correction. Split the "
            "thickness into Z-strips of this thickness (nm), reconstruct each "
            "with its own per-tilt defocus and stitch. 0 disables. Takes "
            "precedence over --ctf3d-num-strips; costs ~N x one reconstruction.")
@_dual("--ctf3d-num-strips", type=int, default=0, show_default=True,
       help="fourier engine: explicit Z-strip count instead of "
            "--ctf3d-defocus-step. 0 disables.")
# ---- weighting hooks ------------------------------------------------------
@click.option("--dose-bfactor-scale", "--dose_bfactor_scale", "dose_bfactor_scale",
              type=float, default=None,
              help="use a custom dose->Bfactor factor instead of Warp's 4 "
                   "(ignored with --dose-weighting motioncor3)")
@_dual("--dose-weighting", type=click.Choice(["warp", "motioncor3"]),
       default="warp", show_default=True,
       help="per-tilt dose exposure filter. 'warp' is Warp's linear model "
            "(Bfactor = -dose*4). 'motioncor3' uses the Grant & Grigorieff (2015) "
            "critical-exposure curve MotionCor3 applies to frames, applied per "
            "tilt via accumulated dose.")
# ---- --engine etomo -------------------------------------------------------
@_dual("--etomo-recon", type=click.Choice(["wbp", "fakesirt", "sirt"]),
       default="wbp", show_default=True,
       help="--engine etomo: 'wbp' weighted back-projection; 'fakesirt' tilt's "
            "SIRT-like radial filter (-FakeSIRTiterations); 'sirt' true "
            "iterative SIRT (-SIRTIterations)")
@_dual("--etomo-sirt-iters", type=int, default=10, show_default=True,
       help="iterations for --etomo-recon fakesirt/sirt")
@_dual("--etomo-radial", type=float, nargs=2, default=(0.35, 0.035),
       show_default=True, metavar="CUTOFF FALLOFF",
       help="tilt RADIAL filter for wbp/fakesirt (etomo's own default)")
@_dual("--etomo-sirt-radial", type=float, nargs=2, default=(0.40, 0.035),
       show_default=True, metavar="CUTOFF FALLOFF",
       help="tilt RADIAL filter for true SIRT (sirtsetup's own default)")
@_dual("--etomo-xaxistilt-sign", type=click.Choice(["1", "-1"]), default="1",
       show_default=True,
       help="sign mapping Warp's LevelAngleX onto tilt's XAXISTILT (exposed "
            "for A/B tests; +1 is validated)")
@_dual("--etomo-view-weight", type=click.Choice(["none", "warp"]),
       default="none", show_default=True,
       help="per-view amplitude weighting via tilt's WeightFile. 'none' is pure "
            "IMOD; 'warp' feeds the active weighting scheme's per-tilt scale "
            "(cos(tilt) for --dose-weighting warp), as the fourier engine applies")
@_dual("--etomo-no-local-motion", is_flag=True,
       help="--engine etomo: do NOT bake Warp's local-motion grids into the tilts")
@_dual("--etomo-gpu", type=int, default=-1, show_default=True,
       help="tilt UseGPU: <0 CPU, 0 best available GPU, N GPU #N")
@_dual("--etomo-workdir", default=None,
       help="where to write the IMOD project (default <output>/etomo)")
@_dual("--imod-dir", default=None,
       help="IMOD installation (default $IMOD_DIR or /usr/local/IMOD)")
# ---- --engine novactf -----------------------------------------------------
@_dual("--novactf-step", type=float, default=10.0, show_default=True,
       help="--engine novactf: defocus step (nm) between CTF-corrected copies "
            "of each tilt (novaCTF -DefocusStep). 0 = 2-D CTF.")
@_dual("--novactf-correction",
       type=click.Choice(["multiplication", "phaseflip", "none"]),
       default="multiplication", show_default=True,
       help="--engine novactf: novaCTF -CorrectionType; 'none' skips CTF correction")
@_dual("--novactf-radial", type=float, nargs=2, default=(0.3, 0.05),
       show_default=True, metavar="CUTOFF FALLOFF",
       help="--engine novactf: novaCTF -RADIAL, cycles/pixel")
@_dual("--novactf-weighting", type=click.Choice(["none", "warp"]),
       default="none", show_default=True,
       help="--engine novactf: 'none' = novaCTF's own per-view weighting; 'warp' "
            "also applies the --dose-weighting scheme's per-tilt scale and "
            "dose exposure filter")
@_dual("--novactf-no-local-motion", is_flag=True,
       help="--engine novactf: global alignment only (what the novaCTF binary "
            "can represent)")
@_dual("--novactf-no-astig", is_flag=True,
       help="--engine novactf: ignore per-tilt astigmatism")
@_dual("--novactf-device", default="cpu", show_default=True,
       help="--engine novactf: 'cpu' (numba if installed, else numpy) or a "
            "PyTorch device such as 'cuda' / 'cuda:1' (needs torch with CUDA)")
@_dual("--novactf-geometry-step", type=int, default=16, show_default=True,
       help="--engine novactf: voxels between exact geometry evaluations "
            "(trilinear in between)")
def xml_reconstruct(xml_path, tomostar, settings, angpix, raw_angpix, dimensions,
                    tilt_stack, tilt_dir, output, threads, engine,
                    deconv, deconv_strength, deconv_falloff, deconv_highpass,
                    dont_invert, dont_normalize, renorm_variance, no_highpass,
                    no_local_motion, subvolume_size, subvolume_padding,
                    pad_factor, weight_floor, float16,
                    ctf3d_defocus_step, ctf3d_num_strips,
                    dose_bfactor_scale, dose_weighting,
                    etomo_recon, etomo_sirt_iters, etomo_radial, etomo_sirt_radial,
                    etomo_xaxistilt_sign, etomo_view_weight, etomo_no_local_motion,
                    etomo_gpu, etomo_workdir, imod_dir,
                    novactf_step, novactf_correction, novactf_radial,
                    novactf_weighting, novactf_no_local_motion, novactf_no_astig,
                    novactf_device, novactf_geometry_step):
    """Reconstruct a tomogram from a Warp tilt-series XML (Warp, IMOD/etomo or
    novaCTF engine)."""
    click.echo(f"threads: {set_threads(threads)}")

    model = load_tiltseries_xml(xml_path, tomostar)
    tilt_images, header_angpix = load_tilt_images(model, tilt_stack, tilt_dir)

    if raw_angpix is None and settings:
        raw_angpix = read_settings_pixelsize(settings)
    if raw_angpix is None:
        raw_angpix = header_angpix
    if not raw_angpix or raw_angpix <= 0:
        raise click.ClickException(
            "could not determine raw pixel size; pass --raw_angpix")
    click.echo(f"raw pixel size: {raw_angpix:.3f} A  ->  output tomogram pixel "
               f"size {angpix:.3f} A")

    # --- tomogram box selection: --dimensions > settings <Tomo> > .xml ---
    box_src = "xml"
    if dimensions:
        dx, dy, dz = dimensions
        model.volume_dims_A = np.array([dx, dy, dz], np.float32) * raw_angpix
        box_src = "--dimensions"
    elif settings:
        dims = read_settings_tomo_dims(settings)
        if dims is not None:
            model.volume_dims_A = np.array(dims, np.float32) * raw_angpix
            box_src = "settings <Tomo>"
    nv = (model.volume_dims_A / angpix)
    click.echo(f"tomogram box: {model.volume_dims_A} A  (from {box_src})  ->  "
               f"~{int(round(nv[0]))}x{int(round(nv[1]))}x{int(round(nv[2]))} "
               f"voxels at {angpix} A")

    opts = ReconOptions(
        angpix=angpix, raw_angpix=raw_angpix,
        invert=not dont_invert, normalize=not dont_normalize,
        do_deconv=deconv, deconv_strength=deconv_strength,
        deconv_falloff=deconv_falloff, deconv_highpass=deconv_highpass,
        subvolume_size=subvolume_size, subvolume_padding=subvolume_padding,
        pad_factor=pad_factor, weight_floor=weight_floor,
        renorm_variance=renorm_variance, highpass=not no_highpass,
        local_motion=not no_local_motion,
        mode={"etomo": "etomo", "novactf": "novactf"}.get(engine, "global"),
        etomo=EtomoOptions(
            recon=etomo_recon, sirt_iters=etomo_sirt_iters,
            radial=tuple(etomo_radial), sirt_radial=tuple(etomo_sirt_radial),
            xaxistilt_sign=float(etomo_xaxistilt_sign),
            view_weight=etomo_view_weight, gpu=etomo_gpu,
            local_motion=not (etomo_no_local_motion or no_local_motion),
            workdir=etomo_workdir or os.path.join(output, "etomo"),
            imod_dir=imod_dir),
        ctf3d_defocus_step_nm=ctf3d_defocus_step,
        ctf3d_num_strips=ctf3d_num_strips,
        novactf=NovaCTFOptions(
            defocus_step_nm=novactf_step, radial=tuple(novactf_radial),
            correction=novactf_correction,
            correct_astigmatism=not novactf_no_astig,
            local_motion=not (novactf_no_local_motion or no_local_motion),
            weighting=novactf_weighting,
            geometry_step=novactf_geometry_step,
            device=novactf_device),
    )

    ctf3d_on = bool(ctf3d_defocus_step or ctf3d_num_strips)
    if engine == "novactf":
        ignored = [n for n, on in (("--deconv", deconv),
                                   ("--ctf3d-defocus-step/--ctf3d-num-strips",
                                    ctf3d_on)) if on]
        if novactf_weighting == "none" and (dose_weighting != "warp"
                                            or dose_bfactor_scale is not None):
            ignored.append("--dose-weighting/--dose-bfactor-scale (needs "
                           "--novactf-weighting warp)")
        if ignored:
            click.echo("note: --engine novactf ignores " + ", ".join(ignored))

    if engine == "etomo":
        # be explicit rather than silently dropping them: `tilt` has no CTF or
        # dose model, and deconvolution is a post-step of the Fourier engine.
        wt_used = etomo_view_weight == "warp"
        ignored = [n for n, on in (
            ("--deconv", deconv),
            ("--dose-weighting motioncor3 (B-factor part; its per-tilt scale "
             "IS applied)" if wt_used else "--dose-weighting motioncor3",
             dose_weighting != "warp"),
            ("--dose-bfactor-scale", dose_bfactor_scale is not None),
            ("--ctf3d-defocus-step/--ctf3d-num-strips", ctf3d_on)) if on]
        if ignored:
            click.echo("note: --engine etomo ignores " + ", ".join(ignored)
                       + " (IMOD tilt has no CTF/dose model)")

    wfn = warp_weighting
    if dose_weighting == "motioncor3":
        wfn = motioncor3_dose_weighting
        click.echo("using MotionCor3-style (Grant & Grigorieff) per-tilt dose "
                   "weighting")
    elif dose_bfactor_scale is not None:
        wfn = make_dose_bfactor_weighting(dose_bfactor_scale)
        click.echo(f"using custom dose->Bfactor scale = {dose_bfactor_scale}")

    outputs = reconstruct(model, tilt_images, opts, weighting_fn=wfn,
                          progress=click.echo)

    os.makedirs(output, exist_ok=True)
    base = model.name
    rec_path = os.path.join(output, base + ".mrc")
    write_mrc(rec_path, outputs["reconstruction"], angpix, as_float16=float16)
    write_png_slice(os.path.join(output, base + ".png"), outputs["reconstruction"])
    click.echo(f"wrote {rec_path}")
    if "deconv" in outputs:
        dpath = os.path.join(output, base + "_deconv.mrc")
        write_mrc(dpath, outputs["deconv"], angpix, as_float16=float16)
        click.echo(f"wrote {dpath}")


if __name__ == "__main__":
    xml_reconstruct()
