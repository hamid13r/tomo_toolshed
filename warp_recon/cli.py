"""Click front-end for the reconstruction engine, mirroring WarpTools
``ts_reconstruct``.

The pure logic lives in the :mod:`warp_recon` package (this is the click layer
only). User errors raise :class:`click.ClickException` rather than tracebacks.

Every multi-word option accepts both a dash and an underscore spelling
(``--tilt-stack`` / ``--tilt_stack``) so older command lines keep working.
"""
from __future__ import annotations

import os
from pathlib import Path

import click
import numpy as np

from warp_recon import load_tiltseries_xml, reconstruct, ReconOptions
from warp_recon import warp_weighting, make_dose_bfactor_weighting
from warp_recon.mrc_io import read_mrc, write_mrc, write_png_slice
from warp_recon.metadata import read_settings_pixelsize, read_settings_tomo_dims


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
        imgs, vs = [], None
        for mp in model.movie_paths:
            root = Path(mp).stem
            cand = sorted(set(d.glob(root + ".mrc")) | set(d.glob(root + ".*")))
            if not cand:
                raise click.ClickException(
                    f"could not find tilt image for '{root}' in {d}")
            im, v = read_mrc(str(cand[0]))
            if im.ndim == 3:
                im = im[0]
            imgs.append(im)
            if vs is None:
                vs = v
        return imgs, vs
    raise click.ClickException("provide --tilt_stack or --tilt_dir")


@click.command(name="xml-reconstruct")
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
@click.option("--subvolume-size", "--subvolume_size", "subvolume_size",
              type=int, default=64, show_default=True)
@click.option("--subvolume-padding", "--subvolume_padding", "subvolume_padding",
              type=float, default=3.0, show_default=True)
@click.option("--pad-factor", "--pad_factor", "pad_factor", type=float,
              default=1.15, show_default=True,
              help="in-plane grid padding to hold the tilted footprint; "
                   "raise if edges look clipped at high tilt")
@click.option("--float16", is_flag=True, help="write 16-bit MRC like Warp")
@click.option("--dose-bfactor-scale", "--dose_bfactor_scale", "dose_bfactor_scale",
              type=float, default=None,
              help="use a custom dose->Bfactor factor instead of Warp's 4")
def xml_reconstruct(xml_path, tomostar, settings, angpix, raw_angpix, dimensions,
                    tilt_stack, tilt_dir, output, deconv, deconv_strength,
                    deconv_falloff, deconv_highpass, dont_invert, dont_normalize,
                    subvolume_size, subvolume_padding, pad_factor, float16,
                    dose_bfactor_scale):
    """Faithful ts_reconstruct re-implementation (weighting/filtering hooks)."""
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
        pad_factor=pad_factor, mode="global",
    )

    wfn = warp_weighting
    if dose_bfactor_scale is not None:
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
