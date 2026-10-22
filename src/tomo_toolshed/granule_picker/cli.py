"""``tomo_toolshed pick-granules`` -- threshold dark granules inside a mask.

    tomo_toolshed pick-granules TOMOGRAM MASK [-o OUTPUT] [options]

Loads a tomogram and a binary mask (e.g. a mitochondrion segmentation), opens a
GUI with the mask and the thresholded voxels overlaid on the tomogram, and lets
you drag a threshold until the dark granules are covered. On save, writes a
uint8 0/1 MRC of the voxels inside the (optionally eroded) mask that are darker
than the threshold, with the mask's voxel size and origin. Clean it up
afterwards with ``tomo_toolshed curate``.
"""

from __future__ import annotations

import os
from pathlib import Path

import click
import numpy as np

from . import core


def default_output_path(mask_path):
    """``<mask stem>_granules<suffix>`` next to the mask."""
    p = Path(mask_path)
    return str(p.with_name(p.stem + "_granules" + (p.suffix or ".mrc")))


@click.command(name="pick-granules", context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("tomogram", type=click.Path(exists=True, dir_okay=False))
@click.argument("mask", type=click.Path(exists=True, dir_okay=False))
@click.option("--output", "-o", "output_path", default=None, type=click.Path(dir_okay=False),
              help="Output binary MRC (parent directories are created). "
                   "[default: <mask>_granules.mrc]")
@click.option("--threshold", "-t", type=float, default=None,
              help="Starting threshold in tomogram units (after --blur). "
                   "[default: the --percentile value]")
@click.option("--percentile", "-p", type=click.FloatRange(0, 100), default=None,
              help="Starting threshold as a percentile of the tomogram values inside "
                   "the mask. [default: 5]")
@click.option("--bright", is_flag=True, default=False,
              help="Pick voxels brighter than the threshold instead of darker.")
@click.option("--blur", type=click.FloatRange(min=0), default=0.0, show_default=True,
              help="Gaussian blur sigma (voxels) applied to the tomogram before "
                   "thresholding and display. 1-2 suppresses noise speckle.")
@click.option("--mask-erode", type=click.IntRange(min=0), default=0, show_default=True,
              help="Erode the mask this many pixels first (XY; 3D with --erode-3d), to keep the dark "
                   "membranes at its boundary out of the result.")
@click.option("--erode-3d", is_flag=True, default=False,
              help="Erode the mask in 3D (also its top and bottom in Z) instead of XY only.")
@click.option("--min-size", type=click.IntRange(min=0), default=0, show_default=True,
              help="Drop granules smaller than this many voxels (26-connected) on save.")
@click.option("--no-gui", is_flag=True, default=False,
              help="Skip the GUI and write the result for the starting threshold "
                   "(batch mode).")
@click.option("--overwrite", is_flag=True, default=False,
              help="Overwrite OUTPUT if it exists. Without it an existing output is "
                   "skipped (exit 0), like curate.")
def pick_granules(tomogram, mask, output_path, threshold, percentile, bright, blur,
                  mask_erode, erode_3d, min_size, no_gui, overwrite):
    """Pick dark granules inside a MASK of a TOMOGRAM with an interactive threshold.

    Shows the tomogram with the mask (blue) and every mask voxel darker than the
    threshold (red), in a Z view and a Y view. Drag the threshold slider (or type
    a value or percentile) under the histogram of in-mask values; 'q' or closing
    the window saves, Esc quits without saving. The output is a uint8 0/1 MRC
    with the mask's voxel size and origin, ready for `tomo_toolshed curate`.

    \b
    example:
    tomo_toolshed pick-granules tomo.mrc mito.mrc -o mito_granules.mrc
    tomo_toolshed pick-granules tomo.mrc mito.mrc --blur 1.5 --mask-erode 3
    tomo_toolshed pick-granules tomo.mrc mito.mrc -t -0.8 --min-size 20 --no-gui
    """
    output_path = output_path or default_output_path(mask)
    real_out = os.path.realpath(output_path)
    if real_out in (os.path.realpath(mask), os.path.realpath(tomogram)):
        raise click.ClickException("--output must differ from TOMOGRAM and MASK")
    if os.path.exists(output_path) and not overwrite:
        click.echo(f"[skip] Output already exists: {output_path}  (use --overwrite)")
        return
    if threshold is not None and percentile is not None:
        raise click.UsageError("give --threshold or --percentile, not both")

    try:
        click.echo(f"Reading tomogram: {tomogram}")
        tomo, _, _ = core.read_volume(tomogram)
        click.echo(f"Reading mask:     {mask}")
        raw_mask, voxel_size, origin = core.read_volume(mask)
        if tomo.shape != raw_mask.shape:
            raise core.GranuleError(
                f"tomogram and mask shapes differ: {tomo.shape} vs {raw_mask.shape}")
        if blur > 0:
            click.echo(f"Blurring tomogram (sigma={blur:g} voxels)...")
        tomo = core.blur(tomo, blur)

        m = core.prepare_mask(raw_mask, mask_erode, erode_3d)
        vals = core.masked_values(tomo, m)
        if vals.size == 0:
            raise core.GranuleError("the mask is empty"
                                    + (" after erosion" if mask_erode else ""))
        thr = core.initial_threshold(vals, threshold, percentile)
    except core.GranuleError as exc:
        raise click.ClickException(str(exc))

    click.echo(f"Mask: {vals.size} voxels; values {vals[0]:.4g} .. {vals[-1]:.4g}, "
               f"median {np.median(vals):.4g}")
    click.echo(f"Starting threshold: {thr:.5g} "
               f"({core.percentile_of(vals, thr):.2f} percentile of mask)")

    params = {"threshold": thr, "mask_erode": mask_erode, "erode_3d": erode_3d,
              "min_size": min_size}
    if not no_gui:
        from .gui import GranulePickerGUI  # lazy: keeps matplotlib out of headless use

        click.echo("Launching GUI... ('q' or close to save, Esc to quit without saving)")
        gui = GranulePickerGUI(tomo, raw_mask, thr, bright=bright,
                               mask_erode=mask_erode, min_size=min_size, erode_3d=erode_3d)
        del m, vals
        params = gui.run()
        if params is None:
            click.echo("Quit without saving.")
            return
        m = core.prepare_mask(raw_mask, params["mask_erode"], params["erode_3d"])
    else:
        del vals

    granules = core.pick(tomo, m, params["threshold"], bright=bright,
                         min_size=params["min_size"])
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    core.write_mask(output_path, granules, voxel_size=voxel_size, origin=origin)

    n = int(granules.sum())
    side = "above" if bright else "below"
    mode = "3D" if params["erode_3d"] else "XY"
    click.echo(f"threshold {params['threshold']:.6g} ({side}), mask erosion "
               f"{params['mask_erode']} px {mode}, min size {params['min_size']} vox")
    click.echo(f"Wrote granules: {output_path}  ({n} of {int(m.sum())} mask voxels, "
               f"{100.0 * n / max(1, int(m.sum())):.2f}%)")
    click.echo("To reproduce without the GUI: tomo_toolshed pick-granules "
               f"{tomogram} {mask} -o {output_path} -t {params['threshold']:.6g}"
               + (f" --blur {blur:g}" if blur else "")
               + (f" --mask-erode {params['mask_erode']}" if params["mask_erode"] else "")
               + (" --erode-3d" if params["mask_erode"] and params["erode_3d"] else "")
               + (f" --min-size {params['min_size']}" if params["min_size"] else "")
               + (" --bright" if bright else "") + " --no-gui")


if __name__ == "__main__":
    pick_granules()
