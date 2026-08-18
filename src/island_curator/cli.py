"""Command-line entry point for island-curator.

    island-curator TOMOGRAM SEGMENTATION OUTPUT_DIR [options]

Loads a tomogram + segmentation pair, labels the segmentation into connected
"islands", launches the interactive curator GUI, and writes the curated binary
mask to ``OUTPUT_DIR/<segmentation basename>``. If that output already exists
the run is skipped (exit 0) without opening the GUI.
"""

from __future__ import annotations

import os
import sys

import click
import numpy as np

from . import io as mrc_io
from . import labeling


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("tomogram", type=click.Path(exists=True, dir_okay=False))
@click.argument("segmentation", type=click.Path(exists=True, dir_okay=False))
@click.argument("output_dir", type=click.Path(file_okay=False))
@click.option("--z-min", type=int, default=None, help="Initial Z-range lower bound (inclusive).")
@click.option("--z-max", type=int, default=None, help="Initial Z-range upper bound (inclusive).")
@click.option("--min-size", type=int, default=0, help="Initial minimum island size in voxels.")
@click.option(
    "--connectivity",
    type=click.Choice(["6", "18", "26"]),
    default="26",
    help="Connected-components connectivity.",
)
@click.option(
    "--color-by-number/--all-green",
    default=False,
    help="Initial overlay mode (default: all-green).",
)
def main(tomogram, segmentation, output_dir, z_min, z_max, min_size, connectivity, color_by_number):
    """Curate a 3D SEGMENTATION over a TOMOGRAM and export to OUTPUT_DIR."""
    connectivity = int(connectivity)

    out_name = os.path.basename(segmentation)
    out_path = os.path.join(output_dir, out_name)

    # Skip-if-exists: mirror the old "already processed" behavior.
    if os.path.exists(out_path):
        click.echo(f"[skip] Output already exists: {out_path}")
        sys.exit(0)

    click.echo(f"Reading tomogram:     {tomogram}")
    tomo = mrc_io.read_mrc(tomogram)
    click.echo(f"Reading segmentation: {segmentation}")
    seg = mrc_io.read_mrc(segmentation)

    if tomo.shape != seg.shape:
        raise click.ClickException(
            f"Tomogram and segmentation shapes differ: {tomo.shape} vs {seg.shape}"
        )

    # Binarize the segmentation (it is the input directly -- do NOT threshold
    # the tomogram).
    binary = (seg > 0).astype(np.uint8)

    # Initial Z-range crop.
    nz = binary.shape[0]
    zmin = 0 if z_min is None else z_min
    zmax = (nz - 1) if z_max is None else z_max
    if z_min is not None or z_max is not None:
        binary = labeling.apply_zrange(binary, zmin, zmax)
        click.echo(f"Applied initial Z-range crop [{zmin}, {zmax}].")

    # Label.
    labels, n = labeling.label_islands(binary, connectivity=connectivity)
    click.echo(f"Labeled {n} islands (connectivity={connectivity}).")

    # Initial min-size filter (drop below threshold, relabel contiguous).
    if min_size > 0:
        _, sizes = labeling.compute_bboxes_and_sizes(labels, n)
        keep = labeling.filter_by_size(sizes, min_size)
        labels, id_map = labeling.renumber(labels, keep)
        n = len(id_map)
        click.echo(f"Applied initial min-size filter (>= {min_size} vox): {n} islands remain.")

    # Launch GUI (import lazily so headless tooling never imports matplotlib).
    from .gui import IslandCuratorGUI

    click.echo("Launching curator GUI... (press 'q' to save & quit)")
    gui = IslandCuratorGUI(
        tomogram=tomo,
        labels=labels,
        n_islands=n,
        connectivity=connectivity,
        color_by_number=color_by_number,
    )
    curated = gui.run()

    # Write curated binary mask (uint8 0/1) to OUTPUT_DIR/<seg basename>.
    os.makedirs(output_dir, exist_ok=True)
    curated = (np.asarray(curated) > 0).astype(np.uint8)
    voxel_size = mrc_io.get_voxel_size(segmentation)
    mrc_io.write_mrc(out_path, curated, voxel_size=voxel_size, overwrite=True)
    click.echo(f"Wrote curated mask: {out_path}  ({int(curated.sum())} foreground voxels)")


if __name__ == "__main__":
    main()
