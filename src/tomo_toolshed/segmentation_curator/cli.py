"""Command-line entry point for segmentation-curator.

    segmentation-curator TOMOGRAM SEGMENTATION OUTPUT_DIR [options]

Loads a tomogram + segmentation pair, labels the segmentation into connected
"islands", launches the interactive curator GUI, and writes the curated binary
mask to ``OUTPUT_DIR/<segmentation basename>``. If that output already exists
the run is skipped (exit 0) without opening the GUI.

If SEGMENTATION is a ``.star`` file (RELION 4.x particles, one tomogram), each
particle is drawn as a sphere of ``--radius`` Å and becomes its own island;
after curation the star file is written to ``OUTPUT_DIR/<star basename>`` with
the deselected (false-positive) particles removed.
"""

from __future__ import annotations

import os
import sys

import click
import numpy as np

from . import io as mrc_io
from . import labeling
from . import particles as particle_ops


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("tomogram", type=click.Path(exists=True, dir_okay=False))
@click.argument("segmentation", type=click.Path(exists=True, dir_okay=False))
@click.argument("output_dir", type=click.Path(file_okay=False))
@click.option("--z-min", type=int, default=None, help="Initial Z-range lower bound (inclusive).")
@click.option("--z-max", type=int, default=None, help="Initial Z-range upper bound (inclusive).")
@click.option("--min-size", type=int, default=0, help="Initial minimum island size in voxels.")
@click.option(
    "--threshold",
    type=click.IntRange(0, 128),
    default=None,
    help="Confidence threshold (0-128): islands with no voxel value above it are "
    "removed. Prompted interactively if omitted.",
)
@click.option(
    "--blur",
    type=float,
    default=0.0,
    help="Gaussian blur sigma applied to the tomogram for display (0 = none).",
)
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
@click.option(
    "--radius",
    "radius_a",
    type=float,
    default=None,
    help="[star input] Sphere radius in Å drawn around each particle "
    "(required for .star input; adjustable in the GUI).",
)
@click.option(
    "--coord-pixel-size",
    type=float,
    default=None,
    help="[star input] Pixel size (Å/px) of the star coordinates, if different "
    "from the tomogram's. Default: same as the tomogram.",
)
@click.option(
    "--tomo-pixel-size",
    type=float,
    default=None,
    help="[star input] Tomogram pixel size (Å/px). Default: read from the MRC header.",
)
def main(tomogram, segmentation, output_dir, z_min, z_max, min_size, threshold, blur,
         connectivity, color_by_number, radius_a, coord_pixel_size, tomo_pixel_size):
    """Curate a 3D SEGMENTATION over a TOMOGRAM and export to OUTPUT_DIR.

    SEGMENTATION is a mask (.mrc) or a RELION 4 particle .star file. For a
    .star, each particle is shown as a sphere of --radius Å; the output is the
    star file with the deselected (false-positive) particles removed.
    """
    connectivity = int(connectivity)

    out_name = os.path.basename(segmentation)
    out_path = os.path.join(output_dir, out_name)

    # Skip-if-exists: mirror the old "already processed" behavior.
    if os.path.exists(out_path):
        click.echo(f"[skip] Output already exists: {out_path}")
        sys.exit(0)

    if particle_ops.is_star_path(segmentation):
        _curate_particles(tomogram, segmentation, out_path, output_dir, z_min, z_max,
                          blur, color_by_number, radius_a, coord_pixel_size,
                          tomo_pixel_size)
        return
    star_only = {"--radius": radius_a, "--coord-pixel-size": coord_pixel_size,
                 "--tomo-pixel-size": tomo_pixel_size}
    given = [k for k, v in star_only.items() if v is not None]
    if given:
        raise click.UsageError(f"{', '.join(given)} only apply to .star input.")

    # Ask for the confidence threshold up front (skip the prompt if supplied).
    if threshold is None:
        threshold = click.prompt(
            "Segmentation confidence threshold (0-128)", type=click.IntRange(0, 128)
        )

    click.echo(f"Reading tomogram:     {tomogram}")
    tomo = mrc_io.read_mrc(tomogram)
    click.echo(f"Reading segmentation: {segmentation}")
    seg = mrc_io.read_mrc(segmentation)

    if tomo.shape != seg.shape:
        raise click.ClickException(
            f"Tomogram and segmentation shapes differ: {tomo.shape} vs {seg.shape}"
        )

    tomo = _maybe_blur(tomo, blur)

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

    # Confidence-threshold cull: keep only islands that contain at least one
    # voxel whose segmentation value is above the threshold; drop the rest.
    keep = labeling.filter_by_value(labels, seg, n, threshold)
    labels, id_map = labeling.renumber(labels, keep)
    n = len(id_map)
    click.echo(
        f"Threshold {threshold}: kept {n} islands with a voxel above it "
        f"(removed the rest)."
    )

    # Initial min-size filter (drop below threshold, relabel contiguous).
    if min_size > 0:
        _, sizes = labeling.compute_bboxes_and_sizes(labels, n)
        keep = labeling.filter_by_size(sizes, min_size)
        labels, id_map = labeling.renumber(labels, keep)
        n = len(id_map)
        click.echo(f"Applied initial min-size filter (>= {min_size} vox): {n} islands remain.")

    # Launch GUI (import lazily so headless tooling never imports matplotlib).
    from .gui import SegmentationCuratorGUI

    click.echo("Launching curator GUI... (press 'q' to save & quit)")
    gui = SegmentationCuratorGUI(
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


def _maybe_blur(tomo, blur):
    """Optional Gaussian blur of the tomogram (display only; does not touch the
    segmentation). A 3D blur keeps both orthogonal views consistent."""
    if blur and blur > 0:
        from scipy.ndimage import gaussian_filter

        click.echo(f"Applying Gaussian blur to tomogram (sigma={blur}).")
        tomo = gaussian_filter(tomo.astype(np.float32), sigma=blur)
    return tomo


def _curate_particles(tomogram, star_path, out_path, output_dir, z_min, z_max, blur,
                      color_by_number, radius_a, coord_pixel_size, tomo_pixel_size):
    """Star-file mode: particles -> spheres -> GUI -> star minus rejected rows."""
    if radius_a is None or radius_a <= 0:
        raise click.UsageError("--radius (sphere radius in Å, > 0) is required for .star input.")

    if tomo_pixel_size is None:
        vs = mrc_io.get_voxel_size(tomogram)
        tomo_pixel_size = float(vs.x) if vs is not None else 0.0
        if tomo_pixel_size <= 0:
            raise click.UsageError(
                "tomogram MRC header has no pixel size; pass --tomo-pixel-size.")

    click.echo(f"Reading particles:    {star_path}")
    try:
        blocks, key, df = particle_ops.load_particles(star_path)
    except particle_ops.ParticleStarError as exc:
        raise click.ClickException(str(exc))
    click.echo(f"Reading tomogram:     {tomogram}")
    tomo = mrc_io.read_mrc(tomogram)
    tomo = _maybe_blur(tomo, blur)

    centers = particle_ops.particle_centers_zyx(df, tomo_pixel_size, coord_pixel_size)
    labels = particle_ops.paint_spheres(tomo.shape, centers, radius_a / tomo_pixel_size)
    n = len(df)
    drawn = len(np.unique(labels)) - 1
    click.echo(f"{n} particles; sphere radius {radius_a:g} Å "
               f"({radius_a / tomo_pixel_size:.1f} px at {tomo_pixel_size:g} Å/px).")
    if drawn < n:
        click.echo(f"  note: {n - drawn} particle(s) fall outside the tomogram and are "
                   "not shown; they are kept unless the Z-range excludes them.")

    selected = set(range(1, n + 1))
    if z_min is not None or z_max is not None:
        zmin = 0 if z_min is None else z_min
        zmax = (tomo.shape[0] - 1) if z_max is None else z_max
        selected &= particle_ops.ids_in_zrange(centers, zmin, zmax)
        click.echo(f"Initial Z-range [{zmin}, {zmax}]: {len(selected)} particles selected.")

    from .gui import SegmentationCuratorGUI

    click.echo("Launching curator GUI... (press 'q' to save & quit)")
    gui = SegmentationCuratorGUI(
        tomogram=tomo,
        labels=labels,
        n_islands=n,
        color_by_number=color_by_number,
        selected=selected,
        particle_centers=centers,
        radius_a=radius_a,
        pixel_size=tomo_pixel_size,
    )
    gui.run()

    os.makedirs(output_dir, exist_ok=True)
    kept = particle_ops.write_kept_particles(blocks, key, df, gui.selected, out_path)
    click.echo(f"Wrote curated particles: {out_path}  "
               f"(kept {kept} of {n}; removed {n - kept})")
