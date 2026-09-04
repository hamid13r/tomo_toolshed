"""``tomo_toolshed trace-filaments`` — trace filaments in a mask to a helical star.

Skeletonizes the connected components of a binary segmentation mask, extracts
each filament's centerline, resamples it into evenly spaced particles with
per-particle ZYZ Euler angles from the local tangent, and writes a RELION-style
helical ``.star`` file. A ChimeraX ``.bild`` overlay is opt-in via ``--bild``.
"""

import click

from .core import default_bild_path
from .core import trace_filaments as run_trace


@click.command(name="trace-filaments")
@click.argument("mask", type=click.Path(exists=True, dir_okay=False))
@click.option("--output", "-o", "star_out", default="particles.star",
              show_default=True, type=click.Path(dir_okay=False),
              help="Output helical star file.")
@click.option("--pixel-size", type=float, default=None,
              help="Pixel size in Å/px. Read from the MRC header if omitted.")
@click.option("--spacing", "spacing_a", type=float, default=82.0,
              show_default=True, help="Å between particles along a filament.")
@click.option("--min-voxels", type=int, default=50, show_default=True,
              help="Drop connected components smaller than this many voxels.")
@click.option("--min-length", "min_path_a", type=float, default=500.0,
              show_default=True, help="Drop centerlines shorter than this (Å).")
@click.option("--smooth-factor", type=float, default=1.0, show_default=True,
              help="splprep smoothing: s = smooth-factor * n_points (0 = interpolate).")
@click.option("--presmooth-window", type=int, default=3, show_default=True,
              help="Moving-average window on ordered points (odd; 1 = off).")
@click.option("--micrograph", default=None,
              help="rlnMicrographName value. Defaults to the mask file name.")
@click.option("--magnification", type=float, default=10000, show_default=True,
              help="rlnMagnification value.")
@click.option("--group-number", type=int, default=1, show_default=True,
              help="rlnGroupNumber value.")
@click.option("--tangent-axis", type=click.Choice(["x", "y", "z"]), default="z",
              show_default=True, help="Particle axis aligned to the filament tangent.")
@click.option("--invert-rot/--no-invert-rot", default=True, show_default=True,
              help="Transpose the rotation matrix before decompose "
                   "(RELION reference->particle convention).")
@click.option("--bild", "write_bild_file", is_flag=True, default=False,
              help="Also write a ChimeraX .bild overlay (off by default).")
@click.option("--bild-dir", default=None, type=click.Path(file_okay=False),
              help="Directory for the .bild file (default: alongside --output). "
                   "Only used with --bild.")
def trace_filaments(mask, star_out, pixel_size, spacing_a, min_voxels, min_path_a,
                    smooth_factor, presmooth_window, micrograph, magnification,
                    group_number, tangent_axis, invert_rot, write_bild_file, bild_dir):
    """Trace filaments in a binary MASK and export a helical star file.

    Separates the mask into 26-connected islands, skeletonizes each, extracts
    the longest-path centerline, resamples it at --spacing into particles with
    ZYZ Euler angles from the local tangent, and writes a RELION helical star.
    Runs non-interactively for batch use. ChimeraX .bild overlay is opt-in.

    \b
    example:
    tomo_toolshed trace-filaments mask.mrc -o particles.star --pixel-size 9.98
    """
    bild_path = default_bild_path(star_out, bild_dir) if write_bild_file else None

    run_trace(
        mask, star_out,
        pixel_size=pixel_size,
        spacing_a=spacing_a,
        min_voxels=min_voxels,
        min_path_a=min_path_a,
        smooth_factor=smooth_factor,
        presmooth_win=presmooth_window,
        micrograph=micrograph,
        magnification=magnification,
        group_number=group_number,
        tangent_axis=tangent_axis,
        invert_rot=invert_rot,
        write_bild_file=write_bild_file,
        bild_path=bild_path,
        log=click.echo,
    )


if __name__ == "__main__":
    trace_filaments()
