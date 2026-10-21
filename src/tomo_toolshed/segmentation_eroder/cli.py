"""``tomo_toolshed erode`` -- erode a binary MRC segmentation in XY (default) or 3D.

Reads the mask, treats every non-zero voxel as foreground, erodes it
``--iterations`` times and writes a uint8 0/1 MRC with the input's voxel size
and origin. Axis convention follows ``curate``: data are ``(Z, Y, X)``, so XY
erosion is per Z slice. The erosion itself lives in ``core.py``.
"""

import os
from pathlib import Path

import click

from .core import CONNECTIVITY_3D, CONNECTIVITY_XY, ErodeError, erode, read_mask, write_mask


def default_output_path(input_path):
    """``<input stem>_eroded<suffix>`` next to the input."""
    p = Path(input_path)
    return str(p.with_name(p.stem + "_eroded" + (p.suffix or ".mrc")))


@click.command(name="erode")
@click.option("--input", "--i", "input_path", required=True,
              type=click.Path(exists=True, dir_okay=False),
              help="Binary segmentation MRC (any non-zero voxel is foreground).")
@click.option("--output", "--o", "output_path", default=None,
              type=click.Path(dir_okay=False),
              help="Eroded MRC (parent directories are created). "
                   "[default: <input>_eroded.mrc]")
@click.option("--iterations", "-n", type=click.IntRange(min=1), default=1,
              show_default=True, help="Number of erosion iterations (1 px each).")
@click.option("--3d", "three_d", is_flag=True, default=False,
              help="Erode in 3D (also along Z). Default: XY plane only, "
                   "slice by slice.")
@click.option("--connectivity", type=click.Choice(
                  [str(c) for c in CONNECTIVITY_XY + CONNECTIVITY_3D]),
              default=None,
              help="Structuring element. XY: 4 (cross) or 8 (square); "
                   "3D: 6 (cross), 18 or 26 (cube). [default: 4 in XY, 6 in 3D]")
@click.option("--quiet", "-q", is_flag=True, default=False,
              help="Only print the final summary.")
def erode_cmd(input_path, output_path, iterations, three_d, connectivity, quiet):
    """Erode a binary segmentation MRC in the XY plane (default) or in 3D.

    Volumes are (Z, Y, X), as in curate: XY erosion shrinks each Z slice
    in-plane and never mixes slices; --3d also erodes along Z. Each iteration
    removes one pixel layer. Voxels outside the volume count as foreground, so
    masks touching the edge are not eroded from the edge. Output is a uint8
    0/1 MRC with the input's voxel size and origin.

    \b
    example:
    tomo_toolshed erode --i membrane.mrc --o membrane_eroded.mrc
    tomo_toolshed erode --i membrane.mrc -n 3
    tomo_toolshed erode --i membrane.mrc --o eroded.mrc -n 2 --3d --connectivity 26
    """
    output_path = output_path or default_output_path(input_path)
    if os.path.realpath(output_path) == os.path.realpath(input_path):
        raise click.ClickException("--output must differ from --input")
    conn = int(connectivity) if connectivity is not None else None

    try:
        data, voxel_size, origin = read_mask(input_path)
        before = int((data > 0).sum())
        eroded = erode(data, iterations=iterations, three_d=three_d, connectivity=conn)
    except ErodeError as exc:
        raise click.ClickException(str(exc))

    after = int(eroded.sum())
    mode = "3D" if three_d else "XY"
    conn_used = conn if conn is not None else (6 if three_d else 4)
    if not quiet:
        click.echo(f"input:         {input_path}")
        click.echo(f"shape (Z,Y,X): {tuple(data.shape)}")
        click.echo(f"voxel size:    {float(voxel_size.x):g} Å")
        click.echo(f"erosion:       {mode}, connectivity {conn_used}, "
                   f"{iterations} iteration(s)")
        click.echo(f"output:        {output_path}")
    if after == 0 and before > 0:
        click.echo("warning: the mask was eroded away completely", err=True)

    parent = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(parent, exist_ok=True)
    write_mask(output_path, eroded, voxel_size=voxel_size, origin=origin)
    pct = 100.0 * (before - after) / before if before else 0.0
    click.echo(f"foreground {before} -> {after} voxels ({pct:.1f}% removed) "
               f"-> {output_path}")


if __name__ == "__main__":
    erode_cmd()
