"""``tomo_toolshed scale-star`` -- rescale star coordinates between pixel sizes.

Rescales the particle coordinates of a star file from ``--input-pixel-size`` to
``--output-pixel-size`` (e.g. after re-binning tomograms), optionally applies a
shift in output pixels, and rewrites the coordinate pixel-size fields to the new
value. The grouping/flavor reading and star I/O are reused from split-star, so
all four flavors (RELION 3/4/5, M/Warp) round-trip identically; this layer wires
options, the pixel-size mismatch prompt, and reporting.
"""

import os

import click

from ..split_star.core import detect_flavor, read_star, write_star_file
from .core import (
    PIXEL_SIZE_COLUMNS,
    ScaleStarError,
    build_header_lines,
    collect_pixel_sizes,
    find_coordinate_columns,
    override_pixel_sizes,
    pixel_size_mismatches,
    scale_particles,
)


@click.command(name="scale-star")
@click.option("--input", "--i", "input_path", required=True,
              type=click.Path(exists=True, dir_okay=False),
              help="Input star file to rescale.")
@click.option("--output", "--o", "output_path", required=True,
              type=click.Path(dir_okay=False),
              help="Output star file (parent directories are created).")
@click.option("--input-pixel-size", "--input_pixel_size", "input_pixel_size",
              type=float, required=True,
              help="Pixel size (Angstrom) the coordinates are currently in.")
@click.option("--output-pixel-size", "--output_pixel_size", "output_pixel_size",
              type=float, required=True,
              help="Target pixel size (Angstrom) to rescale the coordinates to.")
@click.option("--shift-x", "--shift_x", "shift_x", type=float, default=0.0,
              help="Shift added to X after scaling, in output pixels.")
@click.option("--shift-y", "--shift_y", "shift_y", type=float, default=0.0,
              help="Shift added to Y after scaling, in output pixels.")
@click.option("--shift-z", "--shift_z", "shift_z", type=float, default=0.0,
              help="Shift added to Z after scaling, in output pixels.")
@click.option("--yes", "-y", is_flag=True, default=False,
              help="Skip the pixel-size mismatch confirmation prompt.")
@click.option("--comment/--no-comment", default=True, show_default=True,
              help="Prepend a provenance comment (source, pixel sizes, factor, "
                   "shifts) to the output star file.")
@click.option("--dry-run", is_flag=True, default=False,
              help="Report what would change; write nothing.")
@click.option("--quiet", "-q", is_flag=True, default=False,
              help="Only print the final summary.")
def scale_star(input_path, output_path, input_pixel_size, output_pixel_size,
               shift_x, shift_y, shift_z, yes, comment, dry_run, quiet):
    """Rescale particle coordinates from one pixel size to another.

    Multiplies every coordinate column by ``input-pixel-size / output-pixel-size``,
    adds any --shift-* (in output pixels), and sets the coordinate pixel-size
    columns (rlnPixelSize, rlnMicrographPixelSize, rlnTomoTiltSeriesPixelSize) to
    the new value. rlnImagePixelSize and every *Angst column (rlnOrigin*Angst,
    rlnCenteredCoordinate*Angst, ...) are left untouched. All four star flavors
    are supported and every other block/column is carried through unchanged.

    \b
    example:
    tomo_toolshed scale-star --i in.star --o out.star --input-pixel-size 10 --output-pixel-size 5
    tomo_toolshed scale-star --i in.star --o out.star --input-pixel-size 10 --output-pixel-size 5 --shift-z 2
    tomo_toolshed scale-star --i in.star --o out.star --input-pixel-size 10 --output-pixel-size 5 --yes
    """
    if input_pixel_size <= 0 or output_pixel_size <= 0:
        raise click.ClickException("pixel sizes must be > 0")
    if os.path.realpath(input_path) == os.path.realpath(output_path):
        raise click.ClickException("--output must differ from --input")

    shifts = {"X": shift_x, "Y": shift_y, "Z": shift_z}
    factor = input_pixel_size / output_pixel_size

    try:
        blocks, part_key = read_star(input_path)
        particles = blocks[part_key]
        flavor = detect_flavor(blocks, particles)
        coord_columns = find_coordinate_columns(particles)
    except ScaleStarError as exc:
        raise click.ClickException(str(exc))

    # Pixel-size sanity: warn (and, interactively, confirm) if the file's own
    # pixel size disagrees with --input-pixel-size. The user's value always wins.
    pixel_sizes = collect_pixel_sizes(blocks)
    if not pixel_sizes:
        if not quiet:
            click.echo("note: no pixel-size column in this file; cannot verify "
                       "--input-pixel-size (proceeding).")
    else:
        mismatches = pixel_size_mismatches(pixel_sizes, input_pixel_size)
        if mismatches:
            click.echo(f"warning: file pixel size differs from --input-pixel-size="
                       f"{input_pixel_size}:")
            for key, col, values in mismatches:
                vals = ", ".join(repr(v) for v in values)
                click.echo(f"  [{key}] {col}: found {vals}")
            if not dry_run and not yes:
                click.confirm(
                    f"Proceed using --input-pixel-size={input_pixel_size}?",
                    abort=True)

    new_blocks, updated = override_pixel_sizes(blocks, output_pixel_size)
    new_blocks[part_key] = scale_particles(
        new_blocks[part_key], coord_columns, factor, shifts)
    n = len(new_blocks[part_key])

    if not quiet:
        click.echo(f"input:            {input_path}")
        click.echo(f"detected flavor:  {flavor}")
        click.echo(f"scale factor:     {factor} "
                   f"({input_pixel_size} -> {output_pixel_size} Apx)")
        click.echo(f"shifts (out px):  x={shift_x}, y={shift_y}, z={shift_z}")
        click.echo(f"coord columns:    {', '.join(coord_columns)}")
        if updated:
            cols = ", ".join(f"[{k}] {c}" for k, c in updated)
            click.echo(f"pixel-size cols:  {cols} -> {output_pixel_size}")
        else:
            click.echo(f"pixel-size cols:  none "
                       f"(none of {', '.join(PIXEL_SIZE_COLUMNS)} present)")
        click.echo(f"output:           {output_path}")
        click.echo("")

    if dry_run:
        click.echo(f"dry run: would scale {n} particle(s) by {factor} "
                   f"-> {output_path} (nothing written)")
        return

    header = build_header_lines(input_path, input_pixel_size, output_pixel_size,
                                factor, shifts) if comment else None
    write_star_file(new_blocks, output_path, header_lines=header)
    click.echo(f"scaled {n} particle(s) by {factor} -> {output_path}")


if __name__ == "__main__":
    scale_star()
