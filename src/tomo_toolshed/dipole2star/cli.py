"""``tomo_toolshed dipole2star`` — collapse manual dipole picks into a star file.

Each input file holds manual dipole picks as consecutive point pairs: for every
particle, one row marks one end and the next row marks the other end of its long
axis. This collapses each pair into a single centered, oriented particle (ZYZ
Euler angles from ``scipy`` ``align_vectors``) and writes a RELION-style
particle star file per input, named ``<stem>.mrc.star``. Inputs may be RELION
star files or plain 3-column (X Y Z) text files.
"""

import click
import numpy as np

from .core import (
    DEFAULT_MICROGRAPH_SUFFIX,
    DipoleError,
    convert_one,
    expand_paths,
)


@click.command(name="dipole2star")
@click.argument('star_files', nargs=-1, required=True)
@click.option('--scale', type=float, required=True,
              help="Scale picked coordinates by this factor.")
@click.option('--random', default=False, is_flag=True,
              help="Randomize the particle's azimuthal orientation about its pick axis.")
@click.option('--output-apix', 'pixel_size', type=float, required=True,
              help="Pixel size (Angstrom/px) to write into the rlnPixelSize column.")
@click.option('--outdir', default='.', show_default=True,
              help="Directory to write the output star files to.")
@click.option('--micrograph-suffix', default=DEFAULT_MICROGRAPH_SUFFIX, show_default=True,
              help="Appended to each input's stem to form rlnMicrographName "
                   "(match your reconstruction file names).")
@click.option('--format', 'fmt', type=click.Choice(['star', 'txt', 'auto']),
              default='auto', show_default=True,
              help="Input reader. 'auto' picks by extension (.star -> star, else txt).")
@click.option('--seed', type=int, default=None,
              help="Seed for --random. If omitted, a seed is drawn and printed "
                   "so the run can be reproduced.")
def dipole2star(star_files, scale, random, pixel_size, outdir, micrograph_suffix,
                fmt, seed):
    """Convert manual-pick STAR_FILES into oriented-particle star files.

    Each input holds dipole picks as consecutive point pairs (row i and row i+1
    are the two ends of one particle's long axis). Each pair becomes one
    centered, oriented particle, written to ``<stem>.mrc.star`` in --outdir.
    Inputs may be RELION star files or plain 3-column (X Y Z) text files.

    \b
    example:
    tomo_toolshed dipole2star picks.star --scale 2 --output-apix 9.98
    tomo_toolshed dipole2star "picks/*.txt" --scale 2 --output-apix 9.98 --outdir out
    """
    rng = None
    if random:
        if seed is None:
            seed = int(np.random.default_rng().integers(0, 2 ** 63 - 1))
            click.echo(f"random: drew seed {seed} (pass --seed {seed} to reproduce)")
        else:
            click.echo(f"random: using seed {seed}")
        rng = np.random.default_rng(seed)

    for path in expand_paths(star_files):
        try:
            outpath, n = convert_one(
                path, scale, random, pixel_size, outdir,
                micrograph_suffix=micrograph_suffix, fmt=fmt, rng=rng,
            )
        except (DipoleError, ValueError) as exc:
            raise click.ClickException(str(exc))
        click.echo(f"{path}: {n} particles -> {outpath}")


if __name__ == '__main__':
    dipole2star()
