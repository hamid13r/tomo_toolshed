"""``tomo_toolshed split-star`` -- split one star file into per-group star files.

Groups a particle star file by its grouping column (tomogram / micrograph /
source) and writes one ``<label>_<name>/<label>_<name>_all.star`` per group. The
grouping and I/O live in ``core.py``; this layer wires options and reporting.
"""

from pathlib import Path

import click

from .core import (
    DEFAULT_STRIP_PATTERNS,
    DEFAULT_STRIP_SUFFIXES,
    STRIP_PATTERN_LABEL,
    SplitStarError,
    detect_flavor,
    detect_group_column,
    plan_split,
    read_star,
    write_group,
)


@click.command(name="split-star")
@click.option("--input", "--i", "input_path", required=True,
              type=click.Path(exists=True, dir_okay=False),
              help="Input star file to split.")
@click.option("--label", default=None,
              help="Prefix added to each output directory and file "
                   "(e.g. <label>_<name>). Omit for no prefix.")
@click.option("--outdir", "-o", default=".", show_default=True,
              type=click.Path(file_okay=False),
              help="Directory to write the per-group subdirectories into.")
@click.option("--group-by", "--group_by", "group_by", default=None,
              help="Override the auto-detected grouping column.")
@click.option("--strip-suffix", "--strip_suffix", "strip_suffix", multiple=True,
              help="Suffix stripped from each group name to form its "
                   "directory/file base name (repeatable, longest match wins). "
                   "If given, takes full control (the built-in defaults, incl. the "
                   f"{STRIP_PATTERN_LABEL} pattern, are not applied). "
                   f"Default: {', '.join(DEFAULT_STRIP_SUFFIXES)}, {STRIP_PATTERN_LABEL}.")
@click.option("--dry-run", is_flag=True, default=False,
              help="List what would be written; create nothing.")
@click.option("--quiet", "-q", is_flag=True, default=False,
              help="Only print the final summary.")
def split_star(input_path, label, outdir, group_by, strip_suffix, dry_run, quiet):
    """Split a star file into one star file per group.

    Each group (tomogram/micrograph/source) is written to
    ``<outdir>/<label>_<name>/<label>_<name>_all.star``, carrying through any
    optics/general blocks. The grouping column is auto-detected (rlnTomoName,
    then rlnMicrographName, then wrpSourceName) unless --group-by is given.

    \b
    example:
    tomo_toolshed split-star --i run_data.star --label EXP
    tomo_toolshed split-star --i picks.star --group-by rlnTomoName --outdir split
    """
    # Explicit --strip-suffix takes full control (patterns off); otherwise use the
    # built-in defaults, which include the .mrc_<pixelsize>Apx.mrc pattern.
    if strip_suffix:
        strip_suffixes, strip_patterns = strip_suffix, ()
        stripping_report = ", ".join(strip_suffixes)
    else:
        strip_suffixes, strip_patterns = DEFAULT_STRIP_SUFFIXES, DEFAULT_STRIP_PATTERNS
        stripping_report = ", ".join((*DEFAULT_STRIP_SUFFIXES, STRIP_PATTERN_LABEL))

    try:
        blocks, part_key = read_star(input_path)
        particles = blocks[part_key]
        flavor = detect_flavor(blocks, particles)
        group_column = detect_group_column(particles, group_by)
        plan = plan_split(particles, group_column, label, outdir=outdir,
                          strip_suffixes=strip_suffixes,
                          strip_patterns=strip_patterns)
    except SplitStarError as exc:
        raise click.ClickException(str(exc))

    if not quiet:
        click.echo(f"input:           {input_path}")
        click.echo(f"detected flavor: {flavor}")
        click.echo(f"grouping column: {group_column} ({len(plan)} groups)")
        click.echo(f"name stripping:  {stripping_report}")
        click.echo(f"output root:     {outdir}")
        click.echo("")

    if dry_run:
        for _, output_path, n_rows in plan:
            click.echo(f"  would write {n_rows} row(s) -> {output_path}")
        click.echo(f"\ndry run: {len(plan)} file(s) not written")
        return

    total = 0
    for group_name, output_path, _ in plan:
        n = write_group(blocks, part_key, particles, group_column,
                        group_name, output_path)
        total += n
        if not quiet:
            click.echo(f"  {n} row(s) -> {output_path}")

    click.echo(f"wrote {len(plan)} file(s) covering {total} particle(s) "
               f"under {Path(outdir)}")


if __name__ == "__main__":
    split_star()
