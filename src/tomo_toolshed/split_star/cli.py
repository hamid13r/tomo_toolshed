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
    plan_split,
    read_star,
    resolve_group_columns,
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
@click.option("--group-by", "--group_by", "group_by", multiple=True,
              help="Column to split on (repeatable). Any column works, not just "
                   "the name-like ones; repeat it to split by the combination "
                   "(e.g. --group-by rlnTomoName --group-by rlnClassNumber). "
                   "Default: the auto-detected name column.")
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

    Each group is written to ``<outdir>/<label>_<name>/<label>_<name>_all.star``,
    carrying through any optics/general blocks. The grouping column is
    auto-detected (rlnTomoName, then rlnMicrographName, then wrpSourceName) unless
    --group-by is given; --group-by accepts any column and is repeatable, so you
    can split by a combination (one file per unique tuple of values).

    \b
    example:
    tomo_toolshed split-star --i run_data.star --label EXP
    tomo_toolshed split-star --i picks.star --group-by rlnTomoName --outdir split
    tomo_toolshed split-star --i run_data.star --group-by rlnTomoName --group-by rlnClassNumber
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
        group_columns = resolve_group_columns(particles, group_by)
        plan = plan_split(particles, group_columns, label, outdir=outdir,
                          strip_suffixes=strip_suffixes,
                          strip_patterns=strip_patterns)
    except SplitStarError as exc:
        raise click.ClickException(str(exc))

    # Guard against two different column combinations collapsing to the same
    # output name (possible with multi-column splits joined by '_'), which would
    # silently overwrite one group's file with another's.
    from collections import Counter
    counts = Counter(output_path for _, output_path, _ in plan)
    collisions = sorted(p for p, c in counts.items() if c > 1)
    if collisions:
        raise click.ClickException(
            "grouping produces colliding output names (would overwrite):\n  "
            + "\n  ".join(collisions)
            + "\nUse a different --group-by order/columns or split in one column "
              "per run.")

    if not quiet:
        click.echo(f"input:            {input_path}")
        click.echo(f"detected flavor:  {flavor}")
        click.echo(f"grouping columns: {', '.join(group_columns)} ({len(plan)} groups)")
        click.echo(f"name stripping:   {stripping_report}")
        click.echo(f"output root:      {outdir}")
        click.echo("")

    if dry_run:
        for _, output_path, n_rows in plan:
            click.echo(f"  would write {n_rows} row(s) -> {output_path}")
        click.echo(f"\ndry run: {len(plan)} file(s) not written")
        return

    total = 0
    for combo, output_path, _ in plan:
        n = write_group(blocks, part_key, particles, group_columns,
                        combo, output_path)
        total += n
        if not quiet:
            click.echo(f"  {n} row(s) -> {output_path}")

    click.echo(f"wrote {len(plan)} file(s) covering {total} particle(s) "
               f"under {Path(outdir)}")


if __name__ == "__main__":
    split_star()
