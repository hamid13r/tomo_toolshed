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
    Range,
    SplitStarError,
    build_header_lines,
    detect_flavor,
    plan_split,
    read_star,
    resolve_group_columns,
    split_spec_string,
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
@click.option("--range-by", "--range_by", "range_by", default=None,
              help="Numeric column to split into ranges at --breaks (half-open "
                   "bins [low, high)). Combines with --group-by.")
@click.option("--breaks", default=None,
              help="Comma-separated break points for --range-by, e.g. 100,200,300.")
@click.option("--strip-suffix", "--strip_suffix", "strip_suffix", multiple=True,
              help="Suffix stripped from each group name to form its "
                   "directory/file base name (repeatable, longest match wins). "
                   "If given, takes full control (the built-in defaults, incl. the "
                   f"{STRIP_PATTERN_LABEL} pattern, are not applied). "
                   f"Default: {', '.join(DEFAULT_STRIP_SUFFIXES)}, {STRIP_PATTERN_LABEL}.")
@click.option("--comment/--no-comment", default=True, show_default=True,
              help="Prepend a provenance comment (source, split spec, this "
                   "file's values) to each output star file.")
@click.option("--dry-run", is_flag=True, default=False,
              help="List what would be written; create nothing.")
@click.option("--quiet", "-q", is_flag=True, default=False,
              help="Only print the final summary.")
def split_star(input_path, label, outdir, group_by, range_by, breaks,
               strip_suffix, comment, dry_run, quiet):
    """Split a star file into one star file per group.

    Each group is written to ``<outdir>/<label>_<name>/<label>_<name>_all.star``,
    carrying through any optics/general blocks. The grouping column is
    auto-detected (rlnTomoName, then rlnMicrographName, then wrpSourceName) unless
    --group-by is given; --group-by accepts any column and is repeatable. Add
    --range-by COLUMN --breaks a,b,c to also split a numeric column into ranges.
    Splitting is by the combination of all dimensions.

    \b
    example:
    tomo_toolshed split-star --i run_data.star --label EXP
    tomo_toolshed split-star --i run_data.star --group-by rlnTomoName --group-by rlnClassNumber
    tomo_toolshed split-star --i run_data.star --range-by rlnDistanceFromtop --breaks 100,200,300
    """
    # Explicit --strip-suffix takes full control (patterns off); otherwise use the
    # built-in defaults, which include the .mrc_<pixelsize>Apx.mrc pattern.
    if strip_suffix:
        strip_suffixes, strip_patterns = strip_suffix, ()
        stripping_report = ", ".join(strip_suffixes)
    else:
        strip_suffixes, strip_patterns = DEFAULT_STRIP_SUFFIXES, DEFAULT_STRIP_PATTERNS
        stripping_report = ", ".join((*DEFAULT_STRIP_SUFFIXES, STRIP_PATTERN_LABEL))

    # Parse --breaks and assemble the split dimensions.
    if breaks and not range_by:
        raise click.ClickException("--breaks was given without --range-by")
    if range_by and not breaks:
        raise click.ClickException("--range-by requires --breaks (e.g. --breaks 100,200)")
    range_dim = None
    if range_by:
        try:
            break_values = [float(b) for b in breaks.split(",") if b.strip() != ""]
        except ValueError:
            raise click.ClickException(f"--breaks must be numbers, got {breaks!r}")
        if not break_values:
            raise click.ClickException("--breaks is empty")
        range_dim = Range(range_by, break_values)

    try:
        blocks, part_key = read_star(input_path)
        particles = blocks[part_key]
        flavor = detect_flavor(blocks, particles)
        # Categorical dimensions: explicit --group-by, else auto-detect one column
        # (but if only --range-by is given, no categorical dimension is required).
        if group_by:
            dimensions = list(group_by)
        elif range_dim is not None:
            dimensions = []
        else:
            dimensions = resolve_group_columns(particles, None)
        if range_dim is not None:
            dimensions = [*dimensions, range_dim]

        plan = plan_split(particles, dimensions, label, outdir=outdir,
                          strip_suffixes=strip_suffixes,
                          strip_patterns=strip_patterns)
    except SplitStarError as exc:
        raise click.ClickException(str(exc))

    # Guard against two different combinations collapsing to the same output name
    # (values joined by '_' can be ambiguous), which would silently overwrite.
    from collections import Counter
    counts = Counter(item.output_path for item in plan)
    collisions = sorted(p for p, c in counts.items() if c > 1)
    if collisions:
        raise click.ClickException(
            "grouping produces colliding output names (would overwrite):\n  "
            + "\n  ".join(collisions)
            + "\nUse a different --group-by order/columns or split in one column "
              "per run.")

    spec = split_spec_string(dimensions)
    if not quiet:
        click.echo(f"input:           {input_path}")
        click.echo(f"detected flavor: {flavor}")
        click.echo(f"split by:        {spec} ({len(plan)} groups)")
        click.echo(f"name stripping:  {stripping_report}")
        click.echo(f"output root:     {outdir}")
        click.echo("")

    if dry_run:
        for item in plan:
            click.echo(f"  would write {item.n_rows} row(s) -> {item.output_path} "
                       f"[{'; '.join(item.descriptions)}]")
        click.echo(f"\ndry run: {len(plan)} file(s) not written")
        return

    total = 0
    for item in plan:
        header = (build_header_lines(input_path, spec, item.descriptions)
                  if comment else None)
        n = write_group(blocks, part_key, particles, item.mask,
                        item.output_path, header_lines=header)
        total += n
        if not quiet:
            click.echo(f"  {n} row(s) -> {item.output_path}")

    click.echo(f"wrote {len(plan)} file(s) covering {total} particle(s) "
               f"under {Path(outdir)}")


if __name__ == "__main__":
    split_star()
