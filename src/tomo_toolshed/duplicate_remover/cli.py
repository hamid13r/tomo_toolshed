"""``tomo_toolshed duplicate-remover`` -- drop particles closer than a threshold.

Within each group (tomogram / micrograph / source) any two particles closer than
``--threshold`` are treated as duplicates and the worse of the pair is removed,
repeated until none remain. The geometry (star flavor, pixel size, origins) is
resolved in ``core.py``; this layer handles options, the provenance report, the
optional histogram (matplotlib imported lazily so the package stays
headless-import-safe), and writing the result.
"""

from pathlib import Path

import click

from .core import (
    DuplicateRemoverError,
    apply_keep,
    process,
    read_star,
)


def _default_output(star_path):
    """``<name>_cleaned<suffix>`` via pathlib, so paths containing ``.star``
    elsewhere are not corrupted (the old ``str.replace('.star', ...)`` bug)."""
    p = Path(star_path)
    return str(p.with_name(p.stem + "_cleaned" + p.suffix))


def _format_provenance(result, star_path, output_path, threshold, units):
    """Build the multi-line provenance block printed before results."""
    px = result.pixel
    if px.uniform:
        px_str = f"{px.value} Å/px"
    else:
        px_str = f"varies per optics group {px.unique.tolist()} Å/px"

    thr_parts = []
    if result.threshold_angstrom is not None:
        thr_parts.append(f"{result.threshold_angstrom:g} Å")
    if result.threshold_pixel is not None:
        thr_parts.append(f"{result.threshold_pixel:g} px")
    thr_str = " = ".join(thr_parts) if thr_parts else f"{threshold:g} {units}"

    lines = [
        f"input:            {star_path}",
        f"output:           {output_path}",
        f"detected flavor:  {result.flavor}",
        f"grouping column:  {result.group_column} ({len(result.group_stats)} groups)",
        f"coordinate cols:  {', '.join(result.coord_cols)}",
        f"pixel size:       {px_str}  [{px.reason}]",
        f"origins:          {result.origin_reason}",
        f"comparison metric:{result.metric}  [{result.metric_reason}]",
        f"threshold:        {thr_str}  (given as {threshold:g} {units})",
    ]
    return "\n".join(lines)


def _plot_histogram(distances, threshold_angstrom, histogram_path):
    """Save a histogram of removed-particle distances (Agg, no display)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    plt.hist(distances, bins=50, edgecolor="black", alpha=0.7)
    plt.xlabel("Distance to nearest kept neighbour (Å)", fontsize=12)
    plt.ylabel("Number of removed particles", fontsize=12)
    plt.title(f"Removed-particle distances\n"
              f"(threshold {threshold_angstrom:g} Å, {len(distances)} removed)",
              fontsize=14)
    plt.axvline(x=threshold_angstrom, color="red", linestyle="--", linewidth=2,
                label=f"threshold ({threshold_angstrom:g} Å)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(histogram_path, dpi=300, bbox_inches="tight")
    plt.close()


@click.command(name="duplicate-remover")
@click.option("--star-path", "--star_path", "star_path", required=True,
              type=click.Path(exists=True, dir_okay=False),
              help="Input star file of particles.")
@click.option("--output-path", "--output_path", "output_path", default=None,
              type=click.Path(dir_okay=False),
              help="Output star file. Default: input name + _cleaned.")
@click.option("--threshold", "--distance_threshold", "threshold",
              type=float, default=140, show_default=True,
              help="Distance threshold below which particles are duplicates.")
@click.option("--units", type=click.Choice(["angstroms", "pixels"]),
              default="angstroms", show_default=True,
              help="Units of --threshold. 'pixels' multiplies it by the resolved "
                   "pixel size; internal math stays in Å.")
@click.option("--pixel-size", "--pixel_size", "pixel_size", type=float, default=None,
              help="Override the pixel size (Å/px) resolved from the file.")
@click.option("--group-by", "--group_by", "group_by", default=None,
              help="Override the auto-detected grouping column.")
@click.option("--comparison-metric", "--comparison_metric", "comparison_metric",
              type=click.Choice(["auto", "rlnLogLikeliContribution",
                                 "rlnMaxValueProbDistribution", "random", "first"]),
              default="auto", show_default=True,
              help="How to choose which particle of a too-close pair to keep.")
@click.option("--seed", type=int, default=None,
              help="Seed for --comparison-metric random (reproducible runs).")
@click.option("--histogram/--no-histogram", default=True, show_default=True,
              help="Save a histogram of removed-particle distances.")
@click.option("--dry-run", is_flag=True, default=False,
              help="Report what would be removed; write nothing.")
@click.option("--quiet", "-q", is_flag=True, default=False,
              help="Only print the final summary.")
def duplicate_remover(star_path, output_path, threshold, units, pixel_size,
                      group_by, comparison_metric, seed, histogram, dry_run, quiet):
    """Remove particles closer than a threshold within each group.

    The pixel size of the coordinate columns is resolved per star flavor
    (RELION 3/4/5 and M/WarpTools); run without --quiet to see which scale was
    used. Output preserves the input row order and every input block.

    \b
    example:
    tomo_toolshed duplicate-remover --star-path run_data.star --threshold 140
    tomo_toolshed duplicate-remover --star-path picks.star --units pixels --threshold 30
    """
    if output_path is None:
        output_path = _default_output(star_path)

    try:
        blocks, part_key = read_star(star_path)
        result = process(
            blocks, part_key, threshold=threshold, units=units,
            pixel_size=pixel_size, group_by=group_by,
            comparison_metric=comparison_metric, seed=seed)
    except DuplicateRemoverError as exc:
        raise click.ClickException(str(exc))

    if not quiet:
        click.echo(_format_provenance(result, star_path, output_path,
                                      threshold, units))
        click.echo("")

    if result.pixel.ignored_override and pixel_size is not None:
        click.secho("warning: --pixel-size is ignored for M/WarpTools files "
                    "(coordinates are already in Å).", fg="yellow", err=True)
    for warning in result.warnings:
        click.secho(f"warning: {warning}", fg="yellow", err=True)

    if not quiet:
        for name, initial, removed in result.group_stats:
            pct = (removed / initial * 100) if initial else 0.0
            click.echo(f"  {name}: {initial} -> {initial - removed} "
                       f"(removed {removed}, {pct:.1f}%)")
        click.echo("")

    pct_total = (result.n_removed / result.n_before * 100) if result.n_before else 0.0

    if dry_run:
        click.echo(f"dry run: would remove {result.n_removed} of {result.n_before} "
                   f"particles ({pct_total:.1f}%); {output_path} not written")
        return

    kept = apply_keep(blocks, part_key, result.keep_mask)
    from .core import write_star
    write_star(blocks, part_key, kept, output_path)
    click.echo(f"removed {result.n_removed} of {result.n_before} particles "
               f"({pct_total:.1f}%); wrote {result.n_after} to {output_path}")

    if histogram and result.removed_distances:
        thr_ang = result.threshold_angstrom
        if thr_ang is None:
            thr_ang = float(threshold)
        hp = Path(output_path)
        histogram_path = str(hp.with_name(hp.stem + "_removed_distances_histogram.png"))
        _plot_histogram(result.removed_distances, thr_ang, histogram_path)
        click.echo(f"histogram saved to {histogram_path}")

        d = result.removed_distances
        if not quiet:
            import numpy as np
            click.echo(f"removed-distance stats (Å): mean {np.mean(d):.2f}, "
                       f"median {np.median(d):.2f}, min {np.min(d):.2f}, "
                       f"max {np.max(d):.2f}, std {np.std(d):.2f}")
    elif histogram and not quiet:
        click.echo("no particles removed; no histogram generated.")


if __name__ == "__main__":
    duplicate_remover()
