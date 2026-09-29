"""``tomo_toolshed filament-cleanup`` -- remove off-helix particles after helical refinement.

Groups particles into filaments (tomogram x ``rlnHelicalTubeID``), fits a smooth
curve through each filament's refined positions, and removes particles that sit
too far from it (perpendicular distance) or whose helical axis (tilt/psi)
disagrees with their neighbours. Writes the cleaned star file plus a second
star file holding only the removed particles, so both can be overlaid in
ChimeraX/ArtiaX. Geometry lives in ``core.py``; this layer wires options and
reporting.
"""

import os
from pathlib import Path

import click
import numpy as np

from ..split_star.core import _package_version, read_star, write_star_file
from .core import (
    REASON_FLIPPED,
    REASON_ORIENTATION,
    REASON_POSITION,
    FilamentCleanupError,
    clean_filaments,
    split_blocks,
)


def default_rejected_path(output_path):
    """``<output stem>_rejected<suffix>`` next to the cleaned output."""
    p = Path(output_path)
    return str(p.with_name(p.stem + "_rejected" + (p.suffix or ".star")))


def build_header_lines(source, what, params):
    version = _package_version()
    stamp = "tomo_toolshed filament-cleanup" + (f" (v{version})" if version else "")
    return [f"Created by {stamp}", f"source: {source}", f"this file: {what}",
            "parameters: " + ", ".join(f"{k}={v}" for k, v in params.items())]


def _pct(values):
    v = values[np.isfinite(values)]
    if not len(v):
        return "n/a"
    q = np.percentile(v, [50, 90, 99])
    return f"median {q[0]:.1f}, p90 {q[1]:.1f}, p99 {q[2]:.1f}, max {v.max():.1f}"


@click.command(name="filament-cleanup")
@click.option("--input", "--i", "input_path", required=True,
              type=click.Path(exists=True, dir_okay=False),
              help="Helical-refinement particle star file (RELION 3/4/5).")
@click.option("--output", "--o", "output_path", required=True,
              type=click.Path(dir_okay=False),
              help="Cleaned star file (parent directories are created).")
@click.option("--rejected", "rejected_path", default=None,
              type=click.Path(dir_okay=False),
              help="Star file of the removed particles. "
                   "[default: <output>_rejected.star]")
@click.option("--max-distance", "--max_distance", "max_distance", type=float,
              default=30.0, show_default=True,
              help="Max distance (Å) of a refined position from the smooth "
                   "filament curve, measured perpendicular to the curve.")
@click.option("--max-angle", "--max_angle", "max_angle", type=float,
              default=20.0, show_default=True,
              help="Max angle (deg) between a particle's helical axis "
                   "(from tilt/psi) and the mean axis of its neighbours.")
@click.option("--window", type=float, default=500.0, show_default=True,
              help="Smoothing window (Å along the filament) for the curve fit. "
                   "Larger = stiffer curve.")
@click.option("--neighbours", "--neighbors", "neighbours", type=int, default=5,
              show_default=True,
              help="Neighbours on each side used for the orientation check.")
@click.option("--min-particles", "--min_particles", "min_particles", type=int,
              default=5, show_default=True,
              help="Filaments with fewer particles are kept untouched.")
@click.option("--position/--no-position", "check_position", default=True,
              show_default=True, help="Run the position (smooth curve) check.")
@click.option("--orientation/--no-orientation", "check_orientation",
              default=True, show_default=True,
              help="Run the orientation (tilt/psi vs neighbours) check.")
@click.option("--remove-flipped", "--remove_flipped", "remove_flipped",
              is_flag=True, default=False,
              help="Also remove particles whose polarity (psi flipped by 180) "
                   "disagrees with the majority of their filament. Off by "
                   "default: the orientation check ignores polarity.")
@click.option("--pixel-size", "--pixel_size", "pixel_size", type=float,
              default=None,
              help="Override the coordinate pixel size (Å/px). By default it is "
                   "resolved per flavor, as in duplicate-remover.")
@click.option("--group-by", "--group_by", "group_by", default=None,
              help="Tomogram column [default: first of rlnTomoName, "
                   "rlnMicrographName].")
@click.option("--comment/--no-comment", default=True, show_default=True,
              help="Prepend a provenance comment to the output star files.")
@click.option("--dry-run", is_flag=True, default=False,
              help="Report what would be removed; write nothing.")
@click.option("--quiet", "-q", is_flag=True, default=False,
              help="Only print the final summary.")
def filament_cleanup(input_path, output_path, rejected_path, max_distance,
                     max_angle, window, neighbours, min_particles,
                     check_position, check_orientation, remove_flipped,
                     pixel_size, group_by, comment, dry_run, quiet):
    """Remove particles that don't belong to their helix (post helical refinement).

    Per filament (tomogram x rlnHelicalTubeID), a smooth curve is fitted through
    the refined positions (coordinate - rlnOrigin*Angst). A particle is removed if

    \b
      - it lies more than --max-distance Å from the curve (perpendicular;
        sliding along the axis is fine), or
      - its helical axis from rlnAngleTilt/rlnAnglePsi is more than --max-angle
        degrees off the mean axis of its --neighbours neighbours on each side
        (polarity ignored unless --remove-flipped).

    Writes the cleaned star file and a second star file with only the removed
    particles (same blocks and columns), to overlay both in ChimeraX/ArtiaX.

    \b
    example:
    tomo_toolshed filament-cleanup --i run_data.star --o run_data_clean.star
    tomo_toolshed filament-cleanup --i run_data.star --o clean.star --max-distance 20 --max-angle 15
    tomo_toolshed filament-cleanup --i run_data.star --o clean.star --no-orientation --dry-run
    """
    if not (check_position or check_orientation or remove_flipped):
        raise click.ClickException(
            "nothing to do: --no-position and --no-orientation without --remove-flipped")
    rejected_path = rejected_path or default_rejected_path(output_path)
    real = {os.path.realpath(p) for p in (output_path, rejected_path)}
    if os.path.realpath(input_path) in real:
        raise click.ClickException("--output/--rejected must differ from --input")
    if len(real) < 2:
        raise click.ClickException("--rejected must differ from --output")

    try:
        blocks, part_key = read_star(input_path)
        result = clean_filaments(
            blocks, part_key, max_distance=max_distance, max_angle=max_angle,
            window=window, neighbours=neighbours, min_particles=min_particles,
            check_position=check_position, check_orientation=check_orientation,
            remove_flipped=remove_flipped, pixel_size=pixel_size,
            group_by=group_by)
    except FilamentCleanupError as exc:
        raise click.ClickException(str(exc))

    n = len(result.keep_mask)
    info = result.info
    n_fil = len(result.filaments)
    n_skip = sum(f["skipped"] for f in result.filaments)
    if not quiet:
        for w in info["warnings"]:
            click.echo(f"warning: {w}")
        click.echo(f"input:            {input_path}")
        click.echo(f"detected flavor:  {info['flavor']}")
        click.echo(f"grouping:         {result.group_column} x rlnHelicalTubeID "
                   f"({n_fil} filaments, {n_skip} too short to check)")
        click.echo(f"coordinate cols:  {', '.join(info['coord_cols'])}")
        click.echo(f"pixel size:       {info['pixel']}")
        click.echo(f"origins:          {info['origins']}")
        if check_position:
            click.echo(f"position check:   > {max_distance:g} Å from curve "
                       f"(window {window:g} Å) -> {result.count(REASON_POSITION)} "
                       "flagged")
            click.echo(f"  distance (Å):   {_pct(result.distance)}")
        if check_orientation:
            click.echo(f"orient. check:    > {max_angle:g}° from {neighbours}+"
                       f"{neighbours} neighbours -> "
                       f"{result.count(REASON_ORIENTATION)} flagged")
            click.echo(f"  angle (deg):    {_pct(result.angle)}")
        if remove_flipped:
            click.echo(f"polarity check:   minority polarity -> "
                       f"{result.count(REASON_FLIPPED)} flagged")
        worst = sorted((f for f in result.filaments if not f["skipped"]),
                       key=lambda f: -(f["position"] + f["orientation"]
                                       + f["flipped"]) / f["n"])[:5]
        if worst and result.n_removed:
            click.echo("most affected filaments (tomogram, tube: flagged / n):")
            for f in worst:
                flagged = f["position"] + f["orientation"] + f["flipped"]
                if flagged:
                    click.echo(f"  {f['group']}, {f['tube']}: "
                               f"pos {f['position']}, orient {f['orientation']}"
                               + (f", flip {f['flipped']}" if remove_flipped else "")
                               + f" / {f['n']}")
        click.echo(f"output:           {output_path}")
        click.echo(f"rejected:         {rejected_path}")
        click.echo("")

    kept = n - result.n_removed
    pct = 100.0 * result.n_removed / n if n else 0.0
    if dry_run:
        click.echo(f"dry run: would keep {kept} / {n}, remove {result.n_removed} "
                   f"({pct:.1f}%) (nothing written)")
        return

    params = {"max_distance": max_distance, "max_angle": max_angle,
              "window": window, "neighbours": neighbours,
              "min_particles": min_particles, "position": check_position,
              "orientation": check_orientation, "remove_flipped": remove_flipped}
    kept_blocks, rejected_blocks = split_blocks(blocks, part_key, result.keep_mask)
    write_star_file(kept_blocks, output_path, header_lines=build_header_lines(
        input_path, f"kept particles ({kept})", params) if comment else None)
    write_star_file(rejected_blocks, rejected_path, header_lines=build_header_lines(
        input_path, f"removed particles ({result.n_removed})", params)
        if comment else None)
    click.echo(f"kept {kept} / {n}, removed {result.n_removed} ({pct:.1f}%) "
               f"-> {output_path}, {rejected_path}")


if __name__ == "__main__":
    filament_cleanup()
