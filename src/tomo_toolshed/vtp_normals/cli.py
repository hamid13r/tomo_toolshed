"""``tomo_toolshed vtp-normals`` -- preview, curate and flip pycurv .vtp normals.

Three subcommands share one piece numbering (see :mod:`.core`):

    vtp-normals preview   write <name>.normals.png previews of each piece's normals
    vtp-normals curate    interactive 3D curator; mark pieces and write .flipped.vtp
    vtp-normals flip      batch-flip listed pieces into <name>.flipped.vtp

``--split-column`` (default ``OMM_dist``), ``--split-dist`` and
``--min-triangles`` decide how the surface is cut into pieces and must be the
SAME for ``flip`` as for the ``preview`` / ``curate`` the piece ids were read
from, or the ids will not line up.

Only ``click`` and :mod:`.core` are imported at module load; the matplotlib
``preview`` / ``gui`` modules are imported lazily inside the subcommands so that
importing this module (and the top-level ``tomo_toolshed`` CLI) never pulls in a
plotting backend.
"""

import glob
import os
from collections import defaultdict

import click

from . import core

_SPLIT_HELP = ("must match the preview/curate the piece ids were read from, or "
               "the ids will not line up")


@click.group(name="vtp-normals")
def vtp_normals():
    """Split pycurv .vtp surfaces into pieces, preview and flip their normals.

    \b
    preview  Write <name>.normals.png previews of each surface piece's normals.
    curate   Interactive 3D curator: mark pieces to flip, write .flipped.vtp, review.
    flip     Batch-flip listed pieces (TS:MEMBRANE:IDS) into <name>.flipped.vtp.

    Pieces are the connected parts of each surface, split again where the
    --split-column array (default OMM_dist) crosses --split-dist, numbered
    largest first. Pass the same --split-column / --split-dist / --min-triangles
    to flip as to the preview/curate you read ids from.
    """


# --------------------------------------------------------------------- preview
@vtp_normals.command(name="preview")
@click.option("--vtp-glob", default="surfaces/*.vtp", show_default=True, help="Surfaces to preview.")
@click.option("--tomo-dir", default="tomo", show_default=True, type=click.Path(file_okay=False),
              help="Directory with the tomograms shown as background.")
@click.option("--tomo-pattern", default="{stem}.mrc_9.98Apx.mrc", show_default=True,
              help="Tomogram file name; {stem} is the vtp name up to _ts_NNN.")
@click.option("--normal-array", default="n_v", show_default=True,
              help="Cell array holding the normals (n_v = pycurv vector-voting normals).")
@click.option("--dist", type=float, default=150.0, show_default=True, help="Tip distance along the normal (Angstrom).")
@click.option("--apix", type=float, default=9.98, show_default=True, help="Tomogram pixel size (Angstrom).")
@click.option("--coord-scale", type=float, default=10.0, show_default=True,
              help="Multiply surface coordinates by this to get Angstrom (10 for nm surfaces).")
@click.option("--slab", type=float, default=1.5, show_default=True,
              help="Show surface points within this many voxels of the slice.")
@click.option("--split-column", default=core.DEFAULT_SPLIT_COLUMN, show_default=True,
              help=f"Cell array pieces are split on where it crosses --split-dist; {_SPLIT_HELP}.")
@click.option("--split-dist", type=float, default=core.SPLIT_DIST, show_default=True,
              help=f"Split pieces where --split-column crosses this; 0 = only connected pieces; {_SPLIT_HELP}.")
@click.option("--min-triangles", type=int, default=core.MIN_TRIANGLES, show_default=True,
              help=f"Skip pieces with fewer triangles than this; {_SPLIT_HELP}.")
@click.option("--line-every", type=int, default=10, show_default=True,
              help="Draw a membrane-to-tip line for every Nth point.")
@click.option("--bin", "bin_factor", type=click.IntRange(min=1), default=1, show_default=True,
              help="Mean-pool the tomogram by this factor in 3D before display; --apix stays the "
                   "unbinned pixel size (surface coords use apix*bin).")
@click.option("--gaussian", type=float, default=0.0, show_default=True,
              help="Gaussian blur sigma (Angstrom); applied in 2D to each panel's slab image "
                   "(converted to voxels with the effective pixel size). 0 = off.")
def preview(vtp_glob, tomo_dir, tomo_pattern, normal_array, dist, apix, coord_scale, slab, split_column,
            split_dist, min_triangles, line_every, bin_factor, gaussian):
    """Write <vtp>.normals.png previews of the normals of each surface piece."""
    from . import preview as preview_mod

    preview_mod.use_agg_backend()
    vtp_paths = sorted(glob.glob(vtp_glob))
    if not vtp_paths:
        raise click.ClickException(f"No files matched '{vtp_glob}'")

    for vtp_path in vtp_paths:
        preview_mod.preview_vtp(vtp_path, tomo_dir, tomo_pattern=tomo_pattern, normal_array=normal_array,
                                dist=dist, apix=apix, coord_scale=coord_scale, slab=slab,
                                min_triangles=min_triangles, line_every=line_every, split_dist=split_dist,
                                split_column=split_column, bin_factor=bin_factor, gaussian=gaussian)


# ------------------------------------------------------------------------ flip
@vtp_normals.command(name="flip")
@click.option("--flip", "flips", multiple=True, metavar="TS:MEMBRANE:IDS",
              help="Pieces to flip, e.g. lam12_ts_003:IMM:0,3. Repeatable.")
@click.option("--flip-file", type=click.Path(exists=True, dir_okay=False),
              help="Text file with one TS:MEMBRANE:IDS per line.")
@click.option("--vtp-dir", default="surfaces", show_default=True, type=click.Path(exists=True, file_okay=False),
              help="Directory with the original vtp files.")
@click.option("--arrays", default="n_v,normal,avg_normals", show_default=True,
              help="Comma-separated cell arrays to negate.")
@click.option("--curvature/--no-curvature", default=True, show_default=True,
              help="Also update the curvature arrays of flipped pieces.")
@click.option("--preview/--no-preview", default=True, show_default=True,
              help="Write a .flipped.normals.png preview of each output.")
@click.option("--tomo-dir", default="tomo", show_default=True, type=click.Path(file_okay=False),
              help="Tomograms for the preview background.")
@click.option("--split-column", default=core.DEFAULT_SPLIT_COLUMN, show_default=True,
              help=f"Cell array pieces are split on where it crosses --split-dist; {_SPLIT_HELP}.")
@click.option("--split-dist", type=float, default=core.SPLIT_DIST, show_default=True,
              help=f"Split pieces where --split-column crosses this; {_SPLIT_HELP}.")
@click.option("--min-triangles", type=int, default=core.MIN_TRIANGLES, show_default=True,
              help=f"Split parts smaller than this are merged into a neighbour; {_SPLIT_HELP}.")
def flip(flips, flip_file, vtp_dir, arrays, curvature, preview, tomo_dir, split_column,
         split_dist, min_triangles):
    """Write <vtp>.flipped.vtp with the normals of the listed pieces negated."""
    parsed = core.parse_flips(flips, flip_file)
    if not parsed:
        raise click.UsageError("nothing to flip: give --flip and/or --flip-file")
    per_file = defaultdict(set)
    for ts, membrane, piece_ids in parsed:
        per_file[core.find_vtp(vtp_dir, ts, membrane)] |= piece_ids
    array_names = [a.strip() for a in arrays.split(",") if a.strip()]

    preview_mod = None
    if preview:
        from . import preview as preview_mod
        preview_mod.use_agg_backend()

    for vtp_path, piece_ids in per_file.items():
        click.echo(f"\n=== {os.path.basename(vtp_path)} ===")
        out_path = core.flip_surface(vtp_path, piece_ids, array_names, curvature, split_dist, min_triangles,
                                     split_column)
        if preview_mod is not None:
            preview_mod.preview_vtp(out_path, tomo_dir=tomo_dir, split_dist=split_dist,
                                    min_triangles=min_triangles, split_column=split_column)


# ---------------------------------------------------------------------- curate
@vtp_normals.command(name="curate")
@click.option("--tomo", "tomo_file", type=click.Path(exists=True, dir_okay=False),
              help="Tomogram shown as background (overrides --tomo-dir/--tomo-pattern).")
@click.option("--vtp", type=click.Path(exists=True, dir_okay=False),
              help="Surface to curate (overrides --vtp-glob).")
@click.option("--vtp-glob", default="surfaces/*IMM*.vtp", show_default=True, help="Surfaces to go through.")
@click.option("--tomo-dir", default="tomo", show_default=True, type=click.Path(file_okay=False),
              help="Directory with the tomograms shown as background.")
@click.option("--tomo-pattern", default="{stem}.mrc_9.98Apx.mrc", show_default=True,
              help="Tomogram file name; {stem} is the vtp name up to _ts_NNN.")
@click.option("--skip-done/--no-skip-done", default=False, show_default=True,
              help="Skip surfaces that already have a .flipped.vtp.")
@click.option("--dist", type=float, default=150.0, show_default=True,
              help="Initial tip distance along the normal (Angstrom); adjustable in the viewer.")
@click.option("--apix", type=float, default=None,
              help="Tomogram pixel size (Angstrom). Default: from the MRC header of --tomo, else 9.98.")
@click.option("--coord-scale", type=float, default=10.0, show_default=True,
              help="Multiply surface coordinates by this to get Angstrom (10 for nm surfaces).")
@click.option("--split-column", default=core.DEFAULT_SPLIT_COLUMN, show_default=True,
              help=f"Cell array pieces are split on where it crosses --split-dist; {_SPLIT_HELP}.")
@click.option("--split-dist", type=float, default=core.SPLIT_DIST, show_default=True,
              help=f"Split pieces where --split-column crosses this; 0 = only connected pieces; {_SPLIT_HELP}.")
@click.option("--min-triangles", type=int, default=core.MIN_TRIANGLES, show_default=True,
              help=f"Split parts smaller than this are merged into a neighbour; isolated pieces this small "
                   f"are not listed (and never flipped); {_SPLIT_HELP}.")
@click.option("--arrow-every", type=int, default=10, show_default=True,
              help="Draw a normal arrow for every Nth surface point; adjustable in the viewer.")
@click.option("--bin", "bin_factor", type=click.IntRange(min=1), default=1, show_default=True,
              help="Mean-pool the tomogram by this factor in 3D before display; --apix stays the "
                   "unbinned pixel size (surface coords use apix*bin). slab and arrow width are in "
                   "binned display voxels.")
@click.option("--gaussian", type=float, default=0.0, show_default=True,
              help="Gaussian blur sigma (Angstrom); applied in 3D to the display volume after binning "
                   "(converted to voxels with the effective pixel size). 0 = off.")
def curate(tomo_file, vtp, vtp_glob, tomo_dir, tomo_pattern, skip_done, dist, apix, coord_scale, split_column,
           split_dist, min_triangles, arrow_every, bin_factor, gaussian):
    """Curate normal orientations piece by piece in 3D, then flip and review."""
    from . import gui

    gui.run_curation(tomo_file, vtp, vtp_glob, tomo_dir, tomo_pattern, skip_done, dist, apix, coord_scale,
                     split_dist, min_triangles, arrow_every, split_column, bin_factor=bin_factor,
                     gaussian=gaussian)


if __name__ == "__main__":
    vtp_normals()
