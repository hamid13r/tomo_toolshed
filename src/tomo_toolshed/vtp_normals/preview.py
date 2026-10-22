"""Static PNG previews of the normals of each surface piece.

For every ``.vtp`` the surface is split into pieces (see :mod:`.core`) and one
PNG is written next to it (``<name>.normals.png``), one panel per piece. In each
panel the tomogram slice at the piece's median z is shown with:
  * cyan:   triangle centres (``xyz`` cell array) within +-``--slab`` voxels of that z
  * red:    the same points moved ``--dist`` Angstrom along the normal array
  * yellow: thin lines joining a subset of points to their tips
If the normals are consistent, all red tips sit on the same side of the membrane.

Coordinates: surface xyz * ``--coord-scale`` (Angstrom) / ``--apix`` = MRC voxel
index (x, y, z).

This module imports matplotlib with the Agg backend; it is imported lazily by
the CLI so ``tomo_toolshed`` stays headless-import-safe.
"""

import math
import os

import click
import matplotlib
import matplotlib.collections
import matplotlib.pyplot as plt
import mrcfile
import numpy as np

from . import core, volume
from .core import MIN_TRIANGLES, SPLIT_DIST, DEFAULT_SPLIT_COLUMN


def use_agg_backend():
    """Switch matplotlib to the non-interactive Agg backend (for PNG output)."""
    plt.switch_backend("Agg")


def preview_png_path(vtp_path):
    return os.path.splitext(vtp_path)[0] + ".normals.png"


def plot_piece(ax, tomo, z_index, pts_vox, tips_vox, slab, line_every, title, gaussian_vox=0.0):
    nz, ny, nx = tomo.shape if tomo is not None else (None, None, None)
    if tomo is not None:
        z_lo, z_hi = max(z_index - 1, 0), min(z_index + 2, nz)
        img = np.asarray(tomo[z_lo:z_hi], dtype=np.float32).mean(axis=0)
        if gaussian_vox > 0:
            from scipy.ndimage import gaussian_filter
            img = gaussian_filter(img, gaussian_vox)
        lo, hi = np.percentile(img, [1, 99])
        ax.imshow(img, cmap="gray", vmin=lo, vmax=hi, origin="lower", interpolation="nearest")
    in_slab = np.abs(pts_vox[:, 2] - z_index) <= slab
    p, t = pts_vox[in_slab], tips_vox[in_slab]
    if len(p):
        idx = np.arange(0, len(p), line_every)
        segs = np.stack([p[idx, :2], t[idx, :2]], axis=1)
        ax.add_collection(matplotlib.collections.LineCollection(segs, colors="yellow", linewidths=0.6, alpha=0.8))
        ax.scatter(p[:, 0], p[:, 1], s=3, c="cyan", linewidths=0, label="membrane")
        ax.scatter(t[:, 0], t[:, 1], s=3, c="red", linewidths=0, label="tip")
    # zoom to the piece (+ tips) with a margin, so the tip side is visible
    xy = np.concatenate([p[:, :2], t[:, :2]]) if len(p) else pts_vox[:, :2]
    (x0, y0), (x1, y1) = xy.min(axis=0) - 20, xy.max(axis=0) + 20
    if tomo is not None:
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, nx), min(y1, ny)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.set_title(f"{title}\nz = {z_index}, {in_slab.sum()} pts in slab", fontsize=8)
    ax.tick_params(labelsize=6)


def make_preview_figure(vtp_path, tomo_dir="tomo", tomo_pattern="{stem}.mrc_9.98Apx.mrc", normal_array="n_v",
                        dist=150.0, apix=9.98, coord_scale=10.0, slab=1.5, min_triangles=MIN_TRIANGLES,
                        line_every=10, split_dist=SPLIT_DIST, split_column=DEFAULT_SPLIT_COLUMN, tomo_path=None,
                        bin_factor=1, gaussian=0.0):
    """Build the preview figure; return (fig, {piece id: axes}) for pieces with >= min_triangles.

    ``tomo_path`` overrides the tomogram found from tomo_dir/tomo_pattern. With
    ``bin_factor`` > 1 the tomogram is mean-pooled before display and the effective
    pixel size ``apix * bin_factor`` is used for the surface-to-voxel mapping;
    ``--apix`` stays the unbinned pixel size. ``gaussian`` (sigma in Angstrom) is
    applied in 2D to each panel's slab-averaged image, using the effective pixel size.
    """
    click.echo(f"\n=== {os.path.basename(vtp_path)} ===")
    poly = core.read_polydata(vtp_path)
    triangles, centres, normals = core.load_surface(vtp_path, normal_array, poly=poly)
    labels, sizes = core.surface_pieces(poly, triangles, split_dist, min_triangles, split_column)
    split_median = core.piece_split_median(poly, labels, len(sizes), split_column)
    keep = np.flatnonzero(sizes >= min_triangles)
    click.echo(f"  {len(triangles)} triangles, {len(sizes)} piece(s), "
               f"{len(keep)} with >= {min_triangles} triangles")

    apix_eff = apix * bin_factor
    pts_vox = centres * coord_scale / apix_eff
    tips_vox = (centres * coord_scale + normals * dist) / apix_eff
    gaussian_vox = volume.gaussian_sigma_voxels(gaussian, apix_eff)

    if tomo_path is None:
        tomo_path = core.find_tomogram(vtp_path, tomo_dir, tomo_pattern)
    if tomo_path is None:
        click.echo("  WARNING: tomogram not found, plotting without background")
    # bin>1: read the pooled volume into memory (memory-light, Z-slab chunks); bin==1:
    # keep the mmap so a full tomogram is never loaded whole.
    mrc = tomo = None
    if tomo_path and bin_factor > 1:
        tomo = volume.bin_volume(tomo_path, bin_factor)
    elif tomo_path:
        mrc = mrcfile.mmap(tomo_path, permissive=True)
        tomo = mrc.data

    n_cols = min(len(keep), 3)
    n_rows = math.ceil(len(keep) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 6 * n_rows), squeeze=False)
    for ax in axes.ravel()[len(keep):]:
        ax.axis("off")
    for ax, piece in zip(axes.ravel(), keep):
        sel = labels == piece
        z_index = int(round(np.median(pts_vox[sel, 2])))
        if tomo is not None:
            z_index = min(max(z_index, 0), tomo.shape[0] - 1)
        plot_piece(ax, tomo, z_index, pts_vox[sel], tips_vox[sel], slab, line_every,
                   f"piece {piece} ({sizes[piece]} triangles, {split_column} {split_median[piece]:.0f})",
                   gaussian_vox=gaussian_vox)
        click.echo(f"  piece {piece}: {sizes[piece]} triangles, median {split_column} {split_median[piece]:.1f}, "
                   f"central z = {z_index}")
    if mrc:
        mrc.close()

    handles, labels_txt = axes.ravel()[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels_txt, loc="upper right", markerscale=6, fontsize=9)
    fig.suptitle(f"{os.path.basename(vtp_path)}\n'{normal_array}' normals, tips at {dist:g} A",
                 fontsize=10)
    fig.tight_layout()
    return fig, dict(zip(keep.tolist(), axes.ravel()))


def preview_vtp(vtp_path, tomo_dir="tomo", **kwargs):
    """Write <vtp>.normals.png with one panel per connected piece of the surface."""
    fig, _ = make_preview_figure(vtp_path, tomo_dir=tomo_dir, **kwargs)
    out_path = preview_png_path(vtp_path)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    click.echo(f"  saved {out_path}")
