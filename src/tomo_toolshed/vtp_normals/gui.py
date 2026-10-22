"""Interactive 3D curation of membrane normals, then flip and review.

A viewer in the style of ``tomo_toolshed curate``: three orthogonal slices
through the tomogram (Z view = Y-X plane, Y view = Z-X plane, X view = Z-Y
plane, all at true aspect so normals are not distorted) with sliders, and a
table of the surface pieces on the right (numbered as in :mod:`.core`). Pieces
are the connected parts of the surface, split where the ``--split-column`` array
crosses ``--split-dist`` (default ``OMM_dist`` at 20 nm), so the boundary IMM
near the OMM and the cristae far from it can be flipped separately; the table
shows each piece's median split-column value.

In every view, every ``--arrow-every``-th membrane point within +-slab voxels of
the slice is drawn as a cyan dot (yellow for the highlighted piece) with an
arrow ``--dist`` Angstrom along its normal (projected onto the view plane):
green for pieces kept as they are, red for pieces marked to flip. The arrows
follow the flip state live.

Every view opens on the whole tomogram and never auto-zooms; use the matplotlib
toolbar to zoom and pan, and the Full view button to reset. Highlighting a piece
or moving the slices keeps the current zoom.

Controls
  table, click a row         highlight the piece and move all slices onto it
  table, click the flip cell toggle keep / FLIP for that piece
  click a membrane in a view highlight that piece (does not toggle)
  click elsewhere in a view  move the crosshair (the other two slices) there
  scroll in a view / table   step that view's slice / scroll the table
  up / down, Prev / Next     highlight the previous / next piece
  t, Toggle flip             toggle the highlighted piece
  3D view                    rotatable 3D plot of the highlighted piece's normals
  Full view                  reset all views to the whole tomogram
  q, Done (or close)         flip the FLIP pieces into <name>.flipped.vtp
  x, Stop                    stop the session without writing this surface

After Done, the written .flipped.vtp is opened in the same viewer for review
(pieces flipped in the file are marked * in the table). Toggling pieces there
and pressing Done rewrites the file from the original with those corrections;
Done without changes moves on to the next surface. A
<name>.flipped.normals.png preview is saved with every write.

If <name>.flipped.vtp already exists, its flips are loaded as the starting
state, so a surface can be revisited. Nothing is written for a surface with no
flipped pieces unless a .flipped.vtp already exists (it is then rewritten).

Needs a display (e.g. ssh -X, or a desktop/VNC session).

This module imports matplotlib with an interactive backend; it is imported
lazily by the ``curate`` subcommand only.
"""

import glob
import os

import click
import matplotlib.pyplot as plt
import mrcfile
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button, Slider
from vtk.util.numpy_support import vtk_to_numpy

from . import core, preview, volume
from .core import MIN_TRIANGLES, SPLIT_DIST, DEFAULT_SPLIT_COLUMN

KEEP_COLOR = "limegreen"
FLIP_COLOR = "red"
MEMBRANE_COLOR = "cyan"
HIGHLIGHT_COLOR = "yellow"
MAX_POINTS_PER_VIEW = 20000
ARROW_WIDTH_VOX = 1.0  # shaft width in voxels (head is 3.5x wider), so arrows scale with zoom
TABLE_ROWS = 26


class Surface:
    """Triangle centres (voxels, xyz), unit normals and piece labels of one vtp."""

    def __init__(self, vtp_path, apix, coord_scale, min_triangles=MIN_TRIANGLES, split_dist=SPLIT_DIST,
                 split_column=DEFAULT_SPLIT_COLUMN, seed=0):
        poly = core.read_polydata(vtp_path)
        triangles, centres, normals = core.load_surface(vtp_path, "n_v", poly=poly)
        labels, self.sizes = core.surface_pieces(poly, triangles, split_dist, min_triangles, split_column)
        self.split_median = core.piece_split_median(poly, labels, len(self.sizes), split_column)
        flipped_array = poly.GetCellData().GetArray("normal_flipped")
        flipped = (vtk_to_numpy(flipped_array).astype(bool) if flipped_array is not None
                   else np.zeros(len(labels), bool))
        self.pieces = np.flatnonzero(self.sizes >= min_triangles)
        # pieces mostly flipped in this file (only meaningful for .flipped.vtp files)
        flipped_fraction = np.bincount(labels, weights=flipped, minlength=len(self.sizes)) / self.sizes
        self.flipped_in_file = {int(p) for p in self.pieces if flipped_fraction[p] > 0.5}

        keep = np.isin(labels, self.pieces)
        order = np.random.default_rng(seed).permutation(np.count_nonzero(keep))
        # shuffled once, so "first K points of a slab" is a uniform subsample
        self.points = (centres[keep] * coord_scale / apix)[order]
        self.normals = normals[keep][order]
        self.labels = labels[keep][order]
        self.centre_point = {}
        self.z_range = {}
        for piece in self.pieces:
            pts = self.points[self.labels == piece]
            median = np.median(pts, axis=0)
            self.centre_point[piece] = pts[np.argmin(np.linalg.norm(pts - median, axis=1))]
            self.z_range[piece] = (int(pts[:, 2].min()), int(np.ceil(pts[:, 2].max())))


class NormalCurator:
    """Orthoslice viewer + flip table; ``run()`` returns (flip set, stop requested)."""

    # (view name, horizontal point axis, vertical point axis, slice axis); point axes are x=0 y=1 z=2
    VIEWS = (("z", 0, 1, 2), ("y", 0, 2, 1), ("x", 1, 2, 0))

    def __init__(self, tomo, surface, flip, title, dist_vox, slab=1.5, arrow_every=10, review=False,
                 split_column=DEFAULT_SPLIT_COLUMN, apix_binned=None):
        self.tomo = tomo
        self.surf = surface
        self.flip = set(int(p) for p in flip)
        self.dist_vox = dist_vox
        self.slab = slab
        self.arrow_every = arrow_every
        self.review = review
        self.split_column = split_column
        self.apix_binned = apix_binned  # binned pixel size (A), for the blur slider's A->voxel
        self.blur_vox = 0.0  # extra 2D Gaussian on the displayed slices only (voxels)
        self.stop = False
        self.highlight = int(surface.pieces[0]) if len(surface.pieces) else None
        self.table_offset = 0
        self._suspend = False
        if tomo is not None:
            self.shape = tomo.shape  # (nz, ny, nx)
        else:
            self.shape = tuple(int(np.ceil(surface.points[:, a].max())) + 1 for a in (2, 1, 0))
        self.pos = np.array([s // 2 for s in self.shape[::-1]], dtype=int)  # x, y, z
        if self.highlight is not None:
            self.pos = np.round(surface.centre_point[self.highlight]).astype(int)
        self.pos = np.clip(self.pos, 0, np.array(self.shape[::-1]) - 1)
        self.vmin, self.vmax = 0, 255
        self._build(title)
        self.refresh()

    # ------------------------------------------------------------------ setup
    def _build(self, title):
        self.fig = plt.figure(figsize=(16, 9.5))
        self.fig.canvas.manager.set_window_title("normal curator")
        self.fig.suptitle(title, fontsize=10)
        nz, ny, nx = self.shape
        self.ax = {
            "z": self.fig.add_axes([0.03, 0.27, 0.36, 0.64]),
            "y": self.fig.add_axes([0.41, 0.60, 0.33, 0.31]),
            "x": self.fig.add_axes([0.41, 0.27, 0.33, 0.31]),
        }
        self.ax["z"].set_title("Z view (Y-X plane)", fontsize=9)
        self.ax["y"].set_title("Y view (Z-X plane)", fontsize=9)
        self.ax["x"].set_title("X view (Z-Y plane)", fontsize=9)
        self.img, self.mem, self.arrows, self.cross = {}, {}, {}, {}
        for name, h, v, _ in self.VIEWS:
            ax = self.ax[name]
            ax.tick_params(labelsize=6)
            extent = (-0.5, (nx, ny, nz)[h] - 0.5, -0.5, (nx, ny, nz)[v] - 0.5)
            self.img[name] = ax.imshow(np.zeros(((nx, ny, nz)[v], (nx, ny, nz)[h]), np.uint8), cmap="gray",
                                       vmin=0, vmax=255, origin="lower", extent=extent,
                                       interpolation="nearest")
            self.mem[name] = ax.scatter([], [], s=10, linewidths=0.4, edgecolors="black", zorder=4)
            self.arrows[name] = None
            self.cross[name] = (ax.axvline(0, color="orange", lw=0.5, alpha=0.6),
                                ax.axhline(0, color="orange", lw=0.5, alpha=0.6))
            ax.set_xlim(extent[0], extent[1])
            ax.set_ylim(extent[2], extent[3])
            ax.set_aspect("equal")
            ax.set_autoscale_on(False)  # slice/arrow redraws never change the user's zoom

        # table
        self.ax_tab = self.fig.add_axes([0.765, 0.27, 0.225, 0.64])
        self.ax_tab.set_xticks([])
        self.ax_tab.set_yticks([])
        self._tab_artists = []

        # sliders
        def slider(rect, label, vmin, vmax, init, step):
            s = Slider(self.fig.add_axes(rect), label, vmin, vmax, valinit=init, valstep=step)
            s.label.set_fontsize(8)
            s.valtext.set_fontsize(8)
            return s
        x, y, z = self.pos
        self.s_pos = {
            "z": slider([0.06, 0.205, 0.30, 0.02], "Z", 0, nz - 1, z, 1),
            "y": slider([0.06, 0.175, 0.30, 0.02], "Y", 0, ny - 1, y, 1),
            "x": slider([0.06, 0.145, 0.30, 0.02], "X", 0, nx - 1, x, 1),
        }
        for axis, s in zip((2, 1, 0), (self.s_pos["z"], self.s_pos["y"], self.s_pos["x"])):
            s.on_changed(lambda val, axis=axis: self._on_pos_slider(axis, val))
        self.s_cmin = slider([0.47, 0.205, 0.22, 0.02], "contrast min", 0, 255, 0, 1)
        self.s_cmax = slider([0.47, 0.175, 0.22, 0.02], "contrast max", 0, 255, 255, 1)
        self.s_slab = slider([0.47, 0.145, 0.22, 0.02], "slab (vox)", 0.5, 6, self.slab, 0.5)
        self.s_dist = slider([0.47, 0.115, 0.22, 0.02], "tip dist (vox)", 2, 40,
                             round(self.dist_vox), 1)
        self.s_cmin.on_changed(self._on_contrast)
        self.s_cmax.on_changed(self._on_contrast)
        self.s_slab.on_changed(self._on_display_slider)
        self.s_every = slider([0.47, 0.085, 0.22, 0.02], "arrow every", 1, 50, self.arrow_every, 1)
        self.s_dist.on_changed(self._on_display_slider)
        self.s_every.on_changed(self._on_display_slider)
        # extra live blur of the displayed slices only (sigma in A); 0 = off
        self.s_blur = slider([0.47, 0.055, 0.22, 0.02], "blur (A)", 0, 50, 0, 1)
        self.s_blur.on_changed(self._on_display_slider)

        # buttons
        def button(rect, label, callback):
            b = Button(self.fig.add_axes(rect), label)
            b.label.set_fontsize(8)
            b.on_clicked(callback)
            return b
        self.buttons = [
            button([0.77, 0.205, 0.05, 0.035], "◀ Prev", lambda e: self.step_highlight(-1)),
            button([0.825, 0.205, 0.05, 0.035], "Next ▶", lambda e: self.step_highlight(1)),
            button([0.88, 0.205, 0.105, 0.035], "Toggle flip (t)", lambda e: self.toggle(self.highlight)),
            button([0.77, 0.16, 0.105, 0.035], "3D view", lambda e: self.show_3d()),
            button([0.88, 0.16, 0.105, 0.035], "Full view", lambda e: self.full_view()),
            button([0.77, 0.115, 0.105, 0.035], "Done (q)", lambda e: plt.close(self.fig)),
            button([0.88, 0.115, 0.105, 0.035], "Stop (x)", lambda e: self._stop()),
        ]
        self.status = self.fig.text(0.03, 0.06, "", fontsize=9, family="monospace", va="top")

        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("scroll_event", self._on_scroll)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

    # ------------------------------------------------------------- rendering
    def refresh(self):
        if self._suspend:
            return
        x, y, z = self.pos
        nz, ny, nx = self.shape
        if self.tomo is not None:
            def thick(lo, hi, axis):  # 3-slice average along one axis, optionally blurred in 2D
                index = [slice(None)] * 3
                index[axis] = slice(max(lo - 1, 0), hi + 2)
                img = self.tomo[tuple(index)].mean(axis=axis)
                if self.blur_vox > 0:
                    from scipy.ndimage import gaussian_filter
                    img = gaussian_filter(img.astype(np.float32), self.blur_vox)
                return img
            self.img["z"].set_data(thick(z, z, 0))
            self.img["y"].set_data(thick(y, y, 1))
            self.img["x"].set_data(thick(x, x, 2))
        sign = np.where(np.isin(self.surf.labels, sorted(self.flip)), -1.0, 1.0)
        for name, h, v, s in self.VIEWS:
            idx = np.flatnonzero(np.abs(self.surf.points[:, s] - self.pos[s]) <= self.slab)
            idx = self._view_indices(s)
            pts = self.surf.points[idx]
            vec = self.surf.normals[idx] * (sign[idx, None] * self.dist_vox)
            labels = self.surf.labels[idx]
            mem_col = np.where((labels == self.highlight)[:, None],
                               plt.matplotlib.colors.to_rgba(HIGHLIGHT_COLOR),
                               plt.matplotlib.colors.to_rgba(MEMBRANE_COLOR))
            arrow_col = np.where((sign[idx] < 0)[:, None], plt.matplotlib.colors.to_rgba(FLIP_COLOR),
                                 plt.matplotlib.colors.to_rgba(KEEP_COLOR))
            self.mem[name].set_offsets(pts[:, [h, v]] if len(idx) else np.empty((0, 2)))
            self.mem[name].set_facecolors(mem_col)
            if self.arrows[name] is not None:
                self.arrows[name].remove()
                self.arrows[name] = None
            if len(idx):
                # angles/scale in data units, so arrow length = projected tip distance in voxels
                self.arrows[name] = self.ax[name].quiver(
                    pts[:, h], pts[:, v], vec[:, h], vec[:, v], color=arrow_col, angles="xy",
                    scale_units="xy", scale=1, units="xy", width=ARROW_WIDTH_VOX, headwidth=3.5,
                    headlength=4, headaxislength=3.5, minshaft=1.5, linewidth=0, zorder=3)
            vline, hline = self.cross[name]
            vline.set_xdata([self.pos[h]] * 2)
            hline.set_ydata([self.pos[v]] * 2)
        self._draw_table()
        self._draw_status()
        self.fig.canvas.draw_idle()

    def _view_indices(self, axis):
        """Points within the slab around the current slice along axis, every arrow_every-th."""
        idx = np.flatnonzero(np.abs(self.surf.points[:, axis] - self.pos[axis]) <= self.slab)
        return idx[::self.arrow_every][:MAX_POINTS_PER_VIEW]

    def _draw_table(self):
        for art in self._tab_artists:
            art.remove()
        self._tab_artists = []
        ax = self.ax_tab
        ax.set_xlim(0, 1)
        ax.set_ylim(TABLE_ROWS + 1, -0.4)
        add = self._tab_artists.append
        for xpos, text in ((0.03, "piece"), (0.17, "triangles"), (0.39, "z range"), (0.58, self.split_column),
                           (0.80, "flip")):
            add(ax.text(xpos, 0.05, text, fontsize=8, weight="bold", va="center"))
        pieces = self.surf.pieces[self.table_offset:self.table_offset + TABLE_ROWS]
        for row, piece in enumerate(pieces, start=1):
            piece = int(piece)
            if piece == self.highlight:
                add(ax.add_patch(Rectangle((0, row - 0.5), 1, 1, color=HIGHLIGHT_COLOR, alpha=0.35, lw=0)))
            star = "*" if self.review and piece in self.surf.flipped_in_file else ""
            z0, z1 = self.surf.z_range[piece]
            add(ax.text(0.03, row, f"{piece}{star}", fontsize=8, va="center"))
            add(ax.text(0.17, row, f"{self.surf.sizes[piece]}", fontsize=8, va="center"))
            add(ax.text(0.39, row, f"{z0}-{z1}", fontsize=8, va="center"))
            add(ax.text(0.58, row, f"{self.surf.split_median[piece]:.0f}", fontsize=8, va="center"))
            flipped = piece in self.flip
            add(ax.add_patch(Rectangle((0.76, row - 0.4), 0.21, 0.8, lw=0,
                                       color=FLIP_COLOR if flipped else KEEP_COLOR)))
            add(ax.text(0.865, row, "FLIP" if flipped else "keep", fontsize=8, va="center", ha="center",
                        color="white" if flipped else "black", weight="bold"))
        hidden = len(self.surf.pieces) - len(pieces)
        if hidden:
            add(ax.text(0.5, TABLE_ROWS + 0.6, f"scroll for {hidden} more", fontsize=7, ha="center",
                        va="center", style="italic"))

    def _draw_status(self):
        x, y, z = self.pos
        if self.highlight is None:
            piece_txt = "no pieces"
        else:
            state = "FLIP" if self.highlight in self.flip else "keep"
            piece_txt = (f"piece {self.highlight}: {self.surf.sizes[self.highlight]} triangles, median "
                         f"{self.split_column} {self.surf.split_median[self.highlight]:.1f}, {state}")
        mode = "REVIEW of written file (* = flipped in file); toggle what is still wrong" if self.review else ""
        self.status.set_text(f"{piece_txt}   |   {len(self.flip)} of {len(self.surf.pieces)} pieces "
                             f"marked FLIP   |   x={x} y={y} z={z}\n{mode}")

    # ------------------------------------------------------------- actions
    def set_pos(self, pos):
        self.pos = np.clip(np.round(pos).astype(int), 0, np.array(self.shape[::-1]) - 1)
        self._suspend = True
        for name, axis in (("x", 0), ("y", 1), ("z", 2)):
            self.s_pos[name].set_val(int(self.pos[axis]))
        self._suspend = False
        self.refresh()

    def full_view(self):
        for name, h, v, _ in self.VIEWS:
            x0, x1, y0, y1 = self.img[name].get_extent()
            self.ax[name].set_xlim(x0, x1)
            self.ax[name].set_ylim(y0, y1)
        self.fig.canvas.draw_idle()

    def go_to(self, piece):
        self.highlight = int(piece)
        position = int(np.flatnonzero(self.surf.pieces == piece)[0])
        if not self.table_offset <= position < self.table_offset + TABLE_ROWS:
            self.table_offset = max(0, position - TABLE_ROWS // 2)
        self.set_pos(self.surf.centre_point[piece])

    def step_highlight(self, step):
        if not len(self.surf.pieces):
            return
        current = int(np.flatnonzero(self.surf.pieces == self.highlight)[0]) if self.highlight is not None else -1
        self.go_to(self.surf.pieces[(current + step) % len(self.surf.pieces)])

    def toggle(self, piece):
        if piece is None:
            return
        self.flip ^= {int(piece)}
        self.refresh()

    def show_3d(self, max_points=3000, max_arrows=250):
        if self.highlight is None:
            return
        sel = np.flatnonzero(self.surf.labels == self.highlight)
        pts = self.surf.points[sel[:max_points]]
        sign = -1.0 if self.highlight in self.flip else 1.0
        arrows = sel[::max(self.arrow_every, int(np.ceil(len(sel) / max_arrows)))]
        fig = plt.figure(figsize=(8, 7))
        fig.canvas.manager.set_window_title(f"piece {self.highlight} 3D")
        ax = fig.add_subplot(projection="3d")
        ax.scatter(*pts.T, s=2, c=MEMBRANE_COLOR, depthshade=False)
        start = self.surf.points[arrows]
        vec = self.surf.normals[arrows] * sign * self.dist_vox
        ax.quiver(*start.T, *vec.T, color=FLIP_COLOR if sign < 0 else KEEP_COLOR, linewidth=1.3,
                  arrow_length_ratio=0.3)
        ends = np.concatenate([pts, start + vec])
        lo, hi = ends.min(axis=0), ends.max(axis=0)
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_zlim(lo[2], hi[2])
        ax.set_box_aspect(np.maximum(hi - lo, 1))
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        state = "FLIP (shown flipped)" if sign < 0 else "keep"
        ax.set_title(f"piece {self.highlight}: {state}\nred/green arrows = normals after the current choice",
                     fontsize=9)
        fig.show()

    def _stop(self):
        self.stop = True
        plt.close(self.fig)

    # ------------------------------------------------------------- callbacks
    def _on_pos_slider(self, axis, val):
        self.pos[axis] = int(val)
        self.refresh()

    def _on_contrast(self, _val):
        vmin, vmax = self.s_cmin.val, max(self.s_cmax.val, self.s_cmin.val + 1)
        for img in self.img.values():
            img.set_clim(vmin, vmax)
        self.fig.canvas.draw_idle()

    def _on_display_slider(self, _val):
        self.slab = float(self.s_slab.val)
        self.dist_vox = float(self.s_dist.val)
        self.arrow_every = int(self.s_every.val)
        blur_ang = float(self.s_blur.val)
        self.blur_vox = blur_ang / self.apix_binned if (blur_ang > 0 and self.apix_binned) else blur_ang
        self.refresh()

    def _on_click(self, event):
        toolbar = getattr(self.fig.canvas, "toolbar", None)
        if toolbar is not None and str(getattr(toolbar, "mode", "")):
            return  # zoom/pan active
        if event.button != 1 or event.xdata is None:
            return
        if event.inaxes is self.ax_tab:
            row = int(round(event.ydata))
            position = self.table_offset + row - 1
            if row < 1 or position >= min(len(self.surf.pieces), self.table_offset + TABLE_ROWS):
                return
            piece = int(self.surf.pieces[position])
            if event.xdata >= 0.74:
                self.highlight = piece
                self.toggle(piece)
            else:
                self.go_to(piece)
            return
        for name, h, v, s in self.VIEWS:
            if event.inaxes is not self.ax[name]:
                continue
            offsets = self.mem[name].get_offsets()
            if len(offsets):
                d = np.hypot(offsets[:, 0] - event.xdata, offsets[:, 1] - event.ydata)
                nearest = int(np.argmin(d))
                if d[nearest] <= 5:
                    self.highlight = int(self.surf.labels[self._view_indices(s)[nearest]])
                    self.refresh()
                    return
            pos = self.pos.astype(float)
            pos[h], pos[v] = event.xdata, event.ydata
            self.set_pos(pos)
            return

    def _on_scroll(self, event):
        step = 1 if event.button == "up" else -1
        if event.inaxes is self.ax_tab:
            max_offset = max(0, len(self.surf.pieces) - TABLE_ROWS)
            self.table_offset = int(np.clip(self.table_offset - step, 0, max_offset))
            self.refresh()
            return
        for name, _, _, s in self.VIEWS:
            if event.inaxes is self.ax[name]:
                pos = self.pos.copy()
                pos[s] += step
                self.set_pos(pos)

    def _on_key(self, event):
        if event.key == "t":
            self.toggle(self.highlight)
        elif event.key == "up":
            self.step_highlight(-1)
        elif event.key == "down":
            self.step_highlight(1)
        elif event.key == "x":
            self._stop()

    def run(self):
        plt.show()
        return set(self.flip), self.stop


def header_apix(tomo_path):
    """Pixel size (Angstrom) from the MRC header, or None if it is not set."""
    with mrcfile.open(tomo_path, header_only=True, permissive=True) as mrc:
        apix = float(mrc.voxel_size.x)
    return apix if apix > 0 else None


def check_alignment(surface, shape):
    """Warn if the surface does not fall inside the tomogram (wrong pixel size / surface units)."""
    nz, ny, nx = shape
    lo, hi = surface.points.min(axis=0), surface.points.max(axis=0)
    if np.any(lo < -5) or np.any(hi > np.array([nx, ny, nz]) + 5):
        click.secho(f"  WARNING: surface spans voxels x {lo[0]:.0f}-{hi[0]:.0f}, y {lo[1]:.0f}-{hi[1]:.0f}, "
                    f"z {lo[2]:.0f}-{hi[2]:.0f} but the tomogram is {nx} x {ny} x {nz}; "
                    f"check --apix / --coord-scale", fg="yellow")


def existing_flips(out_path, apix, coord_scale, min_triangles, split_dist, split_column):
    if not os.path.exists(out_path):
        return set()
    return Surface(out_path, apix, coord_scale, min_triangles, split_dist, split_column).flipped_in_file


def run_curation(tomo_file, vtp, vtp_glob, tomo_dir, tomo_pattern, skip_done, dist, apix, coord_scale,
                 split_dist, min_triangles, arrow_every, split_column, bin_factor=1, gaussian=0.0):
    """Curate normal orientations piece by piece in 3D, then flip and review."""
    if plt.get_backend().lower() == "agg":
        raise click.ClickException("no interactive matplotlib backend (is DISPLAY set? try ssh -X)")
    if apix is None:
        apix = header_apix(tomo_file) if tomo_file else None
        if apix is None:
            apix = 9.98
            click.echo(f"Pixel size not given, using {apix:g} A")
        else:
            click.echo(f"Pixel size from the MRC header: {apix:g} A")
    # --apix stays the UNBINNED pixel size; everything mapping surface coords to
    # display voxels uses the binned (effective) pixel size.
    apix_eff = apix * bin_factor
    if vtp:
        vtp_paths = [vtp]
    else:
        vtp_paths = [p for p in sorted(glob.glob(vtp_glob)) if ".flipped." not in os.path.basename(p)]
    if skip_done:
        vtp_paths = [p for p in vtp_paths if not os.path.exists(os.path.splitext(p)[0] + ".flipped.vtp")]
    if not vtp_paths:
        raise click.ClickException(f"No surfaces to do for '{vtp_glob}'")

    for n, vtp_path in enumerate(vtp_paths, 1):
        name = os.path.basename(vtp_path)
        out_path = os.path.splitext(vtp_path)[0] + ".flipped.vtp"
        click.echo(f"\n=== [{n}/{len(vtp_paths)}] {name} ===")
        tomo_path = tomo_file or core.find_tomogram(vtp_path, tomo_dir, tomo_pattern)
        if tomo_path is None:
            click.echo("  WARNING: tomogram not found, showing surfaces without background")
        tomo = volume.load_tomogram_uint8(tomo_path, bin_factor, gaussian, apix_eff) if tomo_path else None
        if tomo is not None:
            click.echo(f"  display volume {tomo.shape} (Z,Y,X), effective pixel size {apix_eff:g} A"
                       f" (bin {bin_factor}" + (f", gaussian {gaussian:g} A)" if gaussian else ")"))
        surface = Surface(vtp_path, apix_eff, coord_scale, min_triangles, split_dist, split_column)
        click.echo(f"  {len(surface.pieces)} pieces (split at {split_column} {split_dist:g})")
        if tomo is not None:
            check_alignment(surface, tomo.shape)
        flip = existing_flips(out_path, apix_eff, coord_scale, min_triangles, split_dist, split_column)
        if flip:
            click.echo(f"  starting from the flips in the existing .flipped.vtp: {sorted(flip)}")
        flip, stop = NormalCurator(tomo, surface, flip, f"[{n}/{len(vtp_paths)}] {name}",
                                   dist / apix_eff, arrow_every=arrow_every, split_column=split_column,
                                   apix_binned=apix_eff).run()
        if stop:
            click.echo("Stopped; this surface was not written.")
            return

        while True:
            if not flip and not os.path.exists(out_path):
                click.echo("  no pieces marked FLIP, nothing written")
                break
            click.echo(f"  flipping pieces {sorted(flip)}")
            core.flip_surface(vtp_path, flip, split_dist=split_dist, min_triangles=min_triangles,
                              split_column=split_column)
            preview.preview_vtp(out_path, tomo_dir=tomo_dir, tomo_pattern=tomo_pattern, tomo_path=tomo_path,
                                dist=dist, apix=apix, coord_scale=coord_scale, min_triangles=min_triangles,
                                split_dist=split_dist, split_column=split_column)
            review = Surface(out_path, apix_eff, coord_scale, min_triangles, split_dist, split_column)
            changes, stop = NormalCurator(tomo, review, set(),
                                          f"REVIEW [{n}/{len(vtp_paths)}] {os.path.basename(out_path)}",
                                          dist / apix_eff, arrow_every=arrow_every, review=True,
                                          split_column=split_column, apix_binned=apix_eff).run()
            if stop:
                click.echo("Stopped after writing this surface.")
                return
            if not changes:
                break
            flip ^= changes
            click.echo(f"  corrections in review: {sorted(changes)}")
