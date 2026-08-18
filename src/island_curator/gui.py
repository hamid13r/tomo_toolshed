"""Interactive matplotlib GUI for curating a labeled segmentation.

``IslandCuratorGUI`` holds all mutable state on ``self`` (labels, sizes, bboxes,
selection, view indices, color mode, highlight id) so callbacks mutate instance
attributes rather than globals/nonlocals. matplotlib is imported lazily at
construction time so the rest of the package stays headless-importable.

Volume axis order is (Z, Y, X). The left panel is the Z view (a Y-X plane at
fixed Z); the right panel is the Y view (a Z-X plane at fixed Y).
"""

from __future__ import annotations

from typing import Optional, Set

import numpy as np

from . import colormaps
from . import labeling


class IslandCuratorGUI:
    def __init__(
        self,
        tomogram: np.ndarray,
        labels: np.ndarray,
        n_islands: int,
        connectivity: int = 26,
        color_by_number: bool = False,
        selected: Optional[Set[int]] = None,
    ):
        # Lazy matplotlib imports -- keep module import headless-safe.
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Slider, Button, TextBox

        self._plt = plt
        self._Slider = Slider
        self._Button = Button
        self._TextBox = TextBox

        self.tomo = _to_uint8(tomogram)
        self.labels = np.asarray(labels).astype(np.int32)
        self.n = int(n_islands)
        self.connectivity = connectivity

        self.nz, self.ny, self.nx = self.labels.shape

        # Selection / view state.
        if selected is None:
            selected = set(range(1, self.n + 1))
        self.selected: Set[int] = set(int(i) for i in selected)
        self.z_idx = self.nz // 2
        self.y_idx = self.ny // 2
        self.x_idx = self.nx // 2
        self.color_mode = "number" if color_by_number else "green"
        self.highlight_id: Optional[int] = None
        self.show_labels = True

        # Derived tables.
        self.bboxes, self.sizes = labeling.compute_bboxes_and_sizes(self.labels, self.n)
        self.label_colors = colormaps.build_label_colors(self.n)

        # Original snapshot for Reset.
        self._orig_labels = self.labels.copy()
        self._orig_n = self.n
        self._orig_selected = set(self.selected)

        # Populated on close.
        self.result: Optional[np.ndarray] = None

        self._artists = {}
        self._build_figure()
        self._full_refresh()

    # ------------------------------------------------------------------ setup
    def _build_figure(self):
        plt = self._plt
        self.fig = plt.figure(figsize=(15, 9))
        self.fig.canvas.manager.set_window_title("island-curator")

        # Image axes.
        self.ax_z = self.fig.add_axes([0.05, 0.42, 0.40, 0.54])
        self.ax_y = self.fig.add_axes([0.54, 0.42, 0.40, 0.54])
        self.ax_z.set_title("Z view (axis 0)  —  Y–X plane")
        self.ax_y.set_title("Y view (axis 1)  —  Z–X plane")
        for ax in (self.ax_z, self.ax_y):
            ax.set_xticks([])
            ax.set_yticks([])

        # Base + overlay + highlight images.
        self._artists["z_base"] = self.ax_z.imshow(
            self.tomo[self.z_idx], cmap="gray", vmin=0, vmax=255, interpolation="nearest"
        )
        self._artists["z_over"] = self.ax_z.imshow(
            np.zeros((self.ny, self.nx, 4)), interpolation="nearest"
        )
        self._artists["z_hi"] = self.ax_z.imshow(
            np.zeros((self.ny, self.nx, 4)), interpolation="nearest"
        )
        self._artists["y_base"] = self.ax_y.imshow(
            self.tomo[:, self.y_idx], cmap="gray", vmin=0, vmax=255,
            interpolation="nearest", aspect="auto",
        )
        self._artists["y_over"] = self.ax_y.imshow(
            np.zeros((self.nz, self.nx, 4)), interpolation="nearest", aspect="auto"
        )
        self._artists["y_hi"] = self.ax_y.imshow(
            np.zeros((self.nz, self.nx, 4)), interpolation="nearest", aspect="auto"
        )

        # Crosshairs.
        self._artists["z_vline"] = self.ax_z.axvline(self.x_idx, color="cyan", lw=0.6, alpha=0.7)
        self._artists["z_hline"] = self.ax_z.axhline(self.y_idx, color="cyan", lw=0.6, alpha=0.7)
        self._artists["y_vline"] = self.ax_y.axvline(self.x_idx, color="cyan", lw=0.6, alpha=0.7)
        self._artists["y_hline"] = self.ax_y.axhline(self.z_idx, color="cyan", lw=0.6, alpha=0.7)

        self._bbox_artists = []  # rectangles + text drawn per refresh

        # Sliders.
        ax_zsl = self.fig.add_axes([0.47, 0.42, 0.012, 0.54])
        self.s_z = self._Slider(
            ax_zsl, "Z", 0, self.nz - 1, valinit=self.z_idx, valstep=1, orientation="vertical"
        )
        ax_ysl = self.fig.add_axes([0.54, 0.385, 0.40, 0.02])
        self.s_y = self._Slider(
            ax_ysl, "Y", 0, self.ny - 1, valinit=self.y_idx, valstep=1, orientation="horizontal"
        )
        self.s_z.on_changed(self._on_z_slider)
        self.s_y.on_changed(self._on_y_slider)

        # ---- Control panel (bottom) -------------------------------------
        def _ax(x, y, w, h):
            return self.fig.add_axes([x, y, w, h])

        # Row 1: Z-range.
        self.tb_zmin = self._TextBox(_ax(0.06, 0.31, 0.05, 0.035), "Z min ", initial="0")
        self.tb_zmax = self._TextBox(_ax(0.17, 0.31, 0.05, 0.035), "Z max ", initial=str(self.nz - 1))
        self.b_zrange = self._Button(_ax(0.24, 0.31, 0.11, 0.035), "Apply Z-range")
        self.b_zrange.on_clicked(self._on_apply_zrange)

        # Row 2: size filter + morphology.
        self.tb_minsize = self._TextBox(_ax(0.06, 0.26, 0.05, 0.035), "Min size ", initial="0")
        self.b_filter = self._Button(_ax(0.17, 0.26, 0.10, 0.035), "Apply Filter")
        self.b_filter.on_clicked(self._on_apply_filter)
        self.tb_radius = self._TextBox(_ax(0.31, 0.26, 0.04, 0.035), "Radius ", initial="1")
        self.tb_iters = self._TextBox(_ax(0.40, 0.26, 0.04, 0.035), "Iters ", initial="1")
        self.b_dilate = self._Button(_ax(0.45, 0.26, 0.07, 0.035), "Dilate")
        self.b_dilate.on_clicked(self._on_dilate)
        self.b_erode = self._Button(_ax(0.53, 0.26, 0.07, 0.035), "Erode")
        self.b_erode.on_clicked(self._on_erode)

        # Row 3: go-to / highlight / color mode.
        self.tb_island = self._TextBox(_ax(0.08, 0.21, 0.05, 0.035), "Island ID ", initial="1")
        self.b_goto = self._Button(_ax(0.14, 0.21, 0.06, 0.035), "Go To")
        self.b_goto.on_clicked(self._on_goto)
        self.b_toggle = self._Button(_ax(0.21, 0.21, 0.07, 0.035), "Toggle")
        self.b_toggle.on_clicked(self._on_toggle_button)
        self.b_color = self._Button(_ax(0.29, 0.21, 0.13, 0.035), self._color_button_label())
        self.b_color.on_clicked(self._on_toggle_color)

        # Row 4: selection ops.
        self.b_renumber = self._Button(_ax(0.06, 0.16, 0.09, 0.035), "Renumber")
        self.b_renumber.on_clicked(self._on_renumber)
        self.b_reset = self._Button(_ax(0.16, 0.16, 0.08, 0.035), "Reset")
        self.b_reset.on_clicked(self._on_reset)
        self.b_selall = self._Button(_ax(0.25, 0.16, 0.09, 0.035), "Select All")
        self.b_selall.on_clicked(self._on_select_all)
        self.b_invert = self._Button(_ax(0.35, 0.16, 0.07, 0.035), "Invert")
        self.b_invert.on_clicked(self._on_invert)
        self.b_labels = self._Button(_ax(0.43, 0.16, 0.11, 0.035), "Hide Labels")
        self.b_labels.on_clicked(self._on_toggle_labels)

        # Info / status text.
        self._info_text = self.fig.text(0.06, 0.11, "", fontsize=9, family="monospace", va="top")
        self._status_text = self.fig.text(0.06, 0.025, "", fontsize=9, family="monospace", va="top")

        # Events.
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("scroll_event", self._on_scroll)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.fig.canvas.mpl_connect("close_event", self._on_close)

    # ------------------------------------------------------------- rendering
    def _color_button_label(self):
        return "All green" if self.color_mode == "number" else "Color by number"

    def _z_overlay(self):
        sl = self.labels[self.z_idx]
        if self.color_mode == "green":
            return colormaps.green_overlay(sl, self.selected)
        return colormaps.numbered_overlay(sl, self.selected, self.label_colors)

    def _y_overlay(self):
        sl = self.labels[:, self.y_idx]
        if self.color_mode == "green":
            return colormaps.green_overlay(sl, self.selected)
        return colormaps.numbered_overlay(sl, self.selected, self.label_colors)

    def _refresh_images(self):
        self._artists["z_base"].set_data(self.tomo[self.z_idx])
        self._artists["y_base"].set_data(self.tomo[:, self.y_idx])
        self._artists["z_over"].set_data(self._z_overlay())
        self._artists["y_over"].set_data(self._y_overlay())
        self._artists["z_hi"].set_data(
            colormaps.highlight_overlay(self.labels[self.z_idx], self.highlight_id)
        )
        self._artists["y_hi"].set_data(
            colormaps.highlight_overlay(self.labels[:, self.y_idx], self.highlight_id)
        )

    def _refresh_crosshairs(self):
        self._artists["z_vline"].set_xdata([self.x_idx, self.x_idx])
        self._artists["z_hline"].set_ydata([self.y_idx, self.y_idx])
        self._artists["y_vline"].set_xdata([self.x_idx, self.x_idx])
        self._artists["y_hline"].set_ydata([self.z_idx, self.z_idx])

    def _clear_bbox_artists(self):
        for art in self._bbox_artists:
            art.remove()
        self._bbox_artists = []

    def _draw_bboxes(self):
        from matplotlib.patches import Rectangle

        self._clear_bbox_artists()
        if not self.show_labels:
            return
        for iid, bb in self.bboxes.items():
            if iid not in self.selected:
                continue
            zsl, ysl, xsl = bb
            # Z view: island present if current z is within its z-extent.
            if zsl.start <= self.z_idx < zsl.stop:
                rect = Rectangle(
                    (xsl.start - 0.5, ysl.start - 0.5),
                    xsl.stop - xsl.start, ysl.stop - ysl.start,
                    fill=False, edgecolor="yellow", lw=0.6,
                )
                self.ax_z.add_patch(rect)
                txt = self.ax_z.text(xsl.start, ysl.start - 2, str(iid), color="yellow", fontsize=6)
                self._bbox_artists += [rect, txt]
            # Y view: island present if current y is within its y-extent.
            if ysl.start <= self.y_idx < ysl.stop:
                rect = Rectangle(
                    (xsl.start - 0.5, zsl.start - 0.5),
                    xsl.stop - xsl.start, zsl.stop - zsl.start,
                    fill=False, edgecolor="yellow", lw=0.6,
                )
                self.ax_y.add_patch(rect)
                txt = self.ax_y.text(xsl.start, zsl.start - 2, str(iid), color="yellow", fontsize=6)
                self._bbox_artists += [rect, txt]

    def _refresh_info(self):
        # Highlighted island info.
        if self.highlight_id and self.highlight_id <= self.n:
            hsize = int(self.sizes[self.highlight_id])
            hi = f"Island {self.highlight_id}: {hsize} vox"
        else:
            hi = "Island --: -- vox"
        # Smallest currently selected island.
        if self.selected:
            sel = np.fromiter(self.selected, dtype=np.int64)
            sel = sel[(sel > 0) & (sel < len(self.sizes))]
            if sel.size:
                ssizes = self.sizes[sel]
                order = np.argmin(ssizes)
                sm_id = int(sel[order])
                sm_sz = int(ssizes[order])
                smallest = f"Smallest: {sm_id} ({sm_sz} vox)"
            else:
                smallest = "Smallest: -- (-- vox)"
        else:
            smallest = "Smallest: -- (-- vox)"
        self._info_text.set_text(f"{hi}    {smallest}")

    def _refresh_status(self, msg: str = None):
        sel_vox = int(self.sizes[list(self.selected)].sum()) if self.selected else 0
        base = (
            f"islands: {self.n}   selected: {len(self.selected)}   "
            f"selected voxels: {sel_vox}   "
            f"z={self.z_idx}/{self.nz-1}  y={self.y_idx}/{self.ny-1}  "
            f"mode: {self.color_mode}"
        )
        if msg:
            base = msg + "\n" + base
        self._status_text.set_text(base)

    def _full_refresh(self, msg: str = None):
        self._refresh_images()
        self._refresh_crosshairs()
        self._draw_bboxes()
        self._refresh_info()
        self._refresh_status(msg)
        self.fig.canvas.draw_idle()

    # --------------------------------------------------------- recompute core
    def _recompute_tables(self):
        self.bboxes, self.sizes = labeling.compute_bboxes_and_sizes(self.labels, self.n)
        self.label_colors = colormaps.build_label_colors(self.n)

    def _relabel_from_binary(self, binary: np.ndarray, keep_selection: bool = False):
        """Relabel from a binary mask; reset selection to all new islands."""
        self.labels, self.n = labeling.label_islands(binary, self.connectivity)
        self._recompute_tables()
        self.selected = set(range(1, self.n + 1))
        if self.highlight_id and self.highlight_id > self.n:
            self.highlight_id = None

    def _current_binary(self) -> np.ndarray:
        return labeling.binary_from_selection(self.labels, self.selected)

    # ------------------------------------------------------------- callbacks
    def _on_z_slider(self, val):
        self.z_idx = int(val)
        self._full_refresh()

    def _on_y_slider(self, val):
        self.y_idx = int(val)
        self._full_refresh()

    def _parse_int(self, textbox, default):
        try:
            return int(float(textbox.text))
        except (ValueError, TypeError):
            return default

    def _on_apply_zrange(self, _event):
        zmin = self._parse_int(self.tb_zmin, 0)
        zmax = self._parse_int(self.tb_zmax, self.nz - 1)
        binary = self._current_binary()
        binary = labeling.apply_zrange(binary, zmin, zmax)
        self._relabel_from_binary(binary)
        self._full_refresh(f"Applied Z-range [{zmin}, {zmax}]; relabeled to {self.n} islands.")

    def _on_apply_filter(self, _event):
        min_size = self._parse_int(self.tb_minsize, 0)
        keep = set(labeling.filter_by_size(self.sizes, min_size, ids=self.selected))
        dropped = len(self.selected) - len(keep)
        self.selected = keep
        self._full_refresh(f"Deselected {dropped} islands smaller than {min_size} vox (reversible).")

    def _on_dilate(self, _event):
        r = self._parse_int(self.tb_radius, 1)
        it = self._parse_int(self.tb_iters, 1)
        binary = labeling.dilate_binary(self._current_binary(), radius=r, iterations=it)
        self._relabel_from_binary(binary)
        self._full_refresh(f"Dilated (r={r}, it={it}); relabeled to {self.n} islands.")

    def _on_erode(self, _event):
        r = self._parse_int(self.tb_radius, 1)
        it = self._parse_int(self.tb_iters, 1)
        binary = labeling.erode_binary(self._current_binary(), radius=r, iterations=it)
        self._relabel_from_binary(binary)
        self._full_refresh(f"Eroded (r={r}, it={it}); relabeled to {self.n} islands.")

    def _on_goto(self, _event):
        iid = self._parse_int(self.tb_island, 0)
        if iid <= 0 or iid > self.n or iid not in self.bboxes:
            self._full_refresh(f"Island {iid} not found.")
            return
        zc, yc, xc = labeling.bbox_center(self.bboxes[iid])
        self.z_idx, self.y_idx, self.x_idx = zc, yc, xc
        self.highlight_id = iid
        # Move sliders without re-triggering a double refresh loop.
        self.s_z.set_val(zc)
        self.s_y.set_val(yc)
        self._full_refresh(f"Went to island {iid}; highlighted in yellow.")

    def _on_toggle_button(self, _event):
        iid = self._parse_int(self.tb_island, 0)
        if iid <= 0 or iid > self.n:
            self._full_refresh(f"Island {iid} not found.")
            return
        self._toggle_id(iid)

    def _toggle_id(self, iid: int):
        if iid in self.selected:
            self.selected.discard(iid)
            state = "removed"
        else:
            self.selected.add(iid)
            state = "added"
        self._full_refresh(f"Island {iid} {state}.")

    def _on_toggle_color(self, _event):
        self.color_mode = "green" if self.color_mode == "number" else "number"
        self.b_color.label.set_text(self._color_button_label())
        self._full_refresh(f"Overlay mode: {self.color_mode}.")

    def _on_renumber(self, _event):
        self.labels, id_map = labeling.renumber(self.labels, self.selected)
        self.n = len(id_map)
        self._recompute_tables()
        self.selected = set(range(1, self.n + 1))
        if self.highlight_id in id_map:
            self.highlight_id = id_map[self.highlight_id]
        else:
            self.highlight_id = None
        self._full_refresh(f"Renumbered to contiguous 1..{self.n}.")

    def _on_reset(self, _event):
        self.labels = self._orig_labels.copy()
        self.n = self._orig_n
        self._recompute_tables()
        self.selected = set(self._orig_selected)
        self.highlight_id = None
        self._full_refresh("Reset to original labeling and selection.")

    def _on_select_all(self, _event):
        self.selected = set(range(1, self.n + 1))
        self._full_refresh("Selected all islands.")

    def _on_invert(self, _event):
        allids = set(range(1, self.n + 1))
        self.selected = allids - self.selected
        self._full_refresh("Inverted selection.")

    def _on_toggle_labels(self, _event):
        self.show_labels = not self.show_labels
        self.b_labels.label.set_text("Hide Labels" if self.show_labels else "Show Labels")
        self._full_refresh()

    def _on_click(self, event):
        if event.inaxes not in (self.ax_z, self.ax_y) or event.xdata is None:
            return
        if event.button != 1:
            return
        x = int(round(event.xdata))
        if event.inaxes is self.ax_z:
            y = int(round(event.ydata))
            if not (0 <= x < self.nx and 0 <= y < self.ny):
                return
            lab = int(self.labels[self.z_idx, y, x])
            if lab > 0:
                self._toggle_id(lab)
            else:
                self.x_idx = x
                self.y_idx = y
                self.s_y.set_val(y)
                self._full_refresh()
        else:  # ax_y
            z = int(round(event.ydata))
            if not (0 <= x < self.nx and 0 <= z < self.nz):
                return
            lab = int(self.labels[z, self.y_idx, x])
            if lab > 0:
                self._toggle_id(lab)
            else:
                self.x_idx = x
                self.z_idx = z
                self.s_z.set_val(z)
                self._full_refresh()

    def _on_scroll(self, event):
        step = 1 if event.button == "up" else -1
        if event.inaxes is self.ax_z:
            self.z_idx = int(np.clip(self.z_idx + step, 0, self.nz - 1))
            self.s_z.set_val(self.z_idx)
        elif event.inaxes is self.ax_y:
            self.y_idx = int(np.clip(self.y_idx + step, 0, self.ny - 1))
            self.s_y.set_val(self.y_idx)

    def _on_key(self, event):
        if event.key == "q":
            self._plt.close(self.fig)

    def _on_close(self, _event):
        # Bake the current selection into the exported binary mask.
        self.result = self._current_binary()

    # ------------------------------------------------------------------- run
    def run(self) -> np.ndarray:
        """Show the GUI (blocking) and return the curated binary mask (uint8 0/1)."""
        self._plt.show()
        if self.result is None:
            self.result = self._current_binary()
        return self.result


def _to_uint8(volume: np.ndarray) -> np.ndarray:
    """Normalize an arbitrary-dtype volume to uint8 0..255 for display only."""
    vol = np.asarray(volume).astype(np.float32)
    vmin = float(vol.min())
    vmax = float(vol.max())
    if vmax <= vmin:
        return np.zeros(vol.shape, dtype=np.uint8)
    scaled = (vol - vmin) / (vmax - vmin) * 255.0
    return scaled.astype(np.uint8)
