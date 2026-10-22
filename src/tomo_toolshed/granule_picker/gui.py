"""Interactive matplotlib GUI for ``pick-granules``.

Shows the tomogram with the mask (faint blue) and the thresholded granules
(red) overlaid, in the same two orthogonal views as ``curate``: the Z view (a
Y-X plane at fixed Z) and the Y view (a Z-X plane at fixed Y). A histogram of
the tomogram values inside the mask, with the threshold marked, sits under the
views; dragging the threshold slider updates the overlay live.

The GUI only picks the parameters (threshold, mask erosion, min size). The
caller computes and writes the final volume, so ``run()`` returns those
parameters, or ``None`` if the user quit without saving (Esc).

matplotlib is imported lazily at construction so the package stays headless.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from . import core

_MASK_RGB = (0.25, 0.55, 1.0)
_GRANULE_RGB = (1.0, 0.15, 0.1)


class GranulePickerGUI:
    def __init__(
        self,
        tomogram: np.ndarray,
        raw_mask: np.ndarray,
        threshold: float,
        bright: bool = False,
        mask_erode: int = 0,
        min_size: int = 0,
        erode_3d: bool = False,
    ):
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button, Slider, TextBox

        self._plt = plt
        self._Slider = Slider
        self._Button = Button
        self._TextBox = TextBox

        self.tomo = np.asarray(tomogram, dtype=np.float32)
        self.disp = _to_uint8(self.tomo)
        self.raw_mask = np.asarray(raw_mask) > 0
        self.bright = bool(bright)
        self.mask_erode = int(mask_erode)
        self.erode_3d = bool(erode_3d)
        self.min_size = int(min_size)
        self.threshold = float(threshold)

        self.nz, self.ny, self.nx = self.tomo.shape
        self.z_idx = self.nz // 2
        self.y_idx = self.ny // 2
        self.x_idx = self.nx // 2
        self.vmin, self.vmax = 0, 255
        self.overlay_alpha = 0.5
        self.show_mask = True

        self.cancelled = False
        self._set_mask(self.mask_erode)

        self._artists = {}
        self._build_figure()
        self._full_refresh()

    # ------------------------------------------------------------------ state
    def _set_mask(self, erode_px: int):
        self.mask_erode = int(erode_px)
        self.mask = core.prepare_mask(self.raw_mask, self.mask_erode, self.erode_3d)
        self.vals = core.masked_values(self.tomo, self.mask)

    def _value_range(self):
        if self.vals.size == 0:
            lo, hi = float(self.tomo.min()), float(self.tomo.max())
        else:
            lo = float(np.percentile(self.vals, 0.1))
            hi = float(np.percentile(self.vals, 99.9))
        lo, hi = min(lo, self.threshold), max(hi, self.threshold)
        if hi <= lo:
            hi = lo + 1.0
        return lo, hi

    # ------------------------------------------------------------------ setup
    def _build_figure(self):
        plt = self._plt
        self.fig = plt.figure(figsize=(15, 9))
        self.fig.canvas.manager.set_window_title("pick-granules")

        self.ax_z = self.fig.add_axes([0.05, 0.42, 0.40, 0.54])
        self.ax_y = self.fig.add_axes([0.54, 0.42, 0.40, 0.54])
        self.ax_z.set_title("Z view (axis 0)  —  Y–X plane")
        self.ax_y.set_title("Y view (axis 1)  —  Z–X plane")
        for ax in (self.ax_z, self.ax_y):
            ax.set_xticks([])
            ax.set_yticks([])

        empty_z = np.zeros((self.ny, self.nx, 4))
        empty_y = np.zeros((self.nz, self.nx, 4))
        self._artists["z_base"] = self.ax_z.imshow(
            self.disp[self.z_idx], cmap="gray", vmin=self.vmin, vmax=self.vmax,
            interpolation="nearest")
        self._artists["z_mask"] = self.ax_z.imshow(empty_z, interpolation="nearest")
        self._artists["z_gran"] = self.ax_z.imshow(empty_z, interpolation="nearest")
        kw = dict(interpolation="nearest", aspect="auto", origin="lower")
        self._artists["y_base"] = self.ax_y.imshow(
            self.disp[:, self.y_idx], cmap="gray", vmin=self.vmin, vmax=self.vmax, **kw)
        self._artists["y_mask"] = self.ax_y.imshow(empty_y, **kw)
        self._artists["y_gran"] = self.ax_y.imshow(empty_y, **kw)

        self._artists["z_vline"] = self.ax_z.axvline(self.x_idx, color="cyan", lw=0.6, alpha=0.7)
        self._artists["z_hline"] = self.ax_z.axhline(self.y_idx, color="cyan", lw=0.6, alpha=0.7)
        self._artists["y_vline"] = self.ax_y.axvline(self.x_idx, color="cyan", lw=0.6, alpha=0.7)
        self._artists["y_hline"] = self.ax_y.axhline(self.z_idx, color="cyan", lw=0.6, alpha=0.7)

        ax_zsl = self.fig.add_axes([0.47, 0.42, 0.012, 0.54])
        self.s_z = self._Slider(ax_zsl, "Z", 0, self.nz - 1, valinit=self.z_idx,
                                valstep=1, orientation="vertical")
        ax_ysl = self.fig.add_axes([0.54, 0.385, 0.40, 0.02])
        self.s_y = self._Slider(ax_ysl, "Y", 0, self.ny - 1, valinit=self.y_idx, valstep=1)
        self.s_z.on_changed(self._on_z_slider)
        self.s_y.on_changed(self._on_y_slider)

        # Histogram of masked values + threshold slider (bottom left).
        self.ax_hist = self.fig.add_axes([0.06, 0.20, 0.40, 0.15])
        self.ax_hist.tick_params(labelsize=7)
        self.ax_hist.set_yticks([])
        ax_thr = self.fig.add_axes([0.06, 0.145, 0.40, 0.022])
        lo, hi = self._value_range()
        self.s_thr = self._Slider(ax_thr, "threshold", lo, hi, valinit=self.threshold,
                                  valfmt="%.4g")
        self.s_thr.on_changed(self._on_threshold)
        self.tb_thr = self._TextBox(self.fig.add_axes([0.11, 0.095, 0.08, 0.032]),
                                    "value ", initial=f"{self.threshold:.6g}")
        self.tb_thr.on_submit(self._on_threshold_text)
        self.tb_pct = self._TextBox(self.fig.add_axes([0.28, 0.095, 0.06, 0.032]),
                                    "percentile ", initial="")
        self.tb_pct.on_submit(self._on_percentile_text)
        self._draw_histogram()

        # Display controls (right).
        self.fig.text(0.62, 0.355, "Contrast (display):", fontsize=9)
        self.s_cmin = self._Slider(self.fig.add_axes([0.72, 0.33, 0.20, 0.018]),
                                   "min", 0, 255, valinit=self.vmin, valstep=1)
        self.s_cmax = self._Slider(self.fig.add_axes([0.72, 0.305, 0.20, 0.018]),
                                   "max", 0, 255, valinit=self.vmax, valstep=1)
        self.s_cmin.on_changed(self._on_contrast)
        self.s_cmax.on_changed(self._on_contrast)
        self.fig.text(0.62, 0.275, "Overlay (display):", fontsize=9)
        self.s_alpha = self._Slider(self.fig.add_axes([0.72, 0.25, 0.20, 0.018]),
                                    "opacity", 0.0, 1.0, valinit=self.overlay_alpha,
                                    valstep=0.05)
        self.s_alpha.on_changed(self._on_opacity)
        self.b_mask = self._Button(self.fig.add_axes([0.62, 0.20, 0.10, 0.035]), "Hide mask")
        self.b_mask.on_clicked(self._on_toggle_mask)

        # Mask erosion and min size (right).
        self.tb_erode = self._TextBox(self.fig.add_axes([0.66, 0.15, 0.04, 0.035]),
                                      "Erode mask (px) ", initial=str(self.mask_erode))
        self.b_erode3d = self._Button(self.fig.add_axes([0.705, 0.15, 0.03, 0.035]),
                                      self._erode_mode_label())
        self.b_erode3d.on_clicked(self._on_toggle_erode_3d)
        self.b_erode = self._Button(self.fig.add_axes([0.74, 0.15, 0.05, 0.035]), "Apply")
        self.b_erode.on_clicked(self._on_apply_erode)
        self.tb_minsize = self._TextBox(self.fig.add_axes([0.86, 0.15, 0.06, 0.035]),
                                        "Min size (vox) ", initial=str(self.min_size))
        self.tb_minsize.on_submit(self._on_minsize)

        self._status_text = self.fig.text(0.53, 0.10, "", fontsize=9, family="monospace",
                                          va="top")
        self.fig.text(0.06, 0.03,
                      "q / close window: save & quit    Esc: quit without saving    "
                      "scroll: move slice    click: move crosshair",
                      fontsize=8, color="0.35")

        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("scroll_event", self._on_scroll)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

    def _draw_histogram(self):
        ax = self.ax_hist
        ax.cla()
        ax.set_yticks([])
        ax.tick_params(labelsize=7)
        lo, hi = self._value_range()
        if self.vals.size:
            step = max(1, self.vals.size // 2_000_000)  # subsample big masks
            ax.hist(self.vals[::step], bins=200, range=(lo, hi), color="0.45", log=True)
        ax.set_xlim(lo, hi)
        side = "brighter" if self.bright else "darker"
        ax.set_title(f"tomogram values inside the mask (picking {side} than threshold)",
                     fontsize=8)
        self._thr_line = ax.axvline(self.threshold, color="red", lw=1.2)
        self.s_thr.valmin, self.s_thr.valmax = lo, hi
        self.s_thr.ax.set_xlim(lo, hi)

    # -------------------------------------------------------------- rendering
    def _overlay(self, mask_sl, tomo_sl):
        rgba = np.zeros(mask_sl.shape + (4,), dtype=np.float32)
        if self.show_mask:
            rgba[mask_sl, :3] = _MASK_RGB
            rgba[mask_sl, 3] = 0.15
        gran = core.threshold_granules(tomo_sl, mask_sl, self.threshold, self.bright)
        rgba[gran, :3] = _GRANULE_RGB
        rgba[gran, 3] = self.overlay_alpha
        return rgba

    def _refresh_images(self):
        self._artists["z_base"].set_data(self.disp[self.z_idx])
        self._artists["y_base"].set_data(self.disp[:, self.y_idx])
        self._artists["z_gran"].set_data(
            self._overlay(self.mask[self.z_idx], self.tomo[self.z_idx]))
        self._artists["y_gran"].set_data(
            self._overlay(self.mask[:, self.y_idx], self.tomo[:, self.y_idx]))

    def _refresh_crosshairs(self):
        self._artists["z_vline"].set_xdata([self.x_idx, self.x_idx])
        self._artists["z_hline"].set_ydata([self.y_idx, self.y_idx])
        self._artists["y_vline"].set_xdata([self.x_idx, self.x_idx])
        self._artists["y_hline"].set_ydata([self.z_idx, self.z_idx])

    def _refresh_status(self, msg: Optional[str] = None):
        n_mask = int(self.vals.size)
        n_pick = core.count_picked(self.vals, self.threshold, self.bright)
        pct = 100.0 * n_pick / n_mask if n_mask else 0.0
        lines = [
            f"threshold {self.threshold:.5g}  ({core.percentile_of(self.vals, self.threshold):.2f}"
            f" percentile of mask)",
            f"picked {n_pick} of {n_mask} mask voxels ({pct:.2f}%)",
            f"mask erosion {self.mask_erode} px {self._erode_mode_label()}   min size {self.min_size} vox"
            + ("  (applied on save)" if self.min_size > 1 else ""),
            f"z={self.z_idx}/{self.nz - 1}  y={self.y_idx}/{self.ny - 1}",
        ]
        if msg:
            lines.insert(0, msg)
        self._status_text.set_text("\n".join(lines))

    def _full_refresh(self, msg: Optional[str] = None):
        self._refresh_images()
        self._refresh_crosshairs()
        self._refresh_status(msg)
        self.fig.canvas.draw_idle()

    # -------------------------------------------------------------- callbacks
    def _on_z_slider(self, val):
        self.z_idx = int(val)
        self._full_refresh()

    def _on_y_slider(self, val):
        self.y_idx = int(val)
        self._full_refresh()

    def _on_threshold(self, val):
        self.threshold = float(val)
        self._thr_line.set_xdata([self.threshold, self.threshold])
        self._full_refresh()

    def _set_threshold(self, value: float):
        lo, hi = self.s_thr.valmin, self.s_thr.valmax
        if not lo <= value <= hi:  # widen the slider so any value is reachable
            self.s_thr.valmin, self.s_thr.valmax = min(lo, value), max(hi, value)
            self.s_thr.ax.set_xlim(self.s_thr.valmin, self.s_thr.valmax)
        self.s_thr.set_val(value)  # triggers _on_threshold

    def _on_threshold_text(self, text):
        try:
            self._set_threshold(float(text))
        except (TypeError, ValueError):
            self._full_refresh(f"Invalid threshold {text!r}.")

    def _on_percentile_text(self, text):
        if not str(text).strip():
            return
        try:
            self._set_threshold(core.threshold_from_percentile(self.vals, float(text)))
        except (TypeError, ValueError) as exc:
            self._full_refresh(f"Invalid percentile {text!r}: {exc}")

    def _on_contrast(self, _val):
        self.vmin = float(self.s_cmin.val)
        self.vmax = max(float(self.s_cmax.val), self.vmin + 1)
        self._artists["z_base"].set_clim(self.vmin, self.vmax)
        self._artists["y_base"].set_clim(self.vmin, self.vmax)
        self.fig.canvas.draw_idle()

    def _on_opacity(self, val):
        self.overlay_alpha = float(val)
        self._full_refresh()

    def _on_toggle_mask(self, _event):
        self.show_mask = not self.show_mask
        self.b_mask.label.set_text("Hide mask" if self.show_mask else "Show mask")
        self._full_refresh()

    def _erode_mode_label(self):
        return "3D" if self.erode_3d else "XY"

    def _on_toggle_erode_3d(self, _event):
        self.erode_3d = not self.erode_3d
        self.b_erode3d.label.set_text(self._erode_mode_label())
        self._full_refresh(f"Erosion mode {self._erode_mode_label()}; press Apply to use it.")

    def _on_apply_erode(self, _event):
        try:
            px = int(float(self.tb_erode.text))
        except (TypeError, ValueError):
            px = -1
        if px < 0:
            self._full_refresh(f"Invalid erosion {self.tb_erode.text!r}.")
            return
        self._set_mask(px)
        self._draw_histogram()
        self._full_refresh(f"Mask eroded by {px} px in {self._erode_mode_label()}.")

    def _on_minsize(self, text):
        try:
            self.min_size = max(0, int(float(text)))
        except (TypeError, ValueError):
            self._full_refresh(f"Invalid min size {text!r}.")
            return
        self._full_refresh()

    def _on_click(self, event):
        if event.button != 1 or event.xdata is None:
            return
        x = int(round(event.xdata))
        if event.inaxes is self.ax_z:
            y = int(round(event.ydata))
            if 0 <= x < self.nx and 0 <= y < self.ny:
                self.x_idx = x
                self.s_y.set_val(y)  # refreshes
        elif event.inaxes is self.ax_y:
            z = int(round(event.ydata))
            if 0 <= x < self.nx and 0 <= z < self.nz:
                self.x_idx = x
                self.s_z.set_val(z)

    def _on_scroll(self, event):
        step = 1 if event.button == "up" else -1
        if event.inaxes is self.ax_z:
            self.s_z.set_val(int(np.clip(self.z_idx + step, 0, self.nz - 1)))
        elif event.inaxes is self.ax_y:
            self.s_y.set_val(int(np.clip(self.y_idx + step, 0, self.ny - 1)))

    def _on_key(self, event):
        if event.key == "escape":
            self.cancelled = True
            self._plt.close(self.fig)
        elif event.key == "q":
            self._plt.close(self.fig)

    # -------------------------------------------------------------------- run
    def params(self) -> dict:
        return {"threshold": self.threshold, "mask_erode": self.mask_erode,
                "erode_3d": self.erode_3d, "min_size": self.min_size}

    def run(self) -> Optional[dict]:
        """Show the GUI (blocking). Returns the chosen parameters, or ``None`` on Esc."""
        self._plt.show()
        return None if self.cancelled else self.params()


def _to_uint8(volume: np.ndarray) -> np.ndarray:
    """Normalize to uint8 0..255 for display, clipping the 0.1/99.9 percentiles."""
    vol = np.asarray(volume, dtype=np.float32)
    step = max(1, vol.size // 5_000_000)
    lo, hi = np.percentile(vol.ravel()[::step], [0.1, 99.9])
    if hi <= lo:
        return np.zeros(vol.shape, dtype=np.uint8)
    return (np.clip((vol - lo) / (hi - lo), 0, 1) * 255.0).astype(np.uint8)
