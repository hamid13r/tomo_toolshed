"""
Compare one or more reconstructions against a reference tomogram (e.g. this
engine's output vs WarpTools `ts_reconstruct`): real-space stats, an
axis-flip sanity check, Fourier Shell Correlation, and radially-averaged
power spectra. Writes PNGs + a JSON summary; prints a text report.

Example
-------
    tomo_toolshed compare-tomograms \
        --reference warp_tiltseries/reconstruction/TS_01_21.16Apx.mrc \
        --recon mine_plain=recon_mine_21/TS_01.mrc \
        --recon mine_deconv=recon_mine_21/TS_01_deconv.mrc \
        --output compare_21
"""
from __future__ import annotations
import json
import os
from types import SimpleNamespace

import click
import numpy as np

from .mrc_io import read_mrc
from .compare import (
    robust_stats, find_best_orientation, apply_flip, pearson, central_crop,
    specimen_band_pearson, fourier_shell_correlation, radial_power_spectrum,
)

# validated categorical slots (dataviz skill palette, light mode)
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # blue, orange, aqua, yellow
REF_COLOR = "#4a3aa7"  # violet, reserved for the reference series


def _parse_recon(ctx, param, values):
    out = []
    for s in values:
        if "=" not in s:
            raise click.BadParameter("expects NAME=PATH", ctx=ctx, param=param)
        out.append(tuple(s.split("=", 1)))
    return out


@click.command(name="compare-tomograms")
@click.option("--reference", required=True,
              type=click.Path(exists=True, dir_okay=False),
              help="reference volume (e.g. Warp's ts_reconstruct output)")
@click.option("--reference-label", "--reference_label", "reference_label",
              default="warp", show_default=True)
@click.option("--recon", "recon", multiple=True, required=True,
              callback=_parse_recon, metavar="NAME=PATH",
              help="volume to compare against --reference (repeatable)")
@click.option("--output", default="compare", show_default=True,
              help="output directory")
@click.option("--angpix", type=float, default=None,
              help="override voxel size (A); default reads it from the reference header")
@click.option("--crop-frac", "--crop_frac", "crop_frac", type=float, default=0.1,
              show_default=True,
              help="fraction trimmed off each side per axis for real-space "
                   "stats/correlation, to keep missing-wedge edge shadows out")
@click.option("--no-flip-search", "--no_flip_search", "no_flip_search", is_flag=True,
              help="skip the axis-flip search and assume identity orientation")
def compare_tomograms(reference, reference_label, recon, output, angpix,
                      crop_frac, no_flip_search):
    """Compare reconstructions against a reference tomogram (FSC, Pearson,
    power spectrum, flip search)."""
    args = SimpleNamespace(reference=reference, reference_label=reference_label,
                           recon=recon, output=output, angpix=angpix,
                           crop_frac=crop_frac, no_flip_search=no_flip_search)
    os.makedirs(args.output, exist_ok=True)
    ref, ref_angpix = read_mrc(args.reference)
    angpix = args.angpix or ref_angpix
    print(f"reference: {args.reference}  shape={ref.shape}  angpix={angpix:.4f} A")
    print(f"reference stats: {robust_stats(ref)}")

    summary = {"reference": args.reference, "angpix": angpix,
               "reference_stats": robust_stats(ref), "recons": {}}

    fsc_curves = {}
    power_curves = {args.reference_label: radial_power_spectrum(ref, angpix)}
    recon_vols = {}

    for name, path in args.recon:
        vol, vs = read_mrc(path)
        if vol.shape != ref.shape:
            raise click.ClickException(
                f"shape mismatch: reference {ref.shape} vs {name} {vol.shape} "
                f"({path}) -- resample/crop before comparing")

        flip_code, sign = (1, 1, 1), 1
        if not args.no_flip_search:
            flip_code, sign, table = find_best_orientation(ref, vol)
            id_corr = table[((1, 1, 1), 1)]
            best_corr = table[(flip_code, sign)]
            if flip_code != (1, 1, 1) or sign != 1:
                print(f"[{name}] best orientation vs reference: flip={flip_code} sign={sign:+d} "
                      f"(specimen-ROI corr {best_corr:.4f} vs identity/no-flip {id_corr:.4f})")
            else:
                print(f"[{name}] identity orientation, no sign flip (specimen-ROI corr {id_corr:.4f})")
            vol = apply_flip(vol, flip_code) * sign

        stats = robust_stats(vol)
        ref_c = central_crop(ref, args.crop_frac)
        vol_c = central_crop(vol, args.crop_frac)
        corr_full = pearson(ref, vol)
        corr_central = pearson(ref_c, vol_c)
        corr_band = specimen_band_pearson(ref, vol)
        print(f"[{name}] stats: {stats}")
        print(f"[{name}] Pearson r  full-box={corr_full:.4f}  central-{int((1-2*args.crop_frac)*100)}%={corr_central:.4f}  specimen-band={corr_band:.4f}")

        fsc = fourier_shell_correlation(ref, vol, angpix)
        print(f"[{name}] FSC resolution  0.5={fsc['resolution_0.5']}  0.143={fsc['resolution_0.143']}")

        fsc_curves[name] = fsc
        power_curves[name] = radial_power_spectrum(vol, angpix)
        recon_vols[name] = vol

        summary["recons"][name] = {
            "path": path, "flip_code": flip_code, "sign": sign,
            "stats": stats, "pearson_full": corr_full, "pearson_central": corr_central,
            "pearson_specimen_band": corr_band,
            "fsc_resolution_0.5": fsc["resolution_0.5"],
            "fsc_resolution_0.143": fsc["resolution_0.143"],
        }

    with open(os.path.join(args.output, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=float)
    print("wrote", os.path.join(args.output, "summary.json"))

    _plot_fsc(fsc_curves, args.output)
    _plot_power_spectrum(power_curves, args.reference_label, args.output)
    _slice_comparison(ref, recon_vols, args.reference_label, args.output)


def _plot_fsc(fsc_curves, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=150)
    for i, (name, fsc) in enumerate(fsc_curves.items()):
        ax.plot(fsc["freq"], fsc["fsc"], color=COLORS[i % len(COLORS)], lw=2, label=name)
    ax.axhline(0.5, color="#8a8a86", lw=1, ls="--")
    ax.axhline(0.143, color="#8a8a86", lw=1, ls=":")
    ax.text(ax.get_xlim()[1], 0.5, " 0.5", va="center", ha="left", color="#5a5a56", fontsize=8)
    ax.text(ax.get_xlim()[1], 0.143, " 0.143", va="center", ha="left", color="#5a5a56", fontsize=8)
    ax.set_xlabel("Spatial frequency (1/Å)")
    ax.set_ylabel("FSC")
    ax.set_ylim(-0.05, 1.02)
    ax.set_title("Fourier Shell Correlation vs reference")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, color="#e7e6e0", lw=0.7)
    ax.set_axisbelow(True)
    ax.legend(frameon=False)
    fig.tight_layout()
    path = os.path.join(outdir, "fsc.png")
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path)


def _plot_power_spectrum(power_curves, ref_label, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=150)
    colors = {ref_label: REF_COLOR}
    others = [n for n in power_curves if n != ref_label]
    for i, n in enumerate(others):
        colors[n] = COLORS[i % len(COLORS)]
    for name, (freq, power) in power_curves.items():
        y = 10 * np.log10(power + 1e-30)
        ax.plot(freq, y, color=colors[name], lw=2, label=name)
    ax.set_xlabel("Spatial frequency (1/Å)")
    ax.set_ylabel("Power (dB, arbitrary ref.)")
    ax.set_title("Radially-averaged power spectrum")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, color="#e7e6e0", lw=0.7)
    ax.set_axisbelow(True)
    ax.legend(frameon=False)
    fig.tight_layout()
    path = os.path.join(outdir, "power_spectrum.png")
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path)


def _matched_slice(vol, lo, hi):
    sl = vol[vol.shape[0] // 2].astype(np.float32)
    return np.clip((sl - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)


def _slice_comparison(ref, recons, ref_label, outdir):
    """Central-slice panels for reference + each recon, all displayed on the
    SAME absolute intensity range (reference mean +/- 3 sigma, computed over
    its central region) so a genuine contrast difference is visible rather
    than hidden by independent per-panel rescaling."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ref_c = central_crop(ref, 0.1)
    m, sd = float(ref_c.mean()), float(ref_c.std()) or 1.0
    lo, hi = m - 3 * sd, m + 3 * sd

    names = [ref_label] + list(recons.keys())
    vols = [ref] + list(recons.values())
    fig, axes = plt.subplots(1, len(names), figsize=(4.2 * len(names), 4.6), dpi=150)
    if len(names) == 1:
        axes = [axes]
    for ax, name, vol in zip(axes, names, vols):
        img = _matched_slice(vol, lo, hi)
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax.set_title(name, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"Central slice, matched contrast (range = {ref_label} mean±3σ)", fontsize=10)
    fig.tight_layout()
    path = os.path.join(outdir, "slice_comparison_matched.png")
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path)

    # a second panel where each volume is independently z-scored, to compare
    # *relative* structural contrast irrespective of absolute scale
    fig, axes = plt.subplots(1, len(names), figsize=(4.2 * len(names), 4.6), dpi=150)
    if len(names) == 1:
        axes = [axes]
    for ax, name, vol in zip(axes, names, vols):
        vc = central_crop(vol, 0.1)
        vm, vsd = float(vc.mean()), float(vc.std()) or 1.0
        img = _matched_slice(vol, vm - 3 * vsd, vm + 3 * vsd)
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax.set_title(name, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Central slice, each independently normalized (own mean±3σ)", fontsize=10)
    fig.tight_layout()
    path = os.path.join(outdir, "slice_comparison_own_norm.png")
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path)


if __name__ == "__main__":
    compare_tomograms()
