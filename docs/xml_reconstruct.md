# xml-reconstruct — reconstruct a Warp tomogram from its tilt-series XML

Part of [tomo_toolshed](../README.md). A NumPy/SciPy re-implementation of
WarpTools `ts_reconstruct` that follows the recipe of `TiltSeries.ReconstructFull`,
built so you can swap the **weighting** and **filtering** and make new versions of
a tomogram from the same aligned tilt series.

It reads Warp's own metadata (`.tomostar` + the per-tilt-series `.xml`, and
optionally `.settings`), reproduces Warp's projection geometry, CTF, and dose /
B-factor weighting, and reconstructs by CTF-weighted Fourier-slice insertion — the
same operations Warp performs, just in readable Python. This tool only *reads* the
Warp XML; it never modifies it.

> **Note on weight.** Unlike the other (file/CLI) tools in the toolshed, this one
> does real reconstruction work: it builds an in-plane `S×S × Z` Fourier volume and
> can need several GB of RAM on a full-size box (see "Memory", below). It adds no
> heavy or GPU dependency — just `pillow` for the preview PNG.

## Overview

The pipeline mirrors `ts_reconstruct`:

1. Load the per-tilt averages and Fourier-rescale them to the target pixel size.
2. Preprocess each tilt (subtract mean, rectangular border mask, high-pass,
   normalize, optional contrast invert).
3. Insert each tilt as a CTF-weighted central slice into a 3D Fourier volume
   (weighted CTF as the data term, `|unweighted CTF|` as the weight term, coverage
   capped at 1), then `data / max(weight, floor)` and inverse FFT.
4. Optionally deconvolve, then write `<name>.mrc`, a `<name>.png` central-slice
   preview, and `<name>_deconv.mrc` when `--deconv` is given.

## Installation

Install the whole toolshed (see the [root README](../README.md)):

```bash
git clone https://github.com/hamid13r/tomo_toolshed.git
cd tomo_toolshed
pip install -e .
```

## Usage

```bash
tomo_toolshed xml-reconstruct \
    --xml       VLP3x3_p03_ts_002_blended_frames.xml \
    --tomostar  VLP3x3_p03_ts_002_blended_frames.tomostar \
    --settings  warp_tiltseries.settings \
    --tilt-dir  warp_frameseries/average \
    --angpix    10 \
    --output    reconstruction \
    --deconv
```

The command is also available as `tomo_toolshed xml_reconstruct` (underscore
spelling). Every multi-word option accepts both a dash and an underscore spelling
(`--tilt-stack` / `--tilt_stack`).

### Tilt-image input (pick one)

- `--tilt-stack FILE` — a 3-D MRC/`.st` stack whose slices are ordered like the
  model's tilts (i.e. like the `.tomostar` rows / `.xml` `MoviePath` list).
- `--tilt-dir DIR` — a folder of per-tilt averages named `<movie-root>.mrc`,
  matched to the `.xml` `MoviePath` entries (Warp's `warp_frameseries/*.mrc`).

### Picking the tomogram pixel size and box

- `--angpix` is **the** knob for the output tomogram pixel size (exactly like
  `ts_reconstruct --angpix`). It sets the sampling and, with the box, the number of
  voxels.
- The box (in unbinned pixels) comes from `--settings` (`<Tomo>` DimensionsX/Y/Z)
  by default; pass `--dimensions X Y Z` to override, or omit `--settings` to fall
  back to the box stored in the `.xml`.
- The raw (unbinned) pixel size is read from `--settings` (`PixelSize`), the
  tilt-image MRC header, or `--raw-angpix`.

### Memory

Memory scales with the box: the engine builds an in-plane `S×S × Z` grid, where
`S ≈ pad_factor·max(Vx,Vy)` voxels and `Z ≈ tomogram thickness`. For an
`11664²×2400` px box, `--angpix 10` needs ~8 GB RAM; `--angpix 14` ~2–3 GB.
**Increase `--angpix` if you are memory-limited.**

## Options

| Option (aliases) | Default | Description |
|---|---|---|
| `--xml` | *(required)* | Per-tilt-series `.xml` (Warp metadata; the source of truth). |
| `--tomostar` | – | `.tomostar`; fallback for raw angles/dose when the `.xml` has none. |
| `--settings` | – | Warp `.settings`; supplies the raw pixel size and the `<Tomo>` box. |
| `--angpix` | *(required)* | Output tomogram pixel size (Å). |
| `--raw-angpix` / `--raw_angpix` | header/settings | Unbinned tilt pixel size (Å). |
| `--dimensions X Y Z` | – | Tomogram box in **unbinned pixels**; overrides `.settings`/`.xml`. |
| `--tilt-stack` / `--tilt_stack` | – | 3-D MRC/`.st` stack ordered like the tilts. |
| `--tilt-dir` / `--tilt_dir` | – | Folder of per-tilt averages `<root>.mrc`. |
| `--output` | `reconstruction` | Output folder. |
| `--deconv` | off | Also write a deconvolved `<name>_deconv.mrc`. |
| `--deconv-strength` / `--deconv_strength` | `1.0` | Deconvolution strength. |
| `--deconv-falloff` / `--deconv_falloff` | `1.0` | Deconvolution falloff. |
| `--deconv-highpass` / `--deconv_highpass` | `300.0` | Deconvolution high-pass (Å). |
| `--dont-invert` / `--dont_invert` | off | Keep original contrast (skip the sign flip). |
| `--dont-normalize` / `--dont_normalize` | off | Skip per-tilt mask/high-pass/normalize. |
| `--subvolume-size` / `--subvolume_size` | `64` | Sets the preprocessing high-pass cutoff. |
| `--subvolume-padding` / `--subvolume_padding` | `3.0` | Sets the preprocessing high-pass cutoff. |
| `--pad-factor` / `--pad_factor` | `1.15` | In-plane grid padding to hold the tilted footprint; raise if edges look clipped at high tilt. |
| `--float16` | off | Write 16-bit MRC like Warp. |
| `--dose-bfactor-scale` / `--dose_bfactor_scale` | Warp's 4 | Use a custom dose→B-factor factor instead of Warp's hard-coded 4. |

## Making new versions — where to edit

The whole point. Both hooks are small, isolated functions in the package:

- **Weighting** — `tomo_toolshed/xml_reconstruct/weighting.py`. `warp_weighting()`
  is the exact Warp model; copy it, edit the `Scale`/`Bfactor` lines, and pass it
  into `reconstruct(..., weighting_fn=...)`. From the CLI, `--dose-bfactor-scale N`
  swaps Warp's hard-coded dose→B-factor factor of 4 for `N`.
- **Filtering** — `tomo_toolshed/xml_reconstruct/filters.py`: `preprocess_tilt()`
  (per-tilt real-space band-pass/normalize/invert) and `deconvolve()`
  (post-reconstruction Fourier filter).

## What is faithful, and what is approximate

Reproduced from the Warp source: the geometry (`GetPositionInAllTilts` /
`GetAngleInAllTilts`, tilt/tilt-axis matrices, per-tilt shifts, warp/motion grids,
per-tilt defocus, inverted-angle handling), the CTF equation (`CTF.cs`), the
dose/B-factor weighting (`GetCTFsForOneParticle`), and the `ReconstructFull`
bookkeeping (phase-flip, `|CTF|` weight, coverage cap, weight floor).

Documented approximations (a bit-identical match needs Warp's CUDA gridding
kernel): the Fourier-insertion kernel is trilinear + sinc² de-apodization rather
than Warp's oversampled gridding kernel; the default **global engine** evaluates
the CTF at the tomogram-centre defocus per tilt rather than per padded sub-volume
(identical when the model grids are spatially trivial — no M refinement); and the
deconvolution approximates `GPU.DeconvolveCTF` from its strength/falloff/highpass
parameters.

## Known limitations

- **Small images are masked to zero.** The per-tilt preprocessing applies a
  rectangular border mask with a default 32-px border. Real tilt images are large
  so this is invisible, but an image smaller than ~64 px across is masked away
  entirely (its reconstruction comes out flat). Pass `--dont-normalize` to skip
  the mask on small/synthetic inputs.
- **RELION 4.x geometry conventions** are the target.
- **Validation against Warp** is done separately: run WarpTools
  `ts_reconstruct --angpix N` on the same series and compare (FSC / real-space
  correlation). Small differences are expected from the gridding kernel and the
  global-vs-subvolume CTF above. A built-in self-consistency check
  (`tests/xml_reconstruct/test_reconstruct.py::test_self_consistency_phantom_recovered`)
  forward-projects a phantom through the exact geometry and recovers it at
  correlation ≈0.9.
```
