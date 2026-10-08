# xml-reconstruct — reconstruct a tomogram from a Warp tilt-series XML

Part of [tomo_toolshed](../README.md). A NumPy/SciPy re-implementation of
WarpTools `ts_reconstruct` that follows the recipe of `TiltSeries.ReconstructFull`,
built so you can swap the **weighting** and **filtering** and make new versions of
a tomogram from the same aligned tilt series.

It reads Warp's own metadata (`.tomostar` + the per-tilt-series `.xml`, and
optionally `.settings`), reproduces Warp's projection geometry, CTF, and dose /
B-factor weighting, and reconstructs by CTF-weighted Fourier-slice insertion — the
same operations Warp performs, just in readable Python. This tool only *reads* the
Warp XML; it never modifies it.

Three reconstruction engines share the same Warp metadata, geometry and per-tilt
preprocessing (`--engine`):

| Engine | What it does | Extra requirements |
|---|---|---|
| `fourier` (default) | CTF-weighted Fourier-slice insertion, i.e. what Warp's `ts_reconstruct` does. Optional novaCTF-style Z-strip 3D-CTF (`--ctf3d-*`). | – |
| `etomo` | Hands the same alignment and tilts to IMOD's `tilt`: WBP, SIRT-like filter, or true SIRT. No CTF or dose model. | IMOD (`$IMOD_DIR` or `--imod-dir`) |
| `novactf` | novaCTF's real-space 3D-CTF back-projection, projecting every voxel through Warp's full geometry, local-motion grids included. | `numba` recommended (numpy fallback is much slower); `torch` with CUDA for `--novactf-device cuda` |

A companion command, [`compare-tomograms`](#compare-tomograms), scores
reconstructions against a reference (FSC, Pearson, power spectrum, flip search).

> **Note on weight.** Unlike the other (file/CLI) tools in the toolshed, this one
> does real reconstruction work: it builds an in-plane `S×S × Z` Fourier volume and
> can need several GB of RAM on a full-size box (see "Memory", below). The install
> adds only `pillow`. `numba`, `torch` and IMOD are optional and are used only
> when present: install them yourself if you want the faster or GPU `novactf`
> path, or the `etomo` engine.

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

Local motion (Warp's `GridMovementX/Y`) is applied by default in every engine;
`--no-local-motion` turns it off.

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

Other engines, same inputs:

```bash
# IMOD weighted back-projection / SIRT-like filter / true SIRT
tomo_toolshed xml-reconstruct ... --engine etomo --etomo-recon wbp
tomo_toolshed xml-reconstruct ... --engine etomo --etomo-recon fakesirt --etomo-sirt-iters 10

# novaCTF 3D-CTF on Warp geometry (CPU, or a CUDA GPU)
tomo_toolshed xml-reconstruct ... --engine novactf
tomo_toolshed xml-reconstruct ... --engine novactf --novactf-correction phaseflip --novactf-device cuda

# Fourier engine with novaCTF-style 3D-CTF strips and MotionCor3 dose weighting
tomo_toolshed xml-reconstruct ... --ctf3d-defocus-step 10 --dose-weighting motioncor3
```

`--threads N` sets the thread count for every engine (default: the CPUs this
process may use, so SLURM / `taskset` limits are respected).

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
| `--threads` | `0` (= available CPUs) | Threads for FFTs, numba kernels, thread pools and IMOD `tilt`. |
| `--engine` | `fourier` | `fourier`, `etomo` or `novactf` (see above). |
| `--deconv` | off | Also write a deconvolved `<name>_deconv.mrc`. |
| `--deconv-strength` / `--deconv_strength` | `1.0` | Deconvolution strength. |
| `--deconv-falloff` / `--deconv_falloff` | `1.0` | Deconvolution falloff. |
| `--deconv-highpass` / `--deconv_highpass` | `300.0` | Deconvolution high-pass (Å). |
| `--dont-invert` / `--dont_invert` | off | Keep original contrast (skip the sign flip). |
| `--dont-normalize` / `--dont_normalize` | off | Skip per-tilt mask/high-pass/normalize. |
| `--renorm-variance` | off | Force each tilt to unit std after the high-pass (the old default; over-boosts high tilts). |
| `--no-highpass` | off | Skip only the per-tilt high-pass; keep background subtraction and edge mask. |
| `--no-local-motion` | off | Ignore Warp's local-motion grids in every engine. |
| `--subvolume-size` / `--subvolume_size` | `64` | Sets the preprocessing high-pass cutoff. |
| `--subvolume-padding` / `--subvolume_padding` | `3.0` | Sets the preprocessing high-pass cutoff. |
| `--pad-factor` / `--pad_factor` | `1.15` | In-plane grid padding to hold the tilted footprint; raise if edges look clipped at high tilt. |
| `--weight-floor` | `0.01` | Minimum CTF-weight divisor (fourier engine). |
| `--float16` | off | Write 16-bit MRC like Warp. |
| `--ctf3d-defocus-step` | `0` (off) | Fourier engine: novaCTF-style Z-strip 3D-CTF, strip thickness in nm. Wins over `--ctf3d-num-strips`. |
| `--ctf3d-num-strips` | `0` (off) | Fourier engine: explicit strip count instead. |
| `--dose-bfactor-scale` / `--dose_bfactor_scale` | Warp's 4 | Use a custom dose→B-factor factor instead of Warp's hard-coded 4 (ignored with `--dose-weighting motioncor3`). |
| `--dose-weighting` | `warp` | `warp` (Bfactor = −dose·4) or `motioncor3` (Grant & Grigorieff critical-exposure curve). |

**`--engine etomo` only**

| Option | Default | Description |
|---|---|---|
| `--etomo-recon` | `wbp` | `wbp`, `fakesirt` (tilt `-FakeSIRTiterations`) or `sirt` (`-SIRTIterations`). |
| `--etomo-sirt-iters` | `10` | Iterations for `fakesirt`/`sirt`. |
| `--etomo-radial CUTOFF FALLOFF` | `0.35 0.035` | `tilt` RADIAL filter for wbp/fakesirt. |
| `--etomo-sirt-radial CUTOFF FALLOFF` | `0.40 0.035` | RADIAL filter for true SIRT. |
| `--etomo-xaxistilt-sign` | `1` | Sign mapping LevelAngleX onto XAXISTILT (A/B testing). |
| `--etomo-view-weight` | `none` | `warp` feeds the per-tilt amplitude scale (cos tilt) to `tilt` via its WeightFile. |
| `--etomo-no-local-motion` | off | Don't bake local motion into the tilts. |
| `--etomo-gpu` | `-1` (CPU) | `tilt` UseGPU: 0 = best GPU, N = GPU N. |
| `--etomo-workdir` | `<output>/etomo` | Where the IMOD project is written (re-runnable with `submfg tilt.com`). |
| `--imod-dir` | `$IMOD_DIR` or `/usr/local/IMOD` | IMOD installation. |

**`--engine novactf` only**

| Option | Default | Description |
|---|---|---|
| `--novactf-step` | `10` | Defocus step (nm) between CTF-corrected copies; `0` = 2-D CTF. |
| `--novactf-correction` | `multiplication` | `multiplication`, `phaseflip` or `none`. |
| `--novactf-radial CUTOFF FALLOFF` | `0.3 0.05` | novaCTF `-RADIAL`, cycles/pixel. |
| `--novactf-weighting` | `none` | `warp` also applies the `--dose-weighting` scale and dose filter. |
| `--novactf-no-local-motion` | off | Global alignment only. |
| `--novactf-no-astig` | off | Ignore per-tilt astigmatism. |
| `--novactf-device` | `cpu` | `cpu`, or a PyTorch device such as `cuda` / `cuda:1`. |
| `--novactf-geometry-step` | `16` | Voxels between exact geometry evaluations (trilinear in between). |

Options that an engine cannot use (e.g. `--deconv` with `etomo`) are reported
as ignored at the start of the run rather than dropped silently.

## Making new versions — where to edit

The whole point. Both hooks are small, isolated functions in the package:

- **Weighting** — `tomo_toolshed/xml_reconstruct/weighting.py`. `warp_weighting()`
  is the exact Warp model; copy it, edit the `Scale`/`Bfactor` lines, and pass it
  into `reconstruct(..., weighting_fn=...)`. From the CLI, `--dose-bfactor-scale N`
  swaps Warp's hard-coded dose→B-factor factor of 4 for `N`.

A second built-in scheme, `motioncor3_dose_weighting`, replaces Warp's linear
dose Bfactor (a single Gaussian falloff, `Bfactor = -dose*4`) with the
Grant & Grigorieff (2015) critical-exposure curve MotionCor3 applies to
frames (`Correct/GWeightFrame.cu:mGCalcWeight`), applied here per tilt via
each tilt's accumulated dose. Use it with `--dose-weighting motioncor3`.
Validated on `VLP3x3_p03_ts_002` (2026-08-28): Pearson correlation against
the Warp reference improved on every metric (specimen-band 0.080→0.090,
full-box 0.011→0.017) and the FSC crossing points were unchanged (0.5 @
~165 Å, 0.143 @ ~111 Å — still a real, valid curve). Note this comes from
*less* high-frequency power overall, not more: the curve's biggest deviation
from a flat B-factor is a *stronger* attenuation of high-dose (typically
high-tilt) images at high resolution, where those images carry mostly noise
rather than signal — it turns out to be more accurate there than Warp's own
flat linear model, not more permissive.

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

**Optional novaCTF-style 3D-CTF correction** (`--ctf3d-defocus-step` nm, or
`--ctf3d-num-strips`) narrows the `global` engine's tomogram-centre-defocus gap
above, the way novaCTF (Turoňová et al. 2017, doi:10.1016/j.jsb.2017.07.007)
does: split the tomogram thickness into `N` Z-strips, reconstruct each with
its own per-tilt defocus, and stitch the correctly-focused Z-slab from each
into the final volume. novaCTF's own C++ source
(`ctf3d.cpp:generateFocusGrid`/`computeOneRow`) gets this from an explicit
real-space back-projection loop that picks, *per contributing tilt*, whichever
of `N` pre-corrected projections has the defocus closest to a given voxel's
depth along *that tilt's own* beam — since the same 3D point sits at a
different depth along the beam for each tilt angle. This engine instead does
joint multi-tilt Fourier-slice insertion (the central-slice theorem: one
rotated 2D FT populates every Z at once), so there's no per-voxel-per-tilt
term left to select between afterward — each strip gets one representative
defocus applied to a tilt's *entire* footprint, not varied further across X
the way novaCTF's own per-tilt geometry does. That's the same category of
approximation as the tomogram-centre-defocus limitation above, just at `N`
Z-bands instead of one. What *is* exact: `geometry.positions_in_all_tilts`
already computes the true per-tilt defocus at any 3D point via the full
rotated ray (novaCTF's own defocus-file step only approximates this with a
flat, angle-independent nm shift per strip), so evaluating it at each strip's
Z-shifted center gives genuinely correct per-tilt, per-strip defocus values
for free. Cost is `~N`x a single reconstruction pass (preprocessing is shared
across strips; only the CTF-weighted insertion repeats). See
`reconstruct.reconstruct_novactf` for the implementation.

Validated against real data (2026-08-28, `VLP3x3_p03_ts_002`, 21.16 Å/px):
per-tilt preprocessing used to force every tilt to unit variance after the
high-pass (`preprocess_tilt`'s old default). That disproportionately boosts
high-tilt images, which have the least real high-frequency signal left after
the high-pass and are mostly noise there — it was the dominant cause of a
5-10 dB excess in the reconstructed power spectrum vs Warp's own
`ts_reconstruct` output. Fixed by making that renormalization opt-in
(`--renorm-variance`, off by default); with it off, the power spectrum now
tracks Warp's closely from DC out to ~125 Å, with only a small residual
high-frequency excess left (consistent with the trilinear-vs-gridding-kernel
difference above). See `tomo_eval/compare_fixed/` for the comparison that
found this (not tracked in git; regenerate with `tomo_toolshed compare-tomograms`).


## compare-tomograms

Scores one or more volumes against a reference, e.g. this tool's output vs
WarpTools `ts_reconstruct`: robust stats, an axis-flip and sign search,
Pearson correlation (full box, central crop, specimen band), FSC with 0.5 and
0.143 crossings, and radially averaged power spectra. It writes `summary.json`,
`fsc.png`, `power_spectrum.png` and two central-slice comparison PNGs.

```bash
tomo_toolshed compare-tomograms \
    --reference warp_tiltseries/reconstruction/TS_01_21.16Apx.mrc \
    --recon fourier=recon/TS_01.mrc \
    --recon etomo=recon_etomo/TS_01.mrc \
    --output compare_21
```

| Option | Default | Description |
|---|---|---|
| `--reference` | *(required)* | Reference volume. |
| `--reference-label` | `warp` | Label for the reference in plots. |
| `--recon NAME=PATH` | *(required, repeatable)* | Volume to compare; must match the reference's shape. |
| `--output` | `compare` | Output folder. |
| `--angpix` | reference header | Voxel size override (Å). |
| `--crop-frac` | `0.1` | Fraction trimmed off each side for real-space stats. |
| `--no-flip-search` | off | Assume identity orientation. |

Also available as `tomo_toolshed compare_tomograms`. Needs `matplotlib`
(already a toolshed dependency).

## Engines in detail, and validation runs

These notes come from the standalone `warp_recont_test` repo, with commands
rewritten for `tomo_toolshed`. Data paths and dates refer to the original runs.

### IMOD/etomo back-projection (`--engine etomo`)

Everything above reconstructs by CTF-weighted Fourier-slice insertion, which is
what Warp does. `--engine etomo` instead hands the tilt series to IMOD's `tilt`,
which is a genuinely different algorithm: real-space weighted back-projection
with a 1-D radial filter per projection line. The alignment and the per-tilt
preprocessing are shared with the Fourier engine, so the back-projection is the
only thing that changes.

```bash
# weighted back-projection (IMOD's own RADIAL 0.35 0.035)
tomo_toolshed xml-reconstruct ... --engine etomo --etomo-recon wbp

# the SIRT-like filter (tilt -FakeSIRTiterations): a radial filter analytically
# equivalent to N SIRT iterations, at WBP cost
tomo_toolshed xml-reconstruct ... --engine etomo --etomo-recon fakesirt --etomo-sirt-iters 10

# true iterative SIRT (tilt -SIRTIterations)
tomo_toolshed xml-reconstruct ... --engine etomo --etomo-recon sirt --etomo-sirt-iters 10

# any of the above, plus the cos(tilt) per-view amplitude weighting the
# fourier engine applies (fed to tilt through its -WeightFile)
tomo_toolshed xml-reconstruct ... --engine etomo --etomo-recon wbp --etomo-view-weight warp
```

The IMOD project (stack, `.xf`, `.tlt`, `newst.com`, `tilt.com`, the aligned
stack and `tilt`'s raw output) is left in `<output>/etomo/`, so a run can be
inspected, edited and re-run by hand with `submfg tilt.com` or opened in etomo.

#### The geometry comes from Warp

`geometry.tilt_matrix` builds, per tilt *t*,

    R_t = Euler(0, angle_t + LevelAngleY, -AxisAngle_t) @ RotateX(LevelAngleX)

and since `euler(0, b, g) == RotZ(-g) @ RotY(-b)`, that is exactly

    R_t = RotZ(AxisAngle_t) @ RotY(-(angle_t + LevelAngleY)) @ RotX(LevelAngleX)

IMOD splits the same geometry differently: the in-plane rotation is baked into
the *aligned stack* by `newstack`, and `tilt` then tilts about the (now
vertical) Y axis. So peeling the leading `RotZ` off into a `.xf` leaves
precisely IMOD's model:

| IMOD | from Warp |
|---|---|
| `.xf` 2x2 | `RotZ2D(-AxisAngle_t)` |
| `.xf` shift | `-RotZ2D(-AxisAngle_t) @ (projected tomogram centre - image centre)` |
| `.tlt` | `angle_t + LevelAngleY` |
| `XAXISTILT` | `LevelAngleX` |

This is checked, not assumed. Warp's alignment for this project was *imported
from* etomo (`warp_tiltseries/tiltstack/<series>/`), so the conversion can be
diffed against IMOD's own files. For `VLP3x3_p03_ts_002`:

- **`.xf` 2x2** — generated `0.1643666 0.9863993 -0.9863993 0.1643666` vs IMOD's
  own `0.1643666 0.9863998 -0.9863998 0.1643666`: identical to 6 decimals.
- **`.xf` shift** — ours is written at 21.16 A/px, IMOD's at 6.348 A/px, so the
  ratio should be 3.333: `(-1.853, -7.892) * 3.333 = (-6.177, -26.307)` vs
  IMOD's `(-6.169, -26.272)`; likewise tilts 2-4, all within ~0.5%.
- **`.tlt`** — ours is IMOD's minus 3.62 deg on every tilt, i.e. exactly
  `LevelAngleY = -3.6172`, as intended.

The shift is derived from `geometry.positions_in_all_tilts` rather than from
`AxisOffsetX/Y` directly, which is strictly more complete: it also picks up the
volume-warp and local-motion grids, and it puts the tomogram centre at the
aligned-stack centre exactly where `_insert_all_tilts` puts it via
`_extract_patch`.

`XAXISTILT` is the one piece whose *sign* convention is IMOD's rather than
derivable, so it is exposed as `--etomo-xaxistilt-sign` and settled empirically:
in `test_etomo` in `tests/xml_reconstruct/validate.py` the two signs are identical at `LevelAngleX = 0` and
`+1` wins by a margin that grows monotonically with `|LevelAngleX|` (0.0000 at
0 deg, 0.0655 at 6 deg, 0.1905 at 12 deg). `+1` is the default.

#### What "weighted" means in WBP -- it is not cos(tilt)

Worth being precise, because it is a real asymmetry between the two engines.
`tilt`'s back-projection is "weighted" in two senses, **neither of which is
cos(tilt)**:

- the **radial R-weighting** ramp filter applied per projection line in
  frequency space (`RADIAL cutoff falloff`, `--etomo-radial`), and
- **`-DENSWEIGHT`** (on by default, 2 intervals each side), a per-view weight
  proportional to the *local average tilt increment* between views. On this
  series the increments span 2.88-3.05 deg (5.7%), so it is very nearly uniform.

`-COSINTERP` is unrelated despite the name: it is cosine *stretching*, a
back-projection speed optimization that pre-stretches each input line by
1/cos(tilt) so it lands in register with the output planes. It is disabled on the
GPU and is not a weighting.

The Fourier engine, by contrast, does apply `Scale = cos(tilt)` per tilt
(`weighting.warp_weighting`, from Warp's `GetCTFsForOneParticle`) -- on this
series the +-59 deg views enter at 51% of the flattest view's amplitude.
`--etomo-view-weight warp` closes that gap by feeding the active weighting
scheme's per-tilt amplitude scale to `tilt` through its `-WeightFile`. Only the
*scalar* per-view factor can travel that way: the dose exposure filter is a
frequency-dependent B-factor envelope, which has no one-number-per-view
representation and is therefore not transferred.

#### Local motion (Warp's `GridMovementX/Y`), and two conversion fixes

`tilt` only takes a global alignment, but Warp's local motion is a per-tilt
2-D shift looked up at a point's *globally projected image position*, not at
its depth. So it is exactly a per-tilt image warp: `reconstruct.bake_local_motion`
resamples each tilt so that `I'(q) = I(q - mv(q))`, and sampling `I'` with the
global alignment then gives Warp's local geometry at every depth. It is on by
default whenever the grids vary across the image (`--etomo-no-local-motion`
turns it off). The `.xf` then carries the global alignment only, so the
centre's local shift is not applied twice. Cost: one extra interpolation.

Getting this to match Warp also needed two fixes to the global conversion.
Both were invisible on `VLP3x3_p03_ts_002` (square tilts, ~80 deg axis):

- **Aligned-stack size.** newstack wrote the aligned stack at the raw tilt size.
  With a tilt axis near +-90 deg (HRR021_2: -94.5 deg, 720x512 tilts), the
  720-px tomogram length ends up vertical, and the volume was cut to 104 of 720
  rows. `etomo.aligned_stack_size` now uses the rotated tilt's bounding box.
- **Half-pixel centres.** Warp centres an N-pixel axis at index N/2; IMOD's
  `.xf` and `tilt` use (N-1)/2. The `.xf` now rotates about the IMOD centre
  and sends the tomogram centre to aligned index N/2. The old version was off
  by an axis-angle-dependent ~1 voxel (1.04 in X on HRR021_2), which on its
  own halves the FSC against Warp from ~60 A on.

HRR021_2_S02_L02_ts_003, 13.28 A/px, WBP, against Warp's reconstruction of
the current `.xml`:

| variant | Pearson (central 80%) | FSC 0.5 |
|---|---|---|
| before the fixes, local motion off | 0.085 | 98 A |
| local motion off | 0.225 | 79 A |
| local motion baked in | **0.681** | **30.3 A** |
| `--engine novactf --novactf-correction none` (same geometry, no CTF) | 0.707 | 30.1 A |

Block-wise sub-voxel shifts between the etomo and novactf volumes are now
<= 0.18 voxel everywhere. The `VLP3x3_p03_ts_002` numbers above were measured
before the half-pixel fix and have not been re-run since.

#### What `tilt` still does not do

`tilt` has no notion of a CTF. Unlike the Fourier engine, the etomo path applies
**no phase flip, no |CTF| weighting and no dose B-factor**. `--deconv`,
`--dose-bfactor-scale` and the `--ctf3d_*` options are therefore ignored with
`--engine etomo`, as is the B-factor half of `--dose-weighting` (the CLI says so
rather than dropping them silently). That is a real difference between these
entries and the rest of the list, not an oversight. What *is* shared is
`reconstruct._preprocess_tilts`, so the tilt data going in is bit-identical to
what the Fourier engine inserts.

Also note `tilt`'s output densities are on its own scale (no `SCALE` is applied,
to keep them linear): the WBP volume has std ~4.0 where Warp's reference has
0.0133. FSC and Pearson are scale-invariant, so this affects only display
contrast.

#### Validated against the Warp reference

`VLP3x3_p03_ts_002`, 21.16 A/px, 584x584x120, vs
`warp_tiltseries/reconstruction/..._21.16Apx.mrc`:

| variant | specimen-band Pearson | FSC 0.5 | FSC 0.143 | wall clock |
|---|---|---|---|---|
| `fourier` (this repo's engine, current default) | 0.080 | 165 A | 111 A | ~5 min |
| `etomo --etomo-recon wbp` | **0.480** | **71.3 A** | **45.4 A** | 58 s |
| `etomo --etomo-recon wbp --etomo-view-weight warp` | **0.486** | **68.5 A** | **45.2 A** | 58 s |
| `etomo --etomo-recon fakesirt` (10) | **0.646** | **66.8 A** | **45.4 A** | 24 s |
| `etomo --etomo-recon fakesirt --etomo-view-weight warp` (10) | **0.654** | **65.5 A** | **45.3 A** | 24 s |
| `etomo --etomo-recon sirt` (10) | **0.604** | (see below) | **44.5 A** | ~2 min |

Adding the cos(tilt) per-view weighting is a small but perfectly consistent
improvement -- all three Pearson measures and both FSC crossings move the right
way, for both `wbp` and `fakesirt`. The magnitude is modest because cos(tilt) is
a smooth monotonic per-view amplitude taper and the FSC is normalized per shell.

All three agree with Warp's own `ts_reconstruct` output *substantially better
than this repo's Fourier engine does* -- 6-8x the real-space correlation, and an
FSC 0.143 crossing at 45 A against the Fourier engine's 111 A. The likely reason
is the `mode="global"` approximation documented above: Warp really reconstructs
per padded sub-volume, and a real-space back-projection off the same alignment
apparently lands closer to that than one Fourier-cropped patch per tilt does.
That makes these useful as a cross-check on the Fourier engine, not just as
extra entries in the list.

True SIRT's 0.5 crossing is not meaningful: one very-low-frequency shell (2746
A, few voxels, statistically unstable) dips to 0.303 and drags the *first*
downward crossing out to ~2977 A, while the neighbouring shells sit at 0.87 and
0.76. Its 0.143 crossing (44.5 A) is the number to read.

Fixing one reporting bug was needed to see any of this. `compare._crossing_resolution`
searched for the first shell below the threshold starting at the DC shell -- but
the DC shell holds a single Fourier component (the volume mean), so its
"correlation" is exactly +-1 depending only on the relative sign of the two
volumes. Every sign-flipped volume therefore reported *no crossing at all*, which
looks exactly like a collapsed FSC. The DC shell is now skipped; the previously
reported `fourier` numbers (165 A / 111 A) are unchanged by it.

### Real-space novaCTF on Warp's geometry (`--engine novactf`)

novaCTF's 3D-CTF weighted back-projection (Turoňová et al. 2017), with every
voxel projected through Warp's *full* geometry -- including the
`GridMovementX/Y` local-motion grids, which neither the novaCTF binary (global
`.xf`/`.tlt` only, and it forces `XAXISTILT` to 0) nor IMOD `tilt` can use.
For each tilt, the image is CTF-corrected at defocus steps of `--novactf-step`
nm; each voxel then takes the copy whose defocus matches *its own depth along
that tilt's beam* (novaCTF's `generateFocusGrid` / `computeOneRow`), at the
image position `geometry.positions_one_tilt` gives it. See
`xml_reconstruct/novactf.py` for the full list of differences from the binary.

```bash
# novaCTF defaults: multiplication, 10 nm steps, RADIAL 0.3 0.05, local motion on
tomo_toolshed xml-reconstruct ... --engine novactf
# ablations
tomo_toolshed xml-reconstruct ... --engine novactf --novactf-no-local-motion   # global alignment only
tomo_toolshed xml-reconstruct ... --engine novactf --novactf-step 0            # 2-D CTF (one defocus/tilt)
tomo_toolshed xml-reconstruct ... --engine novactf --novactf-correction phaseflip
tomo_toolshed xml-reconstruct ... --engine novactf --novactf-weighting warp    # + cos(tilt) & dose filter
```

Needs `numba` for speed; falls back to a much slower numpy path without it.
On 64 cores, HRR021_2_S02_L02_ts_003 end to end (load, reconstruct, write):
~13 s at 13.28 A/px (512x720x300) and ~36 s at 6.64 A/px (1024x1440x600,
~11 GB RAM), down from 65 s / 240 s before parallelizing. Output is unchanged
(correlation 1.0 with the earlier volumes, max diff ~3e-7). What is parallel:
the CTF of every defocus copy (fused numba kernel; the defocus-independent
phase terms once per tilt), batched multi-threaded inverse FFTs, threaded
geometry and tilt preprocessing, the numba back-projection, and preparation
of tilt j+1 overlapping the back-projection of tilt j. The back-projection
(memory-bound plane gathers) is ~half of what is left.

**GPU back end** (`--novactf-device cuda`, or `cuda:N`): the FFTs, CTF
copies and back-projection run on the GPU through PyTorch, with a fused
`numba.cuda` back-projection kernel (float32 throughout: RTX-class GPUs run
fp64 at 1/32 rate). The coarse geometry stays on the CPU, prepared for tilt
j+1 while the GPU works on tilt j. Needs torch with CUDA; `numba.cuda` is
optional (without it a pure-torch kernel is used, ~8x slower). Same volume
as the CPU path: correlation 0.999999 on HRR021_2_S02_L02_ts_003, and a
GPU-vs-CPU check in `tests/xml_reconstruct/validate.py`. Measured end to end on one RTX A5000
(`tomoannotator` env):

| | CPU (64 cores) | GPU |
|---|---|---|
| 6.64 A/px, 1024x1440x600 | 31 s | **18 s** (~5 GB GPU memory) |
| 13.28 A/px, 512x720x300 | 11 s | **8 s** |

At 6.64 A/px about 8 s of the GPU run is loading, preprocessing, the torch
import and writing the 3.5 GB volume. On the GPU the reconstruction itself
(~10 s) is limited by the CPU-side geometry. The default
`--novactf-geometry-step` is now 16 (was 8): max 0.003 px position error,
volume correlation 0.999999 with the old default.

`--threads N` sets the thread count for all engines (FFTs, numba kernels,
thread pools, and IMOD `tilt` via OMP_NUM_THREADS; `tomo_toolshed.xml_reconstruct.set_threads`
from Python). The default is the CPUs the process may run on
(`os.sched_getaffinity`), so under SLURM or `taskset` it uses the allocation,
not every core of the node. Because tilt j+1 is prepared while tilt j
back-projects, usage can briefly exceed N (measured: 851% average CPU at
`--threads 8`).

#### Validated against Warp (HRR021_2_S02_L02_ts_003, 13.28 A/px)

Reference: `WarpTools ts_reconstruct --dont-invert` run on the *current*
`.xml` (2026-09-29). The reference in `warp_tiltseries/reconstruction/`
(2026-05-15) predates the local-motion grids that were later written into the
`.xml` (by the `miss-alignment` refinement; see `*_alignment_loss.json`), and
agrees with the current `.xml`'s own Warp reconstruction at only r = 0.08 --
compare against a freshly regenerated reference, not that one.

| variant | Pearson (central 80%) | FSC 0.5 | FSC 0.143 |
|---|---|---|---|
| old (May) Warp reference | 0.083 | 145 A | 88 A |
| `--novactf-no-local-motion` | 0.108 | 90 A | 51 A |
| default (multiplication, 3-D CTF) | 0.673 | 31.1 A | 26.6 A (Nyquist) |
| `--novactf-step 0` (2-D CTF) | -- | 31.1 A | 29.5 A |
| `--novactf-weighting warp` | 0.686 | 31.2 A | 26.6 A (Nyquist) |
| `--novactf-correction none` | 0.707 | 30.1 A | 29.3 A |

The local-motion grids shift content by up to ~15 px here and dominate every
other difference. Their sign is also checked without any reference: an
even/odd-tilt half-set FSC is highest with the grids applied, lower without
them and lowest with them negated (200-100 A band: 0.260 / 0.256 / 0.247).
At this pixel size and 4.2 um defocus, Nyquist (26.6 A) sits just past the
first CTF zero (~29 A), so the CTF variants can only differ in the last few
shells; 3-D vs 2-D CTF needs a finer pixel size to show.

#### 3-D vs 2-D CTF (same series, 6.64 A/px, Nyquist 13.3 A)

Against Warp's own 6.64 A/px reconstruction of the current `.xml`:

| variant | Pearson (central 80%) | FSC 0.5 |
|---|---|---|
| multiplication, `--novactf-step 0` (2-D) | 0.593 | 29.5 A |
| multiplication, 10 nm steps (3-D, default) | 0.598 | 17.2 A |
| phaseflip, `--novactf-step 0` (2-D) | 0.608 | 29.4 A |
| phaseflip, 10 nm steps (3-D) | **0.635** | **16.8 A** |

With one defocus per tilt, the FSC drops deeply at every CTF zero (~29,
20.5, 16.6, 14.5 A): the zeros sit in the wrong place for every voxel away
from the tomogram's central depth. The 3-D variants fill those dips, and
phaseflip + 3-D almost removes them, tracking Warp (which re-evaluates the
CTF per sub-volume) out to Nyquist. Phaseflip matches Warp better than
multiplication because Warp itself phase-flips. The 0.143 crossings are not
quoted: only the shells at DC fall below 0.143. An even/odd-tilt half-set
FSC cannot separate 2-D from 3-D here: both halves are pure noise past ~60 A.
Runtime at the time: ~4 min (3-D) / ~1.5 min (2-D); now ~36 s for 3-D (see above).

### Reproducing WarpTools itself (the default `fourier` engine), 2026-09-29

Two changes make the default engine track `ts_reconstruct` closely:

- **Sub-pixel patch shift sign fix** (`reconstruct._extract_patch`). After the
  integer crop, a tilt's projected centre sits at patch index `half + r`; the
  phase ramp moved it to `half + 2r` instead of `half`. Every tilt was
  therefore off-centre by its own fractional position (0-1 px): the tomogram
  was blurred and ~1 voxel off. This bug, not the `global`-vs-sub-volume
  architecture, is the most likely cause of the old gap between this engine
  and the etomo engine / Warp on `VLP3x3_p03_ts_002` (0.080 vs 0.48). Those
  numbers, and the 2026-08-28 conclusions drawn from them
  (`summary_0828.md`), predate the fix and should be re-run.
- **Local motion**, baked into the tilts exactly as for the etomo engine
  (`reconstruct.bake_local_motion`; `--no-local-motion` disables it).

HRR021_2_S02_L02_ts_003, 13.28 A/px, `--dont-invert`, against `ts_reconstruct`
on the current `.xml`:

| variant | Pearson (central 80%) | FSC 0.5 | FSC 0.143 |
|---|---|---|---|
| before the fix, local motion on | 0.148 | 85 A | 55 A |
| **fixed, local motion on (default)** | **0.817** | **29.6 A** | **26.6 A (Nyquist)** |
| fixed, `--no-local-motion` | 0.232 | 80 A | 46 A |

Block-wise sub-voxel offsets to Warp's volume: <= 0.1 voxel. The phantom
self-test for this engine rose from 0.874 to 0.907.

## Known limitations

- **Small images are masked to zero.** The per-tilt preprocessing applies a
  rectangular border mask with a default 32-px border. Real tilt images are large
  so this is invisible, but an image smaller than ~64 px across is masked away
  entirely (its reconstruction comes out flat). Pass `--dont-normalize` to skip
  the mask on small/synthetic inputs.
- **RELION 4.x geometry conventions** are the target.
- **`--engine etomo` has no CTF or dose model** (IMOD `tilt` limitation), and
  its densities are on `tilt`'s own scale.
- **Validation against Warp** is done separately: run WarpTools
  `ts_reconstruct --angpix N` on the same series and compare (FSC / real-space
  correlation). Small differences are expected from the gridding kernel and the
  global-vs-subvolume CTF above. A built-in self-consistency check
  (`tests/xml_reconstruct/test_reconstruct.py::test_self_consistency_phantom_recovered`)
  forward-projects a phantom through the exact geometry and recovers it at
  correlation ≈0.9. `tests/xml_reconstruct/validate.py` runs the same phantom
  through the etomo engine (needs IMOD; this checks the Warp→IMOD conversion)
  and the novactf engine (plus a GPU-vs-CPU check when torch + CUDA are
  available). Run it directly with `python tests/xml_reconstruct/validate.py`,
  or through pytest (`test_engines.py`).
