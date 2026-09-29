# warp_recon — a hackable re-implementation of WarpTools `ts_reconstruct`

A NumPy/SciPy reconstruction of Warp/WarpTools tomograms that follows the exact
recipe of `TiltSeries.ReconstructFull`, built so you can swap the **weighting**
and **filtering** and make new versions of a tomogram from the same aligned tilt
series.

It reads Warp's own metadata (`.tomostar` + the per-tilt-series `.xml`, and
optionally `.settings`), reproduces Warp's projection geometry, CTF, and dose /
B-factor weighting, and reconstructs by CTF-weighted Fourier-slice insertion —
the same operations Warp performs, just in readable Python.

## What is faithful, and what is approximate

Reproduced exactly from the Warp source:

- **Geometry** — `GetPositionInAllTilts` / `GetAngleInAllTilts`, incl. tilt +
  tilt-axis matrices, `LevelAngleX/Y`, per-tilt shifts (`AxisOffsetX/Y`),
  volume-warp grids, local-motion grids, per-tilt defocus, and inverted-angle
  handling. Rotation matrices match `Matrix3` element-for-element.
- **CTF** — the `CTF.cs` equation (wavelength, K1..K4, astigmatism, phase shift,
  amplitude contrast, B-factor envelope, scale).
- **Weighting** — `GetCTFsForOneParticle`: `Scale = cos(θ)` or the dose/location
  weight grids, and the dose B-factor exposure filter (`−Dose·4`, or the dose/
  location B-factor grids), plus optional global weight/B-factor.
- **Reconstruction bookkeeping** — data are phase-flipped and weighted by the
  weighted CTF; the reconstruction weight is `|unweighted CTF|`; coverage is
  capped at 1; the weight volume is floored (`0.01`) before division — exactly
  as in `ReconstructFull`.
- **Metadata I/O** — `.tomostar` (STAR), per-tilt-series `.xml`
  (`CubicGrid`/`LinearGrid4D`, `CTF`, per-tilt arrays). Verified against a real
  41-tilt Warp `.xml`.

Documented approximations (the "close" in *faithful & close*), because a
bit-identical match needs Warp's CUDA gridding kernel:

- The Fourier-slice **insertion kernel** is trilinear + sinc² de-apodization
  (Warp uses its own oversampled gridding kernel with optional gridding
  iterations). Structurally identical, numerically very close.
- The default **`global` engine** reconstructs the whole tomogram in one Fourier
  volume and evaluates the CTF at the tomogram-centre defocus per tilt. Warp
  tiles the volume into padded sub-volumes and re-evaluates the CTF per
  sub-volume, so it captures the defocus gradient with depth/position. For a
  series whose model grids are spatially trivial (no M refinement; the common
  case — see your `VLP3x3_...xml`, where every warp/motion/dose grid is 1×1×1),
  the only difference is the ±(thickness/2)·1e-4 µm defocus gradient in z.
- The **deconvolution** filter approximates `GPU.DeconvolveCTF` from the
  documented strength/falloff/highpass parameters.

**Optional novaCTF-style 3D-CTF correction** (`--ctf3d_defocus_step` nm, or
`--ctf3d_num_strips`) narrows the `global` engine's tomogram-centre-defocus gap
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
(`--renorm_variance`, off by default); with it off, the power spectrum now
tracks Warp's closely from DC out to ~125 Å, with only a small residual
high-frequency excess left (consistent with the trilinear-vs-gridding-kernel
difference above). See `tomo_eval/compare_fixed/` for the comparison that
found this (not tracked in git; regenerate with `compare_tomograms.py`).

## Install

```bash
pip install -r requirements.txt      # numpy, scipy, mrcfile, pillow
```
SciPy is optional: without it, cubic-grid interpolation falls back to linear
(exact for the per-tilt node-sampled grids, approximate for true 2-D/3-D grids).

## Quick start

```bash
python reconstruct_tomo.py \
    --xml       VLP3x3_p03_ts_002_blended_frames.xml \
    --tomostar  VLP3x3_p03_ts_002_blended_frames.tomostar \
    --settings  warp_tiltseries.settings \
    --tilt_dir  warp_frameseries/average \
    --angpix    10 \
    --output    reconstruction \
    --deconv
```

### Picking the tomogram pixel size and box

- `--angpix` is **the** knob for the output tomogram pixel size (exactly like
  `ts_reconstruct --angpix`). It sets the sampling and, together with the box,
  the number of voxels.
- The box (in unbinned pixels) comes from `--settings` (`<Tomo>` DimensionsX/Y/Z)
  by default; pass `--dimensions X Y Z` to override, or omit `--settings` to fall
  back to the box stored in the `.xml`. The raw pixel size is read from
  `--settings` (`PixelSize`), the tilt-image header, or `--raw_angpix`.
- Memory scales with the box: the engine builds an in-plane `S×S` × `Z` grid,
  where `S ≈ 1.15·max(Vx,Vy)` voxels and `Z ≈ tomogram thickness`. For the
  `VLP3x3` box (11664²×2400 px), `--angpix 10` needs ~8 GB RAM; `--angpix 14`
  ~2–3 GB. **Increase `--angpix` if you are memory-limited** — that is the
  pixel-size/RAM trade-off.

Tilt-image input (pick one):

- `--tilt_stack FILE` — a 3-D MRC/`.st` stack whose slices are ordered like the
  model's tilts (i.e. like the `.tomostar` rows / `.xml` `MoviePath` list).
- `--tilt_dir DIR` — a folder of per-tilt averages named `<movie-root>.mrc`,
  matched to the `.xml` `MoviePath` entries (these are Warp's
  `warp_frameseries/*.mrc` averages).

The raw (unbinned) pixel size is taken from the tilt-image MRC header; override
with `--raw_angpix` or read it from `--settings`.

Outputs mirror `ts_reconstruct`: `<name>.mrc`, a `<name>.png` central-slice
preview, and `<name>_deconv.mrc` when `--deconv` is given. Add `--float16` to
write 16-bit MRC like Warp.

## Making new versions — where to edit

The whole point. Both hooks are small, isolated functions.

**Weighting** — `warp_recon/weighting.py`. `warp_weighting()` is the exact Warp
model. To change it, copy it, edit the `Scale` / `Bfactor` lines, and pass it in:

```python
from warp_recon import reconstruct, ReconOptions, make_dose_bfactor_weighting
# e.g. replace Warp's hard-coded dose→B-factor factor of 4 with 6:
wfn = make_dose_bfactor_weighting(dose_bfactor_scale=6.0)
reconstruct(model, tilts, opts, weighting_fn=wfn)
```

or from the CLI: `--dose_bfactor_scale 6`.

A second built-in scheme, `motioncor3_dose_weighting`, replaces Warp's linear
dose Bfactor (a single Gaussian falloff, `Bfactor = -dose*4`) with the
Grant & Grigorieff (2015) critical-exposure curve MotionCor3 applies to
frames (`Correct/GWeightFrame.cu:mGCalcWeight`), applied here per tilt via
each tilt's accumulated dose. Use it with `--dose_weighting motioncor3`.
Validated on `VLP3x3_p03_ts_002` (2026-08-28): Pearson correlation against
the Warp reference improved on every metric (specimen-band 0.080→0.090,
full-box 0.011→0.017) and the FSC crossing points were unchanged (0.5 @
~165 Å, 0.143 @ ~111 Å — still a real, valid curve). Note this comes from
*less* high-frequency power overall, not more: the curve's biggest deviation
from a flat B-factor is a *stronger* attenuation of high-dose (typically
high-tilt) images at high resolution, where those images carry mostly noise
rather than signal — it turns out to be more accurate there than Warp's own
flat linear model, not more permissive.

**Filtering** — `warp_recon/filters.py`: `preprocess_tilt()` (per-tilt real-space
band-pass / normalize / invert) and `deconvolve()` (post-reconstruction Fourier
filter). Edit or replace and pass your versions into `reconstruct()`.

Because weighting and filtering are the only things that change between versions,
you can loop over parameter sets and write one tomogram per setting.

## Second engine: IMOD/etomo back-projection (`--engine etomo`)

Everything above reconstructs by CTF-weighted Fourier-slice insertion, which is
what Warp does. `--engine etomo` instead hands the tilt series to IMOD's `tilt`,
which is a genuinely different algorithm: real-space weighted back-projection
with a 1-D radial filter per projection line. The alignment and the per-tilt
preprocessing are shared with the Fourier engine, so the back-projection is the
only thing that changes.

```bash
# weighted back-projection (IMOD's own RADIAL 0.35 0.035)
python reconstruct_tomo.py ... --engine etomo --etomo_recon wbp

# the SIRT-like filter (tilt -FakeSIRTiterations): a radial filter analytically
# equivalent to N SIRT iterations, at WBP cost
python reconstruct_tomo.py ... --engine etomo --etomo_recon fakesirt --etomo_sirt_iters 10

# true iterative SIRT (tilt -SIRTIterations)
python reconstruct_tomo.py ... --engine etomo --etomo_recon sirt --etomo_sirt_iters 10

# any of the above, plus the cos(tilt) per-view amplitude weighting the
# fourier engine applies (fed to tilt through its -WeightFile)
python reconstruct_tomo.py ... --engine etomo --etomo_recon wbp --etomo_view_weight warp
```

The IMOD project (stack, `.xf`, `.tlt`, `newst.com`, `tilt.com`, the aligned
stack and `tilt`'s raw output) is left in `<output>/etomo/`, so a run can be
inspected, edited and re-run by hand with `submfg tilt.com` or opened in etomo.

### The geometry comes from Warp

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
derivable, so it is exposed as `--etomo_xaxistilt_sign` and settled empirically:
in `validate.test_etomo` the two signs are identical at `LevelAngleX = 0` and
`+1` wins by a margin that grows monotonically with `|LevelAngleX|` (0.0000 at
0 deg, 0.0655 at 6 deg, 0.1905 at 12 deg). `+1` is the default.

### What "weighted" means in WBP -- it is not cos(tilt)

Worth being precise, because it is a real asymmetry between the two engines.
`tilt`'s back-projection is "weighted" in two senses, **neither of which is
cos(tilt)**:

- the **radial R-weighting** ramp filter applied per projection line in
  frequency space (`RADIAL cutoff falloff`, `--etomo_radial`), and
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
`--etomo_view_weight warp` closes that gap by feeding the active weighting
scheme's per-tilt amplitude scale to `tilt` through its `-WeightFile`. Only the
*scalar* per-view factor can travel that way: the dose exposure filter is a
frequency-dependent B-factor envelope, which has no one-number-per-view
representation and is therefore not transferred.

### Local motion (Warp's `GridMovementX/Y`), and two conversion fixes

`tilt` only takes a global alignment, but Warp's local motion is a per-tilt
2-D shift looked up at a point's *globally projected image position*, not at
its depth. So it is exactly a per-tilt image warp: `reconstruct.bake_local_motion`
resamples each tilt so that `I'(q) = I(q - mv(q))`, and sampling `I'` with the
global alignment then gives Warp's local geometry at every depth. It is on by
default whenever the grids vary across the image (`--etomo_no_local_motion`
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
| `--engine novactf --novactf_correction none` (same geometry, no CTF) | 0.707 | 30.1 A |

Block-wise sub-voxel shifts between the etomo and novactf volumes are now
<= 0.18 voxel everywhere. The `VLP3x3_p03_ts_002` numbers above were measured
before the half-pixel fix and have not been re-run since.

### What `tilt` still does not do

`tilt` has no notion of a CTF. Unlike the Fourier engine, the etomo path applies
**no phase flip, no |CTF| weighting and no dose B-factor**. `--deconv`,
`--dose_bfactor_scale` and the `--ctf3d_*` options are therefore ignored with
`--engine etomo`, as is the B-factor half of `--dose_weighting` (the CLI says so
rather than dropping them silently). That is a real difference between these
entries and the rest of the list, not an oversight. What *is* shared is
`reconstruct._preprocess_tilts`, so the tilt data going in is bit-identical to
what the Fourier engine inserts.

Also note `tilt`'s output densities are on its own scale (no `SCALE` is applied,
to keep them linear): the WBP volume has std ~4.0 where Warp's reference has
0.0133. FSC and Pearson are scale-invariant, so this affects only display
contrast.

### Validated against the Warp reference

`VLP3x3_p03_ts_002`, 21.16 A/px, 584x584x120, vs
`warp_tiltseries/reconstruction/..._21.16Apx.mrc`:

| variant | specimen-band Pearson | FSC 0.5 | FSC 0.143 | wall clock |
|---|---|---|---|---|
| `fourier` (this repo's engine, current default) | 0.080 | 165 A | 111 A | ~5 min |
| `etomo --etomo_recon wbp` | **0.480** | **71.3 A** | **45.4 A** | 58 s |
| `etomo --etomo_recon wbp --etomo_view_weight warp` | **0.486** | **68.5 A** | **45.2 A** | 58 s |
| `etomo --etomo_recon fakesirt` (10) | **0.646** | **66.8 A** | **45.4 A** | 24 s |
| `etomo --etomo_recon fakesirt --etomo_view_weight warp` (10) | **0.654** | **65.5 A** | **45.3 A** | 24 s |
| `etomo --etomo_recon sirt` (10) | **0.604** | (see below) | **44.5 A** | ~2 min |

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

## Third engine: real-space novaCTF on Warp's geometry (`--engine novactf`)

novaCTF's 3D-CTF weighted back-projection (Turoňová et al. 2017), with every
voxel projected through Warp's *full* geometry -- including the
`GridMovementX/Y` local-motion grids, which neither the novaCTF binary (global
`.xf`/`.tlt` only, and it forces `XAXISTILT` to 0) nor IMOD `tilt` can use.
For each tilt, the image is CTF-corrected at defocus steps of `--novactf_step`
nm; each voxel then takes the copy whose defocus matches *its own depth along
that tilt's beam* (novaCTF's `generateFocusGrid` / `computeOneRow`), at the
image position `geometry.positions_one_tilt` gives it. See
`warp_recon/novactf.py` for the full list of differences from the binary.

```bash
# novaCTF defaults: multiplication, 10 nm steps, RADIAL 0.3 0.05, local motion on
python reconstruct_tomo.py ... --engine novactf
# ablations
python reconstruct_tomo.py ... --engine novactf --novactf_no_local_motion   # global alignment only
python reconstruct_tomo.py ... --engine novactf --novactf_step 0            # 2-D CTF (one defocus/tilt)
python reconstruct_tomo.py ... --engine novactf --novactf_correction phaseflip
python reconstruct_tomo.py ... --engine novactf --novactf_weighting warp    # + cos(tilt) & dose filter
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

`--threads N` sets the thread count for all engines (FFTs, numba kernels,
thread pools, and IMOD `tilt` via OMP_NUM_THREADS; `warp_recon.set_threads`
from Python). The default is the CPUs the process may run on
(`os.sched_getaffinity`), so under SLURM or `taskset` it uses the allocation,
not every core of the node. Because tilt j+1 is prepared while tilt j
back-projects, usage can briefly exceed N (measured: 851% average CPU at
`--threads 8`).

### Validated against Warp (HRR021_2_S02_L02_ts_003, 13.28 A/px)

Reference: `WarpTools ts_reconstruct --dont_invert` run on the *current*
`.xml` (2026-09-29). The reference in `warp_tiltseries/reconstruction/`
(2026-05-15) predates the local-motion grids that were later written into the
`.xml` (by the `miss-alignment` refinement; see `*_alignment_loss.json`), and
agrees with the current `.xml`'s own Warp reconstruction at only r = 0.08 --
compare against a freshly regenerated reference, not that one.

| variant | Pearson (central 80%) | FSC 0.5 | FSC 0.143 |
|---|---|---|---|
| old (May) Warp reference | 0.083 | 145 A | 88 A |
| `--novactf_no_local_motion` | 0.108 | 90 A | 51 A |
| default (multiplication, 3-D CTF) | 0.673 | 31.1 A | 26.6 A (Nyquist) |
| `--novactf_step 0` (2-D CTF) | -- | 31.1 A | 29.5 A |
| `--novactf_weighting warp` | 0.686 | 31.2 A | 26.6 A (Nyquist) |
| `--novactf_correction none` | 0.707 | 30.1 A | 29.3 A |

The local-motion grids shift content by up to ~15 px here and dominate every
other difference. Their sign is also checked without any reference: an
even/odd-tilt half-set FSC is highest with the grids applied, lower without
them and lowest with them negated (200-100 A band: 0.260 / 0.256 / 0.247).
At this pixel size and 4.2 um defocus, Nyquist (26.6 A) sits just past the
first CTF zero (~29 A), so the CTF variants can only differ in the last few
shells; 3-D vs 2-D CTF needs a finer pixel size to show.

### 3-D vs 2-D CTF (same series, 6.64 A/px, Nyquist 13.3 A)

Against Warp's own 6.64 A/px reconstruction of the current `.xml`:

| variant | Pearson (central 80%) | FSC 0.5 |
|---|---|---|
| multiplication, `--novactf_step 0` (2-D) | 0.593 | 29.5 A |
| multiplication, 10 nm steps (3-D, default) | 0.598 | 17.2 A |
| phaseflip, `--novactf_step 0` (2-D) | 0.608 | 29.4 A |
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

## Reproducing WarpTools itself (the default `fourier` engine), 2026-09-29

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
  (`reconstruct.bake_local_motion`; `--no_local_motion` disables it).

HRR021_2_S02_L02_ts_003, 13.28 A/px, `--dont_invert`, against `ts_reconstruct`
on the current `.xml`:

| variant | Pearson (central 80%) | FSC 0.5 | FSC 0.143 |
|---|---|---|---|
| before the fix, local motion on | 0.148 | 85 A | 55 A |
| **fixed, local motion on (default)** | **0.817** | **29.6 A** | **26.6 A (Nyquist)** |
| fixed, `--no_local_motion` | 0.232 | 80 A | 46 A |

Block-wise sub-voxel offsets to Warp's volume: <= 0.1 voxel. The phantom
self-test for this engine rose from 0.874 to 0.907.

## Validating against Warp

Two levels of validation ship with the code:

1. **Self-consistency** (no data needed): `python validate.py` builds a phantom,
   forward-projects it through the exact Warp geometry, reconstructs, and reports
   the correlation (≈0.9; limited only by the missing wedge). This checks the
   geometry sign conventions and the insertion/weighting normalization. It then
   runs the same phantom through `--engine etomo` (`validate.test_etomo`), which
   is the end-to-end certificate for the Warp→IMOD conversion: the `.xf` /
   `.tlt` / `XAXISTILT` are consumed by the real `newstack` and `tilt` binaries,
   the model deliberately carries a non-zero `LevelAngleX/Y`, per-tilt tilt-axis
   jitter and per-tilt axis offsets, and all 8 axis flips are searched so the
   output handedness is pinned rather than assumed (0.84 etomo vs 0.87 Fourier
   on the same phantom). Skipped with a message if IMOD is not installed.
2. **Against `ts_reconstruct`** (needs your data): run WarpTools
   `ts_reconstruct --angpix 10` on this tilt series, then reconstruct the same
   series here at the same `--angpix`, and compare (FSC / real-space correlation,
   orientation, contrast). Small differences are expected from the gridding
   kernel and the global-vs-subvolume CTF (see approximations above).

## Layout

```
warp_recon/
  metadata.py    parse .tomostar / per-tilt-series .xml / .settings
  grids.py       CubicGrid, LinearGrid4D
  geometry.py    Matrix3, GetPositionInAllTilts, per-tilt rotation
  ctf.py         CTF equation (CTF.cs)
  weighting.py   per-tilt weighted-CTF model      <-- edit for new WEIGHTING
  filters.py     preprocessing + deconvolution    <-- edit for new FILTERING
  reconstruct.py Fourier-slice reconstruction engine
  etomo.py       IMOD `tilt` back-projection engine (WBP / SIRT-filter / SIRT)
  novactf.py     real-space novaCTF 3D-CTF engine on Warp's per-voxel geometry
  mrc_io.py      MRC/PNG read/write
reconstruct_tomo.py   command-line front-end (mirrors ts_reconstruct)
validate.py           synthetic self-consistency test
```

## Source references

Transcribed from `warpem/warp@main`:
`WarpLib/TiltSeries/TiltSeries.ReconstructFull.cs`, `.../TiltSeries.cs`
(`GetPositionInAllTilts`, `GetAngleInAllTilts`, `GetCTFsForOneParticle`,
`LoadMeta`), `WarpLib/CTF.cs`, `WarpLib/CubicGrid.cs`, `WarpLib/LinearGrid4D.cs`,
`WarpLib/Tools/Matrix3.cs`, `WarpLib/Projector.cs`, and
`WarpTools/Commands/Tiltseries/ReconstructTiltseries.cs`.
