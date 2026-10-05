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

## Install

```bash
pip install -r requirements.txt      # click, numpy, scipy, mrcfile, pillow
```
SciPy is optional: without it, cubic-grid interpolation falls back to linear
(exact for the per-tilt node-sampled grids, approximate for true 2-D/3-D grids).

## Quick start

```bash
warp-recont-test \
    --xml       VLP3x3_p03_ts_002_blended_frames.xml \
    --tomostar  VLP3x3_p03_ts_002_blended_frames.tomostar \
    --settings  warp_tiltseries.settings \
    --tilt-dir  warp_frameseries/average \
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

**Filtering** — `warp_recon/filters.py`: `preprocess_tilt()` (per-tilt real-space
band-pass / normalize / invert) and `deconvolve()` (post-reconstruction Fourier
filter). Edit or replace and pass your versions into `reconstruct()`.

Because weighting and filtering are the only things that change between versions,
you can loop over parameter sets and write one tomogram per setting.

## Validating against Warp

Two levels of validation ship with the code:

1. **Self-consistency** (no data needed): `python validate.py` builds a phantom,
   forward-projects it through the exact Warp geometry, reconstructs, and reports
   the correlation (≈0.9; limited only by the missing wedge). This checks the
   geometry sign conventions and the insertion/weighting normalization.
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
  mrc_io.py      MRC/PNG read/write
  cli.py         click command-line front-end (mirrors ts_reconstruct)
tests/                pytest suite (metadata, self-consistency, end-to-end)
```

## Source references

Transcribed from `warpem/warp@main`:
`WarpLib/TiltSeries/TiltSeries.ReconstructFull.cs`, `.../TiltSeries.cs`
(`GetPositionInAllTilts`, `GetAngleInAllTilts`, `GetCTFsForOneParticle`,
`LoadMeta`), `WarpLib/CTF.cs`, `WarpLib/CubicGrid.cs`, `WarpLib/LinearGrid4D.cs`,
`WarpLib/Tools/Matrix3.cs`, `WarpLib/Projector.cs`, and
`WarpTools/Commands/Tiltseries/ReconstructTiltseries.cs`.
