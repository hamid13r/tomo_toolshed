# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **pick-granules**: new tool to threshold dark granules (e.g. mitochondrial
  granules) inside a binary mask. GUI with Z/Y views of the tomogram, the mask
  and the picked voxels overlaid, a histogram of in-mask values with a live
  threshold slider (or typed value/percentile), XY mask erosion and min island
  size. Writes a uint8 0/1 MRC with the mask's voxel size and origin, for
  cleanup with `curate`. Options `--threshold`/`--percentile`, `--bright`,
  `--blur`, `--mask-erode`, `--erode-3d`, `--min-size`, `--no-gui`, `--overwrite`.
- **vtp-normals**: new tool (click group with three subcommands) to preview,
  curate and flip the normals of pycurv/morphometrics `.vtp` membrane surfaces.
  The surface is split into connected pieces, each split again where a cell
  array crosses a threshold (`--split-column`, default `OMM_dist`, at
  `--split-dist`, default 20 nm; small parts merged, pieces numbered largest
  first).
  - `vtp-normals preview`: per-piece `<name>.normals.png` panels (membrane
    points, normal tips, tomogram background).
  - `vtp-normals curate`: interactive 3-orthoslice curator with a piece table;
    mark pieces, write `<name>.flipped.vtp`, then review.
  - `vtp-normals flip`: batch-flip listed `TS:MEMBRANE:IDS` pieces (or
    `--flip-file`) into `<name>.flipped.vtp` with reversed winding, negated
    normal arrays, curvature negate/swap rules, and `piece_id` /
    `normal_flipped` cell arrays.
  - `--split-column` generalizes the previously hard-coded `OMM_dist`; it must
    match between `preview`/`curate` and `flip` (with `--split-dist` and
    `--min-triangles`) or piece ids will not line up. A surface without the
    column falls back to connected components only.
  - `curate` and `preview` gain `--bin N` (mean-pool the tomogram by `N` in 3D
    before display; memory-light via `mrcfile.mmap` in Z-slab chunks, so
    `--bin 2` on a multi-GB tomogram stays well under the full float32 volume)
    and `--gaussian SIGMA` (Gaussian blur, sigma in Å). `--apix` stays the
    unbinned pixel size; surface coordinates, `--dist` and the alignment check
    use the effective pixel size `apix * bin`; `slab` and arrow width stay in
    binned display voxels. Binning never affects the vtp — piece ids and the
    written `.flipped.vtp` are byte-identical to `--bin 1`. `curate` also has a
    live **blur (Å)** slider on the displayed slices. These options are not on
    `flip`.
  - `flip` gains `--coord-scale` (default 10, matching `preview`/`curate`), used
    to place the surface on the `--preview` background when `--preview` is on.
  - `curate` now opens every view on the **whole tomogram** and never auto-zooms:
    clicking a table row, Prev/Next, up/down and membrane clicks only highlight
    the piece and move the slices, keeping your current zoom. Use the matplotlib
    toolbar to zoom/pan and **Full view** to reset.
  - `curate` now reads the pixel size from the MRC header of the tomogram it is
    actually showing, so `--tomo-dir` / `--tomo-pattern` get the right `--apix`
    too (previously only an explicit `--tomo` did; `--tomo-dir` silently fell
    back to 9.98 Å and mis-scaled the surface overlay).
  - Adds a `vtk>=9.1` dependency (the one heavy dependency here, ~100 MB).
- **erode**: new tool. Erodes a binary segmentation MRC `--iterations` times in
  the XY plane per Z slice (default) or in 3D (`--3d`), with `--connectivity`
  4/8 (XY) or 6/18/26 (3D). Writes a uint8 0/1 MRC keeping the voxel size and
  origin (default output `<input>_eroded.mrc`).
- **xml-reconstruct**: everything from the standalone `warp_recont_test` repo
  since the first import, merged with its history:
  - `--engine etomo`: IMOD `tilt` weighted back-projection, SIRT-like filter or
    true SIRT, driven by Warp's alignment (local motion baked into the tilts).
  - `--engine novactf`: novaCTF's real-space 3D-CTF back-projection through
    Warp's full per-voxel geometry, with an optional GPU back end
    (`--novactf-device cuda`; needs PyTorch).
  - `--ctf3d-defocus-step` / `--ctf3d-num-strips`: novaCTF-style Z-strip 3D-CTF
    for the default Fourier engine.
  - `--dose-weighting motioncor3`, `--weight-floor`, `--renorm-variance`,
    `--no-highpass`, `--no-local-motion`, `--threads`.
- **compare-tomograms**: FSC / Pearson / power-spectrum comparison against a
  reference tomogram (was `compare_tomograms.py` upstream).
- **skipped-views**: Warp 1 support — UTF-16 / CRLF tilt-series XML (e.g.
  `X.mrc.xml`) is read and written back byte-for-byte.
- **skipped-views**: `--mdoc-dir` takes per-tilt doses from SerialEM mdocs
  (accumulated `ExposureDose` in acquisition order, first tilt = 0) and writes
  them to the XML `<Dose>` and, in `--delete` mode, the `.tomostar` `_wrpDose`
  column, warning where the old values differ. The mdoc doses are also used by
  `--n-tilts` / `--max-tilt`.
- **skipped-views**: `--angles-from-log` sets the tilt angles of the views in
  `taSolution.log` to etomo's refined values in the XML `<Angles>` and the
  `.tomostar` `_wrpAngleTilt` (sign matched to the file), warning where they
  differ and flagging changes over 5°.
- **skipped-views**: `--angles-from-log` no longer aborts a series whose log
  angles do not follow the XML (|corr| < 0.8); the log values override them
  with a warning. The sign is flipped only when the log clearly follows the
  file with the opposite sign (the old smallest-difference rule picked the
  wrong sign for failed alignments with a large offset).
- **skipped-views**: a diagnosis table at the end of every run lists, per
  series, the views kept / removed according to `taSolution.log`, the XML
  and the tomostar (each read independently, in original view numbering) and
  flags `SKIP` / `ABORT` / `MISMATCH`. `--no-summary` turns it off.
  An unexpected error in one series (e.g. an unreadable XML) is reported as an
  `ERROR` row instead of stopping the run, and the table is printed even if
  no XML matches or the run is interrupted.
- **skipped-views**: in flip mode, `--mdoc-dir` / `--angles-from-log` also
  update the `.tomostar` in `--tomostar-dir` when it exists (backed up first).

### Changed

- **xml-reconstruct**: Warp's local-motion grids are now applied by default in
  every engine, and per-tilt variance renormalization is off by default (it was
  the main source of excess high-frequency noise vs Warp). A sub-pixel patch
  shift sign bug in the Fourier engine is fixed. Output tomograms differ from
  0.1.0 as a result (closer to `ts_reconstruct`).

### Fixed

- **skipped-views**: re-running on a tilt series that `--delete` had already
  trimmed removed more tilts from the start of the series (the log's view
  numbers no longer lined up with the shorter XML). View numbers are now
  mapped through the first backup `<xml>.bak` by movie name, so a repeat
  `--delete` removes nothing and later selections use the original numbering.

- **skipped-views**: a `taSolution.log` with several solution tables (etomo
  local alignments) made every view look absent, so all tilts were excluded.
  Only the first (global) table is used now.
- **skipped-views**: `--max-tilt` looked up log tilt angles by list position
  instead of view number, picking the wrong angles whenever the log does not
  start at view 1.
- **skipped-views**: `--delete` refuses to remove every tilt of a series, and
  flip mode warns if the selection excludes all tilts. A flip-mode `--dry-run`
  no longer creates the backup directory. CRLF `.tomostar` files keep their
  line endings.

## [0.1.0] - 2026-10-05

First release: a collection of lightweight cryo-ET file/CLI tools under a single
`tomo_toolshed` command.

### Added

- **skipped-views** — prune skipped etomo views from WarpTools tilt-series XML by
  updating `UseTilt` from `taSolution.log`, with optional dose/tilt selection, or
  (with `--delete`) physically remove the excluded tilts from the XML and the
  matching `.tomostar`. Warp 2 XML round-trips byte-for-byte; includes a
  `--dry-run` preview and a test suite.
- **curate** — interactive GUI to review and clean a 3D segmentation over a
  tomogram and export a curated mask, or load particle picks (star / `.txt` /
  `.box`) and export a star file with false positives removed.
- **add-defocus** — fill the placeholder `_rlnDefocus` column of an IsoNet star
  file with the average CTF defocus from the matching Warp XML files.
- **trace-filaments** — trace filaments in a binary mask into a RELION 4 helical
  star file of evenly spaced particles.
- **dipole2star** — collapse manual dipole picks into a RELION oriented-particle
  star file.
- **write-ebt** — build an etomo `batchruntomo` `.ebt` project from a directory
  tree.
- **duplicate-remover** — drop particles closer than a distance threshold per
  tomogram/micrograph across RELION 3/4/5 and M/WarpTools star flavors.
- **split-star** — split a particle star file into one star file per group.
- **scale-star** — rescale particle coordinates between pixel sizes.
- **filament-cleanup** — remove off-helix particles after RELION helical
  refinement.
- **xml-reconstruct** — reconstruct a Warp/WarpTools tomogram from its
  tilt-series XML (a NumPy/SciPy re-implementation of `ts_reconstruct`), with
  swappable weighting and filtering hooks.

[Unreleased]: https://github.com/hamid13r/tomo_toolshed/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/hamid13r/tomo_toolshed/releases/tag/v0.1.0
