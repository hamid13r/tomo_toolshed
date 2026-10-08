# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

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

### Changed

- **xml-reconstruct**: Warp's local-motion grids are now applied by default in
  every engine, and per-tilt variance renormalization is off by default (it was
  the main source of excess high-frequency noise vs Warp). A sub-pixel patch
  shift sign bug in the Fourier engine is fixed. Output tomograms differ from
  0.1.0 as a result (closer to `ts_reconstruct`).

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
