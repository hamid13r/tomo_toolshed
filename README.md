# tomo_toolshed

A small, growing collection of **lightweight cryo-ET file/CLI tools** — etomo and
WarpTools helpers, segmentation curation, and more to come — all installed at
once and exposed as subcommands of a single `tomo_toolshed` command. Every tool
here is intentionally lightweight (file/CLI editors, no heavy or GPU deps), so
one install gets you everything.

## Install

```bash
git clone https://github.com/hamid13r/tomo_toolshed.git
cd tomo_toolshed
pip install -e .
```

This installs the `tomo_toolshed` command. List the available tools with:

```bash
tomo_toolshed --help
```

Prefer conda/micromamba? A merged environment is provided:

```bash
micromamba create -f environment.yml -y
micromamba activate tomo-toolshed
pip install -e .
```

## Tools

| Command | Description | Docs |
|---|---|---|
| `tomo_toolshed skipped-views` | Prune skipped etomo views from WarpTools tilt-series XML by updating `UseTilt` from `taSolution.log` (with optional dose/tilt selection), or physically remove the excluded tilts from the XML and `.tomostar` with `--delete`. | [docs/skipped_views.md](docs/skipped_views.md) |
| `tomo_toolshed curate` | Interactive GUI to review and clean a 3D segmentation over a tomogram, then export a curated binary mask — or load particles (star, or x y z `.txt`/`.box`) as spheres and export a star file with false positives removed. | [docs/segmentation_curator.md](docs/segmentation_curator.md) |
| `tomo_toolshed add-defocus` | Fill in the placeholder `_rlnDefocus` column of an IsoNet star file with the average CTF defocus from the matching Warp XML files. | [docs/add_defocus.md](docs/add_defocus.md) |
| `tomo_toolshed trace-filaments` | Trace filaments in a binary segmentation mask and export a RELION 4 helical star file of evenly spaced particles (optional ChimeraX `.bild` overlay). | [docs/filament_tracer.md](docs/filament_tracer.md) |
| `tomo_toolshed dipole2star` | Collapse manual dipole picks (RELION star or plain 3-column text) into a RELION oriented-particle star file, one output per input. | [docs/dipole2star.md](docs/dipole2star.md) |
| `tomo_toolshed write-ebt` | Build an etomo batchruntomo `.ebt` project file from a directory of per-tilt-series subdirectories, with Linux or Windows `.st` path styles. | [docs/write_ebt.md](docs/write_ebt.md) |
| `tomo_toolshed duplicate-remover` | Remove particles closer than a distance threshold within each tomogram/micrograph, across RELION 3/4/5 and M/WarpTools star flavors (resolves the coordinate pixel size per flavor). | [docs/duplicate_remover.md](docs/duplicate_remover.md) |
| `tomo_toolshed split-star` | Split a particle star file into one star file per tomogram/micrograph/source, flat by default or one subdirectory per group with `--dir-per-group` (carries through optics/general blocks). | [docs/split_star.md](docs/split_star.md) |
| `tomo_toolshed scale-star` | Rescale particle coordinates between pixel sizes (with optional shift), rewriting the coordinate pixel-size columns, across RELION 3/4/5 and M/WarpTools star flavors (leaves `rlnImagePixelSize` and `*Angst` columns alone). | [docs/scale_star.md](docs/scale_star.md) |
| `tomo_toolshed filament-cleanup` | After RELION helical refinement, remove particles that sit off a smooth curve through their filament or whose tilt/psi axis disagrees with their neighbours; writes the cleaned star file plus a star file of the removed particles for inspection. | [docs/filament_cleanup.md](docs/filament_cleanup.md) |
| `tomo_toolshed xml-reconstruct` | Reconstruct a tomogram from a Warp/WarpTools tilt-series XML with one of three engines: a NumPy/SciPy re-implementation of `ts_reconstruct` (default), IMOD `tilt` WBP/SIRT (`--engine etomo`), or novaCTF 3D-CTF on Warp's full geometry (`--engine novactf`, optional GPU). Swappable weighting and filtering hooks for making new versions of a tomogram from the same aligned tilts. | [docs/xml_reconstruct.md](docs/xml_reconstruct.md) |
| `tomo_toolshed compare-tomograms` | Compare reconstructions against a reference tomogram: FSC, Pearson correlation, power spectra and an axis-flip/sign search, with plots and a JSON summary. | [docs/xml_reconstruct.md#compare-tomograms](docs/xml_reconstruct.md#compare-tomograms) |

## Development

```bash
pip install -e ".[test]"
pytest
```

## Adding a tool

The layout is designed so a new lightweight tool is easy to drop in. Convention:

1. **Create a subpackage** under `src/tomo_toolshed/<your_tool>/` with a
   `cli.py` exposing a `click` command (split heavier logic into a `core.py`,
   like `skipped_views/` does). Keep any GUI/matplotlib imports lazy so the
   package stays headless-import-safe.
2. **Register it** in `src/tomo_toolshed/cli.py`: import the command and add it
   to the group with `tomo_toolshed.add_command(<cmd>, name="<subcommand>")`.
   Also add a one-line entry to the group docstring so `tomo_toolshed --help`
   reads as a useful index.
3. **Add any new dependencies** to the single `dependencies` list in
   `pyproject.toml` (and to `environment.yml`). There are deliberately no
   per-tool extras — one install gets everything.
4. **Add a docs page** at `docs/<your_tool>.md` and link it from the table above.
5. **Add tests** under `tests/<your_tool>/`.

## Acknowledgments

These tools read, write, or interoperate with files from the cryo-ET ecosystem,
and build on established open-source software. Please cite the relevant upstream
projects when you use the corresponding part of this toolshed (see
[`CITATION.cff`](CITATION.cff) for structured references).

Interoperates with / builds on:

- [Warp / WarpTools](https://github.com/warpem/warp) — `xml-reconstruct`
  reimplements `ts_reconstruct`, and several tools read Warp XML / `.tomostar` /
  `.settings` (Tegunov & Cramer, *Nat. Methods* 2019).
- [IMOD / etomo](https://bio3d.colorado.edu/imod/) — `skipped-views` reads etomo
  `taSolution.log`; `write-ebt` builds a `batchruntomo` `.ebt` project;
  `xml-reconstruct --engine etomo` drives IMOD `tilt` (Kremer,
  Mastronarde & McIntosh, *J. Struct. Biol.* 1996).
- [novaCTF](https://github.com/turonova/novaCTF) — `xml-reconstruct --engine
  novactf` and `--ctf3d-*` re-implement its 3D-CTF correction (Turoňová et al.,
  *J. Struct. Biol.* 2017).
- [MotionCor3](https://github.com/czimaginginstitute/MotionCor3) —
  `xml-reconstruct --dose-weighting motioncor3` uses the critical-exposure
  curve it applies (Grant & Grigorieff, *eLife* 2015).
- [AreTomo](https://github.com/czimaginginstitute) — marker-free tilt-series
  alignment, an alignment source reflected in the metadata (Zheng et al., 2022).
- [RELION](https://github.com/3dem/relion) — the star-file format target for the
  helical / oriented-particle tools (RELION 4.x).
- [IsoNet](https://github.com/IsoNet-cryoET/IsoNet) — `add-defocus` fills IsoNet
  star files.

Built on these Python libraries: [mrcfile](https://github.com/ccpem/mrcfile),
[starfile](https://github.com/teamtomo/starfile),
[NumPy](https://numpy.org/), [SciPy](https://scipy.org/),
[scikit-image](https://scikit-image.org/), [pandas](https://pandas.pydata.org/),
[Click](https://click.palletsprojects.com/),
[NetworkX](https://networkx.org/),
[connected-components-3d](https://github.com/seung-lab/connected-components-3d),
[Matplotlib](https://matplotlib.org/), [Pillow](https://python-pillow.org/), and
[lxml](https://lxml.de/).

## License

MIT — see [LICENSE](LICENSE).
