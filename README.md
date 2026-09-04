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
| `tomo_toolshed skipped-views` | Prune skipped etomo views from WarpTools tilt-series XML by updating `UseTilt` from `taSolution.log` (with optional dose/tilt selection). | [docs/skipped_views.md](docs/skipped_views.md) |
| `tomo_toolshed curate` | Interactive GUI to review and clean a 3D segmentation over a tomogram, then export a curated binary mask. | [docs/segmentation_curator.md](docs/segmentation_curator.md) |
| `tomo_toolshed add-defocus` | Fill in the placeholder `_rlnDefocus` column of an IsoNet star file with the average CTF defocus from the matching Warp XML files. | [docs/add_defocus.md](docs/add_defocus.md) |
| `tomo_toolshed trace-filaments` | Trace filaments in a binary segmentation mask and export a RELION-style helical star file of evenly spaced particles (optional ChimeraX `.bild` overlay). | [docs/filament_tracer.md](docs/filament_tracer.md) |

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

## License

MIT — see [LICENSE](LICENSE).
