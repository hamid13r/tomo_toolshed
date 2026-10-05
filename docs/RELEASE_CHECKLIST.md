# Release checklist — tomo-toolshed 0.1.0

Status of each item: **done** / **todo** / **decide** (needs a maintainer call).
Items marked *(checked YYYY-MM-DD)* were verified by actually running the command.
Audit date: 2026-10-05. Nothing here has been released.

---

## Version and naming

- **done** — Version is `0.1.0`, declared statically in `pyproject.toml`.
- **done** *(checked)* — `tomo_toolshed --version` prints `tomo_toolshed, version 0.1.0`.
- **done** *(checked)* — The name `tomo-toolshed` is **available** on both PyPI and
  TestPyPI (HTTP 404 on each `/pypi/tomo-toolshed/json`).
- **decide** — Version source: static string vs. `setuptools_scm` (tag-derived).
  Static is fine for a first release; adopt `setuptools_scm` only if you want tags
  to drive the version automatically.

## Metadata (`pyproject.toml`)

- **done** — `authors = [{ name = "Hamid Rahmani" }]`.
- **todo** — Add an author **email** (PyPI shows it; useful for contact).
- **todo** — Add `[project.urls]` with at least `Homepage` and `Issues`
  (e.g. the GitHub repo + `/issues`). Currently absent.
- **todo** — Add `classifiers` (none present): `Programming Language :: Python :: 3`
  + per-minor, `License :: OSI Approved :: MIT License`,
  `Operating System :: OS Independent`, `Intended Audience :: Science/Research`,
  `Topic :: Scientific/Engineering`, and `Development Status :: 3 - Alpha` or
  `4 - Beta`.
- **todo** — Switch `license = { text = "MIT" }` to the SPDX string form
  `license = "MIT"` — the table form emits a setuptools deprecation warning at
  build time.
- **decide** — `requires-python = ">=3.9"` is **not yet verified against the code**.
  Confirm the floor (the code uses `from __future__ import annotations`, dataclasses,
  f-strings, `numpy.random.default_rng` — all 3.9-compatible) or raise it.
- **todo** — Dependencies have **no lower bounds**
  (`click`, `pandas`, `lxml`, `numpy`, `scipy`, `scikit-image`, `mrcfile`,
  `connected-components-3d`, `matplotlib`, `starfile`, `networkx`, `pillow`).
  Pin floors for the APIs actually used (at least `starfile`, `click`, `numpy`,
  `scipy`, `mrcfile`) so installs are reproducible.

## Packaging

- **done** *(checked)* — `python -m build` produces both an sdist
  (`tomo_toolshed-0.1.0.tar.gz`) and a wheel (`…-py3-none-any.whl`).
- **done** *(checked)* — `twine check dist/*` → **PASSED** for both artifacts.
- **done** *(checked)* — Installing the wheel in a clean venv works; the wheel
  **excludes** `tests/` and `.star` fixtures and **includes** every subpackage,
  including `tomo_toolshed/xml_reconstruct/` (10 modules).
- **done** — No package data is required (no bundled data files); `MANIFEST.in`
  is unnecessary for the wheel. Add one only if you later want data in the sdist.

## CI

- **todo** — No `.github/workflows/`. Add a **test workflow**: `pytest` on a Python
  version matrix (e.g. 3.9–3.13) for push/PR.
- **todo** — Add a **publish workflow** triggered on `v*` tags, using PyPI
  **Trusted Publishing** (OIDC, no stored token). Publish to TestPyPI first, then
  PyPI on release.

## Docs

- **done** — Every tool has a `docs/<tool>.md` page, including the new
  `docs/xml_reconstruct.md`; the README tool table links them all.
- **done** *(checked)* — `--help` runs for **all 11 subcommands** (click CliRunner,
  all exit 0).
- **done** — RELION 4.x is the stated output target for the star-writing tools
  (e.g. `trace-filaments`); `xml_reconstruct` docs state RELION 4.x conventions.
- **decide** — A RELION 5 converter is mentioned as future work; confirm the README
  states "RELION 4.x now, RELION 5 via a future converter" consistently.
- **done** — `xml_reconstruct` docs include a known-limitations section
  (small-image border mask; global-vs-subvolume CTF approximation).

## Quality

- **done** *(checked)* — `--help` smoke test passes for every subcommand.
- **done** *(checked 214 passed)* — Full `pytest` suite green in a fresh venv.
- **decide / todo** — **Dash/underscore option-spelling and command-alias
  consistency.** Hidden underscore command aliases exist for `duplicate_remover`,
  `filament_cleanup`, `scale_star`, `split_star`, `write_ebt`, `xml_reconstruct`,
  but the hyphenated commands **`add-defocus`, `skipped-views`, `trace-filaments`
  have no underscore alias**. Decide whether to add aliases for those three for
  consistency.
- **todo** — Audit error handling for consistency: confirm each tool raises
  `click.ClickException` / `click.UsageError` for user errors rather than bare
  tracebacks (newer tools do; older ones should be spot-checked).

## Housekeeping

- **todo** — Add a `CHANGELOG.md` (Keep a Changelog style); seed it with the
  `0.1.0` entry listing the bundled tools.
- **decide** — `CITATION.cff` and/or a Zenodo DOI for citeability. Recommended for
  a scientific tool; needs your ORCID / preferred citation.
- **done** — The stray `PROMPT_*.md` files are now git-ignored and untracked
  (removed from tracking in `69400a7`).
- **decide** — **Large test fixtures in git history / sdist.** Several `.star`
  fixtures are multi-MB (up to ~4.5 MB in `tests/duplicate_remover/`, ~3.2 MB in
  `tests/filament_cleanup/`). They are **not** in the wheel but bloat the repo and
  the **sdist**. Options: keep, shrink/subsample them, exclude from the sdist, or
  move to Git LFS.
- **todo** — `.gitignore` audit: confirm build/venv/`PROMPT_*` patterns cover
  everything; no generated artifacts are tracked.
- **done** *(checked)* — No oversized blobs beyond the known `.star` fixtures; the
  largest history blobs are those test fixtures.

## Release steps (do only when the above is green)

1. **todo** — Finalize metadata + CHANGELOG; bump nothing (stay `0.1.0`).
2. **todo** — Tag `v0.1.0`.
3. **todo** — Create a GitHub release with notes (tool list + highlights).
4. **todo** — Publish to **TestPyPI**; install from TestPyPI into a clean venv and
   smoke-test `tomo_toolshed --help` + one real command.
5. **todo** — Publish to **PyPI**.
6. **decide** — Optional conda-forge recipe (later), since several deps are
   conda-forge packages.

---

### Blockers for a public release (minimum before step 2)

1. CI test + publish workflows (trusted publishing).
2. `[project.urls]`, classifiers, author email, and the `license` SPDX fix.
3. Dependency lower bounds.
4. `CHANGELOG.md`.
5. A decision on the large `.star` fixtures (sdist size).

Everything else is polish or an explicit *decide*.
