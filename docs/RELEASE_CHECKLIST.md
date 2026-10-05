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

- **done** — `authors = [{ name = "Hamid Rahmani", email = "hrahmani@scripps.edu" }]`.
  *(Note: this email becomes public on PyPI — change it if you'd prefer a different
  contact.)*
- **done** *(checked)* — `[project.urls]` added: `Homepage`, `Issues`, `Repository`
  (all pointing at the GitHub repo). Verified in the built wheel's METADATA.
- **done** *(checked)* — `classifiers` added (11): dev-status Beta, science audience,
  bio-informatics/image-processing topics, OS-independent, Python 3 + 3.9–3.13.
- **done** *(checked)* — Switched to SPDX `license = "MIT"` + `license-files`;
  build now emits **no license deprecation** and METADATA shows
  `License-Expression: MIT`. Requires `setuptools>=77` (bumped in `[build-system]`).
- **done** — `requires-python = ">=3.9"` kept; the code is 3.9-compatible
  (`from __future__ import annotations`, dataclasses, f-strings,
  `numpy.random.default_rng`). The CI matrix (3.9–3.13) will confirm on every push.
- **done** *(checked)* — Dependency **lower bounds** pinned: `click>=8.0`,
  `pandas>=1.3`, `lxml>=4.6`, `numpy>=1.23`, `scipy>=1.9`, `scikit-image>=0.19`,
  `mrcfile>=1.4`, `connected-components-3d>=3.10`, `matplotlib>=3.5`,
  `starfile>=0.5`, `networkx>=2.6`, `pillow>=9.0`. 214 tests pass with them.

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

- **done** — `.github/workflows/tests.yml`: `pytest` on a Python matrix
  (3.9–3.13) for push to `master` and all PRs, with pip caching.
- **done** — `.github/workflows/publish.yml`: on `v*` tags, builds + `twine check`,
  then publishes to **TestPyPI** then **PyPI** via **Trusted Publishing** (OIDC,
  no stored token), gated on `testpypi`/`pypi` GitHub environments.
- **decide / action** — One-time setup required before the publish workflow works:
  register the GitHub repo as a **trusted publisher** on pypi.org *and*
  test.pypi.org (owner `hamid13r`, repo `tomo_toolshed`, workflow `publish.yml`,
  environments `pypi`/`testpypi`), and create those two environments in the repo
  settings. This can only be done by the maintainer in the web UIs.

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
- **done** *(checked)* — **Command-alias consistency.** Every hyphenated command
  now has a hidden underscore alias; `add_defocus`, `skipped_views`, and
  `trace_filaments` aliases were added (they were the only ones missing). Verified
  all three run and none leak into `--help`.
- **todo** — Audit error handling for consistency: confirm each tool raises
  `click.ClickException` / `click.UsageError` for user errors rather than bare
  tracebacks (newer tools do; older ones should be spot-checked).

## Housekeeping

- **done** — `CHANGELOG.md` added (Keep a Changelog style) with the `0.1.0` entry
  listing all 11 tools.
- **done** — `CITATION.cff` added (with author ORCID), so GitHub renders a
  "Cite this repository" button with APA/BibTeX. A Zenodo DOI can be added later
  (commented `identifiers:` stub is in place) once a release is archived.
- **done** — The stray `PROMPT_*.md` files are now git-ignored and untracked
  (removed from tracking in `69400a7`).
- **done** *(checked)* — **Distribution size is already clean.** The sdist is
  **86 KB and contains zero `.star` fixtures / zero test files**; the wheel
  excludes tests too. The large `.star` fixtures do **not** bloat the published
  artifacts — this was not actually a packaging problem.
- **decide** — Residual: those multi-MB `.star` fixtures still live in **git
  history** (~4.5 MB `tests/duplicate_remover/`, ~3.2 MB `tests/filament_cleanup/`),
  so a fresh clone is larger than necessary. Shrinking them requires a history
  rewrite (not done here). Options: leave as-is, subsample the fixtures going
  forward, or `git filter-repo` + force-push (coordinate with any collaborators).
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

### Blockers — status

1. ✅ CI test + publish workflows (trusted publishing) — **added** (`.github/workflows/`).
2. ✅ `[project.urls]`, classifiers, author email, `license` SPDX fix — **done**.
3. ✅ Dependency lower bounds — **pinned**.
4. ✅ `CHANGELOG.md` — **added**.
5. ✅ Large `.star` fixtures — **not a packaging blocker** (sdist is 86 KB, fixture-free);
   only a git-history-size *decide* remains.

**All code/metadata blockers are cleared.** The remaining gate is **maintainer-only,
web-UI setup** that cannot be scripted: register the PyPI/TestPyPI **trusted
publishers** and create the `pypi`/`testpypi` GitHub environments, then tag `v0.1.0`.
Remaining *decide* items: a Zenodo DOI (after the first release) and the
git-history fixture size.
