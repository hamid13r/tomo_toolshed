# Prompt for Claude Code — integrate `duplicate_remover` into tomo_toolshed

Integrate `src/tomo_toolshed/duplicate_remover/remove_close_particles.py` into the
`tomo_toolshed` CLI as a new subcommand. Follow the conventions already set by
`add_defocus/`, `filament_tracer/` and `dipole2star/`: subpackage with
`__init__.py` + `cli.py` (click command, options, reporting) + `core.py` (pure
logic, no click), tests in `tests/<tool>/`, docs page `docs/<tool>.md`, README
table row, and `matplotlib` imported **lazily** so the package stays
headless-import-safe.

## 0. Inspect first

Read the script, then read the headers of the four real star files I put in
`tests/duplicate_remover/`. **Actually look at the column labels and the optics
blocks in each** — the whole difficulty of this tool is in the differences between
them, and they are not what you'd guess.

## 1. Command name and options

Register as `duplicate-remover` (plus a hidden `duplicate_remover` alias, as with
`write-ebt`). Say so if you think `remove-duplicates` reads better as a verb — I'm
open, but the package directory is `duplicate_remover`.

Use dashed option names to match the other tools, but keep the original
underscored spellings as additional aliases on the same options
(`@click.option('--star-path', '--star_path', 'star_path', ...)`) so my existing
command lines keep working.

Options:

| option | behavior |
|---|---|
| `--star-path` | required, existing file |
| `--output-path` | default: input name + `_cleaned` — build it with `pathlib` (`with_name`), **not** `str.replace('.star', ...)`, which today corrupts any path containing `.star` elsewhere |
| `--threshold` | distance threshold, default 140 |
| `--units [angstroms\|pixels]` | default `angstroms`; `pixels` multiplies `--threshold` by the resolved pixel size. All internal math stays in Å |
| `--pixel-size` | override the pixel size resolved from the file (Å/px) |
| `--group-by` | override the auto-detected grouping column |
| `--comparison-metric` | `auto` (default), `rlnLogLikeliContribution`, `rlnMaxValueProbDistribution`, `random`, `first` |
| `--seed` | seed for `random`, so runs are reproducible (today's `np.random.rand()` is unseeded) |
| `--histogram / --no-histogram` | default on, preserving current behavior |
| `--dry-run` | report what would be removed, write nothing |
| `--quiet` / `-q` | only the final summary (matches `add-defocus`) |

## 2. Pixel size: resolve it per file flavor, and report what you chose

The threshold is in Ångstroms, so the tool must know the Å/px scale **of the
coordinate columns**. That is not always the pixel size that's easiest to grab,
and the current code (`input_star['optics']['rlnImagePixelSize'].values[0]`) gets
it wrong on two of my four files. Implement this precedence, in `core.py`, as an
explicit resolver that returns both the value and a human-readable reason:

1. **`--pixel-size`** always wins (except for M/Warp files, see 5).
2. **RELION 5** — optics contains `rlnTomoTiltSeriesPixelSize`: coordinates are in
   unbinned tilt-series pixels, so use **that** column.
   In `relion_5_2D_example.star` it is `2.11`, while `rlnImagePixelSize` is `4.22`
   = 2.11 × `rlnTomoSubtomogramBinning` (2). `rlnImagePixelSize` is the extracted
   image scale, not the coordinate scale — using it would double every distance.
3. **RELION 4** — optics with `rlnImagePixelSize` and no
   `rlnTomoTiltSeriesPixelSize`: use `rlnImagePixelSize` (`6.65` in
   `relion_4_example.star`).
   Note that file's *particles* table also carries `_rlnPixelSize` = `9.98`, a
   stale leftover from picking at a coarser binning. **Do not use it.** Proof that
   6.65 is right: the coordinates in `relion_4_example.star` are exactly 1.5× those
   in `relion_3_example.star` for the same particles, and 9.98/6.65 = 1.5.
4. **RELION 3** — no optics block at all (`relion_3_example.star` is one unnamed
   `data_` block): use the per-particle `_rlnPixelSize` column (`9.98`); if absent,
   fall back to `rlnDetectorPixelSize / rlnMagnification × 1e4`.
5. **M / WarpTools** (`wrp*` columns, `mtools_example.star`): `wrpCoordinateX1`
   etc. are **already in Ångstroms**, so the scale is 1.0 and no pixel size exists
   in the file. Passing `--pixel-size` here should warn that it's ignored for
   coordinate conversion; `--units pixels` should fail with a clear message
   (there's no pixel size to convert with — tell the user to use `--threshold` in Å).

Resolve the pixel size **per optics group** by merging on `rlnOpticsGroup`, not
`.values[0]` — `relion_5_2D_example.star` has 6 optics groups. If groups disagree,
that's fine; use each row's own value and say so in the report.

Print a short provenance block before processing (suppressed by `--quiet`):
detected flavor, grouping column, coordinate columns, pixel size and where it came
from, whether origins were applied, threshold in both Å and px.

## 3. Grouping and coordinate columns

- Grouping column: first present of `rlnTomoName`, `rlnMicrographName`,
  `wrpSourceName`, overridable with `--group-by`. In my files:
  relion 5 → `rlnTomoName` (11 groups); relion 4 → `rlnMicrographName` (46);
  relion 3 → `rlnMicrographName` (1); mtools → `wrpSourceName` (5). Error clearly
  if none is present.
- Coordinates: `rlnCoordinateX/Y/Z`, or `wrpCoordinateX<N>/Y<N>/Z<N>` — detect the
  numeric suffix by regex rather than hardcoding `1`, and use the same N for all
  three axes.
- Origins (keep the existing behavior from the script): if
  `rlnOriginXAngst/YAngst/ZAngst` are present, the shifted position is
  `coord × apix − origin_angst`. Present in the relion 4 and 5 files, absent in the
  other two. Additionally support RELION-3-style `rlnOriginX/Y/Z`, which are in
  **pixels** (multiply by apix) — none of my samples has them, but plenty of my
  older files do. If only some of the three axes are present, treat the missing
  ones as 0 and warn, instead of silently discarding the shifts as the current
  `and`-chained check does.

## 4. Multi-block I/O

- `starfile.read()` returns a plain DataFrame for the single-unnamed-`data_` files
  (mtools, relion 3) and a dict for the multi-block ones. Handle both — verify
  behavior yourself rather than trusting this sentence, e.g. with
  `always_dict=True` — and pick the particles block as `particles` if present,
  else the sole/largest block.
- On write, **preserve every block in its original order**. The current code
  reassembles only `{'optics', 'particles'}`, which silently drops the
  `data_general` block of `relion_5_2D_example.star` (holding
  `rlnTomoSubTomosAre2DStacks 1`) — that would break downstream RELION 5. A
  single-unnamed-block input must come back out as a single unnamed block.
- Don't hand-roll a star parser; let `starfile` handle the `# version 50001`
  comments.

## 5. Fix these bugs while porting

The algorithm's intent — iterate per group, drop the worse of any pair closer than
the threshold, repeat to convergence — is right; keep it. But:

1. **Over-removal.** The loop skips `i` if already marked, but not `closest_idx`:
   a particle already removed can still cause a *second* particle to be dropped for
   the same pair. Skip pairs where either member is already marked.
2. **Speed / determinism.** The per-particle `tree.query(k=2)` python loop over
   29k particles × several iterations is slow. Prefer one
   `tree.query_pairs(r=threshold)` pass, then resolve pairs greedily in order of
   the comparison metric (worst removed first), looping only until no pairs remain.
   Keep the invariant that both members of a pair are never removed, and make the
   result deterministic for non-`random` metrics (tie-break on a stable key, not
   on row order).
3. **Missing metric column.** The default metric is `rlnLogLikeliContribution`,
   which exists in only *two* of my four files — today that's a `KeyError`. With
   `--comparison-metric auto`, use `rlnLogLikeliContribution` if present, else
   `rlnMaxValueProbDistribution`, else keep-first, printing which was chosen. An
   explicitly requested metric that's missing should be a clean
   `click.ClickException`.
4. **Index fragility.** `drop(index[list(to_remove)])` mixes positional and label
   indexing; use positional masks / `.iloc` throughout and don't rely on
   `reset_index`.
5. **Edge cases:** groups with fewer than 2 particles (`query(k=2)` returns `inf`),
   empty groups, and the `removed/initial*100` division by zero.
6. **matplotlib at module scope** — move it inside the plotting function, force the
   `Agg` backend there, and keep `core.py` import-clean of it so the headless tests
   still pass.
7. Row order: state in the docs whether the output preserves the input row order or
   is regrouped by tomogram (today's `concat` over `unique()` reorders rows).
   Preserving the original order is preferable; if you regroup, say so explicitly.

## 6. Tests

`tests/duplicate_remover/test_duplicate_remover.py`, alongside the four star files
already there. Split into:

- **Detection tests** over the four real files (read-only, no writing into the
  repo — outputs to `tmp_path`), asserting the resolved flavor, grouping column,
  group count (5 / 1 / 46 / 11 as above), pixel size (1.0 / 9.98 / 6.65 / 2.11 —
  in particular **not** 9.98 for relion 4 and **not** 4.22 for relion 5), and
  whether origins were detected.
- **Round-trip test**: with a threshold of 0, particle counts are unchanged and
  every input block survives with its columns — especially `data_general` for
  relion 5, and single-unnamed-block output for mtools and relion 3.
- **Removal math** on small synthetic DataFrames: a known pair inside the
  threshold keeps exactly the higher-metric particle; a cluster of three collapses
  to one; a pair exactly at the threshold is kept (document `<` vs `<=`); origin
  shifts change which pairs are within threshold.
- `--units pixels` on the relion 4 file equals `--threshold (px × 6.65)` in Å.

The two large files (~3.7 MB, ~4.5 MB) make full-run tests slow — subset them in
memory for anything beyond header/detection assertions.

## 7. Wiring, docs, deps

- Register in `src/tomo_toolshed/cli.py` and add a one-line entry to the group
  docstring's `\b` index, in the existing aligned style.
- `docs/duplicate_remover.md`: what it does, the pixel-size precedence table from
  section 2 (this is the part I'll forget), the star flavors supported, all
  options, and worked examples for a RELION 5 and an M/Warp file.
- README: one row in the tools table linking the docs page.
- `pyproject.toml` should need no change (click, pandas, numpy, scipy, starfile,
  matplotlib are all already listed) — confirm rather than assume.
- Git history: the script is currently untracked. `git add` and commit it as-is
  first, then `git mv` it into `core.py` (or `cli.py`) and edit in place, so
  `git log --follow` shows the original.

## Verify before finishing

- `pip install -e ".[test]"` then `pytest` — whole suite passes.
- Run the command for real on all four files in a scratch directory, and paste the
  provenance block plus the before/after counts for each. Sanity-check that the
  relion 4 and relion 5 removals use 6.65 and 2.11 Å/px respectively.
- Confirm the relion 5 output still has `data_general`, and that mtools output is
  still a single unnamed block, by diffing the block structure of input vs output.
- `tomo_toolshed --help` lists the new subcommand; `--help` on it shows every
  option above.
- Grep for leftovers: no `remove_close_particles` references, no broken doc links.
- `git status` clean. Do **not** push — show me the commits for review first.
