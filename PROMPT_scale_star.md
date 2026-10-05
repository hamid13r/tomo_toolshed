# Prompt for Claude Code — add a `scale-star` subcommand

Add a new tool to `tomo_toolshed`: `tomo_toolshed scale-star`. It rescales the
particle coordinates of a star file from one pixel size to another (e.g. after
re-binning tomograms), optionally applies a shift, and rewrites the pixel-size
fields to the new value. It reads star files exactly the way `split-star` does,
so it supports the same four flavors (RELION 3, RELION 4, RELION 5, M/Warp).

Follow the repo's "Adding a tool" convention in `README.md`. Do not change any
other subcommand's behavior.

## 1. Inspect first

Read these before writing anything:

- `src/tomo_toolshed/split_star/core.py`: `read_star`, `particles_key`,
  `detect_flavor`, the `FLAVOR_*` constants, `_WRP_COORD_RE`, `_package_version`,
  `build_header_lines`, `_prepend_comment`, `write_group`. Reuse these by importing
  them. Don't copy-paste them. If something you need is private (`_`-prefixed),
  either import it anyway or promote it to a public name in `split_star/core.py`
  without changing its behavior. Keep split-star's tests green.
- `src/tomo_toolshed/split_star/cli.py`: the option style to match (`--input`/`--i`,
  `--outdir`/`-o`, `--dry-run`, `--quiet`/`-q`, `--comment/--no-comment`, dual
  dash/underscore spellings, `ClickException` for user errors, the report block).
- `src/tomo_toolshed/cli.py`: how tools are registered, the docstring index, and
  the hidden underscore aliases (`split_star` etc.).
- The fixtures in `tests/duplicate_remover/*.star`. They cover all four flavors.
  Check which coordinate and pixel-size columns each one actually has. Here is
  what I saw, but verify it yourself:
  - M: `wrpCoordinateX1/Y1/Z1`, no pixel-size column
  - RELION 3: `rlnCoordinateX/Y/Z` and `rlnPixelSize` in the particles block
  - RELION 4: `rlnCoordinateX/Y/Z`, `rlnPixelSize` in particles, plus
    `rlnMicrographPixelSize` and `rlnImagePixelSize` in optics, plus `rlnOrigin*Angst`
  - RELION 5: `rlnCoordinateX/Y/Z`, `rlnTomoTiltSeriesPixelSize` and
    `rlnImagePixelSize` in optics, plus `rlnOrigin*Angst`

## 2. Behavior

New subpackage `src/tomo_toolshed/scale_star/` with `__init__.py`, `core.py`
(pure logic, no click) and `cli.py` (click command `scale_star`).

**Options**

| Option | Required | Meaning |
|---|---|---|
| `--input`, `--i` | yes | Input star file (`click.Path(exists=True, dir_okay=False)`). |
| `--output`, `--o` | yes | Output star file. Create parent dirs. Refuse to overwrite `--input`. |
| `--input-pixel-size`, `--input_pixel_size` | yes | Pixel size (Å) the coordinates are currently in. |
| `--output-pixel-size`, `--output_pixel_size` | yes | Target pixel size (Å). |
| `--shift-x`, `--shift-y`, `--shift-z` (+ underscore spellings) | no, default 0 | Shift added **after** scaling, in **output pixels**. |
| `--yes`, `-y` | no | Skip the pixel-size mismatch confirmation. |
| `--comment/--no-comment` | no, default on | Provenance header, same format as split-star (`command="scale-star"`; record in/out pixel size, scale factor, shifts). |
| `--dry-run` | no | Report what would change; write nothing. |
| `--quiet`, `-q` | no | Only print the final summary. |

Both pixel sizes must be > 0. Otherwise raise `ClickException`.

**Coordinate scaling.** `factor = input_pixel_size / output_pixel_size`.
`new = old * factor + shift` for every X/Y/Z coordinate column, detected per
flavor:
- RELION 3/4/5: `rlnCoordinateX/Y/Z`
- M: every `wrpCoordinateX<n>/Y<n>/Z<n>` column (there can be several `<n>`)

Handle files that only have X/Y (2D) without crashing. If no coordinate columns
are found, raise an error.

**Leave alone:** every `rlnOrigin*Angst` column (they're already in Å) and any
other `*Angst` column, including `rlnCenteredCoordinate*Angst` and
`rlnHelicalTrackLengthAngst`. Leave every other column and block untouched too,
and keep block order and names. Use `write_group` or the same write path so the
output round-trips like split-star's.

**Pixel-size override.** Set these columns, wherever they appear (particles
block or optics block, every row), to `--output-pixel-size`:
`rlnPixelSize`, `rlnMicrographPixelSize`, `rlnTomoTiltSeriesPixelSize`.
Do **not** touch `rlnImagePixelSize`. That's the extracted-subtomogram box
pixel size, which is a separate thing from the coordinate pixel size. Put the
column list in a module-level constant with a comment explaining the choice, so
it's easy to change later. Report which columns were updated, and in which
blocks.

**Mismatch check.** Read the existing pixel size from the columns above. If any
value (`np.isclose` tolerance) differs from `--input-pixel-size`:
- Print a warning listing each column/block and the value(s) found vs. the value
  the user gave.
- Ask with `click.confirm("Proceed using --input-pixel-size=<value>?", abort=True)`.
- If the user confirms, carry on with **the user's `--input-pixel-size`**. The
  user's value always wins.
- `--yes` skips the prompt but still prints the warning.
- `--dry-run` prints the warning and doesn't prompt.

If the file has no pixel-size column at all (M), print a note that the input
pixel size couldn't be checked, then proceed. Don't try to add pixel-size
columns that aren't there.

**Report** (unless `--quiet`): input, detected flavor, scale factor, shifts,
coordinate columns scaled, pixel-size columns overridden, output path. Always
print a one-line summary (`scaled N particle(s) by F -> <output>`).

## 3. Register and document

- `src/tomo_toolshed/cli.py`: import the command, add a `scale-star` line to the
  group docstring index, register it with `add_command(..., name="scale-star")`,
  and add a hidden `scale_star` alias like the others.
- `docs/scale_star.md`: what it does, the scaling formula, units of shifts, which
  pixel-size columns are overwritten and which are deliberately left alone, the
  mismatch prompt and `--yes`, an options table, and 3–4 examples (plain rescale,
  rescale plus shift, scripted with `--yes`, `--dry-run`).
- Root `README.md`: add a `scale-star` row to the tools table, linking to
  `docs/scale_star.md`.
- No new dependencies. Everything here is already in `pyproject.toml`
  (click, numpy, pandas, starfile). Confirm this rather than assuming it.

## 4. Tests — `tests/scale_star/test_scale_star.py`

Use the four fixtures in `tests/duplicate_remover/` (read them from there; don't
duplicate them) plus small synthetic frames where that's clearer. Cover:

- Coordinates are scaled correctly for each flavor, including all
  `wrpCoordinate*<n>` columns in M.
- Shifts are applied after scaling, in output pixels.
- `rlnOrigin*Angst` and other `*Angst` columns are byte-identical before and after.
- `rlnPixelSize` / `rlnMicrographPixelSize` / `rlnTomoTiltSeriesPixelSize` are
  set to the new value, and `rlnImagePixelSize` is unchanged.
- Non-coordinate columns and block names/order are preserved, and so is the row count.
- Mismatch handling via `CliRunner`: answering `n` aborts with a non-zero exit
  and no output file. Answering `y` proceeds using the user's value. `--yes`
  never prompts. With no mismatch, there's no prompt.
- No-pixel-size (M) input proceeds and prints the note.
- Bad input: pixel size ≤ 0, output == input, no coordinate columns.
- `--dry-run` writes nothing. The `--comment` header is present by default and
  absent with `--no-comment`.

## Verify before finishing

- `pip install -e ".[test]"` succeeds and `pytest` passes, including all the
  existing split-star tests.
- `tomo_toolshed --help` lists `scale-star`. `tomo_toolshed scale-star --help`
  shows every option above. `tomo_toolshed scale_star --help` also works.
- Do a manual run on the RELION 4 fixture (e.g. 10 Å → 5 Å): coordinates double,
  Origin*Angst columns are unchanged, and the pixel-size columns read 5. Show me
  the before/after for 2–3 rows.
- `git diff --stat` touches only the new `scale_star/` package, its tests,
  `docs/scale_star.md`, `README.md`, `src/tomo_toolshed/cli.py`, and
  `split_star/core.py` if you had to promote a helper there.
- Grep docs/README for broken relative links.

## Commit and push

- Commit on the current branch (`merge-tools-monorepo`). Check it with
  `git branch --show-current` first and tell me if it's something else.
  One commit is fine. Message: `scale-star: rescale star-file coordinates between
  pixel sizes`, with a short body.
- Don't commit the untracked `PROMPT_*.md` files, `.pytest_cache/`, or
  `*.egg-info`. Check that `.gitignore` covers the last two.
- Run `git push origin merge-tools-monorepo`. **No force-push, and don't push
  to `master`.** If the push is rejected (e.g. the remote has moved on), stop
  and report back. Don't rebase or force.
- Finally, print `git log --oneline -5`, the pushed branch, and a short summary
  of the new command's flags.
