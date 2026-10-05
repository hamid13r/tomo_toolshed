# Prompt for Claude Code — integrate `write_ebt` into tomo_toolshed

Integrate the two unintegrated scripts in `src/tomo_toolshed/write_ebt/` into the
existing `tomo_toolshed` CLI as a new subcommand. Follow the conventions already
established by `add_defocus/`, `filament_tracer/` and `dipole2star/`
(subpackage = `__init__.py` + `cli.py` + `core.py`, click command in `cli.py`,
pure logic in `core.py`, tests under `tests/<tool>/`, docs page at
`docs/<tool>.md`, README table row).

## 1. Inspect first

Read both files before changing anything:

- `src/tomo_toolshed/write_ebt/write_ebt_linux` (no `.py` extension)
- `src/tomo_toolshed/write_ebt/write_ebt_windows.py`

They are the same script twice, differing only in how the `.st` path is built:

| | linux | windows |
|---|---|---|
| separator | `"/"` | `"\\"` (one backslash) |
| escaping | none | `.replace("\\", "\\\\")` then `.replace(":", "\:")` |
| extras | — | a stray `print(directories)` |

Both also: write the header with `open(o, 'w')`, then `os.listdir()` the current
directory, then append one block of `meta.row.ebtN.*` / `meta.ref.ebtN` lines per
entry with a 1-based identifier, ending each block with
`meta.ref.ebt.lastID=ebt{identifier}`.

## 2. Preserve the git history of both scripts

The two files are currently untracked. Before refactoring:

1. `git add` them and commit as-is ("add raw write_ebt scripts").
2. Then use `git mv` to move `write_ebt_linux` → the new module that inherits its
   logic (e.g. `core.py`) and `write_ebt_windows.py` → `cli.py` (or whichever
   mapping keeps the larger diff on the file it came from), and edit in place.

`git log --follow` on the resulting files must show the pre-refactor commit.
Do not create the new files by hand and delete the originals.

## 3. Build the subcommand

One command, with the platform as a user choice — not two commands:

```python
@click.option('--platform', type=click.Choice(['linux', 'windows']),
              prompt='Target platform for the .st paths', ...)
```

Use `prompt=` so that running `tomo_toolshed write-ebt` with no arguments asks
interactively, matching the existing `--o` option's style. **Do not** auto-detect
the host OS as the default: the file is often written on one machine and opened by
etomo on another, which is the whole reason there were two scripts.

Keep `--o` exactly as it is today (same name, same `prompt=`, same help text) so
existing muscle memory and any scripts keep working.

In `core.py`, express the difference as a single path-formatting function selected
by platform, e.g. `format_st_path(directory, root, platform)`, so the two variants
share one code path. The **byte-for-byte output must be identical** to what the
corresponding original script produces for the same input directory, including:

- the header block exactly as-is, leading indentation and all (these are Java
  `.properties` lines — etomo strips the leading whitespace on read; do not
  reformat or dedent it, and do not "fix" the odd keys like
  `meta.dataset.fcaleFromZ`);
- the same line order within each row block;
- the windows escaping applied in the same order (double the backslashes first,
  then escape the colon as `\:`).

Drop the stray `print(directories)`.

### One behavior change, which I want called out explicitly

The originals call `open(o, 'w')` *before* `os.listdir()`, so the `.ebt` file they
just created is itself picked up as a row — and so is every other loose file in the
directory, despite the `# Find all directories ending with "mrc"` comment. Fix this
by including **only subdirectories** in the row loop. Mention this in the commit
message and in the final summary, since it changes output for any directory that
contains files.

Also: lift the hardcoded `meta.RootName=batchMay08-150740` into an option
(`--root-name`) whose default is that exact string, so current output is unchanged.

Add `--root` / `-r` (default `.`) so the tool can be pointed at a directory other
than the cwd. Keep the default behavior identical to today's.

## 4. Register it

In `src/tomo_toolshed/cli.py`:

- import the command and `tomo_toolshed.add_command(write_ebt, name="write-ebt")`
  to match the dashed convention (`add-defocus`, `trace-filaments`);
- also register the same command a second time as a hidden alias named
  `write_ebt` (`hidden=True` on a copy, or `add_command(..., name="write_ebt")`
  plus keeping it out of the help index) so that the underscored spelling I
  actually type still works. Flag this if you think it's a bad idea.
- add a one-line entry to the group docstring's `\b` index block, in the same
  aligned style as the others.

No new dependencies — `click` and the stdlib are enough, so `pyproject.toml`
needs no change. Confirm that rather than assuming it.

## 5. Tests

`tests/write_ebt/test_write_ebt.py`, using `pytest` + `tmp_path` +
`click.testing.CliRunner`. Cover:

- a tmp directory with a few subdirectories and a loose file → linux output has one
  row block per subdirectory, in `os.listdir()` order, and no row for the loose file
  or for the output `.ebt` itself;
- the windows variant → paths use doubled backslashes and `\:` after the drive letter;
- the header is written verbatim and `meta.ref.ebt.lastID` ends at the last row;
- `--platform` rejects anything other than `linux`/`windows`.

Tests must run headless (no matplotlib, no display), like the rest of the suite.

## 6. Docs

- `docs/write_ebt.md`: what an `.ebt` batchruntomo project file is, when you'd
  generate one, the directory layout it expects (one subdirectory per tilt series,
  containing `<name>.st`), both platform examples, and the note that `--platform`
  should match the machine that will *open* the file in etomo, not the one writing it.
- README: add a row to the tools table linking `docs/write_ebt.md`, in the same
  one-line-description style as the existing rows.

## Verify before finishing

- `pip install -e ".[test]"` then `pytest` — full suite passes, not just the new tests.
- `tomo_toolshed --help` lists `write-ebt` and the command is still spelled with
  underscores; `tomo_toolshed write-ebt --help` shows `--o`, `--platform`,
  `--root`, `--root-name`.
- Diff the new output against the originals on a scratch directory of real
  subdirectories: run each original script and the new subcommand, and show that
  the only differences are the ones listed above (no self-row, no loose-file rows).
- `git log --follow` on the moved files shows the pre-refactor commit.
- Grep for leftovers: no `write_ebt_linux` / `write_ebt_windows` references, no
  broken relative links in README or docs.
- `git status` clean. Print the final tree of `src/tomo_toolshed/write_ebt/`.

Do **not** push. Show me the commits for review first.
