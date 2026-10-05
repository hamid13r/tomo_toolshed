# Prompt for Claude Code — make `dipole2star --outdir` required and self-creating

Change the output-directory behavior of the `dipole2star` subcommand in
`tomo_toolshed`. Today `--outdir` silently defaults to `.`, so a run that forgets
it scatters `<stem>.mrc.star` files into whatever directory the user happened to
be standing in, and a run that passes a directory that doesn't exist yet fails
deep inside the write. Both should go away.

Desired behavior:

1. `--outdir` is **required** — the user must say where the output goes. No
   default, no implicit current directory.
2. The directory is **created if it doesn't exist**, parents included. Pointing
   at `runs/2026-09/picks/` on a fresh filesystem should just work.

This is a small, surgical change. Do not redesign the command, rename other
flags, or touch the other subcommands.

## 1. Inspect first

Read these before changing anything:

- `src/tomo_toolshed/dipole2star/cli.py` — the click command; `--outdir` is
  declared around line 30 as `default='.', show_default=True` with no `type=`.
- `src/tomo_toolshed/dipole2star/core.py` — `convert_one(path, scale, random,
  pixel_size, outdir, *, micrograph_suffix, fmt, rng)`. It builds
  `outpath = Path(outdir) / f"{stem}.mrc.star"` and calls `starfile.write(...)`.
  Nothing creates the directory.
- `src/tomo_toolshed/split_star/cli.py` — the in-repo convention for this flag:
  `@click.option("--outdir", "-o", type=click.Path(file_okay=False), ...)`.
  Match that spelling and type (but required, unlike split-star).
- `tests/dipole2star/test_dipole2star.py` — several tests pass `--outdir`, and
  `test_cli_glob_expansion_and_outdir` pre-creates the directory with `.mkdir()`.
- `docs/dipole2star.md` — prose reference to `--outdir`, two usage examples, and
  an options table row `| `--outdir PATH` | `.` | ... |`.

## 2. Make the change

In `core.py`, put the directory creation in `convert_one` — not in the CLI — so
library callers get the same guarantee:

```python
outdir = Path(outdir)
outdir.mkdir(parents=True, exist_ok=True)
outpath = outdir / f"{stem}.mrc.star"
```

Update the `convert_one` docstring to say the directory is created if missing.
Keep the signature and return value (`(outpath, n_particles)`) exactly as they
are — `outdir` stays a positional parameter.

In `cli.py`, make the option required and directory-typed:

```python
@click.option('--outdir', '-o', required=True,
              type=click.Path(file_okay=False),
              help="Directory to write the output star files to "
                   "(created if it does not exist).")
```

Note the added `-o` short form, for consistency with `split-star`. Drop
`show_default` (there is no default any more). Leave every other option's flags,
defaults and semantics untouched.

Then update the command docstring: the first example
(`tomo_toolshed dipole2star picks.star --scale 2 --output-apix 9.98`) no longer
runs, since it relies on the removed default. Give both examples an explicit
`--outdir`.

## 3. Tests

Add to `tests/dipole2star/`:

- `--outdir` pointing at a nested path that does not exist (e.g.
  `tmp_path / "a" / "b" / "c"`) — the run succeeds and the star files land there.
- Omitting `--outdir` entirely — `CliRunner` exit code is non-zero and the
  message names the missing option. This is the regression guard for "never
  writes to the cwd by accident".
- A direct `convert_one(...)` call with a nonexistent `outdir`, to prove the
  creation lives in core and not just in the CLI.

Then fix the existing tests: `test_cli_glob_expansion_and_outdir` no longer needs
its `outdir.mkdir()` line (drop it — the point is that the tool does this now),
and confirm any test that called `convert_one` or the CLI without `--outdir`
still passes.

## 4. Docs and README

- `docs/dipole2star.md`: update the options table row — `--outdir PATH` has no
  default and is required; say it is created if missing. Update both usage
  examples to pass `--outdir`, and check the prose around line 43 still reads
  correctly.
- Root `README.md`: check whether the `dipole2star` row or any example shows a
  `--outdir`-less invocation, and fix it if so.
- Grep the whole repo for `dipole2star` invocations that omit `--outdir` —
  docstrings, docs, README, other PROMPT files — and fix every one. Any surviving
  example is now a command that errors out.

## Verify before finishing

- `pytest tests/dipole2star/` passes; `pytest` overall passes.
- `tomo_toolshed dipole2star --help` shows `--outdir` as required, with `-o`, and
  no `[default: .]`.
- Running without `--outdir` exits non-zero and writes nothing to the cwd.
- Running with a nonexistent nested `--outdir` creates it and writes the files.
- `git diff` touches only `dipole2star/`, its tests, its docs page, and the
  README. Diff the rest of `dipole2star --help` against the pre-change output and
  confirm no other option changed.

Note this is a **breaking CLI change** — call it out in the commit message. One
commit is fine. Do not push; show me the commit for review.
