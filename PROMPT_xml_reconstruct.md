# Prompt for Claude Code: review, add `xml-reconstruct`, merge to master, plan the first release

Work in this repo (`tomo_toolshed`, branch `merge-tools-monorepo`). There are four
phases. Each one ends with a checkpoint where you stop and show me the result.
Do not start the next phase until I say so.

The tool I'm adding lives in https://github.com/hamid13r/warp_recont_test. Inside
the toolshed it will be called `xml_reconstruct`: subpackage
`src/tomo_toolshed/xml_reconstruct/`, command `tomo_toolshed xml-reconstruct`,
and a hidden `xml_reconstruct` alias like the other tools have.

---

## Phase 0: Preflight

- `git status` must be clean apart from the untracked `PROMPT_*.md` files. Leave
  those alone: don't commit them and don't delete them. If there's a stale
  `.git/index.lock` and no git process is running, tell me before you remove it.
- Confirm `master` is an ancestor of `merge-tools-monorepo`, i.e. a fast-forward
  merge is possible. Run `git fetch` and check the same against `origin/master`.
- Run `pip install -e ".[test]"` in a fresh venv, then run `pytest`. Record the
  baseline pass count.

## Phase 1: Review and clean up the tool in its own repo first

Clone `warp_recont_test` into a scratch dir outside this repo. Read all of it:
code, README, deps, tests, and any example data. Check the default branch name.
Don't assume it.

Write a short review (a bulleted list, with file:line for each point) covering:

- **Bugs**: wrong logic, unhandled edge cases, silent failures, off-by-ones in
  tilt or frame indexing, wrong units (Å vs px), and file handles that never get closed.
- **XML handling**: does it preserve the Warp XML elements and attributes it doesn't
  touch? Does it round-trip a file unchanged when nothing is edited? Is the
  encoding/declaration kept? Compare with how `skipped_views` uses lxml.
- **CLI**: options, defaults, help text, and whether user errors raise
  `click.ClickException` instead of producing tracebacks.
- **Cleanup**: dead code, hard-coded paths, debug prints, duplicated helpers
  (especially anything already in `tomo_toolshed`, like the Warp XML parsing in
  `skipped_views` or `add_defocus`), unused deps, and heavy deps that break the
  toolshed's "lightweight" rule.
- **Tests**: what's covered, what isn't, and whether there are fixtures.

Then fix things **in the scratch clone**, one commit per logical change, so the
cleanup history comes along with the graft:

- Fix the bugs. Add a regression test for each fix.
- Split the code into `core.py` (pure logic, no click) and `cli.py` (click
  command), the same pattern the other tools use.
- If there are no tests, add a minimal one: build a tiny XML fixture, run the
  tool, and assert on the output.

Keep the existing CLI flags and semantics, with one exception: anything that is
clearly a bug. List every user-visible change separately so I can approve it.

**Checkpoint 1:** show me the review, the list of commits, the list of
user-visible changes, and the test results. Wait for my OK.

## Phase 2: Integrate into tomo_toolshed

Follow the "Adding a tool" convention in `README.md` and the way earlier tools
were grafted in:

1. Graft the cleaned clone **with history** using
   `git subtree add --prefix=_incoming_xmlrec <scratch-clone-path> <branch>`.
   Then `git mv` the code to `src/tomo_toolshed/xml_reconstruct/` and the tests to
   `tests/xml_reconstruct/`. Remove `_incoming_xmlrec`. Use only `git mv`: no copying
   by hand. Drop the nested LICENSE, pyproject, requirements, and .gitignore,
   merging anything useful from them into the root files.
2. Fix imports so they use `tomo_toolshed.xml_reconstruct.*`. Reuse existing shared
   helpers wherever behavior is identical. Don't fork them.
3. Register it in `src/tomo_toolshed/cli.py`:
   - `add_command(..., name="xml-reconstruct")`
   - a hidden `xml_reconstruct` alias
   - a one-line entry in the group docstring index
4. Add any new deps to the flat `dependencies` list in `pyproject.toml` and to
   `environment.yml`. No extras. If a dep is heavy, ask me first.
5. Write `docs/xml_reconstruct.md` with usage, every option, and examples using
   `tomo_toolshed xml-reconstruct`. Fix the image paths. Add a row to the tool
   table in the root `README.md`.

Verify:

- Fresh venv: `pip install -e ".[test]"` works, and `pytest` passes, with baseline
  plus the new tests.
- `tomo_toolshed --help` lists `xml-reconstruct`, and both spellings run.
- Diff `tomo_toolshed xml-reconstruct --help` against the cleaned standalone tool:
  no option has been dropped or renamed.
- `git log --follow` on the moved core file shows its pre-merge history.
- Grep for old import paths, the old script name, and `warp_recont_test`. Nothing
  should be left except intentional mentions in the docs.
- No broken relative links in `README.md` or `docs/`.
- `git status` is clean, apart from the untracked `PROMPT_*.md` files.

**Checkpoint 2:** show me the new commits (`git log --oneline master..HEAD`), the
`--help` output, and the test results. Wait for my OK.

## Phase 3: Merge into master and push

Note: this repo's default branch is `master`, not `main`. Merge into `master`.
Don't create a `main` branch unless I ask.

1. `git fetch origin`. If `origin/master` has moved, stop and tell me.
2. `git checkout master && git merge --ff-only merge-tools-monorepo`. If
   fast-forward isn't possible, stop and tell me: no merge commits, no rebases,
   and no force-pushes without asking.
3. Repeat the fresh-venv install and `pytest` on `master`.
4. Push `master`, then `merge-tools-monorepo`. Never use `--force`.
5. Ask me before deleting any remote branches (`merge-tools-monorepo`,
   `delete-mode`).

**Checkpoint 3:** show me `git log --oneline -5 origin/master` and confirm local
and remote match.

## Phase 4: First-release to-do list

Don't release anything. Audit the repo and write `docs/RELEASE_CHECKLIST.md`.
Commit it on a new branch, `release-prep`, without pushing. It should be a
checklist of concrete items, each marked **done**, **todo**, or **decide**
(i.e. needs my call), with a one-line reason. At minimum it should cover:

- **Version and naming**: the `0.1.0` version, version source (static vs
  `setuptools_scm`), whether `tomo-toolshed` is available on PyPI and TestPyPI,
  and whether the `tomo_toolshed --version` output is correct.
- **Metadata**: authors and email, URLs (homepage, issues), classifiers, the
  `requires-python` floor checked against the code, and lower bounds on deps
  (for example, the `starfile` and `click` APIs actually in use).
- **Packaging**: `python -m build` produces an sdist and a wheel; installing the
  wheel in a clean venv works and does not include tests or fixtures; `twine check`
  passes; and any package data is included.
- **CI**: a GitHub Actions workflow running pytest on a Python version matrix, plus
  a tag-triggered publish workflow (trusted publishing).
- **Docs**: every tool has a docs page with a working example; the README index is
  complete; RELION 4.x is stated as the output target (RELION 5 via a future
  converter); and there's a known-limitations section.
- **Housekeeping**: `CHANGELOG.md`, `CITATION.cff` and/or a Zenodo DOI, what to do
  with the stray `PROMPT_*.md` files, stale branches, the `.gitignore` audit, and a
  check that no large files are in history.
- **Quality**: a smoke test running `--help` for each subcommand; consistent
  error handling across tools; and dual dash/underscore option spellings applied
  consistently.
- **Release steps**: tag `v0.1.0`, a GitHub release with notes, publish to
  TestPyPI, then PyPI, and an optional conda-forge recipe later.

Where you can check an item right now (run `build`, `twine check`, the wheel
install, the version output), do it and record the result in the checklist.

**Checkpoint 4:** show me the checklist and say which items are blockers.
