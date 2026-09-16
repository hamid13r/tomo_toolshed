# split-star — split a star file into per-group star files

Part of [tomo_toolshed](../README.md). Groups a particle `.star` file by its
grouping column (tomogram / micrograph / source) and writes **one star file per
group**, each in its own subdirectory. It is the natural inverse of gathering many
per-tilt-series star files into a single project file.

## What it does

1. Reads the star file (any block layout).
2. Detects the grouping column: the first present of `rlnTomoName`,
   `rlnMicrographName`, `wrpSourceName` (override with `--group-by`).
3. For each distinct group value, writes that group's rows to
   `<outdir>/<label>_<name>/<label>_<name>_all.star`, where `<name>` is the group
   value with `--strip-suffix` (default `.mrc.tomostar`) removed.
4. Any non-particles blocks (`optics`, `general`) are carried through into every
   output, so multi-block RELION 4/5 files split into valid RELION files; a
   single-unnamed-block input yields single-unnamed-block outputs.

## Directory layout produced

For `--label EXP` on a file with micrographs `ts_01.mrc.tomostar` and
`ts_02.mrc.tomostar`:

```
EXP_ts_01/
└── EXP_ts_01_all.star
EXP_ts_02/
└── EXP_ts_02_all.star
```

Omit `--label` and there is no prefix (`ts_01/ts_01_all.star`).

## Usage

```bash
tomo_toolshed split-star --i run_data.star --label EXP
```

Examples:

```bash
# Split by micrograph (auto-detected), labelled EXP, into the current directory
tomo_toolshed split-star --i run_data.star --label EXP

# Split a RELION 5 file by tomogram into a chosen output root
tomo_toolshed split-star --i run_it025_data.star --group-by rlnTomoName --outdir split

# Preview without writing
tomo_toolshed split-star --i run_data.star --label EXP --dry-run
```

## Options

| Option (aliases) | Default | Meaning |
|---|---|---|
| `--input` (`--i`) | *(required)* | Input star file to split. |
| `--label` | *(none)* | Prefix added to each output directory and file. Omit for no prefix. |
| `--outdir` (`-o`) | `.` | Directory to write the per-group subdirectories into. |
| `--group-by` (`--group_by`) | *(auto)* | Override the grouping column. |
| `--strip-suffix` (`--strip_suffix`) | `.mrc.tomostar` | Suffix stripped from each group name to form its base name (only when it is at the end of the name). |
| `--dry-run` | off | List what would be written; create nothing. |
| `--quiet` / `-q` | off | Only print the final summary. |

## Behavior notes

- **Groups are written in first-appearance order**; within each group, the rows
  keep their original order.
- **Every row lands in exactly one output** — the groups partition the input.
- **Existing files are overwritten.**
- The `--strip-suffix` is only removed when it is a true suffix of the name; a name
  that does not end with it is used unchanged (so mixed inputs are safe).
