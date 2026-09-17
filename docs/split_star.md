# split-star — split a star file into per-group star files

Part of [tomo_toolshed](../README.md). Groups a particle `.star` file by its
grouping column (tomogram / micrograph / source) and writes **one star file per
group**, each in its own subdirectory. It is the natural inverse of gathering many
per-tilt-series star files into a single project file.

## What it does

1. Reads the star file (any block layout) and detects the **flavor**
   (RELION 3/4/5 or M/WarpTools), which it reports.
2. Chooses the grouping column(s): by default the first present of `rlnTomoName`,
   `rlnMicrographName`, `wrpSourceName`. `--group-by` overrides this with **any**
   column and is **repeatable** to split by a combination (see
   [Grouping by other columns](#grouping-by-other-columns)).
3. For each distinct group value, writes that group's rows to
   `<outdir>/<label>_<name>/<label>_<name>_all.star`, where `<name>` is the group
   value with a known suffix removed (see [Name cleaning](#name-cleaning)).
4. Any non-particles blocks (`optics`, `general`) are carried through into every
   output, so multi-block RELION 4/5 files split into valid RELION files; a
   single-unnamed-block input yields single-unnamed-block outputs.

## Star flavors

Like `duplicate-remover`, this works across star versions — the flavor is detected
and reported, and each version's grouping/naming convention is handled:

| Flavor | Blocks | Grouping column |
|---|---|---|
| RELION 3 | one unnamed `data_` | `rlnMicrographName` |
| RELION 4 | `optics` + `particles` | `rlnMicrographName` |
| RELION 5 | `general` + `optics` + `particles` | `rlnTomoName` |
| M / WarpTools | one unnamed `data_` | `wrpSourceName` |

## Name cleaning

Group names may carry any of several suffixes — and they can differ from row to
row within the same file. The base name is formed by stripping the **longest
matching** of these suffixes (or the Warp/M pattern) from each name individually,
so directory names come out clean whatever the mix:

```
.mrc.tomostar           →  ts_001.mrc.tomostar         → ts_001
.tomostar               →  Position_1.tomostar         → Position_1
.mrc                    →  foo_microtubule.mrc         → foo_microtubule
.mrc_<pixelsize>Apx.mrc →  Position_1.mrc_9.98Apx.mrc  → Position_1
```

Each name is handled on its own, so one file can contain a mix of these and every
group is still named correctly. The last entry is a **pattern**, not a literal,
because the pixel size varies (`9.98`, `10`, `4.22`, …); it wins over the plain
`.mrc` so the base is not left as `Position_1.mrc_9.98Apx`.

Override with one or more `--strip-suffix` values; doing so takes full control (the
built-in defaults, including the `.mrc_<pixelsize>Apx.mrc` pattern, are then off). A
name that matches nothing is used unchanged.

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

## Grouping by other columns

`--group-by` accepts **any** column, not just the name-like ones — e.g. split by
`rlnClassNumber`, `rlnRandomSubset`, `rlnHelicalTubeID`, or `rlnOpticsGroup`:

```bash
tomo_toolshed split-star --i run_data.star --group-by rlnRandomSubset --label HALF
# -> HALF_1/HALF_1_all.star, HALF_2/HALF_2_all.star
```

Repeat `--group-by` to split by a **combination** — one output per unique tuple of
values, named `<label>_<val1>_<val2>_...`:

```bash
tomo_toolshed split-star --i run_data.star \
    --group-by rlnTomoName --group-by rlnClassNumber --label EXP
# -> EXP_<tomo>_<class>/EXP_<tomo>_<class>_all.star
```

The parts are joined with `_`. If two different combinations would collapse to the
same name (possible because values can themselves contain `_`), the tool **errors
instead of overwriting** — reorder the columns or split one column per run.

## Splitting a numeric column into ranges

`--range-by COLUMN --breaks a,b,c` splits a numeric column at the break points into
**half-open bins** `[low, high)` — a value equal to a break falls in the *upper*
bin. `N` breaks make up to `N+1` bins covering everything; bins with no particles
produce no file, so nothing is dropped:

```bash
tomo_toolshed split-star --i run_data.star \
    --range-by rlnDistanceFromtop --breaks 100,200,300 --label D
# -> D_lt100/, D_100-200/, D_200-300/, D_ge300/  (only the non-empty ones)
```

The range dimension combines with `--group-by`, e.g. per tomogram **and** distance
band:

```bash
tomo_toolshed split-star --i run_data.star \
    --group-by rlnTomoName --range-by rlnDistanceFromtop --breaks 100,200
# -> <tomo>_lt100/, <tomo>_100-200/, <tomo>_ge200/ per tomogram
```

## Provenance comment

Every output star file gets a comment header recording how it was made — the source
file, the split specification, and this file's own values/range:

```text
# Created by tomo_toolshed split-star (v0.1.0)
# source: run_data.star
# split by: rlnTomoName, range(rlnDistanceFromtop, breaks=[100, 200])
# this file: rlnTomoName=Position_1.tomostar; 100 <= rlnDistanceFromtop < 200
```

RELION and `starfile` ignore `#` comment lines, so the file still reads normally.
Disable with `--no-comment`.

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
| `--group-by` (`--group_by`) | *(auto)* | Column to split on; accepts any column and is repeatable to split by a combination. Default: the auto-detected name column. |
| `--range-by` (`--range_by`) | *(none)* | Numeric column to split into ranges at `--breaks` (half-open `[low, high)` bins). Combines with `--group-by`. |
| `--breaks` | *(none)* | Comma-separated break points for `--range-by`, e.g. `100,200,300`. |
| `--comment / --no-comment` | on | Prepend a provenance comment (source, split spec, this file's values) to each output. |
| `--strip-suffix` (`--strip_suffix`) | `.mrc.tomostar`, `.tomostar`, `.mrc`, `.mrc_<pixelsize>Apx.mrc` | Suffix(es) stripped from each group name to form its base name (repeatable; longest match wins; only stripped when it ends the name). Passing this replaces the defaults, including the Warp/M pixel-size pattern. |
| `--dry-run` | off | List what would be written; create nothing. |
| `--quiet` / `-q` | off | Only print the final summary. |

## Behavior notes

- **Groups (or combinations) are written in first-appearance order**; within each
  group, the rows keep their original order.
- **Every row lands in exactly one output** — the groups partition the input.
- **Existing files are overwritten.**
- A `--strip-suffix` is only removed when it is a true suffix of the name; a name
  that ends with none of them is used unchanged (so mixed inputs are safe).
