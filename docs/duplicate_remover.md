# duplicate-remover — remove particles closer than a threshold

Part of [tomo_toolshed](../README.md). Within each group (tomogram, micrograph, or
M source), any two particles closer than a distance threshold are treated as
duplicate picks of the same object; the tool removes the worse of every too-close
pair until none remain, then writes a cleaned star file.

## What it does

1. Reads the star file (any block layout — see [Star flavors](#star-flavors)).
2. Resolves the **pixel size of the coordinate columns** so distances can be
   computed in Ångström (this is the subtle part — see the
   [precedence table](#pixel-size-precedence)).
3. Computes each particle's position in Å: `coordinate × pixel_size − origin`.
4. Splits particles by the grouping column and, within each group, finds all pairs
   closer than the threshold and removes the **worse** member of each (by the
   comparison metric), worst first, until no pair remains.
5. Writes the cleaned star file, **preserving the input row order and every input
   block** (including RELION 5's `data_general`), plus an optional histogram of the
   removed-particle distances.

The removal is deterministic for the non-random metrics (ties break on
coordinates, not on row order).

## Pixel-size precedence

The threshold is in Ångström, so the tool needs the Å/px scale **of the coordinate
columns** — which is not always the most obvious pixel-size field in the file. The
resolver applies this precedence and prints which rule fired:

| # | Condition | Pixel size used | Why |
|---|---|---|---|
| 1 | **M / WarpTools** (`wrp*` columns) | `1.0` | `wrpCoordinate*` are already in Å; there is no pixel size in the file. |
| 2 | `--pixel-size` given (non-M) | the override | Explicit user override wins over the file. |
| 3 | **RELION 5** — optics has `rlnTomoTiltSeriesPixelSize` | that column | Coordinates are in unbinned tilt-series pixels. `rlnImagePixelSize` is the *extracted-image* scale (= tilt-series × `rlnTomoSubtomogramBinning`) and would double every distance. |
| 4 | **RELION 4** — optics has `rlnImagePixelSize`, no tilt-series column | `rlnImagePixelSize` | The particles table may also carry a **stale** `rlnPixelSize` from coarser picking — it is ignored. |
| 5 | **RELION 3** — no optics block | per-particle `rlnPixelSize`, else `rlnDetectorPixelSize / rlnMagnification × 1e4` | No optics block to read a scale from. |

The pixel size is resolved **per optics group** (merged on `rlnOpticsGroup`), so a
file with several optics groups is handled correctly; if groups disagree, each
row's own value is used and the report says so.

Notes for M / WarpTools files: `--pixel-size` is **ignored** (with a warning) since
coordinates are already in Å, and `--units pixels` is a hard error (there is no
pixel size to convert with — give `--threshold` in Å instead).

## Star flavors

| Flavor | Blocks | Grouping column | Coordinates | Origins |
|---|---|---|---|---|
| RELION 3 | one unnamed `data_` | `rlnMicrographName` | `rlnCoordinateX/Y/Z` | `rlnOriginX/Y/Z` (pixels), if any |
| RELION 4 | `optics` + `particles` | `rlnMicrographName` | `rlnCoordinateX/Y/Z` | `rlnOriginX/Y/ZAngst` (Å) |
| RELION 5 | `general` + `optics` + `particles` | `rlnTomoName` | `rlnCoordinateX/Y/Z` | `rlnOriginX/Y/ZAngst` (Å) |
| M / WarpTools | one unnamed `data_` | `wrpSourceName` | `wrpCoordinateX<N>/Y<N>/Z<N>` | none |

Grouping falls back through `rlnTomoName → rlnMicrographName → wrpSourceName` (first
present), overridable with `--group-by`. The `wrp` coordinate suffix `<N>` is
detected, not hardcoded. Origins are applied when present; if only some of the
three axes exist, the missing ones are treated as 0 with a warning.

`starfile` handles the `# version` headers on read and write; the tool never
hand-parses the file.

## Usage

```bash
tomo_toolshed duplicate-remover --star-path run_data.star [options]
```

Run without `--quiet` and it prints a provenance block (flavor, grouping column,
coordinate columns, pixel size and where it came from, origins, metric, and the
threshold in both Å and px) before the per-group counts.

### RELION 5 example

```bash
tomo_toolshed duplicate-remover --star-path run_it025_data.star --threshold 140
```

```
detected flavor:  relion5
grouping column:  rlnTomoName (5 groups)
pixel size:       2.11 Å/px  [optics rlnTomoTiltSeriesPixelSize (per rlnOpticsGroup)]
origins:          rlnOrigin*Angst (Å)
comparison metric:rlnLogLikeliContribution  [auto -> rlnLogLikeliContribution]
threshold:        140 Å = 66.3507 px  (given as 140 angstroms)
```

### M / WarpTools example

Coordinates are already in Å, so give the threshold in Å (the default units):

```bash
tomo_toolshed duplicate-remover --star-path particles.star --threshold 80
```

`--units pixels` and `--pixel-size` do not apply to these files.

## Options

| Option (aliases) | Default | Meaning |
|---|---|---|
| `--star-path` (`--star_path`) | *(required)* | Input star file. |
| `--output-path` (`--output_path`) | `<name>_cleaned.star` | Output path (built with `pathlib`, so a `.star` elsewhere in the path is safe). |
| `--threshold` (`--distance_threshold`) | `140` | Distance below which particles are duplicates. |
| `--units {angstroms,pixels}` | `angstroms` | Units of `--threshold`. `pixels` multiplies by the resolved pixel size; internal math stays in Å. |
| `--pixel-size` (`--pixel_size`) | *(from file)* | Override the resolved Å/px (ignored for M/WarpTools). |
| `--group-by` (`--group_by`) | *(auto)* | Override the grouping column. |
| `--comparison-metric` (`--comparison_metric`) | `auto` | `auto`, `rlnLogLikeliContribution`, `rlnMaxValueProbDistribution`, `random`, or `first`. `auto` = LogLikeli → MaxValueProb → keep-first. |
| `--seed` | *(unseeded)* | Seed for `--comparison-metric random`. |
| `--histogram / --no-histogram` | on | Save a histogram of removed-particle distances next to the output. |
| `--dry-run` | off | Report what would be removed; write nothing. |
| `--quiet` / `-q` | off | Only print the final summary. |

## Behavior notes

- **Threshold is exclusive**: a pair *exactly* at the threshold distance is kept.
- **Row order is preserved** in the output (particles are not regrouped by
  tomogram).
- **Which particle is kept**: the one with the higher metric value; for `first`,
  the earlier row; for `random`, a coin flip (seed with `--seed`). A removed
  particle never causes a second removal, and both members of a pair are never
  removed together.
- **Edge cases**: groups with fewer than two particles, empty groups, and
  no-removal runs are all handled without error.
