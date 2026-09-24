# scale-star — rescale star coordinates between pixel sizes

Part of [tomo_toolshed](../README.md). Rescales the **particle coordinates** of a
star file from one pixel size to another (e.g. after re-binning tomograms),
optionally applies a shift, and rewrites the coordinate pixel-size fields to the
new value. It reads star files the same way
[`split-star`](split_star.md) does, so it supports the same four flavors:
**RELION 3**, **RELION 4**, **RELION 5**, and **M/Warp**.

## What it does

1. Reads the star file, keeping every block (optics/general carried through).
2. Computes `factor = input-pixel-size / output-pixel-size`.
3. For every coordinate column, sets `new = old × factor + shift`, where `shift`
   is the per-axis `--shift-*` value **in output pixels** (added *after* scaling).
4. Sets the coordinate pixel-size columns to `--output-pixel-size`.
5. Writes the result to `--output`, preserving block/column order and row count.

### Scaling formula

```
factor = input-pixel-size / output-pixel-size
new_coord = old_coord × factor + shift        # shift in output pixels
```

Halving the pixel size (10 Å → 5 Å) gives `factor = 2`, so coordinates double —
the picks now index a grid with twice as many pixels across.

### Coordinate columns scaled

- **RELION 3/4/5:** `rlnCoordinateX`, `rlnCoordinateY`, `rlnCoordinateZ`.
- **M/Warp:** every `wrpCoordinateX<n>`, `wrpCoordinateY<n>`, `wrpCoordinateZ<n>`
  (there can be several `<n>`).

Files with only X/Y (2D) are handled without error. If no coordinate columns are
found, the tool errors out rather than writing an unchanged file.

### Pixel-size columns overwritten

Wherever they appear (particles or optics block, every row), these are set to
`--output-pixel-size`:

- `rlnPixelSize`
- `rlnMicrographPixelSize`
- `rlnTomoTiltSeriesPixelSize`

### Deliberately left alone

- **`rlnImagePixelSize`** — this is the *extracted subtomogram box* pixel size, a
  separate quantity from the coordinate pixel size, so it is never touched.
- **Every `*Angst` column** — `rlnOriginXAngst/Y/Z`, `rlnCenteredCoordinate*Angst`,
  `rlnHelicalTrackLengthAngst`, etc. These are already in Ångström and do not
  scale with the pixel size.
- Every other column and block is passed through unchanged, in order.

## Pixel-size mismatch check

Before scaling, the tool compares the pixel size already recorded in the file
against `--input-pixel-size`. If they disagree it prints a warning listing each
column/block and the value(s) found, then asks for confirmation:

```
Proceed using --input-pixel-size=<value>? [y/N]
```

- Answering **n** aborts (non-zero exit, no output written).
- Answering **y** proceeds using **your `--input-pixel-size`** — your value always
  wins.
- `--yes`/`-y` skips the prompt but still prints the warning.
- `--dry-run` prints the warning and never prompts.

If the file has no pixel-size column at all (M/Warp), the input pixel size can't
be checked; the tool prints a note and proceeds. It never adds pixel-size columns
that weren't already there.

## Usage

```bash
tomo_toolshed scale-star --i INPUT.star --o OUTPUT.star \
    --input-pixel-size FLOAT --output-pixel-size FLOAT [options]
```

Examples:

```bash
# Plain rescale: 10 Å -> 5 Å (coordinates double)
tomo_toolshed scale-star --i picks.star --o picks_bin1.star \
    --input-pixel-size 10 --output-pixel-size 5

# Rescale plus a shift, applied after scaling, in output pixels
tomo_toolshed scale-star --i picks.star --o picks_shifted.star \
    --input-pixel-size 10 --output-pixel-size 5 --shift-z 2

# Scripted (no prompt even if the file's pixel size disagrees)
tomo_toolshed scale-star --i picks.star --o picks_bin1.star \
    --input-pixel-size 10 --output-pixel-size 5 --yes

# See what would change without writing anything
tomo_toolshed scale-star --i picks.star --o picks_bin1.star \
    --input-pixel-size 10 --output-pixel-size 5 --dry-run
```

## Options

| Option | Default | Meaning |
|---|---|---|
| `--input PATH`, `--i` | *(required)* | Input star file (must exist). |
| `--output PATH`, `--o` | *(required)* | Output star file; parent dirs are created. Must differ from `--input`. |
| `--input-pixel-size FLOAT`, `--input_pixel_size` | *(required)* | Pixel size (Å) the coordinates are currently in. Must be > 0. |
| `--output-pixel-size FLOAT`, `--output_pixel_size` | *(required)* | Target pixel size (Å). Must be > 0. |
| `--shift-x`, `--shift-y`, `--shift-z` FLOAT (`_` spellings too) | `0` | Shift added **after** scaling, in **output pixels**. |
| `--yes`, `-y` | *(off)* | Skip the pixel-size mismatch confirmation prompt. |
| `--comment / --no-comment` | `--comment` | Prepend a provenance comment (source, pixel sizes, factor, shifts). |
| `--dry-run` | *(off)* | Report what would change; write nothing. |
| `--quiet`, `-q` | *(off)* | Only print the final one-line summary. |

## Behavior notes

- **Shift units are output pixels**, and the shift is added *after* scaling, so
  the coordinate is expressed in the target grid throughout.
- **Row count, block names, block order, and every non-coordinate column** are
  preserved; the output round-trips like the input.
- The provenance comment matches the `split-star` style and records the input
  file, both pixel sizes, the scale factor, and the shifts.
