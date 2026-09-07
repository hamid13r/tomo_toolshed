# dipole2star — dipole picks to oriented particles

Part of [tomo_toolshed](../README.md). Collapses **manual dipole picks** into a
RELION-style **particle `.star`** file, one output per input. Each picked
particle is a *dipole*: two points marking the two ends of the particle's long
axis. The tool centers each pair and derives a ZYZ Euler orientation from the
axis direction, writing a ready-to-use oriented-particle star file. It combines
what `pos2i3.py` (pos → trf) and `trf2star.py` (trf → star) used to do as
separate steps.

## What a dipole pick is

Picks are stored as **consecutive point pairs**, two rows per particle:

- row *i* — one end of the particle's long axis
- row *i+1* — the other end

So a file with `2N` rows describes `N` particles. Each pair collapses to a
single particle whose center is the pair midpoint (scaled by `--scale`, then
truncated to whole numbers) and whose orientation aligns a reference axis to the
pick direction. A file with an **odd** number of rows is rejected.

## What it does

1. Reads picked coordinates from each input (star or 3-column text).
2. For every consecutive pair, computes the center (midpoint × `--scale`) and the
   long-axis direction.
3. Builds a rotation that maps a reference axis onto the pick direction and
   decomposes it to ZYZ Euler angles (`rlnAngleRot/Tilt/Psi`, degrees).
4. With `--random`, replaces the (arbitrary) azimuth about the pick axis with a
   random one; the axis direction is unchanged.
5. Writes one particle star file per input, named `<stem>.mrc.star`.

## Inputs and outputs

- **Input:** one or more pick files, each with an even number of rows. Two
  formats, chosen by extension (`--format auto`) or forced with `--format`:
  - **star** (`.star`) — a RELION star file with `rlnCoordinateX/Y/Z` columns.
  - **txt** (anything else) — plain 3-column `X Y Z` text: whitespace- **or**
    comma-delimited, blank lines skipped, `#` comment lines skipped, and a single
    non-numeric header row (e.g. `x y z`) skipped if present.
- **Output:** per input `<stem>`, a particle star file `<stem>.mrc.star` in
  `--outdir` with columns: `rlnCoordinateX/Y/Z`, `rlnMicrographName`,
  `rlnMagnification` (10000), `rlnPixelSize` (from `--output-apix`),
  `rlnGroupNumber` (1), `rlnAngleRot/Tilt/Psi`, `rlnClassNumber` (empty).

The same coordinates given as `.star` or as `.txt` produce identical output
(apart from `rlnMicrographName`, which is built from the file stem).

### Example inputs

A 3-column text file describing two particles:

```text
# optional comment; a header row is allowed too
x y z
10, 20, 30
14, 20, 30
100 100 100
100 108 100
```

The equivalent star file:

```text
data_

loop_
_rlnCoordinateX #1
_rlnCoordinateY #2
_rlnCoordinateZ #3
10  20  30
14  20  30
100 100 100
100 108 100
```

## Usage

```bash
tomo_toolshed dipole2star STAR_FILES... [options]
```

Examples:

```bash
# One star file
tomo_toolshed dipole2star picks.star --scale 2 --output-apix 9.98

# A glob of text files into a chosen output directory
tomo_toolshed dipole2star "picks/*.txt" --scale 2 --output-apix 9.98 --outdir out

# Randomize the about-axis angle, reproducibly
tomo_toolshed dipole2star picks.star --scale 2 --output-apix 9.98 --random --seed 42
```

## Options

| Option | Default | Meaning |
|---|---|---|
| `STAR_FILES...` (argument) | *(required)* | One or more pick files; glob patterns are expanded (a pattern matching nothing is passed through literally). |
| `--scale FLOAT` | *(required)* | Multiply picked coordinates by this factor before centering. |
| `--output-apix FLOAT` | *(required)* | Pixel size (Å/px) written into `rlnPixelSize`. |
| `--random` | *(off)* | Randomize each particle's azimuthal orientation about its pick axis. |
| `--outdir PATH` | `.` | Directory to write the output star files to. |
| `--micrograph-suffix TEXT` | `.mrc_9.98Apx.mrc` | Appended to each input's stem to form `rlnMicrographName` (match your reconstruction file names). |
| `--format {star,txt,auto}` | `auto` | Input reader. `auto` picks by extension (`.star` → star, else txt). |
| `--seed INT` | *(drawn)* | Seed for `--random`. If omitted, a seed is drawn and printed so the run can be reproduced with `--seed <value>`. |

## Behavior notes

- **Coordinates are truncated to whole numbers** (stored as floats): the center
  is `midpoint × --scale`, then each component is truncated toward zero.
- **Output naming** is always `<stem>.mrc.star`, where `<stem>` is the input file
  name without its extension. Existing files are overwritten.
- **Fixed columns:** `rlnMagnification` is `10000`, `rlnGroupNumber` is `1`, and
  `rlnClassNumber` is left empty.
- **`--random` reproducibility:** the random azimuth is drawn from a local
  generator seeded by `--seed` (or a drawn, printed seed), so a run is fully
  reproducible. The default (non-random) orientation is deterministic.

## Troubleshooting

- **`has N picks (odd) …`** — the file has an odd number of rows. Dipole picking
  needs two points per particle; check for a missing or extra pick.
- **`expected 3 numbers (X Y Z) …`** — a text row is not three parseable numbers.
  The message names the file and line. Only one leading header row (e.g. `x y z`)
  is tolerated; every other non-comment row must be three numbers.
- **`… has an axis parallel to z; its orientation is degenerate …`** — a dipole's
  two ends differ only in z, which makes the orientation frame undefined (it
  would otherwise write NaN angles). Re-pick that particle with an off-axis long
  axis. The message names the file and the pair index.
