# trace-filaments — filament tracer

Part of [tomo_toolshed](../README.md). Traces filaments in a **binary
segmentation mask** and exports a RELION-style **helical `.star`** file of evenly
spaced particles, each carrying a ZYZ Euler orientation derived from the local
filament tangent. Designed to run non-interactively so it can be batched over
many tomograms.

## What it does

1. Reads a binary mask MRC and separates it into 26-connected components.
2. Skeletonizes each component and extracts its longest-path centerline via a
   graph (endpoints → farthest-pair shortest path).
3. Drops components below `--min-voxels` and centerlines shorter than
   `--min-length` (Å).
4. Pre-smooths the ordered points, fits a smoothing spline (`--smooth-factor`),
   and resamples each filament at `--spacing` Å.
5. Computes a per-particle tangent → rotation matrix → ZYZ Euler angles
   (`rlnAngleRot/Tilt/Psi`, with matching `*Prior` columns), grouped per
   filament via `rlnHelicalTubeID` and `rlnHelicalTrackLength`.
6. Writes the particles to the output star file.

## Inputs and outputs

- **Input:** a binary mask `MASK` (`.mrc`); foreground is any voxel `> 0`.
- **Output:** a helical particle star file (default `particles.star`) with
  columns: `rlnCoordinateX/Y/Z`, `rlnMicrographName`, `rlnMagnification`,
  `rlnPixelSize`, `rlnGroupNumber`, `rlnAngleRot/Tilt/Psi`,
  `rlnAngleRot/Tilt/PsiPrior`, `rlnHelicalTubeID`, `rlnHelicalTrackLength`.
- **Optional:** a ChimeraX `.bild` overlay (spheres + tangent arrows) for visual
  QC — **only written when `--bild` is passed**.

## Usage

```bash
tomo_toolshed trace-filaments MASK [options]
```

Example:

```bash
# Primary output only (no .bild)
tomo_toolshed trace-filaments mask.mrc -o particles.star --pixel-size 9.98

# Same run, plus a ChimeraX overlay next to the star file
tomo_toolshed trace-filaments mask.mrc -o particles.star --pixel-size 9.98 --bild
```

## Options

| Option | Default | Meaning |
|---|---|---|
| `MASK` (argument) | *(required)* | Input binary mask MRC. |
| `--output, -o PATH` | `particles.star` | Output helical star file. |
| `--pixel-size FLOAT` | *(MRC header)* | Å/px; read from the MRC header if omitted. |
| `--spacing FLOAT` | `82.0` | Å between particles along a filament. |
| `--min-voxels INT` | `50` | Drop connected components smaller than this. |
| `--min-length FLOAT` | `500.0` | Drop centerlines shorter than this (Å). |
| `--smooth-factor FLOAT` | `1.0` | Spline smoothing: `s = smooth-factor * n_points` (`0` = interpolate). |
| `--presmooth-window INT` | `3` | Moving-average window on ordered points (odd; `1` = off). |
| `--micrograph TEXT` | *(mask file name)* | `rlnMicrographName` value written to every row. |
| `--magnification FLOAT` | `10000` | `rlnMagnification` value. |
| `--group-number INT` | `1` | `rlnGroupNumber` value. |
| `--tangent-axis {x,y,z}` | `z` | Particle axis aligned to the filament tangent. |
| `--invert-rot / --no-invert-rot` | `--invert-rot` | Transpose the rotation matrix before decomposition (RELION reference→particle). |
| `--bild` | *(off)* | Also write a ChimeraX `.bild` overlay. **Opt-in.** |
| `--bild-dir PATH` | *(alongside `--output`)* | Directory for the `.bild` file; only used with `--bild`. |

## ChimeraX `.bild` output is opt-in

By default **no `.bild` file is written** and nothing about the star output
depends on it. Pass `--bild` to also emit `<output-stem>.bild` (spheres at each
particle plus a tangent arrow), written alongside the star file unless
`--bild-dir` redirects it. Running with and without `--bild` produces a
byte-identical star file.
