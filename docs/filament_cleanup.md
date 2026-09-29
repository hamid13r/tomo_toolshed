# filament-cleanup

Remove particles that don't belong to their helix after a RELION helical
refinement, and write the removed ones to a separate star file so you can look
at them.

```bash
tomo_toolshed filament-cleanup --i run_data.star --o run_data_clean.star
```

This writes two files:

- `run_data_clean.star` has the particles that were kept.
- `run_data_clean_rejected.star` has the particles that were removed. You can
  change this path with `--rejected`.

Both files keep every block and column of the input, so you can open them
side by side in ChimeraX/ArtiaX, or feed the cleaned one straight back into
RELION.

## How it works

Particles are grouped into filaments by tomogram (`rlnTomoName` or
`rlnMicrographName`) and `rlnHelicalTubeID`. Within each filament they are
ordered by `rlnHelicalTrackLengthAngst`. Each filament then gets two checks.

1. **Position.** The refined position of each particle is
   `coordinate × pixel size − rlnOrigin*Angst`. A smooth curve is fitted
   through all the refined positions of the filament: a robust local-linear
   (LOWESS) fit against arclength, `--window` Å wide. Outliers are
   down-weighted, so they don't pull the curve toward themselves. A particle is
   removed when it lies more than `--max-distance` Å from the curve. Only the
   distance *perpendicular* to the curve counts, because sliding along the
   helical axis is normal in helical refinement.
2. **Orientation.** The helical axis of a particle follows from `rlnAngleTilt`
   and `rlnAnglePsi`. `rlnAngleRot` is the spin about the axis, so it is
   ignored. Each axis is compared to the robust mean axis of `--neighbours`
   particles on each side. A particle is removed when the angle is larger than
   `--max-angle`.

**Polarity.** By default the orientation check ignores polarity: an axis and
its 180° flip (psi+180, tilt → 180−tilt) count as the same line. With
`--remove-flipped`, the tool also removes the minority polarity of each
filament. Check first whether your refinement actually resolved polarity. In
the example microtubule data, roughly 40% of each filament is flipped
(`rlnAnglePsiFlipRatio 0.5`), so `--remove-flipped` would remove nearly half
the particles.

Filaments with fewer than `--min-particles` particles are too short to judge,
so they are kept as they are.

The report prints the median, p90, p99 and max of both measures, plus the most
affected filaments. Run `--dry-run` first and pick your thresholds from those
numbers.

## Pixel size

The coordinate pixel size is resolved the same way as in
[duplicate-remover](duplicate_remover.md):

- **RELION 4:** optics `rlnImagePixelSize`. The per-particle `rlnPixelSize`
  that Warp writes can be stale, so it is ignored.
- **RELION 5:** `rlnTomoTiltSeriesPixelSize`. Files with only
  `rlnCenteredCoordinate*Angst` are used directly in Å.
- **RELION 3:** `rlnPixelSize`.

Use `--pixel-size` to override any of these. M/WarpTools files are rejected
because they have no helical labels.

## Options

| Option | Default | Meaning |
|---|---|---|
| `--input`, `--i` | required | Helical-refinement star file. |
| `--output`, `--o` | required | Cleaned star file. |
| `--rejected` | `<output>_rejected.star` | Star file of removed particles. |
| `--max-distance` | 30 | Max perpendicular distance (Å) from the smooth curve. |
| `--max-angle` | 20 | Max axis angle (deg) from the neighbour mean. |
| `--window` | 500 | Smoothing window (Å along the filament). Larger means a stiffer curve. |
| `--neighbours`, `--neighbors` | 5 | Neighbours on each side for the orientation check. |
| `--min-particles` | 5 | Shorter filaments are kept as they are. |
| `--position/--no-position` | on | Turn the position check on or off. |
| `--orientation/--no-orientation` | on | Turn the orientation check on or off. |
| `--remove-flipped` | off | Also remove the minority polarity per filament. |
| `--pixel-size` | per flavor | Override the coordinate pixel size (Å/px). |
| `--group-by` | auto | Tomogram column. |
| `--comment/--no-comment` | on | Provenance header in both output files. |
| `--dry-run` | off | Report only, write nothing. |
| `--quiet`, `-q` | off | Only print the summary. |

## Examples

```bash
# stricter thresholds
tomo_toolshed filament-cleanup --i run_data.star --o clean.star --max-distance 20 --max-angle 15

# position check only, preview
tomo_toolshed filament-cleanup --i run_data.star --o clean.star --no-orientation --dry-run

# curvier filaments (e.g. actin): shorter smoothing window
tomo_toolshed filament-cleanup --i run_data.star --o clean.star --window 300
```
