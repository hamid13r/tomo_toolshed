# curate — segmentation curator

Part of [tomo_toolshed](../README.md). Interactive CLI + matplotlib GUI to review
and clean a 3D segmentation ("islands" = connected components) overlaid on a
cryo-ET tomogram, then export a curated **binary** mask.

The segmentation is provided **directly as an input file** — the tool does *not*
re-threshold the tomogram. You give it a tomogram, a segmentation, and an output
directory; you curate; it writes the cleaned mask.

The segmentation may carry per-voxel **confidence scores** (e.g. EasyMode stores
`int8` values `0–127`). At load, the tool asks for a **confidence threshold
(0–128)**, splits the segmentation into islands (connected components of
`seg > 0`), and **removes any island that has no voxel above the threshold**
before opening the GUI. Pass `--threshold N` to skip the prompt.

## Install

Install the whole toolshed (see the [root README](../README.md)). With
micromamba (recommended):

```bash
git clone https://github.com/hamid13r/tomo_toolshed.git
cd tomo_toolshed
micromamba create -f environment.yml -y
micromamba activate tomo-toolshed
pip install -e .
```

Or with plain pip:

```bash
pip install -e .
```

Dependencies: `numpy`, `scipy`, `scikit-image`, `mrcfile`,
`connected-components-3d`, `matplotlib`, `click`.

## Usage

```bash
tomo_toolshed curate TOMOGRAM SEGMENTATION OUTPUT_DIR [options]
```

Example:

```bash
tomo_toolshed curate \
  example/MIM019_2_lam1_ts_002.mrc \
  example/easymode_microtubule/MIM019_2_lam1_ts_002__microtubule.mrc \
  out/
```

The curated mask is written to `OUTPUT_DIR/<basename of SEGMENTATION>`
(e.g. `out/MIM019_2_lam1_ts_002__microtubule.mrc`) as a `0/1` binary volume.

**Skip-if-exists:** if that output file already exists, the tool prints a skip
message and exits `0` *without* opening the GUI. Delete the file to re-curate.

### Options

| Option | Default | Meaning |
|---|---|---|
| `--z-min INT` | full range | Initial Z-range lower bound (inclusive). |
| `--z-max INT` | full range | Initial Z-range upper bound (inclusive). |
| `--threshold INT` | *(prompted)* | Confidence threshold `0–128`. Islands with **no voxel above** this value are removed at load. If omitted, the tool prompts for it interactively. |
| `--min-size INT` | `0` | Initial minimum island size in voxels (dust removal). |
| `--blur FLOAT` | `0.0` | Gaussian blur sigma applied to the tomogram for display (3D; `0` = none). Does not affect the segmentation. |
| `--connectivity {6,18,26}` | `26` | Connected-components connectivity. |
| `--color-by-number / --all-green` | `--all-green` | Initial overlay mode. |

## Particle input (remove false-positive picks)

Pass a particle file instead of a mask, plus `--radius`. Either a RELION 4
`.star` file, or a plain `.txt` / `.box` coordinate list with three columns
`x y z` (whitespace- or comma-separated; `#` comments and blank lines are
ignored). The output is **always a star file**.

```bash
tomo_toolshed curate TOMOGRAM particles.star OUTPUT_DIR --radius 60
tomo_toolshed curate TOMOGRAM picks.txt      OUTPUT_DIR --radius 60   # -> OUTPUT_DIR/picks.star
```

Each particle is drawn as a sphere of `--radius` Å and becomes its own island,
numbered by its row in the star file (row 1 = id 1). Overlapping spheres are
split by nearest center, so close particles never merge. Curate as usual —
click a sphere to reject/restore that particle — and on quit the tool writes
`OUTPUT_DIR/<star basename>` (or `OUTPUT_DIR/<stem>.star` for `.txt`/`.box`
input) containing only the **selected** particles. For star input every column
and the `optics`/`general` blocks pass through unchanged; only rows are
removed. For text input the star has one `particles` block with
`rlnCoordinateX/Y/Z`, in the input's original units (not rescaled).

- **One tomogram per star file.** If `rlnTomoName` / `rlnMicrographName` /
  `wrpSourceName` has more than one value, the tool stops; split first with
  [`split-star`](split_star.md).
- **Coordinates** are RELION 4 `rlnCoordinateX/Y/Z` (or the text file's x y z)
  in pixels; `rlnOrigin*Angst`
  shifts are applied if present. If the coordinates are at a different binning
  than the tomogram, give their pixel size with `--coord-pixel-size`. The
  tomogram's pixel size comes from its MRC header, or `--tomo-pixel-size`.
  RELION 5 centered-Å coordinates are not supported.
- **In the GUI**, a *Sphere radius (Å)* box + **Apply radius** replaces
  Dilate/Erode (display only; the output star is unaffected), **Renumber** is
  hidden (ids must stay tied to star rows), and **Apply Z-range** / `--z-min` /
  `--z-max` *deselect* particles whose center is outside the range.
- `--threshold`, `--min-size` and `--connectivity` do not apply to particle input.
- Particles outside the tomogram volume are not drawn and are kept.

| Particle-mode option | Default | Meaning |
|---|---|---|
| `--radius FLOAT` | *(required)* | Sphere radius in Å. |
| `--coord-pixel-size FLOAT` | tomogram's | Pixel size (Å/px) of the input coordinates. |
| `--tomo-pixel-size FLOAT` | MRC header | Tomogram pixel size (Å/px). |

## Mouse & keyboard cheat-sheet

| Action | Effect |
|---|---|
| **Left-click an island** (either view) | Toggle it on/off in the selection |
| **Left-click background** | Navigate: move the other view's slice + crosshair |
| **Scroll** over a view | Change that view's slice |
| **Z / Y sliders** | Change the Z view / Y view slice |
| **`q`** | Save the curated mask & quit |
| Close the window | Save the curated mask & quit |

The left panel is the **Z view** (a Y–X plane at fixed Z); the right panel is
the **Y view** (a Z–X plane at fixed Y, with Z increasing upward to match the
vertical Z slider). Cyan crosshairs mark the current position; overlay
background voxels are **fully transparent**.

**Contrast:** two sliders (top-right, `min` / `max`) adjust the display contrast
window over the grayscale tomogram live. This only changes how the tomogram is
displayed — it never affects the segmentation or the exported mask.

**Overlay opacity:** the `opacity` slider (below the contrast sliders) sets the
alpha of the island color overlay, from `0` (overlay hidden, bare tomogram) to
`1` (opaque). Display-only; the yellow highlight and the exported mask are
unaffected.

## Buttons & boxes

- **Z min / Z max + Apply Z-range** — zero all voxels outside `[z_min, z_max]`,
  then relabel and recompute (literal "remove anything outside").
- **Min size + Apply Filter** — *deselect* islands smaller than the given voxel
  count. Reversible (use **Reset** or **Select All** to bring them back);
  **Renumber** or export bakes it in.
- **Radius / Iters + Dilate / Erode** — morphological dilation/erosion (spherical
  structuring element) of the currently-selected binary mask, then relabel.
  Dilation can merge islands; erosion can split or remove them — both relabel.
- **Color by number ↔ All green** — toggle overlay coloring. "All green" draws
  every selected island in one green; "Color by number" gives each island a
  distinct color. Toggling never changes the selection.
- **Island ID + Go To** — center both views on that island and highlight it in
  **yellow**. The info line shows `Island {id}: {size} vox` for the highlighted
  island, plus `Smallest: {id} ({size} vox)` and `Largest: {id} ({size} vox)`
  for the current smallest/largest selected islands — handy for jumping to dust.
- **◀ / ▶** (either side of the Island ID box) — step to the previous/next
  island and go there, exactly as if you had typed its id and pressed **Go To**.
  Ids missing from the volume (after filtering or morphology) are skipped, and
  stepping wraps around at both ends. The status line shows `{position}/{total}`.
- **Toggle** — add/remove the Island-ID island from the selection (same as
  clicking it in a view).
- **Renumber** — remap the selected islands to a contiguous `1..N`.
- **Reset** — restore the original labeling and selection.
- **Select All / Invert** — bulk selection helpers.
- **Hide/Show Labels** — toggle per-slice bounding boxes + ID labels.

## Export

On quit, the curated mask = the union of the currently selected islands, written
as `uint8` `0/1`. (MRC has no native unsigned-8-bit mode, so `mrcfile` may store
it as `uint16`; the values are still `0/1`.)

## Development

```bash
pip install -e ".[test]"
pytest
```

`labeling.py`, `io.py` and `particles.py` are pure/headless (no matplotlib import) so the tests
run without a display. The GUI (`gui.py`) is imported lazily only when needed.
