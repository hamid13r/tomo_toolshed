# island-curator

Interactive CLI + matplotlib GUI to review and clean a 3D segmentation
("islands" = connected components) overlaid on a cryo-ET tomogram, then export a
curated **binary** mask.

The segmentation is provided **directly as an input file** — the tool does *not*
re-threshold the tomogram. You give it a tomogram, a segmentation, and an output
directory; you curate; it writes the cleaned mask.

## Install

```bash
pip install -e .
```

Dependencies: `numpy`, `scipy`, `scikit-image`, `mrcfile`,
`connected-components-3d`, `matplotlib`, `click`.

## Usage

```bash
island-curator TOMOGRAM SEGMENTATION OUTPUT_DIR [options]
```

Example (using the bundled example pair):

```bash
island-curator \
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
| `--min-size INT` | `0` | Initial minimum island size in voxels (dust removal). |
| `--connectivity {6,18,26}` | `26` | Connected-components connectivity. |
| `--color-by-number / --all-green` | `--all-green` | Initial overlay mode. |

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
the **Y view** (a Z–X plane at fixed Y). Cyan crosshairs mark the current
position; overlay background voxels are **fully transparent**.

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
  island and `Smallest: {id} ({size} vox)` for the current smallest selected
  island — handy for jumping to dust.
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

`labeling.py` and `io.py` are pure/headless (no matplotlib import) so the tests
run without a display. The GUI (`gui.py`) is imported lazily only when needed.

## License

MIT — see [LICENSE](LICENSE).
