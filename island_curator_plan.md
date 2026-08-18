# Segmentation Curator — Plan & Claude Code Prompt

An interactive tool to review and clean up a 3D segmentation ("islands" =
connected components) overlaid on a cryo-ET tomogram, then export a curated
binary mask. Built on the existing matplotlib GUI (two orthogonal views, Z and Y
sliders, click-to-toggle), extended with the new tools listed below.

This is a rewrite/generalization of the old `granule selection` script. Key
difference from the old code: **the segmentation is provided directly as an input
file** — the tool no longer re-thresholds the tomogram. The CLI takes a tomogram,
a segmentation, and an output directory.

---

## 1. Goals

Given a tomogram + segmentation pair, let a user:

1. Restrict islands to a **Z-range** (remove anything outside).
2. Remove **small islands** below a minimum voxel size (dust removal).
3. **Dilate / erode** the currently-shown binary mask (zeros fully transparent).
4. **Color by number** (each island a distinct color) or **uncolor** (all green).
5. **Go to** a specific island's center, highlight it **yellow** for review, then
   remove it (click) or toggle it via a button next to the Go-To box. Show that
   island's **label and size**, plus the **current smallest island's label and
   size**, next to the box.
6. **Renumber** islands 1..N whenever changes are made.
7. **Toggle** any island on/off by clicking on it in either view.
8. **Export** the curated result (binary mask) to the CLI output directory, using
   the **same filename as the input segmentation**; **skip** if it already exists.

---

## 2. Repository layout

```
segmentation-curator/
├── pyproject.toml            # packaging + console_scripts entry point
├── README.md
├── LICENSE                   # MIT (or user's choice)
├── .gitignore
├── src/
│   └── segmentation_curator/
│       ├── __init__.py
│       ├── io.py             # MRC read/write helpers (mrcfile)
│       ├── labeling.py       # connected components, z-range, size filter,
│       │                     #   dilate/erode, renumber
│       ├── colormaps.py      # green overlay + label colormap + yellow highlight
│       ├── gui.py            # segmentationCuratorGUI (matplotlib)
│       └── cli.py            # click CLI
└── tests/
    ├── test_labeling.py
    └── test_io.py
```

Console entry point: `segmentation-curator` → `segmentation_curator.cli:main`.

---

## 3. CLI specification

```
segmentation-curator TOMOGRAM SEGMENTATION OUTPUT_DIR [options]
```

Positional arguments:
- `TOMOGRAM` — path to the tomogram MRC/REC.
- `SEGMENTATION` — path to the segmentation MRC/REC (binary or labeled).
- `OUTPUT_DIR` — directory where the curated mask is written.

Options (with sensible defaults so the GUI opens pre-filtered):
- `--z-min`, `--z-max` INT — initial Z-range crop (default: full range).
- `--min-size` INT — initial minimum island size in voxels (default: 0).
- `--connectivity` [6|18|26] — connected-components connectivity (default: 26).
- `--color-by-number/--all-green` — initial overlay mode (default: all-green).

Behavior:
- Output filename = **basename of the SEGMENTATION file** (same name), written
  into `OUTPUT_DIR`. Example: `seg_tomo001.mrc` → `OUTPUT_DIR/seg_tomo001.mrc`.
- **If the output file already exists, print a skip message and exit 0 without
  opening the GUI** (mirrors the old "already processed" behavior).
- `os.makedirs(OUTPUT_DIR, exist_ok=True)` before writing.
- On tomogram/segmentation shape mismatch: error out with a clear message.

Loading & labeling pipeline (`cli.py` → `labeling.py`):
1. Read tomogram and segmentation with `mrcfile` (`io.read_mrc`).
2. Normalize tomogram to uint8 for display only (0..255); keep raw copy if needed.
3. Binarize segmentation (`seg > 0`).
4. Apply initial `--z-min/--z-max` crop (zero voxels outside range).
5. `labeled, N = cc3d.connected_components(binary, connectivity=..., return_N=True)`.
6. Apply initial `--min-size` filter (drop components below threshold, relabel).
7. Launch GUI. On close, GUI returns the curated **binary** mask.
8. Write binary mask (`io.write_mrc`) as the same dtype convention (uint8 0/1).

---

## 4. Core (non-GUI) functions — `labeling.py`

Keep all volume logic pure/testable, separate from matplotlib:

- `label_islands(binary, connectivity=26) -> (labels, n)` — wraps cc3d.
- `apply_zrange(volume, z_min, z_max) -> volume` — zeros voxels outside
  `[z_min, z_max]` (inclusive). Applied to the binary before/after labeling as
  appropriate; after cropping, **relabel** so islands split by the crop get new
  IDs.
- `filter_by_size(labels, sizes, min_size) -> kept_ids` — returns the set of
  island IDs with `size >= min_size`.
- `dilate_binary(binary, radius=1, iterations=1)` and
  `erode_binary(binary, radius=1, iterations=1)` — use
  `scipy.ndimage.binary_dilation/erosion` with `skimage.morphology.ball(radius)`.
  After a dilation (which can merge islands) **relabel**.
- `renumber(labels, keep_ids) -> (new_labels, id_map)` — remaps kept IDs to
  contiguous 1..N (this logic already exists in the old `renumber_granules`).
- `compute_bboxes_and_sizes(labels, n) -> (bboxes, sizes)` — reuse the old
  precompute loop; consider `scipy.ndimage.find_objects` + `bincount` for speed
  on large volumes instead of a Python loop over `np.where`.

Precompute performance note: the old code loops `np.where(labels==id)` per island,
which is O(N × volume). Replace with a single pass:
`sizes = np.bincount(labels.ravel())` and `slices = ndi.find_objects(labels)` to
get bounding boxes in one shot.

---

## 5. GUI — `gui.py` (`SegmentationCuratorGUI`)

Refactor the old procedural GUI into a class holding state (`labels`, `sizes`,
`bboxes`, `selected`, `num_islands`, `z_idx`, `y_idx`, `color_mode`,
`highlight_id`) so callbacks mutate `self.*` instead of `global`/`nonlocal`.
Class name: `SegmentationCuratorGUI`.

Layout (keep the old two-panel design): left = Z view (axis 0), right = Y view
(axis 1). Vertical Z slider, horizontal Y slider, crosshairs, scroll-to-slice,
`q` to quit. Grayscale tomogram base image + a semi-transparent overlay per view.

Overlay rendering (zeros fully transparent):
- Build the overlay from the **selected** islands only.
- Use an RGBA image (or a masked array with a colormap whose `set_bad`/alpha
  makes 0 transparent) so background voxels are fully transparent, not just
  low-alpha. This satisfies "zeros should be completely transparent."

### Feature → control mapping

**(1) Z-range — remove anything outside.**
Two text boxes `Z min` / `Z max` + an `Apply Z-range` button. On apply: zero
voxels outside range, relabel, recompute bboxes/sizes, reset selection to all,
refresh. Show current range in a status line.

**(2) Minimum island size — dust removal.**
Reuse the old `Min size` text box + `Apply Filter` button: deselect (or drop)
islands smaller than the value. Keep the old behavior of *deselecting* so it's
reversible, and let `Renumber`/export bake it in.

**(3) Dilate / Erode the shown binary.**
`Dilate` and `Erode` buttons (+ optional radius/iterations text box). Operate on
the binary formed by the currently selected islands. After the op, relabel,
recompute, refresh. Overlay keeps zeros transparent. Note in README that dilate
can merge islands and erode can split/remove them — both trigger a relabel.

**(4) Color by number / uncolor.**
`Color by number` ↔ `All green` toggle button. Two colormaps in `colormaps.py`:
- all-green single-color overlay (like the old `mito_cmap`), and
- a randomized label colormap (e.g. shuffled HSV / `matplotlib` qualitative)
  keyed by island ID so each island is visually distinct.
Toggling only changes how the overlay is colored, not the selection.

**(5) Go-To + yellow review + info.**
Keep the old `Island ID` text box + `Go To` button (centers both views on the
island's bbox center, moves crosshairs to its X). Additions:
- On Go-To, render **that island in yellow** via a dedicated highlight overlay
  drawn on top of the normal overlay (a separate `imshow` whose data is a mask of
  only `highlight_id`, colored yellow, zeros transparent).
- A **`Toggle`** button next to the box toggles that island's selection (already
  exists as `button_toggle_granule`); clicking the island in the view also
  removes/toggles it (already exists in `on_click`).
- Next to the box, show a live label: **"Island {id}: {size} vox"** for the
  highlighted island, and **"Smallest: {min_id} ({min_size} vox)"** for the
  current smallest selected island (so the user can jump to dust quickly).

**(6) Renumber.**
Keep the old `Renumber` button → remap selected islands to 1..N, update bboxes,
sizes, selection, `num_islands`, refresh. Also keep `Reset` (restore original).

**(7) Toggle by clicking.**
Keep the old `on_click`: left-click on an island in either view toggles its
selection; clicking background navigates (updates the other slider + crosshair).
Keep scroll-to-change-slice and the size-stats text.

**(8) Export.**
On window close (`q` / close event) the GUI returns the curated **binary** mask =
union of selected islands (after any renumber/relabel), as uint8 0/1. `cli.py`
writes it to `OUTPUT_DIR/<segmentation basename>`.

Keep the existing niceties: per-slice bounding boxes + ID labels (toggle with
`Hide/Show Labels`), `Select All`, `Invert`, and the size-stats readout.

---

## 6. Dependencies

`numpy`, `scipy`, `scikit-image`, `mrcfile`, `connected-components-3d` (cc3d),
`matplotlib`, `click`. (Drop `pandas`/`yaml` from the old script — no longer
needed since thresholds/config are gone.) Pin nothing tighter than needed; list
in `pyproject.toml`.

---

## 7. Tests & verification

- `test_io.py`: round-trip a small synthetic MRC (write → read → assert equal).
- `test_labeling.py`: synthetic volume with known islands — assert
  `label_islands` count, `filter_by_size` drops the right ones, `apply_zrange`
  zeros the right slices, `dilate/erode` change voxel counts as expected,
  `renumber` produces contiguous 1..N and preserves voxel membership.
- Manual: run `island-curator tomo.mrc seg.mrc out/`, curate, confirm
  `out/seg.mrc` is written, re-run and confirm it **skips**.
- CI-friendly: keep GUI import lazy so `labeling.py`/`io.py` tests run headless.

---

## 8. README essentials

Install (`pip install -e .`), the CLI usage line, a keyboard/mouse cheat-sheet
(scroll = change slice, left-click island = toggle, `q` = save & quit), a short
description of each button, and the skip-if-exists export behavior.

---

## 9. Open design decisions (defaults chosen; change if desired)

- **Z-range semantics:** default = zero all voxels outside the range and relabel
  (literal "remove anything outside"). Alternative = deselect islands whose
  centroid Z is outside the range (keeps islands intact). Plan uses the literal
  crop; flip if you prefer centroid-based.
- **Size filter:** deselect (reversible) rather than hard-drop, so `Reset` works.
- **Export dtype:** binary uint8 (0/1). Switch to labeled uint16 later if a
  downstream step needs per-island identity.

---

# Claude Code prompt (copy-paste)

> Build a Python package called **`island-curator`** — a CLI + matplotlib GUI to
> review and clean a 3D segmentation overlaid on a cryo-ET tomogram, then export a
> curated binary mask. Use my existing granule-selection script as the GUI
> starting point (two orthogonal Z/Y views, Z & Y sliders, crosshairs,
> scroll-to-slice, click-to-toggle island, `q` to quit, per-slice bounding-box
> labels, Select All / Invert / Renumber / Reset, size-stats readout). I will
> place an example tomogram/segmentation MRC pair in `examples/`.
>
> **Package layout:** `src/island_curator/` with `io.py`, `labeling.py`,
> `colormaps.py`, `gui.py`, `cli.py`; `pyproject.toml` with a console entry point
> `island-curator = island_curator.cli:main`; `tests/`; `README.md`; MIT
> `LICENSE`; `.gitignore`. Dependencies: numpy, scipy, scikit-image, mrcfile,
> connected-components-3d, matplotlib, click. Do NOT depend on pandas or yaml.
>
> **CLI:** `island-curator TOMOGRAM SEGMENTATION OUTPUT_DIR` with options
> `--z-min --z-max --min-size --connectivity {6,18,26} --color-by-number/--all-green`.
> Load both MRCs with mrcfile. The segmentation is the input directly (do NOT
> threshold the tomogram). Binarize `seg > 0`, apply initial z-range crop, label
> with `cc3d.connected_components(..., return_N=True)`, apply initial min-size
> filter, launch the GUI. On GUI close, write the curated **binary** mask (uint8
> 0/1) to `OUTPUT_DIR/<basename of SEGMENTATION>`. Create OUTPUT_DIR if missing.
> **If that output file already exists, print a skip message and exit without
> opening the GUI.** Error clearly if tomogram and segmentation shapes differ.
>
> **Keep volume logic pure and testable in `labeling.py`** (no matplotlib):
> `label_islands`, `apply_zrange` (zero voxels outside [z_min,z_max] then
> relabel), `filter_by_size`, `dilate_binary`/`erode_binary` (scipy ndimage +
> `skimage.morphology.ball`, relabel after), `renumber` (contiguous 1..N),
> `compute_bboxes_and_sizes` (use `np.bincount` + `scipy.ndimage.find_objects`,
> NOT a per-island `np.where` loop). Refactor the GUI into a class
> `IslandCuratorGUI` holding state instead of globals/nonlocals.
>
> **GUI features to implement (all of these):**
> 1. Z-range: `Z min`/`Z max` text boxes + `Apply Z-range` button that zeros
>    voxels outside the range, relabels, recomputes, refreshes.
> 2. Minimum island size (dust removal): text box + `Apply Filter` button that
>    deselects islands smaller than the value (reversible).
> 3. Dilate / Erode buttons (with a radius/iterations text box) operating on the
>    binary of currently selected islands; relabel + refresh after.
> 4. `Color by number` ↔ `All green` toggle: two colormaps — a single-green
>    overlay and a randomized per-label colormap so each island is distinct.
> 5. Go-To box + button that centers both views on an island and highlights it in
>    **yellow** (separate overlay on top). A `Toggle` button next to the box adds/
>    removes it from the selection, and clicking the island in the view toggles it
>    too. Next to the box show live text: the highlighted island's label + voxel
>    size, and the current smallest selected island's label + size.
> 6. `Renumber` button: remap selected islands to 1..N and update everything.
> 7. Click any island in either view to toggle it; background click navigates.
> 8. Export happens on window close (return curated binary mask to the CLI).
>
> **Overlay requirement:** background/zero voxels in the overlay must be **fully
> transparent** (use RGBA or a masked array — not just low alpha).
>
> **Tests:** pytest for `io` (MRC round-trip) and `labeling` (island count,
> size filter, z-range crop, dilate/erode voxel counts, renumber contiguity) using
> small synthetic volumes; keep matplotlib import lazy so these run headless.
> Write a `README.md` with install, CLI usage, and a mouse/keyboard cheat-sheet.
> Initialize a git repo with a sensible first commit.

---

*Once the repo is scaffolded, drop the example tomogram/segmentation pair in
`examples/` and run `island-curator examples/<tomo>.mrc examples/<seg>.mrc out/`
to verify the full loop and the skip-if-exists behavior.*
