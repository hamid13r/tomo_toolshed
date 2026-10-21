# vtp-normals — preview, curate and flip pycurv surface normals

Part of [tomo_toolshed](../README.md). Previews and flips the per-triangle
**normal vectors** of [pycurv](https://github.com/kalemaria/pycurv) /
morphometrics membrane surface `.vtp` files. pycurv estimates normals by vector
voting, but their sign (which side of the membrane they point to) is only
locally consistent: a surface is often correct on one patch and inverted on
another. `vtp-normals` splits the surface into **pieces**, lets you see which
pieces point the wrong way, and flips those pieces — reversing the triangle
winding, negating the normal arrays and updating the curvature arrays so the
file stays self-consistent.

It is a click **group** with three subcommands:

| Subcommand | What it does |
|---|---|
| [`preview`](#preview) | Write a `<name>.normals.png` per surface, one panel per piece, so you can read off which pieces are inverted. |
| [`curate`](#curate) | Interactive 3-orthoslice viewer with a piece table: mark pieces, write `<name>.flipped.vtp`, then review the result. |
| [`flip`](#flip) | Batch-flip pieces you list as `TS:MEMBRANE:IDS` (or in a `--flip-file`) into `<name>.flipped.vtp`. |

```bash
# interactive, one surface + its tomogram
tomo_toolshed vtp-normals curate --tomo rec.mrc --vtp surfaces/..._IMM.AVV_rh9.vtp

# or: batch preview, read the piece ids off the PNGs, then flip
tomo_toolshed vtp-normals preview --vtp-glob "surfaces/*.vtp"
tomo_toolshed vtp-normals flip --flip lam12_ts_003:IMM:0 --flip lam12_ts_004:IMM:1,2,4
```

## How pieces are defined

A "piece" is a connected part of the surface that is flipped as one unit. Pieces
are built in two steps:

1. **Connected components** — triangles that share a vertex belong to the same
   piece.
2. **Split by a cell array** — each connected component is split again wherever
   the `--split-column` array (default `OMM_dist`, the OMM–IMM distance written
   by morphometrics) crosses `--split-dist` (default 20 nm). Triangles on
   opposite sides of the threshold are never joined, so the **boundary IMM**
   (close to the OMM) and the **cristae** (far from it) become separate pieces
   and can be flipped independently. Set `--split-dist 0` to skip this and use
   connected components only.

Small split fragments (fewer than `--min-triangles` triangles, default 50) are
**merged** into the neighbouring piece they share the most vertex contacts with,
which removes speckle along the threshold. Isolated pieces smaller than
`--min-triangles` are not shown and never flipped.

Pieces are then **numbered largest first** (piece 0 is the biggest), and that
numbering is shared by all three subcommands. **The same `--split-column`,
`--split-dist` and `--min-triangles` must be passed to `flip` as to the
`preview`/`curate` you read the ids from**, or the ids will not line up.

If the surface has no `--split-column` array (e.g. an already-written
`.flipped.vtp`, which has no `OMM_dist`), the split step is skipped with a
one-line note and pieces are the connected components only.

## Coordinate convention

Surface coordinates are converted to tomogram voxel indices exactly as in the
other tools:

```
voxel (x, y, z) = surface_xyz * --coord-scale / --apix
```

`--coord-scale` defaults to 10 (surfaces are in nm, tomograms in Å);
`--apix` is the tomogram pixel size in Å (`preview` defaults to 9.98; `curate`
reads it from the MRC header of `--tomo`, falling back to 9.98). If the surface
does not land inside the tomogram, `curate` prints a warning so you can fix
`--apix` / `--coord-scale`.

## `preview`

Writes `<name>.normals.png` next to each `.vtp` matched by `--vtp-glob`, one
panel per piece (≥ `--min-triangles`). Each panel shows the tomogram slice at the
piece's median z with:

- **cyan** — triangle centres within ±`--slab` voxels of that slice,
- **red** — the same points moved `--dist` Å along the normal,
- **yellow** — thin lines joining a subset of points to their tips.

If the normals are consistent, every red tip sits on the same side of the
membrane; a piece whose tips point the other way is inverted. Read its piece
number from the panel title and pass it to `flip`.

| Option | Default | Meaning |
|---|---|---|
| `--vtp-glob` | `surfaces/*.vtp` | Surfaces to preview. |
| `--tomo-dir` | `tomo` | Directory searched for the background tomograms. |
| `--tomo-pattern` | `{stem}.mrc_9.98Apx.mrc` | Tomogram file name; `{stem}` is the vtp name up to `_ts_NNN`. |
| `--normal-array` | `n_v` | Cell array holding the normals (`n_v` = pycurv vector-voting normals). |
| `--dist` | `150.0` | Tip distance along the normal (Å). |
| `--apix` | `9.98` | Tomogram pixel size (Å). |
| `--coord-scale` | `10.0` | Multiply surface coordinates by this to get Å. |
| `--slab` | `1.5` | Show surface points within this many voxels of the slice. |
| `--split-column` | `OMM_dist` | Cell array pieces are split on. |
| `--split-dist` | `20.0` | Split where `--split-column` crosses this; `0` = connected pieces only. |
| `--min-triangles` | `50` | Skip pieces with fewer triangles than this. |
| `--line-every` | `10` | Draw a membrane-to-tip line for every Nth point. |

If no tomogram is found, the panels are drawn without a background (a warning is
printed).

## `curate`

An interactive viewer in the style of [`curate`](segmentation_curator.md): three
orthogonal slices (Z = Y–X, Y = Z–X, X = Z–Y, all at true aspect) with sliders,
and a table of pieces on the right. Every `--arrow-every`-th membrane point
within the slab is drawn as a dot with a normal arrow projected onto the view —
**green** where the normal is kept, **red** where the piece is marked to flip —
updating live as you toggle. Needs an interactive matplotlib backend (a display;
e.g. `ssh -X` or a desktop/VNC session) — on a headless/`Agg` backend it exits
with an error.

### Controls

| Action | Effect |
|---|---|
| table, click a row | highlight the piece, zoom to it, move all slices onto it |
| table, click the flip cell | toggle keep / FLIP for that piece |
| click a membrane in a view | highlight that piece (does not toggle) |
| click elsewhere in a view | move the crosshair (the other two slices) there |
| scroll in a view / table | step that view's slice / scroll the table |
| up / down, Prev / Next | highlight the previous / next piece |
| `t`, Toggle flip | toggle the highlighted piece |
| 3D view | rotatable 3D plot of the highlighted piece's normals |
| Full view | zoom all views out to the whole tomogram |
| `q`, Done (or close window) | flip the FLIP pieces into `<name>.flipped.vtp` |
| `x`, Stop | stop the session without writing this surface |

The piece table shows, per piece: id, triangle count, z range, the median
`--split-column` value, and the keep/FLIP state.

After **Done**, the written `.flipped.vtp` is reopened for **review** (pieces
flipped in the file are marked `*`). Toggling pieces there and pressing Done
rewrites the file from the original with those corrections; Done with no changes
moves on to the next surface. A `<name>.flipped.normals.png` preview is saved
with every write. If a `.flipped.vtp` already exists, its flips are loaded as the
starting state, so a surface can be revisited (`--skip-done` skips surfaces that
already have one).

| Option | Default | Meaning |
|---|---|---|
| `--tomo` | — | Tomogram shown as background (overrides `--tomo-dir`/`--tomo-pattern`). |
| `--vtp` | — | Single surface to curate (overrides `--vtp-glob`). |
| `--vtp-glob` | `surfaces/*IMM*.vtp` | Surfaces to go through. |
| `--tomo-dir` | `tomo` | Directory searched for the background tomograms. |
| `--tomo-pattern` | `{stem}.mrc_9.98Apx.mrc` | Tomogram file name; `{stem}` is the vtp name up to `_ts_NNN`. |
| `--skip-done` | off | Skip surfaces that already have a `.flipped.vtp`. |
| `--dist` | `150.0` | Initial tip distance along the normal (Å); adjustable in the viewer. |
| `--apix` | header → 9.98 | Tomogram pixel size (Å). Default: from the MRC header of `--tomo`, else 9.98. |
| `--coord-scale` | `10.0` | Multiply surface coordinates by this to get Å. |
| `--split-column` | `OMM_dist` | Cell array pieces are split on. |
| `--split-dist` | `20.0` | Split where `--split-column` crosses this; `0` = connected pieces only. |
| `--min-triangles` | `50` | Smaller split parts are merged into a neighbour; isolated pieces this small are not listed. |
| `--arrow-every` | `10` | Draw a normal arrow for every Nth surface point; adjustable in the viewer. |

## `flip`

Flips the pieces you list and writes `<name>.flipped.vtp`. Each flip is
`TS:MEMBRANE:IDS`, e.g. `lam12_ts_003:IMM:0,3` flips pieces 0 and 3 of the vtp in
`--vtp-dir` whose name contains `lam12_ts_003` and `_IMM.` (exactly one file must
match, and `.flipped.` files are ignored). Pass flips with repeated `--flip`, or
one per line in `--flip-file`:

```text
# comments and blank lines are ignored
lam12_ts_003:IMM:0
lam12_ts_004:IMM:1,2,4
```

| Option | Default | Meaning |
|---|---|---|
| `--flip` | — | A `TS:MEMBRANE:IDS` flip. Repeatable. |
| `--flip-file` | — | Text file with one `TS:MEMBRANE:IDS` per line. |
| `--vtp-dir` | `surfaces` | Directory with the original vtp files. |
| `--arrays` | `n_v,normal,avg_normals` | Comma-separated cell arrays to negate. |
| `--curvature` / `--no-curvature` | on | Also update the curvature arrays of flipped pieces. |
| `--preview` / `--no-preview` | on | Write a `.flipped.normals.png` preview of each output. |
| `--tomo-dir` | `tomo` | Tomograms for the preview background. |
| `--split-column` | `OMM_dist` | Cell array pieces are split on. |
| `--split-dist` | `20.0` | Split where `--split-column` crosses this. |
| `--min-triangles` | `50` | Smaller split parts are merged into a neighbour. |

### What `<name>.flipped.vtp` contains

For each flipped piece:

- the **normal arrays** (`--arrays`: `n_v`, `normal`, `avg_normals`) are negated;
- the **triangle winding is reversed**, so the geometric normal agrees with the
  flipped normal arrays;
- with `--curvature` (default), the **curvature arrays** are updated so they stay
  consistent with the new normal direction:
  - **sign-flipped**: `mean_curvature`, `mean_curvature_VV`, `shape_index_VV`,
    `shape_index_cat`;
  - **swapped and negated**: `kappa_1 ← −kappa_2`, `kappa_2 ← −kappa_1` (and the
    same for `max_curvature` / `min_curvature`), so `kappa_1 ≥ kappa_2` still
    holds;
  - **swapped**: the principal directions `t_1 ↔ t_2`;
  - Gaussian curvature, curvedness, verticality and `OMM_orientation` do not
    depend on the normal sign and are left unchanged. Missing arrays are skipped
    with a note.

Two new cell arrays are written for every triangle: `piece_id` (the piece
numbering) and `normal_flipped` (`1` where the normal was flipped). Flipping the
same pieces of a `.flipped.vtp` again restores the original normals.

## Typical workflow

1. `tomo_toolshed vtp-normals curate --tomo rec.mrc --vtp surfaces/..._IMM.vtp`
   and fix the inverted pieces interactively; **or**
2. `tomo_toolshed vtp-normals preview --vtp-glob "surfaces/*.vtp"`, open the
   `.normals.png` files, note the inverted piece ids, then
   `tomo_toolshed vtp-normals flip --flip <ts>:<membrane>:<ids> ...`.

Keep `--split-column`, `--split-dist` and `--min-triangles` the same throughout.
