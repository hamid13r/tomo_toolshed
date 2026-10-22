# pick-granules — threshold dark granules inside a mask

Part of [tomo_toolshed](../README.md). Opens a tomogram with a binary mask
(e.g. a mitochondrion segmentation) overlaid, lets you drag an intensity
threshold until the dark granules inside the mask are covered, and writes a
binary MRC of those voxels. The result is meant to be cleaned up afterwards with
[`curate`](segmentation_curator.md).

## Usage

```bash
tomo_toolshed pick-granules tomo.mrc mito.mrc -o mito_granules.mrc
tomo_toolshed pick-granules tomo.mrc mito.mrc --blur 1.5 --mask-erode 3
tomo_toolshed pick-granules tomo.mrc mito.mrc -t -0.8 --min-size 20 --no-gui   # batch

# then clean up the result
tomo_toolshed curate tomo.mrc mito_granules.mrc curated/ --threshold 0
```

`TOMOGRAM` and `MASK` must have the same shape. Any non-zero mask voxel is
inside the mask.

| Option | Default | Meaning |
|---|---|---|
| `--output`, `-o` | `<mask>_granules.mrc` | Output MRC (parent directories are created). |
| `--threshold`, `-t` | — | Starting threshold, in tomogram units after `--blur`. |
| `--percentile`, `-p` | `5` | Starting threshold as a percentile of the in-mask values. Use instead of `-t`. |
| `--bright` | off | Pick voxels brighter than the threshold instead of darker. |
| `--blur` | `0` | Gaussian blur sigma (voxels) applied before thresholding and display. 1–2 removes noise speckle. |
| `--mask-erode` | `0` | Erode the mask this many pixels in XY first (also adjustable in the GUI). Keeps the dark membranes at the mask boundary out of the result. |
| `--erode-3d` | off | Erode the mask in 3D, so its top and bottom in Z shrink too (XY erosion leaves the membrane caps). |
| `--min-size` | `0` | Drop granules smaller than this many voxels (26-connected) on save (also adjustable in the GUI). |
| `--no-gui` | off | Write the result for the starting threshold without opening the GUI. |
| `--overwrite` | off | Overwrite an existing output. Without it the run is skipped, as in `curate`. |

## GUI

- **Views:** Z view (Y–X plane) and Y view (Z–X plane), as in `curate`. Scroll
  over a view to step slices; click to move the crosshair; the Z and Y sliders
  also move the slices.
- **Overlay:** the mask is tinted blue; voxels picked by the current threshold
  are red. Opacity, contrast and **Hide mask** are display only.
- **Threshold:** the histogram shows the tomogram values inside the mask (log
  scale) with the threshold as a red line. Drag the slider, or type an exact
  **value** or a **percentile** and press Enter. The status line shows the
  threshold's percentile and how many mask voxels it picks.
- **Erode mask (px)** + **XY/3D** + **Apply:** re-erodes the original mask
  (the XY/3D button switches the mode) and updates the histogram.
- **Min size (vox):** press Enter to set; applied when saving (the live overlay
  shows the result before small-island removal).
- **Keys:** `q` or closing the window saves and quits; `Esc` quits without
  writing anything.

After saving, the command prints a `--no-gui` line that reproduces the same
result, so the same settings can be run on other tomograms.

## Details

- Volumes are `(Z, Y, X)`, as in `curate` and `erode`.
- Granules = inside the (eroded) mask **and** value `<` threshold (`>` with
  `--bright`), then islands smaller than `--min-size` removed.
- In cryo-ET reconstructions dense material is usually dark (low values), so
  the default picks below the threshold. If your tomogram is inverted, use
  `--bright`.
- Output is a `uint8` 0/1 MRC with the mask's voxel size and header origin.
