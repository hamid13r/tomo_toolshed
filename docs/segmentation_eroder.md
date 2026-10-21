# erode — shrink a binary segmentation MRC

Part of [tomo_toolshed](../README.md). Erodes a binary segmentation mask a set
number of times, in the **XY plane** (default) or in **3D**, and writes an MRC
of the same shape with the input's voxel size and origin.

## Usage

```bash
tomo_toolshed erode --i membrane.mrc --o membrane_eroded.mrc
tomo_toolshed erode --i membrane.mrc -n 3                 # -> membrane_eroded.mrc
tomo_toolshed erode --i membrane.mrc --o eroded.mrc -n 2 --3d --connectivity 26
```

| Option | Default | Meaning |
|---|---|---|
| `--input`, `--i` | required | Binary segmentation MRC. Any non-zero voxel is foreground. |
| `--output`, `--o` | `<input>_eroded.mrc` | Output MRC (parent directories are created). Must differ from the input. |
| `--iterations`, `-n` | `1` | Number of erosion iterations; each removes one pixel layer. |
| `--3d` | off | Also erode along Z. Without it, each Z slice is eroded in-plane only. |
| `--connectivity` | `4` (XY), `6` (3D) | Structuring element. XY: `4` (cross) or `8` (3×3 square). 3D: `6` (cross), `18`, or `26` (3×3×3 cube). |
| `--quiet`, `-q` | off | Only print the final summary. |

## Axis convention

Same as [`curate`](segmentation_curator.md): MRC data are read as `(Z, Y, X)`
arrays, Z being the slice axis you scroll through in IMOD/3dmod. XY erosion
treats every Z slice independently, so a mask never gets thinner in Z — useful
for membranes or filaments that are only a few slices thick. `--3d` erodes in
all three directions.

## Details

- **Foreground** is `value > 0`, as in `curate`. Probability maps should be
  thresholded first.
- **Edges:** voxels outside the volume count as foreground, so a mask that runs
  into the edge of the tomogram is not eroded from the edge inward.
- **Output:** a `uint8` 0/1 MRC (same as `curate`), with the input's voxel size
  and header origin. The summary line reports foreground voxels before and after;
  a warning is printed if the mask is eroded away completely.
- Connectivity `4`/`6` erodes evenly along the axes (diamond-shaped growth of
  the removed band); `8`/`26` erodes corners faster (square/cube-shaped).
