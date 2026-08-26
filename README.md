# Removing Skipped Views in etomo from WarpTools

A Python utility for processing XML files in cryo-electron tomography workflows, specifically designed to update `UseTilt` values based on tilt alignment solutions from WARP.

## Overview

This tool processes XML files and updates their `UseTilt` values based on data from corresponding `taSolution.log` files. It's particularly useful for:

- Removing skipped views from tilt series based on etomo alignment logs that are created in fine-alignment step
- Setting specific numbers of views to keep based on dose values, lowest accumulated doses are also the lowest tilts
- Setting all views to True for testing purposes or just going back to default
- Batch processing multiple XML files with automatic backups

This is needed when there are views that need to be excluded from the tomogram:
![Skipping a view with shifted beam](docs/etomo-skip.png)

*Figure: Example of the view that need to be skipped.*

By turning the UseTilt to False for that view, the shadow goes away:

![The tomogram before and after editing the xml file](docs/before-after.png)

*Figure: Result after removing skipped views using this code, the UseTilt section of the xml file is also shown.*



## Features

- **Automatic backup creation** - Safely backs up original XML files before modification, the backup directory needs to be new to avoid overriding original backups
- **etomo-based filtering** - Uses `taSolution.log` files to determine which views to keep
- **Tilt-based selection** - Option to keep only until a certain amount of tilt from the lowest-dose view
- **Batch processing** - Process multiple XML files with customizable patterns
- **Safety first** - Never overwrites existing backups

## Installation

### Prerequisites

- Python 3.6+
- Required Python packages:

```bash
pip install pandas click lxml
```

### Setup

```bash
git clone https://github.com/yourusername/warp_remove_skipped_views.git
cd warp_remove_skipped_views
pip install pandas click lxml
```

## Usage

### Basic Commands

```bash

# Process with custom pattern and directories, this would work in the warp_tiltseries directory in the default WarpTools structure
python remove_skipped_view.py --xml-dir ./ --xml-pattern "*.xml" --backup-dir backup_xml --tiltstack-dir tiltstack

# Set all UseTilt values to True
python remove_skipped_view.py  --xml-dir ./ --xml-pattern "*.xml" --backup-dir backup_xml --all-true

# Keep only views within 40 degrees of the lowest-dose tilt
python remove_skipped_view.py --xml-dir ./ --xml-pattern "*.xml" --backup-dir backup_xml --max-tilt 40
```

### Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--xml-dir` | `./` | Directory containing XML files to process |
| `--xml-pattern` | `*.xml` | Glob pattern to match XML files |
| `--backup-dir` | `backup_xml` | Directory to store XML backups |
| `--tiltstack-dir` | `tiltstack` | Base directory containing tiltstack logs |
| `--tomostar-dir` | `../tomostar` | Directory containing `.tomostar` files (used by `--delete`; resolved relative to the current working directory) |
| `--all-true` | False | Set all UseTilt values to True (ignores log files) |
| `--max-tilt`| 0 | Keep views up to this tilt from the lowest tilt |
| `--delete` | False | Physically remove the excluded tilts from both the XML and the matching `.tomostar` instead of setting `UseTilt=False` (see [Deletion mode](#deletion-mode)) |
| `--dry-run` | False | Report what would change and write nothing |

## Processing Modes

### 1. Log-based Filtering (Default)
By default (no `--max-tilt` / `--all-true`):
- Views present in `taSolution.log` → `UseTilt = True`
- Views not in log → `UseTilt = False`

### 2. Tilt-angle Selection
When `--max-tilt > 0`:
- Finds the lowest-dose tilt and its tilt angle
- Keeps views within `--max-tilt` degrees of that angle as `True`
- **Note**: Views not in taSolution.log are always set to `False`

### 3. All True Mode
When `--all-true`:
- Sets all `UseTilt` values to `True`
- Useful for testing or resetting configurations

## Deletion mode

Everything above only flips `UseTilt` to `True`/`False` — the tilts stay in the
file. With `--delete` the excluded tilts (the complement of the kept set from
whichever mode above you chose) are **physically removed** from both the XML and
the matching `.tomostar` file:

- Every per-tilt list (`Angles`, `Dose`, `UseTilt`, `AxisAngle`, `AxisOffset*`,
  `MoviePath`, `FOVFraction`, …) has the deleted entries dropped.
- Per-tilt indexed elements (`TiltPS1D`, `TiltSimulatedScale`) are dropped and
  their `ID`s renumbered `0..M-1`.
- Every per-tilt grid (`GridCTF*`, and, when present, `GridMovement*`,
  `GridAngle*`, `GridDoseBfacs*`, `GridDoseWeights` — detected automatically as
  those whose `Depth == N`) has its deleted `Z`-slices removed, the surviving
  `Node` `Z` values renumbered contiguously, and `Depth` updated to the new
  count. Global grids (`Depth == 1`) are left untouched.
- In the `.tomostar`, rows are matched to XML tilts **by movie name** (falling
  back to row order with a warning if names can't be matched) and the deleted
  rows are removed. The header, column order and `_wrp<Name> #<index>`
  declarations are preserved.

```bash
# Delete the views etomo dropped, from both the XML and ../tomostar
python remove_skipped_view.py --xml-dir ./ --delete

# Preview only — writes nothing
python remove_skipped_view.py --xml-dir ./ --delete --dry-run

# Delete everything beyond ±40° of the lowest-dose tilt
python remove_skipped_view.py --xml-dir ./ --delete --max-tilt 40
```

Safety:

- Backups are written **in place** next to the originals: `<file>.xml.bak` and
  `<file>.tomostar.bak`. If a `.bak` already exists, a numbered backup
  (`.bak.1`, `.bak.2`, …) is created — nothing is overwritten.
- The XML and the `.tomostar` are only written once **both** have been parsed,
  edited and re-validated in memory (list lengths and grid `Depth`/`Z` ranges
  are asserted against the new tilt count). Writes are atomic (temp file +
  replace), and any per-tilt list whose length doesn't match `N` aborts that
  file untouched.
- `--delete` with `--all-true` is rejected (it would delete nothing).

> ⚠️ **Deletion is NOT idempotent.** Once tilts are removed, the view numbers in
> `taSolution.log` no longer line up with the XML rows, so **running `--delete`
> again on the same tilt series would delete the WRONG tilts.** Restore from the
> `.bak` files before re-running.

> ℹ️ Deletion changes the tilt count, so downstream WarpTools steps
> (`ts_ctf`, `ts_aretomo`, `ts_reconstruct`) must be re-run.

## Expected Directory Structure


```
project/
├── tomostar/                      # *.tomostar files (used by --delete)
│   └── [xml_basename].tomostar
└── warp_tiltseries/
    ├── *.xml                      # XML files to process
    ├── backup_xml/                # Backup directory for flip mode (auto-created)
    └── tiltstack/                 # Tiltstack made by warp
        └── [xml_basename]/
            └── taSolution.log     # each TS done in etomo gets a taSolution.log
```

Run the tool from `warp_tiltseries/`; the default `--tomostar-dir ../tomostar`
then points at the sibling `tomostar/` directory. In `--delete` mode the
`<file>.xml.bak` / `<file>.tomostar.bak` backups are written next to each
original (not in `backup_xml/`).

## License

This project is released under a permissive open-source license. You are free to use, modify, and distribute it for any purpose. See the LICENSE file for more information.

## Acknowledgments

- Designed for [WARP2.0](https://github.com/warpem/warp) cryo-electron tomography workflows 
- Built with pandas, click, and lxml
- Thanks to Alister Burt and Dimitry Tegunov for the support 