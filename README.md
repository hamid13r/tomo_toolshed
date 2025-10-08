# Warp Remove Skipped Views

A Python utility for processing XML files in cryo-electron tomography workflows, specifically designed to update `UseTilt` values based on tilt alignment solutions from WARP.

## Overview

This tool processes XML files and updates their `UseTilt` values based on data from corresponding `taSolution.log` files. It's particularly useful for:

- Removing skipped views from tilt series based on etomo alignment logs that are created in fine-alignment step
- Setting specific numbers of views to keep based on dose values, lowest accumulated doses are also the lowest tilts
- Setting all views to True for testing purposes or just going back to default
- Batch processing multiple XML files with automatic backups

## Features

- **Automatic backup creation** - Safely backs up original XML files before modification, the backup directory needs to be new to avoid overriding original backups
- **etomo-based filtering** - Uses `taSolution.log` files to determine which views to keep
- **Dose-based selection** - Option to keep only the N lowest-dose views
- **Tilt-based selection** - Option to keep only until a certain amount of tilt from the first view
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
pip install -r requirements.txt
```

## Usage

### Basic Commands

```bash
# Process all XML files in current directory
python remove_skipped_view.py ./

# Process with custom pattern and directories
python remove_skipped_view.py ./ --xml-pattern "*.xml" --backup-dir backup_xml --tiltstack-dir tiltstack

# Set all UseTilt values to True
python remove_skipped_view.py ./ --all-true

# Keep only 20 lowest-dose tilts
python remove_skipped_view.py ./ --n-tilts 20
```

### Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--xml-dir` | `./` | Directory containing XML files to process |
| `--xml-pattern` | `*.xml` | Glob pattern to match XML files |
| `--backup-dir` | `backup_xml` | Directory to store XML backups |
| `--tiltstack-dir` | `tiltstack` | Base directory containing tiltstack logs |
| `--all-true` | False | Set all UseTilt values to True (ignores log files) |
| `--n-tilts` | 0 | Keep N lowest-dose views, set others to False |
| `--max-tilt`| 0 | Keep views that are smaller than this value

## Processing Modes

### 1. Log-based Filtering (Default)
When `--n-tilts 0` (default):
- Views present in `taSolution.log` → `UseTilt = True`
- Views not in log → `UseTilt = False`

### 2. Dose-based Selection
When `--n-tilts > 0`:
- Sorts tilts by dose values from XML
- Sets the N lowest-dose tilts to `True`
- Sets remaining tilts to `False`
- **Note**: Views not in taSolution.log are always set to `False`

### 3. Dose-based Selection
When `--n-tilts > 0`:
- Sorts tilts by dose values from XML
- Sets the N lowest-dose tilts to `True`
- Sets remaining tilts to `False`
- **Note**: Views not in taSolution.log are always set to `False`


### 4. All True Mode
When `--all-true`:
- Sets all `UseTilt` values to `True`
- Useful for testing or resetting configurations

## Expected Directory Structure

```
project_directory/
├── *.xml                          # XML files to process
├── backup_xml/                    # Backup directory (auto-created)
└── tiltstack/                     # Tiltstack logs directory
    └── [xml_basename]/
        └── taSolution.log          # Log file for each XML
```

## Example Workflows

### Standard WARP Processing
Remove views that failed alignment:
```bash
python remove_skipped_view.py ./ --xml-pattern "TS_*.xml"
```

### Low-dose Processing
Keep only the 15 lowest-dose tilts:
```bash
python remove_skipped_view.py ./ --n-tilts 15
```

### Reset All Tilts
Set all tilts to True for reprocessing:
```bash
python remove_skipped_view.py ./ --all-true
```

## Safety Features

- **Backup protection**: Won't overwrite existing backup files
- **Validation**: Checks for required log files before processing
- **Error handling**: Gracefully handles missing files or invalid data
- **Change tracking**: Reports number of modifications made

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -m 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Open a Pull Request

## License

This project is released under a permissive open-source license. You are free to use, modify, and distribute it for any purpose. See the LICENSE file for more information.

## Acknowledgments

- Designed for [WARP2.0](https://github.com/warpem/warp) cryo-electron tomography workflows 
- Built with pandas, click, and lxml
- Thanks to Alister Burt and Dimitry Tegunov for the support 