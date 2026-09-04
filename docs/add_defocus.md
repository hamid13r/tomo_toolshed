# add-defocus — Adding CTF defocus values to an IsoNet star file

Part of [tomo_toolshed](../README.md). Fills in the `_rlnDefocus` column of an
IsoNet star file with the per-tomogram average defocus read from the matching
WarpTools tilt-series XML files.

## Overview

`isonet.py prepare_star` builds a star file listing your tomograms, but the
`_rlnDefocus` column is left as a placeholder (usually `0.0`) — it does not know
your CTF estimates. This tool fills that column in:

1. For each row, take `_rlnTomoName` (e.g. `rec/Position_2_2.mrc_10.00Apx.mrc`)
   and extract the tilt-series base name (`Position_2_2`).
2. Find the corresponding Warp XML (`<xml-dir>/Position_2_2.mrc.xml`).
3. Average the per-tilt defocus values in that file's `<GridCTF>` node,
   converting microns to Angstroms (`value × 10000`).
4. Write the rounded mean into `_rlnDefocus`.

Run it after `prepare_star` and before the steps that use defocus (e.g.
`deconv`). Column layout, ordering, and the rest of the star file are preserved;
only the defocus field of each data row changes.

## Installation

Install the whole toolshed (see the [root README](../README.md)):

```bash
git clone https://github.com/hamid13r/tomo_toolshed.git
cd tomo_toolshed
pip install -e .
```

This provides the `tomo_toolshed` command; the tool below is the `add-defocus`
subcommand.

## Usage

### Typical workflow

```bash
# 1. IsoNet builds the star file (defocus column is a placeholder)
isonet.py prepare_star rec --output_star tomos.star --pixel_size 10.0

# 2. Fill in the defocus from the Warp XML files
tomo_toolshed add-defocus --star tomos.star --xml-dir xml
```

Expected layout: reconstructions referenced by the star file, and a directory of
Warp XML files, one per tilt series:

```
project/
├── tomos.star            # from isonet.py prepare_star
├── rec/
│   ├── Position_2_2.mrc_10.00Apx.mrc
│   └── ...
└── xml/
    ├── Position_2_2.mrc.xml
    └── ...
```

### Other examples

```bash
# See what would change without touching the star file
tomo_toolshed add-defocus --star tomos.star --xml-dir xml --dry-run

# Keep the original and write a new file
tomo_toolshed add-defocus --star tomos.star --xml-dir xml -o tomos_defocus.star

# Abort instead of skipping when an XML is missing or a name doesn't parse
tomo_toolshed add-defocus --star tomos.star --xml-dir xml --strict

# XMLs named "<base>.xml" rather than "<base>.mrc.xml"
tomo_toolshed add-defocus --star tomos.star --xml-dir xml --xml-suffix .xml
```

### Options

| Option | Default | Description |
|---|---|---|
| `--star PATH` | *(required)* | IsoNet star file to update. |
| `--xml-dir PATH` | `xml` | Directory containing the Warp XML files. |
| `-o, --output PATH` | *(in place)* | Write the result here instead of editing `--star`. |
| `--xml-suffix TEXT` | `.mrc.xml` | Suffix appended to the tomogram base name to find its XML. |
| `--tomoname-regex TEXT` | `^(?:.*/)?(.+)\.mrc_[\d.]+Apx\.mrc$` | Regex whose first group extracts the base name from `_rlnTomoName`. |
| `--strict` | off | Fail on the first unparseable row or missing XML instead of skipping it. |
| `--dry-run` | off | Report the defocus values without writing anything. |
| `-q, --quiet` | off | Only print the final summary. |

## Behavior notes

- **In place by default.** The star file is rewritten unless `--output` is
  given. It is cheap to regenerate with `prepare_star`, but use `--dry-run`
  first if you want to look before you leap.
- **Skips rather than fails.** A row whose tomogram name doesn't match
  `--tomoname-regex`, or whose XML is missing/unreadable, is left untouched and
  reported on stderr at the end. `--strict` turns the first such row into an
  error and writes nothing.
- **Units.** `<GridCTF>` values are in microns; each is multiplied by 10000 and
  truncated to whole Angstroms before averaging, matching Warp/RELION
  conventions. The written value is the rounded mean.
- **Separators.** Rows are re-joined with tabs if the original row used tabs,
  otherwise with single spaces.

## Troubleshooting

| Message | Cause |
|---|---|
| `could not parse tomogram name: ...` | `_rlnTomoName` doesn't look like `<base>.mrc_<pixelsize>Apx.mrc`. Pass a matching `--tomoname-regex`. |
| `missing XML file: xml/...` | The XML name doesn't match the tomogram base name. Check `--xml-dir` and `--xml-suffix`. |
| `No <GridCTF> node in ...` | The XML has no CTF estimate yet — run CTF estimation in Warp first. |
| `... has no _rlnDefocus column` | The star file isn't an IsoNet `prepare_star` output, or its header is missing the column. |

## Tests

```bash
pip install -e ".[test]"
pytest tests/add_defocus
```
