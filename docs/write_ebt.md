# write-ebt — build an etomo batchruntomo `.ebt` project

Part of [tomo_toolshed](../README.md). Scans a directory laid out as **one
subdirectory per tilt series** and writes an etomo **batchruntomo project file**
(`.ebt`): a dataset-wide header followed by one row per tilt series, each pointing
at that series' `.st` stack.

## What an `.ebt` file is

A `.ebt` is the project file etomo's **batchruntomo** interface reads to drive a
batch of tilt-series reconstructions. It is a Java `.properties` file: a header of
dataset-wide settings (`meta.dataset.*`, `meta.RootName`, …) followed by a numbered
list of *rows*, one per tilt series. Each row records where that series' `.st`
stack lives (`meta.row.ebtN.OrigStack` / `meta.ref.ebtN`) plus its per-step state.
etomo strips the leading whitespace when it reads the file, so the header's
indentation is cosmetic and left exactly as etomo itself writes it.

You'd generate one when you have many tilt series to reconstruct and want to
populate the batchruntomo table in one shot instead of adding each stack by hand.

## Expected directory layout

Point `--root` (default: the current directory) at a directory holding one
subdirectory per tilt series, each containing that series' `.st` stack named after
the subdirectory:

```
ts_data/
├── ts_001/
│   └── ts_001.st
├── ts_002/
│   └── ts_002.st
└── ts_003/
    └── ts_003.st
```

Every **subdirectory** of `--root` becomes a row whose stack path is
`<root>/<name>/<name>.st`. Loose files in `--root` are ignored — including the
`.ebt` file being written, so it is safe to write the output into the same
directory you are scanning.

> **Note:** the tool only records the expected `<name>/<name>.st` path for each
> subdirectory; it does not check that the `.st` file actually exists.

## Platform: match the machine that *opens* the file

The one thing that differs between platforms is how the `.st` paths are spelled:

- `--platform linux` — forward-slash paths (`/data/ts_001/ts_001.st`).
- `--platform windows` — backslash paths with the backslashes doubled and the
  drive-letter colon escaped (`C\:\\data\\ts_001\\ts_001.st`), as a Windows Java
  `.properties` value requires.

Pick the platform of the machine that will **open** the `.ebt` in etomo, not the
one generating it. Projects are routinely created on one machine (say a Linux
cluster) and opened on another (a Windows workstation), which is exactly why
`--platform` is a required choice rather than auto-detected from the host OS.

## Usage

```bash
tomo_toolshed write-ebt --o batch.ebt --platform linux
```

Run with no options and it prompts for the `.ebt` name and the target platform:

```bash
tomo_toolshed write-ebt
Enter the name of the ebt file: batch.ebt
Target platform for the .st paths (linux, windows): linux
```

Examples:

```bash
# Scan the current directory, Linux paths
tomo_toolshed write-ebt --o batch.ebt --platform linux

# Scan another directory, Windows paths for a Windows etomo
tomo_toolshed write-ebt --o batch.ebt --platform windows --root ts_data

# Override the dataset RootName
tomo_toolshed write-ebt --o batch.ebt --platform linux --root-name myDataset
```

## Options

| Option | Default | Meaning |
|---|---|---|
| `--o TEXT` | *(prompted)* | Name/path of the `.ebt` file to write. |
| `--platform {linux,windows}` | *(prompted)* | `.st` path style; match the machine that will **open** the file in etomo. |
| `--root, -r PATH` | `.` | Directory to scan for tilt-series subdirectories. |
| `--root-name TEXT` | `batchMay08-150740` | Value written into `meta.RootName`. |

## Behavior notes

- **Only subdirectories become rows.** Loose files in `--root` (including the
  output `.ebt`) are skipped. Rows are numbered from 1 in `os.listdir()` order and
  the trailing `meta.ref.ebt.lastID` ends at the last row.
- **The output is overwritten** if it already exists.
- **The header is written verbatim**, including its leading indentation and the
  spelling of every key, for byte-for-byte compatibility with files etomo accepts.
