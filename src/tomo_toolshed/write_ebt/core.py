"""Core logic for write-ebt: build an etomo batchruntomo ``.ebt`` project file.

A batchruntomo project (``.ebt``) is a Java ``.properties`` file that etomo reads
to drive a batch of tilt-series reconstructions. It is a header of dataset-wide
settings followed by one *row* per tilt series, each pointing at that series'
``.st`` stack. This module writes that file for a directory laid out as one
subdirectory per tilt series (``<name>/<name>.st``).

The two original per-platform scripts were identical apart from how the ``.st``
path was spelled. That single difference now lives in :func:`format_st_path`,
selected by ``platform``, so both variants share one code path. There is no
``click`` dependency here; the CLI layer in ``cli.py`` calls into these functions.

Note on the leading whitespace in :data:`HEADER`: these are Java ``.properties``
lines and etomo strips the leading indentation on read, so it is preserved
verbatim (indentation and all) rather than reformatted -- the goal is
byte-for-byte parity with the files etomo already accepts, including the odd keys
(e.g. ``meta.dataset.fcaleFromZ``).
"""

import os

# The RootName baked into the original scripts; exposed via --root-name so the
# default output is unchanged.
DEFAULT_ROOT_NAME = "batchMay08-150740"

# The dataset-wide header, byte-for-byte as the original scripts wrote it: a
# leading newline, every line indented with four spaces, and a trailing line of
# four spaces with no newline (so the first row line follows on that same line,
# exactly as before). ``meta.RootName`` is the one templated field. Do not
# reformat, dedent, or "fix" the keys -- etomo strips the indentation on read.
HEADER = '''
    meta.dataset=true
    meta.EnableStartingStep=false
    meta.dataset.Use.fakeSIRTiterations=false
    meta.dataset.header.open=true
    meta.dataset.fcaleFromZ=0.33
    meta.dataset.Preblend.BinByFactor=1
    meta.ref.ebt.lastID=ebt0
    meta.ProjectLog.FrameLocation.Y=26
    meta.dataset.Use.fcaleFromZ=false
    meta.ProjectLog.FrameLocation.X=26
    meta.dataset.Prenewst.BinByFactor=1
    meta.Version.Etomo.Modified=4.11.21
    meta.ProjectLog.FrameSize.Width=683
    meta.dataset.SizeOfPatchesXandY=680,680
    meta.dataset.autoFitRangeAndStep=true
    meta.dataset.ScaleToInteger=false
    meta.dataset.eraseGold.Fid=true
    meta.dataset.LocalAlignments=false
    meta.datasetTableHeader.open=true
    meta.dataset.LengthOfPieces=false
    meta.dataset.Postprocessing.header.open=false
    meta.dataset.sampleType.Cryo=false
    meta.ProjectLog.Visible=true
    meta.Status=Open
    meta.EndingStep.Use=false
    meta.dataset.hasGoldBeads=true
    meta.StartingStep.Use=false
    meta.Version.Etomo.Created=4.11.21
    meta.dataset.enableStretching=false
    meta.EndingStep=10
    meta.RootName={root_name}
    meta.dataset.sampleType.PlasticSection=true
    meta.dataset.eraseGold.3d=false
    meta.StartingStep=11
    meta.dataset.Use.findSecAddThickness=false
    meta.dataset.fitEveryImage=false
    meta.ProjectLog.FrameSize.Height=230
    meta.ImageFile.ImageFilenameStyle=MRC
    meta.dataset=true
    meta.datasetTableHeader.open=true
    meta.ref.ebt.lastID=ebt0
    '''


def format_st_path(directory, root, platform):
    """Return the ``.st`` path string for one tilt-series subdirectory.

    ``directory`` is the subdirectory name, ``root`` the directory it lives in.
    The absolute stack path is ``<root>/<directory>/<directory>.st``. This is the
    only place the two original scripts differed:

    - ``linux``: forward-slash separator, no escaping.
    - ``windows``: a single backslash separator, then the backslashes are doubled
      and the drive-letter colon is escaped as ``\\:`` -- in that order -- so the
      value is a valid Java ``.properties`` value on Windows.
    """
    base = os.path.abspath(os.path.join(root, directory))
    if platform == "windows":
        st_path = base + "\\" + directory + ".st"
        st_path = st_path.replace("\\", "\\\\")
        st_path = st_path.replace(":", "\\:")
        return st_path
    return base + "/" + directory + ".st"


def list_series_dirs(root):
    """Return the subdirectories of ``root``, in ``os.listdir`` order.

    Only subdirectories are returned -- loose files (including a ``.ebt`` written
    into the same directory) are skipped. This is the one intentional behavior
    change from the originals, which listed every entry.
    """
    return [name for name in os.listdir(root)
            if os.path.isdir(os.path.join(root, name))]


def _row_block(identifier, directory, root, platform):
    """Return the ``.ebt`` lines for a single row (one tilt series)."""
    st_path = format_st_path(directory, root, platform)
    return "".join(line + "\n" for line in (
        f"meta.row.ebt{identifier}.RowNumber={identifier}",
        f"meta.row.ebt{identifier}.OrigStack={st_path}",
        f"meta.row.ebt{identifier}.Tomogram.Done=false",
        f"meta.row.ebt{identifier}.Etomo.Enabled=false",
        f"meta.row.ebt{identifier}.Trimvol.Done=false",
        f"meta.row.ebt{identifier}.Run=true",
        f"meta.row.ebt{identifier}.Rec.Enabled=false",
        f"meta.row.ebt{identifier}.Log.Enabled=false",
        f"meta.ref.ebt{identifier}={st_path}",
        f"meta.row.ebt{identifier}.dual=false",
        f"meta.ref.ebt.lastID=ebt{identifier}",
    ))


def build_ebt(directories, root, platform, root_name=DEFAULT_ROOT_NAME):
    """Return the full ``.ebt`` file contents as a string.

    ``directories`` is the ordered list of tilt-series subdirectory names,
    ``root`` the directory they live in (used to resolve absolute ``.st`` paths),
    ``platform`` selects the path style (see :func:`format_st_path`), and
    ``root_name`` fills ``meta.RootName``. Rows are numbered from 1 in the given
    order and the trailing ``meta.ref.ebt.lastID`` ends at the last row.
    """
    text = HEADER.format(root_name=root_name)
    for identifier, directory in enumerate(directories, start=1):
        text += _row_block(identifier, directory, root, platform)
    return text


def write_ebt(output_path, root, platform, root_name=DEFAULT_ROOT_NAME):
    """Scan ``root``, build the ``.ebt`` contents, and write ``output_path``.

    Returns the list of tilt-series subdirectory names that became rows.
    """
    directories = list_series_dirs(root)
    with open(output_path, 'w') as f:
        f.write(build_ebt(directories, root, platform, root_name=root_name))
    return directories
