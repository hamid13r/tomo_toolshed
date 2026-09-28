"""Tests for write-ebt: subdirectory selection, path styles, and header parity.

All fixtures are empty directories under ``tmp_path`` -- no binaries and no
display are needed, so the suite runs headless like the rest of the project.
"""

from pathlib import Path

import pytest
from click.testing import CliRunner

from tomo_toolshed.write_ebt import core
from tomo_toolshed.write_ebt.cli import write_ebt


def _make_tree(root, subdirs=("ts_002", "ts_001", "ts_003"), loose=("loose.txt",)):
    """Create ``subdirs`` and ``loose`` files under ``root``; return ``root``."""
    for name in subdirs:
        (root / name).mkdir()
    for name in loose:
        (root / name).write_text("not a directory\n")
    return root


# ---------------------------------------------------------------------------
# Subdirectory selection: only subdirs become rows, in os.listdir order.
# ---------------------------------------------------------------------------
def test_only_subdirectories_become_rows(tmp_path):
    _make_tree(tmp_path)
    out = tmp_path / "batch.ebt"

    runner = CliRunner()
    result = runner.invoke(write_ebt, ["--o", str(out), "--platform", "linux",
                                       "--root", str(tmp_path)])
    assert result.exit_code == 0, result.output
    text = out.read_text()

    # os.listdir order is filesystem-dependent, so assert against it directly.
    expected = [n for n in __import__("os").listdir(tmp_path)
                if (tmp_path / n).is_dir()]
    row_dirs = [line.split("=", 1)[1]
                for line in text.splitlines()
                if ".RowNumber=" not in line and line.endswith(".st")
                and ".OrigStack=" in line]
    # Extract the subdir name from each OrigStack path (<root>/<name>/<name>.st).
    got = [Path(p).stem for p in row_dirs]
    assert got == expected

    # No row for the loose file or for the output .ebt itself.
    assert "loose.txt" not in text
    assert "batch.ebt" not in text
    # One row block per subdirectory: RowNumber lines == number of subdirs.
    assert text.count(".RowNumber=") == len(expected)


def test_output_ebt_in_scanned_dir_is_not_self_listed(tmp_path):
    """Writing the .ebt into the scanned directory must not add a row for it."""
    _make_tree(tmp_path, loose=())
    out = tmp_path / "batch.ebt"
    core.write_ebt(str(out), str(tmp_path), "linux")
    # Re-scan: the .ebt now exists on disk but is a file, so it is not a row.
    dirs = core.list_series_dirs(str(tmp_path))
    assert "batch.ebt" not in dirs
    assert out.read_text().count(".RowNumber=") == 3


# ---------------------------------------------------------------------------
# Path styles.
# ---------------------------------------------------------------------------
def test_linux_paths_use_forward_slashes(tmp_path):
    (tmp_path / "ts_001").mkdir()
    text = core.build_ebt(["ts_001"], str(tmp_path), "linux")
    expected = str((tmp_path / "ts_001").resolve() / "ts_001.st")
    assert f"meta.row.ebt1.OrigStack={expected}" in text
    assert "\\" not in text


def test_windows_paths_are_doubled_and_colon_escaped(tmp_path):
    # Use a fake root containing a drive-letter colon to exercise the escaping.
    root = "C:\\data"
    text = core.build_ebt(["ts_001"], root, "windows")
    line = next(l for l in text.splitlines() if l.startswith("meta.row.ebt1.OrigStack="))
    value = line.split("=", 1)[1]
    # Backslashes are doubled...
    assert "\\\\" in value
    assert "\\\\\\" not in value.replace("\\\\", "")  # only ever pairs
    # ...and the drive-letter colon is escaped as \: after doubling.
    assert "\\:" in value
    assert value.endswith("ts_001.st")


def test_format_st_path_windows_order_matches_original(tmp_path):
    """Escaping order: double the backslashes first, then escape the colon."""
    got = core.format_st_path("ts_001", "C:\\data", "windows")
    base = __import__("os").path.abspath("C:\\data/ts_001")  # abspath on this host
    expected = (base + "\\" + "ts_001" + ".st").replace("\\", "\\\\").replace(":", "\\:")
    assert got == expected


# ---------------------------------------------------------------------------
# Header parity and lastID.
# ---------------------------------------------------------------------------
def test_header_written_verbatim_and_lastid_ends_at_last_row(tmp_path):
    _make_tree(tmp_path, subdirs=("a", "b"), loose=())
    text = core.build_ebt(core.list_series_dirs(str(tmp_path)), str(tmp_path), "linux")

    # Header appears verbatim, leading indentation and odd keys intact.
    header = core.HEADER.format(root_name=core.DEFAULT_ROOT_NAME)
    assert text.startswith(header)
    assert "    meta.dataset.fcaleFromZ=0.33\n" in text
    assert "    meta.RootName=batchMay08-150740\n" in text

    # The final lastID names the last row; two subdirs -> ends at ebt2.
    assert text.rstrip().endswith("meta.ref.ebt.lastID=ebt2")


def test_root_name_is_configurable_and_default_is_unchanged(tmp_path):
    (tmp_path / "a").mkdir()
    custom = core.build_ebt(["a"], str(tmp_path), "linux", root_name="myProject")
    assert "meta.RootName=myProject\n" in custom
    assert "meta.RootName=batchMay08-150740\n" not in custom

    default = core.build_ebt(["a"], str(tmp_path), "linux")
    assert "meta.RootName=batchMay08-150740\n" in default


def test_empty_directory_produces_header_only(tmp_path):
    text = core.build_ebt([], str(tmp_path), "linux")
    assert text == core.HEADER.format(root_name=core.DEFAULT_ROOT_NAME)
    assert ".RowNumber=" not in text


# ---------------------------------------------------------------------------
# CLI option validation.
# ---------------------------------------------------------------------------
def test_platform_rejects_other_values(tmp_path):
    runner = CliRunner()
    result = runner.invoke(write_ebt, ["--o", str(tmp_path / "x.ebt"),
                                       "--platform", "mac", "--root", str(tmp_path)])
    assert result.exit_code != 0
    assert "mac" in result.output
    assert "linux" in result.output and "windows" in result.output


def test_platform_accepts_linux_and_windows(tmp_path):
    (tmp_path / "ts_001").mkdir()
    runner = CliRunner()
    for platform in ("linux", "windows"):
        out = tmp_path / f"{platform}.ebt"
        result = runner.invoke(write_ebt, ["--o", str(out), "--platform", platform,
                                           "--root", str(tmp_path)])
        assert result.exit_code == 0, result.output
        assert out.exists()
