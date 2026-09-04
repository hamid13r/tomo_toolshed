"""Tests for the filament tracer, focused on the .bild opt-in behavior.

These use a small synthetic mask (a single straight tube) so the tracing runs
end to end quickly and deterministically.
"""

import numpy as np
import mrcfile
import pytest
from click.testing import CliRunner

from tomo_toolshed.filament_tracer import core
from tomo_toolshed.filament_tracer.cli import trace_filaments


def _write_tube_mask(path, pixel_size=10.0):
    """A single straight 3x3 tube along Z: enough voxels + length to trace."""
    vol = np.zeros((40, 12, 12), dtype=np.uint8)
    vol[3:35, 5:8, 5:8] = 1          # 32 * 3 * 3 = 288 voxels, ~320 A long
    with mrcfile.new(str(path), overwrite=True) as m:
        m.set_data(vol)
        m.voxel_size = pixel_size
    return path


# Parameters that make the synthetic tube pass the filters.
COMMON = dict(pixel_size=10.0, min_voxels=10, min_path_a=100.0, micrograph="mask.mrc")


def test_traces_particles(tmp_path):
    mask = _write_tube_mask(tmp_path / "mask.mrc")
    star = tmp_path / "particles.star"
    df, records = core.trace_filaments(str(mask), str(star), **COMMON)
    assert star.exists()
    assert len(records) > 0
    assert len(df) == len(records)


def test_no_bild_by_default(tmp_path):
    mask = _write_tube_mask(tmp_path / "mask.mrc")
    star = tmp_path / "particles.star"
    core.trace_filaments(str(mask), str(star), write_bild_file=False, **COMMON)
    assert star.exists()
    assert list(tmp_path.glob("*.bild")) == []


def test_bild_written_when_flag_on(tmp_path):
    mask = _write_tube_mask(tmp_path / "mask.mrc")
    star = tmp_path / "particles.star"
    bild = core.default_bild_path(str(star))
    core.trace_filaments(str(mask), str(star), write_bild_file=True,
                         bild_path=bild, **COMMON)
    assert star.exists()
    assert (tmp_path / "particles.bild").exists()


def test_star_byte_identical_with_and_without_bild(tmp_path):
    mask = _write_tube_mask(tmp_path / "mask.mrc")

    star_off = tmp_path / "off.star"
    core.trace_filaments(str(mask), str(star_off), write_bild_file=False, **COMMON)

    star_on = tmp_path / "on.star"
    core.trace_filaments(str(mask), str(star_on), write_bild_file=True,
                         bild_path=str(tmp_path / "on.bild"), **COMMON)

    assert star_off.read_bytes() == star_on.read_bytes()
    # The .bild only appears for the flagged run.
    assert not (tmp_path / "off.bild").exists()
    assert (tmp_path / "on.bild").exists()


def test_cli_runs_non_interactively(tmp_path):
    mask = _write_tube_mask(tmp_path / "mask.mrc")
    star = tmp_path / "out.star"
    runner = CliRunner()
    # No `input=`: if the command prompted for anything this would hang/fail.
    result = runner.invoke(trace_filaments, [
        str(mask), "-o", str(star),
        "--pixel-size", "10", "--min-length", "100", "--min-voxels", "10",
    ])
    assert result.exit_code == 0, result.output
    assert star.exists()
    assert not star.with_suffix(".bild").exists()

    # Same command with --bild writes the overlay.
    star2 = tmp_path / "out2.star"
    result2 = runner.invoke(trace_filaments, [
        str(mask), "-o", str(star2),
        "--pixel-size", "10", "--min-length", "100", "--min-voxels", "10",
        "--bild",
    ])
    assert result2.exit_code == 0, result2.output
    assert (tmp_path / "out2.bild").exists()
