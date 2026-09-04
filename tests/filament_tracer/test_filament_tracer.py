"""Tests for the filament tracer, focused on the .bild opt-in behavior.

These use a small synthetic mask (a single straight tube) so the tracing runs
end to end quickly and deterministically.
"""

import numpy as np
import mrcfile
import pandas as pd
import pytest
import starfile
from click.testing import CliRunner

from tomo_toolshed.filament_tracer import core


def _star_payload(path):
    """Star content minus the ``# Created by ... at <timestamp>`` comment line.

    The ``starfile`` writer stamps the wall-clock time into a leading comment, so
    two runs are never byte-identical across a second boundary. Comparing the
    non-comment lines is the meaningful notion of "identical star files".
    """
    from pathlib import Path
    return "\n".join(
        line for line in Path(path).read_text().splitlines()
        if not line.lstrip().startswith("#")
    )
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

    assert _star_payload(star_off) == _star_payload(star_on)
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


# ---------------------------------------------------------------------------
# --random-rot: randomize the about-axis angle (rlnAngleRot) only.
# ---------------------------------------------------------------------------
def test_random_rot_same_seed_is_reproducible(tmp_path):
    mask = _write_tube_mask(tmp_path / "mask.mrc")
    a = tmp_path / "a.star"
    b = tmp_path / "b.star"
    core.trace_filaments(str(mask), str(a), random_rot=True, seed=42, **COMMON)
    core.trace_filaments(str(mask), str(b), random_rot=True, seed=42, **COMMON)
    assert _star_payload(a) == _star_payload(b)


def test_random_rot_off_is_byte_identical_to_default(tmp_path):
    mask = _write_tube_mask(tmp_path / "mask.mrc")
    default = tmp_path / "default.star"     # no random_rot kwarg at all
    flag_off = tmp_path / "off.star"        # random_rot explicitly False
    core.trace_filaments(str(mask), str(default), **COMMON)
    core.trace_filaments(str(mask), str(flag_off), random_rot=False, seed=42, **COMMON)
    assert _star_payload(default) == _star_payload(flag_off)


def test_random_rot_changes_only_rot_column(tmp_path):
    mask = _write_tube_mask(tmp_path / "mask.mrc")
    off = tmp_path / "off.star"
    on = tmp_path / "on.star"
    core.trace_filaments(str(mask), str(off), random_rot=False, **COMMON)
    core.trace_filaments(str(mask), str(on), random_rot=True, seed=7, **COMMON)

    df_off = starfile.read(str(off))
    df_on = starfile.read(str(on))

    assert len(df_on) > 1  # need several particles to show they differ

    # The prior for the (now unconstrained) about-axis angle is dropped.
    assert "rlnAngleRotPrior" in df_off.columns
    assert "rlnAngleRotPrior" not in df_on.columns

    # Randomized column: in range, and actually varies (not all equal, and not
    # equal to the traced/geometric values).
    rot_on = df_on["rlnAngleRot"].to_numpy(dtype=float)
    rot_off = df_off["rlnAngleRot"].to_numpy(dtype=float)
    assert np.all((rot_on >= 0.0) & (rot_on < 360.0))
    assert np.unique(rot_on).size > 1
    assert not np.allclose(np.sort(rot_on), np.sort(rot_off))

    # Every other column shared by the two runs is identical. Diff to prove it.
    shared = [c for c in df_on.columns if c != "rlnAngleRot"]
    pd.testing.assert_frame_equal(
        df_off[shared].reset_index(drop=True),
        df_on[shared].reset_index(drop=True),
        check_dtype=False,
    )


def test_random_rot_draws_and_prints_a_seed_when_omitted(tmp_path):
    mask = _write_tube_mask(tmp_path / "mask.mrc")
    star = tmp_path / "out.star"
    runner = CliRunner()
    result = runner.invoke(trace_filaments, [
        str(mask), "-o", str(star),
        "--pixel-size", "10", "--min-length", "100", "--min-voxels", "10",
        "--random-rot",
    ])
    assert result.exit_code == 0, result.output
    assert "random-rot: drew seed" in result.output
