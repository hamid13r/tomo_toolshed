"""Tests for scale-star: coordinate rescaling, pixel-size override, and the
mismatch prompt, across the four star flavors.

The four real fixtures live in ``tests/duplicate_remover/`` and are read from
there (read-only; everything is written into ``tmp_path``). Smaller synthetic
frames are used where the expected outcome is clearer by construction.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import starfile
from click.testing import CliRunner

from tomo_toolshed.scale_star import core
from tomo_toolshed.scale_star.cli import scale_star

DATA = Path(__file__).parent.parent / "duplicate_remover"
FILES = {
    "relion3": DATA / "relion_3_example.star",
    "relion4": DATA / "relion_4_example.star",
    "relion5": DATA / "relion_5_2D_example.star",
    "m": DATA / "mtools_example.star",
}


def _run(args, **kwargs):
    return CliRunner().invoke(scale_star, args, **kwargs)


# ---------------------------------------------------------------------------
# Core: coordinate detection and scaling math.
# ---------------------------------------------------------------------------
def test_find_coordinate_columns_rln_and_wrp():
    rln = pd.DataFrame({"rlnCoordinateX": [1.0], "rlnCoordinateY": [2.0],
                        "rlnCoordinateZ": [3.0], "rlnOriginXAngst": [9.0]})
    assert core.find_coordinate_columns(rln) == [
        "rlnCoordinateX", "rlnCoordinateY", "rlnCoordinateZ"]

    m = pd.DataFrame({"wrpCoordinateX1": [1.0], "wrpCoordinateY1": [2.0],
                      "wrpCoordinateZ1": [3.0], "wrpCoordinateX2": [4.0],
                      "wrpCoordinateY2": [5.0], "wrpCoordinateZ2": [6.0],
                      "wrpSourceName": ["a"]})
    assert core.find_coordinate_columns(m) == [
        "wrpCoordinateX1", "wrpCoordinateY1", "wrpCoordinateZ1",
        "wrpCoordinateX2", "wrpCoordinateY2", "wrpCoordinateZ2"]


def test_find_coordinate_columns_2d_ok_and_empty_raises():
    twod = pd.DataFrame({"rlnCoordinateX": [1.0], "rlnCoordinateY": [2.0]})
    assert core.find_coordinate_columns(twod) == ["rlnCoordinateX", "rlnCoordinateY"]
    with pytest.raises(core.ScaleStarError):
        core.find_coordinate_columns(pd.DataFrame({"rlnMicrographName": ["a"]}))


def test_angst_columns_are_not_matched_as_coordinates():
    df = pd.DataFrame({"rlnCenteredCoordinateXAngst": [1.0],
                       "rlnHelicalTrackLengthAngst": [2.0],
                       "rlnCoordinateX": [3.0]})
    assert core.find_coordinate_columns(df) == ["rlnCoordinateX"]


def test_scale_particles_multiple_wrp_indices():
    df = pd.DataFrame({"wrpCoordinateX1": [1.0], "wrpCoordinateY1": [2.0],
                       "wrpCoordinateZ1": [3.0], "wrpCoordinateX2": [4.0],
                       "wrpCoordinateY2": [5.0], "wrpCoordinateZ2": [6.0],
                       "other": [99]})
    cols = core.find_coordinate_columns(df)
    out = core.scale_particles(df, cols, 2.0, {"X": 0.0, "Y": 0.0, "Z": 0.0})
    assert list(out[cols].iloc[0]) == [2.0, 4.0, 6.0, 8.0, 10.0, 12.0]
    assert out["other"].iloc[0] == 99   # untouched


def test_scale_particles_shift_is_after_scaling_in_output_pixels():
    df = pd.DataFrame({"rlnCoordinateX": [10.0], "rlnCoordinateY": [10.0],
                       "rlnCoordinateZ": [10.0]})
    cols = core.find_coordinate_columns(df)
    out = core.scale_particles(df, cols, 2.0, {"X": 5.0, "Y": 0.0, "Z": -3.0})
    # new = old * factor + shift
    assert out["rlnCoordinateX"].iloc[0] == 25.0   # 10*2 + 5
    assert out["rlnCoordinateY"].iloc[0] == 20.0   # 10*2 + 0
    assert out["rlnCoordinateZ"].iloc[0] == 17.0   # 10*2 - 3


# ---------------------------------------------------------------------------
# CLI: per-flavor scaling round-trips.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("flavor", ["relion3", "relion4", "relion5", "m"])
def test_cli_scales_coordinates_per_flavor(tmp_path, flavor):
    src = FILES[flavor]
    out = tmp_path / "out.star"
    result = _run(["--i", str(src), "--o", str(out),
                   "--input-pixel-size", "10", "--output-pixel-size", "5", "--yes"])
    assert result.exit_code == 0, result.output

    sb = starfile.read(src, always_dict=True)
    db = starfile.read(out, always_dict=True)
    # Same block names, in the same order, and same row counts.
    assert list(sb.keys()) == list(db.keys())
    sp_key = "particles" if "particles" in sb else next(iter(sb))
    sp, dp = sb[sp_key], db[sp_key]
    assert len(sp) == len(dp)

    coord_cols = core.find_coordinate_columns(sp)
    for c in coord_cols:
        assert np.allclose(dp[c].astype(float), sp[c].astype(float) * 2.0)


def test_cli_shift_applied_after_scaling(tmp_path):
    src = FILES["relion3"]
    out = tmp_path / "out.star"
    result = _run(["--i", str(src), "--o", str(out),
                   "--input-pixel-size", "10", "--output-pixel-size", "5",
                   "--shift-z", "7", "--yes"])
    assert result.exit_code == 0, result.output
    sp = starfile.read(src, always_dict=True)[""]
    dp = starfile.read(out, always_dict=True)[""]
    assert np.allclose(dp["rlnCoordinateZ"].astype(float),
                       sp["rlnCoordinateZ"].astype(float) * 2.0 + 7.0)
    # X/Y have no shift.
    assert np.allclose(dp["rlnCoordinateX"].astype(float),
                       sp["rlnCoordinateX"].astype(float) * 2.0)


# ---------------------------------------------------------------------------
# Columns left alone: *Angst and rlnImagePixelSize; overrides applied.
# ---------------------------------------------------------------------------
def test_angst_columns_are_unchanged_by_scaling():
    # Core-level exactness: scale_particles must not touch *Angst columns.
    df = pd.DataFrame({"rlnCoordinateX": [100.0], "rlnCoordinateY": [200.0],
                       "rlnCoordinateZ": [300.0], "rlnOriginXAngst": [-11.41748],
                       "rlnHelicalTrackLengthAngst": [3.14159]})
    cols = core.find_coordinate_columns(df)
    out = core.scale_particles(df, cols, 2.0, {"X": 1.0, "Y": 1.0, "Z": 1.0})
    assert out["rlnOriginXAngst"].equals(df["rlnOriginXAngst"])
    assert out["rlnHelicalTrackLengthAngst"].equals(df["rlnHelicalTrackLengthAngst"])


def test_cli_relion4_overrides_and_preserves_the_right_pixel_sizes(tmp_path):
    src = FILES["relion4"]
    out = tmp_path / "out.star"
    result = _run(["--i", str(src), "--o", str(out),
                   "--input-pixel-size", "10", "--output-pixel-size", "5", "--yes"])
    assert result.exit_code == 0, result.output
    sb = starfile.read(src, always_dict=True)
    db = starfile.read(out, always_dict=True)
    # Overridden coordinate pixel sizes -> 5.
    assert (db["particles"]["rlnPixelSize"] == 5.0).all()
    assert (db["optics"]["rlnMicrographPixelSize"] == 5.0).all()
    # rlnImagePixelSize and the Angst columns are left exactly as they were.
    assert db["optics"]["rlnImagePixelSize"].equals(sb["optics"]["rlnImagePixelSize"])
    for c in ("rlnOriginXAngst", "rlnOriginYAngst", "rlnOriginZAngst",
              "rlnHelicalTrackLengthAngst"):
        assert np.allclose(db["particles"][c].astype(float),
                           sb["particles"][c].astype(float))


def test_cli_relion5_overrides_tomo_tiltseries_pixel_size(tmp_path):
    src = FILES["relion5"]
    out = tmp_path / "out.star"
    result = _run(["--i", str(src), "--o", str(out),
                   "--input-pixel-size", "2.11", "--output-pixel-size", "1.055"])
    assert result.exit_code == 0, result.output
    db = starfile.read(out, always_dict=True)
    assert (db["optics"]["rlnTomoTiltSeriesPixelSize"] == 1.055).all()
    # rlnImagePixelSize (4.22) is not a coordinate pixel size -> untouched.
    assert (db["optics"]["rlnImagePixelSize"] == 4.22).all()
    # The general block is carried through.
    assert "general" in db


def test_cli_preserves_noncoordinate_columns_and_rowcount(tmp_path):
    src = FILES["relion3"]
    out = tmp_path / "out.star"
    result = _run(["--i", str(src), "--o", str(out),
                   "--input-pixel-size", "9.98", "--output-pixel-size", "5"])
    assert result.exit_code == 0, result.output
    sp = starfile.read(src, always_dict=True)[""]
    dp = starfile.read(out, always_dict=True)[""]
    assert list(sp.columns) == list(dp.columns)
    assert len(sp) == len(dp)
    # A representative non-coordinate, non-pixel-size column is unchanged.
    assert dp["rlnMicrographName"].equals(sp["rlnMicrographName"])


# ---------------------------------------------------------------------------
# Mismatch handling.
# ---------------------------------------------------------------------------
def test_cli_mismatch_abort_on_no(tmp_path):
    out = tmp_path / "out.star"
    result = _run(["--i", str(FILES["relion3"]), "--o", str(out),
                   "--input-pixel-size", "10", "--output-pixel-size", "5"],
                  input="n\n")
    assert result.exit_code != 0
    assert "warning" in result.output
    assert not out.exists()


def test_cli_mismatch_proceed_on_yes_uses_user_value(tmp_path):
    src = FILES["relion3"]
    out = tmp_path / "out.star"
    result = _run(["--i", str(src), "--o", str(out),
                   "--input-pixel-size", "10", "--output-pixel-size", "5"],
                  input="y\n")
    assert result.exit_code == 0, result.output
    assert out.exists()
    sp = starfile.read(src, always_dict=True)[""]
    dp = starfile.read(out, always_dict=True)[""]
    # factor uses the user's 10 (not the file's 9.98): new = old * 10/5 = old*2.
    assert np.allclose(dp["rlnCoordinateX"].astype(float),
                       sp["rlnCoordinateX"].astype(float) * 2.0)
    assert (dp["rlnPixelSize"] == 5.0).all()


def test_cli_yes_never_prompts(tmp_path):
    out = tmp_path / "out.star"
    # No stdin supplied: if it tried to prompt it would error out.
    result = _run(["--i", str(FILES["relion3"]), "--o", str(out),
                   "--input-pixel-size", "10", "--output-pixel-size", "5", "--yes"])
    assert result.exit_code == 0, result.output
    assert "warning" in result.output   # still warned
    assert out.exists()


def test_cli_no_mismatch_no_prompt(tmp_path):
    out = tmp_path / "out.star"
    result = _run(["--i", str(FILES["relion3"]), "--o", str(out),
                   "--input-pixel-size", "9.98", "--output-pixel-size", "5"])
    assert result.exit_code == 0, result.output
    assert "warning" not in result.output
    assert out.exists()


def test_cli_no_pixel_size_column_proceeds_with_note(tmp_path):
    out = tmp_path / "out.star"
    result = _run(["--i", str(FILES["m"]), "--o", str(out),
                   "--input-pixel-size", "10", "--output-pixel-size", "5"])
    assert result.exit_code == 0, result.output
    assert "no pixel-size column" in result.output
    assert out.exists()


# ---------------------------------------------------------------------------
# Bad input.
# ---------------------------------------------------------------------------
def test_cli_rejects_nonpositive_pixel_size(tmp_path):
    out = tmp_path / "out.star"
    result = _run(["--i", str(FILES["relion3"]), "--o", str(out),
                   "--input-pixel-size", "10", "--output-pixel-size", "0"])
    assert result.exit_code != 0
    assert "must be > 0" in result.output


def test_cli_rejects_output_equal_input(tmp_path):
    src = tmp_path / "same.star"
    starfile.write(pd.DataFrame({"rlnCoordinateX": [1.0], "rlnCoordinateY": [2.0],
                                 "rlnCoordinateZ": [3.0]}), str(src), overwrite=True)
    result = _run(["--i", str(src), "--o", str(src),
                   "--input-pixel-size", "10", "--output-pixel-size", "5"])
    assert result.exit_code != 0
    assert "differ" in result.output


def test_cli_rejects_no_coordinate_columns(tmp_path):
    src = tmp_path / "nocoord.star"
    starfile.write(pd.DataFrame({"rlnMicrographName": ["a", "b"]}),
                   str(src), overwrite=True)
    out = tmp_path / "out.star"
    result = _run(["--i", str(src), "--o", str(out),
                   "--input-pixel-size", "10", "--output-pixel-size", "5"])
    assert result.exit_code != 0
    assert "no coordinate columns" in result.output


# ---------------------------------------------------------------------------
# --dry-run and the provenance comment.
# ---------------------------------------------------------------------------
def test_cli_dry_run_writes_nothing(tmp_path):
    out = tmp_path / "out.star"
    result = _run(["--i", str(FILES["relion3"]), "--o", str(out),
                   "--input-pixel-size", "9.98", "--output-pixel-size", "5",
                   "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "dry run" in result.output
    assert not out.exists()


def test_cli_creates_parent_dirs(tmp_path):
    out = tmp_path / "a" / "b" / "out.star"   # none of these exist yet
    result = _run(["--i", str(FILES["relion3"]), "--o", str(out),
                   "--input-pixel-size", "9.98", "--output-pixel-size", "5"])
    assert result.exit_code == 0, result.output
    assert out.exists()


def test_cli_comment_header_present_by_default_absent_with_no_comment(tmp_path):
    out_c = tmp_path / "c.star"
    out_n = tmp_path / "n.star"
    common = ["--i", str(FILES["relion3"]),
              "--input-pixel-size", "9.98", "--output-pixel-size", "5"]
    assert _run(common + ["--o", str(out_c)]).exit_code == 0
    assert _run(common + ["--o", str(out_n), "--no-comment"]).exit_code == 0

    # starfile writes its own leading comment regardless; our provenance header
    # (naming scale-star) is present only with --comment.
    text_c, text_n = out_c.read_text(), out_n.read_text()
    assert text_c.splitlines()[0].startswith("# Created by tomo_toolshed scale-star")
    assert "tomo_toolshed scale-star" in text_c
    assert "tomo_toolshed scale-star" not in text_n
