"""Tests for split-star: grouping, per-group output, and block preservation.

Fixtures are tiny synthetic star files written into ``tmp_path`` (no binaries,
no display), so the suite runs headless like the rest of the project.
"""

from pathlib import Path

import pandas as pd
import pytest
import starfile
from click.testing import CliRunner

from tomo_toolshed.split_star import core
from tomo_toolshed.split_star.cli import split_star


def _write_single_block(path, micrographs, extra=None):
    """A single-unnamed-block star file grouped by rlnMicrographName."""
    df = pd.DataFrame({
        "rlnCoordinateX": [float(i) for i in range(len(micrographs))],
        "rlnCoordinateY": [float(i) for i in range(len(micrographs))],
        "rlnMicrographName": list(micrographs),
    })
    if extra:
        for k, v in extra.items():
            df[k] = v
    starfile.write(df, str(path), overwrite=True)
    return path


def _write_multi_block(path, tomonames):
    """An optics + particles file grouped by rlnTomoName (RELION-4/5-ish)."""
    optics = pd.DataFrame({"rlnOpticsGroup": [1], "rlnImagePixelSize": [6.65]})
    particles = pd.DataFrame({
        "rlnCoordinateX": [float(i) for i in range(len(tomonames))],
        "rlnTomoName": list(tomonames),
        "rlnOpticsGroup": [1] * len(tomonames),
    })
    starfile.write({"optics": optics, "particles": particles}, str(path),
                   overwrite=True)
    return path


# ---------------------------------------------------------------------------
# Grouping-column detection and dir naming.
# ---------------------------------------------------------------------------
def test_group_column_autodetect_prefers_tomoname():
    df = pd.DataFrame({"rlnTomoName": ["a"], "rlnMicrographName": ["b"]})
    assert core.detect_group_column(df) == "rlnTomoName"


def test_group_column_falls_through_to_micrograph_then_source():
    assert core.detect_group_column(pd.DataFrame({"rlnMicrographName": ["b"]})) \
        == "rlnMicrographName"
    assert core.detect_group_column(pd.DataFrame({"wrpSourceName": ["c"]})) \
        == "wrpSourceName"


def test_group_column_override_must_exist():
    df = pd.DataFrame({"rlnMicrographName": ["b"]})
    assert core.detect_group_column(df, override="rlnMicrographName") == "rlnMicrographName"
    with pytest.raises(core.SplitStarError):
        core.detect_group_column(df, override="rlnTomoName")


def test_no_grouping_column_is_an_error():
    with pytest.raises(core.SplitStarError):
        core.detect_group_column(pd.DataFrame({"rlnCoordinateX": [1.0]}))


@pytest.mark.parametrize("name,expected", [
    ("ts_01.mrc.tomostar", "ts_01"),
    ("weird_name", "weird_name"),
    ("a.mrc.tomostar.extra", "a.mrc.tomostar.extra"),   # suffix not at the end
])
def test_group_dirname_strips_default_suffix(name, expected):
    assert core.group_dirname(name) == expected


def test_group_dirname_custom_suffix():
    assert core.group_dirname("ts_01.mrc", strip_suffix=".mrc") == "ts_01"


# ---------------------------------------------------------------------------
# plan_split: paths, label prefixing, order.
# ---------------------------------------------------------------------------
def test_plan_uses_label_prefix_and_all_suffix(tmp_path):
    src = _write_single_block(tmp_path / "in.star",
                              ["ts_01.mrc.tomostar", "ts_01.mrc.tomostar",
                               "ts_02.mrc.tomostar"])
    blocks, key = core.read_star(str(src))
    plan = core.plan_split(blocks[key], "rlnMicrographName", "EXP", outdir="out")
    names = [gn for gn, _, _ in plan]
    paths = [p for _, p, _ in plan]
    rows = [n for _, _, n in plan]
    assert names == ["ts_01.mrc.tomostar", "ts_02.mrc.tomostar"]   # first-appearance order
    assert paths[0] == str(Path("out") / "EXP_ts_01" / "EXP_ts_01_all.star")
    assert rows == [2, 1]


def test_plan_without_label_has_no_prefix(tmp_path):
    src = _write_single_block(tmp_path / "in.star", ["ts_01.mrc.tomostar"])
    blocks, key = core.read_star(str(src))
    plan = core.plan_split(blocks[key], "rlnMicrographName", None, outdir=".")
    _, path, _ = plan[0]
    import os
    assert path == os.path.join(".", "ts_01", "ts_01_all.star")


# ---------------------------------------------------------------------------
# Writing: per-group rows, block preservation.
# ---------------------------------------------------------------------------
def test_split_single_block_writes_one_file_per_group(tmp_path):
    src = _write_single_block(
        tmp_path / "in.star",
        ["ts_01.mrc.tomostar", "ts_01.mrc.tomostar", "ts_02.mrc.tomostar"])
    out = tmp_path / "out"
    runner = CliRunner()
    result = runner.invoke(split_star, ["--i", str(src), "--label", "EXP",
                                        "--outdir", str(out)])
    assert result.exit_code == 0, result.output

    f1 = out / "EXP_ts_01" / "EXP_ts_01_all.star"
    f2 = out / "EXP_ts_02" / "EXP_ts_02_all.star"
    assert f1.exists() and f2.exists()

    d1 = starfile.read(str(f1), always_dict=True)
    assert list(d1.keys()) == [""]                       # single unnamed block
    assert len(d1[""]) == 2
    assert set(d1[""]["rlnMicrographName"]) == {"ts_01.mrc.tomostar"}
    assert len(starfile.read(str(f2))) == 1


def test_split_multi_block_preserves_optics(tmp_path):
    src = _write_multi_block(tmp_path / "in.star",
                             ["A.mrc.tomostar", "A.mrc.tomostar", "B.mrc.tomostar"])
    out = tmp_path / "out"
    runner = CliRunner()
    result = runner.invoke(split_star, ["--i", str(src), "--label", "R",
                                        "--group-by", "rlnTomoName",
                                        "--outdir", str(out)])
    assert result.exit_code == 0, result.output

    fa = out / "R_A" / "R_A_all.star"
    d = starfile.read(str(fa), always_dict=True)
    assert list(d.keys()) == ["optics", "particles"]     # optics carried through
    assert len(d["particles"]) == 2
    assert d["optics"]["rlnImagePixelSize"].iloc[0] == pytest.approx(6.65)


def test_row_counts_partition_the_input(tmp_path):
    mics = (["a.mrc.tomostar"] * 3 + ["b.mrc.tomostar"] * 5 + ["c.mrc.tomostar"])
    src = _write_single_block(tmp_path / "in.star", mics)
    out = tmp_path / "out"
    runner = CliRunner()
    result = runner.invoke(split_star, ["--i", str(src), "--label", "X",
                                        "--outdir", str(out)])
    assert result.exit_code == 0, result.output
    total = sum(len(starfile.read(str(p)))
                for p in out.rglob("*_all.star"))
    assert total == len(mics)                             # no rows lost or duplicated


# ---------------------------------------------------------------------------
# CLI plumbing: dry-run, aliases, errors.
# ---------------------------------------------------------------------------
def test_dry_run_writes_nothing(tmp_path):
    src = _write_single_block(tmp_path / "in.star", ["a.mrc.tomostar"])
    out = tmp_path / "out"
    runner = CliRunner()
    result = runner.invoke(split_star, ["--i", str(src), "--label", "X",
                                        "--outdir", str(out), "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "would write" in result.output
    assert not out.exists()


def test_underscored_aliases_still_work(tmp_path):
    src = _write_multi_block(tmp_path / "in.star", ["A.mrc.tomostar"])
    out = tmp_path / "out"
    runner = CliRunner()
    result = runner.invoke(split_star, ["--i", str(src), "--label", "X",
                                        "--group_by", "rlnTomoName",
                                        "--outdir", str(out)])
    assert result.exit_code == 0, result.output
    assert (out / "X_A" / "X_A_all.star").exists()


def test_missing_group_column_is_a_clean_error(tmp_path):
    df = pd.DataFrame({"rlnCoordinateX": [1.0, 2.0]})
    src = tmp_path / "in.star"
    starfile.write(df, str(src), overwrite=True)
    runner = CliRunner()
    result = runner.invoke(split_star, ["--i", str(src), "--label", "X",
                                        "--outdir", str(tmp_path / "out")])
    assert result.exit_code != 0
    assert "grouping column" in result.output


def test_custom_strip_suffix(tmp_path):
    src = _write_single_block(tmp_path / "in.star", ["ts_01.mrc", "ts_02.mrc"])
    out = tmp_path / "out"
    runner = CliRunner()
    result = runner.invoke(split_star, ["--i", str(src), "--label", "X",
                                        "--outdir", str(out),
                                        "--strip-suffix", ".mrc"])
    assert result.exit_code == 0, result.output
    assert (out / "X_ts_01" / "X_ts_01_all.star").exists()
