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
    ("ts_01.mrc.tomostar", "ts_01"),                    # RELION 4
    ("Position_1_2.tomostar", "Position_1_2"),          # RELION 5 / M
    ("foo_microtubule.mrc", "foo_microtubule"),         # RELION 3
    ("weird_name", "weird_name"),                       # nothing to strip
    ("a.mrc.tomostar.extra", "a.mrc.tomostar.extra"),   # suffix not at the end
])
def test_group_dirname_strips_default_suffixes_by_flavor(name, expected):
    assert core.group_dirname(name) == expected


def test_group_dirname_longest_suffix_wins():
    # .mrc.tomostar must win over .tomostar so the base is not left as 'x.mrc'.
    assert core.group_dirname("x.mrc.tomostar") == "x"


@pytest.mark.parametrize("name,expected", [
    ("Position_1.mrc_9.98Apx.mrc", "Position_1"),       # typical float
    ("foo.mrc_10Apx.mrc", "foo"),                       # integer pixel size
    ("bar.mrc_4.22Apx.mrc", "bar"),
])
def test_group_dirname_strips_warp_pixelsize_pattern(name, expected):
    # The .mrc_<pixelsize>Apx.mrc pattern must win over the literal .mrc, which
    # would otherwise leave e.g. 'Position_1.mrc_9.98Apx'.
    assert core.group_dirname(name) == expected


def test_group_dirname_custom_suffix_disables_pattern():
    assert core.group_dirname("ts_01.mrc", strip_suffixes=".mrc") == "ts_01"
    assert core.group_dirname("ts_01.xyz", strip_suffixes=[".xyz"]) == "ts_01"
    # With an explicit suffix list and no patterns, the Warp pattern does not fire.
    assert core.group_dirname("foo.mrc_9.98Apx.mrc",
                              strip_suffixes=[".mrc"], strip_patterns=()) \
        == "foo.mrc_9.98Apx"


# ---------------------------------------------------------------------------
# plan_split: paths, label prefixing, order.
# ---------------------------------------------------------------------------
def test_plan_uses_label_prefix_and_all_suffix(tmp_path):
    src = _write_single_block(tmp_path / "in.star",
                              ["ts_01.mrc.tomostar", "ts_01.mrc.tomostar",
                               "ts_02.mrc.tomostar"])
    blocks, key = core.read_star(str(src))
    plan = core.plan_split(blocks[key], "rlnMicrographName", "EXP", outdir="out")
    names = [item.combo[0] for item in plan]
    paths = [item.output_path for item in plan]
    rows = [item.n_rows for item in plan]
    assert names == ["ts_01.mrc.tomostar", "ts_02.mrc.tomostar"]   # first-appearance order
    assert paths[0] == str(Path("out") / "EXP_ts_01" / "EXP_ts_01_all.star")
    assert rows == [2, 1]


def test_plan_without_label_has_no_prefix(tmp_path):
    src = _write_single_block(tmp_path / "in.star", ["ts_01.mrc.tomostar"])
    blocks, key = core.read_star(str(src))
    plan = core.plan_split(blocks[key], "rlnMicrographName", None, outdir=".")
    import os
    assert plan[0].output_path == os.path.join(".", "ts_01", "ts_01_all.star")


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


def test_split_by_arbitrary_single_column(tmp_path):
    # rlnClassNumber is not a name column, but --group-by should accept it.
    df = pd.DataFrame({
        "rlnCoordinateX": [1.0, 2.0, 3.0, 4.0],
        "rlnMicrographName": ["a.mrc.tomostar"] * 4,
        "rlnClassNumber": [1, 2, 1, 2],
    })
    src = tmp_path / "in.star"
    starfile.write(df, str(src), overwrite=True)
    out = tmp_path / "out"
    runner = CliRunner()
    result = runner.invoke(split_star, ["--i", str(src), "--label", "C",
                                        "--group-by", "rlnClassNumber",
                                        "--outdir", str(out)])
    assert result.exit_code == 0, result.output
    assert len(starfile.read(str(out / "C_1" / "C_1_all.star"))) == 2
    assert len(starfile.read(str(out / "C_2" / "C_2_all.star"))) == 2


def test_split_by_multiple_columns_uses_combinations(tmp_path):
    df = pd.DataFrame({
        "rlnCoordinateX": [1.0, 2.0, 3.0, 4.0, 5.0],
        "rlnTomoName": ["A.mrc.tomostar", "A.mrc.tomostar", "A.mrc.tomostar",
                        "B.mrc.tomostar", "B.mrc.tomostar"],
        "rlnClassNumber": [1, 2, 1, 1, 2],
    })
    src = tmp_path / "in.star"
    starfile.write(df, str(src), overwrite=True)
    out = tmp_path / "out"
    runner = CliRunner()
    result = runner.invoke(split_star, [
        "--i", str(src), "--label", "X",
        "--group-by", "rlnTomoName", "--group-by", "rlnClassNumber",
        "--outdir", str(out),
    ])
    assert result.exit_code == 0, result.output
    # Combinations: A/1 (2 rows), A/2 (1), B/1 (1), B/2 (1). Name = <label>_<A>_<class>.
    assert len(starfile.read(str(out / "X_A_1" / "X_A_1_all.star"))) == 2
    assert len(starfile.read(str(out / "X_A_2" / "X_A_2_all.star"))) == 1
    assert len(starfile.read(str(out / "X_B_1" / "X_B_1_all.star"))) == 1
    assert len(starfile.read(str(out / "X_B_2" / "X_B_2_all.star"))) == 1
    # Every row landed exactly once.
    total = sum(len(starfile.read(str(p))) for p in out.rglob("*_all.star"))
    assert total == 5


def test_colliding_multi_column_names_are_rejected(tmp_path):
    # ("x", "y_z") and ("x_y", "z") both join to "x_y_z": must error, not overwrite.
    df = pd.DataFrame({
        "rlnCoordinateX": [1.0, 2.0],
        "colA": ["x", "x_y"],
        "colB": ["y_z", "z"],
    })
    src = tmp_path / "in.star"
    starfile.write(df, str(src), overwrite=True)
    out = tmp_path / "out"
    runner = CliRunner()
    result = runner.invoke(split_star, [
        "--i", str(src), "--group-by", "colA", "--group-by", "colB",
        "--outdir", str(out),
    ])
    assert result.exit_code != 0
    assert "collid" in result.output.lower()
    assert not out.exists()


def test_multiple_group_by_bad_column_is_a_clean_error(tmp_path):
    df = pd.DataFrame({"rlnCoordinateX": [1.0], "rlnTomoName": ["A.mrc.tomostar"]})
    src = tmp_path / "in.star"
    starfile.write(df, str(src), overwrite=True)
    runner = CliRunner()
    result = runner.invoke(split_star, [
        "--i", str(src), "--group-by", "rlnTomoName", "--group-by", "nope",
        "--outdir", str(tmp_path / "out"),
    ])
    assert result.exit_code != 0
    assert "nope" in result.output


def _write_numeric(path, values, column="rlnDistanceFromtop", micrograph="a.mrc.tomostar"):
    df = pd.DataFrame({
        "rlnCoordinateX": [float(i) for i in range(len(values))],
        "rlnMicrographName": [micrograph] * len(values),
        column: list(values),
    })
    starfile.write(df, str(path), overwrite=True)
    return path


# ---------------------------------------------------------------------------
# Range splitting.
# ---------------------------------------------------------------------------
def test_range_split_creates_half_open_bins(tmp_path):
    # values: 50, 100, 150, 250, 400 ; breaks 100,200,300
    src = _write_numeric(tmp_path / "in.star", [50, 100, 150, 250, 400])
    out = tmp_path / "out"
    runner = CliRunner()
    result = runner.invoke(split_star, [
        "--i", str(src), "--label", "D",
        "--range-by", "rlnDistanceFromtop", "--breaks", "100,200,300",
        "--outdir", str(out),
    ])
    assert result.exit_code == 0, result.output
    # 50 -> lt100 ; 100 & 150 -> 100-200 (100 is inclusive lower) ; 250 -> 200-300 ; 400 -> ge300
    assert len(starfile.read(str(out / "D_lt100" / "D_lt100_all.star"))) == 1
    assert len(starfile.read(str(out / "D_100-200" / "D_100-200_all.star"))) == 2
    assert len(starfile.read(str(out / "D_200-300" / "D_200-300_all.star"))) == 1
    assert len(starfile.read(str(out / "D_ge300" / "D_ge300_all.star"))) == 1
    # empty bins produce no file; every row lands once.
    total = sum(len(starfile.read(str(p))) for p in out.rglob("*_all.star"))
    assert total == 5


def test_range_value_on_breakpoint_goes_to_upper_bin(tmp_path):
    src = _write_numeric(tmp_path / "in.star", [200])   # exactly on the break
    out = tmp_path / "out"
    runner = CliRunner()
    result = runner.invoke(split_star, [
        "--i", str(src), "--label", "D", "--range-by", "rlnDistanceFromtop",
        "--breaks", "100,200,300", "--outdir", str(out),
    ])
    assert result.exit_code == 0, result.output
    assert (out / "D_200-300" / "D_200-300_all.star").exists()      # [200, 300)
    assert not (out / "D_100-200").exists()


def test_range_combines_with_group_by(tmp_path):
    df = pd.DataFrame({
        "rlnCoordinateX": [1.0, 2.0, 3.0],
        "rlnTomoName": ["A.mrc.tomostar", "A.mrc.tomostar", "B.mrc.tomostar"],
        "rlnDistanceFromtop": [50.0, 250.0, 50.0],
    })
    src = tmp_path / "in.star"
    starfile.write(df, str(src), overwrite=True)
    out = tmp_path / "out"
    runner = CliRunner()
    result = runner.invoke(split_star, [
        "--i", str(src), "--label", "X", "--group-by", "rlnTomoName",
        "--range-by", "rlnDistanceFromtop", "--breaks", "200", "--outdir", str(out),
    ])
    assert result.exit_code == 0, result.output
    assert (out / "X_A_lt200" / "X_A_lt200_all.star").exists()
    assert (out / "X_A_ge200" / "X_A_ge200_all.star").exists()
    assert (out / "X_B_lt200" / "X_B_lt200_all.star").exists()


def test_range_by_requires_breaks(tmp_path):
    src = _write_numeric(tmp_path / "in.star", [1, 2])
    runner = CliRunner()
    result = runner.invoke(split_star, ["--i", str(src), "--range-by",
                                        "rlnDistanceFromtop", "--outdir", str(tmp_path / "o")])
    assert result.exit_code != 0
    assert "breaks" in result.output.lower()


def test_range_by_non_numeric_column_is_an_error(tmp_path):
    src = _write_numeric(tmp_path / "in.star", [1, 2])
    runner = CliRunner()
    result = runner.invoke(split_star, [
        "--i", str(src), "--range-by", "rlnMicrographName", "--breaks", "1",
        "--outdir", str(tmp_path / "o"),
    ])
    assert result.exit_code != 0
    assert "numeric" in result.output.lower()


# ---------------------------------------------------------------------------
# Provenance comment header.
# ---------------------------------------------------------------------------
def test_output_has_provenance_comment(tmp_path):
    src = _write_numeric(tmp_path / "run_data.star", [50, 250])
    out = tmp_path / "out"
    runner = CliRunner()
    result = runner.invoke(split_star, [
        "--i", str(src), "--label", "D", "--range-by", "rlnDistanceFromtop",
        "--breaks", "200", "--outdir", str(out),
    ])
    assert result.exit_code == 0, result.output
    text = (out / "D_lt200" / "D_lt200_all.star").read_text()
    head = text.splitlines()[:5]
    assert any("Created by tomo_toolshed split-star" in l for l in head)
    assert any("source:" in l and "run_data.star" in l for l in head)
    assert any("split by:" in l and "range(rlnDistanceFromtop" in l for l in head)
    assert any("this file:" in l and "rlnDistanceFromtop < 200" in l for l in head)
    # The comment must not break re-reading.
    assert len(starfile.read(str(out / "D_lt200" / "D_lt200_all.star"))) == 1


def test_no_comment_flag_omits_header(tmp_path):
    src = _write_single_block(tmp_path / "in.star", ["a.mrc.tomostar"])
    out = tmp_path / "out"
    runner = CliRunner()
    result = runner.invoke(split_star, ["--i", str(src), "--label", "L",
                                        "--outdir", str(out), "--no-comment"])
    assert result.exit_code == 0, result.output
    text = (out / "L_a" / "L_a_all.star").read_text()
    assert "tomo_toolshed" not in text


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


@pytest.mark.parametrize("flavor_col,value,flavor,expected_dir", [
    # RELION 5 / M names end .tomostar; RELION 3 micrographs end .mrc.
    ("rlnTomoName", "Position_1.tomostar", "relion3", "Position_1"),
    ("rlnMicrographName", "ts_007.mrc", "relion3", "ts_007"),
])
def test_default_stripping_is_version_appropriate(flavor_col, value, flavor,
                                                   expected_dir, tmp_path):
    df = pd.DataFrame({"rlnCoordinateX": [1.0], flavor_col: [value]})
    src = tmp_path / "in.star"
    starfile.write(df, str(src), overwrite=True)
    out = tmp_path / "out"
    runner = CliRunner()
    result = runner.invoke(split_star, ["--i", str(src), "--outdir", str(out)])
    assert result.exit_code == 0, result.output
    assert (out / expected_dir / f"{expected_dir}_all.star").exists()


def test_warp_pixelsize_names_split_into_clean_dirs(tmp_path):
    src = _write_single_block(
        tmp_path / "in.star",
        ["Position_1.mrc_9.98Apx.mrc", "Position_1.mrc_9.98Apx.mrc",
         "Position_2.mrc_9.98Apx.mrc"])
    out = tmp_path / "out"
    runner = CliRunner()
    result = runner.invoke(split_star, ["--i", str(src), "--label", "L",
                                        "--outdir", str(out)])
    assert result.exit_code == 0, result.output
    assert (out / "L_Position_1" / "L_Position_1_all.star").exists()
    assert (out / "L_Position_2" / "L_Position_2_all.star").exists()


def test_mixed_suffixes_within_one_file_are_each_cleaned(tmp_path):
    # Group names in a single file need not share a suffix; each is stripped on
    # its own (longest match / pattern), independent of flavor.
    src = _write_single_block(tmp_path / "in.star", [
        "ts_001.mrc.tomostar",           # -> ts_001
        "Position_1.tomostar",           # -> Position_1
        "foo.mrc",                       # -> foo
        "Position_2.mrc_9.98Apx.mrc",    # -> Position_2 (pattern beats .mrc)
        "plain_name",                    # -> plain_name (nothing to strip)
    ])
    out = tmp_path / "out"
    runner = CliRunner()
    result = runner.invoke(split_star, ["--i", str(src), "--label", "L",
                                        "--outdir", str(out)])
    assert result.exit_code == 0, result.output
    for base in ("L_ts_001", "L_Position_1", "L_foo", "L_Position_2", "L_plain_name"):
        assert (out / base / f"{base}_all.star").exists(), base


def test_flavor_is_detected_and_reported():
    # RELION 5 sample lives with the duplicate-remover fixtures.
    r5 = Path(__file__).parent.parent / "duplicate_remover" / "relion_5_2D_example.star"
    if not r5.exists():
        pytest.skip("relion_5 sample not available")
    blocks, key = core.read_star(str(r5))
    assert core.detect_flavor(blocks, blocks[key]) == core.FLAVOR_RELION5
