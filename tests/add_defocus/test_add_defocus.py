from pathlib import Path

import pytest
from click.testing import CliRunner

from tomo_toolshed.add_defocus import core
from tomo_toolshed.add_defocus.cli import add_defocus

#: A real WarpTools tilt-series XML that ships with the skipped-views tests.
REAL_XML = Path(__file__).parents[1] / "skipped_views" / "Position_2_2.xml"

STAR_TEMPLATE = """
data_

loop_
_rlnMicrographName #1
_rlnTomoName #2
_rlnPixelSize #3
_rlnDefocus #4
_rlnNumberSubtomo #5
{rows}
"""


def write_xml(path, values):
    nodes = "\n".join(
        f'    <Node X="0" Y="0" Z="{i}" Value="{v}" />'
        for i, v in enumerate(values)
    )
    path.write_text(
        '<?xml version="1.0" encoding="utf-8"?>\n'
        "<TiltSeries>\n"
        f'  <GridCTF Width="1" Height="1" Depth="{len(values)}">\n'
        f"{nodes}\n"
        "  </GridCTF>\n"
        "</TiltSeries>\n"
    )


def make_star(path, tomonames, separator="\t"):
    rows = "\n".join(
        separator.join([f"{n}.mrc", f"rec/{n}.mrc_10.00Apx.mrc", "10.0", "0.0", "100"])
        for n in tomonames
    )
    path.write_text(STAR_TEMPLATE.format(rows=rows))
    return path


def defocus_column(path, index=3):
    values = []
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("rec/") or ".mrc" in stripped and not stripped.startswith("_"):
            cols = stripped.split()
            if len(cols) > index:
                values.append(cols[index])
    return values


# --- defocus reading -------------------------------------------------------

def test_defocus_values_converts_microns_to_angstroms(tmp_path):
    xml = tmp_path / "ts.mrc.xml"
    write_xml(xml, ["4.0", "5.0"])
    assert list(core.defocus_values(xml)) == [40000.0, 50000.0]
    assert core.average_defocus(xml) == 45000.0


def test_average_defocus_on_real_warp_xml():
    mean = core.average_defocus(REAL_XML)
    # Values in this series sit around 4.2-4.4 microns.
    assert 40000 < mean < 46000


def test_defocus_values_rejects_xml_without_gridctf(tmp_path):
    xml = tmp_path / "empty.mrc.xml"
    xml.write_text("<TiltSeries></TiltSeries>")
    with pytest.raises(ValueError, match="GridCTF"):
        core.defocus_values(xml)


# --- star file plumbing ----------------------------------------------------

def test_parse_star_header():
    lines = STAR_TEMPLATE.format(rows="a\tb\tc\td\te").splitlines(keepends=True)
    columns, header_end = core.parse_star_header(lines)
    assert columns == [
        "_rlnMicrographName", "_rlnTomoName", "_rlnPixelSize",
        "_rlnDefocus", "_rlnNumberSubtomo",
    ]
    assert lines[header_end].startswith("a")


def test_xml_path_for():
    assert core.xml_path_for("rec/Position_2_2.mrc_10.00Apx.mrc", "xml") == \
        "xml/Position_2_2.mrc.xml"
    assert core.xml_path_for("Position_2_2.mrc_6.5Apx.mrc", "xml") == \
        "xml/Position_2_2.mrc.xml"
    assert core.xml_path_for("something_else.mrc", "xml") is None


def test_update_star_defocus_in_place(tmp_path):
    xml_dir = tmp_path / "xml"
    xml_dir.mkdir()
    write_xml(xml_dir / "ts1.mrc.xml", ["4.0", "6.0"])
    write_xml(xml_dir / "ts2.mrc.xml", ["2.0", "2.0"])
    star = make_star(tmp_path / "tomos.star", ["ts1", "ts2"])

    updated, skipped = core.update_star_defocus(star, str(xml_dir))

    assert skipped == []
    assert [d for _, d in updated] == [50000, 20000]
    assert defocus_column(star) == ["50000", "20000"]


def test_update_star_defocus_writes_to_output_and_leaves_input(tmp_path):
    xml_dir = tmp_path / "xml"
    xml_dir.mkdir()
    write_xml(xml_dir / "ts1.mrc.xml", ["4.0"])
    star = make_star(tmp_path / "tomos.star", ["ts1"])
    out = tmp_path / "out.star"

    core.update_star_defocus(star, str(xml_dir), output=str(out))

    assert defocus_column(out) == ["40000"]
    assert defocus_column(star) == ["0.0"]


def test_dry_run_writes_nothing(tmp_path):
    xml_dir = tmp_path / "xml"
    xml_dir.mkdir()
    write_xml(xml_dir / "ts1.mrc.xml", ["4.0"])
    star = make_star(tmp_path / "tomos.star", ["ts1"])

    updated, _ = core.update_star_defocus(star, str(xml_dir), dry_run=True)

    assert [d for _, d in updated] == [40000]
    assert defocus_column(star) == ["0.0"]


def test_space_separated_star_stays_space_separated(tmp_path):
    xml_dir = tmp_path / "xml"
    xml_dir.mkdir()
    write_xml(xml_dir / "ts1.mrc.xml", ["4.0"])
    star = make_star(tmp_path / "tomos.star", ["ts1"], separator=" ")

    core.update_star_defocus(star, str(xml_dir))

    data = [l for l in star.read_text().splitlines() if l.startswith("ts1")]
    assert "\t" not in data[0]
    assert defocus_column(star) == ["40000"]


# --- skip vs strict --------------------------------------------------------

def test_missing_xml_is_skipped_by_default(tmp_path):
    xml_dir = tmp_path / "xml"
    xml_dir.mkdir()
    write_xml(xml_dir / "ts1.mrc.xml", ["4.0"])
    star = make_star(tmp_path / "tomos.star", ["ts1", "ts_absent"])

    updated, skipped = core.update_star_defocus(star, str(xml_dir))

    assert [d for _, d in updated] == [40000]
    assert len(skipped) == 1 and "missing XML" in skipped[0][1]
    assert defocus_column(star) == ["40000", "0.0"]


def test_strict_raises_and_writes_nothing(tmp_path):
    xml_dir = tmp_path / "xml"
    xml_dir.mkdir()
    write_xml(xml_dir / "ts1.mrc.xml", ["4.0"])
    star = make_star(tmp_path / "tomos.star", ["ts1", "ts_absent"])

    with pytest.raises(ValueError, match="missing XML"):
        core.update_star_defocus(star, str(xml_dir), strict=True)

    assert defocus_column(star) == ["0.0", "0.0"]


def test_star_without_defocus_column_raises(tmp_path):
    star = tmp_path / "bad.star"
    star.write_text("data_\n\nloop_\n_rlnTomoName #1\nrec/ts1.mrc_10.00Apx.mrc\n")
    with pytest.raises(core.StarFormatError, match="_rlnDefocus"):
        core.update_star_defocus(star, str(tmp_path))


# --- CLI -------------------------------------------------------------------

def test_cli_updates_star(tmp_path):
    xml_dir = tmp_path / "xml"
    xml_dir.mkdir()
    write_xml(xml_dir / "ts1.mrc.xml", ["4.0", "6.0"])
    star = make_star(tmp_path / "tomos.star", ["ts1"])

    result = CliRunner().invoke(
        add_defocus, ["--star", str(star), "--xml-dir", str(xml_dir)]
    )

    assert result.exit_code == 0, result.output
    assert "updated 1 row" in result.output
    assert defocus_column(star) == ["50000"]


def test_cli_strict_exits_nonzero(tmp_path):
    xml_dir = tmp_path / "xml"
    xml_dir.mkdir()
    star = make_star(tmp_path / "tomos.star", ["ts_absent"])

    result = CliRunner().invoke(
        add_defocus,
        ["--star", str(star), "--xml-dir", str(xml_dir), "--strict"],
    )

    assert result.exit_code != 0
    assert defocus_column(star) == ["0.0"]
