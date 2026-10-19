"""Tests for Warp 1 support (UTF-16/CRLF XML, multi-table taSolution.log) and
--mdoc-dir dose override, using the real data in warp1_test/."""
import os
import shutil

import pytest
from click.testing import CliRunner

from tomo_toolshed.skipped_views import core as rsv
from tomo_toolshed.skipped_views.cli import skipped_views

HERE = os.path.dirname(os.path.abspath(__file__))
W1 = os.path.join(HERE, "warp1_test")
TS2 = "NAD069_2_lam11_ts_002.mrc"
TS3 = "NAD069_2_lam11_ts_003.mrc"  # log has 5 tables (global + local areas)


@pytest.fixture
def w1(tmp_path):
    d = tmp_path / "w1"
    shutil.copytree(W1, d)
    return d


def _write_tomostar(root, path, crlf=True):
    n = rsv.get_n_tilts(root)
    bns = rsv.movie_basenames(root, n)
    angles = rsv.split_lines(root.find("Angles").text)
    doses = rsv.split_lines(root.find("Dose").text)
    lines = ["", "data_", "", "loop_", "_wrpMovieName #1", "_wrpAngleTilt #2",
             "_wrpAxisAngle #3", "_wrpDose #4"]
    lines += [f"../frames/{b}  {a}  -91.393  {d}" for b, a, d in zip(bns, angles, doses)]
    lines.append("")
    with open(path, "w", newline="") as fh:
        fh.write(("\r\n" if crlf else "\n").join(lines))


# --------------------------------------------------------------------------- #
# Warp 1 XML I/O
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("ts", [TS2, TS3])
def test_warp1_roundtrip_byte_identical(ts):
    path = os.path.join(W1, ts + ".xml")
    orig = open(path, "rb").read()
    assert orig.startswith(b"\xff\xfe")  # UTF-16 LE BOM
    assert rsv.serialize_xml(rsv.parse_xml(path)) == orig


def test_warp1_parse_values():
    root = rsv.parse_xml(os.path.join(W1, TS2 + ".xml")).getroot()
    assert rsv.get_n_tilts(root) == 37
    assert len(rsv.parse_dose(root)) == 37
    assert rsv.validate_structure(root, 37) == []


# --------------------------------------------------------------------------- #
# taSolution.log
# --------------------------------------------------------------------------- #
def test_multi_table_log_uses_global_table_only():
    views, tilts = rsv.read_log(os.path.join(W1, "imod", TS3, "taSolution.log"))
    assert views == set(range(9, 30))
    assert all(isinstance(v, int) for v in views)
    assert tilts[9] == -25.3 and tilts[29] == 34.4  # first table's values


def test_single_table_log():
    views, tilts = rsv.read_log(os.path.join(W1, "imod", TS2, "taSolution.log"))
    assert views == set(range(6, 32))
    assert tilts[6] == -42.6


def test_multi_table_flip_keeps_logged_views():
    root = rsv.parse_xml(os.path.join(W1, TS3 + ".xml")).getroot()
    views, tilts = rsv.read_log(os.path.join(W1, "imod", TS3, "taSolution.log"))
    mask = rsv.compute_keep_mask(root, views, tilts)
    assert sum(mask) == 21
    assert [i + 1 for i, k in enumerate(mask) if k] == list(range(9, 30))


def test_max_tilt_uses_view_numbers_not_list_position():
    """Log starts at view 6: tilt lookup must be by view, not list index."""
    root = rsv.parse_xml(os.path.join(W1, TS2 + ".xml")).getroot()
    views, tilts = rsv.read_log(os.path.join(W1, "imod", TS2, "taSolution.log"))
    doses = rsv.mdoc_doses_for_xml(root, os.path.join(W1, "PACE", TS2 + ".mdoc"))
    root.find("Dose").text = "\n".join(rsv.format_dose(d) for d in doses)
    mask = rsv.compute_keep_mask(root, views, tilts, max_tilt=10)
    min_view = doses.index(0.0) + 1
    thr = abs(tilts[min_view]) + 10
    expected = [(i + 1) in views and abs(tilts[i + 1]) <= thr for i in range(37)]
    assert mask == expected and any(mask)


# --------------------------------------------------------------------------- #
# mdoc doses
# --------------------------------------------------------------------------- #
def test_mdoc_doses_accumulate_in_acquisition_order():
    root = rsv.parse_xml(os.path.join(W1, TS2 + ".xml")).getroot()
    doses = rsv.mdoc_doses_for_xml(root, os.path.join(W1, "PACE", TS2 + ".mdoc"))
    bns = rsv.movie_basenames(root, 37)
    by_bn = dict(zip(bns, doses))
    # First acquired tilt (9 deg, pre-tilt start) gets 0; next one ExposureDose.
    assert by_bn["NAD069_2_lam11_ts_002_001_0001_9.0_11.15.47.tif"] == 0
    assert by_bn["NAD069_2_lam11_ts_002_001_0001_12.0_11.16.31.tif"] == pytest.approx(2.84828)
    assert sorted(doses) == pytest.approx([i * 2.84828 for i in range(37)])


def test_find_mdoc(tmp_path):
    (tmp_path / "TS_1.mdoc").write_text("")
    assert rsv.find_mdoc(str(tmp_path), "x/TS_1.xml").endswith("TS_1.mdoc")
    (tmp_path / "TS_2.mrc.mdoc").write_text("")
    assert rsv.find_mdoc(str(tmp_path), "TS_2.mrc.xml").endswith("TS_2.mrc.mdoc")
    assert rsv.find_mdoc(str(tmp_path), "TS_2.xml").endswith("TS_2.mrc.mdoc")
    assert rsv.find_mdoc(str(tmp_path), "TS_9.xml") is None


def test_flip_with_mdoc_overrides_dose_and_warns(w1):
    res = CliRunner().invoke(skipped_views, [
        "--xml-dir", str(w1), "--xml-pattern", TS2 + ".xml",
        "--tiltstack-dir", str(w1 / "imod"), "--backup-dir", str(w1 / "bak"),
        "--mdoc-dir", str(w1 / "PACE")])
    assert res.exit_code == 0, res.output
    assert "WARNING: XML doses differ from mdoc for 37/37 tilts" in res.output
    xml = str(w1 / (TS2 + ".xml"))
    root = rsv.parse_xml(xml).getroot()
    expected = rsv.mdoc_doses_for_xml(root, str(w1 / "PACE" / (TS2 + ".mdoc")))
    assert rsv.parse_dose(root) == pytest.approx(expected, abs=1e-4)
    assert rsv.split_lines(root.find("UseTilt").text).count("True") == 26
    assert open(xml, "rb").read().startswith(b"\xff\xfe")  # still UTF-16


def test_no_warning_when_doses_match(w1):
    xml = str(w1 / (TS2 + ".xml"))
    tree = rsv.parse_xml(xml)
    doses = rsv.mdoc_doses_for_xml(tree.getroot(), str(w1 / "PACE" / (TS2 + ".mdoc")))
    tree.getroot().find("Dose").text = "\n".join(rsv.format_dose(d) for d in doses)
    open(xml, "wb").write(rsv.serialize_xml(tree))
    res = CliRunner().invoke(skipped_views, [
        "--xml-dir", str(w1), "--xml-pattern", TS2 + ".xml",
        "--tiltstack-dir", str(w1 / "imod"), "--mdoc-dir", str(w1 / "PACE"),
        "--dry-run"])
    assert res.exit_code == 0 and "WARNING" not in res.output, res.output


def test_delete_with_mdoc_updates_tomostar_and_xml(w1):
    xml = str(w1 / (TS3 + ".xml"))
    os.makedirs(w1 / "tomostar")
    tomo = str(w1 / "tomostar" / (TS3 + ".tomostar"))
    _write_tomostar(rsv.parse_xml(xml).getroot(), tomo, crlf=True)
    args = ["--xml-dir", str(w1), "--xml-pattern", TS3 + ".xml",
            "--tiltstack-dir", str(w1 / "imod"), "--tomostar-dir", str(w1 / "tomostar"),
            "--mdoc-dir", str(w1 / "PACE"), "--delete"]

    dry = CliRunner().invoke(skipped_views, args + ["--dry-run"])
    assert dry.exit_code == 0, dry.output
    assert "would delete 16 of 37 tilts, 21 remain" in dry.output
    assert "tomostar doses differ from mdoc" in dry.output

    res = CliRunner().invoke(skipped_views, args)
    assert res.exit_code == 0, res.output
    root = rsv.parse_xml(xml).getroot()
    assert rsv.get_n_tilts(root) == 21
    rsv.assert_after_delete(root, 21)

    raw = open(tomo, "rb").read().decode()
    assert "\r\n" in raw  # CRLF preserved
    lines, cols, mcol, data_idx = rsv.parse_tomostar_text(raw)
    assert len(data_idx) == 21
    dcol = cols.index("_wrpDose")
    tomo_doses = {os.path.basename(lines[i].split()[mcol]): float(lines[i].split()[dcol])
                  for i in data_idx}
    xml_doses = dict(zip(rsv.movie_basenames(root, 21), rsv.parse_dose(root)))
    assert tomo_doses == pytest.approx(xml_doses)
    by_stem = rsv.read_mdoc_doses(str(w1 / "PACE" / (TS3 + ".mdoc")))
    for bn, d in xml_doses.items():
        assert d == pytest.approx(by_stem[os.path.splitext(bn)[0]], abs=1e-4)


def test_missing_mdoc_skips_file(w1, tmp_path):
    empty = tmp_path / "nomdocs"
    empty.mkdir()
    xml = str(w1 / (TS2 + ".xml"))
    before = open(xml, "rb").read()
    res = CliRunner().invoke(skipped_views, [
        "--xml-dir", str(w1), "--xml-pattern", TS2 + ".xml",
        "--tiltstack-dir", str(w1 / "imod"), "--mdoc-dir", str(empty)])
    assert res.exit_code == 0
    assert "mdoc for" in res.output and "not found" in res.output
    assert open(xml, "rb").read() == before
