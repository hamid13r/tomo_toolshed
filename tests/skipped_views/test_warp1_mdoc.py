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
IMOD = os.path.join(W1, "frames", "imod")  # etomo runs: imod/<series>/taSolution.log
TS2 = "NAD069_2_lam11_ts_002.mrc"
TS3 = "NAD069_2_lam11_ts_003.mrc"  # log has 5 tables (global + local areas)


@pytest.fixture
def w1(tmp_path):
    """Slim working copy: the two NAD069_2 XMLs, their mdocs and logs."""
    d = tmp_path / "w1"
    (d / "PACE").mkdir(parents=True)
    for ts in (TS2, TS3):
        shutil.copy(os.path.join(W1, ts + ".xml"), d)
        shutil.copy(os.path.join(W1, "PACE", ts + ".mdoc"), d / "PACE")
        shutil.copytree(os.path.join(IMOD, ts), d / "imod" / ts)
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
    views, tilts = rsv.read_log(os.path.join(IMOD, TS3, "taSolution.log"))
    assert views == set(range(9, 30))
    assert all(isinstance(v, int) for v in views)
    assert tilts[9] == -25.3 and tilts[29] == 34.4  # first table's values


def test_single_table_log():
    views, tilts = rsv.read_log(os.path.join(IMOD, TS2, "taSolution.log"))
    assert views == set(range(6, 32))
    assert tilts[6] == -42.6


def test_multi_table_flip_keeps_logged_views():
    root = rsv.parse_xml(os.path.join(W1, TS3 + ".xml")).getroot()
    views, tilts = rsv.read_log(os.path.join(IMOD, TS3, "taSolution.log"))
    mask = rsv.compute_keep_mask(root, views, tilts)
    assert sum(mask) == 21
    assert [i + 1 for i, k in enumerate(mask) if k] == list(range(9, 30))


def test_max_tilt_uses_view_numbers_not_list_position():
    """Log starts at view 6: tilt lookup must be by view, not list index."""
    root = rsv.parse_xml(os.path.join(W1, TS2 + ".xml")).getroot()
    views, tilts = rsv.read_log(os.path.join(IMOD, TS2, "taSolution.log"))
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


# --------------------------------------------------------------------------- #
# --angles-from-log
# --------------------------------------------------------------------------- #
def _tomostar_col(path, col):
    raw = open(path, "rb").read().decode()
    lines, cols, mcol, data_idx = rsv.parse_tomostar_text(raw)
    c = cols.index(col)
    return {os.path.basename(lines[i].split()[mcol]): float(lines[i].split()[c])
            for i in data_idx}


@pytest.mark.parametrize("ts", [TS2, TS3])
def test_log_angles_for_xml(ts):
    root = rsv.parse_xml(os.path.join(W1, ts + ".xml")).getroot()
    views, tilts = rsv.read_log(os.path.join(IMOD, ts, "taSolution.log"))
    angles = rsv.log_angles_for_xml(root, tilts)
    for i, a in enumerate(angles):
        if (i + 1) in views:
            assert a == tilts[i + 1]
        else:
            assert a is None  # not in the log -> left unchanged


def test_log_angles_sign_matched_to_xml():
    root = rsv.parse_xml(os.path.join(W1, TS2 + ".xml")).getroot()
    views, tilts = rsv.read_log(os.path.join(IMOD, TS2, "taSolution.log"))
    flipped = {v: -t for v, t in tilts.items()}
    assert rsv.log_angles_for_xml(root, flipped) == rsv.log_angles_for_xml(root, tilts)


def test_log_angles_override_even_when_they_do_not_fit():
    """A log that does not follow the XML angles still overrides them (as-is,
    etomo's sign); strict=True keeps the old refusal."""
    root = rsv.parse_xml(os.path.join(W1, TS2 + ".xml")).getroot()
    views, tilts = rsv.read_log(os.path.join(IMOD, TS2, "taSolution.log"))
    vals = list(tilts.values())
    scrambled = dict(zip(tilts, vals[::2] + vals[1::2]))
    assert rsv.log_angle_fit(root, scrambled) < rsv.MIN_ANGLE_CORR
    angles = rsv.log_angles_for_xml(root, scrambled)
    assert [a for a in angles if a is not None] == [scrambled[v] for v in sorted(scrambled)]
    with pytest.raises(ValueError):
        rsv.log_angles_for_xml(root, scrambled, strict=True)


def test_poor_fit_series_overridden_with_warning(nad3):
    """lam29_ts_001: failed etomo alignment, log/XML corr -0.70. The log
    values are written as they are (no sign flip) and a warning is printed."""
    st = "NAD069_3_lam29_ts_001.mrc"
    res = CliRunner().invoke(skipped_views, _nad3_args(
        nad3, "--xml-pattern", st + ".xml", "--delete"))
    assert res.exit_code == 0, res.output
    assert "do not follow the XML angles (corr -0.70)" in res.output
    assert "overriding with the log values anyway" in res.output
    views, tilts = rsv.read_log(os.path.join(IMOD, st, "taSolution.log"))
    root = rsv.parse_xml(str(nad3 / (st + ".xml"))).getroot()
    angles = [float(a) for a in rsv.split_lines(root.find("Angles").text)]
    assert angles == [tilts[v] for v in sorted(views)]
    cols, rows, mcol = _rows(str(nad3 / (st + ".tomostar")))
    assert [float(r[cols.index("_wrpAngleTilt")]) for r in rows] == angles
    rows_tbl, _ = _table(res.output)
    assert rows_tbl[st][8] == "OK" and "corr" in " ".join(rows_tbl[st][9:])


@pytest.mark.parametrize("delete", [False, True])
def test_angles_from_log_xml_and_tomostar_match_log(w1, delete):
    xml = str(w1 / (TS3 + ".xml"))
    os.makedirs(w1 / "tomostar")
    tomo = str(w1 / "tomostar" / (TS3 + ".tomostar"))
    _write_tomostar(rsv.parse_xml(xml).getroot(), tomo)
    old_angles = rsv.split_lines(rsv.parse_xml(xml).getroot().find("Angles").text)
    args = ["--xml-dir", str(w1), "--xml-pattern", TS3 + ".xml",
            "--tiltstack-dir", str(w1 / "imod"), "--tomostar-dir", str(w1 / "tomostar"),
            "--backup-dir", str(w1 / "bak"), "--angles-from-log"]
    if delete:
        args.append("--delete")
    res = CliRunner().invoke(skipped_views, args)
    assert res.exit_code == 0, res.output
    assert "XML tilt angles differ from taSolution.log for 21/37" in res.output
    assert "check the etomo alignment" in res.output
    assert os.path.exists(tomo + ".bak")

    views, tilts = rsv.read_log(str(w1 / "imod" / TS3 / "taSolution.log"))
    orig_bns = rsv.movie_basenames(rsv.parse_xml(
        str(w1 / "bak" / (TS3 + ".xml")) if not delete else xml + ".bak").getroot(), 37)
    log_by_bn = {orig_bns[v - 1]: t for v, t in tilts.items()}

    root = rsv.parse_xml(xml).getroot()
    n = rsv.get_n_tilts(root)
    assert n == (21 if delete else 37)
    xml_by_bn = dict(zip(rsv.movie_basenames(root, n),
                         [float(a) for a in rsv.split_lines(root.find("Angles").text)]))
    tomo_by_bn = _tomostar_col(tomo, "_wrpAngleTilt")
    assert tomo_by_bn == xml_by_bn
    for bn, t in log_by_bn.items():
        assert xml_by_bn[bn] == t
    if not delete:  # views not in the log keep their original text
        new = rsv.split_lines(root.find("Angles").text)
        assert [new[i] for i in range(37) if (i + 1) not in views] == \
               [old_angles[i] for i in range(37) if (i + 1) not in views]


def test_angles_from_log_is_idempotent(w1):
    args = ["--xml-dir", str(w1), "--tiltstack-dir", str(w1 / "imod"),
            "--backup-dir", str(w1 / "bak"), "--tomostar-dir", str(w1 / "none"),
            "--angles-from-log"]
    assert CliRunner().invoke(skipped_views, args).exit_code == 0
    res = CliRunner().invoke(skipped_views, args + ["--dry-run"])
    assert res.exit_code == 0 and "differ" not in res.output, res.output
    assert "not found; only the XML is updated" in res.output


def test_flip_with_mdoc_also_updates_tomostar(w1):
    xml = str(w1 / (TS2 + ".xml"))
    os.makedirs(w1 / "tomostar")
    tomo = str(w1 / "tomostar" / (TS2 + ".tomostar"))
    _write_tomostar(rsv.parse_xml(xml).getroot(), tomo)
    res = CliRunner().invoke(skipped_views, [
        "--xml-dir", str(w1), "--xml-pattern", TS2 + ".xml",
        "--tiltstack-dir", str(w1 / "imod"), "--backup-dir", str(w1 / "bak"),
        "--tomostar-dir", str(w1 / "tomostar"), "--mdoc-dir", str(w1 / "PACE")])
    assert res.exit_code == 0, res.output
    root = rsv.parse_xml(xml).getroot()
    assert _tomostar_col(tomo, "_wrpDose") == pytest.approx(
        dict(zip(rsv.movie_basenames(root, 37), rsv.parse_dose(root))))


# --------------------------------------------------------------------------- #
# Re-running after --delete (view numbers must keep referring to the original)
# --------------------------------------------------------------------------- #
def _delete_args(w1, ts, *extra):
    return ["--xml-dir", str(w1), "--xml-pattern", ts + ".xml",
            "--tiltstack-dir", str(w1 / "imod"),
            "--tomostar-dir", str(w1 / "tomostar"), "--delete", *extra]


def _kept(xml):
    root = rsv.parse_xml(xml).getroot()
    return rsv.movie_basenames(root, rsv.get_n_tilts(root))


@pytest.mark.parametrize("ts", [TS2, TS3])
def test_second_delete_removes_nothing(w1, ts):
    """Regression: a 2nd --delete used to strip more tilts off the START only
    (log views no longer lined up with the trimmed XML)."""
    xml = str(w1 / (ts + ".xml"))
    os.makedirs(w1 / "tomostar")
    _write_tomostar(rsv.parse_xml(xml).getroot(), str(w1 / "tomostar" / (ts + ".tomostar")))
    orig = _kept(xml)
    views, _ = rsv.read_log(str(w1 / "imod" / ts / "taSolution.log"))
    expected = [orig[v - 1] for v in sorted(views)]

    r1 = CliRunner().invoke(skipped_views, _delete_args(w1, ts))
    assert r1.exit_code == 0, r1.output
    assert _kept(xml) == expected  # both ends trimmed

    r2 = CliRunner().invoke(skipped_views, _delete_args(w1, ts))
    assert r2.exit_code == 0, r2.output
    assert "already removed" in r2.output and "nothing to delete" in r2.output
    assert _kept(xml) == expected


def test_selection_after_delete_matches_single_run(w1):
    """--max-tilt after a plain --delete == --max-tilt on the untouched XML."""
    xml = str(w1 / (TS2 + ".xml"))
    os.makedirs(w1 / "tomostar")
    tomo = str(w1 / "tomostar" / (TS2 + ".tomostar"))
    _write_tomostar(rsv.parse_xml(xml).getroot(), tomo)
    shutil.copy(xml, str(w1 / "orig.xml"))
    shutil.copy(tomo, str(w1 / "orig.tomostar"))

    assert CliRunner().invoke(skipped_views, _delete_args(
        w1, TS2, "--max-tilt", "20")).exit_code == 0
    one_step = _kept(xml)

    shutil.copy(str(w1 / "orig.xml"), xml)
    shutil.copy(str(w1 / "orig.tomostar"), tomo)
    for f in os.listdir(w1):
        if ".bak" in f:
            os.remove(w1 / f)
    for f in os.listdir(w1 / "tomostar"):
        if ".bak" in f:
            os.remove(w1 / "tomostar" / f)
    assert CliRunner().invoke(skipped_views, _delete_args(w1, TS2)).exit_code == 0
    res = CliRunner().invoke(skipped_views, _delete_args(w1, TS2, "--max-tilt", "20"))
    assert res.exit_code == 0, res.output
    assert _kept(xml) == one_step
    assert os.path.exists(xml + ".bak.1")  # original .bak kept intact


def test_flip_after_delete_keeps_all_remaining(w1):
    xml = str(w1 / (TS3 + ".xml"))
    os.makedirs(w1 / "tomostar")
    _write_tomostar(rsv.parse_xml(xml).getroot(), str(w1 / "tomostar" / (TS3 + ".tomostar")))
    assert CliRunner().invoke(skipped_views, _delete_args(w1, TS3)).exit_code == 0
    res = CliRunner().invoke(skipped_views, [
        "--xml-dir", str(w1), "--xml-pattern", TS3 + ".xml",
        "--tiltstack-dir", str(w1 / "imod"), "--backup-dir", str(w1 / "bak")])
    assert res.exit_code == 0, res.output
    ut = rsv.split_lines(rsv.parse_xml(xml).getroot().find("UseTilt").text)
    assert ut == ["True"] * 21


def test_stale_bak_with_foreign_tilts_aborts(w1):
    xml = str(w1 / (TS2 + ".xml"))
    shutil.copy(str(w1 / (TS3 + ".xml")), xml + ".bak")  # wrong series
    before = open(xml, "rb").read()
    res = CliRunner().invoke(skipped_views, [
        "--xml-dir", str(w1), "--xml-pattern", TS2 + ".xml",
        "--tiltstack-dir", str(w1 / "imod"), "--backup-dir", str(w1 / "bak")])
    assert res.exit_code == 0
    assert "ABORT" in res.output and "stale .bak" in res.output
    assert open(xml, "rb").read() == before


# --------------------------------------------------------------------------- #
# End-to-end on the NAD069_3 set: 22 Warp 1 series with their original
# XML + tomostar (backup_tomostar_xml/), etomo logs (frames/imod) and mdocs.
# --------------------------------------------------------------------------- #
ORIG = os.path.join(W1, "backup_tomostar_xml")
# No taSolution.log: untouched. (Series whose log angles do not fit the XML,
# e.g. failed etomo alignments, are now overridden with the log values.)
UNTOUCHED = {"NAD069_3_lam23_ts_001.mrc"}


def _series():
    return sorted(f[:-4] for f in os.listdir(ORIG) if f.endswith(".xml"))


@pytest.fixture
def nad3(tmp_path):
    d = tmp_path / "frames"
    d.mkdir()
    for f in os.listdir(ORIG):
        shutil.copy(os.path.join(ORIG, f), d)
    shutil.copytree(IMOD, d / "imod")
    shutil.copytree(os.path.join(W1, "PACE"), tmp_path / "PACE")
    return d


def _rows(path):
    raw = open(path, "rb").read().decode()
    lines, cols, mcol, data_idx = rsv.parse_tomostar_text(raw)
    return cols, [lines[i].split() for i in data_idx], mcol


@pytest.mark.parametrize("delete", [False, True])
def test_nad3_tomostars_match_xml(nad3, delete):
    args = ["--xml-dir", str(nad3), "--tiltstack-dir", str(nad3 / "imod"),
            "--tomostar-dir", str(nad3), "--backup-dir", str(nad3 / "bak"),
            "--mdoc-dir", str(nad3.parent / "PACE"), "--angles-from-log"]
    if delete:
        args.append("--delete")
    res = CliRunner().invoke(skipped_views, args)
    assert res.exit_code == 0, res.output

    for st in _series():
        xml, tomo = str(nad3 / (st + ".xml")), str(nad3 / (st + ".tomostar"))
        orig_xml = os.path.join(ORIG, st + ".xml")
        if st in UNTOUCHED:
            assert open(xml, "rb").read() == open(orig_xml, "rb").read(), st
            assert open(tomo, "rb").read() == \
                open(os.path.join(ORIG, st + ".tomostar"), "rb").read(), st
            continue

        root0 = rsv.parse_xml(orig_xml).getroot()
        names0 = rsv.movie_basenames(root0, rsv.get_n_tilts(root0))
        views, tilts = rsv.read_log(os.path.join(IMOD, st, "taSolution.log"))
        doses = dict(zip(names0, rsv.mdoc_doses_for_xml(
            root0, os.path.join(W1, "PACE", st + ".mdoc"))))
        oc, orows, om = _rows(os.path.join(ORIG, st + ".tomostar"))
        orig_row = {r[om]: r for r in orows}

        root = rsv.parse_xml(xml).getroot()
        n = rsv.get_n_tilts(root)
        names = rsv.movie_basenames(root, n)
        angles = [float(a) for a in rsv.split_lines(root.find("Angles").text)]
        cols, rows, mcol = _rows(tomo)

        # The tomostar holds exactly the XML's tilts, in the same order.
        assert [r[mcol] for r in rows] == names, st
        if delete:
            assert names == [names0[v - 1] for v in sorted(views)], st
        else:
            assert n == len(names0), st
            ut = rsv.split_lines(root.find("UseTilt").text)
            assert [u == "True" for u in ut] == \
                [(i + 1) in views for i in range(n)], st

        ai, di = cols.index("_wrpAngleTilt"), cols.index("_wrpDose")
        for name, a, r in zip(names, angles, rows):
            view = names0.index(name) + 1
            want = tilts[view] if view in tilts else float(orig_row[name][ai])
            assert float(r[ai]) == pytest.approx(want, abs=1e-6), (st, view)
            assert a == pytest.approx(want, abs=1e-6), (st, view)
            assert float(r[di]) == pytest.approx(doses[name], abs=1e-4), (st, view)
            others = [k for k in range(len(cols)) if k not in (ai, di)]
            assert [r[k] for k in others] == [orig_row[name][k] for k in others], st


# --------------------------------------------------------------------------- #
# Diagnosis table
# --------------------------------------------------------------------------- #
def test_view_ranges():
    assert rsv.view_ranges([1, 2, 3, 5, 7, 8]) == "1-3,5,7-8"
    assert rsv.view_ranges([]) == "-"
    assert rsv.view_ranges({4}) == "4"


def _nad3_args(d, *extra):
    return ["--xml-dir", str(d), "--tiltstack-dir", str(d / "imod"),
            "--tomostar-dir", str(d), "--backup-dir", str(d / "bak"),
            "--mdoc-dir", str(d.parent / "PACE"), "--angles-from-log", *extra]


def _table(output):
    lines = output.split("\n")
    i = next(k for k, l in enumerate(lines) if l.startswith("Diagnosis"))
    rows = {}
    for l in lines[i + 3:]:
        if not l.startswith("NAD069"):
            break
        cols = l.split()
        rows[cols[0]] = cols
    return rows, lines


@pytest.mark.parametrize("delete", [False, True])
def test_diagnosis_table_after_run(nad3, delete):
    res = CliRunner().invoke(skipped_views, _nad3_args(nad3, *(["--delete"] if delete else [])))
    assert res.exit_code == 0, res.output
    rows, lines = _table(res.output)
    assert len(rows) == 22
    assert any("22 series, 21 OK, 1 need attention." in l for l in lines)
    r = rows["NAD069_3_lam18_ts_002.mrc"]
    # series N | log kept/removed | XML kept/removed | tomostar kept/removed | check
    assert r[1:7] == ["37", "8-29", "1-7,30-37", "8-29", "1-7,30-37",
                      "8-29" if delete else "1-37"]
    assert r[8].startswith("OK")
    assert rows["NAD069_3_lam23_ts_001.mrc"][-3:] == ["SKIP:", "no", "taSolution.log"]
    assert rows["NAD069_3_lam33_ts_001.mrc"][2:9] == [
        "13-19", "1-12,20-37", "13-19", "1-12,20-37",
        "13-19" if delete else "1-37", "1-12,20-37" if delete else "-", "OK"]


def test_diagnosis_flags_truncated_tomostar(nad3):
    """The failure seen on real data: tomostar shorter than the XML."""
    assert CliRunner().invoke(skipped_views, _nad3_args(nad3)).exit_code == 0
    tomo = nad3 / "NAD069_3_lam18_ts_002.mrc.tomostar"
    lines = tomo.read_text().split("\n")
    tomo.write_text("\n".join(lines[:39]) + "\n")  # keep rows 1-29 only
    res = CliRunner().invoke(skipped_views, _nad3_args(
        nad3, "--xml-pattern", "*lam18_ts_002*.xml", "--dry-run"))
    rows, _ = _table(res.output)
    r = rows["NAD069_3_lam18_ts_002.mrc"]
    assert r[6:8] == ["1-29", "30-37"] and r[8] == "MISMATCH:"


def test_diagnosis_after_second_delete_uses_original_numbering(nad3):
    args = _nad3_args(nad3, "--xml-pattern", "*lam36_ts_002*.xml", "--delete")
    assert CliRunner().invoke(skipped_views, args).exit_code == 0
    res = CliRunner().invoke(skipped_views, args + ["--dry-run"])
    r, _ = _table(res.output)
    assert r["NAD069_3_lam36_ts_002.mrc"][1:9] == [
        "37", "6-22", "1-5,23-37", "6-22", "1-5,23-37", "6-22", "1-5,23-37", "OK"]


def test_no_summary_flag(nad3):
    res = CliRunner().invoke(skipped_views, _nad3_args(nad3, "--dry-run", "--no-summary"))
    assert res.exit_code == 0 and "Diagnosis" not in res.output
