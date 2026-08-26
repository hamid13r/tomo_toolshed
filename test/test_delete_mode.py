"""Tests for --delete mode of remove_skipped_view.py.

Run:  pytest test/test_delete_mode.py
"""
import os
import sys

import pytest
from lxml import etree

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import remove_skipped_view as rsv

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REAL_XMLS = [
    os.path.join(REPO, "test", "Position_2_2.xml"),
    os.path.join(REPO, "test", "VLP3x3_p02_ts_003_blend.xml"),
]


# --------------------------------------------------------------------------- #
# Synthetic fixtures built in Warp's exact on-disk format.
# --------------------------------------------------------------------------- #
def build_xml(movies, angles, doses, use_tilt=None):
    n = len(movies)
    if use_tilt is None:
        use_tilt = ["True"] * n
    L = []
    L.append('﻿<?xml version="1.0" encoding="utf-8"?>')
    L.append('<TiltSeries DataDirectory="/x" AreAnglesInverted="False">')
    L.append("\t<Angles>" + "\n".join(str(a) for a in angles) + "</Angles>")
    L.append("\t<Dose>" + "\n".join(str(d) for d in doses) + "</Dose>")
    L.append("\t<UseTilt>" + "\n".join(use_tilt) + "</UseTilt>")
    L.append("\t<AxisAngle>" + "\n".join(["85"] * n) + "</AxisAngle>")
    L.append("\t<MoviePath>" + "\n".join("../warp_frameseries/" + m for m in movies) + "</MoviePath>")
    for i in range(n):
        L.append(f'\t<TiltPS1D ID="{i}">ps{i}</TiltPS1D>')
    # GridCTF: single node per Z-slice (W=H=1).
    L.append('\t<GridCTF Width="1" Height="1" Depth="%d" MarginX="0" MarginY="0" MarginZ="0">' % n)
    for z in range(n):
        L.append(f'\t\t<Node X="0" Y="0" Z="{z}" Value="{z}.0" />')
    L.append("\t</GridCTF>")
    # GridMovementX: multi-node per Z-slice (W=H=2 -> 4 nodes per tilt).
    L.append('\t<GridMovementX Width="2" Height="2" Depth="%d" MarginX="0" MarginY="0" MarginZ="0">' % n)
    for z in range(n):
        for y in range(2):
            for x in range(2):
                L.append(f'\t\t<Node X="{x}" Y="{y}" Z="{z}" Value="{z}{y}{x}" />')
    L.append("\t</GridMovementX>")
    # GridMovementY: global (Depth=1) -> must stay untouched.
    L.append('\t<GridMovementY Width="1" Height="1" Depth="1" MarginX="0" MarginY="0" MarginZ="0">')
    L.append('\t\t<Node X="0" Y="0" Z="0" Value="0" />')
    L.append("\t</GridMovementY>")
    L.append("</TiltSeries>")
    return "\n".join(L).encode("utf-8")


def build_tomostar(movies, angles, doses, rows_order=None):
    order = rows_order if rows_order is not None else range(len(movies))
    lines = ["", "data_", "", "loop_", "_wrpMovieName #1", "_wrpAngleTilt #2", "_wrpDose #3"]
    for i in order:
        lines.append(f"  ../warp_frameseries/{movies[i]}  {angles[i]}  {doses[i]}")
    lines.append("")  # trailing newline
    return "\n".join(lines)


@pytest.fixture
def scene(tmp_path):
    movies = [f"m{i}.eer" for i in range(4)]
    angles = [-6, -3, 0, 3]
    doses = [10, 5, 0, 2.5]
    xml = tmp_path / "TS.xml"
    tomo = tmp_path / "TS.tomostar"
    xml.write_bytes(build_xml(movies, angles, doses))
    tomo.write_text(build_tomostar(movies, angles, doses))
    return dict(dir=tmp_path, xml=str(xml), tomo=str(tomo), movies=movies,
               angles=angles, doses=doses)


def do_delete(scene, deleted_idx, dry_run=False):
    """Run handle_delete with an explicit keep-mask."""
    n = 4
    mask = [i not in deleted_idx for i in range(n)]
    rsv.handle_delete(scene["xml"], str(scene["dir"]), mask, scene["angles"], dry_run)


# --------------------------------------------------------------------------- #
# Round-trip identity on the real files -- proves serialization is Warp-safe.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("path", REAL_XMLS)
def test_roundtrip_byte_identical(path):
    orig = open(path, "rb").read()
    assert rsv.serialize_xml(rsv.parse_xml(path)) == orig


def test_roundtrip_synthetic(scene):
    orig = open(scene["xml"], "rb").read()
    assert rsv.serialize_xml(rsv.parse_xml(scene["xml"])) == orig


# --------------------------------------------------------------------------- #
# Deletion correctness
# --------------------------------------------------------------------------- #
def _reload(xml_path):
    root = rsv.parse_xml(xml_path).getroot()
    return root


def test_delete_interior(scene):
    do_delete(scene, {1})  # drop tilt index 1 (angle -3, m1.eer)
    root = _reload(scene["xml"])
    assert rsv.split_lines(root.find("Angles").text) == ["-6", "0", "3"]
    assert rsv.split_lines(root.find("MoviePath").text) == [
        "../warp_frameseries/m0.eer",
        "../warp_frameseries/m2.eer",
        "../warp_frameseries/m3.eer",
    ]
    # Indexed IDs renumbered 0..2
    ids = [int(e.get("ID")) for e in root.findall("TiltPS1D")]
    assert ids == [0, 1, 2]
    assert [e.text for e in root.findall("TiltPS1D")] == ["ps0", "ps2", "ps3"]
    # Grid Depth + contiguous Z
    ctf = root.find("GridCTF")
    assert ctf.get("Depth") == "3"
    assert [int(nd.get("Z")) for nd in ctf.findall("Node")] == [0, 1, 2]
    # Values follow the surviving tilts (0.0, 2.0, 3.0)
    assert [nd.get("Value") for nd in ctf.findall("Node")] == ["0.0", "2.0", "3.0"]
    # Global grid untouched
    assert root.find("GridMovementY").get("Depth") == "1"
    # tomostar rows
    rows = _tomostar_movies(scene["tomo"])
    assert rows == ["m0.eer", "m2.eer", "m3.eer"]
    # Structure fully consistent at M=3
    rsv.assert_after_delete(root, 3)


def test_delete_first_and_last(scene):
    do_delete(scene, {0, 3})
    root = _reload(scene["xml"])
    assert rsv.split_lines(root.find("Angles").text) == ["-3", "0"]
    assert [int(e.get("ID")) for e in root.findall("TiltPS1D")] == [0, 1]
    ctf = root.find("GridCTF")
    assert ctf.get("Depth") == "2"
    assert [int(nd.get("Z")) for nd in ctf.findall("Node")] == [0, 1]
    assert [nd.get("Value") for nd in ctf.findall("Node")] == ["1.0", "2.0"]
    rsv.assert_after_delete(root, 2)
    assert _tomostar_movies(scene["tomo"]) == ["m1.eer", "m2.eer"]


def test_multinode_grid_z_renumber(scene):
    """GridMovementX has W=H=2 -> 4 nodes per Z; deleting a tilt drops all 4
    and renumbers the surviving slices contiguously."""
    do_delete(scene, {1})
    root = _reload(scene["xml"])
    grid = root.find("GridMovementX")
    assert grid.get("Depth") == "3"
    nodes = grid.findall("Node")
    assert len(nodes) == 2 * 2 * 3  # W*H*M
    zs = sorted(set(int(nd.get("Z")) for nd in nodes))
    assert zs == [0, 1, 2]
    # No surviving node should carry a value from the deleted slice (z==1: "1yx")
    assert all(not nd.get("Value").startswith("1") for nd in nodes)


def test_name_based_matching_when_tomostar_reordered(scene):
    """tomostar rows in a different order than the XML are still matched by
    movie name, not position."""
    reordered = build_tomostar(scene["movies"], scene["angles"], scene["doses"],
                               rows_order=[2, 0, 3, 1])
    open(scene["tomo"], "w").write(reordered)
    do_delete(scene, {1})  # delete m1.eer
    assert _tomostar_movies(scene["tomo"]) == ["m2.eer", "m0.eer", "m3.eer"]


def test_positional_fallback_warns(scene, capsys):
    """When names can't be matched, fall back to row order with a warning."""
    root = rsv.parse_xml(scene["xml"]).getroot()
    xml_bns = rsv.movie_basenames(root, 4)
    # tomostar with unrelated movie names -> no name match possible.
    weird = build_tomostar(["z0.eer", "z1.eer", "z2.eer", "z3.eer"],
                           scene["angles"], scene["doses"])
    open(scene["tomo"], "w").write(weird)
    new_text, warning, remaining = rsv.delete_tomostar_rows(scene["tomo"], {1}, xml_bns, 4)
    assert warning is not None and "row order" in warning
    assert remaining == 3
    # Positional: row index 1 (z1.eer) removed.
    kept = [l.split()[0].split("/")[-1] for l in new_text.split("\n") if ".eer" in l]
    assert kept == ["z0.eer", "z2.eer", "z3.eer"]


# --------------------------------------------------------------------------- #
# Aborts and safety
# --------------------------------------------------------------------------- #
def test_count_mismatch_aborts_untouched(scene):
    """A tomostar row count != N aborts the series and touches nothing."""
    short = build_tomostar(scene["movies"][:3], scene["angles"][:3], scene["doses"][:3])
    open(scene["tomo"], "w").write(short)
    before_xml = open(scene["xml"], "rb").read()
    before_tomo = open(scene["tomo"]).read()
    do_delete(scene, {1})
    assert open(scene["xml"], "rb").read() == before_xml
    assert open(scene["tomo"]).read() == before_tomo
    assert not os.path.exists(scene["xml"] + ".bak")


def test_xml_structure_mismatch_aborts_untouched(scene):
    """A per-tilt list whose length != N aborts before any edit."""
    root = rsv.parse_xml(scene["xml"]).getroot()
    # Corrupt: give Dose only 3 entries.
    root.find("Dose").text = "1\n2\n3"
    open(scene["xml"], "wb").write(rsv.serialize_xml(root.getroottree()))
    before = open(scene["xml"], "rb").read()
    do_delete(scene, {1})
    assert open(scene["xml"], "rb").read() == before
    assert not os.path.exists(scene["xml"] + ".bak")


def test_versioned_backup(tmp_path):
    f = tmp_path / "a.xml"
    f.write_text("one")
    b1 = rsv.make_backup(str(f), str(f) + ".bak")
    f.write_text("two")
    b2 = rsv.make_backup(str(f), str(f) + ".bak")
    assert b1 == str(f) + ".bak"
    assert b2 == str(f) + ".bak.1"
    assert open(b1).read() == "one"
    assert open(b2).read() == "two"


def test_existing_bak_versions_and_proceeds(scene):
    """Per project decision: an existing .bak does not skip; a numbered backup
    is made and deletion proceeds."""
    open(scene["xml"] + ".bak", "w").write("preexisting")
    open(scene["tomo"] + ".bak", "w").write("preexisting")
    do_delete(scene, {1})
    # Original .bak preserved, new versioned backups created.
    assert open(scene["xml"] + ".bak").read() == "preexisting"
    assert os.path.exists(scene["xml"] + ".bak.1")
    assert os.path.exists(scene["tomo"] + ".bak.1")
    # Deletion actually happened.
    assert rsv.split_lines(_reload(scene["xml"]).find("Angles").text) == ["-6", "0", "3"]


def test_dry_run_writes_nothing(scene):
    before_xml = open(scene["xml"], "rb").read()
    before_tomo = open(scene["tomo"]).read()
    do_delete(scene, {1}, dry_run=True)
    assert open(scene["xml"], "rb").read() == before_xml
    assert open(scene["tomo"]).read() == before_tomo
    assert not os.path.exists(scene["xml"] + ".bak")
    assert not os.path.exists(scene["tomo"] + ".bak")


# --------------------------------------------------------------------------- #
# Keep-mask (max-tilt uses the fixed Dose parsing)
# --------------------------------------------------------------------------- #
def test_keep_mask_max_tilt(scene):
    """--max-tilt keeps views within N degrees of the lowest-dose tilt, and no
    longer crashes on Dose parsing."""
    root = rsv.parse_xml(scene["xml"]).getroot()
    views = {1, 2, 3, 4}  # all four tilts survived alignment (1-based)
    # doses = [10,5,0,2.5] -> lowest dose at index 2 (angle 0); keep within 3 deg.
    mask = rsv.compute_keep_mask(root, views, scene["angles"], False, 3)
    # angles = [-6,-3,0,3]; |angle| <= |0| + 3 -> indices 1,2,3
    assert mask == [False, True, True, True]


def _tomostar_movies(path):
    return [l.split()[0].split("/")[-1] for l in open(path).read().split("\n") if ".eer" in l]
