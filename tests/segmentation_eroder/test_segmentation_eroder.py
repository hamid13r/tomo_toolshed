"""Tests for erode: XY vs 3D erosion on (Z, Y, X) volumes, connectivities,
edge handling, and the CLI round trip (header voxel size / origin kept)."""

import numpy as np
import pytest
from click.testing import CliRunner

from tomo_toolshed.segmentation_eroder import core
from tomo_toolshed.segmentation_eroder.cli import default_output_path, erode_cmd


def _cube(shape=(9, 11, 13), lo=2, hi=7):
    """Box of foreground voxels [lo, hi) on every axis inside a (Z, Y, X) volume."""
    m = np.zeros(shape, dtype=np.uint8)
    m[lo:hi, lo:hi, lo:hi] = 1
    return m


# ---------------------------------------------------------------------------
# Structuring elements.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("three_d,conn,n", [(False, 4, 4), (False, 8, 8),
                                            (True, 6, 6), (True, 18, 18),
                                            (True, 26, 26)])
def test_neighbour_offsets_count(three_d, conn, n):
    offs = core.neighbour_offsets(three_d, conn)
    assert len(offs) == n
    if not three_d:
        assert all(dz == 0 for dz, _, _ in offs)


def test_default_connectivity_is_cross():
    assert len(core.neighbour_offsets(False)) == 4
    assert len(core.neighbour_offsets(True)) == 6


@pytest.mark.parametrize("three_d,conn", [(False, 6), (False, 26), (True, 4), (True, 8)])
def test_connectivity_must_match_mode(three_d, conn):
    with pytest.raises(core.ErodeError):
        core.neighbour_offsets(three_d, conn)


# ---------------------------------------------------------------------------
# Erosion.
# ---------------------------------------------------------------------------
def test_xy_erosion_does_not_touch_z():
    out = core.erode(_cube(), iterations=1)
    # Every originally-filled Z slice keeps a 3x3 core; nothing moves in Z.
    assert out.dtype == bool
    assert np.array_equal(np.flatnonzero(out.any(axis=(1, 2))), np.arange(2, 7))
    assert out[2:7, 3:6, 3:6].all()
    assert out.sum() == 5 * 3 * 3


def test_3d_erosion_also_shrinks_z():
    out = core.erode(_cube(), iterations=1, three_d=True)
    assert np.array_equal(np.flatnonzero(out.any(axis=(1, 2))), np.arange(3, 6))
    assert out.sum() == 3 * 3 * 3


def test_iterations_accumulate():
    once = core.erode(core.erode(_cube(), 1), 1)
    twice = core.erode(_cube(), 2)
    assert np.array_equal(once, twice)
    assert twice.sum() == 5 * 1 * 1


def test_square_vs_cross_on_corners():
    m = np.zeros((1, 7, 7), np.uint8)
    m[0, 1:6, 1:6] = 1
    m[0, 1, 1] = 0  # notch a corner: affects the 8-neighbour but not the 4-neighbour case
    cross = core.erode(m, 1, connectivity=4)
    square = core.erode(m, 1, connectivity=8)
    assert cross[0, 2, 2] and not square[0, 2, 2]


def test_nonzero_is_foreground_and_all_gone():
    m = _cube().astype(np.float32) * 0.3
    assert core.erode(m, 1).sum() == 45
    assert core.erode(m, 10).sum() == 0


def test_edge_touching_mask_not_eroded_from_edge():
    m = np.ones((3, 5, 5), np.uint8)
    assert core.erode(m, 2).all()
    assert core.erode(m, 2, three_d=True).all()


def test_2d_input():
    m = np.zeros((7, 7), np.uint8)
    m[1:6, 1:6] = 1
    out = core.erode(m, 1)
    assert out.shape == (7, 7) and out.sum() == 9
    with pytest.raises(core.ErodeError):
        core.erode(m, 1, three_d=True)


def test_matches_scipy():
    ndi = pytest.importorskip("scipy.ndimage")
    rng = np.random.default_rng(0)
    m = ndi.binary_dilation(rng.random((12, 20, 24)) > 0.85, iterations=2)
    for three_d, conn, rank_conn in [(False, 4, 1), (False, 8, 2),
                                     (True, 6, 1), (True, 18, 2), (True, 26, 3)]:
        st = ndi.generate_binary_structure(3, rank_conn)
        if not three_d:
            st = st.copy()
            st[0] = st[2] = False
        ref = ndi.binary_erosion(m, structure=st, iterations=2, border_value=1)
        assert np.array_equal(core.erode(m, 2, three_d, conn), ref), (three_d, conn)


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------
def test_default_output_path():
    assert default_output_path("/a/b/mem.mrc") == "/a/b/mem_eroded.mrc"


def test_cli_round_trip_keeps_header(tmp_path):
    mrcfile = pytest.importorskip("mrcfile")
    src = tmp_path / "seg.mrc"
    with mrcfile.new(str(src)) as mrc:
        mrc.set_data(_cube().astype(np.int8))
        mrc.voxel_size = 7.84
        mrc.header.origin = (10.0, 20.0, 30.0)

    out = tmp_path / "sub" / "out.mrc"
    res = CliRunner().invoke(erode_cmd, ["--i", str(src), "--o", str(out), "-n", "1"])
    assert res.exit_code == 0, res.output
    assert "125 -> 45 voxels" in res.output
    with mrcfile.open(str(out)) as mrc:
        assert mrc.data.shape == (9, 11, 13)
        assert int((mrc.data > 0).sum()) == 45
        assert set(np.unique(mrc.data)) <= {0, 1}
        assert np.isclose(mrc.voxel_size.x, 7.84, atol=1e-3)
        assert tuple(float(v) for v in mrc.header.origin.item()) == (10.0, 20.0, 30.0)


def test_cli_3d_and_default_output(tmp_path):
    mrcfile = pytest.importorskip("mrcfile")
    src = tmp_path / "seg.mrc"
    with mrcfile.new(str(src)) as mrc:
        mrc.set_data(_cube().astype(np.int8))
    res = CliRunner().invoke(erode_cmd, ["--i", str(src), "--3d", "-q"])
    assert res.exit_code == 0, res.output
    with mrcfile.open(str(tmp_path / "seg_eroded.mrc")) as mrc:
        assert int((mrc.data > 0).sum()) == 27


def test_cli_rejects_bad_connectivity_and_same_output(tmp_path):
    mrcfile = pytest.importorskip("mrcfile")
    src = tmp_path / "seg.mrc"
    with mrcfile.new(str(src)) as mrc:
        mrc.set_data(_cube().astype(np.int8))
    res = CliRunner().invoke(erode_cmd, ["--i", str(src), "--connectivity", "26"])
    assert res.exit_code != 0 and "not valid for XY" in res.output
    res = CliRunner().invoke(erode_cmd, ["--i", str(src), "--o", str(src)])
    assert res.exit_code != 0 and "must differ" in res.output


def test_registered_in_group():
    from tomo_toolshed.cli import tomo_toolshed

    res = CliRunner().invoke(tomo_toolshed, ["--help"])
    assert "erode" in res.output
