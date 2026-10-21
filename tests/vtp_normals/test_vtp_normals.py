"""Tests for vtp-normals: piece labelling / splitting, normal flipping, the
--split-column generalization, a regression against the original standalone
scripts, and the CLI (headless, Agg backend).

Synthetic surfaces are small triangulated grids written as real .vtp files
(appended + zlib, like morphometrics writes them). No large data is committed;
the regression and integration tests against the original scripts / example
.vtp are skipped when ``tests/vtp_flip_normal/`` is absent.
"""

import importlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import vtk
from click.testing import CliRunner
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

# matplotlib is imported lazily inside the tests that need it (and by the
# preview/curate commands themselves), never at module top: importing it at
# collection time would leave it in sys.modules and break other tools'
# "core does not import matplotlib" checks.
from tomo_toolshed.vtp_normals import core
from tomo_toolshed.vtp_normals.cli import vtp_normals

ORIG_DIR = Path(__file__).resolve().parent.parent / "vtp_flip_normal"
EXAMPLE_VTP = ORIG_DIR / "MIM019_2_lam11_ts_003.mrc.labels_IMM.AVV_rh9.flipped.vtp"


# ---------------------------------------------------------------------------
# Synthetic surface builders.
# ---------------------------------------------------------------------------
def grid(nx, ny, vx0=0.0, vy0=0.0, vz=0.0, vertex_offset=0):
    """A rectangular grid of (nx * ny) points triangulated into 2*(nx-1)*(ny-1) cells."""
    pts, idx = [], {}
    for j in range(ny):
        for i in range(nx):
            idx[(i, j)] = len(pts)
            pts.append((vx0 + i, vy0 + j, vz))
    tris = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            a, b, c, d = idx[(i, j)], idx[(i + 1, j)], idx[(i, j + 1)], idx[(i + 1, j + 1)]
            tris += [(a, b, d), (a, d, c)]
    return np.asarray(pts, float), np.asarray(tris, int) + vertex_offset


def two_patches():
    """Two vertex-disjoint grids (50 and 18 triangles); returns (points, triangles)."""
    pa, ta = grid(6, 6)
    pb, tb = grid(4, 4, vx0=100, vertex_offset=len(pa))
    return np.vstack([pa, pb]), np.vstack([ta, tb])


def cell_centres(points, triangles):
    return points[triangles].mean(axis=1)


def standard_arrays(centres, split_values=None, split_column="OMM_dist"):
    """A pycurv-like set of cell arrays with distinct values so flip rules are testable."""
    t = len(centres)
    up = np.tile([0.0, 0.0, 1.0], (t, 1))
    arrays = {
        "xyz": centres.astype(np.float64),
        "n_v": up.copy(), "normal": up.copy(), "avg_normals": up.copy(),
        "mean_curvature": np.full(t, 1.5), "mean_curvature_VV": np.full(t, 1.4),
        "shape_index_VV": np.full(t, 0.5), "shape_index_cat": np.full(t, 3.0),
        "kappa_1": np.full(t, 2.0), "kappa_2": np.full(t, 1.0),
        "max_curvature": np.full(t, 2.5), "min_curvature": np.full(t, 0.5),
        "t_1": np.tile([1.0, 0.0, 0.0], (t, 1)), "t_2": np.tile([0.0, 1.0, 0.0], (t, 1)),
        "gauss_curvature": np.full(t, 0.7),
    }
    if split_values is not None:
        arrays[split_column] = np.asarray(split_values, float)
    return arrays


def write_vtp(path, points, triangles, cell_arrays):
    """Write a triangle-only polydata as an appended, zlib-compressed .vtp."""
    poly = vtk.vtkPolyData()
    vpts = vtk.vtkPoints()
    vpts.SetData(numpy_to_vtk(np.ascontiguousarray(points, np.float64), deep=1))
    poly.SetPoints(vpts)
    offsets = np.arange(0, triangles.size + 1, 3, dtype=np.int64)
    cells = vtk.vtkCellArray()
    cells.SetData(numpy_to_vtk(offsets, deep=1, array_type=vtk.VTK_ID_TYPE),
                  numpy_to_vtk(np.ascontiguousarray(triangles.ravel(), np.int64), deep=1,
                               array_type=vtk.VTK_ID_TYPE))
    poly.SetPolys(cells)
    cd = poly.GetCellData()
    for name, arr in cell_arrays.items():
        a = numpy_to_vtk(np.ascontiguousarray(arr, np.float64), deep=1)
        a.SetName(name)
        cd.AddArray(a)
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(poly)
    writer.SetDataModeToAppended()
    writer.SetCompressorTypeToZLib()
    assert writer.Write(), f"failed to write {path}"
    return str(path)


def read_cell_arrays(path):
    r = vtk.vtkXMLPolyDataReader()
    r.SetFileName(str(path))
    r.Update()
    poly = r.GetOutput()
    cd = poly.GetCellData()
    arrays = {cd.GetArrayName(i): vtk_to_numpy(cd.GetArray(i)) for i in range(cd.GetNumberOfArrays())}
    conn = vtk_to_numpy(poly.GetPolys().GetConnectivityArray()).reshape(-1, 3)
    return arrays, conn


# ---------------------------------------------------------------------------
# label_pieces.
# ---------------------------------------------------------------------------
def test_two_disjoint_patches_numbered_largest_first():
    points, triangles = two_patches()
    labels, sizes = core.label_pieces(triangles)
    assert len(sizes) == 2
    assert sizes.tolist() == [50, 18]  # largest first
    assert np.array_equal(np.unique(labels), [0, 1])
    assert (labels == 0).sum() == 50 and (labels == 1).sum() == 18


def test_single_patch_split_by_column():
    points, triangles = grid(9, 5)
    split = cell_centres(points, triangles)[:, 0]  # x of each triangle
    labels, sizes = core.label_pieces(triangles, split, split_dist=4.0, merge_below=0)
    assert len(sizes) == 2
    assert sizes.sum() == len(triangles)


def test_split_dist_zero_is_connected_components_only():
    points, triangles = grid(9, 5)
    split = cell_centres(points, triangles)[:, 0]
    labels, sizes = core.label_pieces(triangles, split, split_dist=0.0, merge_below=50)
    assert len(sizes) == 1  # one connected patch, no split


def test_speckle_below_min_triangles_is_merged():
    points, triangles = grid(10, 6)
    split = np.full(len(triangles), 100.0)  # all "far"
    # flip two triangles to "near": a tiny speckle that would be its own piece
    speckle = [0, 1]
    split[speckle] = 0.0
    labels_split, sizes_split = core.label_pieces(triangles, split, split_dist=20.0, merge_below=0)
    labels_merged, sizes_merged = core.label_pieces(triangles, split, split_dist=20.0, merge_below=50)
    assert len(sizes_split) == 2  # big far part + the 2-triangle speckle
    assert sizes_split.min() == 2
    assert len(sizes_merged) == 1  # speckle merged back into its neighbour
    assert sizes_merged[0] == len(triangles)


# ---------------------------------------------------------------------------
# --split-column via surface_pieces / piece_split_median.
# ---------------------------------------------------------------------------
def test_custom_split_column(tmp_path):
    points, triangles = grid(9, 5)
    split = cell_centres(points, triangles)[:, 0]
    arrays = standard_arrays(cell_centres(points, triangles), split_values=split, split_column="my_dist")
    path = write_vtp(tmp_path / "s.vtp", points, triangles, arrays)
    poly = core.read_polydata(path)
    tri = vtk_to_numpy(poly.GetPolys().GetConnectivityArray()).reshape(-1, 3)
    labels, sizes = core.surface_pieces(poly, tri, split_dist=4.0, min_triangles=0, split_column="my_dist")
    assert len(sizes) == 2
    median = core.piece_split_median(poly, labels, len(sizes), split_column="my_dist")
    assert not np.isnan(median).any()


def test_default_split_column_is_omm_dist(tmp_path):
    points, triangles = grid(9, 5)
    split = cell_centres(points, triangles)[:, 0]
    arrays = standard_arrays(cell_centres(points, triangles), split_values=split, split_column="OMM_dist")
    path = write_vtp(tmp_path / "s.vtp", points, triangles, arrays)
    poly = core.read_polydata(path)
    tri = vtk_to_numpy(poly.GetPolys().GetConnectivityArray()).reshape(-1, 3)
    labels, sizes = core.surface_pieces(poly, tri, split_dist=4.0, min_triangles=0)  # default column
    assert len(sizes) == 2


def test_missing_split_column_falls_back_with_note(tmp_path, capsys):
    points, triangles = grid(9, 5)
    arrays = standard_arrays(cell_centres(points, triangles))  # no split column at all
    path = write_vtp(tmp_path / "s.vtp", points, triangles, arrays)
    poly = core.read_polydata(path)
    tri = vtk_to_numpy(poly.GetPolys().GetConnectivityArray()).reshape(-1, 3)
    labels, sizes = core.surface_pieces(poly, tri, split_dist=20.0, min_triangles=0)
    assert len(sizes) == 1  # connected components only
    assert "no 'OMM_dist' array" in capsys.readouterr().out
    median = core.piece_split_median(poly, labels, len(sizes))
    assert np.isnan(median).all()


# ---------------------------------------------------------------------------
# flip_surface.
# ---------------------------------------------------------------------------
def _two_piece_vtp(tmp_path, name="surf.vtp"):
    points, triangles = two_patches()
    arrays = standard_arrays(cell_centres(points, triangles))  # no split column -> 2 connected pieces
    return write_vtp(tmp_path / name, points, triangles, arrays), triangles


def test_flip_negates_normals_and_reverses_winding(tmp_path):
    path, triangles = _two_piece_vtp(tmp_path)
    before, conn_before = read_cell_arrays(path)
    out = core.flip_surface(path, {0}, split_dist=0.0)
    after, conn_after = read_cell_arrays(out)

    flipped = after["normal_flipped"].astype(bool)
    assert flipped.sum() == 50  # piece 0
    # normal arrays negated exactly on flipped cells, untouched elsewhere
    for name in ("n_v", "normal", "avg_normals"):
        assert np.allclose(after[name][flipped], -before[name][flipped])
        assert np.allclose(after[name][~flipped], before[name][~flipped])
    # winding reversed on flipped cells only
    assert np.array_equal(conn_after[flipped], conn_before[flipped][:, ::-1])
    assert np.array_equal(conn_after[~flipped], conn_before[~flipped])


def test_flip_untouched_pieces_byte_identical(tmp_path):
    path, _ = _two_piece_vtp(tmp_path)
    before, conn_before = read_cell_arrays(path)
    out = core.flip_surface(path, {0}, split_dist=0.0)
    after, conn_after = read_cell_arrays(out)
    keep = ~after["normal_flipped"].astype(bool)
    for name in before:  # every original array unchanged on the kept piece
        assert np.array_equal(after[name][keep], before[name][keep]), name


def test_flip_curvature_negate_and_swap_rules(tmp_path):
    path, _ = _two_piece_vtp(tmp_path)
    before, _ = read_cell_arrays(path)
    out = core.flip_surface(path, {0}, split_dist=0.0)
    after, _ = read_cell_arrays(out)
    f = after["normal_flipped"].astype(bool)
    # negated scalars
    for name in core.CURVATURE_NEGATE:
        assert np.allclose(after[name][f], -before[name][f]), name
    # swap + negate: kappa_1 <- -kappa_2, kappa_2 <- -kappa_1
    for a, b in core.CURVATURE_SWAP_NEGATE:
        assert np.allclose(after[a][f], -before[b][f])
        assert np.allclose(after[b][f], -before[a][f])
    assert (after["kappa_1"][f] >= after["kappa_2"][f]).all()  # ordering preserved
    # pure swap: t_1 <-> t_2
    for a, b in core.CURVATURE_SWAP:
        assert np.allclose(after[a][f], before[b][f])
        assert np.allclose(after[b][f], before[a][f])


def test_no_curvature_leaves_curvature_alone(tmp_path):
    path, _ = _two_piece_vtp(tmp_path)
    before, _ = read_cell_arrays(path)
    out = core.flip_surface(path, {0}, curvature=False, split_dist=0.0)
    after, _ = read_cell_arrays(out)
    f = after["normal_flipped"].astype(bool)
    for name in core.CURVATURE_NEGATE + ("kappa_1", "kappa_2", "t_1", "t_2", "max_curvature", "min_curvature"):
        assert np.array_equal(after[name], before[name]), name
    # normals still flipped though
    assert np.allclose(after["n_v"][f], -before["n_v"][f])


def test_flip_writes_piece_id_and_normal_flipped(tmp_path):
    path, _ = _two_piece_vtp(tmp_path)
    out = core.flip_surface(path, {1}, split_dist=0.0)
    after, _ = read_cell_arrays(out)
    assert "piece_id" in after and "normal_flipped" in after
    assert set(np.unique(after["piece_id"])) == {0, 1}
    assert after["normal_flipped"].astype(bool).sum() == 18  # piece 1


def test_double_flip_restores_normals(tmp_path):
    path, _ = _two_piece_vtp(tmp_path)
    before, _ = read_cell_arrays(path)
    out1 = core.flip_surface(path, {0}, split_dist=0.0)  # surf.flipped.vtp
    out2 = core.flip_surface(out1, {0}, split_dist=0.0)  # surf.flipped.flipped.vtp
    after, _ = read_cell_arrays(out2)
    for name in ("n_v", "normal", "avg_normals", "kappa_1", "kappa_2", "t_1", "t_2"):
        assert np.allclose(after[name], before[name]), name


def test_flip_out_of_range_piece_raises(tmp_path):
    import click
    path, _ = _two_piece_vtp(tmp_path)
    with pytest.raises(click.ClickException):
        core.flip_surface(path, {5}, split_dist=0.0)


# ---------------------------------------------------------------------------
# parse_flips / --flip-file.
# ---------------------------------------------------------------------------
def test_parse_flips_cli_and_file(tmp_path):
    f = tmp_path / "flips.txt"
    f.write_text("# a comment\n\nlam12_ts_003:IMM:0,3  # inline comment\nlam12_ts_004:IMM:1\n")
    parsed = core.parse_flips(("lam11_ts_001:OMM:2",), str(f))
    assert ("lam11_ts_001", "OMM", {2}) in parsed
    assert ("lam12_ts_003", "IMM", {0, 3}) in parsed
    assert ("lam12_ts_004", "IMM", {1}) in parsed


def test_parse_flips_bad_entries_raise():
    import click
    with pytest.raises(click.BadParameter):
        core.parse_flips(("not_three_parts",), None)
    with pytest.raises(click.BadParameter):
        core.parse_flips(("ts:IMM:notanint",), None)


def test_flip_file_out_of_range_id_raises(tmp_path):
    (tmp_path / "surfaces").mkdir()
    points, triangles = two_patches()
    arrays = standard_arrays(cell_centres(points, triangles))
    write_vtp(tmp_path / "surfaces" / "lam_ts_001.mrc.labels_IMM.vtp", points, triangles, arrays)
    ff = tmp_path / "f.txt"
    ff.write_text("ts_001:IMM:9\n")  # no piece 9
    res = CliRunner().invoke(vtp_normals, ["flip", "--vtp-dir", str(tmp_path / "surfaces"),
                                           "--flip-file", str(ff), "--no-preview", "--split-dist", "0"])
    assert res.exit_code != 0
    assert "no piece" in res.output


# ---------------------------------------------------------------------------
# Regression against the original standalone scripts.
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def originals():
    if not ORIG_DIR.exists():
        pytest.skip(f"original scripts not present ({ORIG_DIR})")
    sys.path.insert(0, str(ORIG_DIR))
    pn = importlib.import_module("preview_normals")
    fn = importlib.import_module("flip_normals")
    return pn, fn


def test_label_pieces_matches_original(originals):
    pn, _ = originals
    points, triangles = grid(10, 6)
    split = cell_centres(points, triangles)[:, 0]
    for sd, mb in [(0.0, 0), (4.0, 0), (4.0, 50), (20.0, 50)]:
        o_lab, o_sz = pn.label_pieces(triangles, split, sd, mb)
        n_lab, n_sz = core.label_pieces(triangles, split, sd, mb)
        assert np.array_equal(o_lab, n_lab), (sd, mb)
        assert np.array_equal(o_sz, n_sz), (sd, mb)


def test_flip_surface_matches_original(originals, tmp_path):
    _, fn = originals
    points, triangles = two_patches()
    # include OMM_dist so both the original (hard-coded) and ported paths split identically
    split = np.where(cell_centres(points, triangles)[:, 0] < 3, 0.0, 100.0)
    arrays = standard_arrays(cell_centres(points, triangles), split_values=split)
    (tmp_path / "o").mkdir()
    (tmp_path / "n").mkdir()
    op = write_vtp(tmp_path / "o" / "s.vtp", points, triangles, arrays)
    npth = write_vtp(tmp_path / "n" / "s.vtp", points, triangles, arrays)
    o_out = fn.flip_surface(op, {0})
    n_out = core.flip_surface(npth, {0})
    oa, oc = read_cell_arrays(o_out)
    na, nc = read_cell_arrays(n_out)
    assert set(oa) == set(na)
    for name in oa:
        assert np.array_equal(oa[name], na[name]), name
    assert np.array_equal(oc, nc)


# ---------------------------------------------------------------------------
# CLI (headless).
# ---------------------------------------------------------------------------
def test_group_and_subcommand_help():
    runner = CliRunner()
    top = runner.invoke(vtp_normals, ["--help"])
    assert top.exit_code == 0
    for sub in ("curate", "preview", "flip"):
        assert sub in top.output
        res = runner.invoke(vtp_normals, [sub, "--help"])
        assert res.exit_code == 0, res.output
        assert "--split-column" in res.output


def test_registered_in_top_level_group():
    from tomo_toolshed.cli import tomo_toolshed
    res = CliRunner().invoke(tomo_toolshed, ["--help"])
    assert "vtp-normals" in res.output
    assert CliRunner().invoke(tomo_toolshed, ["vtp-normals", "--help"]).exit_code == 0
    assert CliRunner().invoke(tomo_toolshed, ["vtp_normals", "--help"]).exit_code == 0  # hidden alias


def test_core_and_cli_import_without_matplotlib():
    code = ("import sys; import tomo_toolshed.vtp_normals.cli, tomo_toolshed.vtp_normals.core; "
            "assert 'matplotlib' not in sys.modules, "
            "sorted(m for m in sys.modules if m.startswith('matplotlib'))")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_cli_flip_end_to_end(tmp_path):
    surfaces = tmp_path / "surfaces"
    surfaces.mkdir()
    points, triangles = two_patches()
    arrays = standard_arrays(cell_centres(points, triangles))
    write_vtp(surfaces / "foo_ts_001.mrc.labels_IMM.vtp", points, triangles, arrays)
    res = CliRunner().invoke(vtp_normals, ["flip", "--vtp-dir", str(surfaces),
                                           "--flip", "ts_001:IMM:0", "--no-preview", "--split-dist", "0"])
    assert res.exit_code == 0, res.output
    out = surfaces / "foo_ts_001.mrc.labels_IMM.flipped.vtp"
    assert out.exists()
    after, _ = read_cell_arrays(out)
    assert after["normal_flipped"].astype(bool).sum() == 50


def test_cli_preview_without_tomogram_writes_png(tmp_path):
    points, triangles = two_patches()
    arrays = standard_arrays(cell_centres(points, triangles))
    write_vtp(tmp_path / "foo_ts_001.vtp", points, triangles, arrays)
    res = CliRunner().invoke(vtp_normals, ["preview", "--vtp-glob", str(tmp_path / "*.vtp"),
                                           "--split-dist", "0", "--min-triangles", "0"])
    assert res.exit_code == 0, res.output
    assert "tomogram not found" in res.output
    assert (tmp_path / "foo_ts_001.normals.png").exists()


def test_cli_curate_errors_on_agg_backend(tmp_path):
    import matplotlib
    matplotlib.use("Agg", force=True)
    points, triangles = two_patches()
    arrays = standard_arrays(cell_centres(points, triangles))
    path = write_vtp(tmp_path / "s.vtp", points, triangles, arrays)
    res = CliRunner().invoke(vtp_normals, ["curate", "--vtp", str(path)])
    assert res.exit_code != 0
    assert "no interactive matplotlib backend" in res.output


# ---------------------------------------------------------------------------
# Optional integration test on the real example (no large MRC loaded).
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_real_example_vtp_loads_labels_and_flips(tmp_path):
    if not EXAMPLE_VTP.exists():
        pytest.skip("example .vtp not present")
    import shutil
    local = tmp_path / "example.vtp"
    shutil.copy(EXAMPLE_VTP, local)
    poly = core.read_polydata(str(local))
    tri = vtk_to_numpy(poly.GetPolys().GetConnectivityArray()).reshape(-1, 3)
    labels, sizes = core.surface_pieces(poly, tri)  # fallback: no OMM_dist array
    assert len(sizes) >= 1
    out = core.flip_surface(str(local), {0})
    after, _ = read_cell_arrays(out)
    assert after["normal_flipped"].astype(bool).sum() == sizes[0]
