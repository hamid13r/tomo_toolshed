"""Tests for pick-granules: thresholding inside a mask, mask erosion, min-size
removal, percentile helpers, the headless CLI round trip and the GUI under Agg."""

import mrcfile
import numpy as np
import pytest
from click.testing import CliRunner

from tomo_toolshed.cli import tomo_toolshed
from tomo_toolshed.granule_picker import core
from tomo_toolshed.granule_picker.cli import default_output_path, pick_granules


def _phantom():
    """(Z, Y, X) tomogram of value 1 with a 'mitochondrion' mask, two dark
    granules inside it (one big, one single voxel) and one dark blob outside."""
    tomo = np.ones((10, 20, 20), dtype=np.float32)
    mask = np.zeros_like(tomo, dtype=np.uint8)
    mask[1:9, 2:18, 2:18] = 1
    tomo[3:6, 5:8, 5:8] = -2.0      # granule inside, 27 voxels
    tomo[6, 14, 14] = -2.0          # speck inside, 1 voxel
    tomo[3:6, 0:2, 0:2] = -2.0      # dark blob outside the mask
    return tomo, mask


def _write(path, data, voxel_size=2.5, origin=(1.0, 2.0, 3.0)):
    with mrcfile.new(str(path), overwrite=True) as mrc:
        mrc.set_data(np.ascontiguousarray(data))
        mrc.voxel_size = voxel_size
        mrc.header.origin = origin


def test_threshold_only_inside_mask():
    tomo, mask = _phantom()
    g = core.threshold_granules(tomo, mask, 0.0)
    assert g.sum() == 28
    assert not g[3:6, 0:2, 0:2].any()


def test_bright_inverts_comparison():
    tomo, mask = _phantom()
    g = core.threshold_granules(tomo, mask, 0.0, bright=True)
    assert g.sum() == int(mask.sum()) - 28


def test_shape_mismatch():
    with pytest.raises(core.GranuleError):
        core.threshold_granules(np.zeros((2, 3, 3)), np.zeros((2, 3, 4)), 0.0)


def test_remove_small_drops_speck():
    tomo, mask = _phantom()
    g = core.pick(tomo, mask, 0.0, min_size=2)
    assert g.dtype == np.uint8
    assert g.sum() == 27 and g[6, 14, 14] == 0


def test_mask_erosion_is_xy_only():
    _, mask = _phantom()
    m = core.prepare_mask(mask, 2)
    assert m[1:9, 4:16, 4:16].all()
    assert m.sum() == 8 * 12 * 12


def test_mask_erosion_3d_shrinks_z():
    _, mask = _phantom()
    m = core.prepare_mask(mask, 1, three_d=True)
    assert m.sum() == 6 * 14 * 14
    assert not m[1].any() and m[2:8, 3:17, 3:17].all()


def test_percentile_and_count_helpers():
    tomo, mask = _phantom()
    vals = core.masked_values(tomo, mask)
    assert core.count_picked(vals, 0.0) == 28
    assert core.count_picked(vals, 0.0, bright=True) == vals.size - 28
    assert core.percentile_of(vals, 0.0) == pytest.approx(100 * 28 / vals.size)
    assert core.threshold_from_percentile(vals, 0) == -2.0
    assert core.initial_threshold(vals, 0.5, None) == 0.5
    with pytest.raises(core.GranuleError):
        core.initial_threshold(vals, 0.5, 10)


def test_default_output_path():
    assert default_output_path("/a/mito.mrc") == "/a/mito_granules.mrc"


def test_cli_no_gui_roundtrip(tmp_path):
    tomo, mask = _phantom()
    _write(tmp_path / "tomo.mrc", tomo)
    _write(tmp_path / "mito.mrc", mask)
    out = tmp_path / "sub" / "g.mrc"
    res = CliRunner().invoke(tomo_toolshed, [
        "pick-granules", str(tmp_path / "tomo.mrc"), str(tmp_path / "mito.mrc"),
        "-o", str(out), "-t", "0", "--min-size", "2", "--no-gui"])
    assert res.exit_code == 0, res.output
    with mrcfile.open(str(out)) as mrc:
        assert mrc.data.dtype == np.uint8
        assert int(mrc.data.sum()) == 27
        assert float(mrc.voxel_size.x) == pytest.approx(2.5)
        assert float(mrc.header.origin.y) == pytest.approx(2.0)

    # Existing output is skipped unless --overwrite.
    res = CliRunner().invoke(pick_granules, [
        str(tmp_path / "tomo.mrc"), str(tmp_path / "mito.mrc"), "-o", str(out),
        "-t", "0", "--no-gui"])
    assert res.exit_code == 0 and "[skip]" in res.output
    res = CliRunner().invoke(pick_granules, [
        str(tmp_path / "tomo.mrc"), str(tmp_path / "mito.mrc"), "-o", str(out),
        "-t", "0", "--no-gui", "--overwrite"])
    assert res.exit_code == 0, res.output
    with mrcfile.open(str(out)) as mrc:
        assert int(mrc.data.sum()) == 28


def test_cli_rejects_threshold_and_percentile(tmp_path):
    tomo, mask = _phantom()
    _write(tmp_path / "tomo.mrc", tomo)
    _write(tmp_path / "mito.mrc", mask)
    res = CliRunner().invoke(pick_granules, [
        str(tmp_path / "tomo.mrc"), str(tmp_path / "mito.mrc"),
        "-t", "0", "-p", "5", "--no-gui"])
    assert res.exit_code != 0


def test_underscore_alias_hidden():
    res = CliRunner().invoke(tomo_toolshed, ["--help"])
    assert "pick-granules" in res.output and "pick_granules" not in res.output
    res = CliRunner().invoke(tomo_toolshed, ["pick_granules", "--help"])
    assert res.exit_code == 0


def test_gui_builds_and_updates_headless():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from tomo_toolshed.granule_picker.gui import GranulePickerGUI

    tomo, mask = _phantom()
    gui = GranulePickerGUI(tomo, mask, threshold=0.0)
    gui._set_threshold(5.0)          # outside the initial slider range
    assert gui.threshold == 5.0
    gui.tb_erode.set_val("2")
    gui._on_apply_erode(None)
    gui._on_minsize("3")
    gui._on_toggle_erode_3d(None)
    gui._on_apply_erode(None)
    assert gui.params() == {"threshold": 5.0, "mask_erode": 2, "erode_3d": True,
                            "min_size": 3}
    assert int(gui.mask.sum()) == 4 * 12 * 12
    gui._plt.close(gui.fig)
