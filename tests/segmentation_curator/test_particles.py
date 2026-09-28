"""Headless tests for the curator's star-file (particle) mode."""

import numpy as np
import pandas as pd
import pytest
import starfile
from click.testing import CliRunner

from tomo_toolshed.segmentation_curator import io as mrc_io
from tomo_toolshed.segmentation_curator import particles as P
from tomo_toolshed.segmentation_curator.cli import main as curate


def _df(xyz, **extra):
    xyz = np.asarray(xyz, dtype=float)
    d = {"rlnCoordinateX": xyz[:, 0], "rlnCoordinateY": xyz[:, 1],
         "rlnCoordinateZ": xyz[:, 2]}
    d.update(extra)
    return pd.DataFrame(d)


def _write_star(path, df, optics=True):
    if optics:
        blocks = {"optics": pd.DataFrame({"rlnOpticsGroup": [1],
                                          "rlnImagePixelSize": [10.0]}),
                  "particles": df}
    else:
        blocks = df
    starfile.write(blocks, str(path), overwrite=True)
    return path


# ------------------------------------------------------------------ geometry
def test_paint_spheres_one_label_per_particle():
    centers = np.array([[10, 10, 10], [10, 10, 30]], dtype=float)   # z,y,x
    lab = P.paint_spheres((20, 20, 40), centers, 3)
    assert set(np.unique(lab)) == {0, 1, 2}
    assert lab[10, 10, 10] == 1 and lab[10, 10, 30] == 2
    n1 = int((lab == 1).sum())
    assert n1 == int((lab == 2).sum())
    assert abs(n1 - 4 / 3 * np.pi * 27) < 0.25 * n1      # ~ sphere volume


def test_overlapping_spheres_split_by_nearest_center():
    centers = np.array([[10, 10, 10], [10, 10, 14]], dtype=float)
    lab = P.paint_spheres((20, 20, 30), centers, 5)
    assert lab[10, 10, 11] == 1 and lab[10, 10, 13] == 2
    # both particles remain present and distinct (no merging)
    assert (lab == 1).any() and (lab == 2).any()


def test_particle_outside_volume_paints_nothing():
    centers = np.array([[5, 5, 5], [500, 5, 5]], dtype=float)
    lab = P.paint_spheres((10, 10, 10), centers, 2)
    assert set(np.unique(lab)) == {0, 1}


def test_centers_scaled_and_origin_shifted():
    df = _df([[40, 20, 10]], rlnOriginXAngst=[20.0], rlnOriginYAngst=[0.0],
             rlnOriginZAngst=[0.0])
    # coords at 5 Å/px, tomogram at 10 Å/px -> halve; origin 20 Å = 4 px at 5 Å/px
    c = P.particle_centers_zyx(df, tomo_pixel_size=10.0, coord_pixel_size=5.0)
    np.testing.assert_allclose(c, [[5.0, 10.0, 18.0]])        # z, y, x


def test_centers_default_to_tomogram_pixels():
    c = P.particle_centers_zyx(_df([[1, 2, 3]]), tomo_pixel_size=7.0)
    np.testing.assert_allclose(c, [[3, 2, 1]])


def test_ids_in_zrange():
    centers = np.array([[1, 0, 0], [5, 0, 0], [9, 0, 0]], dtype=float)
    assert P.ids_in_zrange(centers, 2, 9) == {2, 3}


# ------------------------------------------------------------------ star I/O
def test_load_rejects_multiple_tomograms(tmp_path):
    df = _df([[1, 1, 1], [2, 2, 2]], rlnTomoName=["a", "b"])
    path = _write_star(tmp_path / "p.star", df)
    with pytest.raises(P.ParticleStarError, match="2 tomograms"):
        P.load_particles(str(path))


def test_load_rejects_relion5_coordinates(tmp_path):
    df = pd.DataFrame({"rlnCenteredCoordinateXAngst": [0.0],
                       "rlnCenteredCoordinateYAngst": [0.0],
                       "rlnCenteredCoordinateZAngst": [0.0]})
    path = _write_star(tmp_path / "p.star", df)
    with pytest.raises(P.ParticleStarError, match="RELION 5"):
        P.load_particles(str(path))


@pytest.mark.parametrize("optics", [True, False])
def test_write_kept_particles_drops_rows_keeps_blocks(tmp_path, optics):
    df = _df([[1, 1, 1], [2, 2, 2], [3, 3, 3]], rlnTomoName=["t", "t", "t"],
             rlnClassNumber=[1, 2, 3])
    src = _write_star(tmp_path / "in.star", df, optics=optics)
    blocks, key, loaded = P.load_particles(str(src))
    out = tmp_path / "out.star"
    assert P.write_kept_particles(blocks, key, loaded, {1, 3}, str(out)) == 2
    back = starfile.read(str(out), always_dict=True)
    assert list(back[key]["rlnClassNumber"]) == [1, 3]
    assert list(back[key].columns) == list(df.columns)
    if optics:
        assert "optics" in back


# ------------------------------------------------------------------ CLI
def _setup_cli(tmp_path):
    tomo = np.zeros((20, 20, 20), dtype=np.float32)
    tomo_path = tmp_path / "tomo.mrc"
    mrc_io.write_mrc(str(tomo_path), tomo, voxel_size=10.0)
    df = _df([[5, 5, 5], [15, 15, 15], [5, 15, 10]], rlnTomoName=["tomo"] * 3)
    star_path = _write_star(tmp_path / "picks.star", df)
    return tomo_path, star_path


def test_cli_star_mode_writes_filtered_star(tmp_path, monkeypatch):
    from tomo_toolshed.segmentation_curator import gui

    captured = {}

    class FakeGUI:
        def __init__(self, **kw):
            captured.update(kw)
            self.selected = set(kw["selected"])

        def run(self):
            self.selected.discard(2)          # "reject" particle 2
            return None

    monkeypatch.setattr(gui, "SegmentationCuratorGUI", FakeGUI)
    tomo_path, star_path = _setup_cli(tmp_path)
    out_dir = tmp_path / "out"
    res = CliRunner().invoke(curate, [str(tomo_path), str(star_path), str(out_dir),
                                      "--radius", "20"])
    assert res.exit_code == 0, res.output
    assert captured["pixel_size"] == pytest.approx(10.0)
    assert captured["n_islands"] == 3
    assert set(np.unique(captured["labels"])) == {0, 1, 2, 3}
    back = starfile.read(str(out_dir / "picks.star"), always_dict=True)
    assert len(back["particles"]) == 2
    assert "kept 2 of 3" in res.output


def test_cli_star_mode_requires_radius(tmp_path):
    tomo_path, star_path = _setup_cli(tmp_path)
    res = CliRunner().invoke(curate, [str(tomo_path), str(star_path),
                                      str(tmp_path / "out")])
    assert res.exit_code != 0
    assert "--radius" in res.output


def test_cli_star_options_rejected_for_mask_input(tmp_path):
    tomo_path, _ = _setup_cli(tmp_path)
    res = CliRunner().invoke(curate, [str(tomo_path), str(tomo_path),
                                      str(tmp_path / "out"), "--radius", "5",
                                      "--threshold", "0"])
    assert res.exit_code != 0
    assert "only apply to particle input" in res.output


# ------------------------------------------------------------------ xyz text
@pytest.mark.parametrize("ext", [".txt", ".box"])
def test_load_text_coordinates(tmp_path, ext):
    path = tmp_path / f"picks{ext}"
    path.write_text("# x y z\n1 2 3\n4.5\t5.5\t6.5\n\n7,8,9  # comma-separated\n")
    blocks, key, df = P.load_particles(str(path))
    assert key == "particles" and list(blocks) == ["particles"]
    assert list(df.columns) == list(P.COORD_COLUMNS)
    np.testing.assert_allclose(df.to_numpy(), [[1, 2, 3], [4.5, 5.5, 6.5], [7, 8, 9]])


def test_text_coordinates_wrong_column_count(tmp_path):
    path = tmp_path / "bad.txt"
    path.write_text("1 2 3\n4 5\n")
    with pytest.raises(P.ParticleStarError, match=":2: expected 3 columns"):
        P.load_particles(str(path))


def test_output_star_name():
    assert P.output_star_name("/a/picks.star") == "picks.star"
    assert P.output_star_name("/a/picks.txt") == "picks.star"
    assert P.output_star_name("/a/tomo_01.coords.box") == "tomo_01.coords.star"


def test_cli_text_input_writes_star(tmp_path, monkeypatch):
    from tomo_toolshed.segmentation_curator import gui

    class FakeGUI:
        def __init__(self, **kw):
            self.selected = set(kw["selected"])

        def run(self):
            self.selected.discard(1)          # reject the first pick
            return None

    monkeypatch.setattr(gui, "SegmentationCuratorGUI", FakeGUI)
    tomo_path, _ = _setup_cli(tmp_path)
    txt = tmp_path / "picks.txt"
    txt.write_text("5 5 5\n15 15 15\n5 15 10\n")
    out_dir = tmp_path / "out"
    res = CliRunner().invoke(curate, [str(tomo_path), str(txt), str(out_dir),
                                      "--radius", "20"])
    assert res.exit_code == 0, res.output
    back = starfile.read(str(out_dir / "picks.star"), always_dict=True)
    np.testing.assert_allclose(
        back["particles"][list(P.COORD_COLUMNS)].to_numpy(),
        [[15, 15, 15], [5, 15, 10]])
    assert not (out_dir / "picks.txt").exists()
