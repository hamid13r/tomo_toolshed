"""Tests for dipole2star: pair collapsing, orientation, and the two readers.

Fixtures are small hand-made pick files written into ``tmp_path`` (no binaries
committed). The orientation math is exercised by rebuilding the rotation from the
written Euler angles and checking it maps the reference axis onto the pick axis.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import starfile
from click.testing import CliRunner
from scipy.spatial.transform import Rotation as R

from tomo_toolshed.dipole2star import core
from tomo_toolshed.dipole2star.cli import dipole2star


def _star_payload(path):
    """Star content minus the ``# Created by ... at <timestamp>`` comment line.

    The ``starfile`` writer stamps the wall-clock time into a leading comment, so
    two runs are never byte-identical across a second boundary. Comparing the
    non-comment lines is the meaningful notion of "identical star files".
    """
    return "\n".join(
        line for line in Path(path).read_text().splitlines()
        if not line.lstrip().startswith("#")
    )


def _write_star(path, coords):
    df = pd.DataFrame(coords, columns=['rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ'])
    starfile.write(df.astype(float), str(path), overwrite=True)
    return path


# Two particles: one axis-aligned along x, one oblique.
COORDS = np.array([
    [10, 20, 30], [14, 20, 30],       # vector (1, 0, 0)
    [0, 0, 0], [3, 4, 12],            # vector (3, 4, 12)/13
], dtype=float)


# ---------------------------------------------------------------------------
# Geometry: pair collapsing, centering, orientation.
# ---------------------------------------------------------------------------
def test_pair_collapsing_and_centering():
    rows = core.picks_to_particles(COORDS, scale=2.0, random=False,
                                   pixel_size=9.98, source="x")
    assert len(rows) == 2
    # center = midpoint * scale, truncated to whole numbers, stored as float.
    assert (rows[0]['rlnCoordinateX'], rows[0]['rlnCoordinateY'],
            rows[0]['rlnCoordinateZ']) == (24.0, 40.0, 60.0)
    assert (rows[1]['rlnCoordinateX'], rows[1]['rlnCoordinateY'],
            rows[1]['rlnCoordinateZ']) == (3.0, 4.0, 12.0)
    assert all(isinstance(rows[0][k], float) for k in
               ('rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ'))


@pytest.mark.parametrize("p0,p1", [
    ([10, 20, 30], [14, 20, 30]),     # axis-aligned (x)
    ([0, 0, 0], [3, 4, 12]),          # oblique
    ([5, 5, 5], [5, 9, 5]),           # axis-aligned (y)
])
def test_rotation_maps_reference_axis_onto_pick_vector(p0, p1):
    coords = np.array([p0, p1], dtype=float)
    rows = core.picks_to_particles(coords, scale=1.0, random=False,
                                   pixel_size=1.0, source="x")
    rot, tilt, psi = (rows[0]['rlnAngleRot'], rows[0]['rlnAngleTilt'],
                      rows[0]['rlnAnglePsi'])
    rotation = R.from_euler('ZYZ', [rot, tilt, psi], degrees=True)

    vec = np.array(p1, float) - np.array(p0, float)
    vec /= np.linalg.norm(vec)
    # align_vectors maps the pick axis onto [0,0,1]; the inverse maps back.
    assert np.allclose(rotation.inv().apply([0, 0, 1]), vec, atol=1e-6)


def test_z_parallel_axis_is_rejected():
    coords = np.array([[0, 0, 0], [0, 0, 5]], dtype=float)   # axis along z
    with pytest.raises(core.DipoleError) as exc:
        core.picks_to_particles(coords, scale=1.0, random=False,
                                pixel_size=1.0, source="picks.txt")
    assert "picks.txt" in str(exc.value)
    assert "pair 0" in str(exc.value)


# ---------------------------------------------------------------------------
# Readers: text variants, star-vs-txt parity, error cases.
# ---------------------------------------------------------------------------
def test_star_and_txt_produce_identical_output(tmp_path):
    txt = tmp_path / "foo.txt"
    txt.write_text("10 20 30\n14 20 30\n0 0 0\n3 4 12\n")
    star = tmp_path / "foo.star"
    _write_star(star, COORDS)

    txt_out = tmp_path / "txt_out"
    star_out = tmp_path / "star_out"
    txt_out.mkdir()
    star_out.mkdir()

    core.convert_one(str(txt), 2.0, False, 9.98, str(txt_out))
    core.convert_one(str(star), 2.0, False, 9.98, str(star_out))

    # Same stem "foo" -> identical rlnMicrographName, so payloads match fully.
    assert _star_payload(txt_out / "foo.mrc.star") == _star_payload(star_out / "foo.mrc.star")


@pytest.mark.parametrize("body", [
    "10 20 30\n14 20 30\n",                       # whitespace
    "10,20,30\n14,20,30\n",                       # comma
    "# comment\n10 20 30\n\n14 20 30\n",          # blank + comment lines
    "x y z\n10 20 30\n14 20 30\n",                # header row
    "x,y,z\n10, 20, 30\n14, 20, 30\n",            # header + comma
])
def test_text_reader_variants(tmp_path, body):
    p = tmp_path / "picks.txt"
    p.write_text(body)
    coords = core.read_coords(str(p))
    assert np.array_equal(coords, np.array([[10, 20, 30], [14, 20, 30]], float))


def test_odd_pick_count_is_an_error(tmp_path):
    p = tmp_path / "odd.txt"
    p.write_text("10 20 30\n14 20 30\n5 5 5\n")
    with pytest.raises(core.DipoleError) as exc:
        core.convert_one(str(p), 1.0, False, 9.98, str(tmp_path))
    assert "odd" in str(exc.value)
    assert "odd.txt" in str(exc.value)


def test_non_numeric_row_names_the_line(tmp_path):
    p = tmp_path / "bad.txt"
    p.write_text("10 20 30\n14 20 30\noops here now\n")
    with pytest.raises(core.DipoleError) as exc:
        core.read_coords(str(p))
    msg = str(exc.value)
    assert "bad.txt" in msg
    assert ":3" in msg   # the offending line number


def test_wrong_column_count_row_is_an_error(tmp_path):
    p = tmp_path / "short.txt"
    p.write_text("10 20 30\n14 20\n")
    with pytest.raises(core.DipoleError):
        core.read_coords(str(p))


# ---------------------------------------------------------------------------
# Columns reflect --scale and --output-apix.
# ---------------------------------------------------------------------------
def test_scale_and_pixel_size_reach_the_right_columns(tmp_path):
    p = tmp_path / "foo.txt"
    p.write_text("10 20 30\n14 20 30\n")
    core.convert_one(str(p), 3.0, False, 4.56, str(tmp_path))
    df = starfile.read(str(tmp_path / "foo.mrc.star"))
    # center midpoint (12,20,30) * scale 3 = (36,60,90), truncated.
    assert float(df['rlnCoordinateX'].iloc[0]) == 36.0
    assert float(df['rlnCoordinateY'].iloc[0]) == 60.0
    assert float(df['rlnCoordinateZ'].iloc[0]) == 90.0
    assert float(df['rlnPixelSize'].iloc[0]) == 4.56
    assert float(df['rlnMagnification'].iloc[0]) == 10000
    assert int(df['rlnGroupNumber'].iloc[0]) == 1


def test_micrograph_suffix_is_configurable(tmp_path):
    p = tmp_path / "foo.txt"
    p.write_text("10 20 30\n14 20 30\n")
    core.convert_one(str(p), 1.0, False, 9.98, str(tmp_path),
                     micrograph_suffix="_bin4.mrc")
    df = starfile.read(str(tmp_path / "foo.mrc.star"))
    assert df['rlnMicrographName'].iloc[0] == "foo_bin4.mrc"


# ---------------------------------------------------------------------------
# --random reproducibility.
# ---------------------------------------------------------------------------
def test_random_same_seed_is_reproducible(tmp_path):
    p = tmp_path / "foo.txt"
    p.write_text("10 20 30\n14 20 30\n0 0 0\n3 4 12\n")
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    core.convert_one(str(p), 1.0, True, 9.98, str(a), rng=np.random.default_rng(7))
    core.convert_one(str(p), 1.0, True, 9.98, str(b), rng=np.random.default_rng(7))
    assert _star_payload(a / "foo.mrc.star") == _star_payload(b / "foo.mrc.star")


def test_random_differs_from_default(tmp_path):
    p = tmp_path / "foo.txt"
    p.write_text("10 20 30\n14 20 30\n0 0 0\n3 4 12\n")
    plain = tmp_path / "plain"
    rand = tmp_path / "rand"
    plain.mkdir()
    rand.mkdir()
    core.convert_one(str(p), 1.0, False, 9.98, str(plain))
    core.convert_one(str(p), 1.0, True, 9.98, str(rand), rng=np.random.default_rng(7))
    assert _star_payload(plain / "foo.mrc.star") != _star_payload(rand / "foo.mrc.star")


# ---------------------------------------------------------------------------
# CLI: glob expansion, --outdir, seed logging, smoke test.
# ---------------------------------------------------------------------------
def test_cli_glob_expansion_and_outdir(tmp_path):
    (tmp_path / "a.txt").write_text("10 20 30\n14 20 30\n")
    (tmp_path / "b.txt").write_text("0 0 0\n3 4 12\n")
    outdir = tmp_path / "out"
    outdir.mkdir()

    runner = CliRunner()
    result = runner.invoke(dipole2star, [
        str(tmp_path / "*.txt"),
        "--scale", "2", "--output-apix", "9.98", "--outdir", str(outdir),
    ])
    assert result.exit_code == 0, result.output
    assert (outdir / "a.mrc.star").exists()
    assert (outdir / "b.mrc.star").exists()


def test_cli_smoke_star_input(tmp_path):
    star = tmp_path / "foo.star"
    _write_star(star, COORDS)
    runner = CliRunner()
    result = runner.invoke(dipole2star, [
        str(star), "--scale", "2", "--output-apix", "9.98", "--outdir", str(tmp_path),
    ])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "foo.mrc.star").exists()
    assert "2 particles" in result.output


def test_cli_random_draws_and_prints_a_seed_when_omitted(tmp_path):
    p = tmp_path / "foo.txt"
    p.write_text("10 20 30\n14 20 30\n")
    runner = CliRunner()
    result = runner.invoke(dipole2star, [
        str(p), "--scale", "2", "--output-apix", "9.98",
        "--outdir", str(tmp_path), "--random",
    ])
    assert result.exit_code == 0, result.output
    assert "random: drew seed" in result.output


def test_cli_odd_pick_count_is_a_clean_error(tmp_path):
    p = tmp_path / "odd.txt"
    p.write_text("10 20 30\n14 20 30\n5 5 5\n")
    runner = CliRunner()
    result = runner.invoke(dipole2star, [
        str(p), "--scale", "2", "--output-apix", "9.98", "--outdir", str(tmp_path),
    ])
    assert result.exit_code != 0
    assert "odd" in result.output
