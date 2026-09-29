"""Tests for filament-cleanup: position and orientation outlier detection on
helical-refinement star files.

Synthetic filaments are used where the right answer is known by construction;
``microtubule_for_cleaning.star`` (a real RELION 4 / Warp microtubule refinement)
is used as a smoke test of the full CLI.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import starfile
from click.testing import CliRunner

from tomo_toolshed.filament_cleanup import core
from tomo_toolshed.filament_cleanup.cli import default_rejected_path, filament_cleanup

DATA = Path(__file__).parent / "microtubule_for_cleaning.star"
APIX = 10.0
STEP = 40.0  # Å between particles


def _axis_to_tilt_psi(v):
    """Inverse of core.helical_axis for a unit vector."""
    v = np.asarray(v, dtype=float) / np.linalg.norm(v)
    tilt = np.degrees(np.arccos(np.clip(v[2], -1, 1)))
    psi = np.degrees(np.arctan2(v[1], -v[0]))
    return tilt, psi


def _filament(n=40, curve=0.0, tube=1, tomo="tomo_A", start=(1000.0, 1000.0, 500.0)):
    """Particles along a (optionally gently curved) line in the XY plane, with
    tilt/psi matching the local tangent. Coordinates in pixels at APIX."""
    s = np.arange(n) * STEP
    x = start[0] + s
    y = start[1] + curve * (s - s.mean()) ** 2
    z = np.full(n, start[2])
    tx, ty = np.ones(n), 2 * curve * (s - s.mean())
    rows = []
    for i in range(n):
        tilt, psi = _axis_to_tilt_psi([tx[i], ty[i], 0.0])
        rows.append({
            "rlnCoordinateX": x[i] / APIX, "rlnCoordinateY": y[i] / APIX,
            "rlnCoordinateZ": z[i] / APIX, "rlnMicrographName": tomo,
            "rlnPixelSize": APIX, "rlnAngleRot": float(i * 30 % 360 - 180),
            "rlnAngleTilt": tilt, "rlnAnglePsi": psi, "rlnHelicalTubeID": tube,
            "rlnHelicalTrackLengthAngst": s[i], "rlnOpticsGroup": 1,
            "rlnOriginXAngst": 0.0, "rlnOriginYAngst": 0.0, "rlnOriginZAngst": 0.0,
        })
    return pd.DataFrame(rows)


def _blocks(particles):
    optics = pd.DataFrame({"rlnOpticsGroup": [1], "rlnOpticsGroupName": ["g1"],
                           "rlnImagePixelSize": [APIX], "rlnImageSize": [64],
                           "rlnImageDimensionality": [3]})
    return {"optics": optics, "particles": particles.reset_index(drop=True)}


def _run(args):
    return CliRunner().invoke(filament_cleanup, args)


# ---------------------------------------------------------------------------
# Geometry helpers.
# ---------------------------------------------------------------------------
def test_helical_axis_matches_relion_third_column():
    # tilt=90, psi=90 -> +Y ; tilt=0 -> +Z ; polarity flip negates the axis.
    np.testing.assert_allclose(core.helical_axis([90], [90])[0], [0, 1, 0], atol=1e-12)
    np.testing.assert_allclose(core.helical_axis([0], [37])[0], [0, 0, 1], atol=1e-12)
    a = core.helical_axis([80], [30])[0]
    b = core.helical_axis([100], [30 - 180])[0]
    np.testing.assert_allclose(a, -b, atol=1e-12)


def test_smooth_curve_follows_a_curved_filament():
    p = _filament(n=60, curve=2e-4)
    pos = p[["rlnCoordinateX", "rlnCoordinateY", "rlnCoordinateZ"]].to_numpy() * APIX
    _, _, resid = core.robust_smooth_curve(core.arclength(pos), pos, window=500)
    assert resid.max() < 2.0


def test_smooth_curve_is_robust_to_an_outlier():
    p = _filament(n=40)
    pos = p[["rlnCoordinateX", "rlnCoordinateY", "rlnCoordinateZ"]].to_numpy() * APIX
    picked = pos.copy()
    pos[20, 1] += 80.0
    _, _, resid = core.robust_smooth_curve(core.arclength(picked), pos, window=500)
    assert resid[20] > 70.0
    assert np.delete(resid, 20).max() < 5.0


def test_neighbour_axis_deviation_ignores_polarity():
    axes = np.tile([1.0, 0.0, 0.0], (10, 1))
    axes[3] *= -1                                  # flipped: same line
    axes[6] = [np.cos(np.radians(40)), np.sin(np.radians(40)), 0]
    dev = core.neighbour_axis_deviation(axes, 3)
    assert dev[3] < 1e-6
    assert abs(dev[6] - 40) < 1e-6


# ---------------------------------------------------------------------------
# clean_filaments.
# ---------------------------------------------------------------------------
def test_clean_filament_keeps_everything():
    blocks = _blocks(pd.concat([_filament(), _filament(curve=2e-4, tube=2)]))
    res = core.clean_filaments(blocks, "particles")
    assert res.n_removed == 0
    assert res.info["flavor"] == "relion4"


def test_perpendicular_offset_flagged_but_axial_slide_is_not():
    p = _filament()
    p.loc[10, "rlnOriginYAngst"] = -60.0   # 60 Å sideways (refined pos = coord - origin)
    p.loc[25, "rlnOriginXAngst"] = -20.0   # 20 Å along the axis: allowed
    res = core.clean_filaments(_blocks(p), "particles", check_orientation=False)
    assert np.flatnonzero(~res.keep_mask).tolist() == [10]
    assert res.reasons[10] == core.REASON_POSITION
    assert res.distance[25] < 5.0


def test_misoriented_particle_flagged():
    p = _filament()
    tilt, psi = _axis_to_tilt_psi([np.cos(np.radians(45)), np.sin(np.radians(45)), 0])
    p.loc[15, ["rlnAngleTilt", "rlnAnglePsi"]] = [tilt, psi]
    res = core.clean_filaments(_blocks(p), "particles")
    assert np.flatnonzero(~res.keep_mask).tolist() == [15]
    assert res.reasons[15] == core.REASON_ORIENTATION
    assert abs(res.angle[15] - 45) < 1.0


def test_polarity_flips_kept_by_default_and_removed_on_request():
    p = _filament()
    for i in (5, 12, 30):
        p.loc[i, "rlnAngleTilt"] = 180 - p.loc[i, "rlnAngleTilt"]
        p.loc[i, "rlnAnglePsi"] = p.loc[i, "rlnAnglePsi"] - 180
    assert core.clean_filaments(_blocks(p), "particles").n_removed == 0
    res = core.clean_filaments(_blocks(p), "particles", remove_flipped=True)
    assert np.flatnonzero(~res.keep_mask).tolist() == [5, 12, 30]
    assert (res.reasons[[5, 12, 30]] == core.REASON_FLIPPED).all()


def test_filaments_are_grouped_per_tomogram_and_tube():
    # Same tube ID in two tomograms and a far-away second tube: none may leak
    # into another filament's fit.
    p = pd.concat([_filament(tomo="A"), _filament(tomo="B", start=(1000, 5000, 500)),
                   _filament(tomo="A", tube=2, start=(1000, 3000, 900))])
    res = core.clean_filaments(_blocks(p), "particles")
    assert res.n_removed == 0
    assert len(res.filaments) == 3


def test_short_filament_left_alone():
    p = _filament(n=4)
    p.loc[2, "rlnOriginYAngst"] = -200.0
    res = core.clean_filaments(_blocks(p), "particles", min_particles=5)
    assert res.n_removed == 0
    assert res.filaments[0]["skipped"]


def test_order_follows_track_length_not_file_order():
    p = _filament().sample(frac=1.0, random_state=0)
    p.loc[p.index == 10, "rlnOriginYAngst"] = -60.0
    res = core.clean_filaments(_blocks(p), "particles")
    removed = res.keep_mask == False  # noqa: E712
    assert p.index[removed].tolist() == [10]


def test_missing_helical_columns_is_an_error():
    p = _filament().drop(columns=["rlnHelicalTubeID"])
    with pytest.raises(core.FilamentCleanupError, match="rlnHelicalTubeID"):
        core.clean_filaments(_blocks(p), "particles")


def test_relion5_centered_angstrom_coordinates():
    p = _filament()
    p["rlnCenteredCoordinateXAngst"] = p.pop("rlnCoordinateX") * APIX
    p["rlnCenteredCoordinateYAngst"] = p.pop("rlnCoordinateY") * APIX
    p["rlnCenteredCoordinateZAngst"] = p.pop("rlnCoordinateZ") * APIX
    p["rlnTomoName"] = p.pop("rlnMicrographName")
    p.loc[10, "rlnOriginYAngst"] = -60.0
    blocks = _blocks(p)
    blocks["optics"]["rlnTomoTiltSeriesPixelSize"] = APIX
    res = core.clean_filaments(blocks, "particles")
    assert res.info["flavor"] == "relion5"
    assert np.flatnonzero(~res.keep_mask).tolist() == [10]


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------
def test_cli_writes_cleaned_and_rejected(tmp_path):
    p = _filament()
    p.loc[10, "rlnOriginYAngst"] = -60.0
    src = tmp_path / "in.star"
    starfile.write(_blocks(p), src)
    out = tmp_path / "sub" / "clean.star"
    r = _run(["--i", str(src), "--o", str(out)])
    assert r.exit_code == 0, r.output
    rej = Path(default_rejected_path(out))
    assert rej.name == "clean_rejected.star"
    kept = starfile.read(out, always_dict=True)
    gone = starfile.read(rej, always_dict=True)
    assert list(kept) == list(gone) == ["optics", "particles"]
    assert len(kept["particles"]) == 39 and len(gone["particles"]) == 1
    assert list(kept["particles"].columns) == list(p.columns)
    assert "filament-cleanup" in rej.read_text().splitlines()[0]


def test_cli_dry_run_writes_nothing(tmp_path):
    src = tmp_path / "in.star"
    starfile.write(_blocks(_filament()), src)
    r = _run(["--i", str(src), "--o", str(tmp_path / "o.star"), "--dry-run"])
    assert r.exit_code == 0, r.output
    assert "dry run" in r.output
    assert not (tmp_path / "o.star").exists()


def test_cli_refuses_overwriting_input(tmp_path):
    src = tmp_path / "in.star"
    starfile.write(_blocks(_filament()), src)
    r = _run(["--i", str(src), "--o", str(src)])
    assert r.exit_code != 0
    r = _run(["--i", str(src), "--o", str(tmp_path / "o.star"),
              "--rejected", str(tmp_path / "o.star")])
    assert r.exit_code != 0


def test_cli_real_microtubule_file(tmp_path):
    out = tmp_path / "mt_clean.star"
    rej = tmp_path / "mt_bad.star"
    r = _run(["--i", str(DATA), "--o", str(out), "--rejected", str(rej)])
    assert r.exit_code == 0, r.output
    assert "rlnImagePixelSize" in r.output  # RELION 4: stale rlnPixelSize ignored
    src = starfile.read(DATA, always_dict=True)["particles"]
    kept = starfile.read(out, always_dict=True)["particles"]
    gone = starfile.read(rej, always_dict=True)["particles"]
    assert len(kept) + len(gone) == len(src)
    assert 0 < len(gone) < 0.3 * len(src)
    names = set(kept["rlnImageName"]) | set(gone["rlnImageName"])
    assert names == set(src["rlnImageName"])
