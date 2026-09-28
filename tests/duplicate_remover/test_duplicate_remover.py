"""Tests for duplicate-remover.

Two layers:

- **Detection / I-O** over the four real star files in this directory (read-only;
  everything is written into ``tmp_path``). These pin the flavor-specific
  behaviour that is the whole point of the tool -- especially the pixel size,
  which must be 6.65 for RELION 4 (not the stale 9.98 in its particles table) and
  2.11 for RELION 5 (not the 4.22 extracted-image scale).
- **Removal math** on tiny synthetic DataFrames, where the expected keep/remove
  outcome is obvious by construction.

The suite is headless: ``core`` never imports matplotlib, and no test triggers the
histogram. The two large files are only read for detection and a threshold-0
round-trip; the removal math never touches them.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import starfile
from click.testing import CliRunner

from tomo_toolshed.duplicate_remover import core
from tomo_toolshed.duplicate_remover.cli import duplicate_remover

DATA = Path(__file__).parent
FILES = {
    "relion3": DATA / "relion_3_example.star",
    "relion4": DATA / "relion_4_example.star",
    "relion5": DATA / "relion_5_2D_example.star",
    "m": DATA / "mtools_example.star",
}


# ---------------------------------------------------------------------------
# Detection over the four real files.
# ---------------------------------------------------------------------------
# (flavor, group_column, n_groups, pixel_size, origins_detected)
# NOTE: the RELION 5 file provided has 5 rlnTomoName groups (4482 particles).
DETECTION = [
    ("relion3", "rlnMicrographName", 1, 9.98, False),
    ("relion4", "rlnMicrographName", 46, 6.65, True),
    ("relion5", "rlnTomoName", 5, 2.11, True),
    ("m", "wrpSourceName", 5, 1.0, False),
]


@pytest.mark.parametrize("flavor,group_col,n_groups,apix,has_origins", DETECTION)
def test_detection(flavor, group_col, n_groups, apix, has_origins):
    blocks, key = core.read_star(str(FILES[flavor]))
    particles = blocks[key]

    assert core.detect_flavor(blocks, particles) == flavor
    assert core.detect_group_column(particles) == group_col
    assert particles[group_col].nunique() == n_groups

    pixel = core.resolve_pixel_size(flavor, blocks, particles)
    assert pixel.uniform
    assert pixel.value == pytest.approx(apix)

    origins, _, _ = core.resolve_origins(particles, pixel.apix)
    assert (origins is not None) == has_origins


def test_relion4_does_not_use_stale_particle_pixel_size():
    """RELION 4 particles carry a stale rlnPixelSize (9.98); the coordinate scale
    is the optics rlnImagePixelSize (6.65)."""
    blocks, key = core.read_star(str(FILES["relion4"]))
    assert "rlnPixelSize" in blocks[key].columns          # the trap is present
    assert blocks[key]["rlnPixelSize"].iloc[0] == pytest.approx(9.98)
    pixel = core.resolve_pixel_size("relion4", blocks, blocks[key])
    assert pixel.value == pytest.approx(6.65)


def test_relion5_uses_tilt_series_not_image_pixel_size():
    blocks, key = core.read_star(str(FILES["relion5"]))
    assert blocks["optics"]["rlnImagePixelSize"].iloc[0] == pytest.approx(4.22)
    pixel = core.resolve_pixel_size("relion5", blocks, blocks[key])
    assert pixel.value == pytest.approx(2.11)


def test_pixel_size_override_wins_except_for_m():
    blocks, key = core.read_star(str(FILES["relion4"]))
    pixel = core.resolve_pixel_size("relion4", blocks, blocks[key],
                                    pixel_size_override=3.0)
    assert pixel.value == pytest.approx(3.0)

    blocks, key = core.read_star(str(FILES["m"]))
    pixel = core.resolve_pixel_size("m", blocks, blocks[key], pixel_size_override=3.0)
    assert pixel.value == pytest.approx(1.0)      # ignored for M
    assert pixel.ignored_override


def test_core_does_not_import_matplotlib():
    import sys
    import importlib
    importlib.import_module("tomo_toolshed.duplicate_remover.core")
    assert "matplotlib" not in sys.modules


# ---------------------------------------------------------------------------
# Round-trip: threshold 0 changes nothing and preserves every block.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("flavor", ["relion3", "relion4", "relion5", "m"])
def test_roundtrip_threshold_zero_preserves_blocks_and_counts(flavor, tmp_path):
    src = FILES[flavor]
    out = tmp_path / "out.star"
    runner = CliRunner()
    result = runner.invoke(duplicate_remover, [
        "--star-path", str(src), "--threshold", "0",
        "--no-histogram", "-q", "--output-path", str(out),
    ])
    assert result.exit_code == 0, result.output

    src_blocks = starfile.read(str(src), always_dict=True)
    out_blocks = starfile.read(str(out), always_dict=True)

    # Same blocks, same order.
    assert list(out_blocks.keys()) == list(src_blocks.keys())
    key = core.particles_key(src_blocks)
    # Counts unchanged and columns preserved.
    assert len(out_blocks[key]) == len(src_blocks[key])
    assert list(out_blocks[key].columns) == list(src_blocks[key].columns)


def test_roundtrip_relion5_keeps_data_general(tmp_path):
    out = tmp_path / "out.star"
    core_blocks, key = core.read_star(str(FILES["relion5"]))
    result = core.process(core_blocks, key, threshold=0)
    kept = core.apply_keep(core_blocks, key, result.keep_mask)
    core.write_star(core_blocks, key, kept, str(out))
    back = starfile.read(str(out), always_dict=True)
    assert "general" in back
    assert back["general"] == {"rlnTomoSubTomosAre2DStacks": 1}


@pytest.mark.parametrize("flavor", ["relion3", "m"])
def test_roundtrip_single_unnamed_block_stays_unnamed(flavor, tmp_path):
    out = tmp_path / "out.star"
    blocks, key = core.read_star(str(FILES[flavor]))
    assert key == ""                                  # single unnamed block
    result = core.process(blocks, key, threshold=0)
    kept = core.apply_keep(blocks, key, result.keep_mask)
    core.write_star(blocks, key, kept, str(out))
    back = starfile.read(str(out), always_dict=True)
    assert list(back.keys()) == [""]


# ---------------------------------------------------------------------------
# Removal math on tiny synthetic data.
# ---------------------------------------------------------------------------
def _relion3_blocks(coords, metric=None, origins=None, micrograph="mic"):
    """A single-unnamed-block DataFrame with apix 1.0 (coords already in Å)."""
    coords = np.asarray(coords, dtype=float)
    df = pd.DataFrame({
        "rlnCoordinateX": coords[:, 0],
        "rlnCoordinateY": coords[:, 1],
        "rlnCoordinateZ": coords[:, 2],
        "rlnMicrographName": micrograph,
        "rlnPixelSize": 1.0,
    })
    if metric is not None:
        df["rlnLogLikeliContribution"] = metric
    if origins is not None:
        origins = np.asarray(origins, dtype=float)
        df["rlnOriginXAngst"] = origins[:, 0]
        df["rlnOriginYAngst"] = origins[:, 1]
        df["rlnOriginZAngst"] = origins[:, 2]
    return {"": df}, ""


def _kept_indices(blocks, key, **kw):
    result = core.process(blocks, key, **kw)
    return set(np.nonzero(result.keep_mask)[0].tolist()), result


def test_pair_within_threshold_keeps_higher_metric():
    blocks, key = _relion3_blocks([[0, 0, 0], [100, 0, 0]], metric=[10.0, 5.0])
    kept, _ = _kept_indices(blocks, key, threshold=140)
    assert kept == {0}                                 # p0 has higher metric


def test_pair_outside_threshold_keeps_both():
    blocks, key = _relion3_blocks([[0, 0, 0], [200, 0, 0]], metric=[10.0, 5.0])
    kept, _ = _kept_indices(blocks, key, threshold=140)
    assert kept == {0, 1}


def test_pair_exactly_at_threshold_is_kept():
    """Threshold is exclusive (distance < threshold), so distance == threshold
    keeps both particles."""
    blocks, key = _relion3_blocks([[0, 0, 0], [140, 0, 0]], metric=[10.0, 5.0])
    kept, _ = _kept_indices(blocks, key, threshold=140)
    assert kept == {0, 1}


def test_cluster_of_three_collapses_to_the_best():
    # positions 50 Å apart; all pairwise within 140. Highest metric is the middle.
    blocks, key = _relion3_blocks([[0, 0, 0], [50, 0, 0], [100, 0, 0]],
                                  metric=[1.0, 3.0, 2.0])
    kept, _ = _kept_indices(blocks, key, threshold=140)
    assert kept == {1}                                 # the max-metric particle


def test_removed_member_does_not_remove_a_second_particle():
    """Over-removal guard: a already-removed particle must not cause another to be
    dropped. a-b within, b-c within, a-c NOT within; keeping only b's removal
    should not also drop c via the already-removed a."""
    blocks, key = _relion3_blocks([[0, 0, 0], [100, 0, 0], [200, 0, 0]],
                                  metric=[3.0, 1.0, 2.0])
    # pairs within 140: (0,1) d=100, (1,2) d=100; (0,2) d=200 not within.
    kept, _ = _kept_indices(blocks, key, threshold=140)
    # b (idx1, worst) conflicts with both -> removed; a and c never conflict -> kept.
    assert kept == {0, 2}


def test_origin_shifts_change_which_pairs_are_within_threshold():
    coords = [[0, 0, 0], [100, 0, 0]]
    # Without origins: distance 100 < 140 -> one removed.
    blocks, key = _relion3_blocks(coords, metric=[10.0, 5.0])
    kept_no, _ = _kept_indices(blocks, key, threshold=140)
    assert kept_no == {0}

    # Origins push p1 to x=200 (pos = coord - origin): origin_x for p1 = -100.
    blocks, key = _relion3_blocks(coords, metric=[10.0, 5.0],
                                  origins=[[0, 0, 0], [-100, 0, 0]])
    result = core.process(blocks, key, threshold=140)
    assert result.origins_applied
    assert set(np.nonzero(result.keep_mask)[0]) == {0, 1}   # now 200 apart -> both kept


def test_partial_origins_warn_and_treat_missing_as_zero():
    coords = [[0, 0, 0], [100, 0, 0]]
    blocks, key = _relion3_blocks(coords, metric=[10.0, 5.0])
    blocks[key]["rlnOriginXAngst"] = [0.0, -100.0]      # only X present
    result = core.process(blocks, key, threshold=140)
    assert result.origins_applied
    assert any("missing" in w for w in result.warnings)
    assert set(np.nonzero(result.keep_mask)[0]) == {0, 1}


def test_group_with_one_particle_is_untouched():
    df = pd.DataFrame({
        "rlnCoordinateX": [0.0, 1.0], "rlnCoordinateY": [0.0, 0.0],
        "rlnCoordinateZ": [0.0, 0.0], "rlnPixelSize": 1.0,
        "rlnMicrographName": ["a", "b"],       # two groups of one each
    })
    result = core.process({"": df}, "", threshold=140)
    assert result.keep_mask.all()


# ---------------------------------------------------------------------------
# Units and metric selection.
# ---------------------------------------------------------------------------
def test_units_pixels_matches_equivalent_angstrom_on_relion4():
    blocks, key = core.read_star(str(FILES["relion4"]))
    ang = core.process(blocks, key, threshold=140, units="angstroms")
    blocks, key = core.read_star(str(FILES["relion4"]))
    px = core.process(blocks, key, threshold=140 / 6.65, units="pixels")
    assert ang.n_removed == px.n_removed
    assert px.threshold_angstrom == pytest.approx(140, rel=1e-6)


def test_units_pixels_on_m_is_an_error():
    blocks, key = core.read_star(str(FILES["m"]))
    with pytest.raises(core.DuplicateRemoverError) as exc:
        core.process(blocks, key, threshold=30, units="pixels")
    assert "M/WarpTools" in str(exc.value)


def test_auto_metric_falls_back_to_first_when_absent():
    blocks, key = _relion3_blocks([[0, 0, 0], [10, 0, 0]])   # no metric column
    name, reason = core.resolve_metric(blocks[key].columns, "auto")
    assert name == core.METRIC_FIRST
    kept, _ = _kept_indices(blocks, key, threshold=140)
    assert kept == {0}                                        # keep-first


def test_explicit_missing_metric_is_an_error():
    with pytest.raises(core.DuplicateRemoverError):
        core.resolve_metric(["rlnCoordinateX"], "rlnLogLikeliContribution")


def test_random_metric_is_reproducible_with_seed():
    coords = [[0, 0, 0], [10, 0, 0], [1000, 0, 0], [1010, 0, 0]]
    blocks, key = _relion3_blocks(coords)
    a = core.process(blocks, key, threshold=140,
                     comparison_metric="random", seed=7)
    blocks, key = _relion3_blocks(coords)
    b = core.process(blocks, key, threshold=140,
                     comparison_metric="random", seed=7)
    assert np.array_equal(a.keep_mask, b.keep_mask)


# ---------------------------------------------------------------------------
# CLI plumbing.
# ---------------------------------------------------------------------------
def test_default_output_path_uses_pathlib_not_str_replace(tmp_path):
    # A directory containing '.star' must not be corrupted.
    d = tmp_path / "my.star.project"
    d.mkdir()
    src = d / "run_data.star"
    _relion3_blocks([[0, 0, 0]], metric=[1.0])   # build a df to write
    df = pd.DataFrame({"rlnCoordinateX": [0.0], "rlnCoordinateY": [0.0],
                       "rlnCoordinateZ": [0.0], "rlnPixelSize": [1.0],
                       "rlnMicrographName": ["a"]})
    starfile.write({"": df}, str(src), overwrite=True)

    runner = CliRunner()
    result = runner.invoke(duplicate_remover, [
        "--star-path", str(src), "--threshold", "10", "--no-histogram", "-q",
    ])
    assert result.exit_code == 0, result.output
    # The directory's '.star' survives; only the file stem gets _cleaned.
    assert (d / "run_data_cleaned.star").exists()


def test_underscored_option_aliases_still_work(tmp_path):
    src = FILES["m"]
    out = tmp_path / "out.star"
    runner = CliRunner()
    result = runner.invoke(duplicate_remover, [
        "--star_path", str(src), "--distance_threshold", "0",
        "--no-histogram", "-q", "--output_path", str(out),
    ])
    assert result.exit_code == 0, result.output
    assert out.exists()


def test_dry_run_writes_nothing(tmp_path):
    src = FILES["relion5"]
    out = tmp_path / "out.star"
    runner = CliRunner()
    result = runner.invoke(duplicate_remover, [
        "--star-path", str(src), "--threshold", "140", "--dry-run",
        "--no-histogram", "-q", "--output-path", str(out),
    ])
    assert result.exit_code == 0, result.output
    assert not out.exists()
