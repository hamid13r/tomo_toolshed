"""The etomo / novaCTF engines and the new CLI options, on the phantom from
validate.py (the upstream self-test script, kept runnable on its own)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

from tomo_toolshed.cli import tomo_toolshed


def _load_validate():
    path = Path(__file__).with_name("validate.py")
    spec = importlib.util.spec_from_file_location("xmlrec_validate", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


validate = _load_validate()


def test_novactf_engine_recovers_phantom():
    """Per-voxel geometry equals positions_in_all_tilts, and the real-space
    novaCTF back-projection recovers the phantom in the identity orientation.
    (The GPU comparison inside is skipped without torch + CUDA.)"""
    assert validate.test_novactf(verbose=False)


def test_etomo_engine_recovers_phantom(tmp_path):
    """End-to-end Warp -> IMOD conversion through the real newstack/tilt."""
    from tomo_toolshed.xml_reconstruct.etomo import _imod_env
    _env, imod_dir = _imod_env()
    if not (Path(imod_dir) / "bin" / "tilt").exists():
        pytest.skip(f"IMOD not found at {imod_dir}")
    assert validate.test_etomo(verbose=False, workdir=str(tmp_path / "etomo"))


def test_motioncor3_weighting_keeps_warp_scale():
    """--dose-weighting motioncor3 swaps only the dose model: the per-tilt
    amplitude scale stays Warp's, and each tilt carries its accumulated dose."""
    from tomo_toolshed.xml_reconstruct import (motioncor3_dose_weighting,
                                               warp_weighting)
    model = validate.make_model(8, 8, 8, 10.0, 16, 16)
    centre = model.volume_dims_A / 2.0
    defocus = np.zeros(model.n_tilts, np.float32)
    mc3 = motioncor3_dose_weighting(model, centre, defocus)
    warp = warp_weighting(model, centre, defocus)
    assert len(mc3) == model.n_tilts
    for t, (a, b) in enumerate(zip(mc3, warp)):
        assert a.scale == pytest.approx(b.scale)
        assert a.dose_model == "motioncor3"
        assert a.dose_ea2 == pytest.approx(float(model.dose[t]))


@pytest.mark.parametrize("flag", [
    "--engine", "--threads", "--renorm-variance", "--no-highpass",
    "--no-local-motion", "--weight-floor", "--ctf3d-defocus-step",
    "--ctf3d-num-strips", "--dose-weighting", "--etomo-recon",
    "--etomo-view-weight", "--novactf-step", "--novactf-device",
])
def test_xml_reconstruct_help_lists_new_options(flag):
    res = CliRunner().invoke(tomo_toolshed, ["xml-reconstruct", "--help"])
    assert res.exit_code == 0, res.output
    assert flag in res.output


def test_compare_tomograms_registered():
    res = CliRunner().invoke(tomo_toolshed, ["--help"])
    assert "compare-tomograms" in res.output
    res = CliRunner().invoke(tomo_toolshed, ["compare-tomograms", "--help"])
    assert res.exit_code == 0 and "--reference" in res.output
