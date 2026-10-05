"""Metadata parsing: the per-tilt-series XML is the source of truth."""
from __future__ import annotations

import numpy as np

from warp_recon import load_tiltseries_xml


def test_load_tiltseries_xml_parses_per_tilt_arrays(synthetic_series):
    xml_path, _stack, n = synthetic_series
    model = load_tiltseries_xml(str(xml_path))

    assert model.n_tilts == n
    assert model.angles.shape == (n,)
    assert np.isclose(model.angles[0], -40.0)
    assert np.isclose(model.angles[-1], 40.0)
    assert model.dose.shape == (n,)
    assert bool(model.use_tilt.all())
    # CTF base params read from the <CTF> node
    assert np.isclose(model.ctf.voltage, 300.0)
    assert np.isclose(model.ctf.cs, 2.7)
    # per-tilt defocus grid (GridCTF) carries one node per tilt at 3.0 um
    assert np.allclose(model.grid_ctf_defocus.values.ravel(), 3.0)
    # box dimensions in Angstrom
    assert np.allclose(model.volume_dims_A, [1600, 1600, 400])
