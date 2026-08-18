import numpy as np

from island_curator import io as mrc_io


def test_mrc_roundtrip(tmp_path):
    vol = np.zeros((5, 6, 7), dtype=np.uint8)
    vol[1:3, 2:4, 3:5] = 1
    path = tmp_path / "roundtrip.mrc"
    mrc_io.write_mrc(str(path), vol)
    back = mrc_io.read_mrc(str(path))
    assert back.shape == vol.shape
    # MRC has no native unsigned-8-bit mode, so mrcfile may promote uint8 to
    # uint16 on write; the stored *values* are what must round-trip.
    np.testing.assert_array_equal(back.astype(np.uint8), vol)


def test_mrc_roundtrip_float(tmp_path):
    rng = np.random.default_rng(0)
    vol = rng.standard_normal((4, 4, 4)).astype(np.float32)
    path = tmp_path / "float.mrc"
    mrc_io.write_mrc(str(path), vol)
    back = mrc_io.read_mrc(str(path))
    np.testing.assert_allclose(back, vol, rtol=1e-6, atol=1e-6)
