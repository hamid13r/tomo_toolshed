import numpy as np
import pytest

from island_curator import labeling


def make_volume():
    """Volume with three well-separated cubic islands of known sizes."""
    vol = np.zeros((10, 10, 10), dtype=np.uint8)
    vol[1:3, 1:3, 1:3] = 1   # 2x2x2 = 8 voxels
    vol[5:8, 5:8, 5:8] = 1   # 3x3x3 = 27 voxels
    vol[0:1, 8:9, 8:9] = 1   # 1 voxel (dust)
    return vol


def test_label_islands_count():
    vol = make_volume()
    labels, n = labeling.label_islands(vol, connectivity=26)
    assert n == 3
    assert labels.max() == 3
    assert set(np.unique(labels)) == {0, 1, 2, 3}


def test_compute_sizes():
    vol = make_volume()
    labels, n = labeling.label_islands(vol)
    _, sizes = labeling.compute_bboxes_and_sizes(labels, n)
    # sizes indexed by id; background at 0.
    island_sizes = sorted(int(s) for s in sizes[1:])
    assert island_sizes == [1, 8, 27]


def test_filter_by_size():
    vol = make_volume()
    labels, n = labeling.label_islands(vol)
    _, sizes = labeling.compute_bboxes_and_sizes(labels, n)
    kept = labeling.filter_by_size(sizes, min_size=8)
    # The 1-voxel dust island should be dropped.
    kept_sizes = sorted(int(sizes[i]) for i in kept)
    assert kept_sizes == [8, 27]


def test_apply_zrange_zeros_outside():
    vol = make_volume()
    cropped = labeling.apply_zrange(vol, z_min=4, z_max=9)
    # Islands at z=1..2 and the dust at z=0 are removed; z=5..7 island stays.
    assert cropped[0:4].sum() == 0
    assert cropped[5:8, 5:8, 5:8].sum() == 27
    labels, n = labeling.label_islands(cropped)
    assert n == 1


def test_dilate_increases_voxels():
    vol = make_volume()
    before = int(vol.sum())
    dil = labeling.dilate_binary(vol, radius=1, iterations=1)
    assert int(dil.sum()) > before


def test_erode_decreases_voxels():
    vol = make_volume()
    before = int(vol.sum())
    ero = labeling.erode_binary(vol, radius=1, iterations=1)
    assert int(ero.sum()) < before


def test_renumber_contiguous_and_preserves_membership():
    vol = make_volume()
    labels, n = labeling.label_islands(vol)
    # Keep only ids 1 and 3 (skip 2) -> should renumber to 1 and 2.
    keep = [1, 3]
    orig_mask_1 = labels == 1
    orig_mask_3 = labels == 3
    new_labels, id_map = labeling.renumber(labels, keep)
    assert id_map == {1: 1, 3: 2}
    assert set(np.unique(new_labels)) == {0, 1, 2}
    # Membership preserved: pixels that were island 1 are now new-island 1, etc.
    np.testing.assert_array_equal(new_labels == 1, orig_mask_1)
    np.testing.assert_array_equal(new_labels == 2, orig_mask_3)


def test_binary_from_selection():
    vol = make_volume()
    labels, n = labeling.label_islands(vol)
    bin_sel = labeling.binary_from_selection(labels, {1})
    np.testing.assert_array_equal(bin_sel > 0, labels == 1)
    empty = labeling.binary_from_selection(labels, set())
    assert empty.sum() == 0


def test_bbox_center():
    vol = make_volume()
    labels, n = labeling.label_islands(vol)
    bboxes, _ = labeling.compute_bboxes_and_sizes(labels, n)
    # 3x3x3 island spans z,y,x = 5..7 -> center 6.
    for iid, bb in bboxes.items():
        zc, yc, xc = labeling.bbox_center(bb)
        assert bb[0].start <= zc < bb[0].stop


def test_connectivity_validation():
    with pytest.raises(ValueError):
        labeling.label_islands(make_volume(), connectivity=7)
