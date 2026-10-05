"""Shared fixtures: a tiny synthetic Warp tilt-series (XML + MRC stack).

The images are deliberately large enough (>= a few hundred px) that the
ReconstructFull border mask (default 32 px) leaves real signal; a 12-px image
would be masked to zero. This mirrors the fact that real tilt images are large.
"""
from __future__ import annotations

import numpy as np
import mrcfile
import pytest


def _write_series(dirpath, n=9, raw_px=10.0, img_px=160, box_px=(160, 160, 40)):
    angles = np.linspace(-40, 40, n)
    rng = np.random.default_rng(1)
    stack = rng.standard_normal((n, img_px, img_px)).astype(np.float32)
    st_path = dirpath / "TS.st"
    with mrcfile.new(str(st_path), overwrite=True) as m:
        m.set_data(stack)
        m.voxel_size = (raw_px, raw_px, raw_px)

    bx, by, bz = box_px
    nodes = "".join(f'<Node X="0" Y="0" Z="{i}" Value="3.0" />' for i in range(n))
    angles_txt = "\n".join(f"{a:.2f}" for a in angles)
    dose_txt = "\n".join(f"{i * 3.0:.2f}" for i in range(n))
    use_txt = "\n".join("True" for _ in range(n))
    axis_txt = "\n".join("85.0" for _ in range(n))
    xml = (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        f'<TiltSeries AreAnglesInverted="False" '
        f'ImageDimensionsAngstrom="{img_px * raw_px}, {img_px * raw_px}" '
        f'VolumeDimensionsAngstrom="{bx * raw_px}, {by * raw_px}, {bz * raw_px}">\n'
        '  <CTF PixelSize="10" Voltage="300" Cs="2.7" Amplitude="0.07" '
        'Defocus="3.0" PhaseShift="0" />\n'
        f'  <Angles>{angles_txt}</Angles>\n'
        f'  <Dose>{dose_txt}</Dose>\n'
        f'  <UseTilt>{use_txt}</UseTilt>\n'
        f'  <AxisAngle>{axis_txt}</AxisAngle>\n'
        f'  <GridCTF Width="1" Height="1" Depth="{n}">{nodes}</GridCTF>\n'
        '</TiltSeries>\n'
    )
    xml_path = dirpath / "TS.xml"
    xml_path.write_text(xml)
    return xml_path, st_path, n


@pytest.fixture
def synthetic_series(tmp_path):
    """Return (xml_path, stack_path, n_tilts) for a tiny synthetic series."""
    return _write_series(tmp_path)
