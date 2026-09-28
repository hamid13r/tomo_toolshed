"""Core logic for scale-star: rescale particle coordinates between pixel sizes.

Rescales the coordinate columns of a star file from one pixel size to another
(``factor = input_pixel_size / output_pixel_size``), optionally applies a shift
in output pixels, and rewrites the coordinate pixel-size fields to the new value.
It reads star files the same way split-star does (``always_dict=True``), so it
supports all four flavors (RELION 3/4/5 and M/Warp) and carries every other block
and column through untouched.

There is no ``click`` dependency here; the CLI layer wires options, the
pixel-size mismatch prompt, and reporting. Reader/writer helpers and flavor
detection are reused from :mod:`tomo_toolshed.split_star.core` rather than copied.
"""

import re

import numpy as np
import pandas as pd

# Reuse split-star's reader/writer/flavor helpers so both tools behave identically.
from ..split_star.core import _package_version  # noqa: F401  (re-exported for header)

# Coordinate columns that scale with the pixel size, detected per flavor:
#   RELION 3/4/5 : rlnCoordinateX/Y/Z
#   M / Warp     : wrpCoordinateX<n>/Y<n>/Z<n> (there can be several <n>)
# The ``$`` anchors deliberately exclude any ``*Angst`` variant (e.g.
# rlnCenteredCoordinateXAngst), which is already in Angstrom and left alone.
_RLN_COORD_RE = re.compile(r"^rlnCoordinate([XYZ])$")
_WRP_COORD_RE = re.compile(r"^wrpCoordinate([XYZ])\d+$")

# Pixel-size columns that describe the *coordinate* pixel size and are therefore
# overwritten with --output-pixel-size wherever they appear (particles or optics,
# every row). rlnImagePixelSize is deliberately NOT here: it is the extracted
# subtomogram box pixel size, a separate quantity from the coordinate pixel size,
# so rescaling coordinates must not touch it. Edit this list to change the policy.
PIXEL_SIZE_COLUMNS = (
    "rlnPixelSize",
    "rlnMicrographPixelSize",
    "rlnTomoTiltSeriesPixelSize",
)


class ScaleStarError(ValueError):
    """Raised for any user-facing problem (no coordinate columns, ...)."""


def _axis_of(column):
    """Return the coordinate axis letter (``'X'``/``'Y'``/``'Z'``) of a column."""
    m = _RLN_COORD_RE.match(column) or _WRP_COORD_RE.match(column)
    return m.group(1) if m else None


def find_coordinate_columns(particles):
    """Return the coordinate columns to scale, in column order.

    Matches ``rlnCoordinateX/Y/Z`` and every ``wrpCoordinateX<n>/Y<n>/Z<n>``.
    A 2D file with only X/Y is fine (Z is simply absent). Raises
    :class:`ScaleStarError` if none are present.
    """
    cols = [c for c in particles.columns if _axis_of(c) is not None]
    if not cols:
        raise ScaleStarError(
            "no coordinate columns found (looked for rlnCoordinateX/Y/Z and "
            "wrpCoordinateX<n>/Y<n>/Z<n>)")
    return cols


def collect_pixel_sizes(blocks):
    """Report the coordinate pixel-size values already in the file.

    Returns a list of ``(block_key, column, values)`` -- one entry per
    :data:`PIXEL_SIZE_COLUMNS` column present in any tabular block, with
    ``values`` the sorted unique numeric values found in that column.
    """
    found = []
    for key, block in blocks.items():
        if not isinstance(block, pd.DataFrame):
            continue
        for col in PIXEL_SIZE_COLUMNS:
            if col in block.columns:
                nums = pd.to_numeric(block[col], errors="coerce").to_numpy(dtype=float)
                values = sorted({float(v) for v in nums if not np.isnan(v)})
                found.append((key, col, values))
    return found


def pixel_size_mismatches(pixel_sizes, input_pixel_size, tol_rel=1e-5):
    """Return the ``(block, column, values)`` entries that differ from the input.

    A column mismatches if any of its values is not :func:`numpy.isclose` to
    ``input_pixel_size``.
    """
    mismatches = []
    for key, col, values in pixel_sizes:
        if any(not np.isclose(v, input_pixel_size, rtol=tol_rel) for v in values):
            mismatches.append((key, col, values))
    return mismatches


def scale_particles(particles, coord_columns, factor, shifts):
    """Return a copy of ``particles`` with coordinates scaled and shifted.

    ``new = old * factor + shift`` for each coordinate column, where ``shift`` is
    ``shifts[axis]`` (0 if unset) in *output* pixels. Other columns are untouched.
    """
    out = particles.copy()
    for col in coord_columns:
        shift = shifts.get(_axis_of(col), 0.0)
        out[col] = out[col].astype(float) * factor + shift
    return out


def override_pixel_sizes(blocks, output_pixel_size):
    """Return ``(new_blocks, updated)`` with pixel-size columns set to the target.

    Every :data:`PIXEL_SIZE_COLUMNS` column present in any tabular block is set to
    ``output_pixel_size`` in every row. Non-tabular blocks (e.g. a RELION 5
    ``general`` block) pass through unchanged. ``updated`` is the list of
    ``(block_key, column)`` that were rewritten.
    """
    new_blocks = {}
    updated = []
    for key, block in blocks.items():
        if isinstance(block, pd.DataFrame):
            block = block.copy()
            for col in PIXEL_SIZE_COLUMNS:
                if col in block.columns:
                    block[col] = output_pixel_size
                    updated.append((key, col))
        new_blocks[key] = block
    return new_blocks, updated


def _fmt(x):
    """Compact number formatting: drop a trailing ``.0`` from whole floats."""
    xf = float(x)
    return str(int(xf)) if xf.is_integer() else repr(xf)


def build_header_lines(source, input_pixel_size, output_pixel_size, factor, shifts):
    """Provenance comment lines, in the same style as split-star's header."""
    version = _package_version()
    stamp = "tomo_toolshed scale-star" + (f" (v{version})" if version else "")
    sx, sy, sz = (shifts.get(a, 0.0) for a in ("X", "Y", "Z"))
    return [
        f"Created by {stamp}",
        f"source: {source}",
        f"rescale: {_fmt(input_pixel_size)} -> {_fmt(output_pixel_size)} Apx "
        f"(factor {_fmt(factor)})",
        f"shift (output px): x={_fmt(sx)}, y={_fmt(sy)}, z={_fmt(sz)}",
    ]
