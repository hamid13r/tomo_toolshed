"""
Spline/linear grids used by Warp to store spatially- and temporally-varying
model parameters (defocus, per-tilt shifts, dose weights, B-factors, etc.).

Faithful re-implementation of:
  - WarpLib/CubicGrid.cs      (cubic B-spline grid, "einspline")
  - WarpLib/LinearGrid4D.cs   (quadrilinear 4D grid)

Fidelity notes
--------------
* LinearGrid4D uses quadrilinear interpolation over [0,1]^4 -- reproduced exactly.
* CubicGrid: Warp uses a uniform cubic B-spline (einspline). If SciPy is
  available we use scipy.ndimage.map_coordinates(order=3), the same interpolant.
  If SciPy is absent we fall back to multilinear interpolation. For the grids
  that matter in a standard reconstruction -- per-tilt defocus / dose-weight /
  B-factor grids of shape (1,1,NTilts), which are sampled *exactly on nodes* --
  the result is identical either way (the spline passes through the nodes).
"""
from __future__ import annotations

from dataclasses import dataclass
import warnings
import numpy as np

try:
    from scipy.ndimage import map_coordinates as _map_coordinates
    _HAVE_SCIPY = True
except Exception:                       # pragma: no cover
    _map_coordinates = None
    _HAVE_SCIPY = False
    warnings.warn("SciPy not found; CubicGrid falls back to linear interpolation "
                  "(exact for per-tilt node-sampled grids, approximate otherwise).")


def _multilinear(values, idx):
    """N-linear interpolation of `values` at fractional indices `idx`
    (shape (ndim, N), axis order matching values.shape). Pure NumPy."""
    ndim = values.ndim
    idx = np.asarray(idx, dtype=np.float64)
    base = np.floor(idx).astype(np.int64)
    frac = idx - base
    out = np.zeros(idx.shape[1], dtype=np.float64)
    for corner in range(1 << ndim):
        w = np.ones(idx.shape[1], dtype=np.float64)
        coord = []
        for d in range(ndim):
            bit = (corner >> d) & 1
            ci = np.clip(base[d] + bit, 0, values.shape[d] - 1)
            coord.append(ci)
            w *= frac[d] if bit else (1.0 - frac[d])
        out += w * values[tuple(coord)]
    return out


def _interp(values, idx, order):
    if _HAVE_SCIPY:
        return _map_coordinates(values, idx, order=order, mode="nearest", prefilter=(order > 1))
    return _multilinear(values, idx)


@dataclass
class CubicGrid:
    values: np.ndarray          # (Z, Y, X)
    margins: np.ndarray         # (mx, my, mz)

    @property
    def dims(self):
        z, y, x = self.values.shape
        return np.array([x, y, z], dtype=int)

    @classmethod
    def empty(cls):
        return cls(values=np.zeros((1, 1, 1), np.float32), margins=np.zeros(3, np.float32))

    @classmethod
    def from_xml(cls, node) -> "CubicGrid":
        w = int(node.get("Width", 1)); h = int(node.get("Height", 1)); d = int(node.get("Depth", 1))
        mx = float(node.get("MarginX", 0.0)); my = float(node.get("MarginY", 0.0)); mz = float(node.get("MarginZ", 0.0))
        vals = np.zeros((d, h, w), dtype=np.float32)
        for nd in node.findall("Node"):
            x = int(nd.get("X")); y = int(nd.get("Y")); z = int(nd.get("Z"))
            vals[z, y, x] = float(nd.get("Value"))
        return cls(values=vals, margins=np.array([mx, my, mz], np.float32))

    def interpolate(self, coords_xyz: np.ndarray) -> np.ndarray:
        coords_xyz = np.atleast_2d(np.asarray(coords_xyz, dtype=np.float64))
        x, y, z = self.dims
        if x <= 1 and y <= 1 and z <= 1:
            return np.full(coords_xyz.shape[0], self.values.flat[0], dtype=np.float32)

        def axis_index(norm, n, margin):
            if n <= 1:
                return np.zeros_like(norm)
            usable = 1.0 - 2.0 * margin
            return (norm - margin) / usable * (n - 1)

        mx, my, mz = self.margins
        cx = axis_index(coords_xyz[:, 0], x, mx)
        cy = axis_index(coords_xyz[:, 1], y, my)
        cz = axis_index(coords_xyz[:, 2], z, mz)
        idx = np.vstack([cz, cy, cx])            # values axis order (z, y, x)
        return _interp(self.values, idx, order=3).astype(np.float32)


@dataclass
class LinearGrid4D:
    values: np.ndarray          # (W, Z, Y, X)

    @property
    def dims(self):
        w, z, y, x = self.values.shape
        return np.array([x, y, z, w], dtype=int)

    @classmethod
    def empty(cls):
        return cls(values=np.zeros((1, 1, 1, 1), np.float32))

    @classmethod
    def from_xml(cls, node) -> "LinearGrid4D":
        w = int(node.get("Width", 1)); h = int(node.get("Height", 1))
        d = int(node.get("Depth", 1)); dur = int(node.get("Duration", 1))
        vals = np.zeros((dur, d, h, w), dtype=np.float32)
        for nd in node.findall("Node"):
            x = int(nd.get("X")); y = int(nd.get("Y")); z = int(nd.get("Z")); ww = int(nd.get("W"))
            vals[ww, z, y, x] = float(nd.get("Value"))
        return cls(values=vals)

    def interpolate(self, coords_xyzw: np.ndarray) -> np.ndarray:
        coords = np.atleast_2d(np.asarray(coords_xyzw, dtype=np.float64))
        x, y, z, w = self.dims
        if x <= 1 and y <= 1 and z <= 1 and w <= 1:
            return np.full(coords.shape[0], self.values.flat[0], dtype=np.float32)
        scale = np.array([max(x - 1, 0), max(y - 1, 0), max(z - 1, 0), max(w - 1, 0)], dtype=np.float64)
        ci = coords * scale
        idx = np.vstack([ci[:, 3], ci[:, 2], ci[:, 1], ci[:, 0]])   # (W, Z, Y, X)
        return _interp(self.values, idx, order=1).astype(np.float32)
