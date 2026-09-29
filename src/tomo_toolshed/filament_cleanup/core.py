"""Core logic for filament-cleanup: drop particles that don't belong to their helix.

Works on the output of a RELION helical refinement. Particles are grouped into
filaments by (tomogram/micrograph, ``rlnHelicalTubeID``) and ordered along each
filament. Two independent checks flag outliers:

1. **Position** -- the *refined* position (coordinate minus ``rlnOrigin*Angst``)
   is compared to a smooth curve fitted through the whole filament. Only the
   distance *perpendicular* to the curve counts: sliding along the helical axis
   is expected in helical refinement and is not an error. The curve is a robust
   local-linear (LOWESS-style) fit of X/Y/Z against arclength, so a few bad
   points do not drag the curve toward themselves.
2. **Orientation** -- the helical axis implied by ``rlnAngleTilt`` and
   ``rlnAnglePsi`` (``rlnAngleRot`` is the spin about the axis and is ignored) is
   compared to the mean axis of its neighbours along the filament. By default
   the comparison ignores polarity (an axis and its 180-degree flip are the same
   line); ``remove_flipped`` additionally removes the minority polarity of each
   filament.

All geometry is in Angstrom. The pixel size / origin / grouping resolvers are
reused from :mod:`tomo_toolshed.duplicate_remover.core`, so the flavor handling
is identical across the toolshed. There is no ``click`` or ``matplotlib``
dependency here, so the module stays headless-import-safe.
"""

import numpy as np
import pandas as pd

from ..duplicate_remover.core import (
    DuplicateRemoverError,
    FLAVOR_M,
    detect_coordinate_columns,
    detect_flavor,
    detect_group_column,
    resolve_origins,
    resolve_pixel_size,
)


class FilamentCleanupError(ValueError):
    """Raised for any user-facing problem (missing column, bad parameter, ...)."""


TUBE_COLUMN = "rlnHelicalTubeID"
ORDER_COLUMN = "rlnHelicalTrackLengthAngst"
_CENTERED_ANGST = ("rlnCenteredCoordinateXAngst", "rlnCenteredCoordinateYAngst",
                   "rlnCenteredCoordinateZAngst")

# Per-particle rejection reasons (bit flags, so one particle can have several).
REASON_POSITION = 1
REASON_ORIENTATION = 2
REASON_FLIPPED = 4

# Robust neighbour-mean axis: bisquare cut-off (deg) and re-weighting passes.
_ROBUST_ANGLE = 30.0
_ROBUST_ITERS = 3


# ---------------------------------------------------------------------------
# Geometry helpers.
# ---------------------------------------------------------------------------
def helical_axis(tilt_deg, psi_deg):
    """Unit helical-axis vectors in the tomogram frame from RELION tilt/psi.

    This is the third column of RELION's ZYZ Euler matrix (``Euler_angles2matrix``):
    ``(-cos(psi) sin(tilt), sin(psi) sin(tilt), cos(tilt))``. It does not depend
    on rot. Flipping polarity (psi+180, tilt -> 180-tilt) negates the vector.
    """
    t = np.radians(np.asarray(tilt_deg, dtype=float))
    p = np.radians(np.asarray(psi_deg, dtype=float))
    return np.column_stack([-np.cos(p) * np.sin(t),
                            np.sin(p) * np.sin(t),
                            np.cos(t)])


def arclength(points):
    """Cumulative arclength (same units as ``points``) along an ordered polyline."""
    if len(points) == 0:
        return np.zeros(0)
    steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(steps)])


def _tricube(u):
    u = np.clip(np.abs(u), 0.0, 1.0)
    return (1.0 - u ** 3) ** 3


def robust_smooth_curve(s, points, window, iterations=3):
    """Robust local-linear fit of ``points`` (N x 3) against parameter ``s``.

    Returns ``(fitted, tangent, residual)``: the smooth curve evaluated at each
    ``s``, its unit tangent there, and each point's distance from the curve
    *perpendicular* to that tangent.

    ``window`` is the full width (in units of ``s``) of the tricube kernel; the
    half-width is widened per point if needed so every local fit sees at least
    four points. After each pass, points are down-weighted with Tukey's bisquare
    on their perpendicular residual (scale = 6 x median), so outliers stop
    pulling the curve -- the standard LOWESS robustness step.
    """
    s = np.asarray(s, dtype=float)
    points = np.asarray(points, dtype=float)
    n = len(s)
    if n < 2:
        return points.copy(), np.tile([0.0, 0.0, 1.0], (n, 1)), np.zeros(n)

    half = max(float(window) / 2.0, 1e-9)
    k_min = min(n, 4)
    robust = np.ones(n)
    fitted = np.empty_like(points)
    slope = np.empty_like(points)

    for it in range(iterations + 1):
        for i in range(n):
            d = np.abs(s - s[i])
            h = max(half, np.partition(d, k_min - 1)[k_min - 1] * 1.0001)
            w = _tricube(d / h) * robust
            if w.sum() <= 0 or np.count_nonzero(w) < 2:
                w = _tricube(d / h)
            sw = w.sum()
            s0 = (w * s).sum() / sw
            ds = s - s0
            p0 = (w[:, None] * points).sum(0) / sw
            var = (w * ds * ds).sum()
            b = ((w * ds)[:, None] * (points - p0)).sum(0) / var if var > 0 \
                else np.zeros(3)
            fitted[i] = p0 + b * (s[i] - s0)
            slope[i] = b

        norms = np.linalg.norm(slope, axis=1)
        tangent = np.where(norms[:, None] > 0,
                           slope / np.where(norms > 0, norms, 1)[:, None],
                           np.array([0.0, 0.0, 1.0]))
        off = points - fitted
        off -= (off * tangent).sum(1)[:, None] * tangent
        residual = np.linalg.norm(off, axis=1)

        if it == iterations:
            break
        scale = 6.0 * np.median(residual)
        if scale <= 0:
            break
        robust = np.clip(1.0 - (residual / scale) ** 2, 0.0, None) ** 2

    return fitted, tangent, residual


def neighbour_axis_deviation(axes, n_neighbours):
    """Angle (deg) between each axis and the mean axis of its neighbours.

    Neighbours are the ``n_neighbours`` particles on each side in filament order
    (the particle itself excluded; fewer at the ends). The neighbour mean is the
    principal eigenvector of the orientation tensor ``sum(w a a^T)``, which
    treats an axis and its flip as the same line, and the returned angle is in
    ``[0, 90]`` -- polarity is ignored. Neighbours are re-weighted with a
    bisquare on their angle to the mean (zero weight beyond
    :data:`_ROBUST_ANGLE` degrees) so outlying neighbours don't bias it.
    """
    axes = np.asarray(axes, dtype=float)
    n = len(axes)
    out = np.zeros(n)
    if n < 2 or n_neighbours < 1:
        return out
    for i in range(n):
        lo, hi = max(0, i - n_neighbours), min(n, i + n_neighbours + 1)
        idx = [j for j in range(lo, hi) if j != i]
        nb = axes[idx]
        w = np.ones(len(nb))
        for _ in range(_ROBUST_ITERS + 1):
            tensor = (nb * w[:, None]).T @ nb
            _, vecs = np.linalg.eigh(tensor)
            mean = vecs[:, -1]
            # Bisquare down-weighting of neighbours far from the current mean,
            # so one bad neighbour cannot drag the mean onto its good neighbours.
            theta = np.degrees(np.arccos(np.clip(np.abs(nb @ mean), 0.0, 1.0)))
            new_w = np.clip(1.0 - (theta / _ROBUST_ANGLE) ** 2, 0.0, None) ** 2
            if new_w.sum() <= 0:
                break
            w = new_w
        c = abs(float(np.dot(axes[i], mean)))
        out[i] = np.degrees(np.arccos(min(1.0, c)))
    return out


# ---------------------------------------------------------------------------
# Star-file plumbing.
# ---------------------------------------------------------------------------
def resolve_refined_positions(blocks, particles, pixel_size=None):
    """Return ``(refined_angstrom, picked_angstrom, description)``.

    Uses the toolshed's shared resolvers: picked = ``coord x apix``, refined =
    ``picked - origin``. A RELION 5
    file with only ``rlnCenteredCoordinate*Angst`` (no ``rlnCoordinate*``) is
    handled directly in Angstrom.
    """
    flavor = detect_flavor(blocks, particles)
    if flavor == FLAVOR_M:
        raise FilamentCleanupError(
            "M/WarpTools star files have no helical refinement labels "
            f"({TUBE_COLUMN}, rlnAngleTilt, rlnAnglePsi); use a RELION file")
    try:
        coord_cols = detect_coordinate_columns(particles)
    except DuplicateRemoverError:
        if all(c in particles.columns for c in _CENTERED_ANGST):
            coords = particles[list(_CENTERED_ANGST)].to_numpy(dtype=float)
            origins, o_reason, warnings = resolve_origins(
                particles, np.ones(len(particles)))
            pos = coords if origins is None else coords - origins
            desc = {"flavor": flavor, "coord_cols": list(_CENTERED_ANGST),
                    "pixel": "n/a (rlnCenteredCoordinate*Angst already in Å)",
                    "origins": o_reason, "warnings": warnings}
            return pos, coords, desc
        raise FilamentCleanupError(
            "no coordinate columns found (rlnCoordinateX/Y/Z or "
            "rlnCenteredCoordinateX/Y/ZAngst)")
    try:
        pixel = resolve_pixel_size(flavor, blocks, particles, pixel_size)
    except DuplicateRemoverError as exc:
        raise FilamentCleanupError(str(exc))
    origins, o_reason, warnings = resolve_origins(particles, pixel.apix)
    coords = particles[coord_cols].to_numpy(dtype=float) * pixel.apix[:, None]
    pos = coords if origins is None else coords - origins
    px = (f"{pixel.value} Å/px" if pixel.uniform
          else f"varies {pixel.unique.tolist()} Å/px")
    desc = {"flavor": flavor, "coord_cols": coord_cols,
            "pixel": f"{px}  [{pixel.reason}]", "origins": o_reason,
            "warnings": warnings}
    return pos, coords, desc


class CleanupResult:
    """Everything the CLI needs to report and write."""

    def __init__(self, keep_mask, reasons, distance, angle, filaments, info,
                 group_column):
        self.keep_mask = keep_mask          # bool, aligned to particle rows
        self.reasons = reasons              # int bit flags per particle
        self.distance = distance            # Å from the smooth curve (NaN if unchecked)
        self.angle = angle                  # deg from neighbour axis (NaN if unchecked)
        self.filaments = filaments          # list of per-filament dicts
        self.info = info                    # flavor / pixel / origin provenance
        self.group_column = group_column

    @property
    def n_removed(self):
        return int((~self.keep_mask).sum())

    def count(self, flag):
        return int(((self.reasons & flag) != 0).sum())


def clean_filaments(blocks, part_key, *, max_distance=30.0, max_angle=20.0,
                    window=500.0, neighbours=5, min_particles=5,
                    check_position=True, check_orientation=True,
                    remove_flipped=False, pixel_size=None, group_by=None):
    """Flag particles that do not belong to their filament.

    ``max_distance`` (Å) is the largest allowed perpendicular distance of a
    refined position from the smooth filament curve; ``window`` (Å) is the
    full width of the smoothing kernel along the filament. ``max_angle`` (deg)
    is the largest allowed angle between a particle's helical axis and the mean
    axis of its ``neighbours`` particles on each side. Filaments with fewer than
    ``min_particles`` particles are kept untouched (too short to judge).
    """
    particles = blocks[part_key]
    if check_position and max_distance <= 0:
        raise FilamentCleanupError("--max-distance must be > 0")
    if check_orientation and max_angle <= 0:
        raise FilamentCleanupError("--max-angle must be > 0")
    if window <= 0:
        raise FilamentCleanupError("--window must be > 0")
    if neighbours < 1:
        raise FilamentCleanupError("--neighbours must be >= 1")

    need_angles = check_orientation or remove_flipped
    required = [TUBE_COLUMN] + (["rlnAngleTilt", "rlnAnglePsi"] if need_angles else [])
    missing = [c for c in required if c not in particles.columns]
    if missing:
        raise FilamentCleanupError(
            f"missing column(s) {missing}; is this a helical refinement star file?")

    try:
        group_column = detect_group_column(particles, group_by)
    except DuplicateRemoverError as exc:
        raise FilamentCleanupError(str(exc))

    positions, picked, info = resolve_refined_positions(blocks, particles, pixel_size)
    axes = (helical_axis(particles["rlnAngleTilt"], particles["rlnAnglePsi"])
            if need_angles else None)

    n = len(particles)
    reasons = np.zeros(n, dtype=int)
    distance = np.full(n, np.nan)
    angle = np.full(n, np.nan)
    filaments = []

    keys = particles[[group_column, TUBE_COLUMN]]
    for (group, tube), rows in keys.groupby([group_column, TUBE_COLUMN],
                                            sort=False).indices.items():
        rows = np.asarray(rows)
        # Order along the filament: track length if present, else file order.
        if ORDER_COLUMN in particles.columns:
            order = np.argsort(particles[ORDER_COLUMN].to_numpy()[rows],
                               kind="stable")
            rows = rows[order]
        stat = {"group": group, "tube": tube, "n": len(rows),
                "skipped": len(rows) < min_particles,
                "position": 0, "orientation": 0, "flipped": 0}
        filaments.append(stat)
        if stat["skipped"]:
            continue

        # Parametrize by arclength of the *picked* positions: they were traced
        # along the filament, so this is a smooth, monotonic parameter.
        s = arclength(picked[rows])
        _, tangent, resid = robust_smooth_curve(s, positions[rows], window)

        if check_position:
            distance[rows] = resid
            bad = resid > max_distance
            reasons[rows[bad]] |= REASON_POSITION
            stat["position"] = int(bad.sum())

        if check_orientation:
            dev = neighbour_axis_deviation(axes[rows], neighbours)
            angle[rows] = dev
            bad = dev > max_angle
            reasons[rows[bad]] |= REASON_ORIENTATION
            stat["orientation"] = int(bad.sum())

        if remove_flipped:
            sign = np.sign((axes[rows] * tangent).sum(1))
            majority = 1.0 if (sign >= 0).sum() >= (sign < 0).sum() else -1.0
            bad = sign == -majority
            reasons[rows[bad]] |= REASON_FLIPPED
            stat["flipped"] = int(bad.sum())

    keep = reasons == 0
    return CleanupResult(keep, reasons, distance, angle, filaments, info,
                         group_column)


def split_blocks(blocks, part_key, keep_mask):
    """Return ``(kept_blocks, rejected_blocks)`` with every other block carried
    through unchanged and row order preserved."""
    particles = blocks[part_key]
    kept, rejected = dict(blocks), dict(blocks)
    kept[part_key] = particles[keep_mask].reset_index(drop=True)
    rejected[part_key] = particles[~keep_mask].reset_index(drop=True)
    return kept, rejected
