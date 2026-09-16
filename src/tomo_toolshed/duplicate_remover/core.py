"""Core logic for duplicate-remover: drop particles closer than a threshold.

Within each group (tomogram / micrograph / source), particles closer than a
distance threshold are duplicates of the same physical object; this removes the
worse of every too-close pair until none remain. All the difficulty is in getting
the geometry right across star flavors, so the flavor-specific decisions live in
small, separately testable resolvers here and the CLI (``cli.py``) only wires
options and reporting. There is no ``click`` or ``matplotlib`` dependency in this
module, so it stays headless-import-safe.

The four flavors this handles, and how they differ (see :func:`resolve_pixel_size`
for the full precedence):

- **RELION 3** -- one unnamed ``data_`` block, no optics; coordinate scale is the
  per-particle ``rlnPixelSize`` (or ``rlnDetectorPixelSize / rlnMagnification``).
- **RELION 4** -- ``optics`` + ``particles``; scale is optics ``rlnImagePixelSize``
  (the particles table may carry a *stale* ``rlnPixelSize`` -- never use it).
- **RELION 5** -- adds ``rlnTomoTiltSeriesPixelSize``; coordinates are in unbinned
  tilt-series pixels, so that column is the scale, not ``rlnImagePixelSize`` (which
  is the extracted-image scale = tilt-series scale x ``rlnTomoSubtomogramBinning``).
- **M / WarpTools** -- ``wrp*`` columns; ``wrpCoordinate*`` are already in
  Ångström, so the scale is 1.0 and no pixel size exists in the file.

All internal geometry is in Ångström.
"""

import re

import numpy as np
import pandas as pd
import starfile
from scipy.spatial import cKDTree


class DuplicateRemoverError(ValueError):
    """Raised for any user-facing problem (bad flavor, missing column, ...)."""


# Flavor identifiers.
FLAVOR_RELION3 = "relion3"
FLAVOR_RELION4 = "relion4"
FLAVOR_RELION5 = "relion5"
FLAVOR_M = "m"


# ---------------------------------------------------------------------------
# I/O: read every block, keep them all, find the particles block.
# ---------------------------------------------------------------------------
def read_star(path):
    """Read ``path`` into an ordered dict of blocks, and pick the particles key.

    Uses ``always_dict=True`` so single-unnamed-block files (RELION 3, M) come
    back as ``{'': df}`` and multi-block files keep every block (including a
    non-tabular ``general`` block, which ``starfile`` returns as a plain dict).
    Returns ``(blocks, particles_key)``.
    """
    blocks = starfile.read(path, always_dict=True)
    return blocks, particles_key(blocks)


def particles_key(blocks):
    """Return the key of the particles block: ``'particles'`` if present, else
    the sole block, else the largest ``DataFrame`` block."""
    if "particles" in blocks:
        return "particles"
    frames = {k: v for k, v in blocks.items() if isinstance(v, pd.DataFrame)}
    if not frames:
        raise DuplicateRemoverError("no tabular block found in the star file")
    if len(frames) == 1:
        return next(iter(frames))
    return max(frames, key=lambda k: len(frames[k]))


def write_star(blocks, particles_key, kept_particles, output_path):
    """Write ``blocks`` to ``output_path`` with the particles block replaced.

    Every other block is preserved in its original order (so RELION 5's
    ``data_general`` and single-unnamed-block inputs survive unchanged).
    """
    out = dict(blocks)
    out[particles_key] = kept_particles
    starfile.write(out, output_path, overwrite=True)


# ---------------------------------------------------------------------------
# Detection: flavor, grouping column, coordinate columns.
# ---------------------------------------------------------------------------
_WRP_COORD_RE = re.compile(r"^wrpCoordinateX(\d+)$")


def detect_flavor(blocks, particles):
    """Classify the file into one of the four flavors from its columns/blocks."""
    if any(_WRP_COORD_RE.match(c) for c in particles.columns):
        return FLAVOR_M
    optics = blocks.get("optics")
    if isinstance(optics, pd.DataFrame):
        if "rlnTomoTiltSeriesPixelSize" in optics.columns:
            return FLAVOR_RELION5
        return FLAVOR_RELION4
    return FLAVOR_RELION3


GROUP_COLUMN_CANDIDATES = ("rlnTomoName", "rlnMicrographName", "wrpSourceName")


def detect_group_column(particles, override=None):
    """Return the grouping column: ``override`` if given (and present), else the
    first of :data:`GROUP_COLUMN_CANDIDATES` present in ``particles``."""
    if override is not None:
        if override not in particles.columns:
            raise DuplicateRemoverError(
                f"--group-by {override!r} is not a column in the star file")
        return override
    for candidate in GROUP_COLUMN_CANDIDATES:
        if candidate in particles.columns:
            return candidate
    raise DuplicateRemoverError(
        "no grouping column found (looked for "
        f"{', '.join(GROUP_COLUMN_CANDIDATES)}); pass --group-by")


def detect_coordinate_columns(particles):
    """Return ``[X, Y, Z]`` coordinate column names.

    RELION uses ``rlnCoordinateX/Y/Z``. M/WarpTools uses ``wrpCoordinateX<N>``
    etc.; the numeric suffix is detected (not hardcoded to 1) and the same ``N``
    is used for all three axes.
    """
    if all(c in particles.columns for c in
           ("rlnCoordinateX", "rlnCoordinateY", "rlnCoordinateZ")):
        return ["rlnCoordinateX", "rlnCoordinateY", "rlnCoordinateZ"]
    suffixes = sorted({m.group(1) for c in particles.columns
                       if (m := _WRP_COORD_RE.match(c))})
    if suffixes:
        n = suffixes[0]
        cols = [f"wrpCoordinateX{n}", f"wrpCoordinateY{n}", f"wrpCoordinateZ{n}"]
        missing = [c for c in cols if c not in particles.columns]
        if missing:
            raise DuplicateRemoverError(
                f"incomplete wrp coordinate columns: missing {missing}")
        return cols
    raise DuplicateRemoverError(
        "no coordinate columns found (rlnCoordinateX/Y/Z or wrpCoordinateX<N>/...)")


# ---------------------------------------------------------------------------
# Pixel size: the crux. Resolve per optics group and report the reason.
# ---------------------------------------------------------------------------
class PixelSize:
    """Per-particle pixel-size result plus a human-readable provenance."""

    def __init__(self, apix, reason, ignored_override=False):
        self.apix = np.asarray(apix, dtype=float)   # Å/px, aligned to particles
        self.reason = reason
        self.ignored_override = ignored_override
        self.unique = np.unique(self.apix[np.isfinite(self.apix)])

    @property
    def uniform(self):
        return len(self.unique) == 1

    @property
    def value(self):
        """The single value if uniform, else ``None``."""
        return float(self.unique[0]) if self.uniform else None


def _optics_pixel_size(blocks, particles, column):
    """Map ``optics[column]`` onto each particle by ``rlnOpticsGroup``."""
    optics = blocks.get("optics")
    if not isinstance(optics, pd.DataFrame) or column not in optics.columns:
        raise DuplicateRemoverError(f"optics block has no {column}")
    n = len(particles)
    if "rlnOpticsGroup" not in particles.columns:
        # No per-particle group: fall back to the first optics row.
        value = float(optics[column].iloc[0])
        return PixelSize(np.full(n, value),
                         f"optics {column} = {value} (no rlnOpticsGroup in "
                         "particles; used the first optics row)")
    mapping = optics.set_index("rlnOpticsGroup")[column].astype(float)
    apix = particles["rlnOpticsGroup"].map(mapping).to_numpy(dtype=float)
    if np.isnan(apix).any():
        missing = sorted(set(particles.loc[np.isnan(apix), "rlnOpticsGroup"]))
        raise DuplicateRemoverError(
            f"optics groups {missing} referenced by particles are missing from "
            "the optics block")
    return PixelSize(apix, f"optics {column} (per rlnOpticsGroup)")


def resolve_pixel_size(flavor, blocks, particles, pixel_size_override=None):
    """Resolve the Å/px scale of the *coordinate columns*, per particle.

    Precedence:

    1. M/WarpTools -> 1.0 (coordinates already in Å); a ``--pixel-size`` override
       is ignored for conversion and flagged.
    2. ``--pixel-size`` override (all other flavors).
    3. RELION 5 -> optics ``rlnTomoTiltSeriesPixelSize``.
    4. RELION 4 -> optics ``rlnImagePixelSize``.
    5. RELION 3 -> per-particle ``rlnPixelSize``, else
       ``rlnDetectorPixelSize / rlnMagnification * 1e4``.
    """
    n = len(particles)
    if flavor == FLAVOR_M:
        return PixelSize(np.ones(n),
                         "M/WarpTools coordinates already in Ångström (scale 1.0)",
                         ignored_override=pixel_size_override is not None)
    if pixel_size_override is not None:
        v = float(pixel_size_override)
        return PixelSize(np.full(n, v), f"--pixel-size override ({v} Å/px)")
    if flavor == FLAVOR_RELION5:
        return _optics_pixel_size(blocks, particles, "rlnTomoTiltSeriesPixelSize")
    if flavor == FLAVOR_RELION4:
        return _optics_pixel_size(blocks, particles, "rlnImagePixelSize")
    # RELION 3.
    if "rlnPixelSize" in particles.columns:
        return PixelSize(particles["rlnPixelSize"].to_numpy(dtype=float),
                         "per-particle rlnPixelSize (RELION 3)")
    if {"rlnDetectorPixelSize", "rlnMagnification"} <= set(particles.columns):
        apix = (particles["rlnDetectorPixelSize"].to_numpy(dtype=float)
                / particles["rlnMagnification"].to_numpy(dtype=float) * 1e4)
        return PixelSize(apix,
                         "rlnDetectorPixelSize / rlnMagnification x 1e4 (RELION 3)")
    raise DuplicateRemoverError(
        "cannot resolve a pixel size (no optics block and no rlnPixelSize / "
        "rlnDetectorPixelSize+rlnMagnification); pass --pixel-size")


# ---------------------------------------------------------------------------
# Origins and final positions.
# ---------------------------------------------------------------------------
_ORIGIN_ANGST = ("rlnOriginXAngst", "rlnOriginYAngst", "rlnOriginZAngst")
_ORIGIN_PIX = ("rlnOriginX", "rlnOriginY", "rlnOriginZ")


def resolve_origins(particles, apix):
    """Return ``(origin_angstrom_array_or_None, reason, warnings)``.

    ``rlnOrigin*Angst`` are in Å and used directly; RELION-3 ``rlnOrigin*`` are in
    pixels and multiplied by ``apix``. If only some of the three axes are present,
    the missing ones are treated as 0 with a warning (rather than silently
    dropping all shifts, as the ``and``-chained original did).
    """
    n = len(particles)
    cols = set(particles.columns)
    warnings = []

    def _collect(names, to_angstrom):
        present = [c for c in names if c in cols]
        if not present:
            return None
        if len(present) < 3:
            missing = [c for c in names if c not in cols]
            warnings.append(
                f"origin columns {present} present but {missing} missing; "
                "treating missing axes as 0")
        origin = np.zeros((n, 3), dtype=float)
        for k, name in enumerate(names):
            if name in cols:
                origin[:, k] = particles[name].to_numpy(dtype=float)
        return to_angstrom(origin)

    origin = _collect(_ORIGIN_ANGST, lambda o: o)
    if origin is not None:
        return origin, "rlnOrigin*Angst (Å)", warnings
    origin = _collect(_ORIGIN_PIX, lambda o: o * apix[:, None])
    if origin is not None:
        return origin, "rlnOrigin* (pixels x apix)", warnings
    return None, "none (no origin columns)", warnings


def compute_positions(particles, coord_cols, apix, origins_angstrom):
    """Return the Nx3 particle positions in Å: ``coord * apix - origin``."""
    coords = particles[coord_cols].to_numpy(dtype=float)
    positions = coords * apix[:, None]
    if origins_angstrom is not None:
        positions = positions - origins_angstrom
    return positions


# ---------------------------------------------------------------------------
# Comparison metric -> per-particle quality (higher = keep).
# ---------------------------------------------------------------------------
METRIC_AUTO = "auto"
METRIC_RANDOM = "random"
METRIC_FIRST = "first"
_AUTO_ORDER = ("rlnLogLikeliContribution", "rlnMaxValueProbDistribution")


def resolve_metric(columns, requested):
    """Return ``(metric_name, reason)``.

    ``auto`` picks the first present of ``rlnLogLikeliContribution`` then
    ``rlnMaxValueProbDistribution``, else falls back to keep-first. An explicitly
    requested metric column that is absent is an error.
    """
    columns = set(columns)
    if requested == METRIC_AUTO:
        for name in _AUTO_ORDER:
            if name in columns:
                return name, f"auto -> {name}"
        return METRIC_FIRST, "auto -> keep-first (no metric column present)"
    if requested in (METRIC_RANDOM, METRIC_FIRST):
        return requested, requested
    if requested in columns:
        return requested, requested
    raise DuplicateRemoverError(
        f"comparison metric {requested!r} is not a column in the star file")


def compute_quality(particles, metric, rng):
    """Per-particle quality where higher means keep."""
    n = len(particles)
    if metric == METRIC_RANDOM:
        return rng.random(n)
    if metric == METRIC_FIRST:
        # Earlier rows rank higher, so keep-first is honored.
        return np.arange(n, 0, -1, dtype=float)
    return particles[metric].to_numpy(dtype=float)


# ---------------------------------------------------------------------------
# The removal itself.
# ---------------------------------------------------------------------------
def _resolve_group(positions, quality, threshold):
    """Return ``(keep_bool, removed_distances)`` for one group.

    Uses a single ``query_pairs`` pass, then removes the worse member of every
    too-close pair, worst first. Processing particles in ascending quality order
    guarantees that when a particle is removed, each of its still-kept neighbours
    ranks at least as high -- so we always drop the worse member and never both
    members of a pair. Ties break on coordinates (a stable, row-order-independent
    key) so non-random results are deterministic. The threshold is exclusive
    (distance ``<`` threshold), matching the original.
    """
    n = len(positions)
    keep = np.ones(n, dtype=bool)
    removed_distances = []
    if n < 2:
        return keep, removed_distances

    tree = cKDTree(positions)
    pairs = tree.query_pairs(threshold, output_type="ndarray")
    if len(pairs) == 0:
        return keep, removed_distances
    dist = np.linalg.norm(positions[pairs[:, 0]] - positions[pairs[:, 1]], axis=1)
    within = dist < threshold          # strict: a pair exactly at threshold is kept
    pairs, dist = pairs[within], dist[within]
    if len(pairs) == 0:
        return keep, removed_distances

    adjacency = {}
    for (a, b), d in zip(pairs, dist):
        adjacency.setdefault(int(a), []).append((int(b), float(d)))
        adjacency.setdefault(int(b), []).append((int(a), float(d)))

    # Worst first: ascending quality, ties by x, then y, then z.
    order = np.lexsort((positions[:, 2], positions[:, 1], positions[:, 0], quality))
    for p in order:
        p = int(p)
        neighbours = adjacency.get(p)
        if not neighbours:
            continue
        kept_dists = [d for nb, d in neighbours if keep[nb]]
        if kept_dists:
            keep[p] = False
            removed_distances.append(min(kept_dists))
    return keep, removed_distances


class Result:
    """Everything the CLI needs to write output and print a report."""

    def __init__(self, *, keep_mask, removed_distances, flavor, group_column,
                 coord_cols, pixel, origin_reason, origins_applied, warnings,
                 metric, metric_reason, units, threshold_input,
                 threshold_angstrom, threshold_pixel, group_stats,
                 n_before, n_after):
        self.keep_mask = keep_mask
        self.removed_distances = removed_distances
        self.flavor = flavor
        self.group_column = group_column
        self.coord_cols = coord_cols
        self.pixel = pixel
        self.origin_reason = origin_reason
        self.origins_applied = origins_applied
        self.warnings = warnings
        self.metric = metric
        self.metric_reason = metric_reason
        self.units = units
        self.threshold_input = threshold_input
        self.threshold_angstrom = threshold_angstrom
        self.threshold_pixel = threshold_pixel
        self.group_stats = group_stats            # list of (name, initial, removed)
        self.n_before = n_before
        self.n_after = n_after

    @property
    def n_removed(self):
        return self.n_before - self.n_after


def process(blocks, part_key, *, threshold, units="angstroms", pixel_size=None,
            group_by=None, comparison_metric=METRIC_AUTO, seed=None):
    """Run the full pipeline and return a :class:`Result`.

    Does not mutate ``blocks``; the caller applies ``result.keep_mask`` to write.
    """
    particles = blocks[part_key]
    n = len(particles)

    flavor = detect_flavor(blocks, particles)
    coord_cols = detect_coordinate_columns(particles)
    group_column = detect_group_column(particles, group_by)
    pixel = resolve_pixel_size(flavor, blocks, particles, pixel_size)

    if units == "pixels" and flavor == FLAVOR_M:
        raise DuplicateRemoverError(
            "--units pixels is not supported for M/WarpTools files: there is no "
            "pixel size to convert with (coordinates are already in Å). Use "
            "--threshold in Ångström with --units angstroms.")

    origins, origin_reason, warnings = resolve_origins(particles, pixel.apix)
    metric, metric_reason = resolve_metric(particles.columns, comparison_metric)
    rng = np.random.default_rng(seed)
    quality = compute_quality(particles, metric, rng)
    positions = compute_positions(particles, coord_cols, pixel.apix, origins)

    keep_mask = np.ones(n, dtype=bool)
    removed_distances = []
    group_stats = []
    group_values = particles[group_column].to_numpy()

    for name in pd.unique(group_values):
        idx = np.nonzero(group_values == name)[0]
        # Threshold in Å for this group.
        if units == "pixels":
            group_apix = np.unique(pixel.apix[idx])
            if len(group_apix) > 1:
                warnings.append(
                    f"group {name!r} has multiple pixel sizes {group_apix.tolist()}; "
                    "using their mean for the --units pixels conversion")
            thr = threshold * float(group_apix.mean())
        else:
            thr = threshold
        gkeep, gdist = _resolve_group(positions[idx], quality[idx], thr)
        keep_mask[idx[~gkeep]] = False
        removed_distances.extend(gdist)
        group_stats.append((name, len(idx), int((~gkeep).sum())))

    n_after = int(keep_mask.sum())

    # Representative threshold in both units for the report.
    apix_value = pixel.value
    if units == "pixels":
        threshold_pixel = threshold
        threshold_angstrom = (threshold * apix_value) if apix_value else None
    else:
        threshold_angstrom = threshold
        threshold_pixel = (threshold / apix_value) if apix_value else None

    return Result(
        keep_mask=keep_mask, removed_distances=removed_distances, flavor=flavor,
        group_column=group_column, coord_cols=coord_cols, pixel=pixel,
        origin_reason=origin_reason, origins_applied=origins is not None,
        warnings=warnings, metric=metric, metric_reason=metric_reason,
        units=units, threshold_input=threshold,
        threshold_angstrom=threshold_angstrom, threshold_pixel=threshold_pixel,
        group_stats=group_stats, n_before=n, n_after=n_after)


def apply_keep(blocks, part_key, keep_mask):
    """Return the kept particles (original row order preserved)."""
    return blocks[part_key].iloc[keep_mask]
