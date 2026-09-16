"""Core logic for split-star: split one star file into per-group star files.

Groups a particle star file by its grouping column (tomogram / micrograph /
source) and writes one star file per group into its own subdirectory, named
``<label>_<name>/<label>_<name>_all.star``. This is the natural inverse of
gathering many per-tilt-series star files into a single project file.

There is no ``click`` dependency here; the CLI layer (``cli.py``) wires options
and reporting. Reading with ``always_dict=True`` keeps every input block, so a
multi-block RELION 4/5 file writes valid per-group outputs (optics/general blocks
carried through) instead of crashing on ``star_data['rlnMicrographName']`` the way
the single-block original did, and a single-unnamed-block input still comes back
out as a single unnamed block.
"""

import os
import re
from collections import namedtuple
from importlib import metadata

import numpy as np
import pandas as pd
import starfile


class SplitStarError(ValueError):
    """Raised for any user-facing problem (missing column, no groups, ...)."""


# Same precedence as the sibling tools: first present is used unless overridden.
GROUP_COLUMN_CANDIDATES = ("rlnTomoName", "rlnMicrographName", "wrpSourceName")

# Different star-file versions name their groups differently -- RELION 4
# micrographs end ``.mrc.tomostar``, RELION 5 / M sources end ``.tomostar``, and
# RELION 3 micrographs end ``.mrc``. All of these are stripped by default (longest
# match first) so directory names come out clean regardless of flavor.
DEFAULT_STRIP_SUFFIXES = (".mrc.tomostar", ".tomostar", ".mrc")

# Warp/M reconstruction names embed the pixel size, e.g.
# ``Position_1.mrc_9.98Apx.mrc``; the float varies, so it is matched by pattern
# rather than by a literal suffix. Described in the CLI report as
# ``.mrc_<pixelsize>Apx.mrc``.
DEFAULT_STRIP_PATTERNS = (re.compile(r"\.mrc_\d+(?:\.\d+)?Apx\.mrc$"),)
STRIP_PATTERN_LABEL = ".mrc_<pixelsize>Apx.mrc"

# Flavor identifiers (same scheme as the duplicate-remover tool).
FLAVOR_RELION3 = "relion3"
FLAVOR_RELION4 = "relion4"
FLAVOR_RELION5 = "relion5"
FLAVOR_M = "m"

_WRP_COORD_RE = re.compile(r"^wrpCoordinateX(\d+)$")


def detect_flavor(blocks, particles):
    """Classify the file into one of the four star flavors from its blocks/columns.

    This does not change how splitting works (that only needs the grouping
    column), but it lets the CLI report which flavor it recognised, matching the
    duplicate-remover tool.
    """
    if any(_WRP_COORD_RE.match(c) for c in particles.columns):
        return FLAVOR_M
    optics = blocks.get("optics")
    if isinstance(optics, pd.DataFrame):
        if "rlnTomoTiltSeriesPixelSize" in optics.columns:
            return FLAVOR_RELION5
        return FLAVOR_RELION4
    return FLAVOR_RELION3


def read_star(path):
    """Read ``path`` into an ordered dict of blocks and the particles key."""
    blocks = starfile.read(path, always_dict=True)
    return blocks, particles_key(blocks)


def particles_key(blocks):
    """Return the particles block key: ``'particles'`` if present, else the sole
    block, else the largest ``DataFrame`` block."""
    if "particles" in blocks:
        return "particles"
    frames = {k: v for k, v in blocks.items() if isinstance(v, pd.DataFrame)}
    if not frames:
        raise SplitStarError("no tabular block found in the star file")
    if len(frames) == 1:
        return next(iter(frames))
    return max(frames, key=lambda k: len(frames[k]))


def detect_group_column(particles, override=None):
    """Return a single grouping column: ``override`` if given (and present), else
    the first of :data:`GROUP_COLUMN_CANDIDATES` present in ``particles``."""
    if override is not None:
        if override not in particles.columns:
            raise SplitStarError(
                f"--group-by {override!r} is not a column in the star file")
        return override
    for candidate in GROUP_COLUMN_CANDIDATES:
        if candidate in particles.columns:
            return candidate
    raise SplitStarError(
        "no grouping column found (looked for "
        f"{', '.join(GROUP_COLUMN_CANDIDATES)}); pass --group-by")


def resolve_group_columns(particles, overrides=None):
    """Return the list of grouping columns to split on.

    With one or more ``overrides`` (any columns, not just the name-like ones),
    the file is split by the **combination** of their values -- one output per
    unique tuple. With no override, a single column is auto-detected via
    :func:`detect_group_column`.
    """
    if overrides:
        columns = list(overrides)
        missing = [c for c in columns if c not in particles.columns]
        if missing:
            raise SplitStarError(
                f"--group-by column(s) {missing} not in the star file")
        return columns
    return [detect_group_column(particles)]


def group_dirname(group_name, strip_suffixes=DEFAULT_STRIP_SUFFIXES,
                  strip_patterns=DEFAULT_STRIP_PATTERNS):
    """Turn a group value into a directory-safe base name.

    Strips whichever of the literal ``strip_suffixes`` or the regex
    ``strip_patterns`` matches the **most** characters at the end of the name (so
    ``.mrc.tomostar`` wins over ``.tomostar``, and ``.mrc_9.98Apx.mrc`` wins over
    ``.mrc``), leaving the rest untouched. A single string is accepted for
    ``strip_suffixes``.
    """
    name = str(group_name)
    if isinstance(strip_suffixes, str):
        strip_suffixes = (strip_suffixes,)

    best = 0
    for suffix in strip_suffixes:
        if suffix and name.endswith(suffix):
            best = max(best, len(suffix))
    for pattern in strip_patterns:
        match = pattern.search(name)
        if match and match.end() == len(name):
            best = max(best, match.end() - match.start())
    return name[: -best] if best else name


# A range dimension: split a numeric ``column`` at ``breaks`` into half-open bins.
Range = namedtuple("Range", "column breaks")

# One planned output file.
PlanItem = namedtuple("PlanItem", "combo base output_path n_rows mask descriptions")


def _normalize_dimensions(dimensions):
    """Normalize the split spec into a list of dimensions (strings or Ranges)."""
    if isinstance(dimensions, (str, Range)):
        return [dimensions]
    return list(dimensions)


def _fmt_num(x):
    """Format a break value: integer-valued floats print without a trailing .0."""
    xf = float(x)
    return str(int(xf)) if xf.is_integer() else str(xf)


def _bin_label(lo, hi):
    """Directory-name label for the half-open bin ``[lo, hi)``."""
    if lo == float("-inf"):
        return f"lt{_fmt_num(hi)}"
    if hi == float("inf"):
        return f"ge{_fmt_num(lo)}"
    return f"{_fmt_num(lo)}-{_fmt_num(hi)}"


def _bin_desc(column, lo, hi):
    """Human-readable description of the half-open bin ``[lo, hi)``."""
    if lo == float("-inf"):
        return f"{column} < {_fmt_num(hi)}"
    if hi == float("inf"):
        return f"{column} >= {_fmt_num(lo)}"
    return f"{_fmt_num(lo)} <= {column} < {_fmt_num(hi)}"


def _dimension_arrays(particles, dim, strip_suffixes, strip_patterns):
    """Return ``(key_array, name_array, describe_fn)`` for one split dimension.

    ``key_array`` is what rows are grouped by (raw value for a column, bin index
    for a range); ``name_array`` is the per-row directory-name part; ``describe_fn``
    maps a key to a human-readable description for the provenance header.
    """
    if isinstance(dim, Range):
        column, raw_breaks = dim.column, dim.breaks
        if column not in particles.columns:
            raise SplitStarError(
                f"--range-by {column!r} is not a column in the star file")
        try:
            values = particles[column].to_numpy(dtype=float)
        except (ValueError, TypeError):
            raise SplitStarError(f"--range-by column {column!r} is not numeric")
        if np.isnan(values).any():
            raise SplitStarError(
                f"--range-by column {column!r} has non-numeric/empty values")
        breaks = sorted(float(b) for b in raw_breaks)
        if not breaks:
            raise SplitStarError("--range-by needs at least one break (--breaks)")
        edges = [float("-inf"), *breaks, float("inf")]
        # side='right': a value equal to a break falls in the upper bin ([break, ...)).
        idx = np.searchsorted(np.asarray(breaks), values, side="right")
        names = np.array([_bin_label(edges[i], edges[i + 1]) for i in idx])
        describe = lambda key: _bin_desc(column, edges[key], edges[key + 1])  # noqa: E731
        return idx, names, describe

    column = dim
    if column not in particles.columns:
        raise SplitStarError(f"--group-by {column!r} is not a column in the star file")
    keys = particles[column].to_numpy()
    names = np.array([group_dirname(v, strip_suffixes, strip_patterns) for v in keys])
    describe = lambda key: f"{column}={key}"  # noqa: E731
    return keys, names, describe


def split_spec_string(dimensions):
    """One-line description of the split, for the provenance header/report."""
    parts = []
    for dim in _normalize_dimensions(dimensions):
        if isinstance(dim, Range):
            breaks = sorted(float(b) for b in dim.breaks)
            joined = ", ".join(_fmt_num(b) for b in breaks)
            parts.append(f"range({dim.column}, breaks=[{joined}])")
        else:
            parts.append(str(dim))
    return ", ".join(parts)


def plan_split(particles, dimensions, label, outdir=".",
               strip_suffixes=DEFAULT_STRIP_SUFFIXES,
               strip_patterns=DEFAULT_STRIP_PATTERNS):
    """Return the list of :class:`PlanItem` to be written.

    ``dimensions`` is a single column name, or a list mixing column names and
    :class:`Range` specs; splitting is by the combination of all dimensions, one
    entry per unique key tuple that actually occurs, in first-appearance order
    (so empty range bins produce no file). ``label`` is prefixed as ``<label>_``
    when non-empty.
    """
    dims = _normalize_dimensions(dimensions)
    keys_list, names_list, describers = [], [], []
    for dim in dims:
        keys, names, describe = _dimension_arrays(
            particles, dim, strip_suffixes, strip_patterns)
        keys_list.append(keys)
        names_list.append(names)
        describers.append(describe)

    n_total = len(particles)
    keys_df = pd.DataFrame({i: keys_list[i] for i in range(len(dims))})
    keys_df = keys_df.reset_index(drop=True)
    prefix = f"{label}_" if label else ""

    plan = []
    for pos in keys_df.drop_duplicates().index:
        combo = tuple(keys_df.iloc[pos].tolist())
        base = prefix + "_".join(str(names_list[i][pos]) for i in range(len(dims)))
        mask = np.ones(n_total, dtype=bool)
        for i in range(len(dims)):
            mask &= (keys_list[i] == combo[i])
        descriptions = [describers[i](combo[i]) for i in range(len(dims))]
        output_path = os.path.join(outdir, base, base + "_all.star")
        plan.append(PlanItem(combo, base, output_path,
                             int(mask.sum()), mask, descriptions))
    return plan


def _package_version():
    try:
        return metadata.version("tomo-toolshed")
    except metadata.PackageNotFoundError:
        return ""


def build_header_lines(source, split_spec, descriptions, command="split-star"):
    """Provenance comment lines recording how this output file was produced."""
    version = _package_version()
    stamp = f"tomo_toolshed {command}" + (f" (v{version})" if version else "")
    return [
        f"Created by {stamp}",
        f"source: {source}",
        f"split by: {split_spec}",
        f"this file: {'; '.join(descriptions)}",
    ]


def _prepend_comment(path, lines):
    """Prepend ``# ``-prefixed comment lines to an existing star file."""
    body = open(path).read()
    with open(path, "w") as f:
        f.writelines(f"# {line}\n" for line in lines)
        f.write(body)


def write_group(blocks, part_key, particles, mask, output_path, header_lines=None):
    """Write the rows selected by ``mask`` to ``output_path``, preserving blocks.

    Every non-particles block (optics, general, ...) is carried through unchanged
    and in order; a single-unnamed-block input yields a single-unnamed-block
    output. ``header_lines`` (if given) are prepended as ``#`` comments.
    """
    out = dict(blocks)
    out[part_key] = particles[mask]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    starfile.write(out, output_path, overwrite=True)
    if header_lines:
        _prepend_comment(output_path, header_lines)
    return int(mask.sum())
