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
    """Return the grouping column: ``override`` if given (and present), else the
    first of :data:`GROUP_COLUMN_CANDIDATES` present in ``particles``."""
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


def plan_split(particles, group_column, label, outdir=".",
               strip_suffixes=DEFAULT_STRIP_SUFFIXES,
               strip_patterns=DEFAULT_STRIP_PATTERNS):
    """Return the list of ``(group_name, output_path, n_rows)`` to be written.

    Groups are taken in first-appearance order. ``label`` is prefixed as
    ``<label>_`` when non-empty; when empty/None there is no prefix (the original
    always prefixed, producing a stray ``None_`` when no label was given).
    """
    prefix = f"{label}_" if label else ""
    plan = []
    for group_name in pd.unique(particles[group_column].to_numpy()):
        base = prefix + group_dirname(group_name, strip_suffixes, strip_patterns)
        n_rows = int((particles[group_column] == group_name).sum())
        output_path = os.path.join(outdir, base, base + "_all.star")
        plan.append((group_name, output_path, n_rows))
    return plan


def write_group(blocks, part_key, particles, group_column, group_name, output_path):
    """Write one group's rows to ``output_path``, preserving other input blocks.

    Every non-particles block (optics, general, ...) is carried through unchanged
    and in order; a single-unnamed-block input yields a single-unnamed-block
    output.
    """
    subset = particles[particles[group_column] == group_name]
    out = dict(blocks)
    out[part_key] = subset
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    starfile.write(out, output_path, overwrite=True)
    return len(subset)
