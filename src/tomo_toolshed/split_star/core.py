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

import pandas as pd
import starfile


class SplitStarError(ValueError):
    """Raised for any user-facing problem (missing column, no groups, ...)."""


# Same precedence as the sibling tools: first present is used unless overridden.
GROUP_COLUMN_CANDIDATES = ("rlnTomoName", "rlnMicrographName", "wrpSourceName")

# The original stripped this exact suffix from each micrograph name to form the
# output directory name; kept as the default, overridable in the CLI.
DEFAULT_STRIP_SUFFIX = ".mrc.tomostar"


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


def group_dirname(group_name, strip_suffix=DEFAULT_STRIP_SUFFIX):
    """Turn a group value into a directory-safe base name.

    Strips ``strip_suffix`` (if present) from the end -- matching the original,
    which removed ``.mrc.tomostar`` -- and leaves the rest untouched.
    """
    name = str(group_name)
    if strip_suffix and name.endswith(strip_suffix):
        name = name[: -len(strip_suffix)]
    return name


def plan_split(particles, group_column, label, outdir=".",
               strip_suffix=DEFAULT_STRIP_SUFFIX):
    """Return the list of ``(group_name, output_path, n_rows)`` to be written.

    Groups are taken in first-appearance order. ``label`` is prefixed as
    ``<label>_`` when non-empty; when empty/None there is no prefix (the original
    always prefixed, producing a stray ``None_`` when no label was given).
    """
    prefix = f"{label}_" if label else ""
    plan = []
    for group_name in pd.unique(particles[group_column].to_numpy()):
        base = prefix + group_dirname(group_name, strip_suffix)
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
