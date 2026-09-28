"""Particle mode for the curator: star / xyz text -> spheres -> filtered star.

Input is a particle ``.star`` file or a plain-text coordinate list (``.txt`` /
``.box``: three columns x y z, whitespace- or comma-separated, ``#`` comments
allowed). The output is always a star file.

Each particle is painted as a sphere and becomes its own "island" whose label
is ``row index + 1``, so toggling an island in the GUI toggles exactly one
particle and the kept labels map straight back to star-file rows. Overlapping
spheres are split between particles by nearest center, so two close particles
never merge into one island.

Coordinates follow the RELION 4.x convention (``rlnCoordinateX/Y/Z`` in pixels,
optional ``rlnOriginX/Y/ZAngst`` shifts). No matplotlib here; headless-testable.
Volumes are (Z, Y, X).
"""

from __future__ import annotations

import os
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from ..split_star.core import (
    GROUP_COLUMN_CANDIDATES,
    read_star,
    write_star_file,
)

COORD_COLUMNS = ("rlnCoordinateX", "rlnCoordinateY", "rlnCoordinateZ")
ORIGIN_COLUMNS = ("rlnOriginXAngst", "rlnOriginYAngst", "rlnOriginZAngst")


class ParticleStarError(ValueError):
    """A star file that particle mode cannot use."""


TEXT_COORD_EXTENSIONS = (".txt", ".box")


def is_star_path(path: str) -> bool:
    return str(path).lower().endswith(".star")


def is_text_coords_path(path: str) -> bool:
    return str(path).lower().endswith(TEXT_COORD_EXTENSIONS)


def is_particle_path(path: str) -> bool:
    """True for any particle-mode input (star or xyz text)."""
    return is_star_path(path) or is_text_coords_path(path)


def output_star_name(path: str) -> str:
    """Output file name: the star's own name, or ``<stem>.star`` for text input."""
    name = os.path.basename(str(path))
    return name if is_star_path(name) else os.path.splitext(name)[0] + ".star"


def load_text_coordinates(path: str):
    """Read a 3-column x y z text file into ``(blocks, key, df)``.

    The result is a single ``particles`` block with ``rlnCoordinateX/Y/Z``, so
    it flows through the same curation and star writing as star input.
    """
    rows = []
    with open(path) as f:
        for lineno, line in enumerate(f, start=1):
            line = line.split("#", 1)[0].replace(",", " ").strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 3:
                raise ParticleStarError(
                    f"{path}:{lineno}: expected 3 columns (x y z), got {len(parts)}")
            try:
                rows.append([float(v) for v in parts])
            except ValueError:
                raise ParticleStarError(
                    f"{path}:{lineno}: non-numeric coordinate in {line!r}")
    if not rows:
        raise ParticleStarError(f"{path}: no coordinates found")
    xyz = np.asarray(rows)
    df = pd.DataFrame({c: xyz[:, i] for i, c in enumerate(COORD_COLUMNS)})
    return {"particles": df}, "particles", df


def load_particles(path: str):
    """Read ``path`` and return ``(blocks, particles_key, particles_df)``.

    Text coordinate files (``.txt`` / ``.box``) go through
    :func:`load_text_coordinates`; anything else is read as a star file.

    Raises :class:`ParticleStarError` if the coordinate columns are missing or
    the file holds particles from more than one tomogram.
    """
    if is_text_coords_path(path):
        return load_text_coordinates(path)
    blocks, key = read_star(path)
    df = blocks[key]
    missing = [c for c in COORD_COLUMNS if c not in df.columns]
    if missing:
        hint = ""
        if "rlnCenteredCoordinateXAngst" in df.columns:
            hint = (" This looks like a RELION 5 file (centered Å coordinates);"
                    " only RELION 4-style pixel coordinates are supported.")
        raise ParticleStarError(
            f"star file is missing {', '.join(missing)}.{hint}")
    for col in GROUP_COLUMN_CANDIDATES:
        if col in df.columns:
            names = pd.unique(df[col])
            if len(names) > 1:
                raise ParticleStarError(
                    f"star file holds {len(names)} tomograms ({col}); particle "
                    "mode expects one. Split it first with "
                    "`tomo_toolshed split-star`.")
            break
    return blocks, key, df.reset_index(drop=True)


def particle_centers_zyx(
    df: pd.DataFrame,
    tomo_pixel_size: float,
    coord_pixel_size: Optional[float] = None,
) -> np.ndarray:
    """Return an ``(N, 3)`` float array of particle centers in tomogram voxels.

    Coordinates are taken to be in ``coord_pixel_size`` Å/px (default: the
    tomogram's own pixel size) and rescaled by ``coord_pixel_size /
    tomo_pixel_size``. ``rlnOrigin*Angst`` shifts, if present, are applied the
    RELION way (``position = coordinate - origin / pixel_size``).
    """
    cpx = float(coord_pixel_size) if coord_pixel_size else float(tomo_pixel_size)
    xyz = df[list(COORD_COLUMNS)].astype(float).to_numpy()
    if all(c in df.columns for c in ORIGIN_COLUMNS):
        xyz = xyz - df[list(ORIGIN_COLUMNS)].astype(float).to_numpy() / cpx
    xyz = xyz * (cpx / float(tomo_pixel_size))
    return xyz[:, ::-1].copy()           # x,y,z -> z,y,x


def paint_spheres(
    shape: Sequence[int], centers_zyx: np.ndarray, radius_px: float
) -> np.ndarray:
    """Paint one labeled sphere per particle into an int32 volume of ``shape``.

    Particle ``i`` (0-based row) gets label ``i + 1``. Where spheres overlap,
    each voxel goes to the nearest center. Particles whose sphere lies fully
    outside the volume paint nothing.
    """
    shape = tuple(int(s) for s in shape)
    labels = np.zeros(shape, dtype=np.int32)
    best = np.full(shape, np.inf, dtype=np.float32)
    r = float(radius_px)
    if r <= 0 or len(centers_zyx) == 0:
        return labels
    r2 = r * r
    for i, c in enumerate(np.asarray(centers_zyx, dtype=float)):
        lo = np.maximum(np.floor(c - r).astype(int), 0)
        hi = np.minimum(np.ceil(c + r).astype(int) + 1, shape)
        if np.any(hi <= lo):
            continue
        zz, yy, xx = np.ogrid[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        d2 = ((zz - c[0]) ** 2 + (yy - c[1]) ** 2 + (xx - c[2]) ** 2).astype(np.float32)
        sub = (slice(lo[0], hi[0]), slice(lo[1], hi[1]), slice(lo[2], hi[2]))
        win = (d2 <= r2) & (d2 < best[sub])
        best[sub][win] = d2[win]
        labels[sub][win] = i + 1
    return labels


def ids_in_zrange(centers_zyx: np.ndarray, z_min: int, z_max: int):
    """Particle ids (1-based) whose center Z lies within ``[z_min, z_max]``."""
    z = np.asarray(centers_zyx, dtype=float)[:, 0]
    inside = (z >= z_min) & (z <= z_max)
    return {int(i) + 1 for i in np.flatnonzero(inside)}


def write_kept_particles(blocks, key, df: pd.DataFrame, keep_ids, out_path: str) -> int:
    """Write ``blocks`` to ``out_path`` keeping only particle ids in ``keep_ids``.

    ``keep_ids`` are 1-based row ids (the island labels). Other blocks
    (optics, general) pass through unchanged. Returns the number of rows kept.
    """
    keep = np.zeros(len(df), dtype=bool)
    idx = np.array(sorted(int(i) - 1 for i in keep_ids if 1 <= int(i) <= len(df)),
                   dtype=int)
    keep[idx] = True
    out = dict(blocks)
    out[key] = df.loc[keep].reset_index(drop=True)
    write_star_file(out, out_path)
    return int(keep.sum())
