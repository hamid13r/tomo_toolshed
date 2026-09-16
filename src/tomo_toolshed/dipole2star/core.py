"""Core logic for dipole2star.

Converts manual dipole picks straight into oriented-particle star files. Each
input file holds picks as consecutive point pairs: for every particle, one row
marks one end and the next row marks the other end of its long axis. This
collapses each pair into a single centered, oriented particle (Euler angles from
``scipy`` ``align_vectors``, ZYZ degrees) and writes a RELION-style particle
star file per input, named ``<stem>.mrc.star``. It combines what pos2i3.py (pos
-> trf) and trf2star.py (trf -> star) used to do as separate steps.

Two input formats are accepted, chosen from the file extension (``.star`` ->
``starfile``, anything else -> plain 3-column text) or forced with ``fmt``.
There is no click dependency here; the reader and conversion raise plain
``DipoleError`` (a ``ValueError``) that the CLI layer turns into a
``click.ClickException``.
"""

import glob
from pathlib import Path

import numpy as np
import pandas as pd
import starfile
from scipy.spatial.transform import Rotation as R

HEADER = [
    'rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ', 'rlnMicrographName',
    'rlnMagnification', 'rlnPixelSize', 'rlnGroupNumber',
    'rlnAngleRot', 'rlnAngleTilt', 'rlnAnglePsi', 'rlnClassNumber',
]

# Per-project constant baked into the original script: the reconstruction file
# name is the pick-file stem with this suffix. Exposed via --micrograph-suffix.
DEFAULT_MICROGRAPH_SUFFIX = ".mrc_9.98Apx.mrc"

# A dipole axis (nearly) parallel to z makes ``cross(vector, [0, 0, 1])`` vanish
# and the orientation frame degenerate; guard against it below this magnitude.
_Z_PARALLEL_TOL = 1e-8


class DipoleError(ValueError):
    """A pick file cannot be interpreted as valid dipole picks."""


def _noop(*args, **kwargs):
    pass


def _draw_seed():
    """A reproducible 63-bit seed for the local generator."""
    return int(np.random.default_rng().integers(0, 2 ** 63 - 1))


def _read_txt(path):
    """Parse a plain 3-column coordinate file into an (N, 3) float array.

    Forgiving: whitespace- or comma-delimited, blank lines skipped, ``#`` comment
    lines skipped, and a single non-numeric header row (e.g. ``x y z``) skipped
    if present. Columns are X, Y, Z in that order. A data row that is not three
    parseable numbers raises :class:`DipoleError` naming the file and line.
    """
    coords = []
    header_allowed = True
    with open(path) as fh:
        for lineno, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line or line.startswith('#'):
                continue
            fields = [f for f in line.replace(',', ' ').split() if f]
            try:
                values = [float(f) for f in fields]
            except ValueError:
                values = None
            if values is None or len(values) != 3:
                # Allow one leading non-numeric header row (e.g. "x y z").
                if header_allowed and not coords:
                    header_allowed = False
                    continue
                raise DipoleError(
                    f"{path}:{lineno}: expected 3 numbers (X Y Z), got {line!r}"
                )
            header_allowed = False
            coords.append(values)
    return np.asarray(coords, dtype=float).reshape(-1, 3)


def read_coords(path, fmt="auto"):
    """Read picked coordinates from ``path`` as an (N, 3) float array.

    ``fmt`` is one of ``"star"``, ``"txt"`` or ``"auto"``. ``"auto"`` picks the
    reader from the extension: ``.star`` uses :func:`starfile.read`, anything
    else is parsed as 3-column text.
    """
    if fmt == "auto":
        fmt = "star" if Path(path).suffix.lower() == ".star" else "txt"

    if fmt == "star":
        picks = starfile.read(path)
        return picks[['rlnCoordinateX', 'rlnCoordinateY',
                      'rlnCoordinateZ']].to_numpy(dtype=float)
    if fmt == "txt":
        return _read_txt(path)
    raise DipoleError(f"unknown format {fmt!r}; expected star, txt or auto")


def picks_to_particles(coords, scale, random, pixel_size, rng=None, source=None):
    """coords: (N, 3) array of picked points, N even (one pair per particle).

    Collapses each consecutive pair into one centered, oriented particle. The
    orientation math is unchanged from the original script. Raises
    :class:`DipoleError` for a pair whose axis is parallel to z (the frame would
    be degenerate and yield NaN angles).
    """
    rows = []
    for i in range(0, len(coords), 2):
        p0, p1 = coords[i], coords[i + 1]
        center = (p0 + p1) * 0.5 * scale
        diff = p1 - p0
        norm = np.linalg.norm(diff)
        vector = diff / norm

        if np.linalg.norm(np.cross(vector, [0, 0, 1])) < _Z_PARALLEL_TOL:
            raise DipoleError(
                f"{source}: pair {i // 2} has an axis parallel to z; its "
                "orientation is degenerate (align_vectors would return NaN "
                "angles). Re-pick this particle with an off-axis long axis."
            )

        normal_vector = np.cross(vector, [0, 0, 1]) / norm

        if random:
            theta = (rng.random() if rng is not None else np.random.rand()) * 2 * np.pi
            ref_vectors = [[0, 0, 1], [np.cos(theta), np.sin(theta), 0], [-np.sin(theta), np.cos(theta), 0]]
        else:
            ref_vectors = [[0, 0, 1], [0, 1, 0], [0, -1, 0]]

        rotation, _ = R.align_vectors(ref_vectors, [vector, normal_vector, -1 * normal_vector])
        rot, tilt, psi = rotation.as_euler('ZYZ', degrees=True)

        rows.append({
            'rlnCoordinateX': float(int(center[0])),
            'rlnCoordinateY': float(int(center[1])),
            'rlnCoordinateZ': float(int(center[2])),
            'rlnMagnification': 10000,
            'rlnPixelSize': pixel_size,
            'rlnGroupNumber': 1,
            'rlnAngleRot': rot,
            'rlnAngleTilt': tilt,
            'rlnAnglePsi': psi,
            'rlnClassNumber': pd.NA,
        })
    return rows


def convert_one(path, scale, random, pixel_size, outdir, *,
                micrograph_suffix=DEFAULT_MICROGRAPH_SUFFIX, fmt="auto", rng=None):
    """Convert one pick file to an oriented-particle star file.

    Returns ``(outpath, n_particles)``. The output is named ``<stem>.mrc.star``
    in ``outdir``. Raises :class:`DipoleError` for an odd pick count or a
    degenerate (z-parallel) dipole.
    """
    stem = Path(path).stem
    coords = read_coords(path, fmt=fmt)

    if len(coords) % 2 != 0:
        raise DipoleError(
            f"{path}: has {len(coords)} picks (odd) -- dipole picking requires two points "
            "per particle, so this cannot be a valid dipole pick file."
        )

    rows = picks_to_particles(coords, scale, random, pixel_size, rng=rng, source=path)
    for row in rows:
        row['rlnMicrographName'] = f"{stem}{micrograph_suffix}"

    df = pd.DataFrame(rows, columns=HEADER)
    outpath = Path(outdir) / f"{stem}.mrc.star"
    starfile.write(df, outpath, overwrite=True)
    return outpath, len(rows)


def expand_paths(patterns):
    """Glob-expand each pattern, falling back to the literal path if none match."""
    paths = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        paths.extend(matches if matches else [pattern])
    return paths
