"""Core logic for the filament tracer.

Traces filaments in a binary segmentation mask and exports a RELION-style
helical ``.star`` file of evenly spaced particles with per-particle ZYZ Euler
angles derived from the local filament tangent. Optionally writes a ChimeraX
``.bild`` overlay for visual inspection.

This module is a straight refactor of a hardcoded script into a parameterized
pipeline: the tracing algorithm and its numerical behavior are unchanged, only
the former module-level constants are now function arguments. There is no
plotting or GUI dependency here; ``.bild`` output is plain text and opt-in.
"""

import os

import numpy as np
import mrcfile
import pandas as pd
import starfile
from scipy import ndimage as ndi
from scipy.interpolate import splprep, splev
from scipy.spatial.transform import Rotation as R

# ``skeletonize_3d`` was folded into ``skeletonize`` in newer scikit-image; the
# 3D result is identical, so fall back transparently on either API.
try:  # pragma: no cover - depends on installed scikit-image version
    from skimage.morphology import skeletonize_3d
except ImportError:  # pragma: no cover
    from skimage.morphology import skeletonize as skeletonize_3d

import networkx as nx


# ---------------------------------------------------------------
# Column layout of the output helical star file.
# ---------------------------------------------------------------
STAR_HEADER = ['rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ',
               'rlnMicrographName', 'rlnMagnification', 'rlnPixelSize',
               'rlnGroupNumber',
               'rlnAngleRot', 'rlnAngleTilt', 'rlnAnglePsi',
               'rlnAngleRotPrior', 'rlnAngleTiltPrior', 'rlnAnglePsiPrior',
               'rlnHelicalTubeID', 'rlnHelicalTrackLength']


# ---------------------------------------------------------------
# Skeleton -> graph -> centerline helpers
# ---------------------------------------------------------------
def skeleton_to_graph(skel):
    coords = np.argwhere(skel)                       # (z,y,x)
    idx = {tuple(c): i for i, c in enumerate(coords)}
    G = nx.Graph()
    G.add_nodes_from(range(len(coords)))
    offs = [np.array(o) - 1 for o in np.ndindex(3, 3, 3) if o != (1, 1, 1)]
    for i, c in enumerate(coords):
        for o in offs:
            j = idx.get(tuple(c + o))
            if j is not None and j > i:
                G.add_edge(i, j, weight=float(np.linalg.norm(o)))
    return G, coords


def longest_path_coords(G, coords):
    if G.number_of_nodes() < 2:
        return None
    endpoints = [n for n in G.nodes if G.degree[n] == 1]
    if len(endpoints) < 2:
        endpoints = [list(G.nodes)[0]]

    def farthest(src):
        d = nx.single_source_dijkstra_path_length(G, src)
        far = max(d, key=d.get)
        return far, d[far]

    a, _ = farthest(endpoints[0])
    b, _ = farthest(a)
    path = nx.shortest_path(G, a, b, weight="weight")
    return coords[path]


def moving_average_3d(pts, win):
    if win <= 1 or len(pts) < win:
        return pts
    k = np.ones(win) / win
    sm = np.stack([np.convolve(pts[:, d], k, mode="same") for d in range(3)], axis=1)
    sm[0], sm[-1] = pts[0], pts[-1]
    return sm


# ---------------------------------------------------------------
# Tangent -> rotation matrix -> ZYZ Euler
# ---------------------------------------------------------------
def tangent_to_matrix(t, axis="z"):
    d = t / (np.linalg.norm(t) + 1e-9)
    ref = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(d, ref)) > 0.95:
        ref = np.array([1.0, 0.0, 0.0])
    a = np.cross(ref, d); a /= (np.linalg.norm(a) + 1e-9)
    b = np.cross(d, a)
    if axis == "z":
        M = np.column_stack([a, b, d])
    elif axis == "x":
        M = np.column_stack([d, a, b])
    else:
        M = np.column_stack([a, d, b])
    return M


def matrix_to_euler(t, tangent_axis="z", invert_rot=True):
    M = tangent_to_matrix(t, tangent_axis)
    if invert_rot:
        M = M.T
    return R.from_matrix(M).as_euler("ZYZ", degrees=True)


# ---------------------------------------------------------------
# ChimeraX .bild overlay (opt-in)
# ---------------------------------------------------------------
def write_bild(path, records, r=3.0, clen=8.0):
    """Write a ChimeraX ``.bild`` overlay (pixel coordinates) for ``records``.

    Called only when the caller opts in; it never touches the primary star
    output.
    """
    with open(path, "w") as f:
        for rec in records:
            x, y, z = rec['pos']; t = rec['tan']
            f.write(".color 1 0.8 0\n")
            f.write(f".sphere {x:.2f} {y:.2f} {z:.2f} {r:.2f}\n")
            tip = np.array([x, y, z]) + t * clen
            f.write(".color 0 0.6 1\n")
            f.write(f".arrow {x:.2f} {y:.2f} {z:.2f} "
                    f"{tip[0]:.2f} {tip[1]:.2f} {tip[2]:.2f} 0.6\n")


def _noop(*args, **kwargs):
    pass


def trace_filaments(
    mask_path,
    star_out,
    *,
    pixel_size=None,
    spacing_a=82.0,
    min_voxels=50,
    min_path_a=500.0,
    smooth_factor=1.0,
    presmooth_win=3,
    micrograph=None,
    magnification=10000,
    group_number=1,
    tangent_axis="z",
    invert_rot=True,
    write_bild_file=False,
    bild_path=None,
    log=None,
):
    """Trace filaments in ``mask_path`` and write a helical star file.

    Returns ``(dataframe, records)``. The star file is always written to
    ``star_out``. A ``.bild`` overlay is written to ``bild_path`` only when
    ``write_bild_file`` is true; the star output is identical either way.
    """
    log = log or _noop

    # 1. Load MRC  (mrc axis order z,y,x -> work in x,y,z)
    with mrcfile.open(mask_path, permissive=True) as m:
        vol = np.asarray(m.data)
        if pixel_size is None:
            pixel_size = float(m.voxel_size.x)

    mask = vol > 0
    log(f"pixel size {pixel_size} Å, {mask.sum()} foreground voxels")

    # 2. Separate islands (26-connectivity)
    structure = np.ones((3, 3, 3), int)
    labels, n_islands = ndi.label(mask, structure=structure)
    log(f"{n_islands} connected components")

    # 3+4. Skeletonize each island, build graph, extract centerline
    centerlines = []
    for lab in range(1, n_islands + 1):
        island = labels == lab
        if island.sum() < min_voxels:
            continue
        skel = skeletonize_3d(island) > 0
        if skel.sum() < 2:
            continue
        G, coords = skeleton_to_graph(skel)
        for comp in nx.connected_components(G):
            sub = G.subgraph(comp)
            path_zyx = longest_path_coords(sub, coords)
            if path_zyx is None or len(path_zyx) < 4:
                continue
            path_xyz = path_zyx[:, ::-1].astype(float)   # z,y,x -> x,y,z
            length_A = np.linalg.norm(np.diff(path_xyz, axis=0), axis=1).sum() * pixel_size
            if length_A < min_path_a:
                continue
            centerlines.append(path_xyz)

    log(f"{len(centerlines)} centerlines kept")

    # 5. Smooth + resample each centerline; keep per-tube grouping
    spacing_px = spacing_a / pixel_size
    records = []   # dict per particle

    tube_id = 0
    for pts in centerlines:
        keep = np.concatenate([[True], np.any(np.diff(pts, axis=0) != 0, axis=1)])
        pts = pts[keep]
        if len(pts) < 4:
            continue
        pts = moving_average_3d(pts, presmooth_win)

        seglen = np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()
        n = max(int(seglen / spacing_px) + 1, 2)
        s = smooth_factor * len(pts)
        try:
            tck, u = splprep(pts.T, s=s, k=min(3, len(pts) - 1))
        except Exception:
            continue

        uu = np.linspace(0, 1, n)
        sx, sy, sz = splev(uu, tck)
        dx, dy, dz = splev(uu, tck, der=1)
        samp = np.stack([sx, sy, sz], axis=1)
        tang = np.stack([dx, dy, dz], axis=1)
        tang /= (np.linalg.norm(tang, axis=1, keepdims=True) + 1e-9)

        ref_dir = samp[-1] - samp[0]
        if np.dot(tang.mean(0), ref_dir) < 0:
            tang = -tang
            samp = samp[::-1]            # keep track length increasing with tangent
            tang = tang[::-1]

        tube_id += 1
        # cumulative arc length along THIS tube, in Å, starting at 0
        seg = np.linalg.norm(np.diff(samp, axis=0), axis=1)
        track = np.concatenate([[0.0], np.cumsum(seg)]) * pixel_size

        for k in range(len(samp)):
            records.append(dict(pos=samp[k], tan=tang[k],
                                tube=tube_id, track=track[k]))

    log(f"{len(records)} particles across {tube_id} tubes")

    # 6+7. Rotation per particle -> ZYZ Euler; assemble helical dataframe
    if micrograph is None:
        micrograph = os.path.basename(mask_path)

    df = pd.DataFrame(columns=STAR_HEADER)
    for i, r in enumerate(records):
        x, y, z = r['pos']
        rot, tilt, psi = matrix_to_euler(r['tan'], tangent_axis, invert_rot)
        df.loc[i, 'rlnCoordinateX'] = x
        df.loc[i, 'rlnCoordinateY'] = y
        df.loc[i, 'rlnCoordinateZ'] = z
        df.loc[i, 'rlnMicrographName'] = micrograph
        df.loc[i, 'rlnMagnification'] = magnification
        df.loc[i, 'rlnPixelSize'] = pixel_size
        df.loc[i, 'rlnGroupNumber'] = group_number
        df.loc[i, 'rlnAngleRot'] = rot
        df.loc[i, 'rlnAngleTilt'] = tilt
        df.loc[i, 'rlnAnglePsi'] = psi
        # priors seed the constrained helical search = the geometric angles
        df.loc[i, 'rlnAngleRotPrior'] = rot
        df.loc[i, 'rlnAngleTiltPrior'] = tilt
        df.loc[i, 'rlnAnglePsiPrior'] = psi
        df.loc[i, 'rlnHelicalTubeID'] = r['tube']
        df.loc[i, 'rlnHelicalTrackLength'] = r['track']

    starfile.write(df, star_out, overwrite=True)
    log(f"wrote {star_out}")

    # 8. Optional ChimeraX .bild overlay (opt-in; never affects the star output)
    if write_bild_file:
        if bild_path is None:
            bild_path = default_bild_path(star_out)
        write_bild(bild_path, records)
        log(f"wrote {bild_path}")

    return df, records


def default_bild_path(star_out, bild_dir=None):
    """Default ``.bild`` location: ``<star stem>.bild`` alongside the star file.

    ``bild_dir`` overrides the directory; the file name is derived from the star
    output's base name.
    """
    directory = bild_dir if bild_dir is not None else os.path.dirname(star_out)
    stem = os.path.splitext(os.path.basename(star_out))[0]
    return os.path.join(directory, stem + ".bild")
