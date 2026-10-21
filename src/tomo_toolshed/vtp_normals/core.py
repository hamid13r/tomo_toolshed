"""VTP read/write, piece labelling and normal flipping for pycurv surfaces.

This module is the single source of piece numbering shared by the ``preview``,
``curate`` and ``flip`` subcommands, so a piece id means the same thing in all
three. It deliberately imports no matplotlib, so headless tooling (and the CLI
itself) can import it without pulling in a plotting backend.

Pieces are the connected parts of the surface (triangles sharing a vertex),
each split again where a per-triangle cell array -- the ``--split-column``,
``OMM_dist`` by default -- crosses ``--split-dist``, so parts near and far from
that threshold become separate pieces. Small split parts (< ``--min-triangles``)
are merged into the neighbour they touch most. Pieces are numbered largest
first. The same ``--split-column`` / ``--split-dist`` / ``--min-triangles`` must
be passed to ``flip`` as to the ``preview`` / ``curate`` you read ids from, or
the ids will not line up.
"""

import glob
import os
import re

import click
import numpy as np
import vtk
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

# Defaults shared by preview, curate and flip, so piece numbers mean the same
# thing in all three.
SPLIT_DIST = 20.0  # split pieces where the split column crosses this (surface units, nm); 0 = no split
MIN_TRIANGLES = 50  # smaller pieces are not shown/flipped; smaller split parts are merged away
DEFAULT_SPLIT_COLUMN = "OMM_dist"  # per-triangle cell array pieces are split on (OMM-IMM distance)
DEFAULT_NORMAL_ARRAYS = ("n_v", "normal", "avg_normals")  # negated when a piece is flipped

# Cell arrays updated when a piece's normal is flipped (missing arrays are skipped).
CURVATURE_NEGATE = ("mean_curvature", "mean_curvature_VV", "shape_index_VV", "shape_index_cat")
CURVATURE_SWAP_NEGATE = (("kappa_1", "kappa_2"), ("max_curvature", "min_curvature"))
CURVATURE_SWAP = (("t_1", "t_2"),)


def read_polydata(vtp_path):
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(vtp_path)
    reader.Update()
    return reader.GetOutput()


def load_surface(vtp_path, normal_array, poly=None):
    """Return (triangles (M, 3), centres (M, 3), normals (M, 3)) from a pycurv vtp.

    Pass an already-read ``poly`` (vtkPolyData) to skip reading the file again.
    """
    if poly is None:
        poly = read_polydata(vtp_path)
    offsets = vtk_to_numpy(poly.GetPolys().GetOffsetsArray())
    if np.any(np.diff(offsets) != 3):
        raise click.ClickException(f"{vtp_path}: surface is not made only of triangles")
    triangles = vtk_to_numpy(poly.GetPolys().GetConnectivityArray()).reshape(-1, 3)
    cell_data = poly.GetCellData()
    if cell_data.GetArray("xyz") is not None:
        centres = vtk_to_numpy(cell_data.GetArray("xyz")).astype(np.float64)
    else:
        points = vtk_to_numpy(poly.GetPoints().GetData()).astype(np.float64)
        centres = points[triangles].mean(axis=1)
    if cell_data.GetArray(normal_array) is None:
        names = [cell_data.GetArrayName(i) for i in range(cell_data.GetNumberOfArrays())]
        raise click.ClickException(f"{vtp_path}: no cell array '{normal_array}' (have {names})")
    normals = vtk_to_numpy(cell_data.GetArray(normal_array)).astype(np.float64)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    return triangles, centres, normals


def label_pieces(triangles, split_values=None, split_dist=0.0, merge_below=0):
    """Return (piece label per triangle, piece sizes), pieces numbered largest first.

    Triangles sharing a vertex are connected. With ``split_values`` (one per
    triangle, e.g. the ``--split-column`` array) and ``split_dist`` > 0,
    triangles on one side of the threshold are never connected to those on the
    other, so every connected piece splits into a near and a far part; parts
    smaller than ``merge_below`` triangles are then merged into the neighbouring
    part they share the most vertex contacts with (removes speckle along the
    threshold).
    """
    n_tri = len(triangles)
    n_pts = int(triangles.max()) + 1
    if split_values is None or split_dist <= 0:
        # bipartite triangle-vertex graph: nodes 0..n_tri-1 are triangles, then vertices
        rows = np.repeat(np.arange(n_tri), 3)
        cols = triangles.ravel() + n_tri
        graph = coo_matrix((np.ones(rows.size, dtype=np.int8), (rows, cols)),
                           shape=(n_tri + n_pts, n_tri + n_pts))
        _, labels = connected_components(graph, directed=False)
        labels = labels[:n_tri]
    else:
        incidence = csr_matrix((np.ones(3 * n_tri, dtype=np.int32),
                                (np.repeat(np.arange(n_tri), 3), triangles.ravel())), shape=(n_tri, n_pts))
        adjacency = (incidence @ incidence.T).tocoo()  # triangle pairs sharing >= 1 vertex
        near = np.asarray(split_values) < split_dist
        same = near[adjacency.row] == near[adjacency.col]
        graph = coo_matrix((np.ones(same.sum(), dtype=np.int8), (adjacency.row[same], adjacency.col[same])),
                           shape=(n_tri, n_tri))
        _, labels = connected_components(graph, directed=False)
        if merge_below > 0:
            labels = _merge_small_parts(labels, adjacency, merge_below)
    # relabel by size, largest piece first
    uniq, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
    order = np.argsort(-counts, kind="stable")
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    return rank[inverse], counts[order]


def _merge_small_parts(labels, adjacency, merge_below):
    """Merge parts smaller than merge_below into their most-touching neighbouring part."""
    n = int(labels.max()) + 1
    sizes = np.bincount(labels, minlength=n)
    a, b = labels[adjacency.row], labels[adjacency.col]
    cross = a != b
    contacts = csr_matrix((np.ones(cross.sum()), (a[cross], b[cross])), shape=(n, n))  # duplicates summed
    parent = np.arange(n)
    size = sizes.copy()

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for part in np.argsort(sizes, kind="stable"):
        if sizes[part] >= merge_below:
            break
        r = root(part)
        if size[r] >= merge_below:
            continue  # already grown big enough by earlier merges
        start, stop = contacts.indptr[part], contacts.indptr[part + 1]
        weight = {}
        for neighbour, w in zip(contacts.indices[start:stop], contacts.data[start:stop]):
            nr = root(neighbour)
            if nr != r:
                weight[nr] = weight.get(nr, 0) + w
        if not weight:
            continue  # isolated small piece: stays its own (small) piece
        target = max(weight, key=lambda k: (weight[k], size[k]))
        parent[r] = target
        size[target] += size[r]
    return np.array([root(i) for i in range(n)])[labels]


def surface_pieces(poly, triangles, split_dist=SPLIT_DIST, min_triangles=MIN_TRIANGLES,
                   split_column=DEFAULT_SPLIT_COLUMN):
    """label_pieces() for a pycurv surface, splitting on its ``split_column`` cell array if present."""
    arr = poly.GetCellData().GetArray(split_column)
    if split_dist > 0 and arr is None:
        click.echo(f"  (no '{split_column}' array: pieces are not split)")
    split_values = vtk_to_numpy(arr) if arr is not None else None
    return label_pieces(triangles, split_values, split_dist, min_triangles)


def piece_split_median(poly, labels, n_pieces, split_column=DEFAULT_SPLIT_COLUMN):
    """Median of the ``split_column`` array per piece (NaN if the surface has no such array)."""
    arr = poly.GetCellData().GetArray(split_column)
    if arr is None:
        return np.full(n_pieces, np.nan)
    values = vtk_to_numpy(arr)
    order = np.argsort(labels, kind="stable")
    bounds = np.searchsorted(labels[order], np.arange(n_pieces + 1))
    return np.array([np.median(values[order[bounds[i]:bounds[i + 1]]]) for i in range(n_pieces)])


def find_tomogram(vtp_path, tomo_dir, tomo_pattern):
    stem = re.match(r"(.+?_ts_\d+)", os.path.basename(vtp_path))
    if stem is None:
        return None
    path = os.path.join(tomo_dir, tomo_pattern.format(stem=stem.group(1)))
    return path if os.path.exists(path) else None


def parse_flips(flips, flip_file):
    """Return a list of (ts, membrane, set of piece ids) from CLI entries and the flip file."""
    entries = list(flips)
    if flip_file:
        with open(flip_file) as f:
            for line in f:
                line = line.split("#", 1)[0].strip()
                if line:
                    entries.append(line)
    parsed = []
    for entry in entries:
        parts = entry.split(":")
        if len(parts) != 3:
            raise click.BadParameter(f"'{entry}' is not TS:MEMBRANE:IDS")
        ts, membrane, ids = (p.strip() for p in parts)
        try:
            piece_ids = {int(i) for i in ids.split(",") if i.strip()}
        except ValueError:
            raise click.BadParameter(f"'{entry}': piece ids must be integers")
        parsed.append((ts, membrane, piece_ids))
    return parsed


def find_vtp(vtp_dir, ts, membrane):
    matches = [p for p in sorted(glob.glob(os.path.join(vtp_dir, "*.vtp")))
               if ".flipped." not in os.path.basename(p)
               and ts in os.path.basename(p) and f"_{membrane}." in os.path.basename(p)]
    if len(matches) != 1:
        raise click.ClickException(f"'{ts}:{membrane}' matches {len(matches)} vtp files in {vtp_dir} "
                                   f"(need exactly 1): {[os.path.basename(m) for m in matches]}")
    return matches[0]


def flip_cell_arrays(cell_data, flip, normal_arrays, curvature):
    """Negate the normal arrays (and update curvatures) in place for cells where flip is True."""
    def view(name):
        vtk_array = cell_data.GetArray(name)
        if vtk_array is None:
            click.echo(f"  (no cell array '{name}', skipped)")
            return None
        vtk_array.Modified()
        return vtk_to_numpy(vtk_array)  # view into the vtk array, edited in place

    negate = list(normal_arrays) + (list(CURVATURE_NEGATE) if curvature else [])
    for name in negate:
        values = view(name)
        if values is not None:
            values[flip] *= -1
    if not curvature:
        return
    for pairs, sign in ((CURVATURE_SWAP_NEGATE, -1), (CURVATURE_SWAP, 1)):
        for name_a, name_b in pairs:
            a, b = view(name_a), view(name_b)
            if a is None or b is None:
                continue
            old_a = a[flip].copy()
            a[flip] = sign * b[flip]
            b[flip] = sign * old_a


def flip_surface(vtp_path, piece_ids, arrays=DEFAULT_NORMAL_ARRAYS, curvature=True,
                 split_dist=SPLIT_DIST, min_triangles=MIN_TRIANGLES, split_column=DEFAULT_SPLIT_COLUMN):
    """Write <vtp>.flipped.vtp with the given pieces flipped; return its path."""
    poly = read_polydata(vtp_path)
    triangles = vtk_to_numpy(poly.GetPolys().GetConnectivityArray()).reshape(-1, 3).copy()
    labels, sizes = surface_pieces(poly, triangles, split_dist, min_triangles, split_column)
    bad = sorted(i for i in piece_ids if i >= len(sizes))
    if bad:
        raise click.ClickException(f"{os.path.basename(vtp_path)} has pieces 0-{len(sizes) - 1}, "
                                   f"no piece {bad}")
    flip = np.isin(labels, sorted(piece_ids))
    for i in sorted(piece_ids):
        click.echo(f"  flipping piece {i} ({sizes[i]} triangles)")

    cell_data = poly.GetCellData()
    flip_cell_arrays(cell_data, flip, arrays, curvature)

    # reverse the winding so the geometric normal agrees with the flipped normal arrays
    triangles[flip] = triangles[flip][:, ::-1]
    offsets = np.arange(0, triangles.size + 1, 3, dtype=np.int64)
    cells = vtk.vtkCellArray()
    cells.SetData(numpy_to_vtk(offsets, deep=1, array_type=vtk.VTK_ID_TYPE),
                  numpy_to_vtk(triangles.ravel().astype(np.int64), deep=1, array_type=vtk.VTK_ID_TYPE))
    poly.SetPolys(cells)

    for name, values, vtk_type in (("piece_id", labels.astype(np.int32), vtk.VTK_INT),
                                   ("normal_flipped", flip.astype(np.uint8), vtk.VTK_UNSIGNED_CHAR)):
        vtk_array = numpy_to_vtk(values, deep=1, array_type=vtk_type)
        vtk_array.SetName(name)
        cell_data.RemoveArray(name)
        cell_data.AddArray(vtk_array)

    out_path = os.path.splitext(vtp_path)[0] + ".flipped.vtp"
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(out_path)
    writer.SetInputData(poly)
    writer.SetDataModeToAppended()
    writer.SetCompressorTypeToZLib()
    if not writer.Write():
        raise click.ClickException(f"failed to write {out_path}")
    click.echo(f"  saved {out_path} ({flip.sum()} of {len(flip)} triangles flipped)")
    return out_path
