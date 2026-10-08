"""
Projection geometry, matching WarpLib/Tools/Matrix3.cs and
TiltSeries.GetPositionInAllTilts / GetAngleInAllTilts.

All rotation matrices use Warp's exact element layout so that `R @ v` reproduces
Warp's `Matrix3 * float3`. Rows of the returned 3x3 are (M1., M2., M3.).
"""
from __future__ import annotations
import numpy as np

DEG = np.pi / 180.0


# --------------------------------------------------------------------------- #
# Elementary rotations (exact transcription of Matrix3.RotateX/Y/Z)           #
# --------------------------------------------------------------------------- #
def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0],
                     [0, c, -s],
                     [0, s, c]], dtype=np.float64)


def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]], dtype=np.float64)


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0],
                     [s, c, 0],
                     [0, 0, 1]], dtype=np.float64)


def euler(rot, tilt, psi):
    """Matrix3.Euler(rot, tilt, psi) -- exact transcription."""
    ca, cb, cg = np.cos(rot), np.cos(tilt), np.cos(psi)
    sa, sb, sg = np.sin(rot), np.sin(tilt), np.sin(psi)
    cc, cs, sc, ss = cb * ca, cb * sa, sb * ca, sb * sa
    return np.array([
        [cg * cc - sg * sa,  cg * cs + sg * ca, -cg * sb],
        [-sg * cc - cg * sa, -sg * cs + cg * ca, sg * sb],
        [sc,                 ss,                 cb],
    ], dtype=np.float64)


def tilt_matrix(model, t):
    """TiltMatrix[t] = Euler(0,(Angle+LevelY),-Axis) * RotateX(LevelX)."""
    A = euler(0.0,
              (model.angles[t] + model.level_angle_y) * DEG,
              -model.axis_angles[t] * DEG)
    return A @ rot_x(model.level_angle_x * DEG)


def tilt_matrix_flipped(model, t):
    return (euler(0.0,
                  -(model.angles[t] + model.level_angle_y) * DEG,
                  -model.axis_angles[t] * DEG)
            @ rot_x(-model.level_angle_x * DEG))


# --------------------------------------------------------------------------- #
# Per-tilt rotation used for the (Fourier) central-slice insertion.           #
#                                                                             #
# GetAngleInAllTilts returns EulerFromMatrix(CorrectionMatrix @ TiltMatrix),  #
# and the Projector reconstructs with Euler(that) == the same matrix. So the  #
# net rotation the reconstruction "sees" for tilt t is exactly:               #
#     R_t = CorrectionMatrix(coord,t) @ TiltMatrix(t)                         #
# We evaluate the (usually trivial) angle-correction grids at `coord`.        #
# --------------------------------------------------------------------------- #
def rotation_for_tilt(model, t, coord_phys=None):
    if coord_phys is None:
        coord_phys = model.volume_dims_A / 2.0
    Vx = model.volume_dims_A[0]
    gstep = 1.0 / max(model.n_tilts - 1, 1)
    gc = np.array([[coord_phys[0] / Vx, coord_phys[1] / Vx, t * gstep]])
    gx = model.grid_angle_x.interpolate(gc)[0]
    gy = model.grid_angle_y.interpolate(gc)[0]
    gz = model.grid_angle_z.interpolate(gc)[0]
    corr = rot_z(gz * DEG) @ rot_y(gy * DEG) @ rot_x(gx * DEG)
    return corr @ tilt_matrix(model, t)


# --------------------------------------------------------------------------- #
# 3D physical coord -> per-tilt image position + defocus.                     #
# Exact transcription of GetPositionInAllTilts (single coord, all tilts).     #
# Returns:                                                                     #
#   xy   : (NTilts, 2) image position in Angstrom (before /pixelsize)         #
#   defocus_um : (NTilts,) defocus in micrometers                             #
# --------------------------------------------------------------------------- #
def positions_in_all_tilts(model, coord_phys, size_rounding=(1.0, 1.0, 1.0)):
    n = model.n_tilts
    Vx, Vy, Vz = model.volume_dims_A
    vol_center = model.volume_dims_A / 2.0
    img_center = model.image_dims_A / 2.0
    gstep = 1.0 / max(n - 1, 1)
    dose_span = (model.max_dose - model.min_dose) or 1.0

    coord = np.asarray(coord_phys, dtype=np.float64)
    centered0 = coord - vol_center

    # volume-warp (usually identity)
    dose_frac = (model.dose - model.min_dose) / dose_span
    warp_coords = np.stack([np.full(n, coord[0] / Vx),
                            np.full(n, coord[1] / Vy),
                            np.full(n, coord[2] / Vz),
                            dose_frac], axis=1)
    wx = model.grid_volume_warp_x.interpolate(warp_coords)
    wy = model.grid_volume_warp_y.interpolate(warp_coords)
    wz = model.grid_volume_warp_z.interpolate(warp_coords)

    xy = np.zeros((n, 2), np.float64)
    zdef = np.zeros(n, np.float64)
    trans_norm = np.zeros((n, 3), np.float64)
    for t in range(n):
        R = tilt_matrix(model, t)
        centered = centered0 + np.array([wx[t], wy[t], wz[t]])
        transformed = R @ centered
        tx = transformed[0] + model.axis_offset_x[t] + img_center[0]
        ty = transformed[1] + model.axis_offset_y[t] + img_center[1]
        zc = transformed[2]
        if model.are_angles_inverted:
            Rf = tilt_matrix_flipped(model, t)
            cflip = centered.copy(); cflip[2] *= -1
            zc = (Rf @ cflip)[2]
        xy[t] = (tx, ty)
        zdef[t] = zc
        trans_norm[t] = (tx / model.image_dims_A[0],
                         ty / model.image_dims_A[1],
                         t * gstep)

    # local motion grid (per-tilt / local shift), subtracted
    mvx = model.grid_movement_x.interpolate(trans_norm)
    mvy = model.grid_movement_y.interpolate(trans_norm)
    xy[:, 0] -= mvx
    xy[:, 1] -= mvy

    # per-tilt defocus (um) from grid + local z contribution (A -> um)
    dc = np.stack([np.full(n, coord[0] / Vx),
                   np.full(n, coord[1] / Vy),
                   np.arange(n) * gstep], axis=1)
    gdef = model.grid_ctf_defocus.interpolate(dc)  # micrometer
    defocus_um = gdef + 1e-4 * zdef

    sr = np.asarray(size_rounding, np.float64)
    xy *= sr[:2]
    defocus_um *= sr[2]
    return xy, defocus_um


# --------------------------------------------------------------------------- #
# Vectorized GetPositionInAllTilts for ONE tilt and MANY points.              #
# Same math as positions_in_all_tilts, line for line, so the real-space       #
# 3D-CTF engine (novactf.py) can project every voxel through Warp's full        #
# geometry -- including the local-motion grids -- instead of only the        #
# tomogram centre. Checked against positions_in_all_tilts in validate.py   #
# (tests/xml_reconstruct/validate.py).                                       #
# Returns x_A, y_A (image position, Angstrom) and defocus_um, each (M,).     #
# --------------------------------------------------------------------------- #
def positions_one_tilt(model, t, coords_phys, size_rounding=(1.0, 1.0, 1.0),
                       local_motion=True):
    n = model.n_tilts
    Vx, Vy, Vz = (float(v) for v in model.volume_dims_A)
    vol_center = np.asarray(model.volume_dims_A, np.float64) / 2.0
    img_center = np.asarray(model.image_dims_A, np.float64) / 2.0
    gstep = 1.0 / max(n - 1, 1)
    dose_span = (model.max_dose - model.min_dose) or 1.0

    coord = np.atleast_2d(np.asarray(coords_phys, dtype=np.float64))
    m = coord.shape[0]

    dose_frac = (model.dose[t] - model.min_dose) / dose_span
    warp_coords = np.stack([coord[:, 0] / Vx, coord[:, 1] / Vy, coord[:, 2] / Vz,
                            np.full(m, dose_frac)], axis=1)
    centered = coord - vol_center
    centered = centered + np.stack([model.grid_volume_warp_x.interpolate(warp_coords),
                                    model.grid_volume_warp_y.interpolate(warp_coords),
                                    model.grid_volume_warp_z.interpolate(warp_coords)],
                                   axis=1)

    R = tilt_matrix(model, t)
    transformed = centered @ R.T
    tx = transformed[:, 0] + model.axis_offset_x[t] + img_center[0]
    ty = transformed[:, 1] + model.axis_offset_y[t] + img_center[1]
    if model.are_angles_inverted:
        cflip = centered.copy()
        cflip[:, 2] *= -1
        zc = cflip @ tilt_matrix_flipped(model, t)[2]
    else:
        zc = transformed[:, 2]

    if local_motion:
        trans_norm = np.stack([tx / model.image_dims_A[0], ty / model.image_dims_A[1],
                               np.full(m, t * gstep)], axis=1)
        tx = tx - model.grid_movement_x.interpolate(trans_norm)
        ty = ty - model.grid_movement_y.interpolate(trans_norm)

    gx_, gy_, _gz = model.grid_ctf_defocus.dims
    if gx_ <= 1 and gy_ <= 1:
        # per-tilt-only defocus grid (the usual case): one value for every
        # point, so evaluate it once instead of a cubic spline per point
        gdef = model.grid_ctf_defocus.interpolate(np.array([[0.5, 0.5, t * gstep]]))[0]
    else:
        dc = np.stack([coord[:, 0] / Vx, coord[:, 1] / Vy, np.full(m, t * gstep)], axis=1)
        gdef = model.grid_ctf_defocus.interpolate(dc)
    defocus_um = gdef + 1e-4 * zc

    sr = np.asarray(size_rounding, np.float64)
    return tx * sr[0], ty * sr[1], defocus_um * sr[2]
