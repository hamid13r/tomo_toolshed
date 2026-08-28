"""
Parsers for Warp/WarpTools metadata:

  * .tomostar          -> STAR file created by ts_import (per-tilt movie, angle,
                          axis angle, dose, optional per-tilt axis offsets).
  * per-tilt-series XML-> the authoritative geometry/CTF/warp model written by
                          ts_aretomo / ts_import_alignments / ts_ctf, loaded by
                          TiltSeries.LoadMeta (WarpLib/TiltSeries/TiltSeries.cs).
  * .settings          -> OptionsWarp; we only need the unbinned pixel size.

The `.xml` is the source of truth. The `.tomostar` is used only as a fallback
for the raw angles/dose when a fresh series has no .xml yet
(InitializeFromTomoStar in the C# source).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np

from .grids import CubicGrid, LinearGrid4D
from .ctf import CTFParams


# --------------------------------------------------------------------------- #
# STAR                                                                        #
# --------------------------------------------------------------------------- #
def read_star(path) -> dict:
    """Minimal RELION3/Warp STAR reader. Returns {column_name: list[str]}.

    Handles a single `loop_` block (which is what .tomostar uses)."""
    text = Path(path).read_text(errors="ignore").splitlines()
    cols, rows = [], []
    in_loop = False
    reading_rows = False
    for raw in text:
        line = raw.strip()
        if line == "" or line.startswith("#"):
            continue
        if line == "loop_":
            in_loop = True
            reading_rows = False
            cols = []
            continue
        if in_loop and line.startswith("_"):
            name = line.split()[0][1:]           # strip leading underscore
            # drop a trailing '#index' if present
            name = name.split("#")[0]
            cols.append(name)
            reading_rows = True
            continue
        if reading_rows:
            parts = line.split()
            if len(parts) >= len(cols):
                rows.append(parts[:len(cols)])
    out = {c: [r[i] for r in rows] for i, c in enumerate(cols)}
    return out


# --------------------------------------------------------------------------- #
# Tilt-series model                                                           #
# --------------------------------------------------------------------------- #
@dataclass
class TiltSeriesModel:
    """Everything needed to reproduce ts_reconstruct geometry & weighting."""
    name: str = "tomo"

    # global attributes
    are_angles_inverted: bool = False
    level_angle_x: float = 0.0
    level_angle_y: float = 0.0
    global_bfactor: float = 0.0
    global_weight: float = 1.0
    image_dims_A: np.ndarray = field(default_factory=lambda: np.zeros(2, np.float32))
    volume_dims_A: np.ndarray = field(default_factory=lambda: np.zeros(3, np.float32))

    # per-tilt arrays (length NTilts)
    angles: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float32))
    dose: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float32))
    use_tilt: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))
    axis_angles: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float32))
    axis_offset_x: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float32))
    axis_offset_y: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float32))
    movie_paths: list = field(default_factory=list)

    # base CTF (Cs, voltage, amplitude, pixel size, phase); per-tilt defocus is
    # taken from grid_ctf_defocus
    ctf: CTFParams = field(default_factory=CTFParams)

    # grids
    grid_ctf_defocus: CubicGrid = field(default_factory=CubicGrid.empty)       # um
    grid_ctf_defocus_delta: CubicGrid = field(default_factory=CubicGrid.empty) # um
    grid_ctf_defocus_angle: CubicGrid = field(default_factory=CubicGrid.empty) # deg
    grid_ctf_phase: CubicGrid = field(default_factory=CubicGrid.empty)
    grid_movement_x: CubicGrid = field(default_factory=CubicGrid.empty)        # A
    grid_movement_y: CubicGrid = field(default_factory=CubicGrid.empty)        # A
    grid_volume_warp_x: LinearGrid4D = field(default_factory=LinearGrid4D.empty)
    grid_volume_warp_y: LinearGrid4D = field(default_factory=LinearGrid4D.empty)
    grid_volume_warp_z: LinearGrid4D = field(default_factory=LinearGrid4D.empty)
    grid_angle_x: CubicGrid = field(default_factory=CubicGrid.empty)           # deg
    grid_angle_y: CubicGrid = field(default_factory=CubicGrid.empty)           # deg
    grid_angle_z: CubicGrid = field(default_factory=CubicGrid.empty)           # deg
    grid_dose_bfacs: CubicGrid = field(default_factory=CubicGrid.empty)        # A^2
    grid_dose_bfacs_delta: CubicGrid = field(default_factory=CubicGrid.empty)
    grid_dose_bfacs_angle: CubicGrid = field(default_factory=CubicGrid.empty)
    grid_dose_weights: CubicGrid = field(default_factory=CubicGrid.empty)
    grid_location_bfacs: CubicGrid = field(default_factory=CubicGrid.empty)
    grid_location_weights: CubicGrid = field(default_factory=CubicGrid.empty)

    @property
    def n_tilts(self):
        return len(self.angles)

    @property
    def min_dose(self):
        return float(self.dose.min()) if self.dose.size else 0.0

    @property
    def max_dose(self):
        return float(self.dose.max()) if self.dose.size else 0.0


def _floats(text):
    return np.array([float(v) for v in text.strip().split("\n") if v.strip() != ""],
                    dtype=np.float32)


def _bools(text):
    return np.array([v.strip().lower() == "true"
                     for v in text.strip().split("\n") if v.strip() != ""], dtype=bool)


def _attr_f(root, name, default):
    v = root.get(name)
    return float(v) if v is not None else default


def _cubic(root, tag):
    node = root.find(tag)
    return CubicGrid.from_xml(node) if node is not None else CubicGrid.empty()


def _linear(root, tag):
    node = root.find(tag)
    return LinearGrid4D.from_xml(node) if node is not None else LinearGrid4D.empty()


def load_tiltseries_xml(xml_path, tomostar_path=None, name=None) -> TiltSeriesModel:
    """Parse a per-tilt-series .xml (and optionally .tomostar) into a model.

    Mirrors TiltSeries.LoadMeta + InitializeFromTomoStar.
    """
    m = TiltSeriesModel()
    m.name = name or Path(xml_path).stem

    root = ET.parse(xml_path).getroot()

    # -- global attributes --
    m.are_angles_inverted = str(root.get("AreAnglesInverted", "False")).lower() == "true"
    m.level_angle_x = _attr_f(root, "LevelAngleX", 0.0)
    m.level_angle_y = _attr_f(root, "LevelAngleY", 0.0)
    m.global_bfactor = _attr_f(root, "Bfactor", 0.0)
    m.global_weight = _attr_f(root, "Weight", 1.0)

    def parse_vec(s, n):
        if s is None:
            return np.zeros(n, np.float32)
        parts = s.replace(",", " ").split()
        return np.array([float(p) for p in parts[:n]], np.float32)

    m.image_dims_A = parse_vec(root.get("ImageDimensionsAngstrom"), 2)
    m.volume_dims_A = parse_vec(root.get("VolumeDimensionsAngstrom"), 3)

    # -- per-tilt properties --
    def txt(tag):
        node = root.find(tag)
        return node.text if node is not None and node.text is not None else None

    if txt("Angles") is not None:
        m.angles = _floats(txt("Angles"))
    n = len(m.angles)
    m.dose = _floats(txt("Dose")) if txt("Dose") else np.zeros(n, np.float32)
    m.use_tilt = _bools(txt("UseTilt")) if txt("UseTilt") else np.ones(n, bool)
    m.axis_angles = _floats(txt("AxisAngle")) if txt("AxisAngle") else np.zeros(n, np.float32)
    m.axis_offset_x = _floats(txt("AxisOffsetX")) if txt("AxisOffsetX") else np.zeros(n, np.float32)
    m.axis_offset_y = _floats(txt("AxisOffsetY")) if txt("AxisOffsetY") else np.zeros(n, np.float32)
    mp = txt("MoviePath")
    m.movie_paths = [s.strip() for s in mp.split("\n") if s.strip()] if mp else []

    # -- base CTF --
    ctf_node = root.find("CTF")
    if ctf_node is not None:
        g = lambda k, d: float(ctf_node.get(k)) if ctf_node.get(k) is not None else d
        m.ctf = CTFParams(
            pixel_size=g("PixelSize", 1.0),
            voltage=g("Voltage", 300.0),
            cs=g("Cs", 2.7),
            amplitude=g("Amplitude", 0.07),
            defocus=g("Defocus", 3.0),
            defocus_delta=g("DefocusDelta", 0.0),
            defocus_angle=g("DefocusAngle", 0.0),
            phase_shift=g("PhaseShift", 0.0),
        )

    # -- grids --
    m.grid_ctf_defocus = _cubic(root, "GridCTF")
    m.grid_ctf_defocus_delta = _cubic(root, "GridCTFDefocusDelta")
    m.grid_ctf_defocus_angle = _cubic(root, "GridCTFDefocusAngle")
    m.grid_ctf_phase = _cubic(root, "GridCTFPhase")
    m.grid_movement_x = _cubic(root, "GridMovementX")
    m.grid_movement_y = _cubic(root, "GridMovementY")
    m.grid_volume_warp_x = _linear(root, "GridVolumeWarpX")
    m.grid_volume_warp_y = _linear(root, "GridVolumeWarpY")
    m.grid_volume_warp_z = _linear(root, "GridVolumeWarpZ")
    m.grid_angle_x = _cubic(root, "GridAngleX")
    m.grid_angle_y = _cubic(root, "GridAngleY")
    m.grid_angle_z = _cubic(root, "GridAngleZ")
    m.grid_dose_bfacs = _cubic(root, "GridDoseBfacs")
    m.grid_dose_bfacs_delta = _cubic(root, "GridDoseBfacsDelta")
    m.grid_dose_bfacs_angle = _cubic(root, "GridDoseBfacsAngle")
    m.grid_dose_weights = _cubic(root, "GridDoseWeights")
    m.grid_location_bfacs = _cubic(root, "GridLocationBfacs")
    m.grid_location_weights = _cubic(root, "GridLocationWeights")

    # fallback: if no angles in xml, read the tomostar
    if m.n_tilts <= 1 and tomostar_path is not None and Path(tomostar_path).exists():
        _init_from_tomostar(m, tomostar_path)

    return m


def _init_from_tomostar(m: TiltSeriesModel, tomostar_path):
    """Mirror TiltSeries.InitializeFromTomoStar."""
    star = read_star(tomostar_path)
    if "wrpAngleTilt" not in star or "wrpDose" not in star:
        raise ValueError("STAR file has no wrpDose or wrpAngleTilt column.")
    m.angles = np.array([float(v) for v in star["wrpAngleTilt"]], np.float32)
    m.dose = np.array([float(v) for v in star["wrpDose"]], np.float32)
    n = m.n_tilts
    m.axis_angles = (np.array([float(v) for v in star["wrpAxisAngle"]], np.float32)
                     if "wrpAxisAngle" in star else np.zeros(n, np.float32))
    if "wrpAxisOffsetX" in star and "wrpAxisOffsetY" in star:
        m.axis_offset_x = np.array([float(v) for v in star["wrpAxisOffsetX"]], np.float32)
        m.axis_offset_y = np.array([float(v) for v in star["wrpAxisOffsetY"]], np.float32)
    else:
        m.axis_offset_x = np.zeros(n, np.float32)
        m.axis_offset_y = np.zeros(n, np.float32)
    if "wrpMovieName" in star:
        m.movie_paths = list(star["wrpMovieName"])
    m.use_tilt = np.ones(n, bool)


def read_settings_params(settings_path):
    """Parse a Warp .settings XML into {ParamName: value_string}.

    Warp stores everything as <Param Name="X" Value="Y" />. Names are unique
    enough (PixelSize, DimensionsX/Y/Z, ...) that a flat dict is sufficient.
    """
    params = {}
    try:
        root = ET.parse(settings_path).getroot()
    except Exception:
        return params
    for el in root.iter("Param"):
        name = el.get("Name")
        val = el.get("Value")
        if name is not None and val is not None:
            params[name] = val
    return params


def read_settings_pixelsize(settings_path):
    """Return the unbinned pixel size (Angstrom) from a .settings XML, or None."""
    p = read_settings_params(settings_path)
    v = p.get("PixelSize") or p.get("PixelSizeMean")
    try:
        return float(v) if v is not None else None
    except ValueError:
        return None


def read_settings_tomo_dims(settings_path):
    """Return the tomogram box (DimensionsX, Y, Z) in *unbinned pixels* from the
    .settings <Tomo> block, or None. This is the box ts_reconstruct uses."""
    p = read_settings_params(settings_path)
    try:
        return (int(round(float(p["DimensionsX"]))),
                int(round(float(p["DimensionsY"]))),
                int(round(float(p["DimensionsZ"]))))
    except (KeyError, ValueError):
        return None
