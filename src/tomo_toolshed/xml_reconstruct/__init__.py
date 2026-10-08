"""
tomo_toolshed.xml_reconstruct -- a faithful, hackable NumPy/SciPy re-implementation of WarpTools'
`ts_reconstruct` tomogram reconstruction, built for experimenting with new
weighting and filtering.

Key modules:
    metadata   - parse .tomostar / per-tilt-series .xml / .settings
    geometry   - Warp's projection geometry (Matrix3, GetPositionInAllTilts)
    ctf        - the CTF equation (CTF.cs)
    weighting  - per-tilt weighted-CTF model  <-- edit here for new weighting
    filters    - preprocessing + deconvolution <-- edit here for new filtering
    reconstruct- the Fourier-slice reconstruction engine
    etomo      - IMOD `tilt` back-projection engine (WBP / SIRT)
    novactf    - real-space novaCTF 3D-CTF engine on Warp's geometry
    mrc_io     - MRC/PNG read/write
    parallel   - one thread count for everything (set_threads / --threads)
"""
from .metadata import (load_tiltseries_xml, TiltSeriesModel,
                       read_settings_pixelsize, read_settings_tomo_dims,
                       read_settings_params)
from .reconstruct import reconstruct, ReconOptions
from .etomo import reconstruct_etomo, EtomoOptions
from .novactf import reconstruct_novactf_rs, NovaCTFOptions
from .parallel import set_threads, get_threads
from .weighting import warp_weighting, make_dose_bfactor_weighting, motioncor3_dose_weighting

__all__ = [
    "load_tiltseries_xml", "TiltSeriesModel",
    "read_settings_pixelsize", "read_settings_tomo_dims", "read_settings_params",
    "reconstruct", "ReconOptions",
    "reconstruct_etomo", "EtomoOptions",
    "reconstruct_novactf_rs", "NovaCTFOptions",
    "set_threads", "get_threads",
    "warp_weighting", "make_dose_bfactor_weighting", "motioncor3_dose_weighting",
]
