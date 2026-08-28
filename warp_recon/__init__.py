"""
warp_recon -- a faithful, hackable NumPy/SciPy re-implementation of WarpTools'
`ts_reconstruct` tomogram reconstruction, built for experimenting with new
weighting and filtering.

Key modules:
    metadata   - parse .tomostar / per-tilt-series .xml / .settings
    geometry   - Warp's projection geometry (Matrix3, GetPositionInAllTilts)
    ctf        - the CTF equation (CTF.cs)
    weighting  - per-tilt weighted-CTF model  <-- edit here for new weighting
    filters    - preprocessing + deconvolution <-- edit here for new filtering
    reconstruct- the Fourier-slice reconstruction engine
    mrc_io     - MRC/PNG read/write
"""
from .metadata import (load_tiltseries_xml, TiltSeriesModel,
                       read_settings_pixelsize, read_settings_tomo_dims,
                       read_settings_params)
from .reconstruct import reconstruct, ReconOptions
from .weighting import warp_weighting, make_dose_bfactor_weighting

__all__ = [
    "load_tiltseries_xml", "TiltSeriesModel",
    "read_settings_pixelsize", "read_settings_tomo_dims", "read_settings_params",
    "reconstruct", "ReconOptions",
    "warp_weighting", "make_dose_bfactor_weighting",
]
