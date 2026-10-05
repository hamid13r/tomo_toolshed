"""
Contrast transfer function, matching WarpLib/CTF.cs.

Reproduces the exact frequency-space CTF that Warp multiplies into the tilt data
during reconstruction, including the amplitude/B-factor *weighting* that is
folded into the same object (Scale, Bfactor) by TiltSeries.GetCTFsForOneParticle.

Conventions (from CTF.cs GetKs / Get1D and ToStruct):
    lambda [A]   = 12.2643247 / sqrt(V * (1 + V * 0.978466e-6)),  V in Volts
    K1 = pi * lambda
    K2 = pi/2 * Cs * lambda^3          (Cs in Angstrom = Cs_mm * 1e7)
    K3 = sqrt(1 - amplitude^2)
    argument(s) = K1 * deltaf(s) * s^2 + K2 * s^4 - phaseshift
    ctf(s)      = amplitude*cos(argument) - K3*sin(argument)
    ctf(s)     *= exp(Bfactor/4 * s^2) * Scale        (only when "weighted")

Frequencies s are in 1/Angstrom. In Get1D Warp uses deltaf = -(Defocus_um*1e4),
i.e. Angstrom with a sign flip; we reproduce that exactly.

Astigmatism: deltaf(s, phi) = defocus + 0.5*defocusdelta*cos(2*(phi - astig))
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class CTFParams:
    """Per-CTF parameters in Warp's *display* units (as stored in XML)."""
    pixel_size: float = 1.0          # Angstrom
    voltage: float = 300.0           # kV
    cs: float = 2.7                  # mm
    amplitude: float = 0.07          # amplitude contrast fraction
    defocus: float = 3.0             # micrometer (underfocus positive)
    defocus_delta: float = 0.0       # micrometer (astigmatism magnitude)
    defocus_angle: float = 0.0       # degrees
    phase_shift: float = 0.0         # fraction of pi
    bfactor: float = 0.0             # Angstrom^2 (<=0 attenuates high freq)
    bfactor_delta: float = 0.0       # Angstrom^2 (anisotropic B, magnitude)
    bfactor_angle: float = 0.0       # degrees
    scale: float = 1.0               # linear amplitude scale (dose/geom weight)

    def ks(self):
        V = self.voltage * 1e3
        lam = 12.2643247 / np.sqrt(V * (1.0 + V * 0.978466e-6))       # Angstrom
        cs = self.cs * 1e7                                            # mm -> A
        K1 = np.pi * lam
        K2 = np.pi * 0.5 * cs * lam ** 3
        K3 = np.sqrt(max(0.0, 1.0 - self.amplitude ** 2))
        K4 = self.bfactor * 0.25                                     # A^2
        return K1, K2, K3, K4


def ctf_2d(params: CTFParams, sx: np.ndarray, sy: np.ndarray,
           weighted: bool = True, do_bfactor: bool = True) -> np.ndarray:
    """Evaluate the (real-valued) CTF on a 2D frequency grid.

    sx, sy : spatial frequencies in 1/Angstrom (same shape).
    weighted : if True, apply Scale and the (dose) B-factor exposure weighting;
               if False, return the bare CTF (which, abs'd, becomes the
               reconstruction weighting term -- see ReconstructFull).
    """
    K1, K2, K3, K4 = params.ks()

    s2 = sx * sx + sy * sy
    s4 = s2 * s2

    defocus_A = -(params.defocus * 1e4)          # matches Get1D: -(Defocus_um*1e4)
    if params.defocus_delta != 0.0:
        phi = np.arctan2(sy, sx)
        astig = np.deg2rad(params.defocus_angle)
        delta_A = -(params.defocus_delta * 1e4)
        deltaf = defocus_A + 0.5 * delta_A * np.cos(2.0 * (phi - astig))
    else:
        deltaf = defocus_A

    phaseshift = params.phase_shift * np.pi
    argument = K1 * deltaf * s2 + K2 * s4 - phaseshift
    ctf = params.amplitude * np.cos(argument) - K3 * np.sin(argument)

    if weighted:
        if do_bfactor:
            env = np.exp(K4 * s2)
            if params.bfactor_delta != 0.0:
                phi = np.arctan2(sy, sx)
                bangle = np.deg2rad(params.bfactor_angle)
                bdelta = params.bfactor_delta * 0.25
                env = env * np.exp(bdelta * s2 * np.cos(2.0 * (phi - bangle)))
            ctf = ctf * env
        ctf = ctf * params.scale

    return ctf.astype(np.float32)


def frequency_grid(size: int, angpix: float):
    """Return (sx, sy) in 1/Angstrom for an FFT of a `size`x`size` image,
    in numpy fftfreq layout (matching np.fft.fft2 output, not shifted)."""
    f = np.fft.fftfreq(size, d=angpix)     # cycles/Angstrom
    sx, sy = np.meshgrid(f, f, indexing="xy")
    return sx.astype(np.float32), sy.astype(np.float32)
