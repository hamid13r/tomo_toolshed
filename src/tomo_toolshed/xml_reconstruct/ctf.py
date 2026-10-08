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

`dose_model="motioncor3"` replaces the Bfactor exposure-filter envelope above
with the Grant & Grigorieff (2015) critical-exposure curve, exactly as
MotionCor3 applies it to frames (Correct/GWeightFrame.cu:mGCalcWeight) -- see
`motioncor3_dose_envelope` below. See weighting.py:motioncor3_dose_weighting
for the per-tilt weighting scheme that uses it.
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
    dose_model: str = "gaussian"     # "gaussian" (the Bfactor envelope above) or
                                      # "motioncor3" (critical-exposure curve below)
    dose_ea2: float = 0.0            # accumulated dose (e-/A^2); only used by "motioncor3"

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
        ctf = ctf * weight_envelope(params, sx, sy, do_bfactor=do_bfactor)

    return ctf.astype(np.float32)


def weight_envelope(params: CTFParams, sx: np.ndarray, sy: np.ndarray,
                    do_bfactor: bool = True) -> np.ndarray:
    """The "weighted" part of ctf_2d on its own: Scale times the dose exposure
    filter (B-factor envelope, or the MotionCor3 critical-exposure curve).
    ctf_2d(weighted=True) == ctf_2d(weighted=False) * weight_envelope."""
    s2 = sx * sx + sy * sy
    env = np.ones_like(s2)
    if do_bfactor:
        if params.dose_model == "motioncor3":
            env = motioncor3_dose_envelope(s2, params.dose_ea2, params.voltage)
        else:
            env = np.exp(params.bfactor * 0.25 * s2)
            if params.bfactor_delta != 0.0:
                phi = np.arctan2(sy, sx)
                bangle = np.deg2rad(params.bfactor_angle)
                bdelta = params.bfactor_delta * 0.25
                env = env * np.exp(bdelta * s2 * np.cos(2.0 * (phi - bangle)))
    return env * params.scale


def _motioncor3_kv_factor(voltage_kv: float) -> float:
    """Voltage rescaling of the (300kV-calibrated) critical-dose curve,
    matching MotionCor3's GWeightFrame.cu:BuildWeight exactly."""
    if voltage_kv >= 300:
        return 1.0
    if voltage_kv >= 200:
        return 0.002 * (voltage_kv - 200) + 0.8
    if voltage_kv >= 120:
        return 0.004375 * (voltage_kv - 120) + 0.45
    return 0.45


def motioncor3_dose_envelope(s2: np.ndarray, dose_ea2: float, voltage_kv: float) -> np.ndarray:
    """Grant & Grigorieff (2015) critical-exposure curve, as used by MotionCor3's
    per-frame dose weighting (Correct/GWeightFrame.cu:mGCalcWeight):

        Ncrit(s) = 0.24499 * s^-1.6649 + 2.8141    [e-/A^2], s in 1/A
        weight(s) = exp(-0.5 * dose / (Ncrit(s) * kv_factor))

    Unlike the Gaussian Bfactor envelope (a fixed exp(-k*s^2) shape whose only
    freedom is a per-tilt scale), this curve's fall-off flattens at low s and
    steepens at high s, matching the empirically measured resolution-dependent
    radiation damage rate rather than approximating it with one Gaussian.
    dose_ea2 is the tilt's *accumulated* dose (same quantity as Warp's own
    `Dose[t]`, i.e. model.dose[t] -- not the per-tilt increment).
    """
    s = np.sqrt(np.maximum(s2, 1e-8))
    crit_dose = (0.24499 * np.power(s, -1.6649) + 2.8141) * _motioncor3_kv_factor(voltage_kv)
    return np.exp(-0.5 * dose_ea2 / crit_dose).astype(np.float32)


def frequency_grid(size: int, angpix: float):
    """Return (sx, sy) in 1/Angstrom for an FFT of a `size`x`size` image,
    in numpy fftfreq layout (matching np.fft.fft2 output, not shifted)."""
    f = np.fft.fftfreq(size, d=angpix)     # cycles/Angstrom
    sx, sy = np.meshgrid(f, f, indexing="xy")
    return sx.astype(np.float32), sy.astype(np.float32)
