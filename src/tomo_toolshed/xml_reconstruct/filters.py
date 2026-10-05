"""
Real-space preprocessing and Fourier-space filtering -- THE place to experiment
with "different filtering".

Two groups:
  * preprocess_tilt(...)  : per-tilt-image preprocessing done before back-
    projection, matching the "Load and preprocess" block of ReconstructFull
    (subtract mean, rectangular mask, high-pass, normalize, contrast invert).
  * deconvolve(...)       : the optional post-reconstruction deconvolution
    (approximates GPU.DeconvolveCTF using the documented strength/falloff/
    highpass parameters).

Swap either by passing your own function into reconstruct().
"""
from __future__ import annotations
import numpy as np


# --------------------------------------------------------------------------- #
# Per-tilt preprocessing                                                      #
# --------------------------------------------------------------------------- #
def preprocess_tilt(img, angpix, highpass_px, normalize=True, invert=True,
                    mask_edge=16, mask_border=32):
    """Mirror ReconstructFull preprocessing for one tilt image (float32, 2D).

    highpass_px : the Bandpass low-frequency cutoff Warp uses,
                  1/(SubVolumeSize * SubVolumePadding / 2) in cycles/pixel.
                  Here expressed as a fraction of Nyquist via `highpass_px`.
    """
    out = img.astype(np.float32, copy=True)
    if normalize:
        # subtract a smooth background (approx SubtractMeanGrid(1))
        out = out - out.mean()
        # soft rectangular mask toward the borders (MaskRectangularly)
        out = _mask_rectangular(out, mask_border, mask_edge)
        # high-pass (Bandpass low cutoff .. 1)
        out = _highpass(out, highpass_px)
        # normalize to zero mean / unit std
        std = out.std()
        if std > 0:
            out = (out - out.mean()) / std
    if invert:
        out = out * -1.0
    return out.astype(np.float32)


def _mask_rectangular(img, border, edge):
    ny, nx = img.shape
    def ramp(n, b, e):
        w = np.ones(n, np.float32)
        for i in range(n):
            d = min(i, n - 1 - i)
            if d < b:
                x = np.clip((d - (b - e)) / max(e, 1), 0, 1)
                w[i] = 0.5 - 0.5 * np.cos(np.pi * x)
        return w
    wy = ramp(ny, border, edge)
    wx = ramp(nx, border, edge)
    return img * np.outer(wy, wx)


def _highpass(img, cutoff_frac):
    """Gaussian-ish high-pass; cutoff_frac in fraction of Nyquist (0..1)."""
    ny, nx = img.shape
    fy = np.fft.fftfreq(ny)[:, None] * 2.0   # in Nyquist units
    fx = np.fft.fftfreq(nx)[None, :] * 2.0
    r = np.sqrt(fy ** 2 + fx ** 2)
    hp = 1.0 - np.exp(-(r / max(cutoff_frac, 1e-6)) ** 2)
    F = np.fft.fft2(img) * hp
    return np.real(np.fft.ifft2(F)).astype(np.float32)


# --------------------------------------------------------------------------- #
# Deconvolution (post-reconstruction)                                         #
# --------------------------------------------------------------------------- #
def deconvolve(vol, angpix, ctf_params, strength=1.0, falloff=1.0, highpass=300.0):
    """Approximate Warp's GPU.DeconvolveCTF Wiener-like filter.

    Applies, in 3D Fourier space, a filter of the form
        d(s) = CTF(s) / (CTF(s)^2 + 1/SSNR(s))
    with SSNR(s) = 10^(3*strength) * highpass_ramp(s) * exp(-falloff*...),
    the model used by Warp desktop's deconvolution. This is a close, tunable
    approximation of the native routine (documented as such).
    """
    from .ctf import CTFParams, ctf_2d
    nz, ny, nx = vol.shape
    fz = np.fft.fftfreq(nz, d=angpix)[:, None, None]
    fy = np.fft.fftfreq(ny, d=angpix)[None, :, None]
    fx = np.fft.fftfreq(nx, d=angpix)[None, None, :]
    s = np.sqrt(fx ** 2 + fy ** 2 + fz ** 2).astype(np.float32)

    # radial CTF magnitude at the central-tilt defocus (isotropic)
    p = CTFParams(**{**ctf_params.__dict__})
    p.scale = 1.0
    p.bfactor = 0.0
    K1, K2, K3, K4 = p.ks()
    s2 = s * s
    defocus_A = -(p.defocus * 1e4)
    arg = K1 * defocus_A * s2 + K2 * s2 * s2 - p.phase_shift * np.pi
    ctf = np.abs(p.amplitude * np.cos(arg) - K3 * np.sin(arg))

    # SSNR model
    highpass_ramp = 1.0 - np.exp(-(s * highpass) ** 2)          # suppress low freq
    ssnr = (10.0 ** (3.0 * strength)) * highpass_ramp * np.exp(-falloff * (s / s.max()) )
    ssnr = np.maximum(ssnr, 1e-6)
    wiener = ctf / (ctf * ctf + 1.0 / ssnr)

    F = np.fft.fftn(vol)
    F *= wiener
    return np.real(np.fft.ifftn(F)).astype(np.float32)
