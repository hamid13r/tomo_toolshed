"""
GPU (PyTorch) back end for the real-space novaCTF engine (novactf.py).

Same algorithm and the same numbers as the numba path, moved to the GPU:

  * per tilt: rfft2 of the padded tilt, the radial/weighting filter, every
    CTF-corrected defocus copy (batched irfft2), all on the device;
  * back-projection in Z-chunks: the coarse geometry is trilinearly
    upsampled with grid_sample (align_corners=True reproduces the numba
    kernel's cell lookup exactly), then each voxel gathers the 4 bilinear
    neighbours from its defocus copy;
  * the volume stays on the device until the end.

The coarse geometry (Warp's projection of the node grid, incl. local
motion) stays on the CPU -- it is small, and novactf.py prepares it for
tilt j+1 while the GPU works on tilt j.

Arithmetic is float32 on the device (the numba path uses float64 for the
CTF phase); agreement with the CPU volume is checked in tests/xml_reconstruct/validate.py.
"""
from __future__ import annotations
import numpy as np

try:
    import torch
    import torch.nn.functional as tF
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

# Fused back-projection kernel. The pure-torch version below materializes
# every intermediate (coordinates, indices, weights) in GPU memory and was
# slower than the numba CPU kernel (34.7 s vs ~15 s at 6.64 A/px); this is
# the CPU kernel ported one-to-one, one thread per voxel.
try:
    from numba import cuda as _cuda
    HAVE_NUMBA_CUDA = _cuda.is_available()
except Exception:                        # numba missing, or no CUDA driver/NVVM
    HAVE_NUMBA_CUDA = False

if HAVE_NUMBA_CUDA:
    import math as _math

    from numba import float32 as _f32

    @_cuda.jit(fastmath=True)
    def _bp_cuda(vol, planes, g, PX, PY, KF, kmin):
        # Everything explicitly float32: in numba `int / float` and bare
        # literals are float64, and consumer GPUs (RTX A5000) run fp64 at
        # 1/32 rate -- the first version of this kernel was 20x slower for it.
        ix = _cuda.blockIdx.x * _cuda.blockDim.x + _cuda.threadIdx.x
        iy = _cuda.blockIdx.y
        iz = _cuda.blockIdx.z
        nz, ny, nx = vol.shape
        if ix >= nx:
            return
        K, H, W = planes.shape
        cz, cy, cx = PX.shape
        one = _f32(1.0)
        fz = _f32(iz) / g
        z0 = min(int(fz), cz - 2)
        tz = fz - _f32(z0)
        fy = _f32(iy) / g
        y0 = min(int(fy), cy - 2)
        ty = fy - _f32(y0)
        fx = _f32(ix) / g
        x0 = min(int(fx), cx - 2)
        tx = fx - _f32(x0)
        px = _f32(0.0)
        py = _f32(0.0)
        kf = _f32(0.0)
        for dz in range(2):
            wz = tz if dz else one - tz
            for dy in range(2):
                wy = ty if dy else one - ty
                for dx in range(2):
                    w = wz * wy * (tx if dx else one - tx)
                    px += w * PX[z0 + dz, y0 + dy, x0 + dx]
                    py += w * PY[z0 + dz, y0 + dy, x0 + dx]
                    kf += w * KF[z0 + dz, y0 + dy, x0 + dx]
        if px < _f32(0.0) or py < _f32(0.0) or px > _f32(W - 1) or py > _f32(H - 1):
            return
        k = int(_math.floor(kf + _f32(0.5))) - kmin
        if k < 0:
            k = 0
        elif k > K - 1:
            k = K - 1
        ix0 = min(int(px), W - 2)
        iy0 = min(int(py), H - 2)
        fxp = px - _f32(ix0)
        fyp = py - _f32(iy0)
        v = ((one - fyp) * ((one - fxp) * planes[k, iy0, ix0] + fxp * planes[k, iy0, ix0 + 1])
             + fyp * ((one - fxp) * planes[k, iy0 + 1, ix0] + fxp * planes[k, iy0 + 1, ix0 + 1]))
        vol[iz, iy, ix] += v

def resolve_device(device: str):
    if not HAVE_TORCH:
        raise RuntimeError("--novactf_device cuda needs PyTorch (not importable in this env)")
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"--novactf_device {device}: CUDA is not available to PyTorch")
    return dev


class TorchCTF3D:
    """Device-side state shared by all tilts of one reconstruction."""

    def __init__(self, device, vol_shape, shape_pad, shape_img, sx, sy, zchunk=16):
        self.dev = device
        self.Py, self.Px = shape_pad
        self.H, self.W = shape_img
        self.vol = torch.zeros(vol_shape, dtype=torch.float32, device=device)
        self.zchunk = int(zchunk)
        s2 = torch.as_tensor(sx.astype(np.float64) ** 2 + sy.astype(np.float64) ** 2,
                             device=device)
        phi = torch.as_tensor(np.arctan2(sy, sx), device=device)
        self._s2 = s2                                            # float64
        self._cos2phi, self._sin2phi = torch.cos(2 * phi), torch.sin(2 * phi)

    # -- CTF phase: argument(defocus_um) = const + slope * defocus_um ------------
    def phase_terms(self, p):
        """Same split as novactf.ctf_phase_terms, evaluated on the device."""
        K1, K2, K3, _ = p.ks()
        s2 = self._s2
        const = K2 * s2 * s2 - p.phase_shift * np.pi
        if p.defocus_delta != 0.0:
            a = 2.0 * np.deg2rad(p.defocus_angle)
            cos2 = self._cos2phi * np.cos(a) + self._sin2phi * np.sin(a)   # cos(2(phi-angle))
            const = const + K1 * s2 * 0.5 * (-(p.defocus_delta * 1e4)) * cos2
        slope = -1e4 * K1 * s2
        return const.float(), slope.float(), float(p.amplitude), float(K3)

    def defocus_planes(self, img, filt, defs, terms, correction, batch=16):
        """All CTF-corrected copies of one tilt as a (K, H, W) device tensor."""
        pad = torch.zeros((self.Py, self.Px), dtype=torch.float32, device=self.dev)
        pad[:self.H, :self.W] = torch.as_tensor(img, device=self.dev)
        F = torch.fft.rfft2(pad) * torch.as_tensor(filt, device=self.dev)
        if correction == "none":
            return torch.fft.irfft2(F, s=(self.Py, self.Px))[None, :self.H, :self.W].contiguous()
        const, slope, amp, K3 = terms
        d = torch.as_tensor(np.asarray(defs, np.float32), device=self.dev)
        planes = torch.empty((len(defs), self.H, self.W), dtype=torch.float32, device=self.dev)
        for b0 in range(0, len(defs), batch):
            arg = const[None] + slope[None] * d[b0:b0 + batch, None, None]
            c = amp * torch.cos(arg) - K3 * torch.sin(arg)
            if correction == "phaseflip":
                c = torch.sign(c)
            planes[b0:b0 + len(arg)] = torch.fft.irfft2(F[None] * c, s=(self.Py, self.Px))[
                :, :self.H, :self.W]
        return planes

    def backproject(self, planes, g, PX, PY, KF, kmin):
        """Accumulate one tilt into self.vol (same arithmetic as _bp_numba)."""
        if HAVE_NUMBA_CUDA and self.dev.type == "cuda":
            nz, ny, nx = self.vol.shape
            c = [torch.as_tensor(a.astype(np.float32), device=self.dev) for a in (PX, PY, KF)]
            tpb = 128
            with _cuda.gpus[self.dev.index or 0]:
                # numba and torch both issue to the legacy default stream here,
                # so the kernel is ordered after the planes torch just wrote
                _bp_cuda[((nx + tpb - 1) // tpb, ny, nz), tpb](
                    _cuda.as_cuda_array(self.vol), _cuda.as_cuda_array(planes),
                    np.float32(g), *[_cuda.as_cuda_array(a) for a in c], np.int32(kmin))
            return
        nz, ny, nx = self.vol.shape
        K, H, W = planes.shape
        coarse = torch.as_tensor(np.stack([PX, PY, KF]).astype(np.float32),
                                 device=self.dev)[None]            # (1, 3, cz, cy, cx)
        cz, cy, cx = PX.shape
        # normalized grid_sample coordinates of every voxel (align_corners=True:
        # -1 / +1 are the first / last coarse node, exactly the kernel's lookup)
        gx = torch.arange(nx, device=self.dev, dtype=torch.float32) / g / (cx - 1) * 2 - 1
        gy = torch.arange(ny, device=self.dev, dtype=torch.float32) / g / (cy - 1) * 2 - 1
        flat = planes.reshape(-1)
        HW = H * W
        for z0 in range(0, nz, self.zchunk):
            z1 = min(z0 + self.zchunk, nz)
            gz = torch.arange(z0, z1, device=self.dev, dtype=torch.float32) / g / (cz - 1) * 2 - 1
            Z, Y, X = torch.meshgrid(gz, gy, gx, indexing="ij")
            grid = torch.stack([X, Y, Z], dim=-1)[None]            # (1, zc, ny, nx, 3)
            s = tF.grid_sample(coarse, grid, mode="bilinear", align_corners=True)[0]
            px, py, kf = s[0], s[1], s[2]
            inside = (px >= 0) & (py >= 0) & (px <= W - 1) & (py <= H - 1)
            k = (torch.floor(kf + 0.5).long() - kmin).clamp_(0, K - 1)
            ix0 = torch.clamp(px.long(), 0, W - 2)
            iy0 = torch.clamp(py.long(), 0, H - 2)
            fx = px - ix0
            fy = py - iy0
            idx = k * HW + iy0 * W + ix0
            v = ((1 - fy) * ((1 - fx) * flat[idx] + fx * flat[idx + 1])
                 + fy * ((1 - fx) * flat[idx + W] + fx * flat[idx + W + 1]))
            self.vol[z0:z1] += torch.where(inside, v, torch.zeros_like(v))

    def result(self):
        return self.vol.cpu().numpy()
