"""
One thread count for every parallel part of xml_reconstruct.

Everything that runs in parallel reads `get_threads()`: scipy.fft `workers`,
the thread pools (tilt loading/preprocessing, novactf geometry), numba's
prange kernels, and the OMP_NUM_THREADS handed to IMOD's `tilt`. Set it once
with `set_threads(n)` (the CLI's --threads).

The default is the CPUs this process may actually run on
(os.sched_getaffinity), not os.cpu_count(): under SLURM, taskset or a cgroup
CPU limit the two differ, and using every core of the node oversubscribes
the allocation.
"""
from __future__ import annotations
import os

_threads: int | None = None


def available_cpus() -> int:
    """CPUs available to this process (respects affinity / SLURM binding)."""
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):          # not Linux
        return max(1, os.cpu_count() or 1)


def get_threads() -> int:
    return _threads if _threads is not None else available_cpus()


def set_threads(n: int | None) -> int:
    """Set the thread count for everything (None or <= 0 = available CPUs).
    Returns the count in effect. numba is capped at the pool size it started
    with (NUMBA_NUM_THREADS, all cores by default), so asking for more than
    that only raises the non-numba parts."""
    global _threads
    _threads = int(n) if n is not None and int(n) > 0 else None
    n_eff = get_threads()
    try:
        import numba
        numba.set_num_threads(max(1, min(n_eff, numba.config.NUMBA_NUM_THREADS)))
    except ImportError:
        pass
    return n_eff


def apply_numba_threads():
    """numba.set_num_threads is per calling thread; call this at the start of
    any thread (incl. the main one) that launches numba parallel kernels."""
    try:
        import numba
        numba.set_num_threads(max(1, min(get_threads(), numba.config.NUMBA_NUM_THREADS)))
    except ImportError:
        pass
