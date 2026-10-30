#!/usr/bin/env python3
"""Milestone 1.1: stride latency cliff benchmark.

Reads data[::s] and accumulates a reduction. Pure-Python loops would drown the
memory effect in interpreter overhead, so the traversal runs in a tiny C kernel
(compiled once with gcc, called through ctypes). Fallback if no compiler exists:
NumPy strided .sum() (a view with step s, which still touches one element per stride).
Latency per element = median elapsed ns / elements touched.
"""
import ctypes, os, statistics, subprocess, sys, tempfile, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from salt import load_params

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
RESULTS = os.path.join(ROOT, "results")
STRIDES = [1, 2, 4, 8, 16, 32, 64, 128]
WARMUP, RUNS = 5, 5

C_SRC = r"""
#include <stddef.h>
double stride_sum(const double *restrict d, size_t n, size_t s) {
    double total = 0.0;
    for (size_t i = 0; i < n; i += s) total += d[i];
    return total;
}
"""


def build_kernel():
    d = tempfile.mkdtemp(prefix="stride_")
    src, so = os.path.join(d, "k.c"), os.path.join(d, "k.so")
    with open(src, "w") as f:
        f.write(C_SRC)
    # -O1 and no -ffast-math: keeps a strictly sequential dependent add chain,
    # so the loop cannot be vectorised away and each element is genuinely loaded.
    subprocess.check_call(["gcc", "-O1", "-shared", "-fPIC", src, "-o", so])
    lib = ctypes.CDLL(so)
    lib.stride_sum.restype = ctypes.c_double
    lib.stride_sum.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t]
    return lib


def main(n=None):
    params = load_params()
    n = n or params["buffer_elements"]
    data = np.ones(n, dtype=np.float64)
    try:
        lib = build_kernel()
        kernel = lambda s: lib.stride_sum(data.ctypes.data, n, s)
        backend = "C/ctypes"
    except Exception as e:  # no compiler
        print("[warn] gcc unavailable (%s); using NumPy strided sum" % e, file=sys.stderr)
        kernel = lambda s: float(data[::s].sum())
        backend = "numpy-view"

    rows = []
    for s in STRIDES:
        touched = (n + s - 1) // s
        for _ in range(WARMUP):
            kernel(s)
        times = []
        for _ in range(RUNS):
            t0 = time.perf_counter_ns()
            total = kernel(s)
            times.append(time.perf_counter_ns() - t0)
        assert total == touched, "reduction mismatch for stride %d: %r != %r" % (s, total, touched)
        med = statistics.median(times)
        rows.append((s, touched, med, med / touched))
        print("stride %4d  touched %9d  median %12.0f ns  latency/elem %7.3f ns" % (s, touched, med, med / touched))

    os.makedirs(RESULTS, exist_ok=True)
    base = rows[0][3]
    with open(os.path.join(RESULTS, "stride_results.tsv"), "w") as f:
        f.write("stride\telements_touched\tmedian_time_ns\tlatency_per_element_ns\trelative_to_stride1\tbackend\n")
        for s, t, med, lat in rows:
            f.write("%d\t%d\t%.0f\t%.4f\t%.3f\t%s\n" % (s, t, med, lat, lat / base, backend))
    print("wrote results/stride_results.tsv (backend: %s)" % backend)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
