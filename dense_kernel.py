#!/usr/bin/env python3
"""Milestone 1.2: loop interchange i-j-k vs i-k-j on flat row-major matrices.

Works on the leading M x M sub-block, M = N/2 (N is the salted dimension), with
A, B stored as flat 1D float64 buffers of leading dimension N, so B[k, j] truly
strides by N elements in the naive order. Both kernels are compiled via ctypes
(gcc -O1, no fast-math / no auto-interchange), as allowed by the spec.
If no compiler is available a pure-Python fallback runs on a tiny block.
Also stores a SHA-256 digest of C (rounded to 6 d.p.) and the max abs diff
between the two algorithms for verify_outputs.py.
"""
import ctypes, hashlib, json, os, statistics, subprocess, sys, tempfile, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from salt import load_params

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
RESULTS = os.path.join(ROOT, "results")

C_SRC = r"""
#include <stddef.h>
/* i-j-k: B[k*N+j] strides by N on every inner step */
void ijk(const double *A, const double *B, double *C, size_t M, size_t N) {
    for (size_t i = 0; i < M; i++)
        for (size_t j = 0; j < M; j++) {
            double acc = 0.0;
            for (size_t k = 0; k < M; k++) acc += A[i*N+k] * B[k*N+j];
            C[i*N+j] += acc;
        }
}
/* i-k-j: A[i,k] hoisted into a register, C[i,:] and B[k,:] walked stride-1 */
void ikj(const double *A, const double *B, double *C, size_t M, size_t N) {
    for (size_t i = 0; i < M; i++)
        for (size_t k = 0; k < M; k++) {
            double a = A[i*N+k];
            const double *b = B + k*N;
            double *c = C + i*N;
            for (size_t j = 0; j < M; j++) c[j] += a * b[j];
        }
}
"""


def build_kernels():
    d = tempfile.mkdtemp(prefix="dense_")
    src, so = os.path.join(d, "k.c"), os.path.join(d, "k.so")
    with open(src, "w") as f:
        f.write(C_SRC)
    subprocess.check_call(["gcc", "-O1", "-fno-tree-loop-distribute-patterns", "-shared", "-fPIC", src, "-o", so])
    lib = ctypes.CDLL(so)
    for fn in (lib.ijk, lib.ikj):
        fn.restype = None
        fn.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_size_t] * 2
    return lib


def py_ijk(A, B, C, M, N):
    for i in range(M):
        for j in range(M):
            acc = 0.0
            for k in range(M):
                acc += A[i*N+k] * B[k*N+j]
            C[i*N+j] += acc


def py_ikj(A, B, C, M, N):
    for i in range(M):
        for k in range(M):
            a = A[i*N+k]
            for j in range(M):
                C[i*N+j] += a * B[k*N+j]


def digest(C, M, N):
    block = np.round(C.reshape(N, N)[:M, :M], 6)
    return hashlib.sha256(np.ascontiguousarray(block).tobytes()).hexdigest()


def main(N=None, M=None, runs=3):
    params = load_params()
    N = N or params["N"]
    M = M or N // 2
    rng = np.random.default_rng(params["seed"])
    A = rng.random(N * N)  # flat 1D buffers, row-major
    B = rng.random(N * N)

    try:
        lib = build_kernels()
        backend = "C/ctypes"

        def run(fn):
            C = np.zeros(N * N)
            t0 = time.perf_counter()
            fn(A.ctypes.data, B.ctypes.data, C.ctypes.data, M, N)
            return time.perf_counter() - t0, C
        naive, inter = lib.ijk, lib.ikj
    except Exception as e:
        print("[warn] gcc unavailable (%s); pure-Python fallback on tiny block" % e, file=sys.stderr)
        backend = "pure-python"
        M = min(M, 96)
        Al, Bl = A.tolist(), B.tolist()

        def run(fn):
            C = [0.0] * (N * N)
            t0 = time.perf_counter()
            fn(Al, Bl, C, M, N)
            return time.perf_counter() - t0, np.array(C)
        naive, inter = py_ijk, py_ikj

    res = {}
    for name, fn in (("naive_ijk", naive), ("interchanged_ikj", inter)):
        times, C = [], None
        for _ in range(runs):
            t, C = run(fn)
            times.append(t)
        res[name] = {"times_s": times, "median_s": statistics.median(times), "C": C}
        print("%-17s median %.4f s  runs %s" % (name, res[name]["median_s"], ["%.4f" % t for t in times]))

    Cn, Ci = res["naive_ijk"]["C"], res["interchanged_ikj"]["C"]
    max_diff = float(np.max(np.abs(Cn - Ci)))
    # reference check with NumPy BLAS on the same sub-block
    A2, B2 = A.reshape(N, N)[:M, :M], B.reshape(N, N)[:M, :M]
    ref_diff = float(np.max(np.abs(Ci.reshape(N, N)[:M, :M] - A2 @ B2)))
    speedup = res["naive_ijk"]["median_s"] / res["interchanged_ikj"]["median_s"]
    out = {
        "N": N, "M": M, "seed": params["seed"], "backend": backend, "runs": runs,
        "naive_ijk": {k: res["naive_ijk"][k] for k in ("times_s", "median_s")},
        "interchanged_ikj": {k: res["interchanged_ikj"][k] for k in ("times_s", "median_s")},
        "speedup_naive_over_interchanged": speedup,
        "max_abs_diff_ijk_vs_ikj": max_diff,
        "max_abs_diff_vs_numpy_matmul": ref_diff,
        "result_sha256_6dp": digest(Ci, M, N),
        "epsilon": 1e-12,
        "numerically_equal_within_eps": bool(max_diff <= 1e-12),
    }
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "loop_interchange.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("speedup S = T_naive/T_interchanged = %.2fx ; max|ijk-ikj| = %.3e" % (speedup, max_diff))


if __name__ == "__main__":
    a = sys.argv[1:]
    main(N=int(a[0]) if a else None, M=int(a[1]) if len(a) > 1 else None)
