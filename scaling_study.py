#!/usr/bin/env python3
"""Milestone 3: strong/weak scaling of a 5-point Jacobi stencil + Roofline placement.

Kernel   : out[i,j] = 0.25*(in[i-1,j]+in[i+1,j]+in[i,j-1]+in[i,j+1]), fixed boundary,
           ITERS sweeps, ping-pong between two shared-memory grids.
Parallel : row-block decomposition over P worker processes, one barrier per sweep.
           Every element is computed with exactly the same arithmetic as in the
           serial reference, so the final grid (and its checksum) must match bitwise.
Timing   : barrier-released start -> checksum computed by the parent (the serial
           reduction is inside the timed region, so it contributes to Amdahl's s).
           Median of REPS repetitions.

Outputs  : results/scaling_metrics.tsv, results/scaling_fit.json,
           results/roofline_summary.json
"""
import argparse, hashlib, json, math, multiprocessing as mp, os, statistics, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from salt import load_params

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
RESULTS = os.path.join(ROOT, "results")

FLOPS_PER_POINT = 4          # 3 adds + 1 multiply
BYTES_PER_POINT_IDEAL = 16   # 8 B read (neighbours reused from cache) + 8 B write
BYTES_PER_POINT_WA = 24      # + 8 B write-allocate read of the destination line


# ----------------------------------------------------------------- kernel
def sweep(src, dst, r0, r1):
    """Update rows r0..r1-1 (interior columns) of dst from src."""
    out = dst[r0:r1, 1:-1]
    np.add(src[r0 - 1:r1 - 1, 1:-1], src[r0 + 1:r1 + 1, 1:-1], out=out)
    out += src[r0:r1, :-2]
    out += src[r0:r1, 2:]
    out *= 0.25


def make_grid(N, seed):
    rng = np.random.default_rng(seed)
    return rng.random((N, N))


def checksum(grid):
    g = np.round(grid, 6)
    return hashlib.sha256(np.ascontiguousarray(g).tobytes()).hexdigest()[:16], float(grid.sum())


def serial_reference(N, seed, iters):
    a = make_grid(N, seed)
    b = a.copy()
    for it in range(iters):
        src, dst = (a, b) if it % 2 == 0 else (b, a)
        sweep(src, dst, 1, N - 1)
    final = a if iters % 2 == 0 else b
    return checksum(final), final.copy()


def worker(raw_a, raw_b, N, r0, r1, iters, start_barrier, sweep_barrier):
    a = np.frombuffer(raw_a, dtype=np.float64).reshape(N, N)
    b = np.frombuffer(raw_b, dtype=np.float64).reshape(N, N)
    start_barrier.wait()
    for it in range(iters):
        src, dst = (a, b) if it % 2 == 0 else (b, a)
        sweep(src, dst, r0, r1)
        sweep_barrier.wait()


def row_blocks(N, P):
    rows = N - 2
    base, extra = divmod(rows, P)
    blocks, r = [], 1
    for p in range(P):
        n = base + (1 if p < extra else 0)
        blocks.append((r, r + n))
        r += n
    return blocks


def run_parallel(N, P, seed, iters, ctx):
    raw_a = ctx.RawArray("d", N * N)
    raw_b = ctx.RawArray("d", N * N)
    a = np.frombuffer(raw_a, dtype=np.float64).reshape(N, N)
    b = np.frombuffer(raw_b, dtype=np.float64).reshape(N, N)
    init = make_grid(N, seed)
    a[:] = init
    b[:] = init
    start_barrier, sweep_barrier = ctx.Barrier(P + 1), ctx.Barrier(P)
    procs = [ctx.Process(target=worker, args=(raw_a, raw_b, N, r0, r1, iters, start_barrier, sweep_barrier))
             for r0, r1 in row_blocks(N, P)]
    for pr in procs:
        pr.start()
    start_barrier.wait()
    t0 = time.perf_counter()
    for pr in procs:
        pr.join()
    final = a if iters % 2 == 0 else b
    cs = checksum(final)          # serial reduction, inside the timed region
    elapsed = time.perf_counter() - t0
    return elapsed, cs


def measure(N, P, seed, iters, reps, ctx, ref):
    times, ok = [], True
    for _ in range(reps):
        t, cs = run_parallel(N, P, seed, iters, ctx)
        times.append(t)
        ok &= (cs == ref)
    return statistics.median(times), ok


# ----------------------------------------------------------------- helpers
def physical_cores():
    try:
        with open(os.path.join(RESULTS, "host_profile.json")) as f:
            return int(json.load(f)["topology"]["physical_cores"])
    except Exception:
        try:
            import psutil
            return psutil.cpu_count(logical=False) or os.cpu_count()
        except Exception:
            return os.cpu_count()


def p_list(maxp):
    ps, p = [], 1
    while p <= maxp:
        ps.append(p)
        p *= 2
    if ps[-1] != maxp:
        ps.append(maxp)
    return ps


def amdahl_point(S, P):
    return float("nan") if P == 1 else (1.0 / S - 1.0 / P) / (1.0 - 1.0 / P)


def amdahl_fit(Ps, Ss):
    """Least-squares fit of 1/S = 1/P + s(1-1/P) (linear in s)."""
    num = den = 0.0
    for P, S in zip(Ps, Ss):
        if P == 1:
            continue
        x = 1.0 - 1.0 / P
        num += (1.0 / S - 1.0 / P) * x
        den += x * x
    return num / den if den else float("nan")


def measure_bandwidth(n=1 << 23, reps=5):
    """Single-process memory-bandwidth proxy (NumPy triad-like, 40 B/element moved)."""
    b = np.random.default_rng(0).random(n)
    c = np.random.default_rng(1).random(n)
    a = np.empty(n)
    tmp = np.empty(n)
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        np.multiply(c, 3.0, out=tmp)
        np.add(b, tmp, out=a)
        best = min(best, time.perf_counter() - t0)
    return 40.0 * n / best / 1e9


def peak_flops_per_cycle():
    try:
        with open("/proc/cpuinfo") as f:
            txt = f.read()
        if "avx512f" in txt:
            return 32, "AVX-512 (2 FMA units assumed)"
        if "avx2" in txt and " fma" in txt:
            return 16, "AVX2+FMA (2 FMA units assumed)"
        if "avx" in txt:
            return 8, "AVX"
        return 4, "SSE2"
    except OSError:
        return 8, "unknown (generic assumption)"


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=40, help="stencil sweeps per run")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--N", type=int, default=None, help="override salted N (testing only)")
    ap.add_argument("--max-p", type=int, default=None, help="override max worker count")
    ap.add_argument("--peak-bw-gbs", type=float, default=None,
                    help="theoretical DRAM peak bandwidth of YOUR machine (GB/s), e.g. 51.2 for dual-channel DDR4-3200")
    ap.add_argument("--flops-per-cycle", type=float, default=None, help="override DP FLOPs/cycle/core")
    args = ap.parse_args()

    params = load_params()
    N = args.N or params["N"]
    seed = params["seed"]
    maxp = args.max_p or physical_cores()
    Ps = p_list(maxp)
    ctx = mp.get_context("fork") if "fork" in mp.get_all_start_methods() else mp.get_context()
    print("N=%d iters=%d reps=%d P in %s" % (N, args.iters, args.reps, Ps))

    rows, all_ok = [], True

    # ---- Task 3.1 strong scaling (fixed N)
    ref, _ = serial_reference(N, seed, args.iters)
    T = {}
    for P in Ps:
        T[P], ok = measure(N, P, seed, args.iters, args.reps, ctx, ref)
        all_ok &= ok
        print("strong P=%d  T=%.3fs  checksum match: %s" % (P, T[P], ok))
    Ss = [T[1] / T[P] for P in Ps]
    for P, S in zip(Ps, Ss):
        rows.append(dict(study="strong", P=P, N=N, time_s=T[P], speedup=S, efficiency_pct=100 * S / P,
                         amdahl_s_point=amdahl_point(S, P), weak_efficiency=float("nan"),
                         scaled_speedup_spec=float("nan"), scaled_speedup_throughput=float("nan"),
                         gustafson_alpha=float("nan"), checksum=ref[0], checksum_match=ok))
    s_fit = amdahl_fit(Ps, Ss)
    s_pos = max(s_fit, 0.0)
    smax = (1.0 / s_pos) if s_pos > 0 else float("inf")

    # ---- Task 3.2 weak scaling (N_P = ceil(N*sqrt(P)), work ~ P)
    Tw, work = {}, {}
    for P in Ps:
        NP = math.ceil(N * math.sqrt(P))
        refw, _ = serial_reference(NP, seed, args.iters) if P > 1 else (ref, None)
        Tw[P], ok = measure(NP, P, seed, args.iters, args.reps, ctx, refw)
        all_ok &= ok
        work[P] = (NP - 2) ** 2 * args.iters * FLOPS_PER_POINT
        print("weak   P=%d  N_P=%d  T=%.3fs  checksum match: %s" % (P, NP, Tw[P], ok))
        S_spec = P * work[1] * Tw[1] / (work[P] * Tw[P])
        S_thr = (work[P] / Tw[P]) / (work[1] / Tw[1])
        alpha = float("nan") if P == 1 else (P - S_thr) / (P - 1)
        rows.append(dict(study="weak", P=P, N=NP, time_s=Tw[P], speedup=float("nan"), efficiency_pct=float("nan"),
                         amdahl_s_point=float("nan"), weak_efficiency=Tw[1] / Tw[P],
                         scaled_speedup_spec=S_spec, scaled_speedup_throughput=S_thr,
                         gustafson_alpha=alpha, checksum=refw[0], checksum_match=ok))

    os.makedirs(RESULTS, exist_ok=True)
    cols = ["study", "P", "N", "time_s", "speedup", "efficiency_pct", "amdahl_s_point", "weak_efficiency",
            "scaled_speedup_spec", "scaled_speedup_throughput", "gustafson_alpha", "checksum", "checksum_match"]
    with open(os.path.join(RESULTS, "scaling_metrics.tsv"), "w") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(("%.5f" % r[c]) if isinstance(r[c], float) and not math.isnan(r[c]) else
                              ("NA" if isinstance(r[c], float) else str(r[c])) for c in cols) + "\n")

    fit = {"N": N, "iters": args.iters, "P_values": Ps,
           "amdahl_serial_fraction_s_lsq": s_fit, "amdahl_serial_fraction_used": s_pos,
           "S_max_asymptotic": smax if math.isfinite(smax) else "infinite (fitted s <= 0)",
           "all_checksums_match_reference": bool(all_ok),
           "note": "s is an effective serial fraction: it also absorbs barrier cost, load imbalance, "
                   "memory-bandwidth saturation and CPU oversubscription."}
    with open(os.path.join(RESULTS, "scaling_fit.json"), "w") as f:
        json.dump(fit, f, indent=2)

    # ---- Task 3.3 arithmetic intensity & roofline
    try:
        with open(os.path.join(RESULTS, "host_profile.json")) as f:
            host = json.load(f)
    except OSError:
        host = {}
    cache = host.get("cache", {})
    freq = (host.get("cpu", {}) or {}).get("base_freq_mhz") or 2500.0
    fpc, fpc_note = peak_flops_per_cycle()
    if args.flops_per_cycle:
        fpc, fpc_note = args.flops_per_cycle, "user override"
    peak_gflops = maxp * freq / 1000.0 * fpc
    bw_meas = measure_bandwidth()
    bw_peak = args.peak_bw_gbs or bw_meas
    bw_src = "user-supplied theoretical" if args.peak_bw_gbs else "measured single-process NumPy proxy (lower bound of true peak)"
    ai_ideal = FLOPS_PER_POINT / BYTES_PER_POINT_IDEAL
    ai_wa = FLOPS_PER_POINT / BYTES_PER_POINT_WA
    ridge = peak_gflops / bw_peak
    ws = 2 * N * N * 8
    l3 = cache.get("l3_bytes")
    achieved = work_gflops = (N - 2) ** 2 * args.iters * FLOPS_PER_POINT / T[1] / 1e9
    attainable = min(peak_gflops, ai_ideal * bw_peak)
    regime = "Memory-Bandwidth Bound" if ai_ideal < ridge else "Compute Bound"
    roof = {
        "kernel": "5-point Jacobi stencil, double precision",
        "flops_per_point": FLOPS_PER_POINT,
        "dram_bytes_per_point_ideal": BYTES_PER_POINT_IDEAL,
        "dram_bytes_per_point_with_write_allocate": BYTES_PER_POINT_WA,
        "arithmetic_intensity_flop_per_byte": ai_ideal,
        "arithmetic_intensity_with_write_allocate": ai_wa,
        "working_set_bytes_two_grids": ws,
        "cache_bytes": {"l1d": cache.get("l1d_bytes"), "l2": cache.get("l2_bytes"), "l3": l3},
        "working_set_exceeds_L3": (ws > l3) if l3 else None,
        "peak_gflops_assumed": peak_gflops,
        "peak_gflops_basis": "%d cores x %.0f MHz x %g FLOP/cycle (%s)" % (maxp, freq, fpc, fpc_note),
        "memory_bandwidth_gbs_used": bw_peak,
        "memory_bandwidth_source": bw_src,
        "memory_bandwidth_measured_gbs": bw_meas,
        "ridge_point_flop_per_byte": ridge,
        "attainable_gflops_roofline": attainable,
        "achieved_gflops_single_process": achieved,
        "regime": regime,
        "conclusion": "AI = %.3f FLOP/B vs ridge point %.2f FLOP/B -> %s. Even at 100%% of the bandwidth roof, "
                      "the kernel can reach only %.2f GFLOP/s of the %.1f GFLOP/s compute peak." % (
                          ai_ideal, ridge, regime, attainable, peak_gflops),
        "caveat": "Peak GFLOP/s assumes full SIMD+FMA throughput; the NumPy implementation makes several "
                  "passes over memory, so its real traffic is higher than the ideal 16 B/point. "
                  "Re-run with --peak-bw-gbs <your DRAM spec> for a theoretical roof.",
    }
    with open(os.path.join(RESULTS, "roofline_summary.json"), "w") as f:
        json.dump(roof, f, indent=2)

    print("Amdahl s (LSQ) = %.4f  S_max = %s" % (s_fit, smax))
    print("Roofline:", roof["conclusion"])
    print("all checksums match reference:", all_ok)
    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
