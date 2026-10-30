#!/usr/bin/env python3
"""Milestone 2: false sharing reproduction, remediation and Karp-Flatt diagnosis.

Three variants, each: P processes, worker p increments its own counter K times.
  unpadded   : counters at shared_arr[p]        -> all on ONE 64-byte cache line
  padded     : counters at shared_arr[p * 8]    -> one cache line per worker
  privatized : local scalar, one locked write at the end

Notes on method
* The shared arrays are raw ctypes arrays (lock=False). With the default lock=True
  every `arr[i] += 1` would go through a global lock and measure lock contention
  instead of cache-line ping-pong.
* The arrays are over-allocated and the base index is chosen so that element 0 sits
  exactly on a 64-byte boundary; otherwise "8 counters on one line" would not be
  guaranteed.
* A Barrier releases all workers together; the clock starts when the barrier opens,
  so process start-up cost is excluded. Median of REPS repetitions is reported.
* Correctness: every run must end with sum(counters) == P * K.

Karp-Flatt: here every worker does K increments, so total work is P*K and the
fair serial time is P*T1. Therefore the work-normalised speedup is S_P = P*T1/T_P
(ideal = P), giving e_workscaled(P) = (1/S_P - 1/P)/(1 - 1/P), ideal = 0.
The literal spec formula (S_P = T1/T_P, e = (P*T_P/T1 - 1)/(P-1)) is also written
(column e_spec_literal); for constant-time-per-worker ideal runs it equals 1.
"""
import argparse, ctypes, json, multiprocessing as mp, os, statistics, sys, time

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
RESULTS = os.path.join(ROOT, "results")
LINE = 64
INTS_PER_LINE = LINE // 8  # 8 x int64 per cache line
PS = [1, 2, 4, 8]
K_DEFAULT = 5_000_000
REPS_DEFAULT = 3


def aligned_array(n_ints):
    """Return (ctypes array, base index) with element[base] on a 64-byte boundary."""
    arr = mp.Array(ctypes.c_int64, n_ints + INTS_PER_LINE, lock=False)
    addr = ctypes.addressof(arr)
    base = ((-addr) % LINE) // 8
    return arr, base


def worker_unpadded(counter_array, base, worker_id, iterations, barrier):
    # Pathological: indices base..base+7 are the SAME 64-byte cache line.
    idx = base + worker_id
    barrier.wait()
    for _ in range(iterations):
        counter_array[idx] += 1


def worker_padded(counter_array, base, worker_id, iterations, barrier):
    # One counter per cache line: stride of 8 int64 = 64 bytes.
    idx = base + worker_id * INTS_PER_LINE
    barrier.wait()
    for _ in range(iterations):
        counter_array[idx] += 1


def worker_privatized(counter_array, base, worker_id, iterations, barrier, lock):
    # Local scalar in the private frame; a single synchronised write at the end.
    local = 0
    barrier.wait()
    for _ in range(iterations):
        local += 1
    with lock:
        counter_array[base + worker_id * INTS_PER_LINE] = local


def run_variant(variant, P, K, ctx):
    arr, base = aligned_array(P * INTS_PER_LINE if variant != "unpadded" else INTS_PER_LINE)
    barrier = ctx.Barrier(P + 1)
    lock = ctx.Lock()
    target = {"unpadded": worker_unpadded, "padded": worker_padded, "privatized": worker_privatized}[variant]
    procs = []
    for p in range(P):
        a = (arr, base, p, K, barrier) + ((lock,) if variant == "privatized" else ())
        procs.append(ctx.Process(target=target, args=a))
    for pr in procs:
        pr.start()
    barrier.wait()
    t0 = time.perf_counter()
    for pr in procs:
        pr.join()
    elapsed = time.perf_counter() - t0
    if variant == "unpadded":
        counts = [arr[base + p] for p in range(P)]
    else:
        counts = [arr[base + p * INTS_PER_LINE] for p in range(P)]
    return elapsed, sum(counts), counts


def karp_flatt(S, P):
    return (1.0 / S - 1.0 / P) / (1.0 - 1.0 / P)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-K", type=int, default=K_DEFAULT, help="increments per worker (spec: 5,000,000)")
    ap.add_argument("--reps", type=int, default=REPS_DEFAULT)
    ap.add_argument("--out", default=os.path.join(RESULTS, "false_sharing_karp_flatt.tsv"))
    args = ap.parse_args()
    K = args.K
    ctx = mp.get_context("fork") if "fork" in mp.get_all_start_methods() else mp.get_context()
    ncpu = os.cpu_count()
    print("K=%d reps=%d logical_cpus=%s" % (K, args.reps, ncpu))
    if ncpu and ncpu < max(PS):
        print("[note] P=8 exceeds the %d logical CPUs here; oversubscribed points are not true parallel runs." % ncpu)

    # median time per (variant, P); also remember every run
    T, rows, all_ok = {}, [], True
    for variant in ("unpadded", "padded", "privatized"):
        for P in PS:
            times = []
            for r in range(args.reps):
                t, total, counts = run_variant(variant, P, K, ctx)
                ok = (total == P * K)
                all_ok &= ok
                times.append(t)
                rows.append({"variant": variant, "P": P, "rep": r, "time_s": t,
                             "total_count": total, "expected": P * K, "exact": ok})
            T[(variant, P)] = statistics.median(times)
            print("%-11s P=%d  median %.3f s  (counts exact: %s)" % (variant, P, T[(variant, P)], all_ok))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write("# K=%d reps=%d; speedup_workscaled = P*T1/TP; e_workscaled=(1/S-1/P)/(1-1/P); "
                "e_spec_literal=(P*TP/T1-1)/(P-1)\n" % (K, args.reps))
        f.write("variant\tP\tmedian_time_s\tK_per_worker\ttotal_count\texpected_count\tcount_exact\t"
                "speedup_workscaled\tefficiency_pct\tslowdown_vs_P1\te_workscaled\te_spec_literal\n")
        for variant in ("unpadded", "padded", "privatized"):
            T1 = T[(variant, 1)]
            for P in PS:
                TP = T[(variant, P)]
                S = P * T1 / TP
                eff = 100.0 * S / P
                slow = TP / T1
                if P == 1:
                    e_ws = e_lit = float("nan")
                else:
                    e_ws = karp_flatt(S, P)
                    e_lit = (P * TP / T1 - 1.0) / (P - 1.0)
                exact = all(r["exact"] for r in rows if r["variant"] == variant and r["P"] == P)
                f.write("%s\t%d\t%.6f\t%d\t%d\t%d\t%s\t%.4f\t%.1f\t%.3f\t%s\t%s\n" % (
                    variant, P, TP, K, P * K, P * K, exact, S, eff, slow,
                    "NA" if P == 1 else "%.4f" % e_ws, "NA" if P == 1 else "%.4f" % e_lit))
    with open(os.path.join(os.path.dirname(args.out), "false_sharing_runs.json"), "w") as f:
        json.dump({"K": K, "reps": args.reps, "all_counts_exact": all_ok, "runs": rows}, f, indent=2)
    print("wrote", args.out, "| all counts exact:", all_ok)
    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
