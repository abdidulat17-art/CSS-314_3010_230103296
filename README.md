# CS482 Lab Practicum: Parallel Computing & Performance Optimization

## Run

```bash
./run_all.sh <YOUR_8_DIGIT_ID> [--peak-bw-gbs <DRAM peak of your machine, GB/s>]
```

Requires Python 3, NumPy, psutil and `gcc` (the stride and dense kernels are compiled at run time and called via `ctypes`; without `gcc` they fall back to NumPy / a tiny pure-Python block). Individual steps: `src/profile_env.py --student-id <ID>`, `src/stride_bench.py`, `src/dense_kernel.py`, `src/false_sharing.py`, `src/scaling_study.py`.

## Summary of bottleneck findings

_Fill in after running on your machine (one paragraph): stride-1 vs stride>=8 latency ratio; i-j-k vs i-k-j speedup; unpadded vs padded runtime and Karp-Flatt e(P) at P=8; fitted Amdahl serial fraction s and S_max; weak-scaling efficiency; arithmetic intensity and whether the stencil is memory-bandwidth bound._

## Method notes

- **M2 / Karp-Flatt:** each worker performs K increments, so total work is P·K. `false_sharing_karp_flatt.tsv` reports the work-normalised speedup `S = P·T1/T_P` and `e_workscaled` (ideal = 0), plus the literal formula from the handout as `e_spec_literal` (ideal = 1 in this setup).
- **M2 arrays** are raw (`lock=False`) and 64-byte aligned, so "8 counters on one line" is guaranteed.
- **M3 weak scaling:** `scaled_speedup_spec` is the handout formula; `scaled_speedup_throughput` is the conventional Gustafson scaled speedup (work/time ratio) and is what `gustafson_alpha` uses.
- **M3 roofline:** peak GFLOP/s is an estimate (cores x base frequency x SIMD FLOPs/cycle); bandwidth defaults to a measured single-process proxy unless `--peak-bw-gbs` is given.
- `verify_local.py` is a self-check only; `verify_outputs.py` and `verification_receipt.sig` come from the course.
