#!/usr/bin/env bash
# Usage: ./run_all.sh <STUDENT_ID> [--peak-bw-gbs <GB/s of your DRAM>]
# Runs M0-M3 in order and a local self-check. It does NOT touch git: commit and tag
# each milestone yourself (the tags are timestamped and must fall inside the 180-min window).
set -euo pipefail
cd "$(dirname "$0")"
ID="${1:?student id required}"; shift || true

python3 src/profile_env.py --student-id "$ID" >/dev/null && echo "M0 done  -> git add . && git commit -m 'M0: Environment and salting configuration' && git tag M0-ENV"
python3 src/stride_bench.py
python3 src/dense_kernel.py
echo "M1 done  -> git add . && git commit -m 'M1: Memory stride and loop interchange benchmarking' && git tag M1-LOCALITY"
python3 src/false_sharing.py
echo "M2 done  -> git add . && git commit -m 'M2: False sharing reproduction and Karp-Flatt analysis' && git tag M2-SHARING"
python3 src/scaling_study.py "$@"
echo "M3 done  -> git add . && git commit -m 'M3: Strong and weak scaling evaluation' && git tag M3-SCALING"
python3 verify_local.py || true
echo "Now run the course's verify_outputs.py, then the M4 commit/tag/push."
