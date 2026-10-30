#!/usr/bin/env python3
"""Local self-check of results/ (NOT the course's verify_outputs.py and it does NOT
create verification_receipt.sig - run the official script for that)."""
import csv, json, os, sys

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
fails = 0


def report(ok, msg, warn=False):
    global fails
    tag = "PASS" if ok else ("WARN" if warn else "FAIL")
    if not ok and not warn:
        fails += 1
    print("[%s] %s" % (tag, msg))


def tsv(name, skip_comment=True):
    with open(os.path.join(R, name)) as f:
        lines = [l for l in f if not l.startswith("#")]
    return list(csv.DictReader(lines, delimiter="\t"))


def js(name):
    with open(os.path.join(R, name)) as f:
        return json.load(f)


try:
    hp, sp = js("host_profile.json"), js("salted_params.json")
    ok = (sp["N"] % 64 == 0 and 1024 <= sp["N"] <= 1536 and sp["buffer_elements"] == 16_777_216
          and hp["cache"]["cache_line_bytes"] and hp["topology"]["physical_cores"])
    report(bool(ok), "M0: host profile + salted params (N=%s, seed=%s)" % (sp["N"], sp["seed"]))
except Exception as e:
    report(False, "M0: %s" % e)

try:
    rows = tsv("stride_results.tsv")
    lat = {int(r["stride"]): float(r["latency_per_element_ns"]) for r in rows}
    report(len(lat) == 8, "M1: stride table has 8 strides")
    report(max(lat[s] for s in (8, 16, 32, 64, 128)) > 1.5 * lat[1],
           "M1: cliff visible (stride-1 %.2f ns -> worst stride>=8 %.2f ns)" % (
               lat[1], max(lat[s] for s in (8, 16, 32, 64, 128))), warn=True)
    li = js("loop_interchange.json")
    report(li["max_abs_diff_ijk_vs_ikj"] <= 1e-12, "M1: i-j-k == i-k-j within 1e-12 (diff %.2e)" % li["max_abs_diff_ijk_vs_ikj"])
    report(li["speedup_naive_over_interchanged"] > 1.0, "M1: interchange speedup %.2fx" % li["speedup_naive_over_interchanged"], warn=True)
except Exception as e:
    report(False, "M1: %s" % e)

try:
    rows = tsv("false_sharing_karp_flatt.tsv")
    report(all(r["count_exact"] == "True" and r["total_count"] == r["expected_count"] for r in rows),
           "M2: every variant ended with exactly P*K counts")
    t = {(r["variant"], int(r["P"])): float(r["median_time_s"]) for r in rows}
    report(t[("unpadded", 8)] > t[("unpadded", 1)], "M2: unpadded runtime grows with P (negative speedup)", warn=True)
    e = {(r["variant"], int(r["P"])): r["e_workscaled"] for r in rows if r["e_workscaled"] != "NA"}
    report(float(e[("unpadded", 8)]) > float(e[("padded", 8)]), "M2: e_unpadded(8) > e_padded(8)", warn=True)
except Exception as e_:
    report(False, "M2: %s" % e_)

try:
    rows = tsv("scaling_metrics.tsv")
    report(all(r["checksum_match"] == "True" for r in rows), "M3: all checksums match serial reference")
    rf = js("roofline_summary.json")
    report(rf["regime"] in ("Memory-Bandwidth Bound", "Compute Bound"), "M3: roofline regime = %s" % rf["regime"])
except Exception as e_:
    report(False, "M3: %s" % e_)

print("\n%s" % ("All local checks passed." if not fails else "%d check(s) FAILED." % fails))
sys.exit(1 if fails else 0)
