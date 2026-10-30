#!/usr/bin/env python3
"""Milestone 0: host topology profiling + student-ID salting.

Usage: python3 src/profile_env.py --student-id <8-digit-id>
"""
import argparse, glob, json, os, platform, re, subprocess, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from salt import get_student_params

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
RESULTS = os.path.join(ROOT, "results")


def _read(path):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return None


def _parse_size(s):
    """'32K' / '1024K' / '8M' / '32768' -> bytes."""
    if not s:
        return None
    m = re.match(r"^\s*(\d+)\s*([KMG]?)", s, re.I)
    if not m:
        return None
    mult = {"": 1, "K": 1024, "M": 1024 ** 2, "G": 1024 ** 3}[m.group(2).upper()]
    return int(m.group(1)) * mult


def _sysctl(name):
    try:
        return subprocess.check_output(["sysctl", "-n", name], stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def cpu_info():
    info = {"model": None, "microarchitecture": platform.machine(), "base_freq_mhz": None}
    system = platform.system()
    if system == "Linux":
        txt = _read("/proc/cpuinfo") or ""
        m = re.search(r"model name\s*:\s*(.+)", txt)
        if m:
            info["model"] = m.group(1).strip()
        else:  # ARM etc.
            m = re.search(r"(Hardware|Model|CPU part)\s*:\s*(.+)", txt)
            info["model"] = m.group(2).strip() if m else None
        fam = re.search(r"cpu family\s*:\s*(\d+)", txt)
        mod = re.search(r"^model\s*:\s*(\d+)", txt, re.M)
        step = re.search(r"stepping\s*:\s*(\d+)", txt)
        if fam and mod:
            info["microarchitecture"] = "%s (family %s, model %s, stepping %s)" % (
                platform.machine(), fam.group(1), mod.group(1), step.group(1) if step else "?")
        f = _read("/sys/devices/system/cpu/cpu0/cpufreq/base_frequency")
        if f:
            info["base_freq_mhz"] = int(f) / 1000.0
        else:
            m = re.search(r"cpu MHz\s*:\s*([\d.]+)", txt)
            if m:
                info["base_freq_mhz"] = float(m.group(1))
    elif system == "Darwin":
        info["model"] = _sysctl("machdep.cpu.brand_string")
        hz = _sysctl("hw.cpufrequency")
        if hz and hz.isdigit():
            info["base_freq_mhz"] = int(hz) / 1e6
    else:
        info["model"] = platform.processor() or None
    try:
        import psutil
        if info["base_freq_mhz"] is None:
            fq = psutil.cpu_freq()
            if fq:
                info["base_freq_mhz"] = fq.max or fq.current
    except Exception:
        pass
    if not info["model"]:
        info["model"] = platform.processor() or "unknown"
    return info


def core_counts():
    logical = os.cpu_count()
    physical = None
    try:
        import psutil
        physical = psutil.cpu_count(logical=False)
        logical = psutil.cpu_count(logical=True) or logical
    except Exception:
        pass
    if physical is None and platform.system() == "Linux":
        pairs = set()
        for d in glob.glob("/sys/devices/system/cpu/cpu[0-9]*/topology"):
            pkg = _read(os.path.join(d, "physical_package_id"))
            core = _read(os.path.join(d, "core_id"))
            if pkg is not None and core is not None:
                pairs.add((pkg, core))
        physical = len(pairs) or None
    if physical is None and platform.system() == "Darwin":
        v = _sysctl("hw.physicalcpu")
        physical = int(v) if v and v.isdigit() else None
    physical = physical or logical
    return {"physical_cores": physical, "logical_threads": logical,
            "smt_enabled": bool(physical and logical and logical > physical),
            "threads_per_core": (logical // physical) if physical else None}


def cache_info():
    out = {"l1d_bytes": None, "l1i_bytes": None, "l2_bytes": None, "l3_bytes": None,
           "cache_line_bytes": None, "source": None}
    system = platform.system()
    if system == "Linux":
        base = "/sys/devices/system/cpu/cpu0/cache"
        for idx in sorted(glob.glob(base + "/index*")):
            lvl = _read(idx + "/level")
            typ = _read(idx + "/type")
            size = _parse_size(_read(idx + "/size"))
            line = _read(idx + "/coherency_line_size")
            if line and out["cache_line_bytes"] is None:
                out["cache_line_bytes"] = int(line)
            if lvl == "1" and typ == "Data":
                out["l1d_bytes"] = size
            elif lvl == "1" and typ == "Instruction":
                out["l1i_bytes"] = size
            elif lvl == "2":
                out["l2_bytes"] = size
            elif lvl == "3":
                out["l3_bytes"] = size
            out["source"] = "sysfs"
        if out["l1d_bytes"] is None:  # fallback: lscpu
            try:
                txt = subprocess.check_output(["lscpu", "-B"], text=True, stderr=subprocess.DEVNULL)
                for key, name in (("L1d", "l1d_bytes"), ("L2", "l2_bytes"), ("L3", "l3_bytes")):
                    m = re.search(r"^%s cache:\s*(\d+)" % key, txt, re.M)
                    if m:
                        out[name] = int(m.group(1))
                out["source"] = "lscpu"
            except Exception:
                pass
    elif system == "Darwin":
        for key, name in (("hw.l1dcachesize", "l1d_bytes"), ("hw.l2cachesize", "l2_bytes"),
                          ("hw.l3cachesize", "l3_bytes"), ("hw.cachelinesize", "cache_line_bytes")):
            v = _sysctl(key)
            if v and v.isdigit():
                out[name] = int(v)
        out["source"] = "sysctl"
    if out["cache_line_bytes"] is None:
        out["cache_line_bytes"] = 64  # typical default
        out["cache_line_assumed"] = True
    return out


def memory_info():
    try:
        import psutil
        return {"total_ram_bytes": psutil.virtual_memory().total}
    except Exception:
        return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student-id", required=True, help="8-digit student ID")
    args = ap.parse_args()
    if not re.fullmatch(r"\d{8}", args.student_id):
        print("[warn] student ID is not 8 digits: %r" % args.student_id, file=sys.stderr)

    os.makedirs(RESULTS, exist_ok=True)
    profile = {
        "cpu": cpu_info(),
        "topology": core_counts(),
        "cache": cache_info(),
        "memory": memory_info(),
        "os": {"system": platform.system(), "kernel": platform.release(),
               "version": platform.version(), "machine": platform.machine()},
        "python": {"implementation": platform.python_implementation(),
                   "version": platform.python_version(), "compiler": platform.python_compiler(),
                   "executable": sys.executable},
    }
    with open(os.path.join(RESULTS, "host_profile.json"), "w") as f:
        json.dump(profile, f, indent=2)

    params = get_student_params(args.student_id)
    with open(os.path.join(RESULTS, "salted_params.json"), "w") as f:
        json.dump(params, f, indent=2)

    print(json.dumps(profile, indent=2))
    print("Salted params:", params)


if __name__ == "__main__":
    main()
