#!/usr/bin/env python3
"""Derive the A/B table from the raw llama-bench JSON in results/ and the pod log in results/.

Every number in RESULTS.md comes from here. Nothing is hand-typed.
Usage: python3 summarize.py
"""
import glob, json, os, re, statistics as st
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")

def emit(rows, host, isa):
    print(f"\n### {host}  ({isa})\n")
    print("| model | prompt | repack ON (tok/s) | repack OFF (tok/s) | speedup | ON spread | OFF spread | verdict |")
    print("|---|---|---|---|---|---|---|---|")
    for (model, prompt), arms in sorted(rows.items()):
        on, off = arms.get("on", []), arms.get("off", [])
        if not on or not off:
            print(f"| {model} | {prompt} | {len(on)} runs | {len(off)} runs | — | — | — | incomplete |")
            continue
        mon, moff = st.median(on), st.median(off)
        sp = lambda v: (max(v) - min(v)) / st.median(v) if len(v) > 1 else 0.0
        speed = mon / moff
        # Kill criterion from README: an effect no larger than within-arm spread is inconclusive.
        verdict = "inconclusive" if (speed - 1) < max(sp(on), sp(off)) else "effect > spread"
        print(f"| {model} | {prompt} | {mon:.2f} | {moff:.2f} | **{speed:.2f}x** | "
              f"{sp(on)*100:.1f}% | {sp(off)*100:.1f}% | {verdict} |")

# --- local arm: raw llama-bench JSON ---
local = defaultdict(lambda: defaultdict(list))
for f in glob.glob(os.path.join(RES, "*.json")):
    base = os.path.basename(f)
    m = re.match(r"(.+)-p(\d+)-(on|off)-(\d+)\.json$", base)
    if not m:
        continue
    model, prompt, arm, _ = m.groups()
    try:
        d = json.load(open(f))[0]
    except Exception:
        continue
    local[(model, int(prompt))][arm].append(d["avg_ts"])
if local:
    emit(local, "Intel i7-9750H, 6 cores / 12 threads, 6 threads used",
         "AVX2, no AVX-512 — thermally limited laptop, iteration only")

# --- pod arm: RESULT lines captured from the pod log ---
pod = defaultdict(lambda: defaultdict(list))
isa = host = None
for logf in glob.glob(os.path.join(RES, "pod-*.log")):
    for line in open(logf):
        if line.startswith("Model name:"):
            host = line.split(":", 1)[1].strip()
        if line.startswith("avx512 flags:"):
            flags = line.split(":", 1)[1].strip()
            isa = ("AVX-512 (" + flags + ")") if flags else "AVX2 only"
        m = re.search(r"RESULT p=(\d+) pass=(\d+) repack=(on|off) tok_s=([\d.]+)", line)
        if m:
            prompt, _, arm, v = m.groups()
            pod[("Llama-3.2-1B-Instruct-Q4_K_M", int(prompt))][arm].append(float(v))
if pod:
    emit(pod, host or "RunPod cpu5c, 16 vCPU", isa or "unknown")
else:
    print("\n(no pod log in results/ yet)")
