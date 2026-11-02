#!/usr/bin/env python3
"""Derives the README table from ab-raw.txt. No number in the README is typed by hand."""
import re, statistics, sys, pathlib

raw = (pathlib.Path(__file__).parent / "ab-raw.txt").read_text().splitlines()
shape, data = None, {}
hashes = set()
for line in raw:
    if line.startswith("## "):
        shape = line[3:].strip()
        data[shape] = {"extracted": [], "parent": []}
    m = re.match(r"(extracted|parent)\s+hash=(\w+)\s+best_ms=([\d.]+)", line)
    if m and shape:
        data[shape][m.group(1)].append(float(m.group(3)))
        data[shape].setdefault("hashes", set()).add(m.group(2))
        hashes.add(m.group(2))

# Bit-exactness is per shape: a different shape computes a different product, so one hash
# per shape -- shared by both arms -- is the assertion that the extraction changed no arithmetic.
for shape, arms in data.items():
    hs = arms["hashes"]
    verdict = "IDENTICAL across both arms" if len(hs) == 1 else f"DIFFERS: {sorted(hs)}"
    print(f"output hash, {shape}: {verdict} ({sorted(hs)[0]})")
print()
print("| shape | extracted best_ms (median) | parent best_ms (median) | ratio | passes extracted faster |")
print("|---|---:|---:|---:|---:|")
for shape, arms in data.items():
    e, p = arms["extracted"], arms["parent"]
    me, mp = statistics.median(e), statistics.median(p)
    wins = sum(1 for a, b in zip(e, p) if a < b)
    print(f"| {shape} | {me:.3f} (n={len(e)}, {min(e):.3f}-{max(e):.3f}) | "
          f"{mp:.3f} (n={len(p)}, {min(p):.3f}-{max(p):.3f}) | {me/mp:.4f} | {wins}/{len(e)} |")
