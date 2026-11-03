#!/usr/bin/env python3
"""Per-operator time of the Java executor against onnxruntime on the same 50 Harvard sentences.

Reads raw/*: onnxruntime's kernel time per operator (optimizations off, so node for node with
the executor's graph; and on, for the fusion ablation), and the executor's per-operator time from
tools/per-op-timing.patch, for the released 0.5.12 and for each later build directory given.

usage: derive.py [EXTRA_RUN_DIR ...]   (each a raw/ subdirectory holding per-op.json)
"""
import json, pathlib, sys

root = pathlib.Path(__file__).resolve().parent.parent / "raw"
off = json.load(open(root / "ort-optimizations-off/summary.json"))
on = json.load(open(root / "ort-optimizations-on/summary.json"))
runs = ["ours-released-0.5.12"] + sys.argv[1:]
ours = {r: {e["op"]: e["nanos"] / 1e9 for e in json.load(open(root / r / "per-op.json"))["byOp"]} for r in runs}
ort = {e["op"]: e["micros"] / 1e6 for e in off["byOp"]}

print(f"onnxruntime, 1 thread: {on['realTimeMultiple']:.1f}x real time with graph optimizations, "
      f"{off['realTimeMultiple']:.1f}x without; kernel time {on['kernelMicros']/1e6:.2f} s and {off['kernelMicros']/1e6:.2f} s")
print()
print("| operator | " + " | ".join(f"{r} (s)" for r in runs) + " | onnxruntime (s) |")
print("| --- | " + " | ".join("---:" for _ in runs) + " | ---: |")
ops = sorted(set(ours[runs[0]]) | set(ours[runs[-1]]), key=lambda o: -ours[runs[0]].get(o, 0))
for op in ops[:20]:
    print(f"| {op} | " + " | ".join(f"{ours[r].get(op, 0):.2f}" for r in runs) + f" | {ort.get(op, 0):.2f} |")
print("| **all operators** | " + " | ".join(f"**{sum(ours[r].values()):.2f}**" for r in runs)
      + f" | **{sum(ort.values()):.2f}** |")
