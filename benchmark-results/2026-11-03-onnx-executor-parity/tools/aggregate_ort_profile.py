#!/usr/bin/env python3
"""Reduce an onnxruntime profile to kernel time per (operator, input shapes).

usage: aggregate_ort_profile.py PROFILE.json > by-op-shape.json
"""
import collections, json, sys

events = json.load(open(sys.argv[1]))
micros = collections.Counter()
calls = collections.Counter()
for e in events:
    if e.get("cat") == "Node" and e["name"].endswith("_kernel_time"):
        a = e["args"]
        key = (a.get("op_name", "?"), json.dumps(a.get("input_type_shape")))
        micros[key] += e["dur"]
        calls[key] += 1
rows = [{"op": op, "inputs": json.loads(shapes), "micros": us, "calls": calls[(op, shapes)]}
        for (op, shapes), us in micros.most_common()]
json.dump({"source": "onnxruntime profile, kernel_time events", "rows": rows}, sys.stdout, indent=1)
print()
