#!/usr/bin/env python3
"""Re-derives every number in NOTES.md from the committed raw oracle outputs."""
import json, math, pathlib, sys

RAW = pathlib.Path(__file__).parent / "raw"


def worst_probe_cosine(a, b):
    worst = 1.0
    for x, y in zip(a, b):
        dot = sum(p * q for p, q in zip(x, y))
        na = math.sqrt(sum(p * p for p in x))
        nb = math.sqrt(sum(q * q for q in y))
        worst = min(worst, dot / (na * nb))
    return worst


def load(name):
    return json.loads((RAW / name).read_text())


def main():
    q6 = load("oracle_Q6_K.default.json")
    q6b1 = load("oracle_Q6_K.b1.json")
    f16 = load("oracle_F16.default.json")
    f16b1 = load("oracle_F16.b1.json")
    print(f"probes                                    {len(q6)}")
    print(f"oracle Q6_K default vs -b 1 -ub 1         {worst_probe_cosine(q6, q6b1):.7f}")
    print(f"oracle F16  default vs -b 1 -ub 1         {worst_probe_cosine(f16, f16b1):.7f}")
    print(f"oracle Q6_K vs oracle F16                 {worst_probe_cosine(q6, f16):.7f}")
    print()
    print("gate verdicts (ours vs oracle, same artifact):")
    for line in (RAW / "gate-verdicts.tsv").read_text().splitlines():
        if not line.strip():
            continue
        fields = line.split("\t")
        print(f"  {fields[0]:52s} {fields[1]:15s} {fields[4] if len(fields) > 4 else ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
