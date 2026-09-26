#!/usr/bin/env python3
"""Independent NumPy re-implementation of the fusion rules, for validity gate G2.

Reads a dump written by `logit-fusion gate g2-dump` (member raw logits plus the Java runtime's
fused scores), recomputes every rule in float64 from the float32 member logits, and compares.

  poe:     s = sum_i w_i * log_softmax(logits_i)
  mixture: s = logsumexp over i with w_i > 0 of (log w_i + log_softmax(logits_i))
  article: s = sum_i w_i * logits_i                      (raw logits, as the article published)
  fused  = log_softmax(s)                                (T = 1)

PASS iff max |delta| <= tolerance on both `fused` and `fusedRaw` for every item, step and rule.
Usage: python3 reference_fusion.py --dump DIR [--tolerance 1e-4] [--report PATH]
Exit status: 0 PASS, 1 FAIL. Requires numpy (see requirements-reference.txt).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

RULES = ("poe", "mixture", "article")


def log_softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    peak = x.max(axis=-1, keepdims=True)
    shifted = x - peak
    return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))


def fuse(rule: str, logits: list[np.ndarray], weights: list[float]) -> np.ndarray:
    """Returns the raw fused score s, shape [steps, vocab], in float64."""
    if rule == "poe":
        return sum(w * log_softmax(l) for w, l in zip(weights, logits))
    if rule == "mixture":
        terms = np.stack([np.log(w) + log_softmax(l) for w, l in zip(weights, logits) if w > 0])
        peak = terms.max(axis=0)
        return peak + np.log(np.exp(terms - peak).sum(axis=0))
    if rule == "article":
        return sum(w * np.asarray(l, dtype=np.float64) for w, l in zip(weights, logits))
    raise ValueError(f"unknown rule: {rule}")


def read_matrix(path: Path, steps: int, vocab: int) -> np.ndarray:
    data = np.fromfile(path, dtype="<f4")
    if data.size != steps * vocab:
        raise ValueError(f"{path.name}: expected {steps}x{vocab} float32 values, found {data.size}")
    return data.reshape(steps, vocab)


def check(dump: Path, tolerance: float) -> dict:
    manifest = json.loads((dump / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("dtype", "float32-le") != "float32-le":
        raise ValueError("unsupported dtype " + str(manifest.get("dtype")))
    vocab, weights, members = manifest["vocab"], manifest["weights"], manifest["members"]
    rules = manifest.get("rules", list(RULES))
    worst = {rule: {"fused": 0.0, "fusedRaw": 0.0} for rule in rules}
    items = []
    for item in manifest["items"]:
        steps = item["steps"]
        logits = [read_matrix(dump / item["members"][m], steps, vocab) for m in members]
        per_rule = {}
        for rule in rules:
            raw = fuse(rule, logits, weights)
            expected = {"fusedRaw": raw, "fused": log_softmax(raw)}
            diffs = {}
            for kind in ("fused", "fusedRaw"):
                actual = read_matrix(dump / item[kind][rule], steps, vocab).astype(np.float64)
                delta = float(np.max(np.abs(actual - expected[kind]))) if steps else 0.0
                if not np.isfinite(delta):
                    delta = float("inf")
                diffs[kind] = delta
                worst[rule][kind] = max(worst[rule][kind], delta)
            per_rule[rule] = diffs
        items.append({"id": item["id"], "steps": steps, "maxAbsDiff": per_rule})
    max_diff = max((v for r in worst.values() for v in r.values()), default=0.0)
    return {
        "schemaVersion": 1,
        "kind": "gate-g2",
        "implementation": "numpy float64 reference_fusion.py",
        "numpyVersion": np.__version__,
        "dump": str(dump),
        "tolerance": tolerance,
        "members": members,
        "weights": weights,
        "vocab": vocab,
        "itemCount": len(items),
        "maxAbsDiff": max_diff,
        "maxAbsDiffByRule": worst,
        "passed": bool(items) and max_diff <= tolerance,
        "items": items,
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="G2: independent fusion-rule check")
    parser.add_argument("--dump", required=True, type=Path)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    report = check(args.dump, args.tolerance)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    verdict = "PASS" if report["passed"] else "FAIL"
    print(f"{verdict} G2 maxAbsDiff={report['maxAbsDiff']:.3e} tolerance={args.tolerance} items={report['itemCount']}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
