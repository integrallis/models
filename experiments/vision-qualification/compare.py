#!/usr/bin/env python3
"""Fail closed on missing/incomparable measurements. Integration gate, not catalog admission."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np


def compare(candidate, oracle, max_latency_ratio=None):
    if max_latency_ratio is not None and (not np.isfinite(max_latency_ratio) or max_latency_ratio <= 0):
        raise ValueError("Latency ratio must be positive and finite")
    a = json.loads((candidate / "report.json").read_text())
    b = json.loads((oracle / "report.json").read_text())
    for key in ("schema", "host", "threads", "mmprojSha256", "imageSha256", "imageSize", "tokens", "dimension", "warmups", "repetitions"):
        if a[key] != b[key]:
            raise ValueError(f"Incomparable {key}: {a[key]!r} != {b[key]!r}")
    if a.get("candidateRevision") in (None, "unrecorded"):
        raise ValueError("Missing candidate revision")
    result = {"candidate": a, "oracle": b}
    for name, count in (("pixels", 3 * a["imageSize"] ** 2), ("embeddings", a["tokens"] * a["dimension"])):
        x = np.fromfile(candidate / (name + ".f32"), dtype="<f4").astype(np.float64)
        y = np.fromfile(oracle / (name + ".f32"), dtype="<f4").astype(np.float64)
        if x.size != count or y.size != count or not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError(f"Invalid {name} shape or nonfinite values")
        norm = np.linalg.norm(x) * np.linalg.norm(y)
        if not norm:
            raise ValueError(f"Degenerate {name}")
        result[name] = dict(maxAbs=float(np.max(np.abs(x-y))), cosine=float(x.dot(y)/norm),
                            normalizedRmse=float(np.linalg.norm(x-y)/np.linalg.norm(y)))
    for report in (a, b):
        times = report["encoderProjectorMs"]
        if len(times) != report["repetitions"] or len(times) < 3 or not all(np.isfinite(x) and x > 0 for x in times):
            raise ValueError("Incomplete latency measurements")
    result["candidateMedianMs"] = float(np.median(a["encoderProjectorMs"]))
    result["oracleMedianMs"] = float(np.median(b["encoderProjectorMs"]))
    result["candidateToOracleLatencyRatio"] = result["candidateMedianMs"] / result["oracleMedianMs"]
    result["numericalPass"] = result["pixels"]["maxAbs"] <= 0.02 and result["embeddings"]["cosine"] >= 0.999 and result["embeddings"]["normalizedRmse"] <= 0.01
    result["catalogQualified"] = False
    if max_latency_ratio is not None:
        result["performancePass"] = result["candidateToOracleLatencyRatio"] <= max_latency_ratio
        result["maximumLatencyRatio"] = max_latency_ratio
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("candidate", type=Path)
    p.add_argument("oracle", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--max-latency-ratio", type=float)
    args = p.parse_args()
    verdict = compare(args.candidate, args.oracle, args.max_latency_ratio)
    args.output.write_text(json.dumps(verdict, indent=2) + "\n")
    print(json.dumps({k:v for k,v in verdict.items() if k not in ("candidate", "oracle")}))
    sys.exit(0 if verdict["numericalPass"] and verdict.get("performancePass", True) else 1)
