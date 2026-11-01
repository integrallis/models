#!/usr/bin/env python3
"""Aggregate pass.json + judge.json files under an evidence root into one report. Nothing hand-typed.

usage: derive.py EVIDENCE_ROOT > report.json
Layout expected: ROOT/<arm>/<corpus>/<voice>/t<threads>/pass<n>/{pass.json,judge.json}
"""
import json, pathlib, statistics, sys


def p95(values):
    values = sorted(values)
    if not values: return None
    k = max(0, int(round(0.95 * (len(values) - 1))))
    return values[k]


def main(root):
    root = pathlib.Path(root); out = {"root": str(root), "groups": []}
    for pass_json in sorted(root.glob("*/*/*/t*/pass*/pass.json")):
        d = pass_json.parent; p = json.loads(pass_json.read_text())
        j = json.loads((d / "judge.json").read_text()) if (d / "judge.json").exists() else None
        rtfs = [r["realTimeFactor"] for r in p["rows"] if r["realTimeFactor"] is not None]
        out["groups"].append({
            "arm": d.parents[3].name, "corpus": p["corpus"], "voice": p["voice"], "threads": p["intraOpThreads"], "pass": p["pass"],
            "realTimeMultiple": p["realTimeMultiple"], "p95RealTimeFactor": p95(rtfs), "medianRealTimeFactor": statistics.median(rtfs) if rtfs else None,
            "p95TimeToFirstAudioMillis": (p95([r["timeToFirstAudioSeconds"] for r in p["rows"]]) or 0) * 1000.0,
            "allFinite": p["allFinite"], "sentences": p["sentences"],
            "werPublisherNormalizer": j and j["werPublisherNormalizer"], "werWhisperNormalizer": j and j["werWhisperNormalizer"],
            "judge": j and j["judge"], "judgeModelSha256": j and j["modelSha256"], "path": str(d.relative_to(root)),
        })
    # Across-pass summary per (arm, corpus, voice, threads)
    summary = {}
    for g in out["groups"]:
        key = (g["arm"], g["corpus"], g["voice"], g["threads"]); summary.setdefault(key, []).append(g)
    out["summary"] = []
    for (arm, corpus, voice, threads), gs in sorted(summary.items()):
        wers = [g["werPublisherNormalizer"] for g in gs if g["werPublisherNormalizer"] is not None]
        xs = [g["realTimeMultiple"] for g in gs if g["realTimeMultiple"] is not None]
        out["summary"].append({"arm": arm, "corpus": corpus, "voice": voice, "threads": threads, "passes": len(gs),
                               "werMin": min(wers) if wers else None, "werMax": max(wers) if wers else None,
                               "realTimeMultipleMin": min(xs) if xs else None, "realTimeMultipleMax": max(xs) if xs else None,
                               "p95RealTimeFactorMax": max(g["p95RealTimeFactor"] for g in gs)})
    json.dump(out, sys.stdout, indent=1); print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
