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
            "arm": d.parents[3].name, "corpus": p["corpus"], "voice": p["voice"], "threads": p.get("intraOpThreads", 1), "pass": p.get("pass", 1),
            "realTimeMultiple": p["realTimeMultiple"], "p95RealTimeFactor": p95(rtfs), "medianRealTimeFactor": statistics.median(rtfs) if rtfs else None,
            "p95TimeToFirstAudioMillis": (p95([r.get("timeToFirstAudioSeconds", r.get("runSeconds")) for r in p["rows"] if r.get("timeToFirstAudioSeconds", r.get("runSeconds")) is not None]) or 0) * 1000.0,
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
    # Phonemizer: per-sentence exact match of the candidate's phoneme string against the comparator's,
    # per corpus and voice (accent follows the voice), from the first pass of each that exists.
    # Both arms record the chunk list per sentence; the comparison is on the joined string and on
    # the chunking separately. Reported, not gated.
    out["phonemizer"] = []
    first = {}
    for g in out["groups"]:
        first.setdefault((g["arm"], g["corpus"], g["voice"]), g["path"])
    for (arm, corpus, voice), path in sorted(first.items()):
        if arm != "candidate" or ("comparator", corpus, voice) not in first: continue
        ours = json.loads((root / path / "pass.json").read_text())["rows"]
        theirs = json.loads((root / first[("comparator", corpus, voice)] / "pass.json").read_text())["rows"]
        n = min(len(ours), len(theirs))
        same = [" ".join(a["phonemes"]) == " ".join(b["phonemes"]) for a, b in zip(ours, theirs)]
        chunked = sum(1 for a, b in zip(ours, theirs) if a["phonemes"] == b["phonemes"])
        entry = {"corpus": corpus, "voice": voice, "sentences": n,
                 "exactMatchRate": sum(same) / n if n else None, "exactChunkingMatchRate": chunked / n if n else None,
                 "candidatePass": path, "comparatorPass": first[("comparator", corpus, voice)]}
        # Split the mismatches by cause when the phonemizer's own report is beside the evidence:
        # sentences with an unresolved word cannot match, so the rate among the rest is the
        # tokenizer/tagger/stress agreement on its own.
        accent = "gb" if voice.startswith("b") else "us"
        report = root.parent / "phonemizer" / f"{corpus}-{accent}.json"
        if report.exists():
            unresolved = {r["index"] for r in json.loads(report.read_text())["rows"] if r["unknownWords"]}
            resolved = [i for i in range(n) if i not in unresolved]
            entry["phonemizerReport"] = str(report.relative_to(root.parent))
            entry["sentencesWithUnresolvedWords"] = len(unresolved)
            entry["exactMatchRateAmongResolvedSentences"] = (sum(same[i] for i in resolved) / len(resolved)) if resolved else None
        out["phonemizer"].append(entry)
    json.dump(out, sys.stdout, indent=1); print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
