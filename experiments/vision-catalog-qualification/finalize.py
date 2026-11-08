#!/usr/bin/env python3
"""Combine every required arm; backend success is distinct from catalog publication."""
import json
import sys
from pathlib import Path

from compare import measured, median, read, require, sha


def stronger_performance(root, corpus, variant, task):
    directory = root / "results" / variant
    receipt = read(directory / "reference-performance-f16-receipt.json")
    original = read(root / "results/receipt.json")
    require(receipt["host"] == original["host"] and receipt["corpusSha256"] == original["corpusSha256"], "F16 workload/host mismatch")
    require(receipt["referenceCommit"] == original["oracleCommit"], "F16 reference revision mismatch")
    require(receipt["kvType"] == "f16", "Expected F16 reference")
    rows = [json.loads(line) for line in (directory / "reference-performance-f16.jsonl").read_text().splitlines()]
    config = rows.pop(0)["configuration"]
    for key in ["threads", "context", "maximumOutputTokens"]:
        require(config[key] == receipt[key] == original[key], "F16 setting mismatch: " + key)
    require(config["kvType"] == "f16", "F16 binary configuration mismatch")
    cases = [case for case in corpus["cases"] if case["repetitions"] == 3]
    require({r["id"] for r in rows} == {c["id"] for c in cases}, "F16 performance coverage mismatch")
    f32 = {p["id"]: p for p in task["performance"]}
    results = []
    for case in cases:
        candidate = measured(read(directory / "candidate" / case["id"] / "runs.json"), case)
        decoder = measured([r for r in rows if r["id"] == case["id"]], case)
        encoder = measured(read(root / "results/encoder" / case["id"] / "runs.json"), case)
        paired = [dict(prefillMs=d["prefillMs"] + e["encoderMs"], decodeMs=d["decodeMs"],
                       totalMs=d["prefillMs"] + e["encoderMs"] + d["decodeMs"])
                  for d, e in zip(decoder, encoder, strict=True)]
        lengths_match = all(len(c["tokens"]) == len(d["tokens"]) for c, d in zip(candidate, decoder, strict=True))
        tokens_match = all(c["tokens"] == d["tokens"] for c, d in zip(candidate, decoder, strict=True))
        ratios = {key: median(candidate, key) / median(paired, key) for key in ["prefillMs", "totalMs"]}
        if any(c["tokens"] or d["tokens"] for c, d in zip(candidate, decoder, strict=True)):
            ratios["decodeMs"] = median(candidate, "decodeMs") / median(paired, "decodeMs")
        worst = {key: max(ratios.get(key, 1), f32[case["id"]]["ratios"].get(key, 1))
                 for key in ["prefillMs", "decodeMs", "totalMs"]}
        results.append(dict(id=case["id"], dataset=case["dataset"], warmups=case["warmups"], repetitions=case["repetitions"],
                            f16Ratios=ratios, worstRatios=worst, generationLengthsMatch=lengths_match,
                            tokenSequencesMatch=tokens_match,
                            passed=lengths_match and all(r <= 1.10 for r in worst.values())))
    return results


def integration(directory, receipt):
    result = dict(directory=directory.name, probes=[], decoders=[], missingEvidence=[])
    for image in ["cats", "wide-gradient", "tall-bands"]:
        path = directory / "results" / image / "verdict.json"
        if not path.is_file():
            result["missingEvidence"].append(str(path.relative_to(directory)))
            continue
        verdict = read(path)
        candidate = verdict["candidate"]
        require(candidate["candidateRevision"] == receipt["revision"] and candidate["host"] == receipt["host"], "Integration runtime/host mismatch")
        require(candidate["mmprojSha256"] == receipt["artifacts"]["llava-v1.5-7b-mmproj-model-f16.gguf"], "Integration projector mismatch")
        result["probes"].append(dict(image=image, numericalPass=verdict["numericalPass"],
                                    performancePass=verdict["performancePass"], ratio=verdict["candidateToOracleLatencyRatio"]))
    for kv in ["f32", "f16"]:
        path = directory / "results/cats" / f"decoder-verdict-{kv}.json"
        if not path.is_file():
            result["missingEvidence"].append(str(path.relative_to(directory)))
            continue
        verdict = read(path)
        result["decoders"].append(dict(kv=kv, **{k: verdict[k] for k in ["numericalPass", "performancePass", "greedyTokensMatch", "imageTextPrefillRatio", "decodeRatio", "totalRatio"]}))
    result["passed"] = not result["missingEvidence"] and all(
        r["numericalPass"] and r["performancePass"] for r in result["probes"] + result["decoders"])
    return result


def main():
    root, initial, confirmation = map(Path, sys.argv[1:])
    task_report = read(root / "results/task-verdict.json")
    corpus = read(root / "cases.json")
    receipt = task_report["receipt"]
    require(receipt["corpusSha256"] == sha(root / "cases.json"), "Corpus changed")
    first = integration(initial, receipt)
    repeated = integration(confirmation, receipt)
    result = dict(schema="llava-catalog-attempt-v1", receipt=receipt,
                  initialIntegration=first, confirmationIntegration=repeated, candidates=[],
                  catalogPublished=False, releasedRuntimeVerified=False, modeljarsRuntimeVerified=False)
    for task in task_report["variants"]:
        performance = stronger_performance(root, corpus, task["variant"], task)
        gates = dict(task["gates"], strongestReferencePerformance=all(p["passed"] for p in performance),
                     integrationConfirmation=repeated["passed"])
        result["candidates"].append(dict(variant=task["variant"], backendQualified=all(gates.values()),
                                         gates=gates, datasets=task["datasets"], tokenAgreement=task["tokenAgreement"],
                                         disagreements=task["disagreements"], performance=performance))
    (root / "results/admission-attempt.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if all(c["backendQualified"] for c in result["candidates"]) else 1


if __name__ == "__main__":
    sys.exit(main())
