#!/usr/bin/env python3
"""Derive the decode-ratio table from the committed verdicts.

Reads this directory's two integration runs and, for the comparison, the 0.5.18 attempt whose
numbers these are measured against. Every figure printed is read from a file; none is typed.

    python3 derive.py [--baseline <admission-attempt.json>]
"""
import argparse
import json
import pathlib
import statistics

SCHEDULES = [("integration", "initial, 2 warmups / 5 repetitions"),
             ("integration-confirmation-3w7r", "confirmation, 3 warmups / 7 repetitions")]
KEYS = ["decodeRatio", "imageTextPrefillRatio", "totalRatio"]


def verdict(root, schedule, kv):
    path = root / schedule / "results/cats" / f"decoder-verdict-{kv}.json"
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=pathlib.Path, default=None,
                        help="admission-attempt.json of the run this is compared against")
    arguments = parser.parse_args()
    root = pathlib.Path(__file__).parent
    before = {}
    if arguments.baseline:
        attempt = json.loads(arguments.baseline.read_text())
        for stage, schedule in [("initialIntegration", "integration"),
                                ("confirmationIntegration", "integration-confirmation-3w7r")]:
            for decoder in attempt[stage]["decoders"]:
                before[(schedule, decoder["kv"])] = decoder["decodeRatio"]

    print(f"candidate commit: {(root / 'integration/candidate-commit.txt').read_text().strip()}")
    for line in (root / "integration/cpu.txt").read_text().splitlines():
        if line.startswith(("Model name", "Thread(s) per core", "Core(s) per socket", "CPU(s):")):
            print(f"  {line.strip()}")
    print()
    header = "| schedule | reference KV | decodeRatio | prefill | total | tokens match | perf pass |"
    if before:
        header = header.replace("| decodeRatio |", "| baseline decodeRatio | decodeRatio | ratio improvement |")
    print(header)
    print("| --- " * (header.count("|") - 1) + "|")
    for schedule, label in SCHEDULES:
        for kv in ["f32", "f16"]:
            v = verdict(root, schedule, kv)
            assert v["referenceKvType"] == kv, f"{schedule} {kv} verdict names {v['referenceKvType']}"
            cells = [label, kv]
            if before:
                baseline = before.get((schedule, kv))
                cells += [f"{baseline:.5f}" if baseline else "n/a"]
            cells += [f"{v['decodeRatio']:.5f}"]
            if before and baseline:
                cells += [f"{baseline / v['decodeRatio']:.3f}x"]
            elif before:
                cells += ["n/a"]
            cells += [f"{v['imageTextPrefillRatio']:.5f}", f"{v['totalRatio']:.5f}",
                      str(v["greedyTokensMatch"]), str(v["performancePass"])]
            print("| " + " | ".join(cells) + " |")
    print()
    print("Raw candidate timings (one candidate run per schedule, shared by both reference KV comparisons):")
    print("| schedule | baseline median decode ms | candidate median decode ms | raw speedup |")
    print("| --- | --- | --- | --- |")
    for (schedule, label), name in zip(SCHEDULES, ["initial", "confirmation"]):
        baseline = json.loads((root / f"baseline-0.5.18-{name}-decoder-f16.json").read_text())
        old_ms = statistics.median(baseline["candidate"]["decodeMs"])
        new_ms = statistics.median(verdict(root, schedule, "f16")["candidate"]["decodeMs"])
        print(f"| {label} | {old_ms:.6f} | {new_ms:.6f} | {old_ms / new_ms:.6f}x |")
    print()
    print(f"ceiling applied by the harness: {verdict(root, 'integration', 'f16')['maximumLatencyRatio']}")
    for schedule, label in SCHEDULES:
        status = (root / schedule / "STATUS").read_text().strip()
        print(f"{label}: STATUS = {status}")


if __name__ == "__main__":
    main()
