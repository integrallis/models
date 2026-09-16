#!/usr/bin/env python3
"""Copy a passing controlled-RAG bundle into the certified evidence tree and write its README.

Refuses anything but a QUALIFIED verdict, copies the exact report files (not the engine logs),
writes SHA256SUMS, and renders the README from the reports so no number is typed by hand.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

REPORTS = ["qualification.json", "models-rust-ffm.json", "ollama.json", "llama.cpp.json",
           "default-correctness/models-rust-ffm.json", "default-correctness/smoke.json"]


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def publish(bundle: Path, destination: Path, title: str, host: str, tuning: str) -> None:
    verdict = load(bundle / "qualification.json")["qualification"]
    if verdict.get("verdict") != "QUALIFIED" or not verdict.get("qualified"):
        raise SystemExit(f"refusing to publish a non-qualified bundle: {verdict.get('verdict')}")
    destination.mkdir(parents=True, exist_ok=True)
    for name in REPORTS:
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(bundle / name, target)
    candidate = load(destination / "models-rust-ffm.json")
    ollama = load(destination / "ollama.json")
    llama = load(destination / "llama.cpp.json")
    smoke = load(destination / "default-correctness" / "smoke.json")
    comparisons = {c["comparatorBackend"]: c for c in verdict["comparisons"]}
    summary = candidate["summary"]
    lines = [
        f"# {title} qualification",
        "",
        f"The exact {candidate['artifactSizeBytes']:,}-byte GGUF artifact with SHA-256",
        f"`{candidate['artifactSha256']}`",
        f"clears `{load(destination / 'qualification.json')['policyId']}` at the `{candidate['performanceTier']}` tier.",
        "",
        f"The final three-iteration run used Java {candidate['environment']['javaVersion']} on {host}.",
        f"Recorded tuning for the performance phase: {tuning}. Prompt template `{candidate['settings']['promptTemplate']}`.",
        f"Models answered {summary['successfulAttempts']}/{summary['totalAttempts']} grounded requests correctly;",
        f"model answer rate {verdict['modelAnswerRate']:.3f}, model answer correct rate {verdict['modelAnswerCorrectRate']:.3f}.",
        "",
        "| Metric | Models | Ollama | llama.cpp | Gate result |",
        "| --- | ---: | ---: | ---: | ---: |",
        f"| p95 TTFT | {summary['ttftMillis']['p95']:,.1f} ms | {ollama['summary']['ttftMillis']['p95']:,.1f} ms | {llama['summary']['ttftMillis']['p95']:,.1f} ms | `{candidate['performanceTier']}` |",
        f"| Median decode | {summary['p50DecodeTokensPerSecond']:.2f} tok/s | {ollama['summary']['p50DecodeTokensPerSecond']:.2f} tok/s | {llama['summary']['p50DecodeTokensPerSecond']:.2f} tok/s | {comparisons['ollama']['decodeThroughputRatio']:.3f}x Ollama (minimum {comparisons['ollama']['minimumDecodeThroughputRatio']}x) |",
        f"| p95 end to end | {summary['endToEndMillis']['p95']:,.1f} ms | {ollama['summary']['endToEndMillis']['p95']:,.1f} ms | {llama['summary']['endToEndMillis']['p95']:,.1f} ms | {comparisons['ollama']['endToEndLatencyRatio']:.3f}x Ollama (maximum {comparisons['ollama']['maximumEndToEndLatencyRatio']}x) |",
        f"| Correct grounded answers | {summary['successfulAttempts']}/{summary['totalAttempts']} | {ollama['summary']['successfulAttempts']}/{ollama['summary']['totalAttempts']} | {llama['summary']['successfulAttempts']}/{llama['summary']['totalAttempts']} | pass |",
        "",
        "`models-rust-ffm.json`, `ollama.json`, and `llama.cpp.json` retain every measured request. Their",
        "SHA-256 values are bound by `qualification.json`.",
        "",
        f"The separate library-default smoke used the same Models commit, left native quantized decode",
        f"disabled, and passed all {smoke['totalAttempts']} workload cases with no generation failures. Its",
        f"immutable report is `default-correctness/models-rust-ffm.json` (SHA-256 `{smoke['reportSha256']}`).",
        "",
        f"Models: `{candidate['backendVersion']}`.",
        "",
    ]
    (destination / "README.md").write_text("\n".join(lines), encoding="utf-8")
    sums = "".join(f"{sha256(destination / name)}  {name}\n" for name in REPORTS + ["README.md"])
    (destination / "SHA256SUMS").write_text(sums, encoding="utf-8")
    print(f"published {destination}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("bundle", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--title", required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--tuning", required=True)
    args = parser.parse_args()
    publish(args.bundle, args.destination, args.title, args.host, args.tuning)


if __name__ == "__main__":
    main()
