#!/usr/bin/env python3
"""Turn a controlled RAG qualification bundle into a ModelJars ``qualifications.json`` entry.

The bundle is what ``scripts/run-controlled-rag-qualification.sh`` writes: ``qualification.json``,
``models-<backend>.json``, and ``default-correctness/{models-<backend>.json,smoke.json}``. Every
number in the entry is read from those files or recomputed from their per-run records; nothing is
typed by hand. The report paths are prefixed with the path the bundle occupies inside the Models
repository, because ModelJars fetches them from ``raw.githubusercontent.com`` at the catalog's
``modelsRevision``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def assemble(bundle: Path, model_name: str, report_prefix: str) -> dict:
    qualification_file = load(bundle / "qualification.json")
    verdict = qualification_file["qualification"]
    backend = verdict["candidateBackend"]
    report_name = f"models-{backend}.json"
    report_path = bundle / report_name
    report = load(report_path)
    report_sha = sha256(report_path)
    candidate = qualification_file["candidate"]
    if candidate["sha256"] != report_sha:
        raise SystemExit(
            f"{report_path} hash {report_sha} does not match qualification.json candidate {candidate['sha256']}"
        )
    if report["modelId"] != verdict["modelId"] or report["backend"] != backend:
        raise SystemExit("qualification.json and the candidate report disagree on model or backend")
    if report["artifactSha256"] != verdict["artifactSha256"]:
        raise SystemExit("qualification.json and the candidate report disagree on the artifact hash")

    smoke = load(bundle / "default-correctness" / "smoke.json")
    smoke_report = bundle / "default-correctness" / Path(smoke["report"]).name
    if smoke["report"] != f"default-correctness/{smoke_report.name}":
        raise SystemExit(f"unexpected smoke report path {smoke['report']}")
    if sha256(smoke_report) != smoke["reportSha256"]:
        raise SystemExit(f"{smoke_report} does not match smoke.json reportSha256")
    if smoke["artifactSha256"] != report["artifactSha256"] or smoke["backend"] != backend:
        raise SystemExit("smoke.json artifact or backend does not match the candidate report")

    runs = report["runs"]
    if not runs:
        raise SystemExit("candidate report has no runs")
    settings = report["settings"]
    summary = report["summary"]
    raw_correct = sum(1 for run in runs if run["rawEvaluation"]["correct"]) / len(runs)
    fallback = sum(1 for run in runs if run["grounding"]["decision"] == "EXTRACTIVE_FALLBACK") / len(runs)

    return {
        "modelId": report["modelId"],
        "model": model_name,
        "backend": backend,
        "backendVersion": report["backendVersion"],
        "workload": settings["workload"],
        "corpusSha256": settings["corpusSha256"],
        "promptTemplate": settings["promptTemplate"],
        "groundingPolicy": settings["groundingPolicy"],
        "artifactSha256": report["artifactSha256"],
        "artifactSizeBytes": report["artifactSizeBytes"],
        "report": f"{report_prefix}/{report_name}",
        "reportSha256": report_sha,
        "performanceTier": report["performanceTier"],
        "verdict": verdict["verdict"],
        "qualified": verdict["qualified"],
        "defaultConfigurationSmoke": {
            "configuration": smoke["configuration"],
            "backend": backend,
            "artifactSha256": smoke["artifactSha256"],
            "report": f"{report_prefix}/{smoke['report']}",
            "reportSha256": smoke["reportSha256"],
            "totalAttempts": smoke["totalAttempts"],
            "successfulAttempts": smoke["successfulAttempts"],
            "tuningSystemProperties": smoke["tuningSystemProperties"],
        },
        "attempts": summary["totalAttempts"],
        "p95RetrievalMillis": summary["retrievalMillis"]["p95"],
        "p95TtftMillis": summary["ttftMillis"]["p95"],
        "p95TpotMillis": summary["tpotMillis"]["p95"],
        "p95EndToEndMillis": summary["endToEndMillis"]["p95"],
        "p50PrefillTokensPerSecond": summary["p50PrefillTokensPerSecond"],
        "p50DecodeTokensPerSecond": summary["p50DecodeTokensPerSecond"],
        "peakRssBytes": summary["peakRssBytes"],
        "correctAnswerRate": summary["correctAnswerRate"],
        "rawCorrectAnswerRate": raw_correct,
        "abstentionAccuracy": summary["abstentionAccuracy"],
        "modelAnswerRate": verdict["modelAnswerRate"],
        "modelAnswerCorrectRate": verdict["modelAnswerCorrectRate"],
        "extractiveFallbackRate": fallback,
        "environment": report["environment"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("bundle", type=Path, help="directory written by run-controlled-rag-qualification.sh")
    parser.add_argument("--model-name", required=True, help="display name as listed in ModelJars models.json")
    parser.add_argument(
        "--report-prefix",
        required=True,
        help="path of the bundle inside the Models repository, e.g. benchmark-results/certified-20260916/rag/x",
    )
    parser.add_argument("--output", type=Path, help="write the entry here instead of stdout")
    args = parser.parse_args()
    entry = assemble(args.bundle, args.model_name, args.report_prefix.rstrip("/"))
    text = json.dumps(entry, indent=2) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
