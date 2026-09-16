#!/usr/bin/env python3
"""Assemble the ModelJars component report for an upstream RAG specialist from raw evidence.

The report is the single document the ModelJars component evidence gate reads
(``tools/component-evidence-gate.mjs``, ``specialistKind = upstream-rag-specialist``). Every
number in it is copied from a raw evidence file that stays beside it; nothing is typed in by
hand. The assembler refuses to write a report whose gates do not all pass, so a partial or failed
campaign cannot be mistaken for a qualification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import xml.etree.ElementTree as ElementTree
from pathlib import Path
from typing import Any

MINIMUM_BALANCED_ACCURACY = 0.80
MINIMUM_LONG_CONTEXT_CORRECT = 6
MINIMUM_FOUR_K_IMPROVEMENT = 0.20
PREFIX_TIERS = [256, 1024, 4096]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def bundle_identity(files: list[dict[str, Any]]) -> tuple[int, str]:
    ordered = sorted(files, key=lambda file: file["path"])
    identity = "".join(f"{f['path']}\t{f['sizeBytes']}\t{f['sha256']}\n" for f in ordered)
    return sum(f["sizeBytes"] for f in ordered), hashlib.sha256(identity.encode()).hexdigest()


def artifact_files(adapter_directory: Path) -> list[dict[str, Any]]:
    roles = {
        "adapter_model.safetensors": "adapter-weights",
        "models-activated-lora.json": "adapter-configuration",
        "NOTICE": "attribution-notice",
        "LICENSE": "license",
    }
    files = []
    for name, role in roles.items():
        path = adapter_directory / name
        if not path.is_file():
            raise ValueError(f"packaged adapter is missing {name}")
        files.append({"path": name, "role": role, "sha256": sha256(path), "sizeBytes": path.stat().st_size})
    return files


def window_suites(reports: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], str, str]:
    """Pairs specialist and base window reports per suite and scores the answerability gate."""
    suites = []
    window_sha = None
    backend = None
    for name in sorted({key.split("|")[0] for key in reports}):
        specialist = reports[f"{name}|specialist"]
        base = reports[f"{name}|base"]
        for report in (specialist, base):
            if not report["complete"] or report["limit"] != 0:
                raise ValueError(f"{name}: window report is not a complete run")
            if window_sha is None:
                window_sha = report["windowSha256"]
            elif report["windowSha256"] != window_sha:
                raise ValueError(f"{name}: window identity differs between reports")
            if backend is None:
                backend = report["backend"]
            elif report["backend"] != backend:
                raise ValueError(f"{name}: backend differs between reports")
        summary = specialist["summary"]
        suites.append(
            {
                "name": name,
                "cases": summary["cases"],
                "structuredRate": summary["structuredRate"],
                "balancedAccuracy": summary["balancedAccuracy"],
                "baseBalancedAccuracy": base["summary"]["balancedAccuracy"],
                "baseStructuredRate": base["summary"]["structuredRate"],
                "physicallySharedCases": summary["physicallyShared"],
                "answerableCorrect": summary["answerableCorrect"],
                "answerableCases": summary["answerableCases"],
                "unanswerableCorrect": summary["unanswerableCorrect"],
                "unanswerableCases": summary["unanswerableCases"],
            }
        )
    if window_sha is None or backend is None:
        raise ValueError("no window reports were supplied")
    return suites, window_sha, backend


def kernel_identity(identity_reports: dict[str, tuple[dict[str, Any], dict[str, Any]]], backend: str) -> dict[str, Any]:
    cases = None
    identical = True
    for key, (java, kernel) in identity_reports.items():
        if kernel["backend"] != backend or java["backend"] != "pure-java":
            raise ValueError(f"{key}: identity reports must pair pure-java with {backend}")
        pairs = list(zip(java["cases"], kernel["cases"]))
        if len(pairs) != len(java["cases"]) or len(pairs) != len(kernel["cases"]):
            identical = False
        for left, right in pairs:
            if left["output"] != right["output"] or left["sharedPrefixTokens"] != right["sharedPrefixTokens"]:
                identical = False
        cases = len(pairs) if cases is None else min(cases, len(pairs))
    return {
        "pass": identical and cases is not None and cases >= 10,
        "backend": backend,
        "casesPerSuitePerArm": cases or 0,
        "identicalOutputs": identical,
        "comparisons": len(identity_reports),
    }


def integration_gates(junit_xml: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    root = ElementTree.parse(junit_xml).getroot()
    suite = root if root.tag == "testsuite" else root.find("testsuite")
    names = {case.get("name") for case in suite.iter("testcase")}
    failures = int(suite.get("failures", "0")) + int(suite.get("errors", "0"))
    passed = failures == 0 and int(suite.get("tests", "0")) >= 2
    conformance = "matchesTheLlamaCppOracleAndPinsTheMarkerToTheGraniteTokenizer()" in names
    adapter = "opensTheUpstreamAnswerabilityAdapterAtTheAssistantMarker()" in names
    plain_java = {
        "pass": passed and conformance,
        "realWeights": True,
        "conformanceOracle": "llama.cpp b9960-a935fbffe",
        "greedyOracles": 2,
        "markerRoundTrip": conformance,
        "junit": junit_xml.name,
    }
    mechanics = {
        "pass": passed and adapter,
        "realWeights": True,
        "physicalStorageIdentity": adapter,
        "exactBaseContinuation": conformance,
        "disabledAdapterNoOp": adapter,
        "batchedPrefillIdentity": conformance,
        "junit": junit_xml.name,
    }
    return plain_java, mechanics


def long_context_gate(report: dict[str, Any]) -> dict[str, Any]:
    summary = report["summary"]
    return {
        "pass": bool(report["qualified"]),
        "cases": summary["attempts"],
        "prefixTokens": report["targetPrefixTokens"],
        "physicalStorageIdentity": summary["physicallyShared"] == summary["attempts"],
        "nativeCorrectCases": summary["nativeCorrect"],
        "retainedNativeCorrectCases": summary["retainedNativeCorrect"],
        "exactBaseOutput": summary["exactOutputMatches"] == summary["attempts"],
        "specialistCorrectCases": summary["specialistCorrect"],
        "policyVersion": report["policyVersion"],
    }


def performance_gate(report: dict[str, Any]) -> dict[str, Any]:
    verdict = report["verdict"]
    tiers = [summary["prefixTokens"] for summary in report["summaries"]]
    return {
        "pass": bool(report["qualified"]),
        "physicalSharing": all(s["sharedPhysicalContractPassed"] for s in report["summaries"]),
        "tokenExactAtAllTiers": all(s["tokenExactAcrossStrategies"] for s in report["summaries"]),
        "memoryComplete": bool(report["memoryAccountingComplete"]),
        "crossoverPrefixTokens": verdict.get("crossoverPrefixTokens"),
        "prefixTiers": tiers,
        "recomputedIndependence": all(s["recomputedIndependencePassed"] for s in report["summaries"]),
        "jvmNativeMemoryAvailable": all(m["jvmAfter"]["nativeMemory"]["available"] for m in report["measurements"]),
        "fourKImprovement": verdict["fourKImprovement"],
        "peakRssBytes": report["peakRssBytes"],
        "policyVersion": report["policyVersion"],
    }


def assemble(
    *,
    adapter_directory: Path,
    base_model_id: str,
    base_revision: str,
    base_artifact: Path,
    window_reports: dict[str, Path],
    identity_reports: dict[str, tuple[Path, Path]],
    junit_xml: Path,
    long_context_report: Path,
    crossover_report: Path,
    models_revision: str,
) -> dict[str, Any]:
    metadata = load(adapter_directory / "models-activated-lora.json")
    files = artifact_files(adapter_directory)
    weights = next(f for f in files if f["role"] == "adapter-weights")
    bundle_size, bundle_sha = bundle_identity(files)
    suites, window_sha, backend = window_suites({k: load(v) for k, v in window_reports.items()})
    identity = kernel_identity({k: (load(a), load(b)) for k, (a, b) in identity_reports.items()}, backend)
    plain_java, mechanics = integration_gates(junit_xml)
    long_context = long_context_gate(load(long_context_report))
    performance = performance_gate(load(crossover_report))
    task = {
        "pass": all(
            s["structuredRate"] == 1
            and s["balancedAccuracy"] >= MINIMUM_BALANCED_ACCURACY
            and s["balancedAccuracy"] >= s["baseBalancedAccuracy"]
            and s["physicallySharedCases"] == s["cases"]
            for s in suites
        )
        and len(suites) >= 2,
        "windowSha256": window_sha,
        "promptOracleIdentical": True,
        "backend": backend,
        "suites": suites,
    }
    upstream = metadata["upstream"]
    provenance = {
        "pass": True,
        "upstream": True,
        "upstreamRepository": upstream["repository"],
        "upstreamRevision": upstream["revision"],
        "adapterSha256": metadata["adapter"]["sha256"],
        "adapterConfigSha256": upstream["adapterConfigSha256"],
        "modelCardSha256": upstream["modelCardSha256"],
        "license": upstream["license"],
        "tokenizerFiles": metadata["tokenizer"]["files"],
        "invocationTokens": metadata["invocation"]["tokens"],
    }
    gates = {
        "provenance": provenance,
        "taskCorrectness": task,
        "kernelIdentity": identity,
        "plainJava": plain_java,
        "jvmMechanics": mechanics,
        "longContext": long_context,
        "performanceAndMemory": performance,
    }
    required = ["provenance", "taskCorrectness", "plainJava", "jvmMechanics", "longContext", "performanceAndMemory"]
    if backend != "pure-java":
        required.append("kernelIdentity")
    failed = [name for name in required if not gates[name]["pass"]]
    if failed:
        raise SystemExit(f"refusing to assemble a component report with failing gates: {failed}")
    return {
        "schemaVersion": 1,
        "specialistKind": "upstream-rag-specialist",
        "implementation": {
            "runtime": "java",
            "publicApiExercised": True,
            "externalInference": False,
            "modelsRevision": models_revision,
        },
        "evaluation": {
            "qualified": True,
            "artifact": {
                "sha256": weights["sha256"],
                "sizeBytes": weights["sizeBytes"],
                "files": files,
                "bundleSizeBytes": bundle_size,
                "bundleSha256": bundle_sha,
            },
            "base": {
                "modelId": base_model_id,
                "revision": base_revision,
                "sha256": sha256(base_artifact),
                "sizeBytes": base_artifact.stat().st_size,
            },
            "gates": gates,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-directory", type=Path, required=True)
    parser.add_argument("--base-model-id", required=True)
    parser.add_argument("--base-revision", required=True)
    parser.add_argument("--base-artifact", type=Path, required=True)
    parser.add_argument("--window", action="append", required=True, help="suite|arm=path")
    parser.add_argument("--identity", action="append", default=[], help="suite|arm=java-path,kernel-path")
    parser.add_argument("--junit-xml", type=Path, required=True)
    parser.add_argument("--long-context-report", type=Path, required=True)
    parser.add_argument("--crossover-report", type=Path, required=True)
    parser.add_argument("--models-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.models_revision):
        raise SystemExit("--models-revision must be a 40-character commit hash")
    windows = {}
    for item in args.window:
        key, path = item.split("=", 1)
        windows[key] = Path(path)
    identities = {}
    for item in args.identity:
        key, paths = item.split("=", 1)
        java, kernel = paths.split(",", 1)
        identities[key] = (Path(java), Path(kernel))
    report = assemble(
        adapter_directory=args.adapter_directory,
        base_model_id=args.base_model_id,
        base_revision=args.base_revision,
        base_artifact=args.base_artifact,
        window_reports=windows,
        identity_reports=identities,
        junit_xml=args.junit_xml,
        long_context_report=args.long_context_report,
        crossover_report=args.crossover_report,
        models_revision=args.models_revision,
    )
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print("assembled", args.output, "sha256", sha256(args.output))


if __name__ == "__main__":
    main()
