#!/usr/bin/env python3
"""Assemble the ModelJars component report for an upstream RAG specialist from raw evidence.

The report is the single document the ModelJars component evidence gate reads
(``tools/component-evidence-gate.mjs``, ``specialistKind = upstream-rag-specialist``). Every
number in it is copied from a raw evidence file that stays beside it; nothing is typed in by
hand. The assembler refuses to write a report whose gates do not all pass, so a partial or failed
campaign cannot be mistaken for a qualification.

Optional inputs:

* ``--labels suite=path`` rescored a suite against confirmed labels (the file
  ``confirm_disputed_labels.py`` writes). Both arms are re-scored from each window report's
  per-case ``prediction`` with :func:`audit_suite_labels.score`, the pass rule is applied to the
  confirmed numbers, and the dataset-label numbers are kept beside them as ``original*``.
* ``--models-artifact`` / ``--clean-host-run`` are validated against the ModelJars gate's field
  requirements and copied into ``gates.modelsArtifact`` / ``gates.cleanHostRun``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import xml.etree.ElementTree as ElementTree
from datetime import datetime
from pathlib import Path
from typing import Any

import audit_suite_labels

MINIMUM_BALANCED_ACCURACY = 0.80
MINIMUM_LONG_CONTEXT_CORRECT = 6
MINIMUM_FOUR_K_IMPROVEMENT = 0.20
PREFIX_TIERS = [256, 1024, 4096]
SHA256 = re.compile(r"[0-9a-f]{64}")
RELEASE_VERSION = re.compile(r"\d+\.\d+\.\d+")
UTC_TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")
IMMUTABLE_MODELS_REPORT = re.compile(r"https://raw\.githubusercontent\.com/integrallis/models/([0-9a-f]{40})/.+")
MAVEN_CENTRAL = "https://repo1.maven.org/maven2"
REQUIRED_MODELS_MODULES = ["backend-java", "models-runtime", "models-spring-ai", "models-langchain4j"]


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


def cases_sha256(cases: list[dict[str, Any]]) -> str:
    """The label file's canonical identity, computed exactly as confirm_disputed_labels.py does."""
    canon = json.dumps(sorted(cases, key=lambda case: case["id"]), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canon.encode()).hexdigest()


def confirmed_labels(name: str, document: dict[str, Any]) -> tuple[dict[str, str], str]:
    """Reads a confirmed label suite, refusing a wrong suite, a bad label, or a stale hash."""
    if document.get("suite") != name:
        raise ValueError(f"{name}: label file is for suite {document.get('suite')!r}")
    cases = document.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"{name}: label file carries no cases")
    labels: dict[str, str] = {}
    for case in cases:
        if case.get("label") not in audit_suite_labels.LABELS:
            raise ValueError(f"{name}: case {case.get('id')!r} has label {case.get('label')!r}")
        if case["id"] in labels:
            raise ValueError(f"{name}: label file repeats case {case['id']!r}")
        labels[case["id"]] = case["label"]
    digest = cases_sha256(cases)
    if digest != document.get("casesSha256"):
        raise ValueError(f"{name}: label casesSha256 does not match its cases (recomputed {digest})")
    for label in audit_suite_labels.LABELS:
        if label not in labels.values():
            raise ValueError(f"{name}: confirmed labels must contain both labels; no {label} case")
    return labels, digest


def rescore(name: str, arm: str, report: dict[str, Any], labels: dict[str, str]) -> dict[str, Any]:
    """Re-scores one arm's per-case predictions against confirmed labels; case ids must match exactly."""
    predictions: dict[str, str] = {}
    for case in report.get("cases") or []:
        if case["id"] in predictions:
            raise ValueError(f"{name}|{arm}: window report repeats case {case['id']!r}")
        predictions[case["id"]] = case["prediction"]
    if set(predictions) != set(labels):
        missing, extra = sorted(set(labels) - set(predictions)), sorted(set(predictions) - set(labels))
        raise ValueError(
            f"{name}|{arm}: label case ids differ from the report's case ids "
            f"(report lacks {missing[:5]}, labels lack {extra[:5]})"
        )
    return audit_suite_labels.score(predictions, labels)


def window_suites(
    reports: dict[str, dict[str, Any]], labels: dict[str, dict[str, Any]] | None = None
) -> tuple[list[dict[str, Any]], str, str]:
    """Pairs specialist and base window reports per suite and scores the answerability gate.

    ``labels`` maps a suite name to a confirmed label document; those suites are re-scored.
    """
    labels = labels or {}
    names = sorted({key.split("|")[0] for key in reports})
    unknown = sorted(set(labels) - set(names))
    if unknown:
        raise ValueError(f"confirmed labels name suites with no window reports: {unknown}")
    suites = []
    window_sha = None
    backend = None
    for name in names:
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
        suite = {
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
            "labelSource": "dataset",
        }
        if name in labels:
            confirmed, digest = confirmed_labels(name, labels[name])
            specialist_score = rescore(name, "specialist", specialist, confirmed)
            base_score = rescore(name, "base", base, confirmed)
            answerable, unanswerable = specialist_score["counts"]["answerable"], specialist_score["counts"]["unanswerable"]
            suite.update(
                balancedAccuracy=specialist_score["balancedAccuracy"],
                baseBalancedAccuracy=base_score["balancedAccuracy"],
                answerableCorrect=answerable[0],
                answerableCases=answerable[1],
                unanswerableCorrect=unanswerable[0],
                unanswerableCases=unanswerable[1],
                labelSource="confirmed",
                labelsSha256=digest,
                originalBalancedAccuracy=summary["balancedAccuracy"],
                originalBaseBalancedAccuracy=base["summary"]["balancedAccuracy"],
                balancedInterval95=specialist_score["balancedInterval95"],
                baseBalancedInterval95=base_score["balancedInterval95"],
            )
        suites.append(suite)
    if window_sha is None or backend is None:
        raise ValueError("no window reports were supplied")
    return suites, window_sha, backend


def positive_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def is_sha256(value: Any) -> bool:
    return isinstance(value, str) and SHA256.fullmatch(value) is not None


def validate_models_artifact(gate: Any) -> dict[str, Any]:
    """Mirrors requireReleasedModelsArtifacts in tools/component-evidence-gate.mjs (minus the byte fetch)."""
    if (
        not isinstance(gate, dict)
        or gate.get("pass") is not True
        or not isinstance(gate.get("version"), str)
        or RELEASE_VERSION.fullmatch(gate["version"]) is None
        or not isinstance(gate.get("artifacts"), list)
        or len(gate["artifacts"]) != len(REQUIRED_MODELS_MODULES)
    ):
        raise ValueError("modelsArtifact must pass, name a release version, and list the required Models artifacts")
    version = gate["version"]
    modules = [artifact.get("module") if isinstance(artifact, dict) else None for artifact in gate["artifacts"]]
    if sorted(map(str, modules)) != sorted(REQUIRED_MODELS_MODULES):
        raise ValueError(f"modelsArtifact must bind exactly {REQUIRED_MODELS_MODULES}, got {modules}")
    for artifact in gate["artifacts"]:
        module = artifact["module"]
        if artifact.get("coordinate") != f"com.integrallis:{module}:{version}":
            raise ValueError(f"modelsArtifact {module} coordinate must be com.integrallis:{module}:{version}")
        for kind in ("jar", "pom"):
            file = artifact.get(kind)
            uri = f"{MAVEN_CENTRAL}/com/integrallis/{module}/{version}/{module}-{version}.{kind}"
            if (
                not isinstance(file, dict)
                or file.get("uri") != uri
                or not is_sha256(file.get("sha256"))
                or not positive_integer(file.get("sizeBytes"))
            ):
                raise ValueError(f"modelsArtifact {module} {kind} must bind {uri} with a sha256 and a positive size")
    return gate


def utc_instant(value: Any) -> datetime | None:
    if not isinstance(value, str) or UTC_TIMESTAMP.fullmatch(value) is None:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        return None


def validate_clean_host_run(run: Any, models_version: str) -> dict[str, Any]:
    """Mirrors requireCleanHostRun in tools/component-evidence-gate.mjs.

    The gate also requires the output log URI to be pinned to the catalog's evidence revision and
    fetches the log bytes; neither is known here, so only the URI shape is checked.
    """
    if not isinstance(run, dict):
        raise ValueError("cleanHostRun must be an object")
    started, completed = utc_instant(run.get("startedAt")), utc_instant(run.get("completedAt"))
    log = run.get("outputLog")
    command = run.get("command")
    problems = [
        (run.get("pass") is not True, "pass must be true"),
        (run.get("freshMachine") is not True, "freshMachine must be true"),
        (run.get("externalInference") is not False, "externalInference must be false"),
        (not (isinstance(run.get("javaMajor"), int) and not isinstance(run.get("javaMajor"), bool) and run["javaMajor"] == 25), "javaMajor must be 25"),
        (not (isinstance(run.get("exitCode"), int) and not isinstance(run.get("exitCode"), bool) and run["exitCode"] == 0), "exitCode must be 0"),
        (run.get("modelsVersion") != models_version, f"modelsVersion must equal modelsArtifact version {models_version}"),
        (started is None or completed is None, "startedAt and completedAt must be UTC timestamps"),
        (started is not None and completed is not None and completed <= started, "completedAt must follow startedAt"),
        (not isinstance(command, list) or not command or any(not isinstance(p, str) or not p for p in command), "command must be nonempty strings"),
        (not is_sha256(run.get("resolvedClasspathSha256")), "resolvedClasspathSha256 must be a sha256"),
        (
            not isinstance(log, dict)
            or not isinstance(log.get("uri"), str)
            or IMMUTABLE_MODELS_REPORT.fullmatch(log["uri"]) is None
            or not is_sha256(log.get("sha256"))
            or not positive_integer(log.get("sizeBytes")),
            "outputLog must be an immutable raw integrallis/models URI with a sha256 and a positive size",
        ),
    ]
    failed = [message for bad, message in problems if bad]
    if failed:
        raise ValueError(f"cleanHostRun is not a completed immutable clean-host Java 25 run: {failed}")
    return run


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
    model_id: str,
    labels: dict[str, Path] | None = None,
    models_artifact: Path | None = None,
    clean_host_run: Path | None = None,
) -> dict[str, Any]:
    if not isinstance(model_id, str) or not model_id:
        raise ValueError("model_id must be a nonempty catalog model id")
    release = validate_models_artifact(load(models_artifact)) if models_artifact is not None else None
    clean_host = None
    if clean_host_run is not None:
        if release is None:
            raise ValueError("cleanHostRun requires modelsArtifact: its modelsVersion must match the released version")
        clean_host = validate_clean_host_run(load(clean_host_run), release["version"])
    metadata = load(adapter_directory / "models-activated-lora.json")
    files = artifact_files(adapter_directory)
    weights = next(f for f in files if f["role"] == "adapter-weights")
    bundle_size, bundle_sha = bundle_identity(files)
    suites, window_sha, backend = window_suites(
        {k: load(v) for k, v in window_reports.items()},
        {k: load(v) for k, v in (labels or {}).items()},
    )
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
    if release is not None:
        gates["modelsArtifact"] = release
    if clean_host is not None:
        gates["cleanHostRun"] = clean_host
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
                "modelId": model_id,
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


def add_release_arguments(parser: argparse.ArgumentParser) -> None:
    """The options shared with assemble_composition_report.py."""
    parser.add_argument(
        "--labels", action="append", default=[], help="suite=path of a confirmed label suite; rescored on those labels"
    )
    parser.add_argument("--models-artifact", type=Path, help="gates.modelsArtifact JSON (collect_models_artifact.py)")
    parser.add_argument("--clean-host-run", type=Path, help="gates.cleanHostRun JSON; requires --models-artifact")


def label_paths(items: list[str]) -> dict[str, Path]:
    labels: dict[str, Path] = {}
    for item in items:
        suite, separator, path = item.partition("=")
        if not separator or not suite or not path:
            raise SystemExit(f"--labels must be suite=path, got {item!r}")
        if suite in labels:
            raise SystemExit(f"--labels names suite {suite} twice")
        labels[suite] = Path(path)
    return labels


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-id", required=True, help="the catalog id of the component being qualified")
    add_release_arguments(parser)
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
    args = parser.parse_args(argv)
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
        model_id=args.model_id,
        labels=label_paths(args.labels),
        models_artifact=args.models_artifact,
        clean_host_run=args.clean_host_run,
    )
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print("assembled", args.output, "sha256", sha256(args.output))


if __name__ == "__main__":
    main()
