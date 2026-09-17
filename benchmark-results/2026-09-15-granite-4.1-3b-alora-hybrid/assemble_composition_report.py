#!/usr/bin/env python3
"""Assemble the ModelJars composition report for the hybrid (base + RAG specialist).

The composition report is the document ``tools/composition-evidence-gate.mjs`` reads for a
composition that declares ``specialistKind`` ``upstream-rag-specialist`` or
``first-party-rag-specialist`` (schemaVersion 2; the kind and its provenance options are those of
:mod:`assemble_component_report`). It is
built on top of the component report: the same raw window, identity, JUnit, long-context and
crossover evidence goes through :mod:`assemble_component_report` first (which refuses any failing
gate), and this module reshapes the result into the composition gate's field names and adds the
two things only a composition carries:

* the handoff economics at the 4,096-token tier (control = recomputed prefix, hybrid = physically
  shared prefix; medians and improvement copied from the crossover report), and
* the published member artifacts, read from a JSON file that a separate resolution step writes
  after actually resolving each Maven coordinate from Central and running it through the public
  API. This module refuses to mark a composition qualified unless every member artifact in that
  file says ``resolvedFromCentral`` and ``runViaPublicApi`` are true, so the report cannot claim a
  Central resolution nobody performed.

Nothing in the report is typed in by hand.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import assemble_component_report as component

FOUR_K = 4096
HANDOFF = "exact-kv-block-sharing"


def published_artifacts(path: Path, members: list[str]) -> list[dict[str, Any]]:
    """Reads the artifact resolution record and checks it covers every member with real evidence."""
    artifacts = component.load(path)
    if not isinstance(artifacts, list):
        raise ValueError(f"{path}: published artifacts must be a JSON list")
    seen: set[str] = set()
    for artifact in artifacts:
        for key in ("modelId", "coordinate", "sha256"):
            if not isinstance(artifact.get(key), str) or not artifact[key]:
                raise ValueError(f"{path}: artifact is missing {key}")
        if artifact["coordinate"].count(":") != 2:
            raise ValueError(f"{path}: {artifact['modelId']} coordinate must be group:artifact:version")
        if artifact.get("resolvedFromCentral") is not True or artifact.get("runViaPublicApi") is not True:
            raise SystemExit(
                f"refusing to assemble a composition report: {artifact['modelId']} was not resolved "
                "from Central and run through the public API"
            )
        seen.add(artifact["modelId"])
    if seen != set(members):
        raise ValueError(f"{path}: artifacts {sorted(seen)} do not match members {sorted(members)}")
    return artifacts


def four_k_summary(crossover: dict[str, Any]) -> dict[str, Any]:
    summaries = [s for s in crossover["summaries"] if s["prefixTokens"] == FOUR_K]
    if len(summaries) != 1:
        raise ValueError("crossover report must carry exactly one 4096-token tier")
    return summaries[0]


def assemble(
    *,
    composition_id: str,
    base_member_id: str,
    specialist_member_id: str,
    published_artifacts_path: Path,
    crossover_report: Path,
    **component_inputs: Any,
) -> dict[str, Any]:
    report = component.assemble(crossover_report=crossover_report, **component_inputs)
    gates = report["evaluation"]["gates"]
    crossover = component.load(crossover_report)
    tier = four_k_summary(crossover)
    performance = gates["performanceAndMemory"]
    long_context = gates["longContext"]
    improvement = tier["improvement"]
    if abs(improvement - performance["fourKImprovement"]) > 1e-12:
        raise ValueError("4K improvement differs between the tier summary and the verdict")
    artifacts = published_artifacts(published_artifacts_path, [base_member_id, specialist_member_id])
    composition_gates = {
        "provenance": gates["provenance"],
        "taskCorrectness": gates["taskCorrectness"],
        "kernelIdentity": gates["kernelIdentity"],
        "plainJava": gates["plainJava"],
        "jvmMechanics": gates["jvmMechanics"],
        "longContextRetrieval": {
            "pass": long_context["pass"],
            "cases": long_context["cases"],
            "contextTokens": long_context["prefixTokens"],
            "nativeCorrect": long_context["nativeCorrectCases"],
            "retainedNativeCorrect": long_context["retainedNativeCorrectCases"],
            "specialistCorrect": long_context["specialistCorrectCases"],
            "tokenExactCases": long_context["cases"] if long_context["exactBaseOutput"] else 0,
            "physicallySharedCases": long_context["cases"] if long_context["physicalStorageIdentity"] else 0,
            "policyVersion": long_context["policyVersion"],
        },
        "performanceCrossover": {
            "pass": performance["pass"],
            "contextTokens": FOUR_K,
            "improvement": improvement,
            "crossoverPrefixTokens": performance["crossoverPrefixTokens"],
            "prefixTiers": performance["prefixTiers"],
            "physicallySharedAllTiers": performance["physicalSharing"],
            "recomputedIndependentAllTiers": performance["recomputedIndependence"],
            "tokenExactAllTiers": performance["tokenExactAtAllTiers"],
            "policyVersion": performance["policyVersion"],
        },
        "memoryAccounting": {
            "pass": performance["pass"] and performance["memoryComplete"],
            "complete": performance["memoryComplete"],
            "peakRssBytes": performance["peakRssBytes"],
            "sharedUniqueStateBytes": tier["sharedMedianUniqueStateBytes"],
            "recomputedUniqueStateBytes": tier["recomputedMedianUniqueStateBytes"],
        },
    }
    # The composition gate does not read these today; they are carried so a composition report
    # binds the same released artifacts and clean-host run as its component report.
    for release_gate in ("modelsArtifact", "cleanHostRun"):
        if release_gate in gates:
            composition_gates[release_gate] = gates[release_gate]
    correctness = composition_gates["taskCorrectness"]["pass"] and composition_gates["longContextRetrieval"]["pass"]
    sharing = (
        composition_gates["performanceCrossover"]["pass"]
        and composition_gates["memoryAccounting"]["pass"]
        and composition_gates["memoryAccounting"]["sharedUniqueStateBytes"]
        < composition_gates["memoryAccounting"]["recomputedUniqueStateBytes"]
    )
    if not (correctness and sharing):
        raise SystemExit("refusing to assemble a composition report whose gates do not all pass")
    return {
        "schemaVersion": 2,
        "specialistKind": report["specialistKind"],
        "compositionId": composition_id,
        "implementation": {
            "runtime": "java",
            "stateHandoffMechanism": HANDOFF,
            "publicApiExercised": True,
            "externalInference": False,
            "modelsRevision": report["implementation"]["modelsRevision"],
        },
        "evaluation": {
            "controlMedianMillis": tier["recomputedMedianHandoffMillis"],
            "hybridMedianMillis": tier["sharedMedianHandoffMillis"],
            "improvement": improvement,
            "qualified": True,
            "correctnessPassed": True,
            "handoffCostIncluded": True,
            "members": [
                {"role": "base", "modelId": base_member_id},
                {"role": "answerability", "modelId": specialist_member_id},
            ],
            "publishedArtifacts": artifacts,
            "artifact": report["evaluation"]["artifact"],
            "base": report["evaluation"]["base"],
            "gates": composition_gates,
        },
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--composition-id", required=True)
    parser.add_argument("--base-member-id", required=True)
    parser.add_argument("--specialist-member-id", required=True)
    parser.add_argument("--model-id", help="evaluation.artifact.modelId; defaults to --specialist-member-id")
    component.add_release_arguments(parser)
    parser.add_argument("--published-artifacts", type=Path, required=True)
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
    window_reports = {}
    for spec in args.window:
        key, path = spec.split("=", 1)
        window_reports[key] = Path(path)
    identity_reports = {}
    for spec in args.identity:
        key, paths = spec.split("=", 1)
        java, kernel = paths.split(",", 1)
        identity_reports[key] = (Path(java), Path(kernel))
    report = assemble(
        composition_id=args.composition_id,
        base_member_id=args.base_member_id,
        specialist_member_id=args.specialist_member_id,
        published_artifacts_path=args.published_artifacts,
        adapter_directory=args.adapter_directory,
        base_model_id=args.base_model_id,
        base_revision=args.base_revision,
        base_artifact=args.base_artifact,
        window_reports=window_reports,
        identity_reports=identity_reports,
        junit_xml=args.junit_xml,
        long_context_report=args.long_context_report,
        crossover_report=args.crossover_report,
        models_revision=args.models_revision,
        model_id=args.model_id or args.specialist_member_id,
        labels=component.label_paths(args.labels),
        models_artifact=args.models_artifact,
        clean_host_run=args.clean_host_run,
        **component.specialist_options(args),
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    evaluation = report["evaluation"]
    print(
        f"wrote {args.output}: improvement {evaluation['improvement']:.3f} "
        f"(control {evaluation['controlMedianMillis']:.0f} ms, hybrid {evaluation['hybridMedianMillis']:.0f} ms)"
    )


if __name__ == "__main__":
    main()
