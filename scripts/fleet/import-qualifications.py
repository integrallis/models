#!/usr/bin/env python3
"""Turn QUALIFIED fleet verdicts into committed evidence and catalog entries.

Reads a directory of fleet results (``<id>.json``, ``<id>.comparator.json``, ``<id>.verdict.json``),
keeps only the QUALIFIED ones, copies the three files into
``benchmark-results/certified-<date>/rag/<slug>/`` in the models repository, and appends a catalog entry
to ``catalog/qualifications.json`` in the ModelJars repository.

Refuses rather than guesses. A model already carrying a qualification is skipped, not overwritten; a
report whose ``artifactSha256`` or ``artifactSizeBytes`` disagrees with the catalogued artifact is an
error, because evidence gathered against a lookalike build is not evidence about the published model;
and the qualified count is asserted not to fall, since the catalog cannot shrink.

The tests still have to be written by hand: every certified model gets a case in
CertifiedRagEvidenceTest that re-derives its verdict from the committed reports, and the aggregate count
pinned in ModelRagQualificationRegistryTest has to move with the catalog.
"""

import argparse
import collections
import datetime
import hashlib
import json
import pathlib
import shutil
import sys


def read(path):
    return json.loads(path.read_bytes())


def entry_for(report_path, report, verdict, catalogued, repo_relative):
    summary = report["summary"]
    settings = report["settings"]
    return {
        "modelId": report["modelId"],
        "model": catalogued["name"],
        "backend": report["backend"],
        "backendVersion": report["backendVersion"],
        "workload": settings["workload"],
        "corpusSha256": settings["corpusSha256"],
        "promptTemplate": settings["promptTemplate"],
        "groundingPolicy": settings["groundingPolicy"],
        "artifactSha256": report["artifactSha256"],
        "artifactSizeBytes": report["artifactSizeBytes"],
        "reportSha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
        "performanceTier": report["performanceTier"],
        "verdict": verdict["verdict"],
        "qualified": True,
        "attempts": summary["totalAttempts"],
        "p95RetrievalMillis": summary["retrievalMillis"]["p95"],
        "p95TtftMillis": summary["ttftMillis"]["p95"],
        "p95TpotMillis": summary["tpotMillis"]["p95"],
        "p95EndToEndMillis": summary["endToEndMillis"]["p95"],
        "p50PrefillTokensPerSecond": summary["p50PrefillTokensPerSecond"],
        "p50DecodeTokensPerSecond": summary["p50DecodeTokensPerSecond"],
        "peakRssBytes": summary["peakRssBytes"],
        "correctAnswerRate": summary["correctAnswerRate"],
        "rawCorrectAnswerRate": summary["rawCorrectAnswerRate"],
        "abstentionAccuracy": summary["abstentionAccuracy"],
        "modelAnswerRate": summary["modelAnswerRate"],
        "modelAnswerCorrectRate": summary["modelAnswerCorrectRate"],
        "extractiveFallbackRate": summary["extractiveFallbackRate"],
        "environment": report["environment"],
        "report": repo_relative,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=pathlib.Path, help="directory of fleet result JSON")
    parser.add_argument("--models-repo", type=pathlib.Path, required=True)
    parser.add_argument("--modeljars-repo", type=pathlib.Path, required=True)
    parser.add_argument("--date", default=datetime.date.today().isoformat().replace("-", ""))
    parser.add_argument("--apply", action="store_true", help="write; otherwise report only")
    args = parser.parse_args()

    catalog = args.modeljars_repo / "catalog"
    catalog_dir = catalog
    models = {m["id"]: m for m in read(catalog / "models.json")["models"]}
    qualifications = read(catalog / "qualifications.json")
    already = {e["modelId"] for e in qualifications["entries"]}
    allowed = list(next(e for e in qualifications["entries"] if e.get("qualified")).keys())

    planned, skipped = [], []
    for verdict_path in sorted(args.results.glob("*.verdict.json")):
        model_id = verdict_path.name[: -len(".verdict.json")]
        verdict = read(verdict_path)["qualification"]
        if verdict["verdict"] != "QUALIFIED":
            skipped.append((model_id, verdict["verdict"]))
            continue
        if model_id in already:
            skipped.append((model_id, "already qualified"))
            continue
        if model_id not in models:
            raise SystemExit(f"{model_id} is not a catalogued candidate; add it to models.json first")
        candidate_path = args.results / f"{model_id}.json"
        comparator_path = args.results / f"{model_id}.comparator.json"
        for required in (candidate_path, comparator_path):
            if not required.is_file():
                raise SystemExit(f"{model_id}: missing {required.name}")
        report = read(candidate_path)
        catalogued = models[model_id]
        if report["artifactSha256"] != catalogued["sha256"]:
            raise SystemExit(f"{model_id}: report artifact differs from the catalogued artifact")
        if report["artifactSizeBytes"] != catalogued["sizeBytes"]:
            raise SystemExit(f"{model_id}: report artifact size differs from the catalogued size")
        planned.append((model_id, candidate_path, comparator_path, verdict_path, report, verdict, catalogued))

    for model_id, reason in skipped:
        print(f"  skip {model_id}: {reason}")
    if not planned:
        print("nothing to import")
        return

    added = []
    for model_id, candidate_path, comparator_path, verdict_path, report, verdict, catalogued in planned:
        slug = catalogued["markerCoordinate"].split(":")[1].replace(".", "-")
        destination = args.models_repo / f"benchmark-results/certified-{args.date}/rag/{slug}"
        relative = f"benchmark-results/certified-{args.date}/rag/{slug}/{slug}-rust-ffm-grounded.json"
        print(f"  + {model_id} -> {relative}  decode={report['summary']['p50DecodeTokensPerSecond']:.2f} tok/s")
        if args.apply:
            destination.mkdir(parents=True, exist_ok=True)
            shutil.copy(candidate_path, destination / f"{slug}-rust-ffm-grounded.json")
            shutil.copy(comparator_path, destination / f"{slug}-ollama-grounded.json")
            shutil.copy(verdict_path, destination / "qualification.json")
        written = destination / f"{slug}-rust-ffm-grounded.json"
        entry = entry_for(written if args.apply else candidate_path, report, verdict, catalogued, relative)
        drift = set(entry) ^ set(allowed)
        if drift:
            raise SystemExit(f"{model_id}: catalog entry schema drift: {sorted(drift)}")
        added.append(entry)

    if not args.apply:
        print(f"\n{len(added)} importable; re-run with --apply to write")
        return

    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    before = sum(1 for e in qualifications["entries"] if e.get("qualified"))
    qualifications["entries"].extend(added)
    after = sum(1 for e in qualifications["entries"] if e.get("qualified"))
    if after < before:
        raise SystemExit("the catalog must not shrink")
    qualifications["qualifiedModels"] = after
    qualifications["generatedAt"] = stamp

    models_document = read(catalog / "models.json")
    ids = {e["modelId"] for e in added}
    for model in models_document["models"]:
        if model["id"] not in ids:
            continue
        # A declared capability is a qualified one: the build requires the tool-calling flag to agree with
        # membership of the tool qualification set, and a RAG run says nothing about tool use. Publishing
        # a model that claims it unmeasured is what that gate exists to stop, so the claim is dropped
        # until a tool-calling qualification exists for it.
        tools = catalog_dir / "tool-qualifications.json"
        tool_qualified = set()
        if tools.is_file():
            tool_qualified = {
                e["modelId"] for e in json.loads(tools.read_text()).get("entries", [])
                if e.get("qualified")
            }
        caps = model.get("capabilities") or []
        if "tool-calling" in caps and model["id"] not in tool_qualified:
            model["capabilities"] = [c for c in caps if c != "tool-calling"]
            print(f"  dropped unqualified tool-calling claim from {model['id']}")
        backends = dict(model["backends"])
        backends["rust-ffm"] = True          # measured here
        order = ["pure-java", "rust-ffm", "llama.cpp"]
        model["backends"] = collections.OrderedDict(
            [(k, backends[k]) for k in order if k in backends]
            + [(k, v) for k, v in backends.items() if k not in order]
        )
        model.setdefault("catalogPublishedAt", stamp)

    (catalog / "qualifications.json").write_text(json.dumps(qualifications, indent=2) + "\n")
    (catalog / "models.json").write_text(json.dumps(models_document, indent=2) + "\n")
    print(f"\nqualifiedModels {before} -> {after}")
    print("remember: add a CertifiedRagEvidenceTest case per model, and move the pinned")
    print("AGGREGATE_QUALIFIED_MODELS in ModelRagQualificationRegistryTest to", after)


if __name__ == "__main__":
    sys.exit(main())
