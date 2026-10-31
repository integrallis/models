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


def default_configuration_smoke(report_path, repo_relative):
    """The library-default correctness smoke beside the tuned report, or None.

    ModelJars' tools/qualification-smoke-gate.mjs requires this on every new or changed
    entry, and refuses the publish without it: a tuned performance comparison says nothing
    about what a caller gets from the public library defaults. It is written by
    run-controlled-rag-qualification.sh, or for a fleet run by
    scripts/fleet/collect-default-smoke.sh -- the two-arm worker does not produce one, which
    is why eleven models measured on 2026-10-29 could not be published until it was
    collected separately.

    The sidecar is used as written rather than rebuilt here: it was produced next to the run
    it describes, and its reportSha256 is checked against the file it names.

    One field is rewritten. The sidecar's `report` is relative to the qualification directory,
    because run-controlled-rag-qualification.sh writes it beside its own output. The catalog
    stores a repo-relative path, and ModelJars' gate fetches it from the models repository at
    `modelsRevision` -- so the bare relative form resolves against the repository root and
    404s: "Could not fetch .../default-correctness/models-rust-ffm.json: HTTP 404".
    """
    sidecar = report_path.parent / "default-correctness" / "smoke.json"
    measured = report_path.parent / "default-correctness" / "models-rust-ffm.json"
    if not sidecar.exists() or not measured.exists():
        return None
    smoke = json.loads(sidecar.read_text())
    digest = hashlib.sha256(measured.read_bytes()).hexdigest()
    if smoke.get("reportSha256") != digest:
        raise SystemExit(
            f"{sidecar}: reportSha256 {smoke.get('reportSha256')} does not match "
            f"{measured} ({digest}); the smoke does not describe that file"
        )
    return {
        "configuration": smoke["configuration"],
        "backend": smoke["backend"],
        "artifactSha256": smoke["artifactSha256"],
        "report": str(pathlib.PurePosixPath(repo_relative).parent / smoke["report"]),
        "reportSha256": smoke["reportSha256"],
        "totalAttempts": smoke["totalAttempts"],
        "successfulAttempts": smoke["successfulAttempts"],
        "tuningSystemProperties": smoke["tuningSystemProperties"],
    }


def entry_for(report_path, report, verdict, catalogued, repo_relative):
    summary = report["summary"]
    settings = report["settings"]
    entry = {
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
    smoke = default_configuration_smoke(report_path, repo_relative)
    if smoke is not None:
        entry["defaultConfigurationSmoke"] = smoke
    return entry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=pathlib.Path, help="directory of fleet result JSON")
    parser.add_argument("--models-repo", type=pathlib.Path, required=True)
    parser.add_argument("--modeljars-repo", type=pathlib.Path, required=True)
    parser.add_argument("--date", default=datetime.date.today().isoformat().replace("-", ""))
    parser.add_argument("--apply", action="store_true", help="write; otherwise report only")
    parser.add_argument(
        "--models-revision",
        help="models commit holding the copied evidence, recorded per entry. Required for a "
        "publisher-reference import, because ModelJars fetches the receipt and the arm reports "
        "from the models repository at that revision. It goes on the ENTRY, never on the "
        "document: moving the document revision revalidates every entry, and the grandfathered "
        "ones predate defaultConfigurationSmoke, so they fail a rule they were never published "
        "under. That mistake rejected an untouched entry once already.",
    )
    args = parser.parse_args()

    catalog = args.modeljars_repo / "catalog"
    catalog_dir = catalog
    models = {m["id"]: m for m in read(catalog / "models.json")["models"]}
    qualifications = read(catalog / "qualifications.json")
    already = {e["modelId"] for e in qualifications["entries"]}
    # Sampling one entry cannot express an optional key, and entries legitimately differ:
    # defaultConfigurationSmoke is carried by the ones measured since the smoke gate landed
    # and absent from the ones that predate it. Required is what every qualified entry has;
    # known is what any of them has. A key outside `known` is drift; a missing required key
    # is drift. An optional key is neither.
    qualified_entries = [e for e in qualifications["entries"] if e.get("qualified")]
    known = set().union(*(set(e) for e in qualified_entries))
    # Legitimate on a publisher-reference entry and absent from every other, so they are optional
    # rather than drift. Listed explicitly: the guard's whole job is to refuse a key nobody
    # intended, and that only works if intent is written down.
    known |= {"policyVersion", "qualificationEvidence", "contextLength", "modelsRevision"}
    # Named required_keys: line 148 reuses `required` as a loop variable over paths.
    required_keys = set(qualified_entries[0]).intersection(*(set(e) for e in qualified_entries))

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
        # Named by the comparator's actual backend. This was hard-coded "ollama", which filed a
        # patched-llama.cpp publisher comparator under a backend it was not, in the one place a
        # reader goes to see which implementation the ratio was measured against.
        comparator_backend = read(comparator_path)["backend"].replace(".", "-")
        comparator_name = f"{slug}-{comparator_backend}-grounded.json"
        receipt_relative = (
            f"benchmark-results/certified-{args.date}/rag/{slug}/qualification.json"
        )
        if args.apply:
            destination.mkdir(parents=True, exist_ok=True)
            # The library-defaults smoke travels with the run it describes. entry_for() resolves
            # the sidecar beside the report it is given, which under --apply is the destination,
            # so a smoke left in the results directory is silently not found and the entry lands
            # without it -- and ModelJars then refuses the publish for exactly that omission.
            smoke_source = candidate_path.parent / "default-correctness"
            if smoke_source.is_dir():
                shutil.copytree(
                    smoke_source, destination / "default-correctness", dirs_exist_ok=True
                )
            shutil.copy(candidate_path, destination / f"{slug}-rust-ffm-grounded.json")
            shutil.copy(comparator_path, destination / comparator_name)
            # The receipt the ModelJars publisher gate verifies, with the arm paths rewritten from
            # wherever the run wrote them to where this repository stores them. The gate requires
            # candidate.path to equal the catalog's report path exactly, so an absolute /root path
            # from a bench host fails it.
            receipt = read(verdict_path)
            receipt["candidate"]["path"] = relative
            receipt["comparators"][0]["path"] = (
                f"benchmark-results/certified-{args.date}/rag/{slug}/{comparator_name}"
            )
            (destination / "qualification.json").write_text(
                json.dumps(receipt, indent=2) + "\n"
            )
        written = destination / f"{slug}-rust-ffm-grounded.json"
        entry = entry_for(written if args.apply else candidate_path, report, verdict, catalogued, relative)
        # A publisher-reference verdict has to say so in the catalog. Without policyVersion the
        # ModelJars gate that binds the artifact to its reviewed reference returns early and never
        # runs, so the one route that admits a non-stock comparator would go unverified.
        policy_id = read(verdict_path).get("policyId")
        if policy_id == "production-rag-publisher-reference-v1":
            entry["policyVersion"] = policy_id
            # The gate requires the catalogued context to equal the context the run actually
            # used, so that a ratio measured at one window cannot be published as another.
            entry["contextLength"] = report["settings"]["contextLength"]
            if args.models_revision:
                entry["modelsRevision"] = args.models_revision
            elif args.apply:
                raise SystemExit(
                    "a publisher-reference import needs --models-revision: the gate fetches the "
                    "receipt and both arm reports from the models repository at that commit"
                )
            if args.apply:
                entry["qualificationEvidence"] = {
                    "report": receipt_relative,
                    "reportSha256": hashlib.sha256(
                        (destination / "qualification.json").read_bytes()
                    ).hexdigest(),
                }
        drift = (set(entry) - known) | (required_keys - set(entry))
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
