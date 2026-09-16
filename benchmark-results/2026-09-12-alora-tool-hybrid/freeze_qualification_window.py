#!/usr/bin/env python3
"""Freeze an unseen BFCL qualification window and prove its preparation exclusions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from evaluate_alora import (
    BFCL_FILES,
    BFCL_REPOSITORY,
    BFCL_REVISION,
    SEED,
    read_jsonl,
    select_cases,
    sha256,
    verify_bfcl_data,
)
from prepare_tool_data import canonical_json, query_fingerprint


QUESTION_FILES = {
    name: digest
    for name, digest in BFCL_FILES.items()
    if name.startswith("BFCL_v3_")
}


def _case_queries(case: dict[str, Any]) -> set[str]:
    return {
        query_fingerprint(message["content"])
        for turn in case["question"]
        for message in turn
        if message.get("role") == "user" and isinstance(message.get("content"), str)
    }


def assert_windows_disjoint(
    exposed: list[dict[str, Any]], candidate: list[dict[str, Any]]
) -> None:
    exposed_ids = {case["id"] for case in exposed}
    candidate_ids = {case["id"] for case in candidate}
    if exposed_ids & candidate_ids:
        raise ValueError("candidate window has an ID overlap with the exposed window")
    exposed_queries = set().union(*(_case_queries(case) for case in exposed))
    candidate_queries = set().union(*(_case_queries(case) for case in candidate))
    if exposed_queries & candidate_queries:
        raise ValueError("candidate window has a query overlap with the exposed window")


def select_unseen_cases(
    cases: list[dict[str, Any]], count: int, seed: int, exposed_count: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if exposed_count <= 0 or count <= 0 or exposed_count + count > len(cases):
        raise ValueError("exposed and candidate counts must fit within the source cases")
    ordered = select_cases(cases, len(cases), seed)
    exposed = ordered[:exposed_count]
    exposed_queries = set().union(*(_case_queries(case) for case in exposed))
    selected: list[dict[str, Any]] = []
    selected_queries: set[str] = set()
    for case in ordered[exposed_count:]:
        queries = _case_queries(case)
        if not queries:
            raise ValueError(f"case contains no user query: {case.get('id')}")
        if queries & (exposed_queries | selected_queries):
            continue
        selected.append(case)
        selected_queries.update(queries)
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError(
            f"only {len(selected)} query-disjoint cases remain; requested {count}"
        )
    assert_windows_disjoint(exposed, selected)
    return exposed, selected


def verify_prepared_exclusions(
    manifests: list[Path], expected_sources: dict[str, str]
) -> list[dict[str, Any]]:
    verified = []
    for manifest_path in manifests:
        try:
            manifest = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"cannot read preparation manifest {manifest_path}: {error}") from error
        if not isinstance(manifest, dict) or manifest.get("schemaVersion") != 2:
            raise ValueError(f"preparation manifest must use schemaVersion 2: {manifest_path}")
        declared = {
            entry.get("path"): entry.get("sha256")
            for entry in manifest.get("excludedEvaluationSources", [])
            if isinstance(entry, dict)
        }
        if any(declared.get(name) != digest for name, digest in expected_sources.items()):
            raise ValueError(
                f"preparation manifest does not bind every qualification source: {manifest_path}"
            )
        verified.append(
            {
                "path": str(manifest_path),
                "sha256": sha256(manifest_path),
                "excludedEvaluationSources": [
                    {"path": name, "sha256": digest}
                    for name, digest in sorted(expected_sources.items())
                ],
            }
        )
    return verified


def freeze(
    *,
    data: Path,
    output: Path,
    count: int,
    exposed_count: int,
    prepared_manifests: list[Path],
) -> dict[str, Any]:
    if output.exists():
        raise ValueError(f"output already exists: {output}")
    if count <= 0 or exposed_count <= 0:
        raise ValueError("counts must be positive")
    verify_bfcl_data(data)
    cases: dict[str, list[dict[str, Any]]] = {}
    ordered_by_kind = {
        kind: select_cases(
            read_jsonl(data / f"BFCL_v3_{kind}.json"),
            len(read_jsonl(data / f"BFCL_v3_{kind}.json")),
            SEED,
        )
        for kind in ("simple", "multiple", "irrelevance")
    }
    exposed = [
        case
        for kind in ("simple", "multiple", "irrelevance")
        for case in ordered_by_kind[kind][:exposed_count]
    ]
    blocked_queries = set().union(*(_case_queries(case) for case in exposed))
    candidate: list[dict[str, Any]] = []
    for kind in ("simple", "multiple", "irrelevance"):
        selected = []
        for case in ordered_by_kind[kind][exposed_count:]:
            queries = _case_queries(case)
            if not queries:
                raise ValueError(f"case contains no user query: {case.get('id')}")
            if queries & blocked_queries:
                continue
            selected.append(case)
            blocked_queries.update(queries)
            if len(selected) == count:
                break
        if len(selected) != count:
            raise ValueError(
                f"only {len(selected)} globally query-disjoint {kind} cases remain; "
                f"requested {count}"
            )
        candidate.extend(selected)
        cases[kind] = [
            {
                "id": case["id"],
                "querySha256": sorted(_case_queries(case)),
            }
            for case in selected
        ]
    assert_windows_disjoint(exposed, candidate)
    prepared = verify_prepared_exclusions(prepared_manifests, QUESTION_FILES)
    case_set_sha256 = hashlib.sha256(canonical_json(cases).encode()).hexdigest()
    manifest = {
        "schemaVersion": 1,
        "status": "frozen-before-training",
        "source": {
            "repository": BFCL_REPOSITORY,
            "revision": BFCL_REVISION,
            "files": [
                {"path": name, "sha256": digest}
                for name, digest in sorted(BFCL_FILES.items())
            ],
        },
        "selection": {
            "seed": SEED,
            "perKind": count,
            "previouslyExposedPerKind": exposed_count,
            "method": (
                "lowest seed:id SHA-256 after excluding every query fingerprint in the "
                "previously exposed window and duplicate query fingerprints"
            ),
            "caseSetSha256": case_set_sha256,
        },
        "preparedExclusionProof": prepared,
        "cases": cases,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--count", required=True, type=int)
    parser.add_argument("--exposed-count", required=True, type=int)
    parser.add_argument("--prepared-manifest", required=True, type=Path, action="append")
    args = parser.parse_args()
    result = freeze(
        data=args.data,
        output=args.out,
        count=args.count,
        exposed_count=args.exposed_count,
        prepared_manifests=args.prepared_manifest,
    )
    print(canonical_json(result))


if __name__ == "__main__":
    main()
