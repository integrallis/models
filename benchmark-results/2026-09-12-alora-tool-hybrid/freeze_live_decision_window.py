#!/usr/bin/env python3
"""Freeze a pinned, contamination-checked BFCL live window for Java Q4 screening."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from evaluate_alora import BFCL_FILES, BFCL_REPOSITORY, BFCL_REVISION, read_jsonl
from evaluate_live_development import (
    DEVELOPMENT_SEED,
    LIVE_FILES,
    assert_disjoint,
    load_live_slice,
)
from prepare_tool_data import (
    HAMMER_IRRELEVANCE_REPOSITORY,
    HAMMER_IRRELEVANCE_REVISION,
    HAMMER_IRRELEVANCE_SHA256,
    TRAINING_REPOSITORY,
    TRAINING_REVISION,
    TRAINING_SHA256,
    canonical_json,
    query_fingerprint,
    sha256,
    write_jsonl,
)


KINDS = ("simple", "multiple", "irrelevance")


def decision_records(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "phase": "screen",
            "id": case["id"],
            "kind": case["kind"],
            "messages": case["messages"],
            "tools": case["tools"],
            "expected": case["expected"],
        }
        for case in sorted(cases, key=lambda item: (item["kind"], item["id"]))
    ]


def assert_unique_cases(cases: list[dict[str, Any]]) -> None:
    seen: dict[str, str] = {}
    for case in cases:
        fingerprint = hashlib.sha256(
            canonical_json(
                {"messages": case["messages"], "tools": case["tools"]}
            ).encode()
        ).hexdigest()
        previous = seen.setdefault(fingerprint, case["id"])
        if previous != case["id"]:
            raise ValueError(
                "live window contains duplicate messages and tools in "
                f"{previous} and {case['id']}"
            )


def assert_disjoint_from_training_sources(
    development: list[dict[str, Any]], training: Path, irrelevance: Path
) -> None:
    development_queries = {
        query_fingerprint(message["content"])
        for case in development
        for message in case["messages"]
        if message.get("role") == "user"
    }
    source_queries: set[str] = set()
    for row in read_jsonl(training):
        source_queries.update(
            query_fingerprint(message["content"])
            for message in row.get("messages", [])
            if message.get("role") == "user" and isinstance(message.get("content"), str)
        )
    for row in json.loads(irrelevance.read_text()):
        if isinstance(row.get("query"), str):
            source_queries.add(query_fingerprint(row["query"]))
    overlap = development_queries & source_queries
    if overlap:
        raise ValueError(
            f"live window overlaps complete training sources in {len(overlap)} queries"
        )


def static_cases(data: Path) -> list[dict[str, Any]]:
    return [
        {
            "id": row["id"],
            "messages": [message for turn in row["question"] for message in turn],
        }
        for kind in KINDS
        for row in read_jsonl(data / f"BFCL_v3_{kind}.json")
    ]


def verify_files(root: Path, expected: dict[str, str]) -> None:
    for relative, expected_hash in expected.items():
        actual = sha256(root / relative)
        if actual != expected_hash:
            raise ValueError(
                f"{relative} SHA-256 differs: expected {expected_hash}, got {actual}"
            )


def freeze(
    *,
    live_data: Path,
    static_data: Path,
    training: Path,
    irrelevance: Path,
    output: Path,
    count: int,
    seed: int,
) -> dict[str, Any]:
    if output.exists():
        raise ValueError(f"output already exists: {output}")
    if count <= 0:
        raise ValueError("count must be positive")
    verify_files(live_data, LIVE_FILES)
    verify_files(static_data, BFCL_FILES)
    verify_files(training.parent, {training.name: TRAINING_SHA256})
    verify_files(irrelevance.parent, {irrelevance.name: HAMMER_IRRELEVANCE_SHA256})
    cases = [
        case
        for kind in KINDS
        for case in load_live_slice(live_data, kind, count, seed)
    ]
    assert_unique_cases(cases)
    assert_disjoint(cases, static_cases(static_data))
    assert_disjoint_from_training_sources(cases, training, irrelevance)

    records = decision_records(cases)
    identities = [
        {
            "id": case["id"],
            "kind": case["kind"],
            "querySha256": sorted(
                query_fingerprint(message["content"])
                for message in case["messages"]
                if message.get("role") == "user"
            ),
        }
        for case in records
    ]
    output.mkdir(parents=True)
    records_path = output / "records.jsonl"
    write_jsonl(records_path, records)
    manifest = {
        "schemaVersion": 1,
        "status": "frozen-before-v18-scoring",
        "purpose": "production Java Q4 applicability screen; never training or calibration",
        "selection": {
            "seed": seed,
            "perKind": count,
            "caseSetSha256": hashlib.sha256(
                canonical_json(identities).encode()
            ).hexdigest(),
            "cases": identities,
        },
        "source": {
            "repository": BFCL_REPOSITORY,
            "revision": BFCL_REVISION,
            "files": [
                {"path": path, "sha256": digest}
                for path, digest in sorted(LIVE_FILES.items())
            ],
        },
        "staticEvaluationOverlapQueries": 0,
        "completeTrainingSourceOverlapQueries": 0,
        "staticEvaluationFiles": [
            {"path": path, "sha256": digest}
            for path, digest in sorted(BFCL_FILES.items())
        ],
        "trainingSources": [
            {
                "repository": TRAINING_REPOSITORY,
                "revision": TRAINING_REVISION,
                "path": training.name,
                "sha256": TRAINING_SHA256,
            },
            {
                "repository": HAMMER_IRRELEVANCE_REPOSITORY,
                "revision": HAMMER_IRRELEVANCE_REVISION,
                "path": irrelevance.name,
                "sha256": HAMMER_IRRELEVANCE_SHA256,
            },
        ],
        "records": {"path": records_path.name, "sha256": sha256(records_path)},
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live-data", required=True, type=Path)
    parser.add_argument("--static-data", required=True, type=Path)
    parser.add_argument("--training", required=True, type=Path)
    parser.add_argument("--irrelevance", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--count", type=int, default=25)
    parser.add_argument("--seed", type=int, default=DEVELOPMENT_SEED)
    args = parser.parse_args()
    manifest = freeze(
        live_data=args.live_data,
        static_data=args.static_data,
        training=args.training,
        irrelevance=args.irrelevance,
        output=args.out,
        count=args.count,
        seed=args.seed,
    )
    print(canonical_json(manifest))


if __name__ == "__main__":
    main()
