#!/usr/bin/env python3
"""Prepare disjoint hard no-call training rows from a pinned BFCL live source."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from prepare_tool_data import (
    NO_TOOL_SENTINEL,
    bfcl_query_fingerprints,
    canonical_json,
    normalize_tools,
    query_fingerprint,
    record_fingerprint,
    sha256,
    write_jsonl,
)


SOURCE_REPOSITORY = "ShishirPatil/gorilla"
SOURCE_REVISION = "c15b2a151662cac9839c96d7dfb1493b5329c975"
SOURCE_SHA256 = "0d259e1c3ab6ba06c2a51911e2f49ecaa2eac3723a4a14f183a7687955e61338"


def prepare(
    *,
    source: Path,
    excluded_paths: list[Path],
    output: Path,
    train_count: int,
    validation_count: int,
    seed: int,
    expected_source_sha256: str,
) -> dict[str, Any]:
    if output.exists():
        raise ValueError(f"output already exists: {output}")
    if train_count <= 0 or validation_count <= 0:
        raise ValueError("train and validation counts must be positive")
    actual_source_sha256 = sha256(source)
    if actual_source_sha256 != expected_source_sha256:
        raise ValueError(
            "source SHA-256 differs: "
            f"expected {expected_source_sha256}, got {actual_source_sha256}"
        )
    excluded_queries = bfcl_query_fingerprints(excluded_paths)
    statistics = {
        "read": 0,
        "invalid": 0,
        "multiMessage": 0,
        "noTools": 0,
        "overlap": 0,
        "duplicate": 0,
    }
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    with source.open() as lines:
        for source_line, line in enumerate(lines, start=1):
            statistics["read"] += 1
            try:
                row = json.loads(line)
                turns = row["question"]
                if (
                    len(turns) != 1
                    or len(turns[0]) != 1
                    or turns[0][0].get("role") != "user"
                    or not isinstance(turns[0][0].get("content"), str)
                ):
                    statistics["multiMessage"] += 1
                    continue
                if not row["function"]:
                    statistics["noTools"] += 1
                    continue
                user = turns[0][0]["content"]
                if query_fingerprint(user) in excluded_queries:
                    statistics["overlap"] += 1
                    continue
                record = {
                    "sourceLine": source_line,
                    "user": user,
                    "tools": normalize_tools(
                        [
                            {
                                "type": "function",
                                "function": {
                                    "name": function["name"],
                                    "description": function.get("description", ""),
                                    "parameters": function.get(
                                        "parameters",
                                        {"type": "object", "properties": {}},
                                    ),
                                },
                            }
                            for function in row["function"]
                        ]
                    ),
                    "calls": [],
                    "assistant": NO_TOOL_SENTINEL,
                }
                fingerprint = record_fingerprint(record)
                if fingerprint in seen:
                    statistics["duplicate"] += 1
                    continue
                seen.add(fingerprint)
                records.append(record)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                statistics["invalid"] += 1
    required = train_count + validation_count
    if len(records) < required:
        raise ValueError(f"source supplied {len(records)} valid disjoint rows; need {required}")
    records.sort(
        key=lambda record: hashlib.sha256(
            f"{seed}:live-irrelevance:{record_fingerprint(record)}".encode()
        ).digest()
    )
    selected = records[:required]
    train = selected[:train_count]
    validation = selected[train_count:]
    output.mkdir(parents=True)
    train_path = output / "train.jsonl"
    validation_path = output / "validation.jsonl"
    write_jsonl(train_path, train)
    write_jsonl(validation_path, validation)
    manifest = {
        "schemaVersion": 2,
        "seed": seed,
        "noCallFraction": 1.0,
        "source": {
            "repository": SOURCE_REPOSITORY,
            "revision": SOURCE_REVISION,
            "path": source.name,
            "sha256": actual_source_sha256,
            "role": "training only; not used for qualification reporting",
        },
        "excludedEvaluationSources": [
            {"path": path.name, "sha256": sha256(path)} for path in excluded_paths
        ],
        "excludedQueryCount": len(excluded_queries),
        "selection": "lowest SHA-256 of seed:live-irrelevance:record-fingerprint",
        "statistics": {**statistics, "eligible": len(records), "selected": len(selected)},
        "train": {
            "count": len(train),
            "noCallCount": len(train),
            "sha256": sha256(train_path),
            "sourceLines": [record["sourceLine"] for record in train],
        },
        "validation": {
            "count": len(validation),
            "noCallCount": len(validation),
            "sha256": sha256(validation_path),
            "sourceLines": [record["sourceLine"] for record in validation],
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--bfcl", required=True, type=Path, nargs="+")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--train-count", required=True, type=int)
    parser.add_argument("--validation-count", required=True, type=int)
    parser.add_argument("--seed", type=int, default=20_260_915)
    args = parser.parse_args()
    manifest = prepare(
        source=args.source,
        excluded_paths=args.bfcl,
        output=args.out,
        train_count=args.train_count,
        validation_count=args.validation_count,
        seed=args.seed,
        expected_source_sha256=SOURCE_SHA256,
    )
    print(canonical_json(manifest))


if __name__ == "__main__":
    main()
