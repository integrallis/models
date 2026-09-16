#!/usr/bin/env python3
"""Merge verified positive-call and hard no-call preparations without split leakage."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from prepare_tool_data import canonical_json, record_fingerprint, sha256, write_jsonl


def _load_manifest(parent: Path, label: str) -> tuple[Path, dict[str, Any]]:
    path = parent / "manifest.json"
    try:
        manifest = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {label} parent manifest: {error}") from error
    if not isinstance(manifest, dict) or manifest.get("schemaVersion") != 2:
        raise ValueError(f"{label} parent manifest must use preparation schemaVersion 2")
    return path, manifest


def _load_split(
    parent: Path,
    manifest: dict[str, Any],
    label: str,
    split: str,
) -> list[dict[str, Any]]:
    path = parent / f"{split}.jsonl"
    try:
        expected = manifest[split]["sha256"]
    except (KeyError, TypeError) as error:
        raise ValueError(f"{label} parent manifest has no {split} split hash") from error
    try:
        actual = sha256(path)
    except OSError as error:
        raise ValueError(f"cannot read {label} {split} split: {error}") from error
    if actual != expected:
        raise ValueError(
            f"{label} {split} split SHA-256 differs: expected {expected}, got {actual}"
        )

    rows: list[dict[str, Any]] = []
    with path.open() as source:
        for line_number, line in enumerate(source, 1):
            try:
                record = json.loads(line)
                calls = record["calls"]
            except (json.JSONDecodeError, KeyError, TypeError) as error:
                raise ValueError(
                    f"{label} {split} record {line_number} is invalid: {error}"
                ) from error
            if not isinstance(record, dict) or not isinstance(calls, list):
                raise ValueError(f"{label} {split} record {line_number} is invalid")
            rows.append(record)
    return rows


def _priority(record: dict[str, Any], seed: int, role: str, split: str) -> bytes:
    return hashlib.sha256(
        f"{seed}:merge:{role}:{split}:{record_fingerprint(record)}".encode()
    ).digest()


def _select(
    rows: list[dict[str, Any]], count: int, seed: int, role: str, split: str
) -> list[dict[str, Any]]:
    if count < 0:
        raise ValueError("requested split counts cannot be negative")
    if len(rows) < count:
        raise ValueError(
            f"{role} {split} parent has {len(rows)} records but {count} were requested"
        )
    return sorted(rows, key=lambda row: _priority(row, seed, role, split))[:count]


def _mixed_order(rows: list[dict[str, Any]], seed: int, split: str) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: hashlib.sha256(
            f"{seed}:merge:output:{split}:{record_fingerprint(row)}".encode()
        ).digest(),
    )


def merge_prepared(
    *,
    positive: Path,
    negative: Path,
    output: Path,
    train_call_count: int,
    train_no_call_count: int,
    validation_call_count: int,
    validation_no_call_count: int,
    seed: int,
) -> dict[str, Any]:
    if output.exists():
        raise ValueError(f"output already exists: {output}")
    requested = (
        train_call_count,
        train_no_call_count,
        validation_call_count,
        validation_no_call_count,
    )
    if any(count < 0 for count in requested) or sum(requested) == 0:
        raise ValueError("requested split counts must be non-negative and not all zero")

    positive_manifest_path, positive_manifest = _load_manifest(positive, "positive")
    negative_manifest_path, negative_manifest = _load_manifest(negative, "negative")
    positive_splits = {
        split: _load_split(positive, positive_manifest, "positive", split)
        for split in ("train", "validation")
    }
    negative_splits = {
        split: _load_split(negative, negative_manifest, "negative", split)
        for split in ("train", "validation")
    }

    all_rows = [
        record
        for splits in (positive_splits, negative_splits)
        for rows in splits.values()
        for record in rows
    ]
    fingerprints = [record_fingerprint(record) for record in all_rows]
    if len(fingerprints) != len(set(fingerprints)):
        raise ValueError("parent preparations contain a duplicate record")

    positive_calls = {
        split: [record for record in rows if record["calls"]]
        for split, rows in positive_splits.items()
    }
    negative_no_calls = {
        split: [record for record in rows if not record["calls"]]
        for split, rows in negative_splits.items()
    }

    train = _mixed_order(
        _select(positive_calls["train"], train_call_count, seed, "positive", "train")
        + _select(negative_no_calls["train"], train_no_call_count, seed, "negative", "train"),
        seed,
        "train",
    )
    validation = _mixed_order(
        _select(
            positive_calls["validation"],
            validation_call_count,
            seed,
            "positive",
            "validation",
        )
        + _select(
            negative_no_calls["validation"],
            validation_no_call_count,
            seed,
            "negative",
            "validation",
        ),
        seed,
        "validation",
    )
    if {record_fingerprint(row) for row in train} & {
        record_fingerprint(row) for row in validation
    }:
        raise ValueError("merged train and validation splits overlap")

    output.mkdir(parents=True)
    train_path = output / "train.jsonl"
    validation_path = output / "validation.jsonl"
    write_jsonl(train_path, train)
    write_jsonl(validation_path, validation)
    total = len(train) + len(validation)
    no_calls = train_no_call_count + validation_no_call_count
    manifest = {
        "schemaVersion": 2,
        "seed": seed,
        "noCallFraction": no_calls / total,
        "selection": "lowest SHA-256 of seed:merge:role:split:record-fingerprint",
        "statistics": {
            "positiveTrainExcludedNoCalls": len(positive_splits["train"])
            - len(positive_calls["train"]),
            "positiveValidationExcludedNoCalls": len(positive_splits["validation"])
            - len(positive_calls["validation"]),
            "negativeTrainExcludedCalls": len(negative_splits["train"])
            - len(negative_no_calls["train"]),
            "negativeValidationExcludedCalls": len(negative_splits["validation"])
            - len(negative_no_calls["validation"]),
        },
        "parents": {
            "positive": {
                "role": "call examples only",
                "manifestSha256": sha256(positive_manifest_path),
                "trainSha256": positive_manifest["train"]["sha256"],
                "validationSha256": positive_manifest["validation"]["sha256"],
            },
            "negative": {
                "role": "hard no-applicable-tool examples only",
                "manifestSha256": sha256(negative_manifest_path),
                "trainSha256": negative_manifest["train"]["sha256"],
                "validationSha256": negative_manifest["validation"]["sha256"],
            },
        },
        "train": {
            "count": len(train),
            "callCount": train_call_count,
            "noCallCount": train_no_call_count,
            "sha256": sha256(train_path),
        },
        "validation": {
            "count": len(validation),
            "callCount": validation_call_count,
            "noCallCount": validation_no_call_count,
            "sha256": sha256(validation_path),
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positive", required=True, type=Path)
    parser.add_argument("--negative", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--train-call-count", required=True, type=int)
    parser.add_argument("--train-no-call-count", required=True, type=int)
    parser.add_argument("--validation-call-count", required=True, type=int)
    parser.add_argument("--validation-no-call-count", required=True, type=int)
    parser.add_argument("--seed", type=int, default=20_260_913)
    args = parser.parse_args()
    result = merge_prepared(
        positive=args.positive,
        negative=args.negative,
        output=args.out,
        train_call_count=args.train_call_count,
        train_no_call_count=args.train_no_call_count,
        validation_call_count=args.validation_call_count,
        validation_no_call_count=args.validation_no_call_count,
        seed=args.seed,
    )
    print(canonical_json(result))


if __name__ == "__main__":
    main()
