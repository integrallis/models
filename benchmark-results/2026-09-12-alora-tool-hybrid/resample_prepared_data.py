#!/usr/bin/env python3
"""Create a provenance-bound, stratified subset of already verified prepared tool data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from prepare_tool_data import (
    canonical_json,
    record_fingerprint,
    sha256,
    split_stratified,
    write_jsonl,
)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read parent manifest: {error}") from error
    if not isinstance(value, dict) or value.get("schemaVersion") != 2:
        raise ValueError("parent manifest must use preparation schemaVersion 2")
    return value


def _verified_rows(parent: Path, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for split in ("train", "validation"):
        path = parent / f"{split}.jsonl"
        try:
            expected = manifest[split]["sha256"]
        except (KeyError, TypeError) as error:
            raise ValueError(f"parent manifest has no {split} split hash") from error
        actual = sha256(path)
        if actual != expected:
            raise ValueError(
                f"parent {split} split SHA-256 differs: expected {expected}, got {actual}"
            )
        with path.open() as source:
            for line in source:
                record = json.loads(line)
                fingerprint = record_fingerprint(record)
                if fingerprint in seen:
                    raise ValueError("parent preparation contains a duplicate record")
                seen.add(fingerprint)
                rows.append(record)
    return rows


def _select(rows: list[dict[str, Any]], count: int, no_call_count: int, seed: int) -> list[dict]:
    no_calls = [record for record in rows if not record["calls"]]
    calls = [record for record in rows if record["calls"]]

    def priority(record: dict[str, Any]) -> bytes:
        fingerprint = record_fingerprint(record)
        return hashlib.sha256(f"{seed}:resample:{fingerprint}".encode()).digest()

    no_calls.sort(key=priority)
    calls.sort(key=priority)
    call_count = count - no_call_count
    if len(no_calls) < no_call_count or len(calls) < call_count:
        raise ValueError(
            "parent preparation cannot supply requested strata: "
            f"no-call {len(no_calls)}/{no_call_count}, call {len(calls)}/{call_count}"
        )
    return no_calls[:no_call_count] + calls[:call_count]


def resample(
    *,
    parent: Path,
    output: Path,
    train_count: int,
    validation_count: int,
    no_call_fraction: float,
    seed: int,
) -> dict[str, Any]:
    if output.exists():
        raise ValueError(f"output already exists: {output}")
    if train_count <= 0 or validation_count <= 0:
        raise ValueError("train and validation counts must be positive")
    if not 0.0 <= no_call_fraction <= 1.0:
        raise ValueError("no-call fraction must be between zero and one")

    parent_manifest_path = parent / "manifest.json"
    parent_manifest = _load_json(parent_manifest_path)
    rows = _verified_rows(parent, parent_manifest)
    total_count = train_count + validation_count
    total_no_calls = round(train_count * no_call_fraction) + round(
        validation_count * no_call_fraction
    )
    selected = _select(rows, total_count, total_no_calls, seed)
    train, validation = split_stratified(
        selected, train_count, validation_count, no_call_fraction, seed
    )

    output.mkdir(parents=True)
    train_path = output / "train.jsonl"
    validation_path = output / "validation.jsonl"
    write_jsonl(train_path, train)
    write_jsonl(validation_path, validation)

    manifest = {
        key: value
        for key, value in parent_manifest.items()
        if key not in {"seed", "noCallFraction", "train", "validation", "statistics"}
    }
    manifest.update(
        {
            "schemaVersion": 2,
            "seed": seed,
            "noCallFraction": no_call_fraction,
            "statistics": {
                "parentCount": len(rows),
                "parentNoCallCount": sum(not record["calls"] for record in rows),
                "selectedCount": len(selected),
                "selectedNoCallCount": sum(not record["calls"] for record in selected),
            },
            "resampling": {
                "parentManifestSha256": sha256(parent_manifest_path),
                "parentTrainSha256": parent_manifest["train"]["sha256"],
                "parentValidationSha256": parent_manifest["validation"]["sha256"],
                "selection": "lowest SHA-256 of seed:resample:record-fingerprint per stratum",
            },
            "train": {
                "count": len(train),
                "noCallCount": sum(not record["calls"] for record in train),
                "sha256": sha256(train_path),
                "sourceLines": [record["sourceLine"] for record in train],
            },
            "validation": {
                "count": len(validation),
                "noCallCount": sum(not record["calls"] for record in validation),
                "sha256": sha256(validation_path),
                "sourceLines": [record["sourceLine"] for record in validation],
            },
        }
    )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--train-count", required=True, type=int)
    parser.add_argument("--validation-count", required=True, type=int)
    parser.add_argument("--no-call-fraction", required=True, type=float)
    parser.add_argument("--seed", type=int, default=20_260_913)
    args = parser.parse_args()
    manifest = resample(
        parent=args.parent,
        output=args.out,
        train_count=args.train_count,
        validation_count=args.validation_count,
        no_call_fraction=args.no_call_fraction,
        seed=args.seed,
    )
    print(canonical_json(manifest))


if __name__ == "__main__":
    main()
