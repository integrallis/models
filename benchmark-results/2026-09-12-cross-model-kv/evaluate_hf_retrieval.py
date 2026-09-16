#!/usr/bin/env python3
"""Exact-answer long-context gates using the pinned KVBridge task generators."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import time
from pathlib import Path
from typing import Any, Sequence

from evaluate_hf_quality import Oracle


def exact_answer_retention(records: Sequence[dict[str, int]]) -> dict[str, Any]:
    """Require every answer the native target gets right to survive mapped handoff."""
    native_correct = [row for row in records if row["nativeWinner"] == row["gold"]]
    retained = [row for row in native_correct if row["mappedWinner"] == row["gold"]]
    count = len(native_correct)
    retention = len(retained) / count if count else 0.0
    return {
        "nativeCorrect": count,
        "retainedNativeCorrect": len(retained),
        "retention": retention,
        "pass": count > 0 and len(retained) == count,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reference_tasks(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("pinned_kvbridge_tasks", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load reference task generators from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.TASKS


def _winner(scores: Sequence[float], lengths: Sequence[int]) -> int:
    return max(range(len(scores)), key=lambda index: scores[index] / lengths[index])


def evaluate_task(
    oracle: Oracle,
    task: Any,
    task_name: str,
    context_tokens: int,
    samples: int,
    depths: int,
    seed: int,
) -> dict[str, Any]:
    generated = task(
        oracle.tokenizer,
        n_ctx=context_tokens,
        n_samples=samples,
        n_depths=depths,
        seed=seed,
    )
    if any(sample.mode != "choice" for sample in generated):
        raise ValueError(f"{task_name} must use exact-choice scoring")
    records = []
    started = time.time()
    for number, sample in enumerate(generated, 1):
        context = oracle.tokenizer(sample.context)["input_ids"]
        context.extend(
            oracle.tokenizer(sample.query, add_special_tokens=False)["input_ids"]
        )
        options = [
            oracle.tokenizer(option, add_special_tokens=False)["input_ids"]
            for option in sample.options
        ]
        paired = oracle.paired_scores_many(context, options)
        native_scores = [score[0] for score in paired]
        mapped_scores = [score[1] for score in paired]
        lengths = [len(option) for option in options]
        records.append(
            {
                "sample": number - 1,
                "depth": sample.meta.get("depth"),
                "contextTokens": len(context),
                "gold": sample.gold,
                "nativeWinner": _winner(native_scores, lengths),
                "mappedWinner": _winner(mapped_scores, lengths),
            }
        )
        print(f"{task_name} {context_tokens} {number}/{len(generated)}", flush=True)
    gate = exact_answer_retention(records)
    return {
        "task": task_name,
        "requestedContextTokens": context_tokens,
        "samples": len(records),
        "depths": depths,
        "elapsedSeconds": time.time() - started,
        "gate": gate,
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--target-revision", required=True)
    parser.add_argument("--mapper", required=True, type=Path)
    parser.add_argument("--reference-tasks", required=True, type=Path)
    parser.add_argument("--reference-commit", required=True)
    parser.add_argument("--tasks", default="kvr,niah")
    parser.add_argument("--context-lengths", default="4096")
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--depths", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--native-target-sink", type=int, default=4)
    parser.add_argument("--recomputed-target-tail", type=int, default=512)
    parser.add_argument("--mapping-token-chunk", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--attention-implementation", default="sdpa")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.samples < 1 or args.depths < 1:
        parser.error("--samples and --depths must be positive")
    if len(args.reference_commit) != 40:
        parser.error("--reference-commit must be a full 40-character commit")
    task_names = [name.strip() for name in args.tasks.split(",") if name.strip()]
    lengths = [int(value) for value in args.context_lengths.split(",")]
    if not task_names or any(length < 2 for length in lengths):
        parser.error("at least one task and context length >= 2 are required")
    tasks = _reference_tasks(args.reference_tasks)
    unknown = [name for name in task_names if name not in tasks]
    if unknown:
        parser.error(f"unknown reference tasks: {unknown}")

    oracle = Oracle(args)
    results = [
        evaluate_task(oracle, tasks[name], name, length, args.samples, args.depths, args.seed)
        for length in lengths
        for name in task_names
    ]
    report = {
        "schemaVersion": 1,
        "source": {"model": args.source_model, "revision": args.source_revision},
        "target": {"model": args.target_model, "revision": args.target_revision},
        "mapper": {"path": str(args.mapper), "sha256": _sha256(args.mapper)},
        "referenceTasks": {
            "path": str(args.reference_tasks),
            "sha256": _sha256(args.reference_tasks),
            "repository": "https://github.com/spotta85/kvbridge.git",
            "commit": args.reference_commit,
        },
        "strategy": {
            "nativeTargetSinkTokens": args.native_target_sink,
            "recomputedTargetTailTokens": args.recomputed_target_tail,
            "mappingTokenChunk": args.mapping_token_chunk,
        },
        "results": results,
        "qualified": all(result["gate"]["pass"] for result in results),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps({"qualified": report["qualified"], "gates": [r["gate"] for r in results]}))
    raise SystemExit(0 if report["qualified"] else 2)


if __name__ == "__main__":
    main()
