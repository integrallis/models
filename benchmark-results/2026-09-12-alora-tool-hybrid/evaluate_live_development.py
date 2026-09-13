#!/usr/bin/env python3
"""Evaluate adapter calibration on a pinned BFCL live slice disjoint from qualification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evaluate_alora import (
    BFCL_REVISION,
    generate_mode,
    load_slice,
    read_jsonl,
    resolve_adapter_identity,
    select_cases,
    sha256,
    verify_bfcl_data,
    verify_sha256,
)
from prepare_tool_data import normalize_tools, query_fingerprint


DEVELOPMENT_SEED = 20260915
LIVE_FILES = {
    "BFCL_v3_live_simple.json": "10e5cedb6dc73cd16bc430755643c658d6032a69363dfc41a22a84988eaa312a",
    "BFCL_v3_live_multiple.json": "791d1faaf5ba5579196cd45a5fb37fa096d88848d6077f33cb6e2d22ebf58206",
    "BFCL_v3_live_irrelevance.json": "0d259e1c3ab6ba06c2a51911e2f49ecaa2eac3723a4a14f183a7687955e61338",
    "possible_answer/BFCL_v3_live_simple.json": "4a1988780cdd8a723967587614cb84a9f53aa5d6fd37f83c78a0dd7acc2f3d15",
    "possible_answer/BFCL_v3_live_multiple.json": "97e90d59c5bd76c55a2920ce93e5566e9046307d3f558578f085f9d3a56c3084",
}


def load_live_slice(data: Path, kind: str, count: int, seed: int) -> list[dict[str, Any]]:
    cases = select_cases(read_jsonl(data / f"BFCL_v3_live_{kind}.json"), count, seed)
    answers = (
        {}
        if kind == "irrelevance"
        else {
            answer["id"]: answer["ground_truth"]
            for answer in read_jsonl(
                data / "possible_answer" / f"BFCL_v3_live_{kind}.json"
            )
        }
    )
    result = []
    for case in cases:
        tools = normalize_tools(
            [
                {
                    "type": "function",
                    "function": {
                        "name": function["name"],
                        "description": function.get("description", ""),
                        "parameters": function.get(
                            "parameters", {"type": "object", "properties": {}}
                        ),
                    },
                }
                for function in case["function"]
            ]
        )
        result.append(
            {
                "id": case["id"],
                "kind": kind,
                "messages": case["question"][0],
                "tools": tools,
                "expected": answers.get(case["id"], []),
            }
        )
    return result


def _queries(cases: list[dict[str, Any]]) -> set[str]:
    return {
        query_fingerprint(message["content"])
        for case in cases
        for message in case["messages"]
        if message.get("role") == "user"
    }


def assert_disjoint(
    development: list[dict[str, Any]], qualification: list[dict[str, Any]]
) -> None:
    overlap = _queries(development) & _queries(qualification)
    if overlap:
        raise ValueError(f"development slice overlaps qualification in {len(overlap)} queries")


def assert_disjoint_from_prepared(
    development: list[dict[str, Any]], prepared_roots: list[Path]
) -> None:
    development_queries = _queries(development)
    prepared_queries: set[str] = set()
    for root in prepared_roots:
        for split in ("train", "validation"):
            split_path = root / f"{split}.jsonl"
            try:
                rows = read_jsonl(split_path)
            except (OSError, json.JSONDecodeError) as error:
                raise ValueError(f"cannot read prepared {split_path}: {error}") from error
            for row in rows:
                user = row.get("user")
                if not isinstance(user, str) or not user.strip():
                    raise ValueError(f"prepared {split_path} contains a row without a user query")
                prepared_queries.add(query_fingerprint(user))
    overlap = development_queries & prepared_queries
    if overlap:
        raise ValueError(
            f"development slice overlaps prepared training data in {len(overlap)} queries"
        )


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    tool_cases = [record for record in records if record["kind"] != "irrelevance"]
    irrelevant = [record for record in records if record["kind"] == "irrelevance"]
    return {
        "cases": len(records),
        "syntaxRate": sum(record["syntaxValid"] for record in records) / len(records),
        "schemaRate": sum(record["schemaValid"] for record in records) / len(records),
        "toolExactRate": sum(record["exact"] for record in tool_cases) / len(tool_cases),
        "irrelevanceFalseToolRate": sum(
            bool(record.get("parsedCalls")) for record in irrelevant
        )
        / len(irrelevant),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=True, type=Path)
    parser.add_argument("--training-manifest", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--qualification-data", required=True, type=Path)
    parser.add_argument(
        "--prepared",
        required=True,
        type=Path,
        nargs="+",
        help="every prepared corpus used by this adapter or any continuation ancestor",
    )
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--count", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"output already exists: {args.out}")
    if args.count <= 0 or args.batch_size <= 0 or args.max_new_tokens <= 0:
        parser.error("count, batch size, and max new tokens must be positive")
    for relative, expected in LIVE_FILES.items():
        verify_sha256(args.data / relative, expected)
    verify_bfcl_data(args.qualification_data)
    identity = resolve_adapter_identity(args.adapter, args.training_manifest)
    development = [
        case
        for kind in ("simple", "multiple", "irrelevance")
        for case in load_live_slice(args.data, kind, args.count, DEVELOPMENT_SEED)
    ]
    qualification = [
        case
        for kind in ("simple", "multiple", "irrelevance")
        for case in load_slice(args.qualification_data, kind, 100)
    ]
    assert_disjoint(development, qualification)
    assert_disjoint_from_prepared(development, args.prepared)

    import peft
    import torch
    import transformers
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(identity["model"], revision=identity["revision"])
    tokenizer.padding_side = "left"
    base = AutoModelForCausalLM.from_pretrained(
        identity["model"],
        revision=identity["revision"],
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base.to(device)
    model = PeftModel.from_pretrained(base, args.adapter)
    model.eval()
    records = generate_mode(
        model,
        tokenizer,
        development,
        "adapter",
        args.batch_size,
        args.max_new_tokens,
    )
    args.out.mkdir(parents=True)
    records_path = args.out / "records.jsonl"
    with records_path.open("w") as target:
        for record in records:
            target.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    report = {
        "schemaVersion": 1,
        "qualifyingRun": False,
        "purpose": "adapter interpolation calibration only",
        "selection": {"seed": DEVELOPMENT_SEED, "perKind": args.count},
        "source": {
            "repository": "ShishirPatil/gorilla",
            "revision": BFCL_REVISION,
            "files": [
                {"path": path, "sha256": digest}
                for path, digest in sorted(LIVE_FILES.items())
            ],
        },
        "qualificationOverlapQueries": 0,
        "preparedTrainingOverlapQueries": 0,
        "preparedCorpora": [
            {
                "root": str(root),
                "trainSha256": sha256(root / "train.jsonl"),
                "validationSha256": sha256(root / "validation.jsonl"),
            }
            for root in args.prepared
        ],
        "base": {"model": identity["model"], "revision": identity["revision"]},
        "adapter": {
            "sha256": identity["adapterSha256"],
            "trainingManifestSha256": identity["trainingManifestSha256"],
        },
        "environment": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": str(device),
            "transformers": transformers.__version__,
            "peft": peft.__version__,
        },
        "records": {"path": records_path.name, "sha256": sha256(records_path)},
        "summary": summarize(records),
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
