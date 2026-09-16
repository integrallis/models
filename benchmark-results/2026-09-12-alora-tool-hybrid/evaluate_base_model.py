#!/usr/bin/env python3
"""Screen a pinned unadapted base model on the unchanged held-out BFCL slice."""

from __future__ import annotations

import argparse
import contextlib
import json
from pathlib import Path
from typing import Any

from evaluate_alora import (
    BFCL_FILES,
    BFCL_REPOSITORY,
    BFCL_REVISION,
    QUALIFICATION_COUNT,
    SEED,
    generate_mode,
    inference_device_name,
    load_slice,
    sha256,
    verify_bfcl_data,
)


CANDIDATES = {
    "qwen3-1.7b": ("Qwen/Qwen3-1.7B", "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"),
}


def candidate(name: str) -> tuple[str, str]:
    try:
        return CANDIDATES[name]
    except KeyError as error:
        raise ValueError(f"unsupported base candidate: {name}") from error


def summarize_base(records: list[dict[str, Any]], elapsed: float) -> dict[str, Any]:
    if not records:
        raise ValueError("base evaluation produced no records")
    tool_cases = [record for record in records if record["kind"] != "irrelevance"]
    irrelevant = [record for record in records if record["kind"] == "irrelevance"]
    if not tool_cases or not irrelevant:
        raise ValueError("base evaluation must contain tool and irrelevance cases")
    return {
        "cases": len(records),
        "syntaxRate": sum(record["syntaxValid"] for record in records) / len(records),
        "schemaRate": sum(record["schemaValid"] for record in records) / len(records),
        "toolExactRate": sum(record["exact"] for record in tool_cases) / len(tool_cases),
        "irrelevanceFalseToolRate": sum(
            bool(record.get("parsedCalls")) for record in irrelevant
        )
        / len(irrelevant),
        "elapsedSeconds": elapsed,
    }


class BaseModelView:
    """Give the shared evaluator the no-op adapter context expected by its base arm."""

    def __init__(self, model: Any):
        self._model = model

    def disable_adapter(self):
        return contextlib.nullcontext()

    def __getattr__(self, name: str):
        return getattr(self._model, name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", choices=sorted(CANDIDATES), required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--smoke-count", type=int)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"output already exists: {args.out}")
    if args.batch_size <= 0 or args.max_new_tokens <= 0:
        parser.error("batch size and max new tokens must be positive")
    count = args.smoke_count or QUALIFICATION_COUNT
    verify_bfcl_data(args.data)
    model_id, model_revision = candidate(args.candidate)

    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=model_revision)
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=model_revision,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    device = torch.device(inference_device_name(torch.cuda.is_available()))
    model.to(device)
    model.eval()
    cases = []
    for kind in ("simple", "multiple", "irrelevance"):
        cases.extend(load_slice(args.data, kind, count))
    records = generate_mode(
        BaseModelView(model),
        tokenizer,
        cases,
        "base",
        args.batch_size,
        args.max_new_tokens,
    )
    elapsed = records[0]["modeElapsedSeconds"]
    args.out.mkdir(parents=True)
    records_path = args.out / "records.jsonl"
    with records_path.open("w") as target:
        for record in records:
            target.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    report = {
        "schemaVersion": 1,
        "qualifyingRun": False,
        "purpose": "base-screen-before-adapter-training",
        "selection": {"seed": SEED, "perKind": count},
        "base": {"model": model_id, "revision": model_revision},
        "evaluationSource": {
            "repository": BFCL_REPOSITORY,
            "revision": BFCL_REVISION,
            "files": [
                {"path": path, "sha256": digest}
                for path, digest in sorted(BFCL_FILES.items())
            ],
        },
        "environment": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "device": str(device),
            "transformers": transformers.__version__,
        },
        "records": {"path": records_path.name, "sha256": sha256(records_path)},
        "summary": summarize_base(records, elapsed),
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
