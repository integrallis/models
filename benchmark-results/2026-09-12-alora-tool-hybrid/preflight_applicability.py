#!/usr/bin/env python3
"""Run every CPU-only V13 corpus, tokenizer, alignment, and objective preflight."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
from pathlib import Path
from typing import Any

from prepare_tool_data import bfcl_query_fingerprints, query_fingerprint
from train_applicability_alora import (
    CALL_LABEL,
    V13_TRAIN_CALL_COUNT,
    V13_TRAIN_NO_CALL_COUNT,
    V13_VALIDATION_CALL_COUNT,
    V13_VALIDATION_NO_CALL_COUNT,
    class_weights,
    combined_loss,
    decision_contract,
    torch_decision_loss,
    verify_v13_corpus,
)


def prepared_query_overlap(
    prepared_paths: list[Path], evaluation_paths: list[Path]
) -> dict[str, int]:
    prepared_queries: set[str] = set()
    for path in prepared_paths:
        with path.open() as source:
            for line in source:
                if line.strip():
                    prepared_queries.add(query_fingerprint(json.loads(line)["user"]))
    evaluation_queries = bfcl_query_fingerprints(evaluation_paths)
    return {
        "preparedQueries": len(prepared_queries),
        "evaluationQueries": len(evaluation_queries),
        "overlap": len(prepared_queries & evaluation_queries),
    }


def tensor_objective_preflight(contract: Any, torch_module: Any) -> dict[str, Any]:
    torch = torch_module
    call_weight, no_call_weight = class_weights(
        V13_TRAIN_CALL_COUNT, V13_TRAIN_NO_CALL_COUNT
    )
    vocab_size = max(contract.call_token_id, contract.no_call_token_id) + 2
    logits = torch.zeros((1, 3, vocab_size), dtype=torch.float32, requires_grad=True)
    positions = torch.tensor([2], dtype=torch.long)
    labels = torch.tensor([CALL_LABEL], dtype=torch.long)
    decision = torch_decision_loss(
        logits=logits,
        decision_positions=positions,
        decision_labels=labels,
        call_token_id=contract.call_token_id,
        no_call_token_id=contract.no_call_token_id,
        call_weight=call_weight,
        no_call_weight=no_call_weight,
        torch_module=torch,
    )
    decision.backward()
    nonzero = torch.nonzero(logits.grad, as_tuple=False).tolist()
    expected_nonzero = [
        [0, 1, contract.call_token_id],
        [0, 1, contract.no_call_token_id],
    ]
    if nonzero != expected_nonzero:
        raise ValueError(
            f"auxiliary gradient reached nondecision logits: expected {expected_nonzero}, got {nonzero}"
        )
    call_gradient = float(logits.grad[0, 1, contract.call_token_id])
    no_call_gradient = float(logits.grad[0, 1, contract.no_call_token_id])
    if not call_gradient < 0 < no_call_gradient:
        raise ValueError("auxiliary gradient signs do not increase the correct decision margin")

    language_model_loss = torch.tensor(1.25, requires_grad=True)
    if combined_loss(language_model_loss, decision.detach(), 0.0) is not language_model_loss:
        raise ValueError("zero coefficient does not preserve the exact language-model loss")

    fixture = torch.tensor(
        [[[0.0] * vocab_size, [0.0] * vocab_size, [0.0] * vocab_size]],
        dtype=torch.float32,
    )
    fixture[0, 1, contract.call_token_id] = 1.5
    fixture[0, 1, contract.no_call_token_id] = -0.5
    fixture_loss = torch_decision_loss(
        logits=fixture,
        decision_positions=positions,
        decision_labels=labels,
        call_token_id=contract.call_token_id,
        no_call_token_id=contract.no_call_token_id,
        call_weight=call_weight,
        no_call_weight=no_call_weight,
        torch_module=torch,
    )
    serialized = io.BytesIO()
    torch.save({"logits": fixture}, serialized)
    serialized.seek(0)
    reloaded = torch.load(serialized, map_location="cpu", weights_only=True)["logits"]
    reloaded_loss = torch_decision_loss(
        logits=reloaded,
        decision_positions=positions,
        decision_labels=labels,
        call_token_id=contract.call_token_id,
        no_call_token_id=contract.no_call_token_id,
        call_weight=call_weight,
        no_call_weight=no_call_weight,
        torch_module=torch,
    )
    if not torch.equal(fixture, reloaded) or not torch.equal(fixture_loss, reloaded_loss):
        raise ValueError("the CPU save/reload fixture changed logits or component loss")
    return {
        "predictionLogitPosition": 1,
        "decisionTokenPosition": 2,
        "nonzeroAuxiliaryGradientCoordinates": nonzero,
        "callGradient": call_gradient,
        "noCallGradient": no_call_gradient,
        "callWeight": call_weight,
        "noCallWeight": no_call_weight,
        "fixtureLogitsSha256": hashlib.sha256(fixture.numpy().tobytes()).hexdigest(),
        "fixtureDecisionLoss": float(fixture_loss),
        "fixtureReloadExact": True,
        "zeroCoefficientReturnsLanguageModelLoss": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--bfcl", required=True, type=Path, nargs="+")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    import torch
    import transformers
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    contract = decision_contract(tokenizer)
    train, validation, train_report, validation_report = verify_v13_corpus(
        args.prepared, tokenizer, 1024
    )
    if len(train) + len(validation) != 3311:
        raise ValueError("V13 must contain exactly 3,311 usable train/validation examples")
    overlap = prepared_query_overlap(
        [args.prepared / "train.jsonl", args.prepared / "validation.jsonl"],
        args.bfcl,
    )
    if overlap["overlap"] != 0:
        raise ValueError(f"V13 training queries overlap static BFCL evaluation: {overlap}")
    tensor_report = tensor_objective_preflight(contract, torch)
    if not all(
        math.isfinite(value)
        for value in (
            tensor_report["callGradient"],
            tensor_report["noCallGradient"],
            tensor_report["fixtureDecisionLoss"],
        )
    ):
        raise ValueError("V13 synthetic objective produced a non-finite result")
    report = {
        "schemaVersion": 1,
        "experiment": "qwen3-17b-applicability-decision-v13",
        "passed": True,
        "tokenizer": {
            "path": str(args.tokenizer),
            "transformers": transformers.__version__,
            "sharedPrefixTokens": list(contract.shared_prefix_tokens),
            "callToken": contract.call_token_id,
            "noCallToken": contract.no_call_token_id,
        },
        "corpus": {"train": train_report, "validation": validation_report},
        "contamination": overlap,
        "objective": tensor_report,
        "torch": torch.__version__,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.out is not None:
        if args.out.exists():
            parser.error(f"output already exists: {args.out}")
        args.out.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
