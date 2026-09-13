#!/usr/bin/env python3
"""Train a paired Qwen3 aLoRA with an isolated call/no-call decision loss."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import gc
import hashlib
import json
import math
import platform
import random
from pathlib import Path
from typing import Any

from train_alora import (
    CANDIDATES,
    INVOCATION_STRING,
    INVOCATION_TOKENS,
    TARGET_MODULES,
    build_example,
    prepare_for_evaluation,
    resolve_candidate,
    resolve_initial_adapter,
    sha256,
    verify_prepared_splits,
)


CALL_LABEL = 0
NO_CALL_LABEL = 1
TOOL_CALL_PREFIX = "<tool_call>\n"
CALL_CONTEXT = TOOL_CALL_PREFIX + '{"arguments":{},"name":"x"}\n</tool_call>'
NO_CALL_CONTEXT = TOOL_CALL_PREFIX + "[]\n</tool_call>"
V13_PREPARATION_MANIFEST_SHA256 = (
    "6abf29b6040a2831f52ad0b77a7276a2213e5c6bc775c4c68b40e3a3dc9b788c"
)
V13_TRAIN_SHA256 = "c46b606c74a3d1bda38c6f50a55854bf9112f0ae9d92993eb32e577c211f3002"
V13_VALIDATION_SHA256 = "7006d89f1be432424889f8fbc846ff348cf2770a56f409c9e38361ae4857cfb0"
V13_TRAIN_SOURCE_LINES_SHA256 = (
    "ceab03a876d41ada5e7a3f99cad4357e0b70a478b6f8c20f82da7aa0a1626941"
)
V13_VALIDATION_SOURCE_LINES_SHA256 = (
    "1ce289514ff5ddbe72a34e240196925b8d341058e66278d627b477f5fc622f51"
)
V13_TRAIN_CALL_COUNT = 2356
V13_TRAIN_NO_CALL_COUNT = 484
V13_VALIDATION_CALL_COUNT = 390
V13_VALIDATION_NO_CALL_COUNT = 81


@dataclass(frozen=True)
class DecisionContract:
    shared_prefix_tokens: tuple[int, ...]
    call_prefix_tokens: tuple[int, ...]
    no_call_prefix_tokens: tuple[int, ...]
    call_token_id: int
    no_call_token_id: int


def _tokenize(tokenizer: Any, value: str) -> tuple[int, ...]:
    tokens = tokenizer.encode(value, add_special_tokens=False)
    if hasattr(tokens, "tolist"):
        tokens = tokens.tolist()
    return tuple(tokens)


def decision_contract(tokenizer: Any) -> DecisionContract:
    """Resolve the one-token contextual branch after the canonical tool prefix."""
    shared = _tokenize(tokenizer, TOOL_CALL_PREFIX)
    call_context = _tokenize(tokenizer, CALL_CONTEXT)
    no_call_context = _tokenize(tokenizer, NO_CALL_CONTEXT)
    if (
        call_context[: len(shared)] != shared
        or no_call_context[: len(shared)] != shared
        or len(call_context) <= len(shared)
        or len(no_call_context) <= len(shared)
    ):
        raise ValueError(
            "call and no-call contexts must preserve the canonical tool prefix"
        )
    call_token = call_context[len(shared)]
    no_call_token = no_call_context[len(shared)]
    if call_token == no_call_token:
        raise ValueError("call and no-call decision tokens must be distinct")
    return DecisionContract(
        shared,
        (*shared, call_token),
        (*shared, no_call_token),
        call_token,
        no_call_token,
    )


def decision_prediction_position(decision_position: int) -> int:
    """Return the causal-logit index that predicts a decision token."""
    if decision_position <= 0:
        raise ValueError("a decision token must have a preceding causal logit")
    return decision_position - 1


def build_decision_example(
    record: dict[str, Any], tokenizer: Any, max_length: int
) -> dict[str, Any] | None:
    """Build a normal SFT example and bind its sole call/no-call decision token."""
    example = build_example(record, tokenizer, max_length)
    if example is None:
        return None
    contract = decision_contract(tokenizer)
    try:
        prompt_length = example["labels"].index(next(label for label in example["labels"] if label != -100))
    except (StopIteration, ValueError) as error:
        raise ValueError(f"source line {record['sourceLine']} has no supervised completion") from error
    expected_prefix = (
        contract.call_prefix_tokens if record["calls"] else contract.no_call_prefix_tokens
    )
    actual_prefix = tuple(
        example["input_ids"][prompt_length : prompt_length + len(expected_prefix)]
    )
    if actual_prefix != expected_prefix:
        raise ValueError(
            f"source line {record['sourceLine']} does not have the canonical decision prefix"
        )
    decision_position = prompt_length + len(contract.shared_prefix_tokens)
    decision_label = CALL_LABEL if record["calls"] else NO_CALL_LABEL
    expected_token = (
        contract.call_token_id if decision_label == CALL_LABEL else contract.no_call_token_id
    )
    if example["input_ids"][decision_position] != expected_token:
        raise ValueError(f"source line {record['sourceLine']} has the wrong decision token")
    return {
        **example,
        "decision_position": decision_position,
        "decision_label": decision_label,
    }


def class_weights(call_count: int, no_call_count: int) -> tuple[float, float]:
    """Return manually applied inverse-frequency weights with mean weight one."""
    if call_count <= 0 or no_call_count <= 0:
        raise ValueError("both applicability classes must be present")
    total = call_count + no_call_count
    return total / (2 * call_count), total / (2 * no_call_count)


def auxiliary_loss(
    correct_logit: float, wrong_logit: float, example_weight: float
) -> float:
    """Stable two-class cross entropy used by the decision-boundary objective."""
    if example_weight <= 0 or not math.isfinite(example_weight):
        raise ValueError("the example weight must be finite and positive")
    margin = correct_logit - wrong_logit
    return example_weight * (max(-margin, 0.0) + math.log1p(math.exp(-abs(margin))))


def combined_loss(language_model_loss: Any, decision_loss: Any, coefficient: float) -> Any:
    if coefficient < 0 or not math.isfinite(coefficient):
        raise ValueError("the applicability-loss coefficient must be finite and non-negative")
    if coefficient == 0.0:
        return language_model_loss
    return language_model_loss + coefficient * decision_loss


def load_decision_examples(
    path: Path, tokenizer: Any, max_length: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    skipped = 0
    no_call = 0
    multiple = 0
    with path.open() as source:
        for line in source:
            record = json.loads(line)
            example = build_decision_example(record, tokenizer, max_length)
            if example is None:
                skipped += 1
                continue
            examples.append(example)
            no_call += not record["calls"]
            multiple += len(record["calls"]) > 1
    if not examples:
        raise ValueError(f"no usable examples in {path}")
    return examples, {
        "source": str(path),
        "sha256": sha256(path),
        "usable": len(examples),
        "skippedOverMaxLength": skipped,
        "call": len(examples) - no_call,
        "noCall": no_call,
        "multipleCall": multiple,
        "sourceLinesSha256": hashlib.sha256(
            json.dumps(
                [example["source_line"] for example in examples], separators=(",", ":")
            ).encode()
        ).hexdigest(),
    }


def verify_v13_corpus(
    prepared: Path, tokenizer: Any, max_length: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    manifest = verify_prepared_splits(prepared)
    expected_manifest = {
        "manifest": V13_PREPARATION_MANIFEST_SHA256,
        "train": V13_TRAIN_SHA256,
        "validation": V13_VALIDATION_SHA256,
    }
    actual_manifest = {
        "manifest": sha256(prepared / "manifest.json"),
        "train": sha256(prepared / "train.jsonl"),
        "validation": sha256(prepared / "validation.jsonl"),
    }
    if actual_manifest != expected_manifest:
        raise ValueError(
            f"V13 requires the exact V8 hard-mixed corpus: expected {expected_manifest}, "
            f"got {actual_manifest}"
        )
    train, train_report = load_decision_examples(
        prepared / "train.jsonl", tokenizer, max_length
    )
    validation, validation_report = load_decision_examples(
        prepared / "validation.jsonl", tokenizer, max_length
    )
    expected_reports = {
        "train": {
            "usable": V13_TRAIN_CALL_COUNT + V13_TRAIN_NO_CALL_COUNT,
            "call": V13_TRAIN_CALL_COUNT,
            "noCall": V13_TRAIN_NO_CALL_COUNT,
            "sourceLinesSha256": V13_TRAIN_SOURCE_LINES_SHA256,
        },
        "validation": {
            "usable": V13_VALIDATION_CALL_COUNT + V13_VALIDATION_NO_CALL_COUNT,
            "call": V13_VALIDATION_CALL_COUNT,
            "noCall": V13_VALIDATION_NO_CALL_COUNT,
            "sourceLinesSha256": V13_VALIDATION_SOURCE_LINES_SHA256,
        },
    }
    actual_reports = {
        split: {key: report[key] for key in expected_reports[split]}
        for split, report in (("train", train_report), ("validation", validation_report))
    }
    if actual_reports != expected_reports:
        raise ValueError(
            f"V13 tokenizer filtering changed: expected {expected_reports}, got {actual_reports}"
        )
    return train, validation, train_report, validation_report


class DecisionDataset:
    def __init__(self, examples: list[dict[str, Any]]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        example = self.examples[index]
        return {
            key: example[key]
            for key in (
                "input_ids",
                "attention_mask",
                "labels",
                "decision_position",
                "decision_label",
            )
        }


class DecisionCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        length = max(len(feature["input_ids"]) for feature in features)
        length = ((length + 7) // 8) * 8
        return {
            "input_ids": torch.tensor(
                [
                    feature["input_ids"]
                    + [self.pad_token_id] * (length - len(feature["input_ids"]))
                    for feature in features
                ],
                dtype=torch.long,
            ),
            "attention_mask": torch.tensor(
                [
                    feature["attention_mask"]
                    + [0] * (length - len(feature["attention_mask"]))
                    for feature in features
                ],
                dtype=torch.long,
            ),
            "labels": torch.tensor(
                [
                    feature["labels"]
                    + [-100] * (length - len(feature["labels"]))
                    for feature in features
                ],
                dtype=torch.long,
            ),
            "decision_positions": torch.tensor(
                [feature["decision_position"] for feature in features], dtype=torch.long
            ),
            "decision_labels": torch.tensor(
                [feature["decision_label"] for feature in features], dtype=torch.long
            ),
        }


def torch_decision_loss(
    *,
    logits: Any,
    decision_positions: Any,
    decision_labels: Any,
    call_token_id: int,
    no_call_token_id: int,
    call_weight: float,
    no_call_weight: float,
    torch_module: Any,
) -> Any:
    """Compute the manually weighted loss at the causal decision boundary only."""
    torch = torch_module
    batch = torch.arange(logits.shape[0], device=logits.device)
    prediction_positions = decision_positions - 1
    if bool(torch.any(prediction_positions < 0)):
        raise ValueError("decision positions must have preceding causal logits")
    boundary = logits[batch, prediction_positions]
    candidates = boundary[:, [call_token_id, no_call_token_id]]
    per_example = torch.nn.functional.cross_entropy(
        candidates.float(), decision_labels, reduction="none"
    )
    weights = torch.where(
        decision_labels == CALL_LABEL,
        torch.as_tensor(call_weight, device=logits.device, dtype=per_example.dtype),
        torch.as_tensor(no_call_weight, device=logits.device, dtype=per_example.dtype),
    )
    return (per_example * weights).mean()


def make_applicability_trainer(
    trainer_class: Any,
    *,
    contract: DecisionContract,
    call_weight: float,
    no_call_weight: float,
    coefficient: float,
    torch_module: Any,
) -> Any:
    class ApplicabilityTrainer(trainer_class):
        def __init__(self, *args: Any, **kwargs: Any):
            super().__init__(*args, **kwargs)
            self.finite_gradient_checks = 0

        def compute_loss(
            self, model: Any, inputs: dict[str, Any], return_outputs: bool = False, **_: Any
        ) -> Any:
            decision_positions = inputs.pop("decision_positions")
            decision_labels = inputs.pop("decision_labels")
            outputs = model(**inputs)
            decision = torch_decision_loss(
                logits=outputs.logits,
                decision_positions=decision_positions,
                decision_labels=decision_labels,
                call_token_id=contract.call_token_id,
                no_call_token_id=contract.no_call_token_id,
                call_weight=call_weight,
                no_call_weight=no_call_weight,
                torch_module=torch_module,
            )
            loss = combined_loss(outputs.loss, decision, coefficient)
            if not bool(torch_module.isfinite(outputs.loss)):
                raise FloatingPointError("language-model loss is not finite")
            if not bool(torch_module.isfinite(decision)):
                raise FloatingPointError("applicability loss is not finite")
            if not bool(torch_module.isfinite(loss)):
                raise FloatingPointError("combined training loss is not finite")
            return (loss, outputs) if return_outputs else loss

        def training_step(self, model: Any, inputs: dict[str, Any], *args: Any, **kwargs: Any) -> Any:
            loss = super().training_step(model, inputs, *args, **kwargs)
            for parameter in model.parameters():
                if parameter.grad is not None and not bool(
                    torch_module.all(torch_module.isfinite(parameter.grad))
                ):
                    raise FloatingPointError("training gradient is not finite")
            self.finite_gradient_checks += 1
            return loss

    return ApplicabilityTrainer


def make_stop_callback(callback_class: Any, optimizer_steps: int) -> Any:
    if optimizer_steps <= 0:
        raise ValueError("screen optimizer steps must be positive")

    class StopAtStep(callback_class):
        def on_step_end(self, args: Any, state: Any, control: Any, **_: Any) -> Any:
            if state.global_step >= optimizer_steps:
                control.should_training_stop = True
            return control

    return StopAtStep()


def _move_model_inputs(batch: dict[str, Any], device: Any) -> tuple[dict[str, Any], Any, Any]:
    decision_positions = batch.pop("decision_positions").to(device)
    decision_labels = batch.pop("decision_labels").to(device)
    return (
        {name: value.to(device) for name, value in batch.items()},
        decision_positions,
        decision_labels,
    )


def initial_logits_sha256(
    model: Any,
    examples: list[dict[str, Any]],
    collator: DecisionCollator,
    torch_module: Any,
    count: int = 8,
) -> str:
    torch = torch_module
    device = next(model.parameters()).device
    digest = hashlib.sha256()
    model.eval()
    with torch.no_grad():
        for example in examples[:count]:
            inputs, positions, _ = _move_model_inputs(collator([example]), device)
            outputs = model(**inputs)
            boundary = outputs.logits[0, int(positions[0].item()) - 1].float().cpu()
            digest.update(boundary.numpy().tobytes())
    return digest.hexdigest()


def evaluate_teacher_forcing(
    model: Any,
    examples: list[dict[str, Any]],
    collator: DecisionCollator,
    contract: DecisionContract,
    call_weight: float,
    no_call_weight: float,
    torch_module: Any,
) -> dict[str, Any]:
    torch = torch_module
    device = next(model.parameters()).device
    losses: list[float] = []
    auxiliary: list[float] = []
    totals = [0, 0]
    correct = [0, 0]
    model.eval()
    with torch.no_grad():
        for example in examples:
            inputs, positions, labels = _move_model_inputs(collator([example]), device)
            outputs = model(**inputs)
            losses.append(float(outputs.loss.detach().float().cpu()))
            decision = torch_decision_loss(
                logits=outputs.logits,
                decision_positions=positions,
                decision_labels=labels,
                call_token_id=contract.call_token_id,
                no_call_token_id=contract.no_call_token_id,
                call_weight=call_weight,
                no_call_weight=no_call_weight,
                torch_module=torch,
            )
            auxiliary.append(float(decision.detach().float().cpu()))
            prediction_position = int(positions[0].item()) - 1
            candidates = outputs.logits[0, prediction_position][
                [contract.call_token_id, contract.no_call_token_id]
            ]
            label = int(labels[0].item())
            totals[label] += 1
            correct[label] += int(int(candidates.argmax().item()) == label)
    call_accuracy = correct[CALL_LABEL] / totals[CALL_LABEL]
    no_call_accuracy = correct[NO_CALL_LABEL] / totals[NO_CALL_LABEL]
    return {
        "examples": len(examples),
        "languageModelLoss": sum(losses) / len(losses),
        "applicabilityLoss": sum(auxiliary) / len(auxiliary),
        "call": {"correct": correct[CALL_LABEL], "total": totals[CALL_LABEL], "accuracy": call_accuracy},
        "noCall": {
            "correct": correct[NO_CALL_LABEL],
            "total": totals[NO_CALL_LABEL],
            "accuracy": no_call_accuracy,
        },
        "balancedAccuracy": (call_accuracy + no_call_accuracy) / 2,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--candidate", choices=sorted(CANDIDATES), default="qwen3-1.7b")
    parser.add_argument("--initial-adapter", required=True, type=Path)
    parser.add_argument("--initial-training-manifest", required=True, type=Path)
    parser.add_argument("--applicability-loss-weight", required=True, type=float)
    parser.add_argument("--screen-steps", type=int)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--alpha", type=int, default=64)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260918)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"output already exists: {args.out}")
    if args.candidate != "qwen3-1.7b":
        parser.error("V13 is frozen to qwen3-1.7b")
    if args.applicability_loss_weight not in {0.0, 0.1}:
        parser.error("V13 permits only the control coefficient 0.0 or treatment coefficient 0.1")
    if args.max_length != 1024 or args.rank != 32 or args.alpha != 64:
        parser.error("V13 is frozen to max-length 1024, rank 32, and alpha 64")
    if args.epochs != 1.0 or args.learning_rate != 2e-5:
        parser.error("V13 is frozen to one epoch and learning rate 2e-5")
    if args.gradient_accumulation != 16 or args.seed != 20260918:
        parser.error("V13 is frozen to gradient accumulation 16 and seed 20260918")

    prepared_manifest = verify_prepared_splits(args.prepared)
    model_id, model_revision = resolve_candidate(args.candidate)
    initial_adapter = resolve_initial_adapter(
        args.initial_adapter,
        args.initial_training_manifest,
        model_id,
        model_revision,
        args.rank,
        args.alpha,
    )

    import accelerate
    import peft
    import safetensors
    import torch
    import transformers
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainerCallback, TrainingArguments

    if not torch.cuda.is_available():
        raise RuntimeError("V13 training requires CUDA")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=model_revision)
    actual_invocation = tokenizer.encode(INVOCATION_STRING, add_special_tokens=False)
    if actual_invocation != INVOCATION_TOKENS:
        raise ValueError(
            f"invocation tokens changed: expected {INVOCATION_TOKENS}, got {actual_invocation}"
        )
    contract = decision_contract(tokenizer)
    train_examples, validation_examples, train_report, validation_report = verify_v13_corpus(
        args.prepared, tokenizer, args.max_length
    )
    call_weight, no_call_weight = class_weights(
        train_report["call"], train_report["noCall"]
    )

    collator = DecisionCollator(tokenizer.pad_token_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=model_revision,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    model.to(torch.device("cuda"))
    base_logits = initial_logits_sha256(model, validation_examples, collator, torch)
    model = PeftModel.from_pretrained(model, args.initial_adapter, is_trainable=True)
    model.config.use_cache = False
    model.to(torch.device("cuda"))
    with model.disable_adapter():
        initial_disabled_logits = initial_logits_sha256(
            model, validation_examples, collator, torch
        )
    if initial_disabled_logits != base_logits:
        raise ValueError("adapter-disabled initial logits differ from the exact base")
    initial_logits = initial_logits_sha256(model, validation_examples, collator, torch)
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.print_trainable_parameters()

    args.out.mkdir(parents=True)
    training = TrainingArguments(
        output_dir=str(args.out / "checkpoints"),
        overwrite_output_dir=False,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation,
        num_train_epochs=args.epochs,
        max_steps=-1,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.0,
        optim="adamw_torch",
        bf16=True,
        tf32=False,
        logging_steps=10,
        save_strategy="no",
        eval_strategy="no",
        report_to="none",
        remove_unused_columns=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        seed=args.seed,
        data_seed=args.seed,
    )
    applicability_trainer = make_applicability_trainer(
        Trainer,
        contract=contract,
        call_weight=call_weight,
        no_call_weight=no_call_weight,
        coefficient=args.applicability_loss_weight,
        torch_module=torch,
    )
    callbacks = (
        [make_stop_callback(TrainerCallback, args.screen_steps)]
        if args.screen_steps is not None
        else []
    )
    trainer = applicability_trainer(
        model=model,
        args=training,
        train_dataset=DecisionDataset(train_examples),
        eval_dataset=DecisionDataset(validation_examples),
        data_collator=collator,
        callbacks=callbacks,
    )
    train_result = trainer.train()
    prepare_for_evaluation(model, True)
    evaluation = evaluate_teacher_forcing(
        model,
        validation_examples,
        collator,
        contract,
        call_weight,
        no_call_weight,
        torch,
    )
    final_logits = initial_logits_sha256(model, validation_examples, collator, torch)
    with model.disable_adapter():
        final_disabled_logits = initial_logits_sha256(
            model, validation_examples, collator, torch
        )
    if final_disabled_logits != base_logits:
        raise ValueError("adapter-disabled trained logits differ from the exact base")
    adapter = args.out / "adapter"
    model.save_pretrained(adapter, safe_serialization=True)
    tokenizer.save_pretrained(adapter)
    adapter_files = [
        {"name": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in sorted(adapter.iterdir())
        if path.is_file()
    ]
    finite_gradient_checks = trainer.finite_gradient_checks
    del trainer
    del model
    gc.collect()
    torch.cuda.empty_cache()
    reloaded_base = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=model_revision,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    reloaded = PeftModel.from_pretrained(reloaded_base, adapter, is_trainable=False)
    reloaded.config.use_cache = False
    reloaded.to(torch.device("cuda"))
    reloaded_logits = initial_logits_sha256(
        reloaded, validation_examples, collator, torch
    )
    if reloaded_logits != final_logits:
        raise ValueError("saved/reloaded adapter logits differ from the trained checkpoint")
    del reloaded
    gc.collect()
    torch.cuda.empty_cache()
    manifest = {
        "schemaVersion": 1,
        "kind": "activated-lora-tool-specialist",
        "experiment": "qwen3-17b-applicability-decision-v13",
        "arm": "control" if args.applicability_loss_weight == 0.0 else "treatment",
        "base": {"model": model_id, "revision": model_revision},
        "invocation": {"string": INVOCATION_STRING, "tokens": INVOCATION_TOKENS},
        "decision": {
            "sharedPrefixTokens": list(contract.shared_prefix_tokens),
            "callToken": contract.call_token_id,
            "noCallToken": contract.no_call_token_id,
            "callWeight": call_weight,
            "noCallWeight": no_call_weight,
            "coefficient": args.applicability_loss_weight,
        },
        "adapter": {
            "rank": args.rank,
            "alpha": args.alpha,
            "dropout": 0.0,
            "targetModules": TARGET_MODULES,
            "files": adapter_files,
        },
        "data": {
            "preparedManifestSha256": sha256(args.prepared / "manifest.json"),
            "preparedSchemaVersion": prepared_manifest["schemaVersion"],
            "train": train_report,
            "validation": validation_report,
        },
        "training": {
            "epochs": args.epochs,
            "screenSteps": args.screen_steps,
            "optimizerStepsCompleted": int(train_result.global_step),
            "learningRate": args.learning_rate,
            "gradientAccumulation": args.gradient_accumulation,
            "maxLength": args.max_length,
            "gradientCheckpointing": True,
            "seed": args.seed,
            "initialLogitsSha256": initial_logits,
            "exactBaseLogitsSha256": base_logits,
            "initialDisabledAdapterLogitsSha256": initial_disabled_logits,
            "finalLogitsSha256": final_logits,
            "finalDisabledAdapterLogitsSha256": final_disabled_logits,
            "reloadedLogitsSha256": reloaded_logits,
            "metrics": train_result.metrics,
            "finiteGradientChecks": finite_gradient_checks,
            "teacherForcedEvaluation": evaluation,
        },
        "continuedFrom": initial_adapter,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "transformers": transformers.__version__,
            "peft": peft.__version__,
            "accelerate": accelerate.__version__,
            "safetensors": safetensors.__version__,
        },
    }
    (args.out / "training-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
