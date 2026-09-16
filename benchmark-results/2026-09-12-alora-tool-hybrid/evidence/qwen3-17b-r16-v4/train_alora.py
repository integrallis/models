#!/usr/bin/env python3
"""Train the pinned Qwen3 tool aLoRA and emit a provenance-bound adapter."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
from pathlib import Path
from typing import Any


MODEL_ID = "Qwen/Qwen3-0.6B"
MODEL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
CANDIDATES = {
    "qwen3-0.6b": (MODEL_ID, MODEL_REVISION),
    "qwen3-1.7b": ("Qwen/Qwen3-1.7B", "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"),
}
INVOCATION_STRING = "<|im_start|>assistant\n"
INVOCATION_TOKENS = [151644, 77091, 198]
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def resolve_candidate(name: str) -> tuple[str, str]:
    try:
        return CANDIDATES[name]
    except KeyError as error:
        raise ValueError(f"unsupported training candidate: {name}") from error


def prepare_for_evaluation(model: Any, gradient_checkpointing: bool) -> None:
    if gradient_checkpointing:
        model.gradient_checkpointing_disable()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_prepared_splits(prepared: Path) -> dict[str, Any]:
    manifest_path = prepared / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read preparation manifest: {error}") from error
    if not isinstance(manifest, dict) or manifest.get("schemaVersion") not in {1, 2}:
        raise ValueError("unsupported preparation schemaVersion")
    for split in ("train", "validation"):
        path = prepared / f"{split}.jsonl"
        try:
            expected = manifest[split]["sha256"]
        except (KeyError, TypeError) as error:
            raise ValueError(f"preparation manifest has no {split} split hash") from error
        if not isinstance(expected, str) or len(expected) != 64:
            raise ValueError(f"preparation manifest has an invalid {split} split hash")
        try:
            actual = sha256(path)
        except OSError as error:
            raise ValueError(f"cannot read prepared {split} split: {error}") from error
        if actual != expected:
            raise ValueError(
                f"{split} split SHA-256 differs: expected {expected}, got {actual}"
            )
    return manifest


def last_subsequence(values: list[int], subsequence: list[int]) -> int:
    for start in range(len(values) - len(subsequence), -1, -1):
        if values[start : start + len(subsequence)] == subsequence:
            return start
    return -1


def build_example(record: dict[str, Any], tokenizer: Any, max_length: int) -> dict | None:
    prompt_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": record["user"]}],
        tools=record["tools"],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if hasattr(prompt_ids, "tolist"):
        prompt_ids = prompt_ids.tolist()
    completion_ids = tokenizer.encode(
        record["assistant"] + "<|im_end|>\n", add_special_tokens=False
    )
    input_ids = list(prompt_ids) + list(completion_ids)
    if len(input_ids) > max_length:
        return None
    invocation_start = last_subsequence(input_ids, INVOCATION_TOKENS)
    if invocation_start < 0 or invocation_start >= len(prompt_ids):
        raise ValueError(
            f"source line {record['sourceLine']} does not contain the pinned invocation boundary"
        )
    if last_subsequence(input_ids[:invocation_start], INVOCATION_TOKENS) >= 0:
        raise ValueError(f"source line {record['sourceLine']} contains multiple invocation boundaries")
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": [-100] * len(prompt_ids) + list(completion_ids),
        "source_line": record["sourceLine"],
        "invocation_start": invocation_start,
        "completion_tokens": len(completion_ids),
    }


def load_examples(path: Path, tokenizer: Any, max_length: int, limit: int | None) -> tuple[list[dict], dict]:
    examples: list[dict] = []
    skipped = 0
    no_call = 0
    multiple = 0
    with path.open() as source:
        for line in source:
            record = json.loads(line)
            example = build_example(record, tokenizer, max_length)
            if example is None:
                skipped += 1
                continue
            examples.append(example)
            no_call += not record["calls"]
            multiple += len(record["calls"]) > 1
            if limit is not None and len(examples) >= limit:
                break
    if not examples:
        raise ValueError(f"no usable examples in {path}")
    return examples, {
        "source": str(path),
        "sha256": sha256(path),
        "usable": len(examples),
        "skippedOverMaxLength": skipped,
        "noCall": no_call,
        "multipleCall": multiple,
        "sourceLinesSha256": hashlib.sha256(
            json.dumps([example["source_line"] for example in examples], separators=(",", ":")).encode()
        ).hexdigest(),
    }


class ToolDataset:
    def __init__(self, examples: list[dict]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict:
        example = self.examples[index]
        return {key: example[key] for key in ("input_ids", "attention_mask", "labels")}


class RightPaddingCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict]):
        import torch

        length = max(len(feature["input_ids"]) for feature in features)
        length = ((length + 7) // 8) * 8
        inputs = []
        masks = []
        labels = []
        for feature in features:
            padding = length - len(feature["input_ids"])
            inputs.append(feature["input_ids"] + [self.pad_token_id] * padding)
            masks.append(feature["attention_mask"] + [0] * padding)
            labels.append(feature["labels"] + [-100] * padding)
        return {
            "input_ids": torch.tensor(inputs, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--candidate", choices=sorted(CANDIDATES), default="qwen3-0.6b")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--alpha", type=int, default=64)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--validation-limit", type=int)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=20260912)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"output already exists: {args.out}")
    if args.max_length <= len(INVOCATION_TOKENS) or args.rank <= 0 or args.alpha <= 0:
        parser.error("max length, rank, and alpha must be positive")
    prepared_manifest = verify_prepared_splits(args.prepared)
    model_id, model_revision = resolve_candidate(args.candidate)

    import accelerate
    import peft
    import safetensors
    import torch
    import transformers
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

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
    train_examples, train_report = load_examples(
        args.prepared / "train.jsonl", tokenizer, args.max_length, args.train_limit
    )
    validation_examples, validation_report = load_examples(
        args.prepared / "validation.jsonl", tokenizer, args.max_length, args.validation_limit
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=model_revision,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    config = LoraConfig(
        task_type="CAUSAL_LM",
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=0.0,
        bias="none",
        target_modules=TARGET_MODULES,
        alora_invocation_tokens=INVOCATION_TOKENS,
    )
    model = get_peft_model(model, config)
    model.config.use_cache = False
    gradient_checkpointing = args.candidate == "qwen3-1.7b"
    if gradient_checkpointing:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    model.print_trainable_parameters()

    args.out.mkdir(parents=True)
    checkpoints = args.out / "checkpoints"
    training = TrainingArguments(
        output_dir=str(checkpoints),
        overwrite_output_dir=False,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
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
        gradient_checkpointing=gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        seed=args.seed,
        data_seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=training,
        train_dataset=ToolDataset(train_examples),
        eval_dataset=ToolDataset(validation_examples),
        data_collator=RightPaddingCollator(tokenizer.pad_token_id),
    )
    train_result = trainer.train()
    prepare_for_evaluation(model, gradient_checkpointing)
    evaluation = trainer.evaluate()
    adapter = args.out / "adapter"
    model.save_pretrained(adapter, safe_serialization=True)
    tokenizer.save_pretrained(adapter)

    adapter_files = []
    for path in sorted(adapter.iterdir()):
        if path.is_file():
            adapter_files.append({"name": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)})
    manifest = {
        "schemaVersion": 1,
        "kind": "activated-lora-tool-specialist",
        "base": {"model": model_id, "revision": model_revision},
        "invocation": {"string": INVOCATION_STRING, "tokens": INVOCATION_TOKENS},
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
            "maxSteps": args.max_steps,
            "learningRate": args.learning_rate,
            "gradientAccumulation": args.gradient_accumulation,
            "maxLength": args.max_length,
            "gradientCheckpointing": gradient_checkpointing,
            "seed": args.seed,
            "metrics": train_result.metrics,
            "evaluation": evaluation,
        },
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
