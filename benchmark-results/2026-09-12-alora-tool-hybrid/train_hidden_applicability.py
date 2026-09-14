#!/usr/bin/env python3
"""Extract and train the frozen V16 hidden-state applicability head."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import gc
import hashlib
import json
from pathlib import Path
import platform
from typing import Any, Iterable, Sequence

from train_alora import INVOCATION_STRING, INVOCATION_TOKENS, sha256
from train_applicability_alora import (
    CALL_LABEL,
    verify_v13_corpus,
)


EXPERIMENT = "qwen3-17b-hidden-applicability-v16"
TOOL_CALL_PREFIX = "<tool_call>\n"
TOOL_CALL_PREFIX_TOKENS = [151657, 198]
FOLD_SEED = EXPERIMENT + ":fold-v1"
MODEL_ID = "Qwen/Qwen3-1.7B"
MODEL_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
PRODUCTION_GGUF_SHA256 = (
    "061b54daade076b5d3362dac252678d17da8c68f07560be70818cace6590cb1a"
)
ADAPTER_SHA256 = "f1a4a1bc15e3942ed5a3fbe2c4750f39a8b24804e20566242ec740c5ce851c21"
TRAIN_SHA256 = "c46b606c74a3d1bda38c6f50a55854bf9112f0ae9d92993eb32e577c211f3002"
VALIDATION_SHA256 = "7006d89f1be432424889f8fbc846ff348cf2770a56f409c9e38361ae4857cfb0"
TRAIN_SOURCE_LINES_SHA256 = (
    "ceab03a876d41ada5e7a3f99cad4357e0b70a478b6f8c20f82da7aa0a1626941"
)
VALIDATION_SOURCE_LINES_SHA256 = (
    "1ce289514ff5ddbe72a34e240196925b8d341058e66278d627b477f5fc622f51"
)
HIDDEN_DIMENSION = 2048
MAX_LENGTH = 1024
FOLD_COUNT = 5
REGULARIZATION = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
TORCH_VERSION = "2.6.0+cu124"
TRANSFORMERS_VERSION = "4.53.3"
PEFT_VERSION = "0.18.1"


@dataclass(frozen=True)
class ClassificationScore:
    calls: int
    no_calls: int
    correct_calls: int
    correct_no_calls: int
    call_accuracy: float
    no_call_accuracy: float
    balanced_accuracy: float


def decision_feature_tokens(prompt_tokens: Iterable[int], tokenizer: Any) -> list[int]:
    """Append only the common decision prefix, excluding either class token."""
    suffix = list(tokenizer.encode(TOOL_CALL_PREFIX, add_special_tokens=False))
    if suffix != TOOL_CALL_PREFIX_TOKENS:
        raise ValueError(
            "decision prefix token IDs differ: "
            f"expected {TOOL_CALL_PREFIX_TOKENS}, got {suffix}"
        )
    return [*prompt_tokens, *suffix]


def stratified_folds(
    rows: Iterable[dict[str, int]], fold_count: int
) -> dict[tuple[int, int], int]:
    """Assign each class independently by frozen hash order and round robin."""
    if fold_count < 2:
        raise ValueError("fold_count must be at least two")
    by_label: dict[int, list[int]] = {}
    for row in rows:
        source_line = int(row["source_line"])
        label = int(row["label"])
        by_label.setdefault(label, []).append(source_line)
    if len(by_label) != 2:
        raise ValueError("exactly two applicability labels are required")

    result: dict[tuple[int, int], int] = {}
    for label, source_lines in sorted(by_label.items()):
        if len(source_lines) < fold_count:
            raise ValueError("each applicability class must populate every fold")
        if len(set(source_lines)) != len(source_lines):
            raise ValueError("source lines must be unique within each label")
        ordered = sorted(
            source_lines,
            key=lambda source_line: (
                hashlib.sha256(
                    f"{FOLD_SEED}\0{source_line}".encode()
                ).hexdigest(),
                source_line,
            ),
        )
        for ordinal, source_line in enumerate(ordered):
            result[source_line, label] = ordinal % fold_count
    return result


def balanced_score(
    labels: Iterable[int], scores: Iterable[float]
) -> ClassificationScore:
    """Score calls as strictly positive and no-calls as zero or negative."""
    pairs = list(zip(labels, scores, strict=True))
    calls = sum(label == 1 for label, _ in pairs)
    no_calls = sum(label == 0 for label, _ in pairs)
    if calls == 0 or no_calls == 0:
        raise ValueError("both applicability classes are required")
    correct_calls = sum(label == 1 and score > 0.0 for label, score in pairs)
    correct_no_calls = sum(label == 0 and score <= 0.0 for label, score in pairs)
    call_accuracy = correct_calls / calls
    no_call_accuracy = correct_no_calls / no_calls
    return ClassificationScore(
        calls,
        no_calls,
        correct_calls,
        correct_no_calls,
        call_accuracy,
        no_call_accuracy,
        (call_accuracy + no_call_accuracy) / 2.0,
    )


def select_regularization(results: dict[float, ClassificationScore]) -> float:
    """Apply the frozen cross-validation tie order."""
    if not results:
        raise ValueError("regularization results must not be empty")
    return max(
        results,
        key=lambda value: (
            results[value].balanced_accuracy,
            results[value].correct_no_calls,
            results[value].correct_calls,
            value,
        ),
    )


def artifact_sha256(artifact: dict[str, Any]) -> str:
    """Fingerprint canonical JSON without recursively including the fingerprint field."""
    canonical = {key: value for key, value in artifact.items() if key != "artifactSha256"}
    payload = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _source_lines_sha256(source_lines: Any) -> str:
    payload = json.dumps(source_lines.tolist(), separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def validation_passes(score: ClassificationScore) -> bool:
    """Apply the untouched V16 validation floors exactly."""
    return (
        score.calls == 390
        and score.no_calls == 81
        and score.correct_calls >= 371
        and score.correct_no_calls >= 77
        and score.balanced_accuracy > 0.95
    )


def _require_version(name: str, actual: str, expected: str) -> None:
    if actual != expected:
        raise RuntimeError(f"V16 requires {name} {expected}; got {actual}")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n"
    )


def _feature_metadata(
    split: str,
    report: dict[str, Any],
    feature_path: Path,
    environment: dict[str, Any],
) -> dict[str, Any]:
    return {
        "split": split,
        "rows": report["usable"],
        "calls": report["call"],
        "noCalls": report["noCall"],
        "sourceSha256": report["sha256"],
        "sourceLinesSha256": report["sourceLinesSha256"],
        "featureFile": feature_path.name,
        "featureSha256": sha256(feature_path),
        "environment": environment,
    }


def _extract_split(
    examples: Sequence[dict[str, Any]],
    model: Any,
    tokenizer: Any,
    torch_module: Any,
    split: str,
) -> tuple[Any, Any, Any]:
    features = torch_module.empty(
        (len(examples), HIDDEN_DIMENSION), dtype=torch_module.float32
    )
    labels = torch_module.empty(len(examples), dtype=torch_module.int64)
    source_lines = torch_module.empty(len(examples), dtype=torch_module.int64)
    device = torch_module.device("cuda")
    for index, example in enumerate(examples):
        decision_position = int(example["decision_position"])
        input_ids = decision_feature_tokens(
            example["input_ids"][: decision_position - len(TOOL_CALL_PREFIX_TOKENS)],
            tokenizer,
        )
        if len(input_ids) != decision_position:
            raise ValueError(
                f"{split} source line {example['source_line']} has a changed decision boundary"
            )
        tokens = torch_module.tensor([input_ids], dtype=torch_module.long, device=device)
        mask = torch_module.ones_like(tokens)
        with torch_module.inference_mode():
            output = model(
                input_ids=tokens,
                attention_mask=mask,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )
        hidden = output.hidden_states[-1][0, -1].detach().float().cpu()
        if hidden.numel() != HIDDEN_DIMENSION or not bool(torch_module.isfinite(hidden).all()):
            raise ValueError(
                f"{split} source line {example['source_line']} produced an invalid hidden state"
            )
        features[index].copy_(hidden)
        labels[index] = 1 if example["decision_label"] == CALL_LABEL else 0
        source_lines[index] = int(example["source_line"])
        if (index + 1) % 25 == 0 or index + 1 == len(examples):
            print(f"{split}: {index + 1}/{len(examples)}", flush=True)
        del output, hidden, tokens, mask
    return features.contiguous(), labels.contiguous(), source_lines.contiguous()


def extract_features(args: argparse.Namespace) -> None:
    if args.out.exists():
        raise ValueError(f"output already exists: {args.out}")
    adapter_model = args.adapter / "adapter_model.safetensors"
    if not adapter_model.is_file():
        raise ValueError(f"adapter weights do not exist: {adapter_model}")
    if sha256(adapter_model) != ADAPTER_SHA256:
        raise ValueError(f"V16 requires the frozen V9 adapter: {sha256(adapter_model)}")

    import peft
    import safetensors
    import torch
    import transformers
    from peft import PeftModel
    from safetensors.torch import save_file
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _require_version("Torch", torch.__version__, TORCH_VERSION)
    _require_version("Transformers", transformers.__version__, TRANSFORMERS_VERSION)
    _require_version("PEFT", peft.__version__, PEFT_VERSION)
    if not torch.cuda.is_available():
        raise RuntimeError("V16 feature extraction requires CUDA")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)

    tokenizer = AutoTokenizer.from_pretrained(args.adapter)
    if tokenizer.encode(INVOCATION_STRING, add_special_tokens=False) != INVOCATION_TOKENS:
        raise ValueError("activated adapter invocation tokens changed")
    decision_feature_tokens([], tokenizer)
    train, validation, train_report, validation_report = verify_v13_corpus(
        args.prepared, tokenizer, MAX_LENGTH
    )
    if (
        train_report["sha256"] != TRAIN_SHA256
        or validation_report["sha256"] != VALIDATION_SHA256
        or train_report["sourceLinesSha256"] != TRAIN_SOURCE_LINES_SHA256
        or validation_report["sourceLinesSha256"] != VALIDATION_SOURCE_LINES_SHA256
    ):
        raise ValueError("V16 prepared corpus identity changed")

    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(base, args.adapter, is_trainable=False)
    model.config.use_cache = False
    model.eval()
    model.to(torch.device("cuda"))
    active = model.peft_config[model.active_adapter]
    if list(active.alora_invocation_tokens or []) != INVOCATION_TOKENS:
        raise ValueError("loaded PEFT adapter has a changed aLoRA invocation contract")

    environment = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "safetensors": safetensors.__version__,
        "tf32": False,
        "dtype": "bfloat16",
        "attention": "sdpa",
        "batchSize": 1,
    }
    args.out.mkdir(parents=True)
    split_metadata: dict[str, Any] = {}
    for split, examples, report in (
        ("train", train, train_report),
        ("validation", validation, validation_report),
    ):
        features, labels, source_lines = _extract_split(
            examples, model, tokenizer, torch, split
        )
        feature_path = args.out / f"{split}-features.safetensors"
        save_file(
            {
                "features": features,
                "labels": labels,
                "source_lines": source_lines,
            },
            feature_path,
            metadata={
                "experiment": EXPERIMENT,
                "split": split,
                "source_sha256": report["sha256"],
                "source_lines_sha256": report["sourceLinesSha256"],
            },
        )
        split_metadata[split] = _feature_metadata(
            split, report, feature_path, environment
        )
    manifest = {
        "schemaVersion": 1,
        "experiment": EXPERIMENT,
        "base": {"model": MODEL_ID, "revision": MODEL_REVISION},
        "productionGgufSha256": PRODUCTION_GGUF_SHA256,
        "adapterSha256": ADAPTER_SHA256,
        "representation": {
            "hiddenDimension": HIDDEN_DIMENSION,
            "maxLength": MAX_LENGTH,
            "decisionPrefix": TOOL_CALL_PREFIX,
            "decisionPrefixTokens": TOOL_CALL_PREFIX_TOKENS,
            "state": "final-rms-normalized-activated-hidden",
        },
        "splits": split_metadata,
    }
    _write_json(args.out / "features-manifest.json", manifest)
    print(json.dumps(manifest, sort_keys=True), flush=True)
    del model, base
    gc.collect()
    torch.cuda.empty_cache()


def _load_feature_split(directory: Path, split: str, torch_module: Any) -> tuple[Any, Any, Any]:
    from safetensors import safe_open
    from safetensors.torch import load_file

    path = directory / f"{split}-features.safetensors"
    if not path.is_file():
        raise ValueError(f"missing {split} feature file: {path}")
    tensors = load_file(path, device="cpu")
    expected_keys = {"features", "labels", "source_lines"}
    if set(tensors) != expected_keys:
        raise ValueError(f"{split} feature tensors changed: {sorted(tensors)}")
    features = tensors["features"]
    labels = tensors["labels"]
    source_lines = tensors["source_lines"]
    if features.dtype != torch_module.float32 or features.shape[1:] != (HIDDEN_DIMENSION,):
        raise ValueError(f"{split} hidden feature shape or dtype changed")
    if labels.dtype != torch_module.int64 or source_lines.dtype != torch_module.int64:
        raise ValueError(f"{split} label/source-line dtype changed")
    if labels.shape != source_lines.shape or labels.shape != (features.shape[0],):
        raise ValueError(f"{split} feature row shapes differ")
    if not bool(torch_module.isfinite(features).all()):
        raise ValueError(f"{split} features are not finite")
    if set(labels.tolist()) != {0, 1}:
        raise ValueError(f"{split} labels are not binary")
    with safe_open(path, framework="pt", device="cpu") as source:
        metadata = source.metadata()
    if (
        metadata.get("experiment") != EXPERIMENT
        or metadata.get("split") != split
    ):
        raise ValueError(f"{split} feature metadata changed")
    return features, labels, source_lines


def _normalization(features: Any, torch_module: Any) -> tuple[Any, Any]:
    mean = features.mean(dim=0)
    scale = ((features - mean).square().mean(dim=0)).sqrt().clamp_min(1e-6)
    if not bool(torch_module.isfinite(mean).all() and torch_module.isfinite(scale).all()):
        raise ValueError("feature normalization is not finite")
    return mean, scale


def _fit_head(features: Any, labels: Any, regularization: float, torch_module: Any) -> tuple[Any, Any, Any, Any]:
    features = features.to(dtype=torch_module.float64)
    labels = labels.to(dtype=torch_module.float64)
    mean, scale = _normalization(features, torch_module)
    normalized = (features - mean) / scale
    weight = torch_module.zeros(
        HIDDEN_DIMENSION, dtype=torch_module.float64, requires_grad=True
    )
    bias = torch_module.zeros((), dtype=torch_module.float64, requires_grad=True)
    calls = int(labels.sum().item())
    no_calls = labels.numel() - calls
    if calls == 0 or no_calls == 0:
        raise ValueError("head training requires both applicability classes")
    call_weight = labels.numel() / (2.0 * calls)
    no_call_weight = labels.numel() / (2.0 * no_calls)
    sample_weight = torch_module.where(labels == 1.0, call_weight, no_call_weight)
    optimizer = torch_module.optim.LBFGS(
        [weight, bias],
        lr=1.0,
        max_iter=200,
        max_eval=250,
        tolerance_grad=1e-10,
        tolerance_change=1e-12,
        history_size=20,
        line_search_fn="strong_wolfe",
    )

    def closure() -> Any:
        optimizer.zero_grad(set_to_none=True)
        logits = normalized.mv(weight) + bias
        losses = torch_module.nn.functional.binary_cross_entropy_with_logits(
            logits, labels, reduction="none"
        )
        loss = (losses * sample_weight).mean() + regularization * weight.square().sum()
        if not bool(torch_module.isfinite(loss)):
            raise ValueError("applicability-head loss is not finite")
        loss.backward()
        return loss

    optimizer.step(closure)
    if not bool(torch_module.isfinite(weight).all() and torch_module.isfinite(bias)):
        raise ValueError("applicability-head parameters are not finite")
    return mean.detach(), scale.detach(), weight.detach(), bias.detach()


def _scores(features: Any, parameters: tuple[Any, Any, Any, Any], torch_module: Any) -> Any:
    mean, scale, weight, bias = parameters
    return ((features.to(torch_module.float64) - mean) / scale).mv(weight) + bias


def _score_tensor(labels: Any, scores: Any) -> ClassificationScore:
    return balanced_score(labels.tolist(), scores.tolist())


def _cross_validate(features: Any, labels: Any, source_lines: Any, torch_module: Any) -> tuple[float, dict[float, ClassificationScore]]:
    rows = [
        {"source_line": int(source_line), "label": int(label)}
        for source_line, label in zip(source_lines.tolist(), labels.tolist(), strict=True)
    ]
    assignments = stratified_folds(rows, FOLD_COUNT)
    results: dict[float, ClassificationScore] = {}
    for regularization in REGULARIZATION:
        out_of_fold = torch_module.empty(labels.numel(), dtype=torch_module.float64)
        for fold in range(FOLD_COUNT):
            held_out = torch_module.tensor(
                [
                    assignments[int(source_line), int(label)] == fold
                    for source_line, label in zip(
                        source_lines.tolist(), labels.tolist(), strict=True
                    )
                ],
                dtype=torch_module.bool,
            )
            parameters = _fit_head(
                features[~held_out], labels[~held_out], regularization, torch_module
            )
            out_of_fold[held_out] = _scores(features[held_out], parameters, torch_module)
        results[regularization] = _score_tensor(labels, out_of_fold)
        print(
            f"lambda={regularization:g} balanced={results[regularization].balanced_accuracy:.6f} "
            f"calls={results[regularization].correct_calls}/{results[regularization].calls} "
            f"no-calls={results[regularization].correct_no_calls}/{results[regularization].no_calls}",
            flush=True,
        )
    return select_regularization(results), results


def _score_as_serialized(features: Any, artifact: dict[str, Any], torch_module: Any) -> Any:
    mean = torch_module.tensor(artifact["normalization"]["mean"], dtype=torch_module.float32)
    scale = torch_module.tensor(artifact["normalization"]["scale"], dtype=torch_module.float32)
    weight = torch_module.tensor(artifact["classifier"]["weight"], dtype=torch_module.float32)
    bias = torch_module.tensor(artifact["classifier"]["bias"], dtype=torch_module.float32)
    products = ((features - mean) / scale) * weight
    return products.to(torch_module.float64).sum(dim=1) + bias.to(torch_module.float64)


def _score_json(score: ClassificationScore) -> dict[str, Any]:
    return {
        "calls": score.calls,
        "noCalls": score.no_calls,
        "correctCalls": score.correct_calls,
        "correctNoCalls": score.correct_no_calls,
        "callAccuracy": score.call_accuracy,
        "noCallAccuracy": score.no_call_accuracy,
        "balancedAccuracy": score.balanced_accuracy,
    }


def fit_head(args: argparse.Namespace) -> int:
    import safetensors
    import torch

    _require_version("Torch", torch.__version__, TORCH_VERSION)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    if args.out.exists():
        raise ValueError(f"output already exists: {args.out}")
    manifest_path = args.features / "features-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("schemaVersion") != 1
        or manifest.get("experiment") != EXPERIMENT
        or manifest.get("base") != {"model": MODEL_ID, "revision": MODEL_REVISION}
        or manifest.get("productionGgufSha256") != PRODUCTION_GGUF_SHA256
        or manifest.get("adapterSha256") != ADAPTER_SHA256
    ):
        raise ValueError("V16 feature manifest identity changed")
    train, train_labels, train_source_lines = _load_feature_split(
        args.features, "train", torch
    )
    validation, validation_labels, validation_source_lines = _load_feature_split(
        args.features, "validation", torch
    )
    expected = {
        "train": (2840, 2356, 484, TRAIN_SHA256, TRAIN_SOURCE_LINES_SHA256),
        "validation": (
            471,
            390,
            81,
            VALIDATION_SHA256,
            VALIDATION_SOURCE_LINES_SHA256,
        ),
    }
    for split, tensors, source_lines in (
        ("train", (train, train_labels), train_source_lines),
        ("validation", (validation, validation_labels), validation_source_lines),
    ):
        rows, calls, no_calls, source_sha, lines_sha = expected[split]
        metadata = manifest["splits"][split]
        path = args.features / metadata["featureFile"]
        if (
            tensors[0].shape[0] != rows
            or int(tensors[1].sum().item()) != calls
            or rows - int(tensors[1].sum().item()) != no_calls
            or metadata["sourceSha256"] != source_sha
            or metadata["sourceLinesSha256"] != lines_sha
            or _source_lines_sha256(source_lines) != lines_sha
            or metadata["featureFile"] != f"{split}-features.safetensors"
            or metadata["featureSha256"] != sha256(path)
        ):
            raise ValueError(f"V16 {split} feature identity changed")

    selected, cross_validation = _cross_validate(
        train, train_labels, train_source_lines, torch
    )
    parameters = _fit_head(train, train_labels, selected, torch)
    mean, scale, weight, bias = parameters
    mean32 = mean.to(torch.float32)
    scale32 = scale.to(torch.float32)
    weight32 = weight.to(torch.float32)
    bias32 = bias.to(torch.float32)
    if not bool(
        torch.isfinite(mean32).all()
        and torch.isfinite(scale32).all()
        and torch.isfinite(weight32).all()
        and torch.isfinite(bias32)
    ):
        raise ValueError("serialized applicability-head parameters are not finite")
    artifact: dict[str, Any] = {
        "schemaVersion": 1,
        "kind": "activated-hidden-applicability-head",
        "experiment": EXPERIMENT,
        "base": {
            "model": MODEL_ID,
            "revision": MODEL_REVISION,
            "productionGgufSha256": PRODUCTION_GGUF_SHA256,
        },
        "adapterSha256": ADAPTER_SHA256,
        "featuresManifestSha256": sha256(manifest_path),
        "representation": manifest["representation"],
        "training": {
            "folds": FOLD_COUNT,
            "foldSeed": FOLD_SEED,
            "regularizationCandidates": list(REGULARIZATION),
            "selectedRegularization": selected,
            "optimizer": {
                "name": "lbfgs",
                "lineSearch": "strong_wolfe",
                "historySize": 20,
                "maxIterations": 200,
                "gradientTolerance": 1e-10,
                "changeTolerance": 1e-12,
                "precision": "float64",
                "threads": 1,
            },
            "crossValidation": {
                str(value): _score_json(cross_validation[value])
                for value in REGULARIZATION
            },
        },
        "normalization": {
            "minimumScale": 1e-6,
            "mean": mean32.tolist(),
            "scale": scale32.tolist(),
        },
        "classifier": {
            "weight": weight32.tolist(),
            "bias": float(bias32.item()),
            "threshold": 0.0,
            "decision": "call iff score > threshold",
            "accumulation": "float32 intermediates, float64 ordered sum",
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "safetensors": safetensors.__version__,
        },
    }
    validation_scores = _score_as_serialized(validation, artifact, torch)
    validation_score = _score_tensor(validation_labels, validation_scores)
    artifact["validation"] = _score_json(validation_score)
    artifact["validationPassed"] = validation_passes(validation_score)
    artifact["artifactSha256"] = artifact_sha256(artifact)
    args.out.mkdir(parents=True)
    artifact_path = args.out / "applicability-head.json"
    _write_json(artifact_path, artifact)
    reloaded = json.loads(artifact_path.read_text())
    if reloaded["artifactSha256"] != artifact_sha256(reloaded):
        raise ValueError("saved applicability-head fingerprint changed after reload")
    reloaded_scores = _score_as_serialized(validation, reloaded, torch)
    if not bool(torch.equal(validation_scores, reloaded_scores)):
        raise ValueError("saved applicability-head scores changed after reload")
    report = {
        "schemaVersion": 1,
        "experiment": EXPERIMENT,
        "artifact": artifact_path.name,
        "artifactFileSha256": sha256(artifact_path),
        "artifactCanonicalSha256": artifact["artifactSha256"],
        "selectedRegularization": selected,
        "validation": _score_json(validation_score),
        "validationPassed": artifact["validationPassed"],
        "verdict": "PASS" if artifact["validationPassed"] else "FAIL",
    }
    _write_json(args.out / "result.json", report)
    print(json.dumps(report, sort_keys=True), flush=True)
    return 0 if artifact["validationPassed"] else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    extract = commands.add_parser("extract", help="extract frozen BF16 activated hidden states")
    extract.add_argument("--prepared", required=True, type=Path)
    extract.add_argument("--adapter", required=True, type=Path)
    extract.add_argument("--out", required=True, type=Path)
    fit = commands.add_parser("fit", help="fit and validate the frozen affine head")
    fit.add_argument("--features", required=True, type=Path)
    fit.add_argument("--out", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "extract":
        extract_features(args)
        return
    raise SystemExit(fit_head(args))


if __name__ == "__main__":
    main()
