#!/usr/bin/env python3
"""Evaluate a head-local KV mapper against pinned Hugging Face model forwards.

This oracle test verifies that a mapped prefix can actually continue in the target model. KV R2
alone is never treated as qualification evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, DynamicCache

from cross_model_kv.rope import RopeSpec, apply_rope, invert_rope


def map_content_cache(
    source_keys: torch.Tensor,
    source_values: torch.Tensor,
    selection: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map content-space tensors from ``[source-layer, token, head, dim]``."""
    if source_keys.shape != source_values.shape or source_keys.ndim != 4:
        raise ValueError("source K/V must be equal rank-four tensors")
    target_layers, top_k = selection.shape
    source_layers, tokens, heads, dimension = source_keys.shape
    head_local_weight = (target_layers, 2, heads, top_k * dimension, dimension)
    head_local_bias = (target_layers, 2, heads, dimension)
    full_dimensions = heads * dimension
    full_weight = (target_layers, 2, full_dimensions, top_k * full_dimensions)
    full_bias = (target_layers, 2, full_dimensions)
    head_local = tuple(weight.shape) == head_local_weight and tuple(bias.shape) == head_local_bias
    full_cross_head = tuple(weight.shape) == full_weight and tuple(bias.shape) == full_bias
    if not head_local and not full_cross_head:
        raise ValueError(
            f"mapper shape differs: weight={tuple(weight.shape)}, bias={tuple(bias.shape)}, "
            f"expected head-local {head_local_weight}/{head_local_bias} or "
            f"full-cross-head {full_weight}/{full_bias}"
        )
    if selection.min() < 0 or selection.max() >= source_layers:
        raise ValueError("mapper selection contains an invalid source layer")
    mapped = []
    for kind, source in enumerate((source_keys, source_values)):
        target = torch.empty(
            (target_layers, tokens, heads, dimension),
            device=source.device,
            dtype=torch.float32,
        )
        for target_layer in range(target_layers):
            picked = source.index_select(0, selection[target_layer].to(source.device))
            if full_cross_head:
                inputs = picked.permute(1, 0, 2, 3).reshape(tokens, -1).float()
                output = torch.nn.functional.linear(
                    inputs,
                    weight[target_layer, kind].float(),
                    bias[target_layer, kind].float(),
                )
                target[target_layer] = output.reshape(tokens, heads, dimension)
            else:
                for head in range(heads):
                    inputs = picked[:, :, head].permute(1, 0, 2).reshape(tokens, -1).float()
                    target[target_layer, :, head] = torch.addmm(
                        bias[target_layer, kind, head].float(),
                        inputs,
                        weight[target_layer, kind, head].float(),
                    )
        mapped.append(target)
    return mapped[0], mapped[1]


def _canonical_content(past_key_values, rope: RopeSpec) -> tuple[torch.Tensor, torch.Tensor]:
    legacy = (
        past_key_values.to_legacy_cache()
        if hasattr(past_key_values, "to_legacy_cache")
        else past_key_values
    )
    positions = torch.arange(legacy[0][0].shape[-2], device=legacy[0][0].device)
    keys = []
    values = []
    for key, value in legacy:
        canonical_key = key[0].transpose(0, 1)
        keys.append(invert_rope(canonical_key, positions, rope).to(torch.float32))
        values.append(value[0].transpose(0, 1).to(torch.float32))
    return torch.stack(keys), torch.stack(values)


@torch.no_grad()
def _mapped_target_cache(
    source_cache: DynamicCache,
    source_rope: RopeSpec,
    selection: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    target_rope: RopeSpec,
    *,
    source_token_from: int = 0,
    token_chunk: int = 128,
    prefix_cache: DynamicCache | None = None,
) -> DynamicCache:
    """Translate source cache chunks directly into bounded-memory BF16 target cache blocks."""
    if token_chunk < 1:
        raise ValueError("token_chunk must be positive")
    source = source_cache.to_legacy_cache()
    if not source:
        raise ValueError("source cache must contain at least one layer")
    if any(key.shape[0] != 1 or value.shape != key.shape for key, value in source):
        raise ValueError("source cache must contain equal K/V tensors for one batch")
    source_layers = len(source)
    tokens = source[0][0].shape[-2]
    heads = source[0][0].shape[-3]
    dimension = source[0][0].shape[-1]
    if any(
        key.shape[-3:] != (heads, tokens, dimension) or value.shape != key.shape
        for key, value in source
    ):
        raise ValueError("source cache layer shapes differ")
    if source_token_from < 0 or source_token_from >= tokens:
        raise ValueError("source_token_from must leave a nonempty mapped middle")
    target_layers, top_k = selection.shape
    if int(selection.min()) < 0 or int(selection.max()) >= source_layers:
        raise ValueError("mapper selection contains an invalid source layer")
    head_local = tuple(weight.shape) == (
        target_layers,
        2,
        heads,
        top_k * dimension,
        dimension,
    ) and tuple(bias.shape) == (target_layers, 2, heads, dimension)
    full_dimensions = heads * dimension
    full_cross_head = tuple(weight.shape) == (
        target_layers,
        2,
        full_dimensions,
        top_k * full_dimensions,
    ) and tuple(bias.shape) == (target_layers, 2, full_dimensions)
    if not head_local and not full_cross_head:
        raise ValueError("mapper shape differs from source cache geometry")
    prefix = None if prefix_cache is None else prefix_cache.to_legacy_cache()
    if prefix is None and source_token_from != 0:
        raise ValueError("a mapped middle with an offset requires a native prefix cache")
    if prefix is not None:
        if len(prefix) != target_layers:
            raise ValueError("native prefix and mapped middle layer counts differ")
        if any(key.shape[-2] != source_token_from for key, _ in prefix):
            raise ValueError("native prefix length differs from source_token_from")

    legacy = []
    for target_layer in range(target_layers):
        picks = [int(value) for value in selection[target_layer]]
        output_chunks = [[], []]
        for offset in range(source_token_from, tokens, token_chunk):
            stop = min(tokens, offset + token_chunk)
            positions = torch.arange(offset, stop, device=weight.device)
            for kind in (0, 1):
                selected = []
                for source_layer in picks:
                    raw = source[source_layer][kind][0, :, offset:stop, :].transpose(0, 1)
                    selected.append(
                        invert_rope(raw, positions, source_rope).float()
                        if kind == 0
                        else raw.float()
                    )
                gathered = torch.stack(selected)
                if full_cross_head:
                    inputs = gathered.permute(1, 0, 2, 3).reshape(stop - offset, -1)
                    mapped = torch.nn.functional.linear(
                        inputs,
                        weight[target_layer, kind].float(),
                        bias[target_layer, kind].float(),
                    ).reshape(stop - offset, heads, dimension)
                else:
                    mapped = torch.empty(
                        (stop - offset, heads, dimension),
                        device=weight.device,
                        dtype=torch.float32,
                    )
                    for head in range(heads):
                        inputs = gathered[:, :, head].permute(1, 0, 2).reshape(
                            stop - offset, -1
                        )
                        mapped[:, head] = torch.addmm(
                            bias[target_layer, kind, head].float(),
                            inputs,
                            weight[target_layer, kind, head].float(),
                        )
                if kind == 0:
                    mapped = apply_rope(mapped, positions, target_rope)
                output_chunks[kind].append(mapped.to(torch.bfloat16))
        key = torch.cat(output_chunks[0]).transpose(0, 1).unsqueeze(0)
        value = torch.cat(output_chunks[1]).transpose(0, 1).unsqueeze(0)
        if prefix is not None:
            key = torch.cat((prefix[target_layer][0], key), dim=-2)
            value = torch.cat((prefix[target_layer][1], value), dim=-2)
        legacy.append((key, value))
    return DynamicCache.from_legacy_cache(tuple(legacy))


def _target_cache(
    content_keys: torch.Tensor,
    values: torch.Tensor,
    rope: RopeSpec,
    *,
    start_position: int = 0,
    prefix_cache: DynamicCache | None = None,
) -> DynamicCache:
    if content_keys.shape != values.shape or content_keys.ndim != 4:
        raise ValueError("target content K/V must be equal rank-four tensors")
    if start_position < 0:
        raise ValueError("start_position must not be negative")
    prefix = None if prefix_cache is None else prefix_cache.to_legacy_cache()
    if prefix is None:
        if start_position != 0:
            raise ValueError("a nonzero start_position requires a native prefix cache")
    else:
        if len(prefix) != content_keys.shape[0]:
            raise ValueError("native prefix and mapped middle layer counts differ")
        if any(key.shape[-2] != start_position for key, _ in prefix):
            raise ValueError("native prefix length differs from start_position")
    positions = torch.arange(
        start_position,
        start_position + content_keys.shape[1],
        device=content_keys.device,
    )
    legacy = []
    for layer in range(content_keys.shape[0]):
        rotated = apply_rope(content_keys[layer], positions, rope).to(torch.bfloat16)
        value = values[layer].to(torch.bfloat16)
        key = rotated.transpose(0, 1).unsqueeze(0)
        value = value.transpose(0, 1).unsqueeze(0)
        if prefix is not None:
            key = torch.cat((prefix[layer][0], key), dim=-2)
            value = torch.cat((prefix[layer][1], value), dim=-2)
        legacy.append((key, value))
    return DynamicCache.from_legacy_cache(tuple(legacy))


def _argmax(values: torch.Tensor) -> int:
    return int(torch.argmax(values).item())


def _cosine(first: torch.Tensor, second: torch.Tensor) -> float:
    return float(torch.nn.functional.cosine_similarity(first.float(), second.float(), dim=0))


def _kl(expected: torch.Tensor, actual: torch.Tensor) -> float:
    expected_log = torch.log_softmax(expected.float(), dim=0)
    actual_log = torch.log_softmax(actual.float(), dim=0)
    return float(torch.sum(torch.exp(expected_log) * (expected_log - actual_log)))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--target-revision", required=True)
    parser.add_argument("--mapper", required=True, type=Path)
    parser.add_argument("--sequences", required=True, type=Path)
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--teacher-forced-steps", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    source_snapshot = snapshot_download(args.source_model, revision=args.source_revision)
    target_snapshot = snapshot_download(args.target_model, revision=args.target_revision)
    source = AutoModelForCausalLM.from_pretrained(
        source_snapshot,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    ).to(args.device).eval()
    target = AutoModelForCausalLM.from_pretrained(
        target_snapshot,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    ).to(args.device).eval()
    source_rope = RopeSpec.from_hf_config(source.config.to_dict())
    target_rope = RopeSpec.from_hf_config(target.config.to_dict())
    mapper = load_file(str(args.mapper), device=args.device)
    selection = mapper["selection"].to(torch.long)
    weight = mapper["weight"]
    bias = mapper["bias"]
    sequences = [
        json.loads(line)
        for line in args.sequences.read_text().splitlines()
        if line.strip()
    ]
    sequences = [row for row in sequences if row["split"] == args.split][: args.limit]
    if len(sequences) != args.limit:
        raise SystemExit(f"found {len(sequences)} sequences, expected {args.limit}")

    results = []
    for index, sequence in enumerate(sequences):
        tokens = torch.tensor([sequence["token_ids"]], dtype=torch.long, device=args.device)
        prefix = tokens[:, :-1]
        tail = tokens[:, -1:]
        started = time.perf_counter()
        with torch.inference_mode():
            source_output = source(
                input_ids=prefix, use_cache=True, logits_to_keep=1, return_dict=True
            )
        source_seconds = time.perf_counter() - started
        source_keys, source_values = _canonical_content(source_output.past_key_values, source_rope)
        started = time.perf_counter()
        mapped_keys, mapped_values = map_content_cache(
            source_keys, source_values, selection, weight, bias
        )
        mapped_cache = _target_cache(mapped_keys, mapped_values, target_rope)
        mapping_seconds = time.perf_counter() - started
        started = time.perf_counter()
        with torch.inference_mode():
            mapped_output = target(
                input_ids=tail,
                past_key_values=mapped_cache,
                position_ids=torch.tensor([[prefix.shape[1]]], device=args.device),
                cache_position=torch.tensor([prefix.shape[1]], device=args.device),
                use_cache=True,
                logits_to_keep=1,
                return_dict=True,
            )
        mapped_tail_seconds = time.perf_counter() - started
        started = time.perf_counter()
        with torch.inference_mode():
            native_output = target(
                input_ids=tokens, use_cache=True, logits_to_keep=1, return_dict=True
            )
        native_seconds = time.perf_counter() - started

        mapped_logits = mapped_output.logits[0, -1]
        native_logits = native_output.logits[0, -1]
        agreements = int(_argmax(mapped_logits) == _argmax(native_logits))
        native_token = _argmax(native_logits)
        mapped_state = mapped_output.past_key_values
        native_state = native_output.past_key_values
        final_mapped = mapped_logits
        final_native = native_logits
        with torch.inference_mode():
            for step in range(1, args.teacher_forced_steps):
                token = torch.tensor([[native_token]], dtype=torch.long, device=args.device)
                position = tokens.shape[1] + step - 1
                native_output = target(
                    input_ids=token,
                    past_key_values=native_state,
                    position_ids=torch.tensor([[position]], device=args.device),
                    cache_position=torch.tensor([position], device=args.device),
                    use_cache=True,
                    logits_to_keep=1,
                    return_dict=True,
                )
                mapped_output = target(
                    input_ids=token,
                    past_key_values=mapped_state,
                    position_ids=torch.tensor([[position]], device=args.device),
                    cache_position=torch.tensor([position], device=args.device),
                    use_cache=True,
                    logits_to_keep=1,
                    return_dict=True,
                )
                native_state = native_output.past_key_values
                mapped_state = mapped_output.past_key_values
                final_native = native_output.logits[0, -1]
                final_mapped = mapped_output.logits[0, -1]
                next_native = _argmax(final_native)
                agreements += int(next_native == _argmax(final_mapped))
                native_token = next_native
        result = {
            "sequenceId": sequence["seq_id"],
            "tokens": tokens.shape[1],
            "firstTop1Equal": _argmax(mapped_logits) == _argmax(native_logits),
            "teacherForcedTop1": agreements,
            "teacherForcedSteps": args.teacher_forced_steps,
            "finalLogitCosine": _cosine(final_native, final_mapped),
            "finalKl": _kl(final_native, final_mapped),
            "sourcePrefillMillis": source_seconds * 1000,
            "mappingMillis": mapping_seconds * 1000,
            "mappedTailMillis": mapped_tail_seconds * 1000,
            "nativeTargetPrefillMillis": native_seconds * 1000,
        }
        results.append(result)
        print(json.dumps(result, sort_keys=True), flush=True)
        del source_output, source_keys, source_values, mapped_keys, mapped_values
        del mapped_cache, mapped_output, native_output, mapped_state, native_state
        torch.cuda.empty_cache()

    report = {
        "schemaVersion": 1,
        "source": {"model": args.source_model, "revision": args.source_revision},
        "target": {"model": args.target_model, "revision": args.target_revision},
        "mapper": {"path": str(args.mapper), "sha256": _sha256(args.mapper)},
        "split": args.split,
        "sequenceManifestSha256": _sha256(args.sequences),
        "results": results,
        "summary": {
            "firstTop1Rate": sum(row["firstTop1Equal"] for row in results) / len(results),
            "teacherForcedTop1Rate": sum(row["teacherForcedTop1"] for row in results)
            / (len(results) * args.teacher_forced_steps),
            "meanFinalLogitCosine": sum(row["finalLogitCosine"] for row in results)
            / len(results),
            "meanFinalKl": sum(row["finalKl"] for row in results) / len(results),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
