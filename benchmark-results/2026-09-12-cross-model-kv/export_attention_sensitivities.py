#!/usr/bin/env python3
"""Export CacheBridge receiver-attention K/V sensitivities for a pinned target model.

This is an offline calibration utility. It replaces only the eager-attention function while the
Hugging Face target oracle runs, retains the normal model outputs, and stores the sensitivity rows
needed for weighted Java-runtime mapper fitting. It is never used by production inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import HfApi, snapshot_download
from transformers import AutoModelForCausalLM


def receiver_sensitivities(
    query: torch.Tensor,
    value: torch.Tensor,
    attention: torch.Tensor,
    output: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return raw K/V sensitivities as ``[boundary, KV-head, prefix-position]``.

    Shapes are query/output ``[query-head, future-query, head-dim]``, value
    ``[KV-head, prefix-position, head-dim]``, and attention
    ``[query-head, future-query, prefix-position]``.
    """
    if query.ndim != 3 or value.ndim != 3 or attention.ndim != 3 or output.ndim != 3:
        raise ValueError("query, value, attention, and output must be rank-three tensors")
    query_heads, queries, dimension = query.shape
    kv_heads, positions, value_dimension = value.shape
    if query_heads % kv_heads != 0:
        raise ValueError("query-head count must be divisible by KV-head count")
    if value_dimension != dimension or output.shape != query.shape:
        raise ValueError("query, value, and output head dimensions must agree")
    if attention.shape != (query_heads, queries, positions):
        raise ValueError("attention shape does not match query heads/queries/prefix positions")

    query = query.to(torch.float32)
    value = value.to(torch.float32)
    attention = attention.to(torch.float32)
    output = output.to(torch.float32)
    key_result = torch.zeros((queries, kv_heads, positions), device=query.device)
    value_result = torch.zeros_like(key_result)
    group_size = query_heads // kv_heads
    for kv_head in range(kv_heads):
        start = kv_head * group_size
        stop = start + group_size
        group_attention = attention[start:stop]
        squared_attention = torch.square(group_attention)
        group_query = query[start:stop]
        group_output = output[start:stop]
        values = value[kv_head]
        value_norm = torch.square(values).sum(-1)
        output_norm = torch.square(group_output).sum(-1)
        output_value_dot = torch.einsum("gud,td->gut", group_output, values)
        squared_distance = (
            output_norm[:, :, None] + value_norm[None, None, :] - 2.0 * output_value_dot
        ).clamp_min_(0.0)
        normalized_query_norm = torch.square(group_query).sum(-1) / dimension
        value_result[:, kv_head] = squared_attention.sum(0)
        key_result[:, kv_head] = (
            squared_attention * squared_distance * normalized_query_norm[:, :, None]
        ).sum(0)
    return key_result, value_result


def causal_boundary_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    boundaries: torch.Tensor,
    scaling: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate the first future query against only the prefix before each boundary."""
    if query.ndim != 3 or key.ndim != 3 or value.shape != key.shape:
        raise ValueError("query, key, and value must be compatible rank-three tensors")
    heads, probes, dimension = query.shape
    if key.shape[0] != heads or key.shape[2] != dimension:
        raise ValueError("query, key, and value heads and dimensions must agree")
    boundaries = boundaries.to(device=query.device, dtype=torch.long)
    if boundaries.shape != (probes,):
        raise ValueError("boundaries must contain one prefix length per query")
    if int(boundaries.min()) < 1 or int(boundaries.max()) > key.shape[1]:
        raise ValueError("each boundary must leave a nonempty in-range prefix")
    scores = torch.einsum("hbd,htd->hbt", query, key) * scaling
    positions = torch.arange(key.shape[1], device=query.device)
    scores.masked_fill_(positions[None, None, :] >= boundaries[None, :, None], -math.inf)
    attention = torch.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
    output = torch.matmul(attention, value)
    return attention, output


def equal_mass_boundary_sum(
    key: torch.Tensor,
    value: torch.Tensor,
    sampled_positions: torch.Tensor,
) -> torch.Tensor:
    """Normalize every boundary separately, then sum them with equal mass.

    The returned layout is ``[sampled-position, KV-head, key-or-value]``.
    """
    if key.ndim != 3 or value.shape != key.shape:
        raise ValueError("key and value sensitivities must be equal rank-three tensors")
    sampled_positions = sampled_positions.to(device=key.device, dtype=torch.long)
    if sampled_positions.ndim != 1 or sampled_positions.numel() == 0:
        raise ValueError("sampled_positions must be a nonempty vector")
    if int(sampled_positions.min()) < 0 or int(sampled_positions.max()) >= key.shape[2]:
        raise ValueError("sampled_positions are outside the sensitivity tensor")
    selected_key = key.index_select(2, sampled_positions)
    selected_value = value.index_select(2, sampled_positions)
    key_mass = selected_key.sum(2, keepdim=True)
    value_mass = selected_value.sum(2, keepdim=True)
    if (key_mass <= 0).any() or (value_mass <= 0).any():
        raise ValueError("every boundary, head, and component needs positive sampled mass")
    combined = torch.stack(
        (selected_key / key_mass, selected_value / value_mass), dim=-1
    ).sum(0)
    return combined.permute(1, 0, 2).contiguous()


def stacked_host_rows(captures: list[torch.Tensor]) -> torch.Tensor:
    """Stack per-layer captures in mmap layout and materialize them on the host."""
    if not captures:
        raise ValueError("at least one layer capture is required")
    rows = torch.stack(captures, dim=1)
    return rows.to(device="cpu", dtype=torch.float32)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _boundaries(length: int, count: int) -> torch.Tensor:
    if length <= 12:
        return torch.tensor([length - 1], dtype=torch.long)
    raw = torch.exp(torch.linspace(math.log(12), math.log(length - 1), count))
    return torch.unique(raw.round().to(torch.long), sorted=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--sequences", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--split", default="calibration")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--boundaries", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.limit < 1 or args.stride < 1 or args.boundaries < 1:
        parser.error("limit, stride, and boundaries must be positive")

    sequences = [
        json.loads(line)
        for line in args.sequences.read_text().splitlines()
        if line.strip()
    ]
    sequences = [row for row in sequences if row["split"] == args.split][: args.limit]
    if len(sequences) != args.limit:
        raise SystemExit(f"found {len(sequences)} sequences, expected {args.limit}")

    revision = args.revision or HfApi().model_info(args.model).sha
    snapshot = Path(snapshot_download(args.model, revision=revision))
    model = AutoModelForCausalLM.from_pretrained(
        snapshot,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    ).to(args.device)
    model.eval()
    config = model.config
    layers = int(config.num_hidden_layers)
    kv_heads = int(config.num_key_value_heads)
    sequence_length = len(sequences[0]["token_ids"])
    sampled_positions = list(range(0, sequence_length, args.stride))
    output_shape = (len(sequences) * len(sampled_positions), layers, kv_heads, 2)
    args.out.mkdir(parents=True, exist_ok=True)
    data_path = args.out / "sensitivities.f32"
    partial = Path(str(data_path) + ".partial")
    if partial.exists():
        partial.unlink()
    mapped = np.memmap(partial, mode="w+", dtype=np.float32, shape=output_shape)

    import transformers.models.qwen3.modeling_qwen3 as qwen3

    original_eager = qwen3.eager_attention_forward
    captures: list[torch.Tensor | None] = []
    active_boundaries: torch.Tensor | None = None

    def capturing_eager(
        module,
        query,
        key,
        value,
        attention_mask,
        scaling,
        dropout=0.0,
        **kwargs,
    ):
        del kwargs
        repeated_key = qwen3.repeat_kv(key, module.num_key_value_groups)
        repeated_value = qwen3.repeat_kv(value, module.num_key_value_groups)
        attention_weights = torch.matmul(query, repeated_key.transpose(2, 3)) * scaling
        if attention_mask is not None:
            attention_weights = attention_weights + attention_mask[:, :, :, : key.shape[-2]]
        attention_weights = torch.nn.functional.softmax(
            attention_weights, dim=-1, dtype=torch.float32
        ).to(query.dtype)
        if dropout:
            attention_weights = torch.nn.functional.dropout(
                attention_weights, p=dropout, training=module.training
            )
        head_output = torch.matmul(attention_weights, repeated_value)
        if active_boundaries is None:
            raise RuntimeError("attention capture has no active query boundaries")
        rows = active_boundaries.to(query.device)
        probe_query = query[0].index_select(1, rows)
        probe_attention, probe_output = causal_boundary_attention(
            probe_query,
            repeated_key[0],
            repeated_value[0],
            rows,
            scaling,
        )
        key_sensitivity, value_sensitivity = receiver_sensitivities(
            probe_query,
            value[0],
            probe_attention,
            probe_output,
        )
        captures[module.layer_idx] = equal_mass_boundary_sum(
            key_sensitivity,
            value_sensitivity,
            torch.tensor(sampled_positions, device=query.device),
        )
        return head_output.transpose(1, 2).contiguous(), attention_weights

    qwen3.eager_attention_forward = capturing_eager
    try:
        cursor = 0
        for index, sequence in enumerate(sequences):
            token_ids = sequence["token_ids"]
            if len(token_ids) != sequence_length:
                raise ValueError(f"{sequence['seq_id']}: sequence length differs")
            active_boundaries = _boundaries(sequence_length, args.boundaries)
            captures = [None] * layers
            input_ids = torch.tensor([token_ids], dtype=torch.long, device=args.device)
            with torch.inference_mode():
                model(
                    input_ids=input_ids,
                    use_cache=False,
                    output_attentions=False,
                    logits_to_keep=1,
                    return_dict=True,
                )
            if any(capture is None for capture in captures):
                raise RuntimeError(f"{sequence['seq_id']}: not every target layer was captured")
            rows = stacked_host_rows(
                [capture for capture in captures if capture is not None]
            )
            next_cursor = cursor + len(sampled_positions)
            mapped[cursor:next_cursor] = rows.numpy()
            cursor = next_cursor
            print(f"attention sensitivities {index + 1}/{len(sequences)}", flush=True)
    finally:
        qwen3.eager_attention_forward = original_eager
    mapped.flush()
    del mapped
    os.replace(partial, data_path)
    manifest = {
        "schemaVersion": 2,
        "method": (
            "CacheBridge first-future-query receiver-attention sensitivities with "
            "per-boundary unit-mass aggregation"
        ),
        "reference": {
            "paper": "https://arxiv.org/abs/2609.00891",
            "version": "v1",
            "equations": [15, 16, 17, 18, 19],
        },
        "eligiblePrefix": "positions strictly before each query boundary",
        "boundaryAggregation": (
            "normalize each layer, KV head, and K/V component to unit mass at every "
            "boundary, then sum boundaries with equal mass"
        ),
        "model": args.model,
        "revision": revision,
        "split": args.split,
        "sequenceIds": [row["seq_id"] for row in sequences],
        "sequenceManifestSha256": _sha256(args.sequences),
        "shape": list(output_shape),
        "layout": "observation,target-layer,kv-head,key-or-value",
        "dtype": "float32",
        "stride": args.stride,
        "sampledPositions": sampled_positions,
        "queryBoundaryCount": args.boundaries,
        "queryBoundaries": _boundaries(sequence_length, args.boundaries).tolist(),
        "file": {
            "name": data_path.name,
            "bytes": data_path.stat().st_size,
            "sha256": _sha256(data_path),
        },
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
