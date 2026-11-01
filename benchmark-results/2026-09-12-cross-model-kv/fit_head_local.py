#!/usr/bin/env python3
"""Fit a compact head-local Qwen KV mapper from audited calibration exports.

The offline fitter keeps every target KV head isolated, selects source layers by held-out
predictive R2, and writes one BF16 SafeTensors artifact. It never participates in production
inference; the resulting immutable artifact is consumed by the Java backend.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file


def effective_sample_size(weights: torch.Tensor) -> float:
    weights = weights.to(torch.float64)
    if weights.ndim != 1 or weights.numel() == 0:
        raise ValueError("weights must be a nonempty vector")
    if not torch.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("weights must be finite and nonnegative")
    denominator = torch.square(weights).sum()
    if denominator <= 0:
        raise ValueError("at least one weight must be positive")
    return float(torch.square(weights.sum()) / denominator)


def shrink_weights(sensitivities: torch.Tensor, parameter_count: int) -> torch.Tensor:
    """CacheBridge ESS shrinkage: preserve contrast only when the fit supports it."""
    if parameter_count < 1:
        raise ValueError("parameter_count must be positive")
    values = sensitivities.to(torch.float64)
    if values.ndim != 1 or values.numel() == 0:
        raise ValueError("sensitivities must be a nonempty vector")
    if not torch.isfinite(values).all() or (values < 0).any():
        raise ValueError("sensitivities must be finite and nonnegative")
    if values.sum() <= 0:
        raise ValueError("at least one sensitivity must be positive")
    normalized = values / values.mean()
    cv2 = torch.square(normalized - 1.0).mean().item()
    floor = min(values.numel(), 2 * parameter_count)
    alpha = 1.0
    if cv2 > 0:
        alpha = min(1.0, math.sqrt(max(0.0, (values.numel() / floor - 1.0) / cv2)))
    return (1.0 + alpha * (normalized - 1.0)).to(sensitivities.dtype)


def centered_ridge(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    weights: torch.Tensor | None = None,
    regularization: float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference centered affine ridge used by tests and small diagnostics."""
    if inputs.ndim != 2 or targets.ndim != 2 or inputs.shape[0] != targets.shape[0]:
        raise ValueError("inputs and targets must be aligned rank-two matrices")
    if regularization < 0 or not math.isfinite(regularization):
        raise ValueError("regularization must be finite and nonnegative")
    if weights is None:
        weights = torch.ones(inputs.shape[0], device=inputs.device, dtype=inputs.dtype)
    else:
        weights = weights.to(device=inputs.device, dtype=inputs.dtype)
    if weights.ndim != 1 or weights.shape[0] != inputs.shape[0]:
        raise ValueError("weights must have one entry per observation")
    if not torch.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("weights must be finite and nonnegative")
    total = weights.sum()
    if total <= 0:
        raise ValueError("at least one weight must be positive")
    mean_x = (inputs * weights[:, None]).sum(0) / total
    mean_y = (targets * weights[:, None]).sum(0) / total
    xc = inputs - mean_x
    yc = targets - mean_y
    covariance = xc.T @ (xc * weights[:, None]) / total
    covariance.diagonal().add_(regularization)
    cross = xc.T @ (yc * weights[:, None]) / total
    weight = torch.linalg.solve(covariance, cross)
    return weight, mean_y - mean_x @ weight


def _sequences(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError("sequence manifest is empty")
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_attention_manifest(
    manifest: dict,
    *,
    expected_model: str,
    expected_revision: str,
    expected_sequence_manifest_sha256: str,
    expected_sequence_ids: list[str],
    expected_sampled_positions: list[int],
    target_layers: int,
    kv_heads: int,
) -> None:
    """Refuse sensitivities that do not implement the audited CacheBridge protocol."""
    if manifest.get("schemaVersion") != 2:
        raise ValueError("attention sensitivities must use corrected schema version 2")
    if manifest.get("method") != (
        "CacheBridge first-future-query receiver-attention sensitivities with "
        "per-boundary unit-mass aggregation"
    ):
        raise ValueError("attention sensitivity method is not the audited CacheBridge method")
    if manifest.get("eligiblePrefix") != "positions strictly before each query boundary":
        raise ValueError("attention sensitivities include an ineligible query position")
    if manifest.get("boundaryAggregation") != (
        "normalize each layer, KV head, and K/V component to unit mass at every "
        "boundary, then sum boundaries with equal mass"
    ):
        raise ValueError("attention sensitivities do not use equal-mass boundary aggregation")
    if manifest.get("reference") != {
        "paper": "https://arxiv.org/abs/2609.00891",
        "version": "v1",
        "equations": [15, 16, 17, 18, 19],
    }:
        raise ValueError("attention sensitivity reference is not the audited paper version")
    if manifest.get("model") != expected_model:
        raise ValueError("attention sensitivity target model differs from target KV")
    if manifest.get("revision") != expected_revision:
        raise ValueError("attention sensitivity target revision differs from target KV")
    if manifest.get("sequenceManifestSha256") != expected_sequence_manifest_sha256:
        raise ValueError("attention sensitivity sequence manifest differs from calibration KV")
    if manifest.get("sequenceIds") != expected_sequence_ids:
        raise ValueError("attention sensitivity sequence order differs from calibration KV")
    if manifest.get("sampledPositions") != expected_sampled_positions:
        raise ValueError("attention sensitivity sampled-position order differs from calibration KV")
    shape = manifest.get("shape")
    if not isinstance(shape, list) or len(shape) != 4 or shape[0] < 1:
        raise ValueError("attention sensitivity shape is invalid")
    if shape[1:] != [target_layers, kv_heads, 2]:
        raise ValueError("attention sensitivity shape does not match target KV geometry")
    if shape[0] != len(expected_sequence_ids) * len(expected_sampled_positions):
        raise ValueError("attention sensitivity observation count differs from its row identity")
    if manifest.get("layout") != "observation,target-layer,kv-head,key-or-value":
        raise ValueError("attention sensitivity layout is unsupported")
    if manifest.get("dtype") != "float32":
        raise ValueError("attention sensitivities must be float32")


def consolidate(exports: Path, sequences_path: Path, out: Path) -> dict:
    """Consolidate per-sequence shards into mmap-friendly content-space K and V files."""
    from cross_model_kv.calibration import load_sequence_kv

    sequences = _sequences(sequences_path)
    first_k, first_v, first_meta = load_sequence_kv(
        exports, sequences[0]["seq_id"], strip_rope=True
    )
    per_sequence = first_k.shape[0]
    shape = (
        len(sequences) * per_sequence,
        first_meta.source.num_layers,
        first_meta.source.num_kv_heads,
        first_meta.source.head_dim,
    )
    if first_k.shape != first_v.shape or tuple(first_k.shape) != (per_sequence, *shape[1:]):
        raise ValueError("first calibration shard has an unexpected shape")
    out.mkdir(parents=True, exist_ok=True)
    files = {"keys": out / "keys.f16", "values": out / "values.f16"}
    partials = {name: Path(str(path) + ".partial") for name, path in files.items()}
    for partial in partials.values():
        if partial.exists():
            partial.unlink()
    mapped = {
        name: np.memmap(path, mode="w+", dtype=np.float16, shape=shape)
        for name, path in partials.items()
    }
    cursor = 0
    for index, sequence in enumerate(sequences):
        if index == 0:
            keys, values, meta = first_k, first_v, first_meta
        else:
            keys, values, meta = load_sequence_kv(
                exports, sequence["seq_id"], strip_rope=True
            )
        if tuple(keys.shape) != (per_sequence, *shape[1:]) or keys.shape != values.shape:
            raise ValueError(f"{sequence['seq_id']}: inconsistent calibration shape")
        if meta.token_ids_sha256 != sequence["token_ids_sha256"]:
            raise ValueError(f"{sequence['seq_id']}: token hash differs from the sequence manifest")
        next_cursor = cursor + per_sequence
        mapped["keys"][cursor:next_cursor] = keys.to(torch.float16).numpy()
        mapped["values"][cursor:next_cursor] = values.to(torch.float16).numpy()
        cursor = next_cursor
        if (index + 1) % 25 == 0 or index + 1 == len(sequences):
            print(f"consolidated {index + 1}/{len(sequences)}", flush=True)
    for name, array in mapped.items():
        array.flush()
        del array
        os.replace(partials[name], files[name])
    split_rows: dict[str, list[int]] = {}
    cursor = 0
    for sequence in sequences:
        bounds = split_rows.setdefault(sequence["split"], [cursor, cursor])
        if bounds[1] != cursor:
            raise ValueError("each split must be contiguous in sequences.jsonl")
        cursor += per_sequence
        bounds[1] = cursor
    manifest = {
        "schemaVersion": 1,
        "source": first_meta.source.to_dict(),
        "rope": first_meta.rope,
        "shape": list(shape),
        "dtype": "float16",
        "contentSpaceKeys": True,
        "sampledPositions": first_meta.positions,
        "splits": split_rows,
        "sequenceManifestSha256": _sha256(sequences_path),
        "files": {
            name: {"name": path.name, "sha256": _sha256(path), "bytes": path.stat().st_size}
            for name, path in files.items()
        },
    }
    manifest_path = out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def _open_consolidated(root: Path) -> tuple[dict, dict[str, np.memmap]]:
    manifest = json.loads((root / "manifest.json").read_text())
    shape = tuple(manifest["shape"])
    arrays = {
        name: np.memmap(root / entry["name"], mode="r", dtype=np.float16, shape=shape)
        for name, entry in manifest["files"].items()
    }
    return manifest, arrays


def _tensor(array: np.memmap, start: int, end: int, device: str) -> torch.Tensor:
    # The copy makes the slice writable and gives CUDA one contiguous host transfer.
    return torch.from_numpy(np.array(array[start:end], copy=True)).to(device, torch.float32)


def _moments_for_layer_selection(
    source: np.memmap,
    target: np.memmap,
    start: int,
    end: int,
    head: int,
    device: str,
    chunk: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    layers, dimension = source.shape[1], source.shape[3]
    target_layers = target.shape[1]
    sum_x = torch.zeros((layers, dimension), device=device, dtype=torch.float64)
    sum_y = torch.zeros((target_layers, dimension), device=device, dtype=torch.float64)
    xx = torch.zeros((layers, dimension, dimension), device=device)
    xy = torch.zeros((layers, target_layers, dimension, dimension), device=device)
    count = 0
    for offset in range(start, end, chunk):
        stop = min(end, offset + chunk)
        x = _tensor(source, offset, stop, device)[:, :, head, :]
        y = _tensor(target, offset, stop, device)[:, :, head, :]
        sum_x += x.sum(0).to(torch.float64)
        sum_y += y.sum(0).to(torch.float64)
        xx += torch.einsum("nli,nlj->lij", x, x)
        xy += torch.einsum("nli,ntj->ltij", x, y)
        count += stop - offset
        del x, y
    return sum_x, sum_y, xx, xy, count


def select_layers(
    source: np.memmap,
    target: np.memmap,
    fit_rows: tuple[int, int],
    eval_rows: tuple[int, int],
    top_k: int,
    regularization: float,
    device: str,
    chunk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit head-local single-layer probes and score them on the requested rows."""
    layers, heads, dimension = source.shape[1], source.shape[2], source.shape[3]
    target_layers = target.shape[1]
    score = torch.zeros((target_layers, layers), dtype=torch.float64)
    for head in range(heads):
        sum_x, sum_y, xx, xy, n = _moments_for_layer_selection(
            source, target, *fit_rows, head, device, chunk
        )
        mean_x = (sum_x / n).to(torch.float32)
        mean_y = (sum_y / n).to(torch.float32)
        covariance = xx / n - torch.einsum("li,lj->lij", mean_x, mean_x)
        covariance.diagonal(dim1=-2, dim2=-1).add_(regularization)
        cross = xy / n - torch.einsum("li,tj->ltij", mean_x, mean_y)
        weight = torch.linalg.solve(covariance[:, None], cross)
        bias = mean_y[None] - torch.einsum("li,ltij->ltj", mean_x, weight)

        eval_start, eval_end = eval_rows
        target_sum = torch.zeros((target_layers, dimension), device=device, dtype=torch.float64)
        target_sq = torch.zeros((target_layers, dimension), device=device, dtype=torch.float64)
        residual = torch.zeros((layers, target_layers), device=device, dtype=torch.float64)
        eval_count = 0
        for offset in range(eval_start, eval_end, chunk):
            stop = min(eval_end, offset + chunk)
            x = _tensor(source, offset, stop, device)[:, :, head, :]
            y = _tensor(target, offset, stop, device)[:, :, head, :]
            prediction = torch.einsum("nli,ltij->nltj", x, weight) + bias
            residual += torch.square(prediction - y[:, None]).sum((0, 3)).to(torch.float64)
            target_sum += y.sum(0).to(torch.float64)
            target_sq += torch.square(y).sum(0).to(torch.float64)
            eval_count += stop - offset
            del x, y, prediction
        total = (target_sq - torch.square(target_sum) / eval_count).sum(1)
        r2 = 1.0 - residual / total[None]
        score += r2.T.cpu()
        print(f"selected-layer probe head {head + 1}/{heads}", flush=True)
        del sum_x, sum_y, xx, xy, covariance, cross, weight, bias, residual
    score /= heads
    selection = torch.topk(score, top_k, dim=1).indices.to(torch.int32)
    return selection, score


def combine_selection_scores(
    scores: dict[str, torch.Tensor], top_k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Choose one K/V source-layer ordering using their mean predictive R2."""
    if set(scores) != {"keys", "values"} or scores["keys"].shape != scores["values"].shape:
        raise ValueError("key and value selection scores must have the same shape")
    if top_k < 1 or top_k > scores["keys"].shape[1]:
        raise ValueError("top_k is outside the source-layer count")
    combined = (scores["keys"] + scores["values"]) / 2.0
    return torch.topk(combined, top_k, dim=1).indices.to(torch.int32), combined


def cachebridge_selection_protocol(
    *,
    calibration_rows: tuple[int, int],
    rows_per_sequence: int,
    calibration_sequences: int,
) -> tuple[tuple[int, int], tuple[int, int], float]:
    """Return the paper's calibration-only centered OLS selection protocol."""
    if rows_per_sequence < 1:
        raise ValueError("rows_per_sequence must be positive")
    row_count = calibration_rows[1] - calibration_rows[0]
    if row_count < 1 or row_count % rows_per_sequence:
        raise ValueError("calibration rows do not contain complete sequences")
    expected_sequences = row_count // rows_per_sequence
    if calibration_sequences != expected_sequences:
        raise ValueError(
            f"CacheBridge layer selection requires all {expected_sequences} "
            "calibration sequences"
        )
    return calibration_rows, calibration_rows, 0.0


def select_only(
    source_root: Path,
    target_root: Path,
    out: Path,
    top_k: int,
    regularization: float,
    probe_sequences: int,
    device: str,
    chunk: int,
) -> dict:
    """Persist layer ranking independently so a larger mapper fit need not allocate an ablation."""
    source_manifest, source = _open_consolidated(source_root)
    target_manifest, target = _open_consolidated(target_root)
    if source_manifest["shape"] != target_manifest["shape"]:
        raise ValueError("source and target consolidated KV shapes differ")
    fit_rows = tuple(source_manifest["splits"]["calibration"])
    eval_rows = tuple(source_manifest["splits"]["validation"])
    rows_per_sequence = len(source_manifest["sampledPositions"])
    probe_fit_rows = (
        fit_rows[0],
        min(fit_rows[1], fit_rows[0] + probe_sequences * rows_per_sequence),
    )
    if probe_fit_rows[1] <= probe_fit_rows[0]:
        raise ValueError("probe fit split is empty")
    scores = {}
    for kind in ("keys", "values"):
        _, scores[kind] = select_layers(
            source[kind],
            target[kind],
            probe_fit_rows,
            eval_rows,
            top_k,
            regularization,
            device,
            chunk,
        )
    selection, combined = combine_selection_scores(scores, top_k)
    report = {
        "schemaVersion": 1,
        "kind": "cross-model-kv-layer-selection",
        "method": "per-head K/V predictive R2, averaged by target/source layer",
        "source": source_manifest["source"],
        "target": target_manifest["source"],
        "topK": top_k,
        "lambda": regularization,
        "selectionProbeSequences": probe_sequences,
        "selectionProbeRows": list(probe_fit_rows),
        "selectionEvaluationSplit": "validation",
        "selectionEvaluationRows": list(eval_rows),
        "sequenceManifestSha256": source_manifest["sequenceManifestSha256"],
        "selection": selection.tolist(),
        "scores": {
            "keys": scores["keys"].tolist(),
            "values": scores["values"].tolist(),
            "combined": combined.tolist(),
        },
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True))
    return report


def center_normal_equations_in_place(
    xx: torch.Tensor,
    xy: torch.Tensor,
    mean_x: torch.Tensor,
    mean_y: torch.Tensor,
    total_weight: torch.Tensor,
    regularization: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Center weighted moments without duplicating the large covariance tensors."""
    if xx.ndim != 4 or xy.ndim != 4 or xx.shape[:2] != xy.shape[:2]:
        raise ValueError("normal-equation tensors must share target-layer/head axes")
    if xx.shape[2] != xx.shape[3] or xy.shape[2] != xx.shape[2]:
        raise ValueError("normal-equation feature dimensions differ")
    if mean_x.shape != xx.shape[:3] or mean_y.shape != (
        *xy.shape[:2],
        xy.shape[3],
    ):
        raise ValueError("weighted means differ from normal-equation geometry")
    if total_weight.shape != xx.shape[:2] or (total_weight <= 0).any():
        raise ValueError("every normal equation needs positive total weight")
    if regularization < 0 or not math.isfinite(regularization):
        raise ValueError("regularization must be finite and nonnegative")
    for target_layer in range(xx.shape[0]):
        for head in range(xx.shape[1]):
            normalizer = float(total_weight[target_layer, head])
            covariance = xx[target_layer, head]
            cross = xy[target_layer, head]
            covariance.div_(normalizer)
            covariance.addmm_(
                mean_x[target_layer, head, :, None],
                mean_x[target_layer, head, None, :],
                beta=1.0,
                alpha=-1.0,
            )
            covariance.diagonal().add_(regularization)
            cross.div_(normalizer)
            cross.addmm_(
                mean_x[target_layer, head, :, None],
                mean_y[target_layer, head, None, :],
                beta=1.0,
                alpha=-1.0,
            )
    return xx, xy


def fit_kind(
    source: np.memmap,
    target: np.memmap,
    selection: torch.Tensor,
    fit_rows: tuple[int, int],
    eval_rows: tuple[int, int],
    regularization: float,
    device: str,
    chunk: int,
    observation_weights: np.memmap | None = None,
) -> tuple[torch.Tensor, torch.Tensor, list[float], list[float]]:
    target_layers, top_k = selection.shape
    heads, dimension = source.shape[2], source.shape[3]
    features = top_k * dimension
    selection_device = selection.to(device=device, dtype=torch.long)
    fit_start, fit_end = fit_rows
    sum_x = torch.zeros(
        (target_layers, heads, features), device=device, dtype=torch.float64
    )
    sum_y = torch.zeros(
        (target_layers, heads, dimension), device=device, dtype=torch.float64
    )
    total_weight = torch.zeros((target_layers, heads), device=device, dtype=torch.float64)
    xx = torch.zeros((target_layers, heads, features, features), device=device)
    xy = torch.zeros((target_layers, heads, features, dimension), device=device)
    count = 0
    for offset in range(fit_start, fit_end, chunk):
        stop = min(fit_end, offset + chunk)
        raw_source = _tensor(source, offset, stop, device)
        gathered = raw_source[:, selection_device]
        x = gathered.permute(1, 3, 0, 2, 4).reshape(
            target_layers, heads, stop - offset, features
        )
        y = _tensor(target, offset, stop, device).permute(1, 2, 0, 3)
        if observation_weights is None:
            batch_weights = torch.ones(
                (target_layers, heads, stop - offset), device=device
            )
        else:
            weight_start = offset - fit_start
            weight_stop = weight_start + (stop - offset)
            batch_weights = torch.from_numpy(
                np.array(observation_weights[weight_start:weight_stop], copy=True)
            ).to(device, torch.float32).permute(1, 2, 0)
        total_weight += batch_weights.sum(2).to(torch.float64)
        sum_x += torch.einsum("thn,thnp->thp", batch_weights, x).to(torch.float64)
        sum_y += torch.einsum("thn,thnp->thp", batch_weights, y).to(torch.float64)
        xx += torch.einsum("thn,thnp,thnq->thpq", batch_weights, x, x)
        xy += torch.einsum("thn,thnp,thnq->thpq", batch_weights, x, y)
        count += stop - offset
        del raw_source, gathered, x, y, batch_weights
    if (total_weight <= 0).any():
        raise ValueError("every target layer/head needs positive fitting weight")
    mean_x = (sum_x / total_weight[:, :, None]).to(torch.float32)
    mean_y = (sum_y / total_weight[:, :, None]).to(torch.float32)
    covariance, cross = center_normal_equations_in_place(
        xx,
        xy,
        mean_x,
        mean_y,
        total_weight,
        regularization,
    )
    flat_count = target_layers * heads
    flat_covariance = covariance.reshape(flat_count, features, features)
    flat_cross = cross.reshape(flat_count, features, dimension)
    flat_weight = torch.empty_like(flat_cross)
    for equation in range(flat_count):
        flat_weight[equation] = torch.linalg.solve(
            flat_covariance[equation], flat_cross[equation]
        )
    weight = flat_weight.reshape(target_layers, heads, features, dimension)
    bias = mean_y - torch.einsum("thi,thij->thj", mean_x, weight)
    del covariance, cross, flat_covariance, flat_cross, xx, xy
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()

    residual_fp32 = torch.zeros((target_layers, heads), device=device, dtype=torch.float64)
    residual_bf16 = torch.zeros_like(residual_fp32)
    target_sum = torch.zeros(
        (target_layers, heads, dimension), device=device, dtype=torch.float64
    )
    target_sq = torch.zeros_like(target_sum)
    eval_start, eval_end = eval_rows
    eval_count = 0
    flat_weight = weight.reshape(flat_count, features, dimension)
    flat_bias = bias.reshape(flat_count, dimension)
    bf16_weight = flat_weight.to(torch.bfloat16)
    bf16_bias = flat_bias.to(torch.bfloat16)
    for offset in range(eval_start, eval_end, chunk):
        stop = min(eval_end, offset + chunk)
        raw_source = _tensor(source, offset, stop, device)
        gathered = raw_source[:, selection_device]
        x = gathered.permute(1, 3, 0, 2, 4).reshape(
            flat_count, stop - offset, features
        )
        y = _tensor(target, offset, stop, device).permute(1, 2, 0, 3)
        flat_y = y.reshape(flat_count, stop - offset, dimension)
        prediction = torch.bmm(x, flat_weight) + flat_bias[:, None]
        quantized = (
            torch.bmm(x.to(torch.bfloat16), bf16_weight) + bf16_bias[:, None]
        ).to(torch.float32)
        residual_fp32 += torch.square(prediction - flat_y).sum((1, 2)).reshape(
            target_layers, heads
        ).to(torch.float64)
        residual_bf16 += torch.square(quantized - flat_y).sum((1, 2)).reshape(
            target_layers, heads
        ).to(torch.float64)
        target_sum += y.sum(2).to(torch.float64)
        target_sq += torch.square(y).sum(2).to(torch.float64)
        eval_count += stop - offset
        del raw_source, gathered, x, y, flat_y, prediction, quantized
    total = (target_sq - torch.square(target_sum) / eval_count).sum(2)
    fp32_matrix = 1.0 - residual_fp32 / total
    bf16_matrix = 1.0 - residual_bf16 / total
    for target_layer in range(target_layers):
        print(
            f"fit target layer {target_layer + 1}/{target_layers}: "
            f"held-out R2={fp32_matrix[target_layer].mean().item():.4f}",
            flush=True,
        )
    return (
        weight.to("cpu", torch.bfloat16),
        bias.to("cpu", torch.bfloat16),
        fp32_matrix.cpu().flatten().tolist(),
        bf16_matrix.cpu().flatten().tolist(),
    )


def fit(
    source_root: Path,
    target_root: Path,
    out: Path,
    top_k: int,
    regularization: float,
    probe_sequences: int,
    device: str,
    chunk: int,
    sensitivity_root: Path | None = None,
    sequences_path: Path | None = None,
) -> dict:
    if out.exists():
        raise FileExistsError(f"refusing to overwrite mapper evidence directory: {out}")
    source_manifest, source = _open_consolidated(source_root)
    target_manifest, target = _open_consolidated(target_root)
    if source_manifest["shape"] != target_manifest["shape"]:
        raise ValueError("source and target consolidated KV shapes differ")
    if source_manifest["sequenceManifestSha256"] != target_manifest["sequenceManifestSha256"]:
        raise ValueError("source and target calibration sequence manifests differ")
    if source_manifest["sampledPositions"] != target_manifest["sampledPositions"]:
        raise ValueError("source and target sampled-position order differs")

    attention_manifest: dict | None = None
    sensitivity_shape: tuple[int, ...] | None = None
    sensitivity_file: Path | None = None
    if sensitivity_root is not None:
        if sequences_path is None:
            raise ValueError("--sequences is required for attention-weighted fitting")
        if _sha256(sequences_path) != source_manifest["sequenceManifestSha256"]:
            raise ValueError("sequence manifest file differs from consolidated calibration KV")
        sequence_rows = _sequences(sequences_path)
        calibration_sequence_ids = [
            row["seq_id"] for row in sequence_rows if row["split"] == "calibration"
        ]
        attention_manifest = json.loads((sensitivity_root / "manifest.json").read_text())
        expected_model = target_manifest["source"]["model_id"]
        expected_revision = target_manifest["source"]["revision"]
        validate_attention_manifest(
            attention_manifest,
            expected_model=expected_model,
            expected_revision=expected_revision,
            expected_sequence_manifest_sha256=source_manifest["sequenceManifestSha256"],
            expected_sequence_ids=calibration_sequence_ids,
            expected_sampled_positions=source_manifest["sampledPositions"],
            target_layers=target_manifest["shape"][1],
            kv_heads=target_manifest["shape"][2],
        )
        sensitivity_shape = tuple(attention_manifest["shape"])
        sensitivity_file = sensitivity_root / attention_manifest["file"]["name"]
        if sensitivity_file.stat().st_size != attention_manifest["file"]["bytes"]:
            raise ValueError("attention sensitivity file size differs from its manifest")
        if _sha256(sensitivity_file) != attention_manifest["file"]["sha256"]:
            raise ValueError("attention sensitivity file hash differs from its manifest")

    fit_rows = tuple(source_manifest["splits"]["calibration"])
    eval_rows = tuple(source_manifest["splits"]["validation"])
    rows_per_sequence = len(source_manifest["sampledPositions"])
    probe_fit_rows = (
        fit_rows[0],
        min(fit_rows[1], fit_rows[0] + probe_sequences * rows_per_sequence),
    )
    selection_eval_rows = eval_rows
    selection_regularization = regularization
    selection_method = "held-out predictive R2 ridge variant"
    if sensitivity_root is not None:
        (
            probe_fit_rows,
            selection_eval_rows,
            selection_regularization,
        ) = cachebridge_selection_protocol(
            calibration_rows=fit_rows,
            rows_per_sequence=rows_per_sequence,
            calibration_sequences=probe_sequences,
        )
        selection_method = (
            "calibration-only centered same-head single-source-layer OLS; "
            "mean held-in R2 across K/V and KV heads"
        )
    if probe_fit_rows[1] <= probe_fit_rows[0]:
        raise ValueError("probe fit split is empty")

    scores: dict[str, torch.Tensor] = {}
    selections: dict[str, torch.Tensor] = {}
    for kind in ("keys", "values"):
        selection, score = select_layers(
            source[kind],
            target[kind],
            probe_fit_rows,
            selection_eval_rows,
            top_k,
            selection_regularization,
            device,
            chunk,
        )
        selections[kind] = selection
        scores[kind] = score
    # CacheBridge uses one K/V selection. Its exact path ranks calibration-only held-in R2.
    selection, _ = combine_selection_scores(scores, top_k)

    attention_weights: dict[str, np.memmap] = {}
    attention_report: dict | None = None
    if sensitivity_root is not None:
        if attention_manifest is None or sensitivity_shape is None or sensitivity_file is None:
            raise AssertionError("attention sensitivity preflight did not run")
        raw = np.memmap(
            sensitivity_file,
            mode="r",
            dtype=np.float32,
            shape=sensitivity_shape,
        )
        fit_rows = (fit_rows[0], fit_rows[0] + sensitivity_shape[0])
        if fit_rows[1] > source_manifest["splits"]["calibration"][1]:
            raise ValueError("attention observations exceed the calibration split")
        effective_sizes: dict[str, list[float]] = {"keys": [], "values": []}
        out.mkdir(parents=True, exist_ok=False)
        scratch = out / "fit-scratch"
        scratch.mkdir()
        derived_weight_files: dict[str, dict] = {}
        for kind_index, kind in enumerate(("keys", "values")):
            path = scratch / f"shrunk-{kind}.f32"
            shaped = np.memmap(
                path,
                mode="w+",
                dtype=np.float32,
                shape=sensitivity_shape[:-1],
            )
            for target_layer in range(sensitivity_shape[1]):
                for head in range(sensitivity_shape[2]):
                    values = torch.from_numpy(
                        np.array(raw[:, target_layer, head, kind_index], copy=True)
                    )
                    shrunk = shrink_weights(values, top_k * source_manifest["shape"][3])
                    shaped[:, target_layer, head] = shrunk.numpy()
                    effective_sizes[kind].append(effective_sample_size(shrunk))
            shaped.flush()
            attention_weights[kind] = shaped
            derived_weight_files[kind] = {
                "name": str(path.relative_to(out)),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        attention_report = {
            "manifest": attention_manifest,
            "fitObservations": sensitivity_shape[0],
            "derivedWeightFiles": derived_weight_files,
            "effectiveSampleSize": {
                kind: {
                    "min": min(values),
                    "mean": sum(values) / len(values),
                    "max": max(values),
                }
                for kind, values in effective_sizes.items()
            },
        }

    weights = []
    biases = []
    metrics: dict[str, dict] = {}
    for kind in ("keys", "values"):
        weight, bias, fp32_r2, bf16_r2 = fit_kind(
            source[kind],
            target[kind],
            selection,
            fit_rows,
            eval_rows,
            regularization,
            device,
            chunk,
            attention_weights.get(kind),
        )
        weights.append(weight)
        biases.append(bias)
        metrics[kind] = {
            "heldoutR2": sum(fp32_r2) / len(fp32_r2),
            "bf16HeldoutR2": sum(bf16_r2) / len(bf16_r2),
            "bf16Delta": (sum(fp32_r2) - sum(bf16_r2)) / len(fp32_r2),
            "heldoutR2PerTargetHead": fp32_r2,
        }
    if sensitivity_root is None:
        out.mkdir(parents=True, exist_ok=False)
    artifact = out / "mapper.safetensors"
    save_file(
        {
            "selection": selection.contiguous(),
            "weight": torch.stack(weights, dim=1).contiguous(),
            "bias": torch.stack(biases, dim=1).contiguous(),
        },
        str(artifact),
    )
    report = {
        "schemaVersion": 1,
        "kind": "head-local-cross-model-kv-ridge",
        "method": (
            "CacheBridge head-local predictive mapping with receiver-attention weighting"
            if attention_report is not None
            else "CacheBridge head-local predictive mapping; uniform calibration baseline"
        ),
        "source": source_manifest["source"],
        "target": target_manifest["source"],
        "topK": top_k,
        "lambda": regularization,
        "fitSplit": "calibration",
        "selectionProbeSequences": probe_sequences,
        "selectionMethod": selection_method,
        "selectionFitRows": list(probe_fit_rows),
        "selectionEvaluationRows": list(selection_eval_rows),
        "selectionEvaluationSplit": (
            "calibration" if selection_eval_rows == fit_rows else "validation"
        ),
        "mapperEvaluationSplit": "validation",
        "contentSpaceKeys": True,
        "weightsDtype": "bfloat16",
        "selection": selection.tolist(),
        "metrics": metrics,
        "attentionWeighting": attention_report,
        "artifact": {
            "name": artifact.name,
            "bytes": artifact.stat().st_size,
            "sha256": _sha256(artifact),
        },
    }
    (out / "fit-report.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    consolidate_parser = subparsers.add_parser("consolidate")
    consolidate_parser.add_argument("--exports", required=True, type=Path)
    consolidate_parser.add_argument("--sequences", required=True, type=Path)
    consolidate_parser.add_argument("--out", required=True, type=Path)
    fit_parser = subparsers.add_parser("fit")
    fit_parser.add_argument("--source", required=True, type=Path)
    fit_parser.add_argument("--target", required=True, type=Path)
    fit_parser.add_argument("--out", required=True, type=Path)
    fit_parser.add_argument("--top-k", type=int, default=4)
    fit_parser.add_argument("--lambda", dest="regularization", type=float, default=0.01)
    fit_parser.add_argument("--probe-sequences", type=int, default=100)
    fit_parser.add_argument("--device", default="cuda")
    fit_parser.add_argument("--chunk", type=int, default=256)
    fit_parser.add_argument("--sensitivities", type=Path, default=None)
    fit_parser.add_argument("--sequences", type=Path, default=None)
    select_parser = subparsers.add_parser("select")
    select_parser.add_argument("--source", required=True, type=Path)
    select_parser.add_argument("--target", required=True, type=Path)
    select_parser.add_argument("--out", required=True, type=Path)
    select_parser.add_argument("--top-k", type=int, default=12)
    select_parser.add_argument("--lambda", dest="regularization", type=float, default=0.01)
    select_parser.add_argument("--probe-sequences", type=int, default=500)
    select_parser.add_argument("--device", default="cuda")
    select_parser.add_argument("--chunk", type=int, default=256)
    args = parser.parse_args()
    started = time.time()
    if args.command == "consolidate":
        result = consolidate(args.exports, args.sequences, args.out)
    elif args.command == "select":
        result = select_only(
            args.source,
            args.target,
            args.out,
            args.top_k,
            args.regularization,
            args.probe_sequences,
            args.device,
            args.chunk,
        )
    else:
        result = fit(
            args.source,
            args.target,
            args.out,
            args.top_k,
            args.regularization,
            args.probe_sequences,
            args.device,
            args.chunk,
            args.sensitivities,
            args.sequences,
        )
    print(json.dumps({"elapsedSeconds": round(time.time() - started, 3), "result": result}))


if __name__ == "__main__":
    main()
