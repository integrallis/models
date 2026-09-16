#!/usr/bin/env python3
"""Fit the full cross-head Qwen KV transform from audited calibration exports.

Unlike the smaller head-local ablation, every target KV head can use every source KV head from
each selected source layer.  We stream the consolidated memmaps and solve one target layer at a
time so the production-size fit does not depend on loading the full calibration corpus into RAM.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

from fit_head_local import effective_sample_size, shrink_weights


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _open_consolidated(root: Path) -> tuple[dict, dict[str, np.memmap]]:
    manifest = json.loads((root / "manifest.json").read_text())
    shape = tuple(manifest["shape"])
    arrays = {
        name: np.memmap(root / entry["name"], mode="r", dtype=np.float16, shape=shape)
        for name, entry in manifest["files"].items()
    }
    return manifest, arrays


def _tensor(array: np.memmap, start: int, end: int, device: str) -> torch.Tensor:
    return torch.from_numpy(np.array(array[start:end], copy=True)).to(device, torch.float32)


def _validate_selection(
    selection: torch.Tensor, source_layers: int, target_layers: int
) -> None:
    if selection.ndim != 2 or selection.shape[0] != target_layers or selection.shape[1] < 1:
        raise ValueError("selection must provide at least one source layer per target layer")
    if int(selection.min()) < 0 or int(selection.max()) >= source_layers:
        raise ValueError("selection contains a source layer outside the source model")
    for layer in selection:
        if len(set(int(value) for value in layer)) != layer.numel():
            raise ValueError("selection must not repeat a source layer")


def fit_full_kind(
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
    """Fit one K or V transform, returning BF16 weights in Java row-major layout.

    Weights are ``[target-layer, output, input]`` so a contiguous layer slice is directly usable
    as a ``BFloat16Matrix`` without a runtime transpose.
    """
    if source.shape != target.shape or len(source.shape) != 4:
        raise ValueError("source and target must be aligned rank-four arrays")
    if regularization < 0 or not math.isfinite(regularization):
        raise ValueError("regularization must be finite and nonnegative")
    if chunk < 1:
        raise ValueError("chunk must be positive")
    observations, source_layers, heads, head_dimension = source.shape
    target_layers = target.shape[1]
    _validate_selection(selection, source_layers, target_layers)
    top_k = selection.shape[1]
    input_dimensions = top_k * heads * head_dimension
    output_dimensions = heads * head_dimension
    fit_count = fit_rows[1] - fit_rows[0]
    if fit_count < input_dimensions * 4:
        raise ValueError(
            f"full-head production fit needs at least four observations per parameter: "
            f"{fit_count} < {input_dimensions * 4}"
        )
    if not (0 <= fit_rows[0] < fit_rows[1] <= observations):
        raise ValueError("fit rows are outside the calibration array")
    if not (0 <= eval_rows[0] < eval_rows[1] <= observations):
        raise ValueError("evaluation rows are outside the calibration array")
    if observation_weights is not None and observation_weights.shape != (
        fit_count,
        target_layers,
        heads,
    ):
        raise ValueError(
            "observation weights must have shape "
            f"[{fit_count}, {target_layers}, {heads}]"
        )

    weights: list[torch.Tensor] = []
    biases: list[torch.Tensor] = []
    fp32_r2: list[float] = []
    bf16_r2: list[float] = []
    for target_layer in range(target_layers):
        picks = selection[target_layer].to(device=device, dtype=torch.long)
        fit_heads = range(1) if observation_weights is None else range(heads)
        head_weights: list[torch.Tensor] = []
        head_biases: list[torch.Tensor] = []
        for weighted_head in fit_heads:
            fitted_outputs = output_dimensions if observation_weights is None else head_dimension
            sum_x = torch.zeros(input_dimensions, device=device, dtype=torch.float64)
            sum_y = torch.zeros(fitted_outputs, device=device, dtype=torch.float64)
            xx = torch.zeros(
                (input_dimensions, input_dimensions), device=device, dtype=torch.float32
            )
            xy = torch.zeros(
                (input_dimensions, fitted_outputs), device=device, dtype=torch.float32
            )
            total_weight = torch.zeros((), device=device, dtype=torch.float64)
            for offset in range(fit_rows[0], fit_rows[1], chunk):
                stop = min(fit_rows[1], offset + chunk)
                raw = _tensor(source, offset, stop, device)
                x = raw.index_select(1, picks).reshape(stop - offset, input_dimensions)
                target_rows = _tensor(target, offset, stop, device)[:, target_layer]
                if observation_weights is None:
                    y = target_rows.reshape(stop - offset, output_dimensions)
                    row_weights = torch.ones(stop - offset, device=device)
                else:
                    y = target_rows[:, weighted_head]
                    weight_start = offset - fit_rows[0]
                    weight_stop = weight_start + stop - offset
                    row_weights = torch.from_numpy(
                        np.array(
                            observation_weights[
                                weight_start:weight_stop, target_layer, weighted_head
                            ],
                            copy=True,
                        )
                    ).to(device, torch.float32)
                if not torch.isfinite(row_weights).all() or (row_weights < 0).any():
                    raise ValueError("observation weights must be finite and nonnegative")
                weighted_x = x * row_weights[:, None]
                sum_x += weighted_x.sum(0).to(torch.float64)
                sum_y += (y * row_weights[:, None]).sum(0).to(torch.float64)
                total_weight += row_weights.sum().to(torch.float64)
                xx.addmm_(x.T, weighted_x)
                xy.addmm_(x.T, y * row_weights[:, None])
                del raw, x, target_rows, y, row_weights, weighted_x
            if total_weight <= 0:
                raise ValueError("every target layer/head needs positive fitting weight")

            mean_x = (sum_x / total_weight).to(torch.float32)
            mean_y = (sum_y / total_weight).to(torch.float32)
            covariance = xx / total_weight.to(torch.float32) - torch.outer(mean_x, mean_x)
            covariance.diagonal().add_(regularization)
            cross = xy / total_weight.to(torch.float32) - torch.outer(mean_x, mean_y)
            # Ridge makes the covariance positive definite. Cholesky avoids a general LU solve and
            # gives one factorization for all output dimensions.
            factor = torch.linalg.cholesky(covariance)
            input_output_weight = torch.cholesky_solve(cross, factor)
            fitted_bias = mean_y - mean_x @ input_output_weight
            head_weights.append(input_output_weight.T.contiguous())
            head_biases.append(fitted_bias)
            del sum_x, sum_y, xx, xy, total_weight, covariance, cross, factor
            del input_output_weight, fitted_bias
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
        output_input_weight = torch.cat(head_weights, dim=0)
        bias = torch.cat(head_biases, dim=0)

        residual_fp32 = torch.zeros((), device=device, dtype=torch.float64)
        residual_bf16 = torch.zeros((), device=device, dtype=torch.float64)
        target_sum = torch.zeros(output_dimensions, device=device, dtype=torch.float64)
        target_sq = torch.zeros(output_dimensions, device=device, dtype=torch.float64)
        eval_count = 0
        bf16_weight = output_input_weight.to(torch.bfloat16)
        bf16_bias = bias.to(torch.bfloat16)
        for offset in range(eval_rows[0], eval_rows[1], chunk):
            stop = min(eval_rows[1], offset + chunk)
            raw = _tensor(source, offset, stop, device)
            x = raw.index_select(1, picks).reshape(stop - offset, input_dimensions)
            y = _tensor(target, offset, stop, device)[:, target_layer].reshape(
                stop - offset, output_dimensions
            )
            prediction = torch.nn.functional.linear(x, output_input_weight, bias)
            quantized = torch.nn.functional.linear(
                x.to(torch.bfloat16), bf16_weight, bf16_bias
            ).to(torch.float32)
            residual_fp32 += torch.square(prediction - y).sum().to(torch.float64)
            residual_bf16 += torch.square(quantized - y).sum().to(torch.float64)
            target_sum += y.sum(0).to(torch.float64)
            target_sq += torch.square(y).sum(0).to(torch.float64)
            eval_count += stop - offset
            del raw, x, y, prediction, quantized
        total = (target_sq - torch.square(target_sum) / eval_count).sum()
        layer_fp32 = float(1.0 - residual_fp32 / total)
        layer_bf16 = float(1.0 - residual_bf16 / total)
        fp32_r2.append(layer_fp32)
        bf16_r2.append(layer_bf16)
        weights.append(bf16_weight.cpu())
        biases.append(bf16_bias.cpu())
        print(
            f"full-head target layer {target_layer + 1}/{target_layers}: "
            f"held-out R2={layer_fp32:.4f}, bf16={layer_bf16:.4f}",
            flush=True,
        )
        del output_input_weight, bias, bf16_weight, bf16_bias
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    return torch.stack(weights), torch.stack(biases), fp32_r2, bf16_r2


def fit(
    source_root: Path,
    target_root: Path,
    selection_report: Path,
    sequences_path: Path,
    out: Path,
    regularization: float,
    device: str,
    chunk: int,
    sensitivity_root: Path | None = None,
) -> dict:
    source_manifest, source = _open_consolidated(source_root)
    target_manifest, target = _open_consolidated(target_root)
    if source_manifest["shape"] != target_manifest["shape"]:
        raise ValueError("source and target consolidated KV shapes differ")
    selection_document = json.loads(selection_report.read_text())
    selection = torch.tensor(selection_document["selection"], dtype=torch.int32)
    _validate_selection(selection, source_manifest["shape"][1], target_manifest["shape"][1])
    fit_rows = tuple(source_manifest["splits"]["calibration"])
    eval_rows = tuple(source_manifest["splits"]["validation"])
    rows = [json.loads(line) for line in sequences_path.read_text().splitlines() if line.strip()]
    fit_ids = [row["seq_id"] for row in rows if row["split"] == "calibration"]
    eval_ids = [row["seq_id"] for row in rows if row["split"] == "validation"]

    out.mkdir(parents=True, exist_ok=True)
    attention_weights: dict[str, np.memmap] = {}
    attention_report: dict | None = None
    if sensitivity_root is not None:
        manifest_path = sensitivity_root / "manifest.json"
        attention_manifest = json.loads(manifest_path.read_text())
        if attention_manifest.get("layout") != (
            "observation,target-layer,kv-head,key-or-value"
        ):
            raise ValueError("attention sensitivity layout is unsupported")
        if attention_manifest.get("dtype") != "float32":
            raise ValueError("attention sensitivities must be float32")
        expected_target = target_manifest["source"]
        if (
            attention_manifest.get("model") != expected_target["model_id"]
            or attention_manifest.get("revision") != expected_target["revision"]
        ):
            raise ValueError("attention sensitivity target identity differs from target KV")
        if attention_manifest.get("sequenceManifestSha256") != _sha256(sequences_path):
            raise ValueError("attention sensitivity sequence manifest differs from calibration")
        if attention_manifest.get("sequenceIds") != fit_ids:
            raise ValueError("attention sensitivity sequence ordering differs from calibration")
        if attention_manifest.get("sampledPositions") != source_manifest["sampledPositions"]:
            raise ValueError("attention sensitivity positions differ from calibration KV")
        sensitivity_shape = tuple(attention_manifest["shape"])
        expected_geometry = (
            selection.shape[0],
            source_manifest["shape"][2],
            2,
        )
        if sensitivity_shape[1:] != expected_geometry:
            raise ValueError("attention sensitivity shape does not match target KV geometry")
        if sensitivity_shape[0] > fit_rows[1] - fit_rows[0]:
            raise ValueError("attention observations exceed the calibration split")
        sensitivity_path = sensitivity_root / attention_manifest["file"]["name"]
        if sensitivity_path.stat().st_size != attention_manifest["file"]["bytes"]:
            raise ValueError("attention sensitivity byte size differs from its manifest")
        if _sha256(sensitivity_path) != attention_manifest["file"]["sha256"]:
            raise ValueError("attention sensitivity SHA-256 differs from its manifest")
        raw = np.memmap(
            sensitivity_path,
            mode="r",
            dtype=np.float32,
            shape=sensitivity_shape,
        )
        fit_rows = (fit_rows[0], fit_rows[0] + sensitivity_shape[0])
        input_dimensions = (
            selection.shape[1]
            * source_manifest["shape"][2]
            * source_manifest["shape"][3]
        )
        effective_sizes: dict[str, list[float]] = {"keys": [], "values": []}
        for kind_index, kind in enumerate(("keys", "values")):
            path = out / f"shrunk-{kind}.f32"
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
                    shrunk = shrink_weights(values, input_dimensions)
                    shaped[:, target_layer, head] = shrunk.numpy()
                    effective_sizes[kind].append(effective_sample_size(shrunk))
            shaped.flush()
            attention_weights[kind] = shaped
        attention_report = {
            "method": "per-target-head receiver-attention weights with ESS shrinkage",
            "manifestSha256": _sha256(manifest_path),
            "sensitivityFile": attention_manifest["file"],
            "fitObservations": sensitivity_shape[0],
            "parameterCountPerTargetHead": input_dimensions,
            "effectiveSampleSize": {
                kind: {
                    "min": min(values),
                    "mean": sum(values) / len(values),
                    "max": max(values),
                }
                for kind, values in effective_sizes.items()
            },
        }

    kind_weights = []
    kind_biases = []
    metrics = {}
    for kind in ("keys", "values"):
        weights, biases, fp32, bf16 = fit_full_kind(
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
        kind_weights.append(weights)
        kind_biases.append(biases)
        metrics[kind] = {
            "heldoutR2": sum(fp32) / len(fp32),
            "bf16HeldoutR2": sum(bf16) / len(bf16),
            "bf16Delta": (sum(fp32) - sum(bf16)) / len(fp32),
            "heldoutR2PerTargetLayer": fp32,
        }

    artifact = out / "mapper.safetensors"
    # [target-layer, K-or-V, output, input] is the zero-copy Java matrix layout.
    stacked_weights = torch.stack(kind_weights, dim=1).contiguous()
    stacked_biases = torch.stack(kind_biases, dim=1).contiguous()
    save_file(
        {"selection": selection.contiguous(), "weight": stacked_weights, "bias": stacked_biases},
        str(artifact),
        metadata={
            "kind": "full-cross-head-cross-model-kv-ridge",
            "weight_layout": "target-layer,key-or-value,output,input",
        },
    )
    report = {
        "schemaVersion": 1,
        "kind": "full-cross-head-cross-model-kv-ridge",
        "method": (
            "receiver-attention-weighted centered affine ridge with all source KV heads "
            "visible to every target KV head"
            if attention_report is not None
            else "centered affine ridge with all source KV heads visible to every target KV head"
        ),
        "source": source_manifest["source"],
        "target": target_manifest["source"],
        "topK": int(selection.shape[1]),
        "lambda": regularization,
        "contentSpaceKeys": True,
        "weightsDtype": "bfloat16",
        "weightLayout": "target-layer,key-or-value,output,input",
        "selection": selection.tolist(),
        "fit": {"split": "calibration", "sequenceIds": fit_ids, "rows": list(fit_rows)},
        "evaluation": {"split": "validation", "sequenceIds": eval_ids, "rows": list(eval_rows)},
        "attentionWeighting": attention_report,
        "sequenceManifestSha256": _sha256(sequences_path),
        "selectionReportSha256": _sha256(selection_report),
        "metrics": metrics,
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
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--target", required=True, type=Path)
    parser.add_argument("--selection-report", required=True, type=Path)
    parser.add_argument("--sequences", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--lambda", dest="regularization", type=float, default=0.01)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--chunk", type=int, default=128)
    parser.add_argument("--sensitivities", type=Path, default=None)
    args = parser.parse_args()
    started = time.time()
    result = fit(
        args.source,
        args.target,
        args.selection_report,
        args.sequences,
        args.out,
        args.regularization,
        args.device,
        args.chunk,
        args.sensitivities,
    )
    print(
        json.dumps(
            {
                "elapsedSeconds": round(time.time() - started, 3),
                "artifact": result["artifact"],
                "metrics": result["metrics"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
