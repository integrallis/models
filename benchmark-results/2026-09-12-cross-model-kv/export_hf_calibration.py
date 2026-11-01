#!/usr/bin/env python3
"""Export pinned Hugging Face decoder KV tensors into the audited calibration layout.

This is an offline calibration/oracle utility. ModelJars production inference does not load
Transformers, PyTorch, vLLM, or any other external inference runtime.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from huggingface_hub import HfApi, snapshot_download
from transformers import AutoModelForCausalLM

from cross_model_kv.calibration import CalibrationExportWriter
from cross_model_kv.identity import ModelIdentity
from cross_model_kv.rope import RopeSpec


def canonical_layers(past_key_values) -> list[torch.Tensor]:
    """Return each cache layer as [token, K-or-V, KV-head, head-dimension]."""
    legacy = (
        past_key_values.to_legacy_cache()
        if hasattr(past_key_values, "to_legacy_cache")
        else past_key_values
    )
    result: list[torch.Tensor] = []
    for layer, pair in enumerate(legacy):
        if len(pair) < 2:
            raise ValueError(f"cache layer {layer} does not contain key and value tensors")
        key, value = pair[0], pair[1]
        if key.ndim != 4 or value.ndim != 4:
            raise ValueError(f"cache layer {layer} must use [batch, head, token, dimension]")
        if key.shape[0] != 1 or value.shape[0] != 1:
            raise ValueError("calibration export requires batch size 1")
        if key.shape != value.shape:
            raise ValueError(f"cache layer {layer} key/value shapes differ")
        result.append(
            torch.stack((key[0].transpose(0, 1), value[0].transpose(0, 1)), dim=1)
        )
    if not result:
        raise ValueError("model returned no KV cache layers")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--sequences", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--split", default=None)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checksums", action="store_true")
    args = parser.parse_args()

    if args.stride < 1:
        parser.error("--stride must be at least 1")

    revision = args.revision or HfApi().model_info(args.model).sha
    snapshot = Path(snapshot_download(args.model, revision=revision))
    config_doc = json.loads((snapshot / "config.json").read_text())
    identity = ModelIdentity.from_model_dir(
        snapshot,
        "bfloat16",
        model_id=args.model,
        revision=revision,
    )
    rope = RopeSpec.from_hf_config(config_doc).to_dict()
    model = AutoModelForCausalLM.from_pretrained(
        snapshot,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    ).to(args.device)
    model.eval()

    sequences = [
        json.loads(line)
        for line in Path(args.sequences).read_text().splitlines()
        if line.strip()
    ]
    if args.split:
        sequences = [sequence for sequence in sequences if sequence["split"] == args.split]
    if not sequences:
        raise SystemExit("no matching sequences")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for index, sequence in enumerate(sequences, start=1):
        token_ids = sequence["token_ids"]
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=args.device)
        with torch.inference_mode():
            outputs = model(input_ids=input_ids, use_cache=True, return_dict=True)
        layers = canonical_layers(outputs.past_key_values)
        if len(layers) != identity.num_layers:
            raise RuntimeError(
                f"{sequence['seq_id']}: model returned {len(layers)} layers; "
                f"identity declares {identity.num_layers}"
            )
        writer = CalibrationExportWriter(
            export_root=out,
            sequence_id=sequence["seq_id"],
            source=identity,
            token_ids_sha256=sequence["token_ids_sha256"],
            num_prompt_tokens=len(token_ids),
            stride=args.stride,
            dtype="bfloat16",
            rope=rope,
            checksums=args.checksums,
        )
        writer.begin()
        try:
            for layer_index, canonical in enumerate(layers):
                writer.add_layer(layer_index, canonical)
            writer.finalize()
        except BaseException:
            writer.abort()
            raise
        del outputs, layers, input_ids
        print(f"{args.model}: {sequence['seq_id']} ({index}/{len(sequences)})", flush=True)

    print(
        json.dumps(
            {
                "model": args.model,
                "revision": revision,
                "sequences": len(sequences),
                "stride": args.stride,
                "out": str(out),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
