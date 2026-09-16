#!/usr/bin/env python3
"""Generate exhaustive projection deltas for independent Java aLoRA verification."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path
from typing import Any


MODULES = [
    ("QUERY", "self_attn.q_proj"),
    ("KEY", "self_attn.k_proj"),
    ("VALUE", "self_attn.v_proj"),
    ("ATTENTION_OUTPUT", "self_attn.o_proj"),
    ("FFN_GATE", "mlp.gate_proj"),
    ("FFN_UP", "mlp.up_proj"),
    ("FFN_DOWN", "mlp.down_proj"),
]

TOKENIZER_PROBE_MESSAGES = [
    {"role": "user", "content": "What is the weather for 88252?"}
]
TOKENIZER_PROBE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get-weather-for-zipcode",
            "description": "Gets weather for a given zipcode",
            "parameters": {
                "type": "object",
                "properties": {"zipcode": {"type": "string"}},
                "required": ["zipcode"],
            },
        },
    }
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def properties_text(properties: dict[str, Any]) -> str:
    lines = ["# Generated aLoRA projection oracle; do not edit."]
    for name in sorted(properties):
        value = str(properties[name])
        if any(character in name + value for character in "\n\r"):
            raise ValueError("properties cannot contain line breaks")
        lines.append(f"{name}={value}")
    return "\n".join(lines) + "\n"


def tokenizer_probe(tokenizer: Any) -> tuple[str, list[int]]:
    """Render and tokenize the same Qwen tool-selection turn exercised by Java."""
    prompt = tokenizer.apply_chat_template(
        TOKENIZER_PROBE_MESSAGES,
        tools=TOKENIZER_PROBE_TOOLS,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    token_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("tokenizer probe produced an empty prompt")
    if not token_ids or any(
        not isinstance(token_id, int) or token_id < 0 for token_id in token_ids
    ):
        raise ValueError("tokenizer probe produced invalid token ids")
    return prompt, token_ids


def generate(adapter_directory: Path, output_directory: Path) -> None:
    import numpy as np
    from safetensors import safe_open
    from transformers import AutoTokenizer

    metadata_path = adapter_directory / "models-activated-lora.json"
    weights_path = adapter_directory / "adapter_model.safetensors"
    metadata = json.loads(metadata_path.read_text())
    rank = int(metadata["adapter"]["rank"])
    alpha = int(metadata["adapter"]["alpha"])
    expected_adapter_hash = metadata["adapter"]["sha256"]
    if sha256(weights_path) != expected_adapter_hash:
        raise ValueError("adapter hash differs from models-activated-lora.json")
    if output_directory.exists():
        raise ValueError(f"output already exists: {output_directory}")
    output_directory.mkdir(parents=True)

    properties: dict[str, Any] = {
        "schema.version": 1,
        "adapter.sha256": expected_adapter_hash,
        "rank": rank,
        "alpha": alpha,
        "max.absolute.error": "2.0e-4",
        "minimum.cosine": "0.999999",
    }

    base = metadata.get("base", {})
    base_model = base.get("model")
    base_revision = base.get("revision")
    if not isinstance(base_model, str) or not base_model:
        raise ValueError("adapter metadata is missing the base model")
    if not isinstance(base_revision, str) or len(base_revision) != 40:
        raise ValueError("adapter metadata is missing the exact base revision")
    tokenizer = AutoTokenizer.from_pretrained(base_model, revision=base_revision)
    rendered_prompt, token_ids = tokenizer_probe(tokenizer)
    prompt_path = output_directory / "tokenizer-prompt.txt"
    prompt_path.write_text(rendered_prompt)
    token_path = output_directory / "tokenizer-token-ids.i32le"
    token_path.write_bytes(struct.pack(f"<{len(token_ids)}i", *token_ids))
    properties.update(
        {
            "tokenizer.base.model": base_model,
            "tokenizer.base.revision": base_revision,
            "tokenizer.prompt.file": prompt_path.name,
            "tokenizer.prompt.bytes": prompt_path.stat().st_size,
            "tokenizer.prompt.sha256": sha256(prompt_path),
            "tokenizer.tokens.file": token_path.name,
            "tokenizer.tokens.count": len(token_ids),
            "tokenizer.tokens.sha256": sha256(token_path),
        }
    )
    binary_path = output_directory / "projection-oracle.f32le"
    float_offset = 0
    probe_count = 0
    expected_names: set[str] = set()
    with safe_open(weights_path, framework="numpy") as tensors, binary_path.open("wb") as binary:
        names = set(tensors.keys())
        layers = len(names) // (len(MODULES) * 2)
        if layers <= 0 or len(names) != layers * len(MODULES) * 2:
            raise ValueError(f"unexpected adapter tensor count: {len(names)}")
        properties["layers"] = layers
        for layer in range(layers):
            for module_index, (projection, module) in enumerate(MODULES):
                prefix = f"base_model.model.model.layers.{layer}.{module}.lora_"
                a_name = prefix + "A.weight"
                b_name = prefix + "B.weight"
                expected_names.update((a_name, b_name))
                a = tensors.get_tensor(a_name).astype(np.float32, copy=False)
                b = tensors.get_tensor(b_name).astype(np.float32, copy=False)
                if a.ndim != 2 or b.ndim != 2 or a.shape[0] != rank or b.shape[1] != rank:
                    raise ValueError(f"invalid low-rank shapes for {prefix}: {a.shape}, {b.shape}")
                input_dimension = int(a.shape[1])
                output_dimension = int(b.shape[0])
                indexes = np.arange(1, input_dimension + 1, dtype=np.float64)
                values = np.sin(
                    indexes * ((layer + 1) * 0.017) + (module_index + 1) * 0.113
                ).astype("<f4")
                delta = (b @ (a @ values) * np.float32(alpha / rank)).astype("<f4")
                values.tofile(binary)
                properties[f"probe.{probe_count}.input.offset"] = float_offset
                properties[f"probe.{probe_count}.input.length"] = input_dimension
                float_offset += input_dimension
                delta.tofile(binary)
                properties[f"probe.{probe_count}.output.offset"] = float_offset
                properties[f"probe.{probe_count}.output.length"] = output_dimension
                float_offset += output_dimension
                properties[f"probe.{probe_count}.layer"] = layer
                properties[f"probe.{probe_count}.projection"] = projection
                probe_count += 1
        if names != expected_names:
            raise ValueError(
                "adapter tensor set differs from expected graph: "
                f"missing={sorted(expected_names - names)}, unexpected={sorted(names - expected_names)}"
            )
    properties["probe.count"] = probe_count
    properties["binary.file"] = binary_path.name
    properties["binary.floats"] = float_offset
    properties["binary.sha256"] = sha256(binary_path)
    (output_directory / "projection-oracle.properties").write_text(properties_text(properties))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    generate(args.adapter.resolve(), args.out.resolve())


if __name__ == "__main__":
    main()
