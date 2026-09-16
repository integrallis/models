#!/usr/bin/env python3
"""Freeze one Hugging Face Qwen policy prompt for Java byte/token parity."""

from __future__ import annotations

import argparse
import hashlib
import struct
from pathlib import Path
from typing import Any

from evaluate_alora import apply_system_policy, load_system_policy, resolve_adapter_identity, sha256
from generate_projection_oracle import (
    TOKENIZER_PROBE_MESSAGES,
    TOKENIZER_PROBE_TOOLS,
    properties_text,
)


INVOCATION_TOKENS = [151644, 77091, 198]


def _contains(tokens: list[int], sequence: list[int]) -> bool:
    return any(
        tokens[offset : offset + len(sequence)] == sequence
        for offset in range(len(tokens) - len(sequence) + 1)
    )


def policy_tokenizer_probe(tokenizer: Any, policy: str) -> tuple[str, list[int]]:
    """Render the canonical tool turn with exactly one specialist policy system message."""
    messages = apply_system_policy(TOKENIZER_PROBE_MESSAGES, policy)
    prompt = tokenizer.apply_chat_template(
        messages,
        tools=TOKENIZER_PROBE_TOOLS,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    token_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("policy tokenizer probe produced an empty prompt")
    if prompt.count(policy) != 1 or "# Tools" not in prompt or prompt.index(policy) > prompt.index("# Tools"):
        raise ValueError("policy must occur exactly once before the tool declarations")
    if not _contains(token_ids, INVOCATION_TOKENS):
        raise ValueError("policy prompt does not retain the exact activated invocation sequence")
    return prompt, token_ids


def generate(
    adapter: Path,
    training_manifest: Path,
    policy_path: Path,
    output_directory: Path,
) -> None:
    from transformers import AutoTokenizer

    if output_directory.exists():
        raise ValueError(f"output already exists: {output_directory}")
    identity = resolve_adapter_identity(adapter, training_manifest)
    policy = load_system_policy(policy_path)
    tokenizer = AutoTokenizer.from_pretrained(
        identity["model"], revision=identity["revision"]
    )
    prompt, token_ids = policy_tokenizer_probe(tokenizer, policy)
    output_directory.mkdir(parents=True)
    prompt_path = output_directory / "policy-prompt.txt"
    tokens_path = output_directory / "policy-token-ids.i32le"
    prompt_path.write_text(prompt)
    tokens_path.write_bytes(struct.pack(f"<{len(token_ids)}i", *token_ids))
    properties = {
        "schema.version": 1,
        "base.model": identity["model"],
        "base.revision": identity["revision"],
        "adapter.sha256": identity["adapterSha256"],
        "training.manifest.sha256": identity["trainingManifestSha256"],
        "policy.file.sha256": sha256(policy_path),
        "policy.content.sha256": hashlib.sha256(policy.encode()).hexdigest(),
        "prompt.file": prompt_path.name,
        "prompt.bytes": prompt_path.stat().st_size,
        "prompt.sha256": sha256(prompt_path),
        "tokens.file": tokens_path.name,
        "tokens.count": len(token_ids),
        "tokens.sha256": sha256(tokens_path),
        "invocation.tokens": ",".join(str(token) for token in INVOCATION_TOKENS),
    }
    (output_directory / "policy-prompt-oracle.properties").write_text(
        properties_text(properties)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=True, type=Path)
    parser.add_argument("--training-manifest", required=True, type=Path)
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    generate(
        args.adapter.resolve(),
        args.training_manifest.resolve(),
        args.policy.resolve(),
        args.out.resolve(),
    )


if __name__ == "__main__":
    main()
