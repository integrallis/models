#!/usr/bin/env python3
"""Package a publisher-supplied activated LoRA with a fail-closed runtime contract.

This is the Granite 4.1 generalization of the 2026-09-14 packager. Granite RAG Library
adapters (peft 0.18) declare their activation boundary as ``alora_invocation_tokens`` rather
than an ``invocation_string``, and keep their model card at the task level rather than inside
the adapter directory. Both forms are accepted, but every value is cross-checked and a mismatch
fails closed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any

RUNTIME_METADATA = "models-activated-lora.json"
ADAPTER_WEIGHTS = "adapter_model.safetensors"
TARGET_MODULES = {
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(), object_pairs_hook=reject_duplicate_keys)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def require_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must not be blank")
    return value


def require_revision(value: str, name: str) -> str:
    if re.fullmatch(r"[0-9a-fA-F]{40}", value) is None:
        raise ValueError(f"{name} must be a pinned 40-character commit hash")
    return value.lower()


def require_relative_file(value: str, name: str) -> str:
    path = Path(value)
    if not value or path.is_absolute() or ".." in path.parts or "." in path.parts or "\\" in value:
        raise ValueError(f"{name} must be a normalized relative path")
    return value


def require_positive_integer(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def require_target_modules(value: Any) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError("adapter_config.target_modules must be a nonempty array")
    if any(not isinstance(module, str) or module not in TARGET_MODULES for module in value):
        raise ValueError("adapter_config.target_modules contains an unsupported projection")
    if len(set(value)) != len(value):
        raise ValueError("adapter_config.target_modules must not contain duplicates")
    return list(value)


def require_token_ids(tokens: Any, name: str = "invocation tokens") -> list[int]:
    if (
        not isinstance(tokens, list)
        or not tokens
        or any(not isinstance(token, int) or isinstance(token, bool) or token < 0 for token in tokens)
    ):
        raise ValueError(f"{name} must be a nonempty array of nonnegative integers")
    return list(tokens)


def resolve_invocation(
    config: dict[str, Any], cli_tokens: list[int], cli_text: str | None
) -> tuple[str, list[int]]:
    """Cross-check every declared activation boundary; any disagreement fails closed."""
    tokens = require_token_ids(cli_tokens)
    declared_text = config.get("invocation_string")
    declared_tokens = config.get("alora_invocation_tokens")
    if declared_text is None and declared_tokens is None:
        raise ValueError(
            "adapter_config must declare invocation_string or alora_invocation_tokens"
        )
    if declared_tokens is not None:
        declared_tokens = require_token_ids(
            declared_tokens, "adapter_config.alora_invocation_tokens"
        )
        if declared_tokens != tokens:
            raise ValueError(
                "adapter_config.alora_invocation_tokens differ from the pinned invocation tokens"
            )
    if declared_text is not None:
        declared_text = require_text(declared_text, "adapter_config.invocation_string")
        if cli_text is not None and cli_text != declared_text:
            raise ValueError("--invocation-text differs from adapter_config.invocation_string")
        return declared_text, tokens
    if cli_text is None:
        raise ValueError(
            "adapter_config declares only alora_invocation_tokens; --invocation-text is required "
            "so the runtime can render the marker, and Java must prove it encodes to those tokens"
        )
    return require_text(cli_text, "--invocation-text"), tokens


def package_upstream_adapter(
    *,
    adapter_directory: Path,
    base_artifact: Path,
    base_model: str,
    base_revision: str,
    tokenizer_directory: Path,
    tokenizer_files: list[str],
    publisher: str,
    repository: str,
    revision: str,
    license_name: str,
    license_file: Path,
    notice_file: Path,
    invocation_tokens: list[int],
    output_directory: Path,
    invocation_text: str | None = None,
    model_card: Path | None = None,
) -> dict[str, Any]:
    if output_directory.exists():
        raise ValueError(f"output already exists: {output_directory}")
    for path, name in (
        (adapter_directory, "adapter directory"),
        (base_artifact, "base artifact"),
        (tokenizer_directory, "tokenizer directory"),
        (license_file, "license file"),
        (notice_file, "notice file"),
    ):
        if not path.exists():
            raise ValueError(f"{name} does not exist: {path}")
    if not adapter_directory.is_dir() or not tokenizer_directory.is_dir():
        raise ValueError("adapter and tokenizer paths must be directories")
    if not base_artifact.is_file() or not license_file.is_file() or not notice_file.is_file():
        raise ValueError("base artifact, license, and notice must be regular files")

    base_model = require_text(base_model, "base model")
    base_revision = require_revision(base_revision, "base revision")
    publisher = require_text(publisher, "upstream publisher")
    repository = require_text(repository, "upstream repository")
    revision = require_revision(revision, "upstream revision")
    license_name = require_text(license_name, "upstream license")

    config_path = adapter_directory / "adapter_config.json"
    card_path = model_card if model_card is not None else adapter_directory / "README.md"
    weights_path = adapter_directory / ADAPTER_WEIGHTS
    for path, name in (
        (config_path, "adapter configuration"),
        (card_path, "adapter model card"),
        (weights_path, "adapter weights"),
    ):
        if not path.is_file():
            raise ValueError(f"{name} does not exist: {path}")
    config = load_json(config_path)
    if config.get("peft_type") != "LORA":
        raise ValueError("adapter_config.peft_type must be LORA")
    if config.get("base_model_name_or_path") != base_model:
        raise ValueError("adapter_config base model differs from the pinned base model")
    invocation, tokens = resolve_invocation(config, invocation_tokens, invocation_text)
    rank = require_positive_integer(config.get("r"), "adapter_config.r")
    alpha = require_positive_integer(config.get("lora_alpha"), "adapter_config.lora_alpha")
    target_modules = require_target_modules(config.get("target_modules"))

    tokenizer_entries = []
    for filename in sorted(tokenizer_files):
        filename = require_relative_file(filename, "tokenizer file")
        path = tokenizer_directory / filename
        if not path.is_file():
            raise ValueError(f"tokenizer file does not exist: {path}")
        tokenizer_entries.append({"name": filename, "sha256": sha256(path)})
    if not tokenizer_entries:
        raise ValueError("at least one tokenizer file is required")

    metadata = {
        "schemaVersion": 5,
        "kind": "activated-lora-specialist",
        "base": {
            "model": base_model,
            "revision": base_revision,
            "artifactSha256": sha256(base_artifact),
        },
        "tokenizer": {"files": tokenizer_entries},
        "adapter": {
            "file": ADAPTER_WEIGHTS,
            "sha256": sha256(weights_path),
            "rank": rank,
            "alpha": alpha,
            "targetModules": target_modules,
        },
        "invocation": {"text": invocation, "tokens": tokens},
        "upstream": {
            "publisher": publisher,
            "repository": repository,
            "revision": revision,
            "modelCardSha256": sha256(card_path),
            "adapterConfigSha256": sha256(config_path),
            "license": license_name,
        },
    }

    output_directory.mkdir(parents=True)
    shutil.copy2(weights_path, output_directory / ADAPTER_WEIGHTS)
    shutil.copy2(license_file, output_directory / "LICENSE")
    shutil.copy2(notice_file, output_directory / "NOTICE")
    (output_directory / RUNTIME_METADATA).write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-directory", type=Path, required=True)
    parser.add_argument("--base-artifact", type=Path, required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--base-revision", required=True)
    parser.add_argument("--tokenizer-directory", type=Path, required=True)
    parser.add_argument("--tokenizer-file", action="append", required=True)
    parser.add_argument("--publisher", required=True)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--license", required=True)
    parser.add_argument("--license-file", type=Path, required=True)
    parser.add_argument("--notice-file", type=Path, required=True)
    parser.add_argument("--invocation-tokens", required=True)
    parser.add_argument(
        "--invocation-text",
        help="Marker text for adapters that declare only alora_invocation_tokens",
    )
    parser.add_argument(
        "--model-card",
        type=Path,
        help="Model card path when the publisher keeps it outside the adapter directory",
    )
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    try:
        tokens = json.loads(args.invocation_tokens)
    except json.JSONDecodeError as error:
        raise SystemExit(f"--invocation-tokens must be JSON: {error}") from error
    package_upstream_adapter(
        adapter_directory=args.adapter_directory,
        base_artifact=args.base_artifact,
        base_model=args.base_model,
        base_revision=args.base_revision,
        tokenizer_directory=args.tokenizer_directory,
        tokenizer_files=args.tokenizer_file,
        publisher=args.publisher,
        repository=args.repository,
        revision=args.revision,
        license_name=args.license,
        license_file=args.license_file,
        notice_file=args.notice_file,
        invocation_tokens=tokens,
        output_directory=args.output_directory,
        invocation_text=args.invocation_text,
        model_card=args.model_card,
    )


if __name__ == "__main__":
    main()
