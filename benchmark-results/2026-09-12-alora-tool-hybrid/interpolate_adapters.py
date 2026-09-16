#!/usr/bin/env python3
"""Interpolate two provenance-bound checkpoints of the same activated LoRA adapter."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from prepare_tool_data import canonical_json
from train_alora import INVOCATION_STRING, INVOCATION_TOKENS, sha256, resolve_initial_adapter


def interpolate_state_dicts(
    left: dict[str, Any], right: dict[str, Any], fraction: float
) -> dict[str, Any]:
    if not 0.0 < fraction < 1.0:
        raise ValueError("interpolation fraction must be strictly between zero and one")
    if left.keys() != right.keys():
        raise ValueError("adapter tensor keys differ")
    result = {}
    for name in left:
        if left[name].shape != right[name].shape:
            raise ValueError(f"adapter tensor shape differs for {name}")
        result[name] = left[name].float().lerp(right[name].float(), fraction).to(left[name].dtype)
    return result


def _manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read adapter manifest: {error}") from error
    if not isinstance(value, dict):
        raise ValueError("adapter manifest must be an object")
    return value


def interpolate(
    *,
    left_adapter: Path,
    left_manifest_path: Path,
    right_adapter: Path,
    right_manifest_path: Path,
    fraction: float,
    output: Path,
) -> dict[str, Any]:
    if output.exists():
        raise ValueError(f"output already exists: {output}")
    left_manifest = _manifest(left_manifest_path)
    right_manifest = _manifest(right_manifest_path)
    try:
        base = left_manifest["base"]
        adapter_metadata = left_manifest["adapter"]
        model = base["model"]
        revision = base["revision"]
        rank = adapter_metadata["rank"]
        alpha = adapter_metadata["alpha"]
    except (KeyError, TypeError) as error:
        raise ValueError("left adapter manifest is missing its contract") from error
    if right_manifest.get("base") != base:
        raise ValueError("adapter base identities differ")
    if {
        key: right_manifest.get("adapter", {}).get(key)
        for key in ("rank", "alpha", "targetModules")
    } != {key: adapter_metadata.get(key) for key in ("rank", "alpha", "targetModules")}:
        raise ValueError("adapter contracts differ")
    left_identity = resolve_initial_adapter(
        left_adapter, left_manifest_path, model, revision, rank, alpha
    )
    right_identity = resolve_initial_adapter(
        right_adapter, right_manifest_path, model, revision, rank, alpha
    )

    from safetensors.torch import load_file, save_file

    left_weights = left_adapter / "adapter_model.safetensors"
    right_weights = right_adapter / "adapter_model.safetensors"
    state = interpolate_state_dicts(
        load_file(left_weights), load_file(right_weights), fraction
    )
    output_adapter = output / "adapter"
    output_adapter.mkdir(parents=True)
    for entry in adapter_metadata["files"]:
        name = entry["name"]
        if name != "adapter_model.safetensors":
            shutil.copy2(left_adapter / name, output_adapter / name)
    save_file(state, output_adapter / "adapter_model.safetensors")
    files = [
        {"name": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in sorted(output_adapter.iterdir())
        if path.is_file()
    ]
    manifest = {
        "schemaVersion": 1,
        "kind": "activated-lora-tool-specialist",
        "base": base,
        "invocation": {"string": INVOCATION_STRING, "tokens": INVOCATION_TOKENS},
        "adapter": {
            "rank": rank,
            "alpha": alpha,
            "dropout": adapter_metadata.get("dropout", 0.0),
            "targetModules": adapter_metadata["targetModules"],
            "files": files,
        },
        "derivation": {
            "kind": "parameter-linear-interpolation",
            "rightFraction": fraction,
            "left": left_identity,
            "right": right_identity,
        },
    }
    (output / "training-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-adapter", required=True, type=Path)
    parser.add_argument("--left-manifest", required=True, type=Path)
    parser.add_argument("--right-adapter", required=True, type=Path)
    parser.add_argument("--right-manifest", required=True, type=Path)
    parser.add_argument("--fraction", required=True, type=float)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    manifest = interpolate(
        left_adapter=args.left_adapter,
        left_manifest_path=args.left_manifest,
        right_adapter=args.right_adapter,
        right_manifest_path=args.right_manifest,
        fraction=args.fraction,
        output=args.out,
    )
    print(canonical_json(manifest))


if __name__ == "__main__":
    main()
