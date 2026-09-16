#!/usr/bin/env python3
"""Build a fail-closed, provenance-bound runtime aLoRA directory."""

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
NON_TOKENIZER_FILES = {"README.md", "adapter_config.json", ADAPTER_WEIGHTS}
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


def require_hash(value: Any, name: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-fA-F]{64}", value) is None:
        raise ValueError(f"{name} must be a 64-character SHA-256")
    return value.lower()


def require_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must not be blank")
    return value


def require_revision(value: str, name: str) -> str:
    if re.fullmatch(r"[0-9a-fA-F]{40}", value) is None:
        raise ValueError(f"{name} must be a pinned 40-character commit hash")
    return value.lower()


def require_selection(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")

    def integer(field: str, *, positive: bool = False) -> int:
        number = value.get(field)
        if not isinstance(number, int) or isinstance(number, bool):
            raise ValueError(f"{name}.{field} must be an integer")
        if number < (1 if positive else 0):
            comparison = "positive" if positive else "nonnegative"
            raise ValueError(f"{name}.{field} must be {comparison}")
        return number

    usable = integer("usable", positive=True)
    skipped = integer("skippedOverMaxLength")
    no_call = integer("noCall")
    multiple = integer("multipleCall")
    if no_call > usable:
        raise ValueError(f"{name}.noCall must not exceed usable")
    if multiple > usable:
        raise ValueError(f"{name}.multipleCall must not exceed usable")
    return {
        "usable": usable,
        "skippedOverMaxLength": skipped,
        "noCall": no_call,
        "multipleCall": multiple,
        "sourceLinesSha256": require_hash(
            value.get("sourceLinesSha256"), f"{name}.sourceLinesSha256"
        ),
    }


def require_relative_file(value: str, name: str) -> str:
    path = Path(value)
    if not value or path.is_absolute() or ".." in path.parts or "." in path.parts or "\\" in value:
        raise ValueError(f"{name} must be a normalized relative path")
    return value


def package_adapter(
    *,
    training_manifest: Path,
    preparation_manifest: Path,
    adapter_directory: Path,
    base_artifact: Path,
    dataset: str,
    dataset_revision: str,
    source_file: str,
    formatter: Path,
    notice: Path,
    output_directory: Path,
    positive_parent_manifest: Path | None = None,
    negative_parent_manifest: Path | None = None,
) -> dict[str, Any]:
    training = load_json(training_manifest)
    prepared = load_json(preparation_manifest)
    if training.get("schemaVersion") != 1:
        raise ValueError("unsupported training-manifest schemaVersion")
    if training.get("kind") != "activated-lora-tool-specialist":
        raise ValueError("unsupported training-manifest kind")
    prepared_schema = prepared.get("schemaVersion")
    if prepared_schema not in {1, 2}:
        raise ValueError("unsupported preparation-manifest schemaVersion")
    dataset = require_text(dataset, "dataset")
    dataset_revision = require_revision(dataset_revision, "dataset revision")
    source_file = require_relative_file(source_file, "source file")
    if output_directory.exists():
        raise ValueError(f"output already exists: {output_directory}")
    base_license = notice.with_name("LICENSE")
    for path, name in (
        (adapter_directory, "adapter directory"),
        (base_artifact, "base artifact"),
        (formatter, "formatter"),
        (notice, "attribution notice"),
        (base_license, "base license"),
    ):
        if not path.exists():
            raise ValueError(f"{name} does not exist: {path}")

    data = training["data"]
    declared_prepared_schema = data.get("preparedSchemaVersion")
    if prepared_schema == 2 and declared_prepared_schema != prepared_schema:
        raise ValueError("training and preparation schemaVersion differ")
    if declared_prepared_schema is not None and declared_prepared_schema != prepared_schema:
        raise ValueError("training and preparation schemaVersion differ")
    expected_prepared_hash = require_hash(
        data["preparedManifestSha256"], "prepared manifest hash"
    )
    if sha256(preparation_manifest) != expected_prepared_hash:
        raise ValueError("preparation manifest hash differs from training manifest")
    declared_sources = prepared.get("trainingSources")
    if declared_sources is not None:
        if prepared_schema != 2 or not isinstance(declared_sources, list) or not declared_sources:
            raise ValueError("prepared trainingSources requires a nonempty schema-2 list")
        training_sources = []
        roles: set[str] = set()
        for index, declared in enumerate(declared_sources):
            if not isinstance(declared, dict):
                raise ValueError(f"prepared training source {index} must be an object")
            role = require_text(declared.get("role"), f"prepared training source {index} role")
            if role in roles:
                raise ValueError(f"duplicate prepared training source role: {role}")
            roles.add(role)
            training_sources.append(
                {
                    "role": role,
                    "dataset": require_text(
                        declared.get("repository"),
                        f"prepared training source {index} repository",
                    ),
                    "datasetRevision": require_revision(
                        declared.get("revision"),
                        f"prepared training source {index} revision",
                    ),
                    "sourceFile": require_relative_file(
                        declared.get("path"), f"prepared training source {index} file"
                    ),
                    "sourceSha256": require_hash(
                        declared.get("sha256"), f"prepared training source {index} hash"
                    ),
                }
            )
        primary = [source for source in training_sources if source["role"] == "tool-calls"]
        if len(primary) != 1:
            raise ValueError("prepared trainingSources must contain exactly one tool-calls source")
        if primary[0]["dataset"] != dataset:
            raise ValueError("dataset differs from preparation manifest")
        if primary[0]["datasetRevision"] != dataset_revision:
            raise ValueError("dataset revision differs from preparation manifest")
        if primary[0]["sourceFile"] != source_file:
            raise ValueError("source file differs from preparation manifest")
    elif prepared.get("parents") is not None:
        if prepared_schema != 2 or not isinstance(prepared["parents"], dict):
            raise ValueError("prepared parents requires a schema-2 object")
        if positive_parent_manifest is None or negative_parent_manifest is None:
            raise ValueError("merged preparation requires both parent manifest files")

        def resolve_parent(
            label: str, path: Path, source_field: str, role: str
        ) -> dict[str, str]:
            declared_parent = prepared["parents"].get(label)
            if not isinstance(declared_parent, dict):
                raise ValueError(f"prepared {label} parent must be an object")
            expected_manifest_hash = require_hash(
                declared_parent.get("manifestSha256"), f"{label} parent manifest hash"
            )
            if sha256(path) != expected_manifest_hash:
                raise ValueError(f"{label} parent manifest SHA-256 differs")
            parent = load_json(path)
            if parent.get("schemaVersion") != 2:
                raise ValueError(f"{label} parent manifest must use schemaVersion 2")
            for split in ("train", "validation"):
                expected_split = require_hash(
                    declared_parent.get(f"{split}Sha256"), f"{label} parent {split} hash"
                )
                actual_split = require_hash(
                    parent.get(split, {}).get("sha256"), f"{label} parent {split} manifest hash"
                )
                if actual_split != expected_split:
                    raise ValueError(f"{label} parent {split} SHA-256 differs")
            source = parent.get(source_field)
            if not isinstance(source, dict):
                raise ValueError(f"{label} parent {source_field} must be an object")
            return {
                "role": role,
                "dataset": require_text(source.get("repository"), f"{label} source repository"),
                "datasetRevision": require_revision(
                    source.get("revision"), f"{label} source revision"
                ),
                "sourceFile": require_relative_file(
                    source.get("path"), f"{label} source file"
                ),
                "sourceSha256": require_hash(
                    source.get("sha256"), f"{label} source hash"
                ),
            }

        training_sources = [
            resolve_parent(
                "positive", positive_parent_manifest, "trainingSource", "tool-calls"
            ),
            resolve_parent("negative", negative_parent_manifest, "source", "hard-no-call"),
        ]
        primary = training_sources[0]
        if primary["dataset"] != dataset:
            raise ValueError("dataset differs from preparation manifest")
        if primary["datasetRevision"] != dataset_revision:
            raise ValueError("dataset revision differs from preparation manifest")
        if primary["sourceFile"] != source_file:
            raise ValueError("source file differs from preparation manifest")
    else:
        prepared_source_hash = require_hash(
            prepared["trainingSource"]["sha256"], "training source hash"
        )
        prepared_source_file = require_relative_file(
            prepared["trainingSource"]["path"], "prepared training source file"
        )
        if (
            (prepared_schema == 2 and prepared_source_file != source_file)
            or (
                prepared_schema == 1
                and Path(prepared_source_file).name != Path(source_file).name
            )
        ):
            raise ValueError("source file differs from preparation manifest")
        training_sources = [
            {
                "role": "tool-calls",
                "dataset": dataset,
                "datasetRevision": dataset_revision,
                "sourceFile": source_file,
                "sourceSha256": prepared_source_hash,
            }
        ]
        if prepared_schema == 2:
            declared_training_source = prepared["trainingSource"]
            if declared_training_source.get("repository") != dataset:
                raise ValueError("dataset differs from preparation manifest")
            if require_revision(
                declared_training_source.get("revision"), "prepared training source revision"
            ) != dataset_revision:
                raise ValueError("dataset revision differs from preparation manifest")
            irrelevance_source = prepared.get("irrelevanceSource")
            if isinstance(irrelevance_source, dict):
                training_sources.append(
                    {
                        "role": "no-call",
                        "dataset": require_text(
                            irrelevance_source["repository"],
                            "irrelevance source repository",
                        ),
                        "datasetRevision": require_revision(
                            irrelevance_source["revision"], "irrelevance source revision"
                        ),
                        "sourceFile": require_relative_file(
                            irrelevance_source["path"], "irrelevance source file"
                        ),
                        "sourceSha256": require_hash(
                            irrelevance_source["sha256"], "irrelevance source hash"
                        ),
                    }
                )
    train_hash = require_hash(prepared["train"]["sha256"], "prepared train hash")
    validation_hash = require_hash(
        prepared["validation"]["sha256"], "prepared validation hash"
    )
    if require_hash(data["train"]["sha256"], "training train hash") != train_hash:
        raise ValueError("train split hash differs between preparation and training manifests")
    if require_hash(data["validation"]["sha256"], "training validation hash") != validation_hash:
        raise ValueError("validation split hash differs between preparation and training manifests")
    train_selection = require_selection(data["train"], "training train selection")
    validation_selection = require_selection(
        data["validation"], "training validation selection"
    )

    declared_files = training["adapter"]["files"]
    declared_names: set[str] = set()
    for declared in declared_files:
        name = require_relative_file(declared["name"], "adapter file")
        if "/" in name or name in declared_names:
            raise ValueError(f"duplicate or nested adapter file: {name}")
        declared_names.add(name)
        actual = adapter_directory / name
        if not actual.is_file():
            raise ValueError(f"declared adapter file does not exist: {name}")
        if actual.stat().st_size != declared["bytes"]:
            raise ValueError(f"adapter file size differs from training manifest: {name}")
        if sha256(actual) != require_hash(declared["sha256"], f"adapter file {name}"):
            raise ValueError(f"adapter file hash differs from training manifest: {name}")
    actual_names = {path.name for path in adapter_directory.iterdir() if path.is_file()}
    if actual_names != declared_names:
        raise ValueError(
            "adapter directory differs from training manifest: "
            f"missing={sorted(declared_names - actual_names)}, "
            f"unexpected={sorted(actual_names - declared_names)}"
        )
    if ADAPTER_WEIGHTS not in actual_names:
        raise ValueError(f"training output does not contain {ADAPTER_WEIGHTS}")

    tokenizer_files = sorted(actual_names - NON_TOKENIZER_FILES)
    if not tokenizer_files:
        raise ValueError("training output does not contain tokenizer files")
    tokenizer = [
        {"name": name, "sha256": sha256(adapter_directory / name)}
        for name in tokenizer_files
    ]
    adapter = training["adapter"]
    target_modules = adapter["targetModules"]
    if len(target_modules) != len(TARGET_MODULES) or set(target_modules) != TARGET_MODULES:
        raise ValueError(f"adapter target modules must be exactly {sorted(TARGET_MODULES)}")
    rank = int(adapter["rank"])
    alpha = int(adapter["alpha"])
    if rank <= 0 or alpha <= 0:
        raise ValueError("adapter rank and alpha must be positive")
    invocation_tokens = training["invocation"]["tokens"]
    if not invocation_tokens or any(not isinstance(token, int) or token < 0 for token in invocation_tokens):
        raise ValueError("invocation tokens must be nonnegative integers")

    metadata = {
        "schemaVersion": 4,
        "kind": "activated-lora-tool-specialist",
        "base": {
            "model": training["base"]["model"],
            "revision": require_revision(training["base"]["revision"], "base revision"),
            "artifactSha256": sha256(base_artifact),
        },
        "tokenizer": {"files": tokenizer},
        "adapter": {
            "file": ADAPTER_WEIGHTS,
            "sha256": sha256(adapter_directory / ADAPTER_WEIGHTS),
            "rank": rank,
            "alpha": alpha,
            "targetModules": target_modules,
        },
        "invocation": {"tokens": invocation_tokens},
        "training": {
            "sources": training_sources,
            "preparedSchemaVersion": prepared_schema,
            "preparedManifestSha256": expected_prepared_hash,
            "trainSha256": train_hash,
            "validationSha256": validation_hash,
            "trainSelection": train_selection,
            "validationSelection": validation_selection,
            "formatter": formatter.name,
            "formatterSha256": sha256(formatter),
        },
    }

    output_directory.mkdir(parents=True)
    shutil.copy2(adapter_directory / ADAPTER_WEIGHTS, output_directory / ADAPTER_WEIGHTS)
    shutil.copy2(base_license, output_directory / "LICENSE")
    shutil.copy2(notice, output_directory / "NOTICE")
    (output_directory / RUNTIME_METADATA).write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-manifest", required=True, type=Path)
    parser.add_argument("--preparation-manifest", required=True, type=Path)
    parser.add_argument("--adapter", required=True, type=Path)
    parser.add_argument("--base-artifact", required=True, type=Path)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--source-file", required=True)
    parser.add_argument("--formatter", required=True, type=Path)
    parser.add_argument("--notice", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--positive-parent-manifest", type=Path)
    parser.add_argument("--negative-parent-manifest", type=Path)
    args = parser.parse_args()
    metadata = package_adapter(
        training_manifest=args.training_manifest.resolve(),
        preparation_manifest=args.preparation_manifest.resolve(),
        adapter_directory=args.adapter.resolve(),
        base_artifact=args.base_artifact.resolve(),
        dataset=args.dataset,
        dataset_revision=args.dataset_revision,
        source_file=args.source_file,
        formatter=args.formatter.resolve(),
        notice=args.notice.resolve(),
        output_directory=args.out.resolve(),
        positive_parent_manifest=(
            args.positive_parent_manifest.resolve() if args.positive_parent_manifest else None
        ),
        negative_parent_manifest=(
            args.negative_parent_manifest.resolve() if args.negative_parent_manifest else None
        ),
    )
    print(json.dumps(metadata, sort_keys=True))


if __name__ == "__main__":
    main()
