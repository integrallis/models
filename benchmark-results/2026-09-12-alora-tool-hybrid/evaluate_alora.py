#!/usr/bin/env python3
"""Evaluate the pinned aLoRA/base pair on a deterministic held-out BFCL slice."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any

from prepare_tool_data import canonical_json, normalize_tools, query_fingerprint, validate_calls


SEED = 20260912
QUALIFICATION_COUNT = 100
TOOL_BLOCK = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
BFCL_REPOSITORY = "ShishirPatil/gorilla"
BFCL_REVISION = "c15b2a151662cac9839c96d7dfb1493b5329c975"
BFCL_FILES = {
    "BFCL_v3_simple.json": "fbc37b2ad252bf9af985582e0e07b456173fe627d957491472ea9cef5fb83158",
    "BFCL_v3_multiple.json": "aef168155ebd74b7ac2401198b201343bc7d16d7a3d7e0d4e6d8ee82c6969b2a",
    "BFCL_v3_irrelevance.json": "975f51c51f688649fd190078efd87081241e0a326f9114a2ea3c1ca2440d8690",
    "possible_answer/BFCL_v3_simple.json": "2911a2bc00df82c4f999ffa64fedb0164cb88e96212ffa5972087eb91fd496ee",
    "possible_answer/BFCL_v3_multiple.json": "244e00ce9395df948bcafc7bee64e8f9c87ef70887587d83cae45b13699f3047",
}
BFCL_STRING_PUNCTUATION = re.compile(r"[ ,./\-_*^]")


def inference_device_name(cuda_available: bool) -> str:
    return "cuda" if cuda_available else "cpu"


def parse_completion(completion: str) -> list[dict[str, Any]]:
    """Parse a completion containing only strict JSON tool-call blocks."""
    calls: list[dict[str, Any]] = []
    cursor = 0
    saw_block = False
    saw_empty = False
    for match in TOOL_BLOCK.finditer(completion):
        if completion[cursor : match.start()].strip():
            raise ValueError("completion contains content outside tool-call blocks")
        cursor = match.end()
        saw_block = True
        value = json.loads(match.group(1))
        if value == []:
            saw_empty = True
            continue
        if not isinstance(value, dict) or set(value) != {"name", "arguments"}:
            raise ValueError("tool-call JSON must contain exactly name and arguments")
        if not isinstance(value["name"], str) or not value["name"]:
            raise ValueError("tool-call name must be a non-empty string")
        if not isinstance(value["arguments"], dict) or not all(
            isinstance(name, str) for name in value["arguments"]
        ):
            raise ValueError("tool-call arguments must be a JSON object with string keys")
        calls.append(value)
    if not saw_block or completion[cursor:].strip():
        raise ValueError("completion must contain only complete tool-call blocks")
    if saw_empty and calls:
        raise ValueError("empty-call sentinel cannot be combined with calls")
    return calls


def _standardize_bfcl_string(value: str) -> str:
    """Apply the pinned BFCL AST checker's value normalization."""
    return BFCL_STRING_PUNCTUATION.sub("", value).lower().replace("'", '"')


def _dict_matches(
    actual: dict[str, Any], expected_alternatives: list[Any], schema: dict[str, Any]
) -> bool:
    properties = schema.get("properties", {})
    for candidate in expected_alternatives:
        if not isinstance(candidate, dict):
            continue
        if actual.keys() - candidate.keys():
            continue
        if any(
            not _value_matches(value, candidate[name], properties.get(name, {}))
            for name, value in actual.items()
        ):
            continue
        if any(name not in actual and "" not in alternatives for name, alternatives in candidate.items()):
            continue
        return True
    return False


def _value_matches(actual: Any, alternatives: list[Any], schema: dict[str, Any]) -> bool:
    if not isinstance(alternatives, list) or not alternatives:
        raise ValueError("ground-truth alternatives must be a non-empty list")
    expected_type = schema.get("type")
    if expected_type == "object":
        return isinstance(actual, dict) and _dict_matches(actual, alternatives, schema)
    if expected_type == "array":
        if not isinstance(actual, list):
            return False
        item_schema = schema.get("items", {})
        if item_schema.get("type") == "object":
            return any(
                isinstance(candidate, list)
                and len(actual) == len(candidate)
                and all(
                    isinstance(value, dict)
                    and _dict_matches(value, [expected_value], item_schema)
                    for value, expected_value in zip(actual, candidate)
                )
                for candidate in alternatives
            )
        normalized_actual = [
            _standardize_bfcl_string(value) if isinstance(value, str) else value
            for value in actual
        ]
        return any(
            isinstance(candidate, list)
            and normalized_actual
            == [
                _standardize_bfcl_string(value) if isinstance(value, str) else value
                for value in candidate
            ]
            for candidate in alternatives
        )
    if expected_type == "string":
        return isinstance(actual, str) and _standardize_bfcl_string(actual) in {
            _standardize_bfcl_string(value)
            for value in alternatives
            if isinstance(value, str)
        }
    if expected_type == "integer":
        return isinstance(actual, int) and not isinstance(actual, bool) and actual in alternatives
    if expected_type == "number":
        return (
            isinstance(actual, (int, float))
            and not isinstance(actual, bool)
            and any(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and float(actual) == float(value)
                for value in alternatives
            )
        )
    if expected_type == "boolean":
        return isinstance(actual, bool) and actual in alternatives
    return actual in alternatives


def _expected_call_matches(
    actual: dict[str, Any], expected: dict[str, Any], schema: dict[str, Any]
) -> bool:
    if len(expected) != 1:
        raise ValueError(f"ground-truth call must name exactly one function: {expected}")
    name, expected_arguments = next(iter(expected.items()))
    if actual["name"] != name or not isinstance(expected_arguments, dict):
        return False
    actual_arguments = actual["arguments"]
    if actual_arguments.keys() - expected_arguments.keys():
        return False
    properties = schema.get("properties", {})
    for argument, alternatives in expected_arguments.items():
        if not isinstance(alternatives, list) or not alternatives:
            raise ValueError(f"ground-truth alternatives must be a non-empty list: {argument}")
        if argument not in actual_arguments:
            if "" not in alternatives:
                return False
        elif not _value_matches(actual_arguments[argument], alternatives, properties.get(argument, {})):
            return False
    return True


def exact_calls_match(
    actual: list[dict[str, Any]],
    expected: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
) -> bool:
    """Compare unordered calls with the pinned BFCL AST checker's value rules."""
    if len(actual) != len(expected):
        return False
    schemas = {
        tool["function"]["name"]: tool["function"].get("parameters", {})
        for tool in (tools or [])
    }
    unmatched = list(expected)
    for call in actual:
        match_index = next(
            (
                index
                for index, candidate in enumerate(unmatched)
                if _expected_call_matches(call, candidate, schemas.get(call["name"], {}))
            ),
            None,
        )
        if match_index is None:
            return False
        unmatched.pop(match_index)
    return not unmatched


def select_cases(
    cases: list[dict[str, Any]], count: int, seed: int, offset: int = 0
) -> list[dict[str, Any]]:
    if count <= 0 or offset < 0 or offset + count > len(cases):
        raise ValueError(
            f"offset and count must select within [0, {len(cases)}]: "
            f"offset={offset}, count={count}"
        )
    ordered = sorted(
        cases,
        key=lambda case: hashlib.sha256(f"{seed}:{case['id']}".encode()).digest(),
    )
    return ordered[offset : offset + count]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as source:
        return [json.loads(line) for line in source if line.strip()]


def _materialize_cases(
    data: Path, kind: str, cases: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    if kind == "irrelevance":
        answers: dict[str, list[dict[str, Any]]] = {}
    else:
        answers = {
            answer["id"]: answer["ground_truth"]
            for answer in read_jsonl(data / "possible_answer" / f"BFCL_v3_{kind}.json")
        }
    selected = []
    for case in cases:
        raw_tools = case["function"]
        tools = normalize_tools(
            [
                {
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool.get("description", ""),
                        "parameters": tool.get(
                            "parameters", {"type": "object", "properties": {}}
                        ),
                    },
                }
                for tool in raw_tools
            ]
        )
        selected.append(
            {
                "id": case["id"],
                "kind": kind,
                "messages": case["question"][0],
                "tools": tools,
                "expected": answers.get(case["id"], []),
            }
        )
    return selected


def load_slice(data: Path, kind: str, count: int, offset: int = 0) -> list[dict[str, Any]]:
    cases = select_cases(
        read_jsonl(data / f"BFCL_v3_{kind}.json"), count, SEED, offset
    )
    return _materialize_cases(data, kind, cases)


def _case_query_hashes(case: dict[str, Any]) -> list[str]:
    hashes = {
        query_fingerprint(message["content"])
        for turn in case.get("question", [])
        for message in turn
        if message.get("role") == "user" and isinstance(message.get("content"), str)
    }
    return sorted(hashes)


def resolve_qualification_window(
    data: Path, manifest_path: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Verify and materialize the exact query-bound cases frozen before training."""
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read qualification-window manifest: {error}") from error
    if not isinstance(manifest, dict) or manifest.get("schemaVersion") != 1:
        raise ValueError("qualification window must use schemaVersion 1")
    if manifest.get("status") != "frozen-before-training":
        raise ValueError("qualification window was not frozen before training")

    source = manifest.get("source")
    if not isinstance(source, dict):
        raise ValueError("qualification window source is missing")
    if source.get("repository") != BFCL_REPOSITORY or source.get("revision") != BFCL_REVISION:
        raise ValueError("qualification window does not bind the pinned BFCL source")
    declared_files = source.get("files")
    if not isinstance(declared_files, list):
        raise ValueError("qualification window source files are invalid")
    declared_file_map = {
        entry.get("path"): entry.get("sha256")
        for entry in declared_files
        if isinstance(entry, dict)
    }
    if len(declared_file_map) != len(declared_files) or declared_file_map != BFCL_FILES:
        raise ValueError("qualification window does not bind every exact BFCL file")

    selection = manifest.get("selection")
    cases_by_kind = manifest.get("cases")
    if not isinstance(selection, dict) or not isinstance(cases_by_kind, dict):
        raise ValueError("qualification window selection is missing")
    if selection.get("seed") != SEED:
        raise ValueError("qualification window seed differs from the pinned evaluator seed")
    per_kind = selection.get("perKind")
    if not isinstance(per_kind, int) or isinstance(per_kind, bool) or per_kind <= 0:
        raise ValueError("qualification window perKind must be a positive integer")
    expected_case_set = selection.get("caseSetSha256")
    actual_case_set = hashlib.sha256(canonical_json(cases_by_kind).encode()).hexdigest()
    if expected_case_set != actual_case_set:
        raise ValueError(
            "qualification window case-set SHA-256 differs: "
            f"expected {expected_case_set}, got {actual_case_set}"
        )
    if set(cases_by_kind) != {"simple", "multiple", "irrelevance"}:
        raise ValueError("qualification window must contain exactly the three BFCL kinds")

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    selected_query_hashes: set[str] = set()
    for kind in ("simple", "multiple", "irrelevance"):
        entries = cases_by_kind[kind]
        if not isinstance(entries, list) or len(entries) != per_kind:
            raise ValueError(f"qualification window {kind} count differs from perKind")
        raw_cases = read_jsonl(data / f"BFCL_v3_{kind}.json")
        indexed = {case.get("id"): case for case in raw_cases}
        if len(indexed) != len(raw_cases):
            raise ValueError(f"BFCL {kind} source contains duplicate case IDs")
        exact_cases = []
        for entry in entries:
            if not isinstance(entry, dict) or set(entry) != {"id", "querySha256"}:
                raise ValueError(f"qualification window {kind} case entry is invalid")
            case_id = entry["id"]
            if not isinstance(case_id, str) or case_id in selected_ids:
                raise ValueError(f"qualification window has a duplicate or invalid ID: {case_id}")
            case = indexed.get(case_id)
            if case is None:
                raise ValueError(f"qualification window case is absent from BFCL: {case_id}")
            declared_queries = entry["querySha256"]
            actual_queries = _case_query_hashes(case)
            if (
                not isinstance(declared_queries, list)
                or not declared_queries
                or sorted(declared_queries) != actual_queries
            ):
                raise ValueError(
                    f"qualification window query fingerprints differ for {case_id}"
                )
            if selected_query_hashes.intersection(actual_queries):
                raise ValueError("qualification window contains duplicate query fingerprints")
            selected_ids.add(case_id)
            selected_query_hashes.update(actual_queries)
            exact_cases.append(case)
        selected.extend(_materialize_cases(data, kind, exact_cases))

    return selected, {
        "seed": SEED,
        "perKind": per_kind,
        "caseSetSha256": actual_case_set,
        "manifestPath": str(manifest_path),
        "manifestSha256": sha256(manifest_path),
    }


def verify_sha256(path: Path, expected: str) -> None:
    if not path.is_file():
        raise ValueError(f"pinned evaluation file does not exist: {path}")
    actual = sha256(path)
    if actual != expected:
        raise ValueError(
            f"evaluation file SHA-256 differs for {path}: expected {expected}, got {actual}"
        )


def verify_bfcl_data(data: Path) -> None:
    for relative_path, expected in BFCL_FILES.items():
        verify_sha256(data / relative_path, expected)


def resolve_adapter_identity(adapter: Path, training_manifest: Path) -> dict[str, str]:
    """Resolve a pinned base identity and verify the exact adapter before loading either."""
    try:
        manifest = json.loads(training_manifest.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read training manifest: {error}") from error
    if not isinstance(manifest, dict) or manifest.get("schemaVersion") != 1:
        raise ValueError("unsupported training manifest schemaVersion")
    if manifest.get("kind") != "activated-lora-tool-specialist":
        raise ValueError("training manifest is not an activated tool adapter")
    try:
        model = manifest["base"]["model"]
        revision = manifest["base"]["revision"]
        declared_files = manifest["adapter"]["files"]
    except (KeyError, TypeError) as error:
        raise ValueError("training manifest is missing adapter identity") from error
    if not isinstance(model, str) or not model:
        raise ValueError("training manifest base model is invalid")
    if not isinstance(revision, str) or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError("training manifest base revision is not an exact commit")
    if not isinstance(declared_files, list) or not declared_files:
        raise ValueError("training manifest adapter files must be a non-empty array")
    seen: set[str] = set()
    adapter_sha256 = None
    for entry in declared_files:
        if not isinstance(entry, dict):
            raise ValueError("training manifest adapter file must be an object")
        name = entry.get("name")
        if (
            not isinstance(name, str)
            or not name
            or Path(name).name != name
            or name in seen
        ):
            raise ValueError(f"training manifest adapter filename is invalid: {name}")
        seen.add(name)
        declared_size = entry.get("bytes")
        if (
            not isinstance(declared_size, int)
            or isinstance(declared_size, bool)
            or declared_size < 0
        ):
            raise ValueError(f"training manifest adapter file size is invalid: {name}")
        expected = entry.get("sha256")
        if not isinstance(expected, str) or re.fullmatch(r"[0-9a-f]{64}", expected) is None:
            raise ValueError(f"training manifest adapter file SHA-256 is invalid: {name}")
        path = adapter / name
        if not path.is_file():
            raise ValueError(f"declared adapter file does not exist: {path}")
        if path.stat().st_size != declared_size:
            raise ValueError(
                f"adapter file size differs for {name}: expected {declared_size}, "
                f"got {path.stat().st_size}"
            )
        actual = sha256(path)
        if actual != expected:
            raise ValueError(
                f"adapter file SHA-256 differs for {name}: expected {expected}, got {actual}"
            )
        if name == "adapter_model.safetensors":
            adapter_sha256 = actual
    if adapter_sha256 is None:
        raise ValueError("training manifest must identify exactly one adapter weight file")
    return {
        "model": model,
        "revision": revision,
        "adapterSha256": adapter_sha256,
        "trainingManifestSha256": sha256(training_manifest),
    }


def render_prompts(tokenizer: Any, cases: list[dict[str, Any]]) -> list[str]:
    return [
        tokenizer.apply_chat_template(
            case["messages"],
            tools=case["tools"],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for case in cases
    ]


def generate_mode(
    model: Any,
    tokenizer: Any,
    cases: list[dict[str, Any]],
    mode: str,
    batch_size: int,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    import torch

    prompts = render_prompts(tokenizer, cases)
    results = []
    adapter_context = model.disable_adapter if mode == "base" else contextlib.nullcontext
    started = time.perf_counter()
    with adapter_context(), torch.inference_mode():
        for start in range(0, len(cases), batch_size):
            batch_prompts = prompts[start : start + batch_size]
            encoded = tokenizer(
                batch_prompts, return_tensors="pt", padding=True, add_special_tokens=False
            ).to(model.device)
            generated = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
            prompt_width = encoded["input_ids"].shape[1]
            completions = tokenizer.batch_decode(
                generated[:, prompt_width:], skip_special_tokens=True
            )
            for case, prompt, completion in zip(
                cases[start : start + batch_size], batch_prompts, completions
            ):
                record = {
                    **case,
                    "mode": mode,
                    "prompt": prompt,
                    "rawCompletion": completion,
                    "syntaxValid": False,
                    "schemaValid": False,
                    "exact": False,
                }
                try:
                    calls = parse_completion(completion)
                    record["parsedCalls"] = calls
                    record["syntaxValid"] = True
                    validate_calls(calls, case["tools"])
                    record["schemaValid"] = True
                    record["exact"] = exact_calls_match(calls, case["expected"], case["tools"])
                except (ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
                    record["error"] = f"{type(error).__name__}: {error}"
                results.append(record)
    elapsed = time.perf_counter() - started
    for record in results:
        record["modeElapsedSeconds"] = elapsed
    return results


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for mode in ("base", "adapter"):
        selected = [record for record in records if record["mode"] == mode]
        tool_cases = [record for record in selected if record["kind"] != "irrelevance"]
        irrelevant = [record for record in selected if record["kind"] == "irrelevance"]
        summary[mode] = {
            "cases": len(selected),
            "syntaxRate": sum(record["syntaxValid"] for record in selected) / len(selected),
            "schemaRate": sum(record["schemaValid"] for record in selected) / len(selected),
            "toolExactRate": sum(record["exact"] for record in tool_cases) / len(tool_cases),
            "irrelevanceFalseToolRate": sum(
                bool(record.get("parsedCalls")) for record in irrelevant
            )
            / len(irrelevant),
            "elapsedSeconds": selected[0]["modeElapsedSeconds"],
        }
    adapter = summary["adapter"]
    base = summary["base"]
    summary["qualificationGates"] = {
        "syntax100Percent": adapter["syntaxRate"] == 1.0,
        "schema100Percent": adapter["schemaRate"] == 1.0,
        "toolExactAtLeast85Percent": adapter["toolExactRate"] >= 0.85,
        "toolExactNoWorseThanBase": adapter["toolExactRate"] >= base["toolExactRate"],
        "irrelevanceFalseToolAtMost5Percent": adapter["irrelevanceFalseToolRate"] <= 0.05,
    }
    summary["qualificationPassed"] = all(summary["qualificationGates"].values())
    return summary


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--smoke-count",
        type=int,
        help="run a non-qualifying smaller slice; qualification always uses 100 of each kind",
    )
    parser.add_argument(
        "--selection-offset",
        type=int,
        default=0,
        help="skip this many deterministically ranked cases before selecting the evaluation window",
    )
    parser.add_argument(
        "--selection-manifest",
        type=Path,
        help="frozen query-bound manifest required for a qualifying run",
    )
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"output already exists: {args.out}")
    if args.batch_size <= 0 or args.max_new_tokens <= 0 or args.selection_offset < 0:
        parser.error("batch size and max new tokens must be positive; offset must be nonnegative")
    if args.selection_manifest is not None and (
        args.smoke_count is not None or args.selection_offset != 0
    ):
        parser.error("a selection manifest cannot be combined with smoke count or offset")
    if args.smoke_count is None and args.selection_manifest is None:
        parser.error("a qualifying run requires --selection-manifest")
    count = args.smoke_count or QUALIFICATION_COUNT
    verify_bfcl_data(args.data)
    identity = resolve_adapter_identity(args.adapter, args.training_manifest)

    import peft
    import torch
    import transformers
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        identity["model"], revision=identity["revision"]
    )
    tokenizer.padding_side = "left"
    base = AutoModelForCausalLM.from_pretrained(
        identity["model"],
        revision=identity["revision"],
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    device = torch.device(inference_device_name(torch.cuda.is_available()))
    base.to(device)
    model = PeftModel.from_pretrained(base, args.adapter)
    model.eval()
    if args.selection_manifest is not None:
        cases, selection_identity = resolve_qualification_window(
            args.data, args.selection_manifest
        )
    else:
        cases = []
        for kind in ("simple", "multiple", "irrelevance"):
            cases.extend(load_slice(args.data, kind, count, args.selection_offset))
        selection_identity = {
            "seed": SEED,
            "offset": args.selection_offset,
            "perKind": count,
        }

    records = []
    for mode in ("base", "adapter"):
        records.extend(
            generate_mode(
                model, tokenizer, cases, mode, args.batch_size, args.max_new_tokens
            )
        )
    args.out.mkdir(parents=True)
    records_path = args.out / "records.jsonl"
    with records_path.open("w") as target:
        for record in records:
            target.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    report = {
        "schemaVersion": 1,
        "qualifyingRun": args.selection_manifest is not None,
        "selection": selection_identity,
        "base": {"model": identity["model"], "revision": identity["revision"]},
        "evaluationSource": {
            "repository": BFCL_REPOSITORY,
            "revision": BFCL_REVISION,
            "files": [
                {"path": path, "sha256": digest}
                for path, digest in sorted(BFCL_FILES.items())
            ],
        },
        "adapter": {
            "path": str(args.adapter),
            "sha256": identity["adapterSha256"],
            "trainingManifest": {
                "path": str(args.training_manifest),
                "sha256": identity["trainingManifestSha256"],
            },
        },
        "environment": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "device": str(device),
            "transformers": transformers.__version__,
            "peft": peft.__version__,
        },
        "records": {"path": records_path.name, "sha256": sha256(records_path)},
        "summary": summarize(records),
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
