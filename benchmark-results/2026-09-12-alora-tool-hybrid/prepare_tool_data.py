#!/usr/bin/env python3
"""Prepare a pinned, contamination-checked Qwen tool-calling SFT corpus."""

from __future__ import annotations

import argparse
import ast
from copy import deepcopy
import hashlib
import heapq
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable


TOOLS_MARKER = "List of tools: "
NO_TOOL_SENTINEL = "<tool_call>\n[]\n</tool_call>"
TRAINING_REPOSITORY = "edbuildingstuff/bfcl-ft-data"
TRAINING_REVISION = "a7ceb3b1e1605f609fda6f2befec704f575290de"
TRAINING_SHA256 = "915679b445c64676d7212e8953a0c3379c1024235e3c8168d1af2d2ef2ae7973"
HAMMER_IRRELEVANCE_REPOSITORY = "MadeAgents/xlam-irrelevance-7.5k"
HAMMER_IRRELEVANCE_REVISION = "34323bf09efc7e4a394998a0fa91ff997617c369"
HAMMER_IRRELEVANCE_SHA256 = (
    "2e6f3d0adbd40248a592ea001e3f3a4f1624a5d50f434fa1e7d019e93e922327"
)
HAMMER_REPOSITORY = "MadeAgents/Hammer"
HAMMER_REVISION = "ff415d9998180b5a68bbfdda3309ec04b472fb49"
HAMMER_MASKING_SOURCE = "train/data_processing.py"
HAMMER_MASKING_SHA256 = (
    "bf9b79c249b444a25a36d0d43ab031faf2ba45bf3041fa3cb0932f464db579d9"
)
HAMMER_AMBIGUITY_THRESHOLD = 0.45
HAMMER_QUERY_TOOL_AMBIGUITY_THRESHOLD = 0.125
TOKEN = re.compile(r"[a-z0-9]+")
STOPWORDS = {
    "and",
    "are",
    "for",
    "from",
    "get",
    "into",
    "the",
    "this",
    "that",
    "tool",
    "using",
    "with",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_source_sha256(path: Path, expected: str, label: str) -> str:
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"{label} SHA-256 differs: expected {expected}, got {actual}")
    return actual


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def normalize_text(value: str) -> str:
    return " ".join(value.casefold().split())


def _function_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return _function_name(node.value) + "." + node.attr
    raise ValueError("tool call name must contain only identifiers and dots")


def _literal(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_literal(element) for element in node.elts]
    if isinstance(node, ast.Dict):
        return {_literal(key): _literal(value) for key, value in zip(node.keys, node.values)}
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        value = _literal(node.operand)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError("unary tool argument must be numeric")
        return -value if isinstance(node.op, ast.USub) else value
    raise ValueError(f"unsupported tool argument syntax: {type(node).__name__}")


def parse_calls(source: str) -> list[dict[str, Any]]:
    expression = ast.parse(source, mode="eval").body
    if not isinstance(expression, (ast.List, ast.Tuple)):
        raise ValueError("assistant tool label must be a list")
    calls: list[dict[str, Any]] = []
    for node in expression.elts:
        if not isinstance(node, ast.Call) or node.args:
            raise ValueError("tool labels must use keyword-only calls")
        arguments: dict[str, Any] = {}
        for keyword in node.keywords:
            if keyword.arg is None or keyword.arg in arguments:
                raise ValueError("tool arguments must have unique explicit names")
            arguments[keyword.arg] = _literal(keyword.value)
        calls.append({"name": _function_name(node.func), "arguments": arguments})
    return calls


def _generic(source: str) -> tuple[str, list[str]] | None:
    if "[" not in source or not source.endswith("]"):
        return None
    base, body = source.split("[", 1)
    body = body[:-1]
    depth = 0
    start = 0
    arguments: list[str] = []
    for index, character in enumerate(body):
        if character == "[":
            depth += 1
        elif character == "]":
            depth -= 1
        elif character == "," and depth == 0:
            arguments.append(body[start:index].strip())
            start = index + 1
    arguments.append(body[start:].strip())
    return base.strip(), arguments


def _json_schema_type(source: str) -> dict[str, Any]:
    value = source.strip()
    lowered = value.casefold().replace("typing.", "")
    optional = "optional" in lowered
    lowered = re.sub(r",\s*optional$", "", lowered)
    lowered = re.sub(r",\s*default(?:\s*=)?\s*.*$", "", lowered)
    if lowered.startswith("optional[") and lowered.endswith("]"):
        lowered = lowered[len("optional[") : -1]
    generic = _generic(lowered)
    if generic is not None:
        base, arguments = generic
        if base in {"list", "sequence", "array", "arraylist", "set"}:
            if len(arguments) != 1:
                raise ValueError(f"{base} requires one item type")
            return {"type": "array", "items": _json_schema_type(arguments[0])}
        if base == "tuple":
            schemas = [_json_schema_type(argument) for argument in arguments]
            unique = {canonical_json(schema): schema for schema in schemas}
            items = next(iter(unique.values())) if len(unique) == 1 else {"anyOf": list(unique.values())}
            return {"type": "array", "items": items, "minItems": len(schemas), "maxItems": len(schemas)}
        if base == "union":
            return {"anyOf": [_json_schema_type(argument) for argument in arguments]}
        raise ValueError(f"unsupported source type: {source}")
    return {
        "type": {
            "str": "string",
            "string": "string",
            "int": "integer",
            "integer": "integer",
            "float": "number",
            "number": "number",
            "bool": "boolean",
            "boolean": "boolean",
            "dict": "object",
            "object": "object",
            "hashmap": "object",
            "array": "array",
            "arraylist": "array",
            "list": "array",
            "set": "array",
        }.get(lowered, "string"),
        **({"x-source-optional": True} if optional else {}),
    }


def _normalize_json_schema(raw: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("parameter schema must be a JSON object")
    schema = dict(raw)
    if "type" in schema:
        source_type = str(schema["type"])
        normalized_type = _json_schema_type(source_type)
        schema.pop("type")
        schema = {**normalized_type, **schema}
    if isinstance(schema.get("properties"), dict):
        schema["properties"] = {
            name: _normalize_json_schema(value)
            for name, value in schema["properties"].items()
        }
    if isinstance(schema.get("items"), dict):
        schema["items"] = _normalize_json_schema(schema["items"])
    return schema


def normalize_tools(raw_tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tools: list[dict[str, Any]] = []
    for raw in raw_tools:
        if raw.get("type") == "function" and isinstance(raw.get("function"), dict):
            source_function = raw["function"]
            function = {
                "name": source_function["name"],
                "description": source_function.get("description", ""),
                "parameters": _normalize_json_schema(
                    source_function.get("parameters", {"type": "object", "properties": {}})
                ),
            }
            tools.append({"type": "function", "function": function})
            continue
        properties: dict[str, Any] = {}
        required: list[str] = []
        for name, raw_property in raw.get("parameters", {}).items():
            if not isinstance(raw_property, dict):
                raise ValueError(f"parameter schema for {raw['name']}.{name} must be an object")
            source_type = str(raw_property.get("type", "string"))
            schema = _json_schema_type(source_type)
            optional = schema.pop("x-source-optional", False) or "default" in raw_property
            if description := raw_property.get("description"):
                schema["description"] = description
            properties[name] = schema
            if not optional:
                required.append(name)
        parameters: dict[str, Any] = {"type": "object", "properties": properties}
        if required:
            parameters["required"] = required
        function = {
            "name": raw["name"],
            "description": raw.get("description", ""),
            "parameters": parameters,
        }
        tools.append({"type": "function", "function": function})
    signatures: dict[str, str] = {}
    for tool in tools:
        function = tool["function"]
        name = function["name"]
        signature = _callable_signature(function)
        existing = signatures.get(name)
        if existing is None:
            signatures[name] = signature
            continue
        if existing != signature:
            raise ValueError(f"ambiguous duplicate tool declaration: {name}")
    return tools


def render_calls(calls: list[dict[str, Any]]) -> str:
    if not calls:
        return NO_TOOL_SENTINEL
    return "\n".join(
        f"<tool_call>\n{canonical_json(call)}\n</tool_call>" for call in calls
    )


def validate_calls(calls: list[dict[str, Any]], tools: list[dict[str, Any]]) -> None:
    schemas = {tool["function"]["name"]: tool["function"]["parameters"] for tool in tools}
    for call in calls:
        name = call["name"]
        if name not in schemas:
            raise ValueError(f"call references undeclared tool: {name}")
        schema = schemas[name]
        properties = schema.get("properties", {})
        arguments = call["arguments"]
        unknown = arguments.keys() - properties.keys()
        if unknown:
            raise ValueError(f"call has undeclared arguments for {name}: {sorted(unknown)}")
        missing = set(schema.get("required", [])) - arguments.keys()
        if missing:
            raise ValueError(f"call is missing required arguments for {name}: {sorted(missing)}")


def training_record(line: str, source_line: int) -> dict[str, Any]:
    source = json.loads(line)
    messages = source["messages"]
    system = next(message["content"] for message in messages if message["role"] == "system")
    user = next(message["content"] for message in messages if message["role"] == "user")
    assistant = next(
        message["content"] for message in reversed(messages) if message["role"] == "assistant"
    )
    marker = system.find(TOOLS_MARKER)
    if marker < 0:
        raise ValueError(f"source line {source_line} has no tools marker")
    raw_tools = json.loads(system[marker + len(TOOLS_MARKER) :])
    calls = parse_calls(assistant)
    tools = normalize_tools(raw_tools)
    validate_calls(calls, tools)
    return {
        "sourceLine": source_line,
        "user": user,
        "tools": tools,
        "calls": calls,
        "assistant": render_calls(calls),
    }


def external_irrelevance_record(source: dict[str, Any], source_line: int) -> dict[str, Any]:
    raw_tools = source["tools"]
    if isinstance(raw_tools, str):
        raw_tools = json.loads(raw_tools)
    answers = source["answers"]
    if isinstance(answers, str):
        answers = json.loads(answers)
    if answers != []:
        raise ValueError("external irrelevance record must contain an empty answer")
    tools = normalize_tools(raw_tools)
    return {
        "sourceLine": source_line,
        "user": source["query"],
        "tools": tools,
        "calls": [],
        "assistant": NO_TOOL_SENTINEL,
    }


def external_irrelevance_records(
    source: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    records: list[dict[str, Any]] = []
    statistics = {"read": 0, "invalid": 0}
    for source_line, row in enumerate(source, start=1):
        statistics["read"] += 1
        try:
            records.append(external_irrelevance_record(row, source_line))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            statistics["invalid"] += 1
    return records, statistics


def _function_tokens(function: dict[str, Any]) -> set[str]:
    text = [function["name"], function.get("description", "")]
    text.extend(_schema_search_text(function.get("parameters", {})))
    return _meaningful_tokens(" ".join(text))


def _function_similarity(left: dict[str, Any], right: dict[str, Any]) -> float:
    left_tokens = _function_tokens(left)
    right_tokens = _function_tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / math.sqrt(
        len(left_tokens) * len(right_tokens)
    )


def maximum_query_tool_similarity(record: dict[str, Any]) -> float:
    """Return the strongest lexical match between a query and one advertised function.

    Scoring every function independently prevents a large catalog of unrelated functions from
    diluting an applicable function's score.
    """
    query_tokens = _meaningful_tokens(record["user"])
    if not query_tokens:
        return 0.0
    similarities = []
    for tool in record["tools"]:
        function_tokens = _function_tokens(tool["function"])
        if function_tokens:
            similarities.append(
                len(query_tokens & function_tokens)
                / math.sqrt(len(query_tokens) * len(function_tokens))
            )
    return max(similarities, default=0.0)


def positive_call_index(
    lines: Iterable[str],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, int]]:
    index: dict[str, list[dict[str, Any]]] = {}
    statistics = {"read": 0, "invalid": 0, "positive": 0}
    for source_line, line in enumerate(lines, start=1):
        statistics["read"] += 1
        try:
            record = training_record(line, source_line)
        except (KeyError, TypeError, ValueError, SyntaxError, json.JSONDecodeError):
            statistics["invalid"] += 1
            continue
        if not record["calls"]:
            continue
        functions = {tool["function"]["name"]: tool["function"] for tool in record["tools"]}
        evidence = {
            "callCount": len(record["calls"]),
            "calledFunctions": [functions[call["name"]] for call in record["calls"]],
        }
        index.setdefault(query_fingerprint(record["user"]), []).append(evidence)
        statistics["positive"] += 1
    return index, statistics


def filter_external_irrelevance(
    records: Iterable[dict[str, Any]],
    positive_index: dict[str, list[dict[str, Any]]],
    ambiguity_threshold: float,
    query_tool_ambiguity_threshold: float,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    if not 0.0 <= ambiguity_threshold <= 1.0:
        raise ValueError("ambiguity threshold must be between zero and one")
    if not 0.0 <= query_tool_ambiguity_threshold <= 1.0:
        raise ValueError("query/tool ambiguity threshold must be between zero and one")
    retained: list[dict[str, Any]] = []
    statistics = {
        "read": 0,
        "missingPositiveEvidence": 0,
        "multiCallPositive": 0,
        "ambiguousDistractor": 0,
        "queryToolAmbiguous": 0,
        "retained": 0,
    }
    for record in records:
        statistics["read"] += 1
        evidence = positive_index.get(query_fingerprint(record["user"]))
        if not evidence:
            statistics["missingPositiveEvidence"] += 1
            continue
        if any(item["callCount"] != 1 for item in evidence):
            statistics["multiCallPositive"] += 1
            continue
        called_functions = [
            function for item in evidence for function in item["calledFunctions"]
        ]
        remaining_functions = [tool["function"] for tool in record["tools"]]
        if any(
            _function_similarity(called, remaining) >= ambiguity_threshold
            for called in called_functions
            for remaining in remaining_functions
        ):
            statistics["ambiguousDistractor"] += 1
            continue
        if maximum_query_tool_similarity(record) >= query_tool_ambiguity_threshold:
            statistics["queryToolAmbiguous"] += 1
            continue
        retained.append(record)
        statistics["retained"] += 1
    return retained, statistics


def query_fingerprint(user: str) -> str:
    return hashlib.sha256(normalize_text(user).encode()).hexdigest()


def record_fingerprint(record: dict[str, Any]) -> str:
    identity = {
        "user": normalize_text(record["user"]),
        "tools": record["tools"],
        "calls": record["calls"],
    }
    return hashlib.sha256(canonical_json(identity).encode()).hexdigest()


def _opaque_identifiers(
    prefix: str,
    seed: int,
    fingerprint: str,
    namespace: str,
    source_names: Iterable[str],
) -> dict[str, str]:
    names = list(source_names)
    if len(names) != len(set(names)):
        raise ValueError("opaque identifier source names must be unique")
    ranked = sorted(
        names,
        key=lambda name: (
            hashlib.sha256(
                f"{seed}:{fingerprint}:{namespace}:{name}".encode()
            ).digest(),
            name,
        ),
    )
    return {name: f"{prefix}{index}" for index, name in enumerate(ranked)}


def _schema_without_descriptions(value: Any) -> Any:
    """Return the validation-relevant schema shape used to compare duplicate callables."""
    if isinstance(value, dict):
        return {
            key: _schema_without_descriptions(child)
            for key, child in value.items()
            if key != "description"
        }
    if isinstance(value, list):
        return [_schema_without_descriptions(child) for child in value]
    return value


def _callable_signature(function: dict[str, Any]) -> str:
    return canonical_json(_schema_without_descriptions(function.get("parameters", {})))


def _function_mask_record(record: dict[str, Any], seed: int) -> dict[str, Any]:
    """Replace callable and argument names while preserving descriptions and values.

    This is the useful part of Hammer's function-masking technique: it prevents the training run
    from succeeding by memorizing semantically meaningful API names. Unlike the reference script,
    the mapping here is content-addressed, deterministic, collision-checked, and applied to both
    JSON Schema and labels without duplicating tools.
    """
    masked = deepcopy(record)
    fingerprint = record_fingerprint(record)
    functions = [tool["function"] for tool in masked["tools"]]
    unique_functions: dict[str, dict[str, Any]] = {}
    for function in functions:
        name = function["name"]
        existing = unique_functions.get(name)
        if existing is not None and _callable_signature(existing) != _callable_signature(
            function
        ):
            raise ValueError(f"conflicting duplicate tool declaration: {name}")
        unique_functions.setdefault(name, function)
    original_names = list(unique_functions)
    function_names = _opaque_identifiers(
        "f", seed, fingerprint, "function", original_names
    )
    if len(function_names.values()) != len(set(function_names.values())):
        raise ValueError("function masking generated duplicate tool names")
    parameter_names: dict[str, dict[str, str]] = {}
    for original_function_name, function in unique_functions.items():
        parameters = function.get("parameters", {})
        properties = parameters.get("properties", {})
        mapping = _opaque_identifiers(
            "p",
            seed,
            fingerprint,
            f"parameter:{original_function_name}",
            properties,
        )
        if len(mapping.values()) != len(set(mapping.values())):
            raise ValueError(
                f"function masking generated duplicate arguments for {original_function_name}"
            )
        parameter_names[original_function_name] = mapping
    for function in functions:
        original_function_name = function["name"]
        mapping = parameter_names[original_function_name]
        parameters = function.get("parameters", {})
        properties = parameters.get("properties", {})
        function["name"] = function_names[original_function_name]
        if properties:
            parameters["properties"] = {
                mapping[name]: schema for name, schema in properties.items()
            }
        if "required" in parameters:
            parameters["required"] = [mapping[name] for name in parameters["required"]]
    for call in masked["calls"]:
        original_function_name = call["name"]
        mapping = parameter_names[original_function_name]
        call["name"] = function_names[original_function_name]
        call["arguments"] = {
            mapping[name]: value for name, value in call["arguments"].items()
        }
    masked["assistant"] = render_calls(masked["calls"])
    validate_calls(masked["calls"], masked["tools"])
    return masked


def apply_function_mask(
    records: Iterable[dict[str, Any]], fraction: float, seed: int
) -> list[dict[str, Any]]:
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("function mask fraction must be between zero and one")
    threshold = int(fraction * (1 << 64))
    masked: list[dict[str, Any]] = []
    for record in records:
        fingerprint = record_fingerprint(record)
        priority = int.from_bytes(
            hashlib.sha256(
                f"{seed}:function-mask-selection:{fingerprint}".encode()
            ).digest()[:8],
            "big",
        )
        masked.append(
            _function_mask_record(record, seed)
            if priority < threshold
            else deepcopy(record)
        )
    return masked


def _meaningful_tokens(value: str) -> set[str]:
    return {
        token
        for token in TOKEN.findall(value.casefold().replace("_", " "))
        if len(token) >= 3 and token not in STOPWORDS
    }


def _schema_search_text(schema: dict[str, Any]) -> list[str]:
    result: list[str] = []
    if description := schema.get("description"):
        result.append(str(description))
    if isinstance(schema.get("enum"), list):
        result.extend(str(value) for value in schema["enum"])
    if isinstance(schema.get("properties"), dict):
        for name, nested in schema["properties"].items():
            result.append(name)
            if isinstance(nested, dict):
                result.extend(_schema_search_text(nested))
    if isinstance(schema.get("items"), dict):
        result.extend(_schema_search_text(schema["items"]))
    return result


def no_call_hardness(record: dict[str, Any]) -> float:
    """Score lexical applicability overlap for a no-call training example.

    A high score means the user's request resembles the advertised tools even though the label says
    no tool applies. Those examples teach the applicability boundary that random mismatches miss.
    """
    query_tokens = _meaningful_tokens(record["user"])
    tool_text: list[str] = []
    for tool in record["tools"]:
        function = tool["function"]
        tool_text.extend((function["name"], function.get("description", "")))
        tool_text.extend(_schema_search_text(function.get("parameters", {})))
    tool_tokens = _meaningful_tokens(" ".join(tool_text))
    if not query_tokens or not tool_tokens:
        return 0.0
    return len(query_tokens & tool_tokens) / math.sqrt(
        len(query_tokens) * len(tool_tokens)
    )


def bfcl_query_fingerprints(paths: Iterable[Path]) -> set[str]:
    result: set[str] = set()
    for path in paths:
        with path.open() as source:
            for line in source:
                if not line.strip():
                    continue
                row = json.loads(line)
                for turn in row["question"]:
                    for message in turn:
                        if message.get("role") == "user":
                            result.add(query_fingerprint(message["content"]))
    return result


def select_normalized_records(
    records: Iterable[dict[str, Any]],
    excluded_queries: set[str],
    count: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    if count <= 0:
        raise ValueError("count must be positive")
    selected: list[tuple[int, int, dict[str, Any]]] = []
    seen: set[str] = set()
    statistics = {"read": 0, "overlap": 0, "duplicate": 0}
    for record in records:
        statistics["read"] += 1
        if query_fingerprint(record["user"]) in excluded_queries:
            statistics["overlap"] += 1
            continue
        fingerprint = record_fingerprint(record)
        if fingerprint in seen:
            statistics["duplicate"] += 1
            continue
        seen.add(fingerprint)
        priority = int.from_bytes(
            hashlib.sha256(f"{seed}:{fingerprint}".encode()).digest()[:8], "big"
        )
        retain_lowest(selected, (-priority, record["sourceLine"], record), count)
    if len(selected) != count:
        raise ValueError(f"requested {count} records but selected {len(selected)}")
    return [entry[2] for entry in sorted(selected, key=lambda entry: -entry[0])], statistics


def select_records(
    lines: Iterable[str],
    excluded_queries: set[str],
    count: int,
    seed: int,
    no_call_count: int | None = None,
    hard_no_call_count: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    if count <= 0:
        raise ValueError("count must be positive")
    if no_call_count is not None and not 0 <= no_call_count <= count:
        raise ValueError("no_call_count must be between zero and count")
    if no_call_count is None and hard_no_call_count:
        raise ValueError("hard_no_call_count requires no_call_count")
    if no_call_count is not None and not 0 <= hard_no_call_count <= no_call_count:
        raise ValueError("hard_no_call_count must be between zero and no_call_count")
    selected: list[tuple[int, int, dict[str, Any]]] = []
    selected_no_calls: list[tuple[int, int, dict[str, Any]]] = []
    random_no_call_pool: list[tuple[int, int, dict[str, Any]]] = []
    selected_hard_no_calls: list[tuple[float, int, int, dict[str, Any]]] = []
    selected_calls: list[tuple[int, int, dict[str, Any]]] = []
    seen: set[str] = set()
    statistics = {
        "read": 0,
        "invalid": 0,
        "overlap": 0,
        "duplicate": 0,
        "selectedHardNoCalls": 0,
    }
    for source_line, line in enumerate(lines, start=1):
        statistics["read"] += 1
        try:
            record = training_record(line, source_line)
        except (KeyError, TypeError, ValueError, SyntaxError, json.JSONDecodeError):
            statistics["invalid"] += 1
            continue
        if query_fingerprint(record["user"]) in excluded_queries:
            statistics["overlap"] += 1
            continue
        fingerprint = record_fingerprint(record)
        if fingerprint in seen:
            statistics["duplicate"] += 1
            continue
        seen.add(fingerprint)
        priority = int.from_bytes(
            hashlib.sha256(f"{seed}:{fingerprint}".encode()).digest()[:8], "big"
        )
        entry = (-priority, source_line, record)
        if no_call_count is None:
            retain_lowest(selected, entry, count)
        elif record["calls"]:
            retain_lowest(selected_calls, entry, count - no_call_count)
        elif hard_no_call_count:
            retain_highest(
                selected_hard_no_calls,
                (no_call_hardness(record), -priority, -source_line, record),
                hard_no_call_count,
            )
            retain_lowest(random_no_call_pool, entry, no_call_count)
        else:
            retain_lowest(selected_no_calls, entry, no_call_count)
    if no_call_count is not None:
        if hard_no_call_count:
            hard_records = [entry[3] for entry in selected_hard_no_calls]
            hard_fingerprints = {record_fingerprint(record) for record in hard_records}
            random_records = [
                entry[2]
                for entry in sorted(random_no_call_pool, key=lambda candidate: -candidate[0])
                if record_fingerprint(entry[2]) not in hard_fingerprints
            ][: no_call_count - hard_no_call_count]
            no_call_records = hard_records + random_records
            call_records = [
                entry[2] for entry in sorted(selected_calls, key=lambda candidate: -candidate[0])
            ]
            statistics["selectedHardNoCalls"] = len(hard_records)
            selected_records = no_call_records + call_records
            selected_records.sort(key=lambda record: split_priority(record, seed, "selection"))
            if len(selected_records) != count:
                raise ValueError(
                    f"requested {count} records but selected {len(selected_records)} "
                    f"(hard no-call {len(hard_records)}/{hard_no_call_count}, "
                    f"other no-call {len(random_records)}/{no_call_count - hard_no_call_count}, "
                    f"call {len(call_records)}/{count - no_call_count})"
                )
            return selected_records, statistics
        selected = selected_no_calls + selected_calls
    if len(selected) != count:
        suffix = ""
        if no_call_count is not None:
            suffix = (
                f" (no-call {len(selected_no_calls)}/{no_call_count}, "
                f"call {len(selected_calls)}/{count - no_call_count})"
            )
        raise ValueError(f"requested {count} records but selected {len(selected)}{suffix}")
    ordered = [entry[2] for entry in sorted(selected, key=lambda entry: -entry[0])]
    return ordered, statistics


def retain_lowest(
    selected: list[tuple[int, int, dict[str, Any]]],
    entry: tuple[int, int, dict[str, Any]],
    count: int,
) -> None:
    if count == 0:
        return
    priority = -entry[0]
    if len(selected) < count:
        heapq.heappush(selected, entry)
    elif priority < -selected[0][0]:
        heapq.heapreplace(selected, entry)


def retain_highest(
    selected: list[tuple[float, int, int, dict[str, Any]]],
    entry: tuple[float, int, int, dict[str, Any]],
    count: int,
) -> None:
    if count == 0:
        return
    if len(selected) < count:
        heapq.heappush(selected, entry)
    elif entry[:3] > selected[0][:3]:
        heapq.heapreplace(selected, entry)


def split_stratified(
    records: list[dict[str, Any]],
    train_count: int,
    validation_count: int,
    no_call_fraction: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if train_count <= 0 or validation_count <= 0:
        raise ValueError("train and validation counts must be positive")
    if not 0.0 <= no_call_fraction <= 1.0:
        raise ValueError("no_call_fraction must be between zero and one")
    train_no_call = round(train_count * no_call_fraction)
    validation_no_call = round(validation_count * no_call_fraction)
    no_calls = [record for record in records if not record["calls"]]
    calls = [record for record in records if record["calls"]]
    required_no_calls = train_no_call + validation_no_call
    required_calls = train_count + validation_count - required_no_calls
    if len(no_calls) != required_no_calls or len(calls) != required_calls:
        raise ValueError(
            "selected population does not match the requested stratification: "
            f"no-call {len(no_calls)}/{required_no_calls}, call {len(calls)}/{required_calls}"
        )
    train = no_calls[:train_no_call] + calls[: train_count - train_no_call]
    validation = no_calls[train_no_call:] + calls[train_count - train_no_call :]
    train.sort(key=lambda record: split_priority(record, seed, "train"))
    validation.sort(key=lambda record: split_priority(record, seed, "validation"))
    return train, validation


def split_priority(record: dict[str, Any], seed: int, split: str) -> bytes:
    return hashlib.sha256(
        f"{seed}:{split}:{record_fingerprint(record)}".encode()
    ).digest()


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    with path.open("w") as target:
        for record in records:
            target.write(canonical_json(record) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", required=True, type=Path)
    parser.add_argument("--irrelevance", type=Path)
    parser.add_argument("--bfcl", required=True, type=Path, nargs="+")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--train-count", type=int, default=12_000)
    parser.add_argument("--validation-count", type=int, default=1_000)
    parser.add_argument("--no-call-fraction", type=float, default=0.25)
    parser.add_argument("--hard-no-call-fraction", type=float, default=0.0)
    parser.add_argument("--function-mask-fraction", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20_260_912)
    args = parser.parse_args()
    if args.train_count <= 0 or args.validation_count <= 0:
        parser.error("train and validation counts must be positive")
    if not 0.0 <= args.no_call_fraction <= 1.0:
        parser.error("no-call fraction must be between zero and one")
    if not 0.0 <= args.hard_no_call_fraction <= 1.0:
        parser.error("hard no-call fraction must be between zero and one")
    if not 0.0 <= args.function_mask_fraction <= 1.0:
        parser.error("function mask fraction must be between zero and one")
    if args.irrelevance and args.hard_no_call_fraction:
        parser.error("external irrelevance data and mined no-calls are mutually exclusive")
    verify_source_sha256(args.training, TRAINING_SHA256, "training source")
    args.out.mkdir(parents=True, exist_ok=False)
    excluded = bfcl_query_fingerprints(args.bfcl)
    train_no_call = round(args.train_count * args.no_call_fraction)
    validation_no_call = round(args.validation_count * args.no_call_fraction)
    hard_no_call_count = round(
        (train_no_call + validation_no_call) * args.hard_no_call_fraction
    )
    total_count = args.train_count + args.validation_count
    total_no_call = train_no_call + validation_no_call
    if args.irrelevance:
        actual_irrelevance_sha256 = sha256(args.irrelevance)
        if actual_irrelevance_sha256 != HAMMER_IRRELEVANCE_SHA256:
            raise ValueError(
                "Hammer irrelevance SHA-256 differs: "
                f"expected {HAMMER_IRRELEVANCE_SHA256}, got {actual_irrelevance_sha256}"
            )
        with args.training.open() as lines:
            call_records, call_statistics = select_records(
                lines,
                excluded,
                total_count - total_no_call,
                args.seed,
                no_call_count=0,
            )
        raw_irrelevance = json.loads(args.irrelevance.read_text())
        irrelevance_records, normalization_statistics = external_irrelevance_records(
            raw_irrelevance
        )
        with args.training.open() as lines:
            positive_index, positive_index_statistics = positive_call_index(lines)
        irrelevance_records, filter_statistics = filter_external_irrelevance(
            irrelevance_records,
            positive_index,
            HAMMER_AMBIGUITY_THRESHOLD,
            HAMMER_QUERY_TOOL_AMBIGUITY_THRESHOLD,
        )
        no_call_records, no_call_statistics = select_normalized_records(
            irrelevance_records, excluded, total_no_call, args.seed
        )
        records = call_records + no_call_records
        records.sort(key=lambda record: split_priority(record, args.seed, "selection"))
        statistics: dict[str, Any] = {
            "callSource": call_statistics,
            "irrelevanceSource": {
                "normalization": normalization_statistics,
                "positiveEvidence": positive_index_statistics,
                "filter": filter_statistics,
                "selection": no_call_statistics,
            },
        }
    else:
        with args.training.open() as lines:
            records, statistics = select_records(
                lines,
                excluded,
                total_count,
                args.seed,
                no_call_count=total_no_call,
                hard_no_call_count=hard_no_call_count,
            )
    train, validation = split_stratified(
        records,
        args.train_count,
        args.validation_count,
        args.no_call_fraction,
        args.seed,
    )
    unmasked_train = train
    unmasked_validation = validation
    train = apply_function_mask(unmasked_train, args.function_mask_fraction, args.seed)
    validation = apply_function_mask(
        unmasked_validation, args.function_mask_fraction, args.seed
    )
    masked_train_count = sum(
        before["tools"] != after["tools"]
        for before, after in zip(unmasked_train, train)
    )
    masked_validation_count = sum(
        before["tools"] != after["tools"]
        for before, after in zip(unmasked_validation, validation)
    )
    train_path = args.out / "train.jsonl"
    validation_path = args.out / "validation.jsonl"
    write_jsonl(train_path, train)
    write_jsonl(validation_path, validation)
    manifest = {
        "schemaVersion": 2,
        "seed": args.seed,
        "noCallFraction": args.no_call_fraction,
        "hardNoCallFraction": args.hard_no_call_fraction,
        "hardNoCallCount": hard_no_call_count,
        "hardnessMetric": "binary lexical cosine over query and tool/schema terms",
        "functionMask": {
            "fraction": args.function_mask_fraction,
            "trainCount": masked_train_count,
            "validationCount": masked_validation_count,
            "selection": "unsigned first 64 bits of SHA-256(seed:function-mask-selection:record-fingerprint)",
            "identifierDerivation": "short ordinal assigned by sorted SHA-256(seed:record-fingerprint:namespace:source-name)",
            "reference": {
                "repository": HAMMER_REPOSITORY,
                "revision": HAMMER_REVISION,
                "path": HAMMER_MASKING_SOURCE,
                "sha256": HAMMER_MASKING_SHA256,
            },
        },
        "trainingSource": {
            "repository": TRAINING_REPOSITORY,
            "revision": TRAINING_REVISION,
            "path": args.training.name,
            "sha256": sha256(args.training),
        },
        "irrelevanceSource": (
            {
                "repository": HAMMER_IRRELEVANCE_REPOSITORY,
                "revision": HAMMER_IRRELEVANCE_REVISION,
                "path": args.irrelevance.name,
                "sha256": sha256(args.irrelevance),
                "construction": "known-correct function removed from xLAM positive example",
                "ambiguityFilter": {
                    "singleCallPositiveRequired": True,
                    "maximumCalledToRemainingToolSimilarityExclusive": HAMMER_AMBIGUITY_THRESHOLD,
                    "maximumQueryToRemainingToolsSimilarityExclusive": HAMMER_QUERY_TOOL_AMBIGUITY_THRESHOLD,
                    "calledToRemainingMetric": "binary lexical cosine over function and schema terms",
                    "queryToRemainingMetric": "maximum per-function binary lexical cosine over name, description, and schema terms",
                },
            }
            if args.irrelevance
            else None
        ),
        "excludedEvaluationSources": [
            {"path": path.name, "sha256": sha256(path)} for path in args.bfcl
        ],
        "excludedQueryCount": len(excluded),
        "statistics": statistics,
        "train": {
            "count": len(train),
            "noCallCount": sum(not record["calls"] for record in train),
            "sha256": sha256(train_path),
            "sourceLines": [record["sourceLine"] for record in train],
        },
        "validation": {
            "count": len(validation),
            "noCallCount": sum(not record["calls"] for record in validation),
            "sha256": sha256(validation_path),
            "sourceLines": [record["sourceLine"] for record in validation],
        },
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(canonical_json(manifest))


if __name__ == "__main__":
    main()
