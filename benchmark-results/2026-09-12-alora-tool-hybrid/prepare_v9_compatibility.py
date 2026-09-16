#!/usr/bin/env python3
"""Reproduce the frozen V8/V9 corpus before duplicate-callable rejection was added."""

from __future__ import annotations

from typing import Any

import prepare_tool_data


def normalize_tools_v9(raw_tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Preserve the exact historical normalization used to produce the V8 inputs.

    Later experiments correctly rejected conflicting duplicate callable declarations. V13 must
    nevertheless reproduce V8 byte-for-byte, so this compatibility entry point retains the old
    normalization only for rebuilding that frozen training corpus.
    """
    tools: list[dict[str, Any]] = []
    for raw in raw_tools:
        if raw.get("type") == "function" and isinstance(raw.get("function"), dict):
            source_function = raw["function"]
            function = {
                "name": source_function["name"],
                "description": source_function.get("description", ""),
                "parameters": prepare_tool_data._normalize_json_schema(
                    source_function.get(
                        "parameters", {"type": "object", "properties": {}}
                    )
                ),
            }
            tools.append({"type": "function", "function": function})
            continue
        properties: dict[str, Any] = {}
        required: list[str] = []
        for name, raw_property in raw.get("parameters", {}).items():
            if not isinstance(raw_property, dict):
                raise ValueError(
                    f"parameter schema for {raw['name']}.{name} must be an object"
                )
            source_type = str(raw_property.get("type", "string"))
            schema = prepare_tool_data._json_schema_type(source_type)
            optional = schema.pop("x-source-optional", False) or "default" in raw_property
            if description := raw_property.get("description"):
                schema["description"] = description
            properties[name] = schema
            if not optional:
                required.append(name)
        parameters: dict[str, Any] = {"type": "object", "properties": properties}
        if required:
            parameters["required"] = required
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": raw["name"],
                    "description": raw.get("description", ""),
                    "parameters": parameters,
                },
            }
        )
    return tools


def main() -> None:
    prepare_tool_data.normalize_tools = normalize_tools_v9
    prepare_tool_data.main()


if __name__ == "__main__":
    main()
