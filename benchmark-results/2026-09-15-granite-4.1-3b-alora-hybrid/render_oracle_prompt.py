#!/usr/bin/env python3
"""Render qualification cases through the published Granite chat template with Transformers.

This is the prompt-byte oracle required by gate 4: the Java runner must produce the same bytes and
token IDs for the same case before any score is read. Transformers is an oracle only; it is never
in the product execution path. ``--all`` renders every case of a suite into one JSON document;
``compare`` checks a directory of Java dumps (``<index>.json``) against such a document.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

BASE_INSTRUCTION = "Answer with exactly one word: answerable or unanswerable"


def load_case(window_path: Path, suite_name: str, index: int) -> dict:
    window = json.loads(window_path.read_text())
    suite = next(s for s in window["suites"] if s["name"] == suite_name)
    return suite["cases"][index]


def render_case(tokenizer, case: dict, arm: str) -> dict:
    messages = [{"role": m["role"], "content": m["text"]} for m in case["messages"]]
    if arm == "base":
        messages = [{"role": "system", "content": BASE_INSTRUCTION}] + messages
    documents = [{"doc_id": d["doc_id"], "text": d["text"]} for d in case["documents"]]
    text = tokenizer.apply_chat_template(
        messages, documents=documents, add_generation_prompt=True, tokenize=False
    )
    tokens = tokenizer(text, add_special_tokens=False)["input_ids"]
    return {
        "caseId": case["id"],
        "text": text,
        "textSha256": hashlib.sha256(text.encode()).hexdigest(),
        "tokens": tokens,
        "tokenCount": len(tokens),
    }


def render(tokenizer_directory: Path, window_path: Path, suite_name: str, arm: str, index: int | None) -> dict:
    from transformers import AutoTokenizer  # imported lazily so unit tests need no Transformers

    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_directory))
    window = json.loads(window_path.read_text())
    suite = next(s for s in window["suites"] if s["name"] == suite_name)
    cases = suite["cases"] if index is None else [suite["cases"][index]]
    rendered = [render_case(tokenizer, case, arm) for case in cases]
    return {
        "suite": suite_name,
        "arm": arm,
        "index": index,
        "tokenizerDirectory": str(tokenizer_directory),
        "windowSha256": window.get("windowSha256"),
        "cases": rendered,
    }


def compare(oracle_path: Path, dumps_directory: Path) -> dict:
    oracle = json.loads(oracle_path.read_text())
    mismatches = []
    compared = 0
    for index, expected in enumerate(oracle["cases"]):
        dump_path = dumps_directory / f"{index}.json"
        if not dump_path.is_file():
            mismatches.append({"index": index, "caseId": expected["caseId"], "reason": "missing Java dump"})
            continue
        actual = json.loads(dump_path.read_text())
        compared += 1
        if actual["caseId"] != expected["caseId"]:
            mismatches.append({"index": index, "caseId": expected["caseId"], "reason": "case id differs"})
        elif actual["textSha256"] != expected["textSha256"]:
            mismatches.append({"index": index, "caseId": expected["caseId"], "reason": "text differs"})
        elif actual["tokens"] != expected["tokens"]:
            first = next(
                (i for i, (a, b) in enumerate(zip(actual["tokens"], expected["tokens"])) if a != b),
                min(len(actual["tokens"]), len(expected["tokens"])),
            )
            mismatches.append(
                {
                    "index": index,
                    "caseId": expected["caseId"],
                    "reason": "tokens differ",
                    "firstDifference": first,
                    "javaTokens": actual["tokenCount"],
                    "oracleTokens": expected["tokenCount"],
                }
            )
    return {
        "suite": oracle["suite"],
        "arm": oracle["arm"],
        "oracleCases": len(oracle["cases"]),
        "compared": compared,
        "identical": compared - len([m for m in mismatches if m["reason"] != "missing Java dump"]),
        "mismatches": mismatches,
        "pass": compared == len(oracle["cases"]) and not mismatches,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    render_parser = subparsers.add_parser("render")
    render_parser.add_argument("--tokenizer-directory", type=Path, required=True)
    render_parser.add_argument("--window", type=Path, required=True)
    render_parser.add_argument("--suite", required=True)
    render_parser.add_argument("--arm", choices=("specialist", "base"), default="specialist")
    render_parser.add_argument("--index", type=int)
    render_parser.add_argument("--all", action="store_true")
    render_parser.add_argument("--output", type=Path, required=True)
    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--oracle", type=Path, required=True)
    compare_parser.add_argument("--dumps", type=Path, required=True)
    compare_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "render":
        if args.all == (args.index is not None):
            raise SystemExit("pass exactly one of --all or --index")
        result = render(args.tokenizer_directory, args.window, args.suite, args.arm, None if args.all else args.index)
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
        print(args.suite, args.arm, "cases", len(result["cases"]), "tokens", sum(c["tokenCount"] for c in result["cases"]))
    else:
        result = compare(args.oracle, args.dumps)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(result["suite"], result["arm"], "identical", result["identical"], "of", result["oracleCases"], "PASS" if result["pass"] else "FAIL")
        if not result["pass"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
