#!/usr/bin/env python3
"""Render a qualification case through the published Granite chat template with Transformers.

This is the prompt-byte oracle required by gate 4: the Java runner must produce the same bytes and
token IDs for the same case before any score is read. Transformers is an oracle only; it is never
in the product execution path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def render(tokenizer_directory: Path, window_path: Path, suite_name: str, index: int, arm: str) -> dict:
    from transformers import AutoTokenizer  # imported lazily so unit tests need no Transformers

    window = json.loads(window_path.read_text())
    suite = next(s for s in window["suites"] if s["name"] == suite_name)
    case = suite["cases"][index]
    messages = [{"role": m["role"], "content": m["text"]} for m in case["messages"]]
    if arm == "base":
        messages = [
            {"role": "system", "content": "Answer with exactly one word: answerable or unanswerable"}
        ] + messages
    documents = [{"doc_id": d["doc_id"], "text": d["text"]} for d in case["documents"]]
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_directory))
    text = tokenizer.apply_chat_template(
        messages, documents=documents, add_generation_prompt=True, tokenize=False
    )
    tokens = tokenizer(text, add_special_tokens=False)["input_ids"]
    return {
        "suite": suite_name,
        "index": index,
        "arm": arm,
        "caseId": case["id"],
        "text": text,
        "textSha256": hashlib.sha256(text.encode()).hexdigest(),
        "tokens": tokens,
        "tokenCount": len(tokens),
        "tokenizerDirectory": str(tokenizer_directory),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer-directory", type=Path, required=True)
    parser.add_argument("--window", type=Path, required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--arm", choices=("specialist", "base"), default="specialist")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = render(args.tokenizer_directory, args.window, args.suite, args.index, args.arm)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(args.suite, result["caseId"], "tokens", result["tokenCount"], "sha256", result["textSha256"])


if __name__ == "__main__":
    main()
