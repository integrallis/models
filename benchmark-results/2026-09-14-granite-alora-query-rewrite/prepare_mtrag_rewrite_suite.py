#!/usr/bin/env python3
"""Create a deterministic held-out MT-RAG query-rewrite suite from pinned source files."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for number, line in enumerate(path.read_text().splitlines(), 1):
        item = json.loads(line)
        identifier = item.get("_id")
        text = item.get("text")
        if not isinstance(identifier, str) or not isinstance(text, str):
            raise ValueError(f"invalid JSONL item at {path}:{number}")
        if identifier in result:
            raise ValueError(f"duplicate query id: {identifier}")
        result[identifier] = text
    return result


def prepare(conversations_path: Path, lastturns_path: Path, rewrites_path: Path, count: int) -> dict:
    if count < 1:
        raise ValueError("count must be positive")
    conversations = json.loads(conversations_path.read_text())
    if not isinstance(conversations, list):
        raise ValueError("conversation source must be an array")
    rewrites = load_jsonl(rewrites_path)
    lastturns = load_jsonl(lastturns_path)
    rewrite_by_lastturn = {
        text.removeprefix("|user|: ").strip(): (identifier, rewrites[identifier])
        for identifier, text in lastturns.items()
        if identifier in rewrites
    }
    cases = []
    for conversation in conversations:
        author = conversation.get("author")
        messages = conversation.get("messages")
        if not isinstance(author, str) or not isinstance(messages, list):
            continue
        prefix = []
        for message in messages:
            speaker = message.get("speaker")
            text = message.get("text")
            timestamp = message.get("timestamp")
            if not isinstance(speaker, str) or not isinstance(text, str):
                continue
            prefix.append({"role": "assistant" if speaker == "agent" else speaker, "text": text})
            if speaker != "user" or not isinstance(timestamp, int):
                continue
            match = rewrite_by_lastturn.get(text.strip())
            if match is None:
                continue
            query_id, expected = match
            cases.append(
                {
                    "id": hashlib.sha256((author + str(timestamp)).encode()).hexdigest(),
                    "queryId": query_id,
                    "messages": list(prefix),
                    "expectedRewrite": expected.removeprefix("|user|: ").strip(),
                }
            )
    ordered = sorted(cases, key=lambda item: hashlib.sha256(item["id"].encode()).hexdigest())
    selected = ordered[:count]
    if len(selected) != count:
        raise ValueError(f"only {len(selected)} eligible cases; requested {count}")
    return {
        "schemaVersion": 1,
        "suiteId": "ibm-mtrag-cloud-query-rewrite-heldout-v1",
        "source": {
            "conversationsSha256": sha256(conversations_path),
            "lastturnsSha256": sha256(lastturns_path),
            "rewritesSha256": sha256(rewrites_path),
            "selection": "sha256(id) ascending; first count",
        },
        "cases": selected,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conversations", type=Path, required=True)
    parser.add_argument("--rewrites", type=Path, required=True)
    parser.add_argument("--lastturns", type=Path, required=True)
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing suite: {args.output}")
    suite = prepare(args.conversations, args.lastturns, args.rewrites, args.count)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(suite, indent=2) + "\n")


if __name__ == "__main__":
    main()
