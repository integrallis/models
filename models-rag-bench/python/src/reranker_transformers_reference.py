"""Deterministic Transformers oracle for cross-encoder reranker qualification."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence


def validate_workload(query: str, documents: Sequence[str]) -> None:
    """Rejects a workload that cannot produce meaningful sentence pairs."""
    if not query or not query.strip():
        raise ValueError("query must not be blank")
    if not documents:
        raise ValueError("at least one document is required")
    if any(not document or not document.strip() for document in documents):
        raise ValueError("documents must not contain blank text")


def _lists(value: Any) -> list[Any]:
    converted = value.tolist() if hasattr(value, "tolist") else value
    return list(converted)


def build_report(
    *,
    model_id: str,
    requested_revision: str,
    resolved_revision: str,
    query: str,
    documents: Sequence[str],
    encoded: Mapping[str, Any],
    logits: Sequence[float],
    dependency_versions: Mapping[str, str],
) -> dict[str, Any]:
    """Builds the audit record without depending on Torch tensor types."""
    validate_workload(query, documents)
    scores = [float(score) for score in _lists(logits)]
    input_ids = [_lists(row) for row in _lists(encoded["input_ids"])]
    attention = [_lists(row) for row in _lists(encoded["attention_mask"])]
    raw_token_types = encoded.get("token_type_ids")
    token_types = (
        [_lists(row) for row in _lists(raw_token_types)]
        if raw_token_types is not None
        else [[0] * len(row) for row in input_ids]
    )
    expected = len(documents)
    if not (len(scores) == len(input_ids) == len(attention) == len(token_types) == expected):
        raise ValueError("reference output cardinality does not match the document workload")

    pairs: list[dict[str, Any]] = []
    for index, (tokens, types, mask, score) in enumerate(
        zip(input_ids, token_types, attention, scores, strict=True)
    ):
        if not (len(tokens) == len(types) == len(mask)):
            raise ValueError(f"encoded pair {index} has inconsistent tensor lengths")
        retained = [position for position, visible in enumerate(mask) if int(visible) != 0]
        pairs.append(
            {
                "documentIndex": index,
                "tokens": [int(tokens[position]) for position in retained],
                "tokenTypes": [int(types[position]) for position in retained],
                "logit": score,
            }
        )

    ranking = sorted(range(expected), key=lambda index: (-scores[index], index))
    return {
        "schemaVersion": 1,
        "reference": {
            "backend": "hugging-face-transformers",
            "modelId": model_id,
            "requestedRevision": requested_revision,
            "resolvedRevision": resolved_revision,
            "dependencies": dict(dependency_versions),
        },
        "workload": {"query": query, "documents": list(documents)},
        "pairs": pairs,
        "ranking": ranking,
    }


def run_reference(
    *,
    model_id: str,
    revision: str,
    query: str,
    documents: Sequence[str],
    maximum_length: int,
    threads: int,
    local_files_only: bool,
) -> dict[str, Any]:
    """Runs the pinned unquantized sequence-classification graph on CPU."""
    validate_workload(query, documents)
    if not revision or not revision.strip():
        raise ValueError("an immutable model revision is required")
    if maximum_length < 3:
        raise ValueError("maximum length must be at least three tokens")
    if threads < 1:
        raise ValueError("threads must be positive")

    import torch
    import transformers
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        revision=revision,
        local_files_only=local_files_only,
        trust_remote_code=False,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        revision=revision,
        local_files_only=local_files_only,
        trust_remote_code=False,
    ).eval()
    encoded = tokenizer(
        [query] * len(documents),
        list(documents),
        add_special_tokens=True,
        padding=True,
        truncation=True,
        max_length=maximum_length,
        return_attention_mask=True,
        return_token_type_ids=True,
        return_tensors="pt",
    )
    with torch.inference_mode():
        logits = model(**encoded).logits.reshape(-1)
    resolved_revision = getattr(model.config, "_commit_hash", None) or revision
    return build_report(
        model_id=model_id,
        requested_revision=revision,
        resolved_revision=str(resolved_revision),
        query=query,
        documents=documents,
        encoded=encoded,
        logits=logits,
        dependency_versions={
            "python": os.sys.version.split()[0],
            "torch": importlib.metadata.version("torch"),
            "transformers": transformers.__version__,
        },
    )


def load_workload(path: Path) -> tuple[str, tuple[str, ...]]:
    """Loads a versioned query/document workload from JSON."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("workload must be a JSON object")
    query = payload.get("query")
    documents = payload.get("documents")
    if not isinstance(query, str) or not isinstance(documents, list):
        raise ValueError("workload requires a string query and document array")
    if any(not isinstance(document, str) for document in documents):
        raise ValueError("every workload document must be a string")
    resolved = tuple(documents)
    validate_workload(query, resolved)
    return query, resolved


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a pinned Transformers cross-encoder reference and retain exact inputs."
    )
    parser.add_argument("--model", required=True, help="Hugging Face model ID or local directory")
    parser.add_argument("--revision", required=True, help="Immutable Hugging Face commit")
    parser.add_argument("--workload", required=True, type=Path, help="Query/documents JSON file")
    parser.add_argument("--output", type=Path, help="Output JSON file; stdout when omitted")
    parser.add_argument("--maximum-length", type=int, default=512)
    parser.add_argument("--threads", type=int, default=max(1, os.cpu_count() or 1))
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    query, documents = load_workload(args.workload)
    report = run_reference(
        model_id=args.model,
        revision=args.revision,
        query=query,
        documents=documents,
        maximum_length=args.maximum_length,
        threads=args.threads,
        local_files_only=args.local_files_only,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
