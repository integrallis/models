#!/usr/bin/env python3
"""Screen an activated query-rewriter against MT-RAG Cloud retrieval relevance.

This is a reproducible *screen*, not a replacement for a production vector-store
benchmark.  It applies one pinned BM25 implementation to the final user turn,
the MT-RAG reference rewrite, and the model rewrite, then compares each against
the benchmark's judged passage ids.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

TOKEN = re.compile(r"[a-z0-9]+")
K_VALUES = (1, 3, 5, 10)


def tokens(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def last_user_message(messages: list[dict]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user" and isinstance(message.get("text"), str):
            return message["text"]
    raise ValueError("case has no user message")


def join_cases(suite: dict, report: dict) -> list[dict]:
    suite_cases = {case["id"]: case for case in suite["cases"]}
    report_cases = {case["id"]: case for case in report["cases"]}
    if suite_cases.keys() != report_cases.keys():
        raise ValueError("suite and report case ids differ")
    joined = []
    for identifier, source in suite_cases.items():
        generated = report_cases[identifier]
        if not generated.get("structured") or not generated.get("physicallyShared"):
            raise ValueError(f"case {identifier} is not a shared structured result")
        actual = generated.get("actualRewrite")
        if not isinstance(actual, str) or not actual.strip():
            raise ValueError(f"case {identifier} has no generated rewrite")
        joined.append(
            {
                "id": identifier,
                "queryId": source["queryId"],
                "raw": last_user_message(source["messages"]),
                "reference": source["expectedRewrite"],
                "generated": actual,
            }
        )
    return joined


def load_qrels(path: Path) -> dict[str, set[str]]:
    qrels: dict[str, set[str]] = defaultdict(set)
    for number, line in enumerate(path.read_text().splitlines()):
        if number == 0:
            continue
        fields = line.split("\t")
        if len(fields) != 3:
            raise ValueError(f"invalid qrels line {number + 1}")
        if int(fields[2]) > 0:
            qrels[fields[0]].add(fields[1])
    return qrels


def build_index(corpus_zip: Path, vocabulary: set[str]) -> tuple[list[tuple[str, int, Counter]], Counter, float]:
    """Retain only query-term postings, so the full corpus vocabulary is not held."""
    documents: list[tuple[str, int, Counter]] = []
    document_frequency: Counter = Counter()
    total_length = 0
    with zipfile.ZipFile(corpus_zip) as archive:
        names = [name for name in archive.namelist() if name.endswith(".jsonl")]
        if len(names) != 1:
            raise ValueError("expected one JSONL corpus member")
        with archive.open(names[0]) as source:
            for line in source:
                item = json.loads(line)
                identifier = item.get("_id")
                text = item.get("text")
                if not isinstance(identifier, str) or not isinstance(text, str):
                    raise ValueError("invalid corpus record")
                doc_tokens = tokens(text)
                relevant = Counter(token for token in doc_tokens if token in vocabulary)
                for term in relevant:
                    document_frequency[term] += 1
                documents.append((identifier, len(doc_tokens), relevant))
                total_length += len(doc_tokens)
    if not documents:
        raise ValueError("empty corpus")
    return documents, document_frequency, total_length / len(documents)


def rank(query: str, documents: Iterable[tuple[str, int, Counter]], document_frequency: Counter, average_length: float, document_count: int, limit: int = 10) -> list[str]:
    query_terms = set(tokens(query))
    k1, b = 1.2, 0.75
    scored = []
    for identifier, length, frequencies in documents:
        score = 0.0
        for term in query_terms:
            frequency = frequencies.get(term, 0)
            if not frequency:
                continue
            inverse_frequency = math.log(1 + (document_count - document_frequency[term] + 0.5) / (document_frequency[term] + 0.5))
            score += inverse_frequency * frequency * (k1 + 1) / (frequency + k1 * (1 - b + b * length / average_length))
        if score:
            scored.append((score, identifier))
    return [identifier for _, identifier in heapq.nlargest(limit, scored)]


def evaluate(joined: list[dict], qrels: dict[str, set[str]], documents: list[tuple[str, int, Counter]], document_frequency: Counter, average_length: float) -> dict:
    variants: dict[str, dict] = {}
    for variant in ("raw", "reference", "generated"):
        cases = []
        totals = {str(k): 0 for k in K_VALUES}
        for case in joined:
            relevant = qrels.get(case["queryId"])
            if not relevant:
                raise ValueError(f"no qrels for {case['queryId']}")
            ranked = rank(case[variant], documents, document_frequency, average_length, len(documents))
            hits = {str(k): bool(set(ranked[:k]) & relevant) for k in K_VALUES}
            for k, hit in hits.items():
                totals[k] += int(hit)
            cases.append({"id": case["id"], "queryId": case["queryId"], "query": case[variant], "top10": ranked, "hits": hits})
        variants[variant] = {"recallAt": {k: totals[k] / len(cases) for k in totals}, "cases": cases}
    return variants


def score(suite_path: Path, report_path: Path, qrels_path: Path, corpus_zip: Path) -> dict:
    suite = json.loads(suite_path.read_text())
    report = json.loads(report_path.read_text())
    joined = join_cases(suite, report)
    vocabulary = set().union(*(tokens(case[variant]) for case in joined for variant in ("raw", "reference", "generated")))
    documents, document_frequency, average_length = build_index(corpus_zip, vocabulary)
    variants = evaluate(joined, load_qrels(qrels_path), documents, document_frequency, average_length)
    raw = variants["raw"]["recallAt"]["10"]
    reference = variants["reference"]["recallAt"]["10"]
    generated = variants["generated"]["recallAt"]["10"]
    return {
        "schemaVersion": 1,
        "method": "bm25-k1.2-b0.75-query-term-postings-v1",
        "sources": {"suiteSha256": sha256(suite_path), "reportSha256": sha256(report_path), "qrelsSha256": sha256(qrels_path), "corpusSha256": sha256(corpus_zip)},
        "caseCount": len(joined),
        "corpusDocumentCount": len(documents),
        "variants": variants,
        "screen": {
            "rule": "generated Recall@10 >= raw Recall@10 and >= 90% of reference Recall@10",
            "rawRecallAt10": raw,
            "referenceRecallAt10": reference,
            "generatedRecallAt10": generated,
            "passed": generated >= raw and generated >= reference * 0.9,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--qrels", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing report: {args.output}")
    result = score(args.suite, args.report, args.qrels, args.corpus)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"screen passed: {result['screen']['passed']}; generated Recall@10={result['screen']['generatedRecallAt10']:.3f}")


if __name__ == "__main__":
    main()
