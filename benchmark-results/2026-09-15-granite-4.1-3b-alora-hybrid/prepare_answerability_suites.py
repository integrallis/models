#!/usr/bin/env python3
"""Freeze the two answerability qualification slices from pinned public source files.

The output binds every source file by SHA-256, selects cases by the lowest unsigned SHA-256 of
``<salt>:<case id>`` within each label stratum, and records the exact messages, documents, and
label for every case so the Java runner and any independent scorer see identical inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

SALT = "20260915"
LABELS = ("answerable", "unanswerable")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rank_key(case_id: str, salt: str = SALT) -> str:
    return hashlib.sha256(f"{salt}:{case_id}".encode()).hexdigest()


def select(cases: list[dict[str, Any]], label: str, count: int) -> list[dict[str, Any]]:
    stratum = sorted((c for c in cases if c["label"] == label), key=lambda c: rank_key(c["id"]))
    if len(stratum) < count:
        raise ValueError(f"only {len(stratum)} {label} cases; requested {count}")
    return stratum[:count]


def mtrag_cases(path: Path) -> list[dict[str, Any]]:
    cases = []
    with path.open() as source:
        for number, line in enumerate(source, 1):
            row = json.loads(line)
            labels = row.get("Answerability")
            if not isinstance(labels, list) or len(labels) != 1:
                raise ValueError(f"RAG.jsonl:{number}: Answerability must be a single-label list")
            if labels[0] not in ("ANSWERABLE", "UNANSWERABLE"):
                continue
            task_id = row["task_id"]
            if not isinstance(task_id, str) or not task_id:
                raise ValueError(f"RAG.jsonl:{number}: task_id required")
            messages = []
            for turn in row["input"]:
                speaker = turn["speaker"]
                if speaker not in ("user", "agent"):
                    raise ValueError(f"RAG.jsonl:{number}: unexpected speaker {speaker}")
                messages.append(
                    {"role": "assistant" if speaker == "agent" else "user", "text": turn["text"]}
                )
            if not messages or messages[-1]["role"] != "user":
                raise ValueError(f"RAG.jsonl:{number}: conversation must end with a user turn")
            documents = [
                {"doc_id": index + 1, "text": context["text"]}
                for index, context in enumerate(row["contexts"])
            ]
            if not documents:
                raise ValueError(f"RAG.jsonl:{number}: a RAG task must carry contexts")
            cases.append(
                {
                    "id": task_id,
                    "label": labels[0].lower(),
                    "messages": messages,
                    "documents": documents,
                }
            )
    return cases


def squad_cases(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    if data.get("version") != "v2.0":
        raise ValueError("SQuAD source must be version v2.0")
    cases = []
    for article in data["data"]:
        for paragraph in article["paragraphs"]:
            document = {"doc_id": 1, "text": paragraph["context"]}
            for question in paragraph["qas"]:
                impossible = question["is_impossible"]
                if impossible and question["answers"]:
                    raise ValueError(f"{question['id']}: impossible question carries answers")
                if not impossible and not question["answers"]:
                    raise ValueError(f"{question['id']}: possible question carries no answers")
                cases.append(
                    {
                        "id": question["id"],
                        "label": "unanswerable" if impossible else "answerable",
                        "messages": [{"role": "user", "text": question["question"]}],
                        "documents": [document],
                    }
                )
    return cases


NO_ANSWER = "No Answer Present."


def msmarco_cases(path: Path) -> list[dict[str, Any]]:
    """MS MARCO v2.1 rows as single-turn cases: every passage of the query is a document, the
    label is unanswerable when the only answer is the dataset's no-answer marker and answerable
    when there is at least one answer and none is the marker. Rows with no passages, no answers,
    or a mix of the marker and answers are ineligible."""
    import pyarrow.parquet as pq  # imported here: only the MS MARCO source needs it

    table = pq.read_table(path, columns=["query_id", "query", "passages", "answers"])
    cases = []
    for row in table.to_pylist():
        answers = row["answers"] or []
        passages = (row["passages"] or {}).get("passage_text") or []
        if not answers or not passages:
            continue
        marker = [a == NO_ANSWER for a in answers]
        if all(marker):
            label = "unanswerable"
        elif not any(marker):
            label = "answerable"
        else:
            continue
        cases.append(
            {
                "id": str(row["query_id"]),
                "label": label,
                "messages": [{"role": "user", "text": row["query"]}],
                "documents": [
                    {"doc_id": index + 1, "text": text} for index, text in enumerate(passages)
                ],
            }
        )
    return cases


def prepare(
    mtrag_path: Path,
    squad_path: Path,
    mtrag_revision: str,
    msmarco_path: Path | None = None,
    msmarco_revision: str | None = None,
) -> dict[str, Any]:
    mtrag = mtrag_cases(mtrag_path)
    squad = squad_cases(squad_path)
    unanswerable = [c for c in mtrag if c["label"] == "unanswerable"]
    mtrag_selected = select(mtrag, "unanswerable", len(unanswerable)) + select(
        mtrag, "answerable", len(unanswerable)
    )
    squad_selected = select(squad, "unanswerable", 100) + select(squad, "answerable", 100)
    plan = [
        ("mtrag-human-rag", mtrag_path, mtrag_revision, mtrag_selected, mtrag),
        ("squad-v2-dev", squad_path, None, squad_selected, squad),
    ]
    if msmarco_path is not None:
        msmarco = msmarco_cases(msmarco_path)
        msmarco_selected = select(msmarco, "unanswerable", 100) + select(msmarco, "answerable", 100)
        plan.append(("msmarco-v2.1-validation", msmarco_path, msmarco_revision, msmarco_selected, msmarco))
    suites = []
    for name, source, revision, selected, eligible in plan:
        ordered = sorted(selected, key=lambda c: rank_key(c["id"]))
        suites.append(
            {
                "name": name,
                "source": {
                    "file": source.name,
                    "sha256": sha256(source),
                    **({"revision": revision} if revision else {}),
                },
                "eligibleCases": len(eligible),
                "eligibleByLabel": {
                    label: sum(1 for c in eligible if c["label"] == label) for label in LABELS
                },
                "selectedByLabel": {
                    label: sum(1 for c in ordered if c["label"] == label) for label in LABELS
                },
                "cases": ordered,
            }
        )
    window = {
        "schemaVersion": 1,
        "salt": SALT,
        "selectionRule": "lowest unsigned SHA-256 of '<salt>:<case id>' within each label stratum",
        "labels": list(LABELS),
        "suites": suites,
    }
    canonical = json.dumps(window, sort_keys=True, separators=(",", ":")).encode()
    window["windowSha256"] = hashlib.sha256(canonical).hexdigest()
    return window


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mtrag-rag-jsonl", type=Path, required=True)
    parser.add_argument("--mtrag-revision", required=True)
    parser.add_argument("--squad-dev-json", type=Path, required=True)
    parser.add_argument("--msmarco-parquet", type=Path, help="MS MARCO v2.1 validation parquet")
    parser.add_argument("--msmarco-revision", help="Hugging Face dataset revision of the parquet")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite a frozen window: {args.output}")
    if (args.msmarco_parquet is None) != (args.msmarco_revision is None):
        raise SystemExit("--msmarco-parquet and --msmarco-revision go together")
    window = prepare(
        args.mtrag_rag_jsonl,
        args.squad_dev_json,
        args.mtrag_revision,
        args.msmarco_parquet,
        args.msmarco_revision,
    )
    args.output.write_text(json.dumps(window, indent=2, sort_keys=True) + "\n")
    for suite in window["suites"]:
        print(suite["name"], suite["selectedByLabel"], "of", suite["eligibleByLabel"])
    print("window", window["windowSha256"])


if __name__ == "__main__":
    main()
