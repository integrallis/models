#!/usr/bin/env python3
"""Build a balanced answerability training set from published *train* splits only:

* QuAC train (multi-turn; ``CANNOTANSWER`` is the unanswerable label; the grounding section plus
  distractor sections from other dialogues are the documents),
* SQuAD v2 train (single-turn; the grounding paragraph plus distractor paragraphs from the same
  article),
* MS MARCO v2.1 train shard (single-turn; every passage of the query, ``No Answer Present.`` is the
  unanswerable label).

Every record has the same shape as a qualification-window case ({messages, documents, label}) so
the trainer renders it exactly as the runner renders the window. Validation records are held out
by dialogue / article / query so nothing in the validation split shares a source document with
training. The qualification window itself is never read here.
"""
from __future__ import annotations
import argparse, hashlib, json, random
from pathlib import Path

SALT = "20260916-answerability-alora"

def stable(x: str) -> int:
    return int(hashlib.sha256(f"{SALT}:{x}".encode()).hexdigest(), 16)

def quac_records(path: Path, rng: random.Random, max_distractors: int) -> list[dict]:
    data = json.loads(path.read_text())["data"]
    sections = [(a["title"], p) for a in data for p in a["paragraphs"]]
    contexts = [p["context"] for _, p in sections]
    records = []
    for title, section in sections:
        history = []
        for qa in section["qas"]:
            question = qa["question"]
            unanswerable = qa["orig_answer"]["text"] == "CANNOTANSWER"
            messages = history + [{"role": "user", "text": question}]
            k = rng.randint(1, max_distractors)
            distractors = rng.sample(contexts, k)
            docs = [section["context"]] + [d for d in distractors if d != section["context"]]
            rng.shuffle(docs)
            records.append({
                "id": f"quac:{qa['id']}", "source": "quac-train", "group": section["id"],
                "messages": messages, "documents": [{"doc_id": i + 1, "text": t} for i, t in enumerate(docs)],
                "label": "unanswerable" if unanswerable else "answerable", "turns": len(messages),
            })
            answer = "I cannot answer that from the documents." if unanswerable else qa["orig_answer"]["text"]
            history = messages + [{"role": "assistant", "text": answer}]
    return records

def squad_records(path: Path, rng: random.Random, max_distractors: int) -> list[dict]:
    import pyarrow.parquet as pq
    t = pq.read_table(path, columns=["id", "title", "context", "question", "answers"]).to_pylist()
    by_title: dict[str, list[str]] = {}
    for r in t:
        by_title.setdefault(r["title"], [])
        if r["context"] not in by_title[r["title"]]:
            by_title[r["title"]].append(r["context"])
    records = []
    for r in t:
        unanswerable = len(r["answers"]["text"]) == 0
        others = [c for c in by_title[r["title"]] if c != r["context"]]
        k = min(rng.randint(0, max_distractors), len(others))
        docs = [r["context"]] + rng.sample(others, k)
        rng.shuffle(docs)
        records.append({
            "id": f"squad:{r['id']}", "source": "squad-v2-train", "group": r["title"],
            "messages": [{"role": "user", "text": r["question"]}],
            "documents": [{"doc_id": i + 1, "text": c} for i, c in enumerate(docs)],
            "label": "unanswerable" if unanswerable else "answerable", "turns": 1,
        })
    return records

def msmarco_records(path: Path) -> list[dict]:
    import pyarrow.parquet as pq
    t = pq.read_table(path, columns=["query_id", "query", "passages", "answers"]).to_pylist()
    records = []
    for r in t:
        answers = r["answers"] or []; passages = (r["passages"] or {}).get("passage_text") or []
        if not answers or not passages: continue
        marker = [a == "No Answer Present." for a in answers]
        if all(marker): label = "unanswerable"
        elif not any(marker): label = "answerable"
        else: continue
        records.append({
            "id": f"msmarco:{r['query_id']}", "source": "msmarco-v2.1-train", "group": str(r["query_id"]),
            "messages": [{"role": "user", "text": r["query"]}],
            "documents": [{"doc_id": i + 1, "text": p} for i, p in enumerate(passages)],
            "label": label, "turns": 1,
        })
    return records

def split_and_balance(records: list[dict], per_label: int, validation_per_label: int, rng: random.Random) -> tuple[list[dict], list[dict]]:
    groups = sorted({r["group"] for r in records}, key=stable)
    validation_groups = set(groups[: max(1, len(groups) // 20)])
    train_pool = [r for r in records if r["group"] not in validation_groups]
    validation_pool = [r for r in records if r["group"] in validation_groups]
    def take(pool: list[dict], count: int) -> list[dict]:
        out = []
        for label in ("answerable", "unanswerable"):
            stratum = sorted((r for r in pool if r["label"] == label), key=lambda r: stable(r["id"]))
            if len(stratum) < count: raise SystemExit(f"only {len(stratum)} {label} records in pool for {count}")
            out += stratum[:count]
        rng.shuffle(out); return out
    return take(train_pool, per_label), take(validation_pool, validation_per_label)

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--quac-train", type=Path, required=True); p.add_argument("--squad-train", type=Path, required=True)
    p.add_argument("--msmarco-train", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
    p.add_argument("--per-source-per-label", type=int, default=4000); p.add_argument("--validation-per-source-per-label", type=int, default=150)
    p.add_argument("--max-distractors", type=int, default=4); p.add_argument("--seed", type=int, default=20260916)
    p.add_argument("--source-per-label", action="append", default=[],
                   help="override training records per label for one source, e.g. msmarco-v2.1-train=8000")
    a = p.parse_args(); rng = random.Random(a.seed)
    a.output.mkdir(parents=True, exist_ok=True)
    sources = {"quac-train": quac_records(a.quac_train, rng, a.max_distractors),
               "squad-v2-train": squad_records(a.squad_train, rng, a.max_distractors),
               "msmarco-v2.1-train": msmarco_records(a.msmarco_train)}
    train, validation, manifest = [], [], {"salt": SALT, "seed": a.seed, "sources": {}}
    overrides = {}
    for spec in a.source_per_label:
        name, count = spec.split("=", 1)
        overrides[name] = int(count)
    unknown = set(overrides) - set(sources)
    if unknown:
        raise SystemExit(f"unknown sources in --source-per-label: {sorted(unknown)}")
    manifest["perLabel"] = {name: overrides.get(name, a.per_source_per_label) for name in sources}
    for name, records in sources.items():
        tr, va = split_and_balance(records, overrides.get(name, a.per_source_per_label), a.validation_per_source_per_label, rng)
        train += tr; validation += va
        manifest["sources"][name] = {"records": len(records), "train": len(tr), "validation": len(va)}
    rng.shuffle(train); rng.shuffle(validation)
    for split, rows in (("train", train), ("validation", validation)):
        with (a.output / f"{split}.jsonl").open("w") as f:
            for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
        manifest[split] = {"records": len(rows), "sha256": hashlib.sha256((a.output / f"{split}.jsonl").read_bytes()).hexdigest()}
    for name, path in (("quac-train", a.quac_train), ("squad-v2-train", a.squad_train), ("msmarco-v2.1-train", a.msmarco_train)):
        manifest["sources"][name]["file"] = path.name; manifest["sources"][name]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (a.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))

if __name__ == "__main__":
    main()
