#!/usr/bin/env python3
"""Automated confirmation of disputed answerability labels by verifiable evidence.

Two blind model judges disagreed with dataset labels on some window cases (see
audit_suite_labels.py). Judgement alone is weak evidence, so every disputed case is re-decided
here by a protocol whose central claim is checked by code:

  1. extract  An extractor reads question + passages and returns the shortest verbatim quote that
              answers the question, or null. It sees no label.
  2. verify   Code checks that the quote occurs verbatim (whitespace- and case-normalised) in the
              named passage. An unverifiable quote counts as no quote.
  3. judge    Two verifiers independently see only the question, the one passage and the quote,
              and answer yes/no: does the quote directly answer the question?
  4. decide   answerable iff the quote verified AND both verifiers say yes; otherwise unanswerable.

Controls with trusted labels (SQuAD v2 cases on which both audit judges agreed with the dataset)
are mixed in blind. The protocol is admissible only if it calls at most 2 of 20 control
unanswerables answerable, and at least 18 of 20 control answerables answerable.

    confirm_disputed_labels.py prepare --window W --adjudicated suite=F ... --out-dir D [--seed N] [--parts 3]
    confirm_disputed_labels.py verify-quotes --out-dir D
    confirm_disputed_labels.py decide --out-dir D --adjudicated suite=F ...
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import pathlib
import random
import re
import sys

CONTROLS_PER_LABEL = 20
CONTROL_MAX_FALSE_ANSWERABLE = 2
CONTROL_MIN_TRUE_ANSWERABLE = 18
MIN_QUOTE_CHARS = 4
CONTROL_SUITE = "squad-v2-dev"


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def quote_verifies(quote: str | None, passage: str) -> bool:
    if not quote:
        return False
    q = norm(quote).strip(" \"'“”‘’.,;:")
    return len(q) >= MIN_QUOTE_CHARS and q in norm(passage)


def question_of(case: dict) -> str:
    return "\n".join(f"{m['role']}: {m['text']}" for m in case["messages"])


def passages_of(case: dict) -> list[dict]:
    docs = case["documents"]
    if not isinstance(docs, list):
        raise ValueError(f"case {case['id']}: documents must be a list")
    return [{"doc_id": d["doc_id"], "text": d["text"]} for d in docs]


def disputed_ids(window_cases: list[dict], adjudicated: dict) -> dict[str, str]:
    """id -> 'flipped' | 'excluded' for every case whose dataset label was not simply kept."""
    kept = {c["id"]: c for c in adjudicated["cases"]}
    out = {}
    for c in window_cases:
        k = kept.get(c["id"])
        if k is None:
            out[c["id"]] = "excluded"
        elif k["label"] != k["datasetLabel"]:
            out[c["id"]] = "flipped"
    return out


def prepare(window: dict, adjudicated: dict[str, dict], judges: dict[str, dict[str, str]], seed: int, parts: int):
    suites = {s["name"]: s["cases"] for s in window["suites"]}
    rng = random.Random(seed)
    items, key = [], {}
    for suite, adj in sorted(adjudicated.items()):
        for cid, why in sorted(disputed_ids(suites[suite], adj).items()):
            case = next(c for c in suites[suite] if c["id"] == cid)
            uid = f"{suite}:{cid}"
            items.append({"id": uid, "question": question_of(case), "passages": passages_of(case)})
            key[uid] = {"suite": suite, "caseId": cid, "datasetLabel": case["label"], "group": f"disputed-{why}"}
    control_cases = suites[CONTROL_SUITE]
    disputed = {k["caseId"] for k in key.values() if k["suite"] == CONTROL_SUITE}
    for label in ("answerable", "unanswerable"):
        pool = [c for c in control_cases if c["label"] == label and c["id"] not in disputed
                and all(j.get(c["id"]) == label for j in judges.values())]
        pool.sort(key=lambda c: c["id"])
        if len(pool) < CONTROLS_PER_LABEL:
            raise ValueError(f"only {len(pool)} {label} controls")
        for case in rng.sample(pool, CONTROLS_PER_LABEL):
            uid = f"{CONTROL_SUITE}:{case['id']}"
            items.append({"id": uid, "question": question_of(case), "passages": passages_of(case)})
            key[uid] = {"suite": CONTROL_SUITE, "caseId": case["id"], "datasetLabel": label, "group": f"control-{label}"}
    rng.shuffle(items)
    return [items[k::parts] for k in range(parts)], key


def verify_quotes(items: dict[str, dict], extractions: dict[str, dict]) -> dict[str, dict]:
    if set(items) != set(extractions):
        raise ValueError(f"extraction coverage mismatch: missing {sorted(set(items) - set(extractions))[:5]}")
    out = {}
    for uid, item in items.items():
        e = extractions[uid]
        quote, doc_id = e.get("quote"), e.get("doc_id")
        passage = next((p for p in item["passages"] if p["doc_id"] == doc_id), None)
        if (passage is None or not quote_verifies(quote, passage["text"])) and quote:
            passage = next((p for p in item["passages"] if quote_verifies(quote, p["text"])), None)
        ok = passage is not None and quote_verifies(quote, passage["text"])
        out[uid] = {"quote": quote, "verified": ok, "doc_id": passage["doc_id"] if ok else None,
                    "passage": passage["text"] if ok else None, "question": item["question"]}
    return out


def decide_label(verification: dict, verdicts: list[str | None]) -> str:
    if not verification["verified"]:
        return "unanswerable"
    return "answerable" if all(v == "yes" for v in verdicts) else "unanswerable"


def decide(key: dict, verification: dict, verifiers: dict[str, dict[str, str]]) -> dict:
    verified = {u for u, v in verification.items() if v["verified"]}
    for name, table in verifiers.items():
        if set(table) != verified:
            raise ValueError(f"verifier {name} coverage mismatch")
    labels = {u: decide_label(verification[u], [t.get(u) for t in verifiers.values()]) for u in key}
    control = {g: [labels[u] for u, k in key.items() if k["group"] == g] for g in ("control-answerable", "control-unanswerable")}
    false_ans = sum(l == "answerable" for l in control["control-unanswerable"])
    true_ans = sum(l == "answerable" for l in control["control-answerable"])
    admissible = false_ans <= CONTROL_MAX_FALSE_ANSWERABLE and true_ans >= CONTROL_MIN_TRUE_ANSWERABLE
    return {"labels": labels, "controls": {"falseAnswerableOf20": false_ans, "trueAnswerableOf20": true_ans},
            "protocolAdmissible": admissible}


def confirmed_suite(suite: str, adjudicated: dict, key: dict, labels: dict) -> dict:
    kept = {c["id"]: c for c in adjudicated["cases"] if c["label"] == c["datasetLabel"]}
    cases = [{"id": cid, "label": c["label"], "datasetLabel": c["datasetLabel"], "source": "audit-kept"} for cid, c in kept.items()]
    for uid, k in key.items():
        if k["suite"] == suite and k["group"].startswith("disputed"):
            cases.append({"id": k["caseId"], "label": labels[uid], "datasetLabel": k["datasetLabel"], "source": "evidence-confirmed"})
    cases.sort(key=lambda c: c["id"])
    canon = json.dumps(cases, sort_keys=True, separators=(",", ":"))
    return {"suite": suite, "cases": cases, "casesSha256": hashlib.sha256(canon.encode()).hexdigest()}


def read_list(pattern: str) -> list[dict]:
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"nothing matches {pattern}")
    rows = []
    for f in files:
        rows.extend(json.loads(pathlib.Path(f).read_text()))
    return rows


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("prepare", "verify-quotes", "decide"):
        s = sub.add_parser(name)
        s.add_argument("--out-dir", type=pathlib.Path, required=True)
        if name in ("prepare", "decide"):
            s.add_argument("--adjudicated", action="append", required=True, help="suite=path")
        if name == "prepare":
            s.add_argument("--window", type=pathlib.Path, required=True)
            s.add_argument("--judge", action="append", required=True, help="NAME=glob of audit judge files for the control suite")
            s.add_argument("--seed", type=int, default=20260919)
            s.add_argument("--parts", type=int, default=3)
    a = p.parse_args(argv)
    d = a.out_dir
    if a.cmd == "prepare":
        window = json.loads(a.window.read_text())
        adjudicated = {k: json.loads(pathlib.Path(v).read_text()) for k, v in (s.split("=", 1) for s in a.adjudicated)}
        judges = {}
        for spec in a.judge:
            n, g = spec.split("=", 1)
            judges[n] = {r["id"]: r["judgment"] for r in read_list(g)}
        parts, key = prepare(window, adjudicated, judges, a.seed, a.parts)
        d.mkdir(parents=True, exist_ok=True)
        for i, part in enumerate(parts):
            (d / f"extract-part{i}.json").write_text(json.dumps(part, indent=1) + "\n")
        (d / "key.json").write_text(json.dumps(key, indent=1) + "\n")
        groups = {}
        for k in key.values():
            groups[k["group"]] = groups.get(k["group"], 0) + 1
        print(json.dumps({"items": len(key), "groups": groups, "parts": [len(x) for x in parts]}))
        return 0
    items = {i["id"]: i for i in read_list(str(d / "extract-part*.json"))}
    if a.cmd == "verify-quotes":
        extractions = {r["id"]: r for r in read_list(str(d / "extraction-part*.json"))}
        verification = verify_quotes(items, extractions)
        (d / "verification.json").write_text(json.dumps(verification, indent=1) + "\n")
        todo = [{"id": u, "question": v["question"], "passage": v["passage"], "quote": v["quote"]}
                for u, v in sorted(verification.items()) if v["verified"]]
        random.Random(1).shuffle(todo)
        (d / "verify-input.json").write_text(json.dumps(todo, indent=1) + "\n")
        claimed = sum(1 for e in extractions.values() if e.get("quote"))
        print(json.dumps({"items": len(items), "quotesClaimed": claimed, "quotesVerified": len(todo)}))
        return 0
    key = json.loads((d / "key.json").read_text())
    verification = json.loads((d / "verification.json").read_text())
    verifiers = {}
    for f in sorted(glob.glob(str(d / "verdicts-*.json"))):
        name = pathlib.Path(f).stem.split("-", 1)[1]
        verifiers[name] = {r["id"]: r["verdict"] for r in json.loads(pathlib.Path(f).read_text())}
    if len(verifiers) != 2:
        raise SystemExit(f"expected 2 verifier files, found {sorted(verifiers)}")
    result = decide(key, verification, verifiers)
    adjudicated = {k: json.loads(pathlib.Path(v).read_text()) for k, v in (s.split("=", 1) for s in a.adjudicated)}
    summary = {"controls": result["controls"], "protocolAdmissible": result["protocolAdmissible"], "suites": {}}
    for suite, adj in adjudicated.items():
        conf = confirmed_suite(suite, adj, key, result["labels"])
        (d / f"confirmed-{suite}.json").write_text(json.dumps(conf, indent=1) + "\n")
        moved = {}
        for uid, k in key.items():
            if k["suite"] == suite and k["group"].startswith("disputed"):
                t = f"{k['group']}:{k['datasetLabel']}->{result['labels'][uid]}"
                moved[t] = moved.get(t, 0) + 1
        summary["suites"][suite] = {"cases": len(conf["cases"]), "casesSha256": conf["casesSha256"], "disputedOutcomes": moved}
    (d / "decision.json").write_text(json.dumps({**summary, "labels": result["labels"]}, indent=1) + "\n")
    print(json.dumps(summary, indent=1))
    return 0 if result["protocolAdmissible"] else 3


if __name__ == "__main__":
    sys.exit(main())
