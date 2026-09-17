#!/usr/bin/env python3
"""Label-audit stage for answerability qualification suites.

A suite may gate a component only after its labels are audited. Harness rule written after the
MS MARCO v2.1 validation suite turned out to carry wrong "No Answer Present." labels: two blind
judges contradicted 35 of its 100 unanswerable labels. Every adapter's shortfall was the
instrument's, and four training pilots were spent against it.

    audit_suite_labels.py export     --window W --suite S --out-dir D [--parts 3] [--seed 20260917]
    audit_suite_labels.py adjudicate --window W --suite S --judge A=glob --judge B=glob --out F
    audit_suite_labels.py rescore    --adjudicated F --arm name=path ... [--json out]

export     writes blind, shuffled parts (id, question, documents; no label, no prediction).
adjudicate applies the pre-registered rule and the admission rule, and writes the adjudicated
           suite with a canonical cases sha256. The exit status is 0 when the suite is admissible
           (original or adjudicated labels), and 3 when it is unusable.
rescore    re-scores recorded arm outputs on the adjudicated labels, with Wilson 95 % intervals.

Adjudication (pre-registered 2026-09-17T12:05Z):
  keep the dataset label if at least one judge agrees with it; flip it if both judges contradict
  it definitely; otherwise exclude the case.
Admission (pre-registered 2026-09-17T12:40Z, before any audit other than MS MARCO's):
  ORIGINAL_ADMISSIBLE if, for every label, flipped + excluded <= 10 % of that label's cases;
  ADJUDICATED_ONLY if total exclusions <= 25 % and every label keeps >= 30 cases;
  UNUSABLE otherwise.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import pathlib
import random
import sys

LABELS = ("answerable", "unanswerable")
OPPOSITE = {"answerable": "unanswerable", "unanswerable": "answerable"}
JUDGEMENTS = LABELS + ("ambiguous",)
ORIGINAL_MAX_DISPUTED = 0.10
ADJUDICATED_MAX_EXCLUDED = 0.25
ADJUDICATED_MIN_PER_LABEL = 30


def suite_cases(window: dict, name: str) -> list[dict]:
    for suite in window["suites"]:
        if suite["name"] == name:
            return suite["cases"]
    raise SystemExit(f"suite {name!r} not in window")


def question_of(case: dict) -> str:
    return "\n".join(f"{m['role']}: {m['text']}" for m in case["messages"])


def export(cases: list[dict], parts: int, seed: int) -> list[list[dict]]:
    blind = [{"id": c["id"], "question": question_of(c), "documents": c["documents"]} for c in cases]
    random.Random(seed).shuffle(blind)
    return [blind[k::parts] for k in range(parts)]


def adjudicate_case(label: str, a: str, b: str) -> str | None:
    """Returns the adjudicated label, or None when the case is excluded."""
    for j in (a, b):
        if j not in JUDGEMENTS:
            raise ValueError(f"unknown judgement {j!r}")
    if a == label or b == label:
        return label
    if a == b == OPPOSITE[label]:
        return OPPOSITE[label]
    return None


def adjudicate(cases: list[dict], judges: dict[str, dict[str, str]]) -> dict:
    ids = {c["id"] for c in cases}
    for name, table in judges.items():
        if set(table) != ids:
            missing, extra = sorted(ids - set(table)), sorted(set(table) - ids)
            raise ValueError(f"judge {name}: missing {missing[:5]} extra {extra[:5]}")
    a_name, b_name = sorted(judges)
    per_label = {l: {"cases": 0, "kept": 0, "flipped": 0, "excluded": 0} for l in LABELS}
    out = []
    for c in cases:
        label = c["label"]
        verdict = adjudicate_case(label, judges[a_name][c["id"]], judges[b_name][c["id"]])
        row = per_label[label]
        row["cases"] += 1
        if verdict is None:
            row["excluded"] += 1
            continue
        row["kept" if verdict == label else "flipped"] += 1
        out.append({"id": c["id"], "label": verdict, "datasetLabel": label})
    admission = admit(per_label, out)
    canon = json.dumps(sorted(out, key=lambda r: r["id"]), sort_keys=True, separators=(",", ":"))
    return {"perDatasetLabel": per_label, "admission": admission,
            "casesSha256": hashlib.sha256(canon.encode()).hexdigest(), "cases": out}


def admit(per_label: dict, adjudicated: list[dict]) -> str:
    if all((r["flipped"] + r["excluded"]) <= ORIGINAL_MAX_DISPUTED * r["cases"] for r in per_label.values()):
        return "ORIGINAL_ADMISSIBLE"
    total = sum(r["cases"] for r in per_label.values())
    excluded = sum(r["excluded"] for r in per_label.values())
    counts = {l: sum(1 for c in adjudicated if c["label"] == l) for l in LABELS}
    if excluded <= ADJUDICATED_MAX_EXCLUDED * total and min(counts.values()) >= ADJUDICATED_MIN_PER_LABEL:
        return "ADJUDICATED_ONLY"
    return "UNUSABLE"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (centre - half, centre + half)


def score(predictions: dict[str, str], labels: dict[str, str]) -> dict:
    missing = sorted(set(labels) - set(predictions))
    if missing:
        raise ValueError(f"arm lacks predictions for {missing[:5]}")
    counts = {l: [0, 0] for l in LABELS}
    for cid, label in labels.items():
        counts[label][1] += 1
        counts[label][0] += predictions[cid] == label
    recall = {l: counts[l][0] / counts[l][1] for l in LABELS}
    intervals = {l: wilson(*counts[l]) for l in LABELS}
    return {"balancedAccuracy": 0.5 * (recall["answerable"] + recall["unanswerable"]),
            "counts": counts,
            # Conservative bound: the mean of the per-label Wilson bounds.
            "balancedInterval95": [0.5 * (intervals["answerable"][0] + intervals["unanswerable"][0]),
                                   0.5 * (intervals["answerable"][1] + intervals["unanswerable"][1])]}


def load_judge(pattern: str) -> dict[str, str]:
    table: dict[str, str] = {}
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"no judge files match {pattern}")
    for f in files:
        for r in json.loads(pathlib.Path(f).read_text()):
            if r["id"] in table:
                raise SystemExit(f"{f}: duplicate id {r['id']}")
            table[r["id"]] = r["judgment"]
    return table


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--window", type=pathlib.Path, required=True); e.add_argument("--suite", required=True)
    e.add_argument("--out-dir", type=pathlib.Path, required=True)
    e.add_argument("--parts", type=int, default=3); e.add_argument("--seed", type=int, default=20260917)
    a = sub.add_parser("adjudicate")
    a.add_argument("--window", type=pathlib.Path, required=True); a.add_argument("--suite", required=True)
    a.add_argument("--judge", action="append", required=True, help="NAME=glob of judge JSON files")
    a.add_argument("--out", type=pathlib.Path, required=True)
    r = sub.add_parser("rescore")
    r.add_argument("--adjudicated", type=pathlib.Path, required=True)
    r.add_argument("--arm", action="append", required=True, help="NAME=path of a recorded window output")
    r.add_argument("--json", type=pathlib.Path)
    args = p.parse_args(argv)

    if args.cmd == "export":
        window = json.loads(args.window.read_text())
        args.out_dir.mkdir(parents=True, exist_ok=True)
        for k, part in enumerate(export(suite_cases(window, args.suite), args.parts, args.seed)):
            (args.out_dir / f"{args.suite}-blind-part{k}.json").write_text(json.dumps(part, indent=1) + "\n")
            print(f"part {k}: {len(part)} cases")
        return 0
    if args.cmd == "adjudicate":
        window = json.loads(args.window.read_text())
        judges = dict(spec.split("=", 1) for spec in args.judge)
        if len(judges) != 2:
            raise SystemExit("exactly two judges are required")
        result = adjudicate(suite_cases(window, args.suite), {n: load_judge(g) for n, g in judges.items()})
        result = {"suite": args.suite, "sourceWindowSha256": window["windowSha256"], **result}
        args.out.write_text(json.dumps(result, indent=1) + "\n")
        print(json.dumps({k: result[k] for k in ("suite", "perDatasetLabel", "admission", "casesSha256")}))
        return 3 if result["admission"] == "UNUSABLE" else 0
    adjudicated = json.loads(args.adjudicated.read_text())
    labels = {c["id"]: c["label"] for c in adjudicated["cases"]}
    rows = []
    for spec in args.arm:
        name, path = spec.split("=", 1)
        cases = json.loads(pathlib.Path(path).read_text())["cases"]
        original = {c["id"]: c["label"] for c in cases}
        preds = {c["id"]: c["prediction"] for c in cases}
        rows.append({"arm": name, "original": score(preds, original), "adjudicated": score(preds, labels)})
        o, j = rows[-1]["original"], rows[-1]["adjudicated"]
        print(f"{name:28} original {o['balancedAccuracy']:.3f}  adjudicated {j['balancedAccuracy']:.3f} "
              f"[{j['balancedInterval95'][0]:.3f}, {j['balancedInterval95'][1]:.3f}]  counts {j['counts']}")
    if args.json:
        args.json.write_text(json.dumps(rows, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
