#!/usr/bin/env python3
"""Split the multi-option-set corpus into train/validation, holding one option set out entirely.

The held-out set is the experiment: every frozen-probe variant measured on 2026-09-21 could
separate a fixed trained class set and nothing else, so the question is whether an adapter that
reshapes the representation can answer a question whose options it has never seen.
"""
import argparse, hashlib, json, random
from pathlib import Path

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--holdout", required=True, help="option set excluded from training entirely")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()

    rows = [json.loads(l) for l in a.corpus.open()]
    train, validation, holdout = [], [], []
    rng = random.Random(20260921)
    for r in rows:
        if r["optionSet"] == a.holdout:
            holdout.append(r)
        else:
            (validation if rng.random() < 0.12 else train).append(r)

    a.out.mkdir(parents=True, exist_ok=True)
    for name, part in (("train", train), ("validation", validation), ("holdout", holdout)):
        path = a.out / f"{name}.jsonl"
        with path.open("w") as f:
            for r in part:
                f.write(json.dumps(r, separators=(",", ":")) + "\n")
        sets = sorted({r["optionSet"] for r in part})
        print(f"  {name:11} {len(part):5} rows  sets {sets}")
        print(f"              sha256 {hashlib.sha256(path.read_bytes()).hexdigest()[:16]}")

    trained_sets = {r["optionSet"] for r in train}
    assert a.holdout not in trained_sets, "holdout leaked into training"
    print(f"\n  holdout '{a.holdout}' is absent from training: verified")

if __name__ == "__main__":
    main()
