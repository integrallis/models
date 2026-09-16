#!/usr/bin/env python3
"""Check that two activated-answerability reports produced identical outputs case by case.

The preflight amendment admits the Rust FFM kernel arm for gates 4 and 5 only if its outputs are
identical to the pure-Java reference on the first ten cases of every suite. Identity means the same
case ids in the same order, and byte-identical ``output`` and ``prediction`` for every case; timing
differs by construction and is reported, not compared.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def load(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def compare(reference: dict, candidate: dict) -> list[str]:
    problems: list[str] = []
    for key in ("suite", "arm", "windowSha256", "windowFileSha256", "modelsRevision"):
        if reference.get(key) != candidate.get(key):
            problems.append(f"{key} differs: {reference.get(key)!r} vs {candidate.get(key)!r}")
    ref_cases = reference["cases"]
    cand_cases = candidate["cases"]
    if [c["id"] for c in ref_cases] != [c["id"] for c in cand_cases]:
        problems.append("case ids or order differ")
        return problems
    for ref, cand in zip(ref_cases, cand_cases):
        for field in ("output", "prediction", "structured", "physicallyShared", "sharedPrefixTokens"):
            if ref.get(field) != cand.get(field):
                problems.append(f"case {ref['id']} {field}: {ref.get(field)!r} != {cand.get(field)!r}")
    return problems


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: compare_identity_arms.py REFERENCE.json CANDIDATE.json")
    reference = load(Path(sys.argv[1]))
    candidate = load(Path(sys.argv[2]))
    problems = compare(reference, candidate)
    ref_ms = sum(c["millis"] for c in reference["cases"])
    cand_ms = sum(c["millis"] for c in candidate["cases"])
    print(
        f"{reference['backend']} vs {candidate['backend']} on {reference['suite']}: "
        f"{len(reference['cases'])} cases, total {ref_ms} ms vs {cand_ms} ms"
    )
    if problems:
        print("NOT IDENTICAL")
        for problem in problems:
            print(f"  {problem}")
        raise SystemExit(1)
    print("IDENTICAL")


if __name__ == "__main__":
    main()
