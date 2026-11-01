#!/usr/bin/env python3
"""Turn the JMH sweep into the one number the protocol's decision rule reads.

c(K) = T_verify(K) / T_decode(1), the cost of verifying a K-token block in units of single decode
steps. `batched` is the speculative path; `sequential` is K ordinary steps, and sequential at K=1
is one decode step, which is the denominator.

Nothing here decides anything on its own: it prints c(K), the implied speedup for the acceptance
lengths FlashMap published (imported, not measured here), and leaves the verdict to README.md.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
GAMMA = {"weakest (MT-Bench, 1 step)": 6.00, "strongest (GSM8K, 2 steps)": 8.31}
DRAFTER_COST = 0.06  # 0.5B drafter against an 8B target, both bandwidth-bound: a parameter ratio


def load(path: Path) -> dict:
    rows = {}
    for entry in json.loads(path.read_text()):
        mode = entry["benchmark"].rsplit(".", 1)[1]
        k = int(entry["params"]["draftLength"])
        metric = entry["primaryMetric"]
        rows[(mode, k)] = (metric["score"], metric["scoreError"], metric["scoreUnit"])
    return rows


def main() -> int:
    report = {}
    for path in sorted(HERE.glob("results-*.json")):
        tag = path.name[len("results-"):-len(".json")]
        rows = load(path)
        ks = sorted({k for _mode, k in rows})
        step = rows.get(("sequential", 1))
        if step is None:
            print(f"{tag}: no sequential K=1 row; cannot form c(K)")
            continue
        unit_cost, unit_err, unit = step
        print(f"\n== {tag}   one decode step = {unit_cost:.2f} +/- {unit_err:.2f} {unit}")
        print(f"{'K':>3} {'batched':>10} {'sequential':>11} {'c(K)':>7} {'seq/bat':>8}"
              f"  {'speedup at gamma=6.00':>22} {'at 8.31':>9}")
        table = {}
        for k in ks:
            bat = rows.get(("batched", k))
            seq = rows.get(("sequential", k))
            if not bat or not seq:
                continue
            c = bat[0] / unit_cost
            table[k] = {"batchedMs": round(bat[0], 3), "batchedError": round(bat[1], 3),
                        "sequentialMs": round(seq[0], 3), "sequentialError": round(seq[1], 3),
                        "blockCostInSteps": round(c, 3),
                        "sequentialOverBatched": round(seq[0] / bat[0], 3)}
            speedups = {name: g / (DRAFTER_COST + c) for name, g in GAMMA.items()}
            table[k]["projectedSpeedup"] = {name: round(v, 3) for name, v in speedups.items()}
            print(f"{k:>3} {bat[0]:>10.2f} {seq[0]:>11.2f} {c:>7.2f} {seq[0]/bat[0]:>8.2f}"
                  f"  {speedups['weakest (MT-Bench, 1 step)']:>22.2f}"
                  f" {speedups['strongest (GSM8K, 2 steps)']:>9.2f}")
        report[tag] = {"oneDecodeStepMs": round(unit_cost, 3), "unit": unit, "byDraftLength": table}

    (HERE / "RESULTS.json").write_text(json.dumps(
        {"drafterCostAssumption": DRAFTER_COST,
         "acceptanceLengthsImportedFromFlashMapTable1": GAMMA,
         "cells": report}, indent=2, sort_keys=True) + "\n")
    print(f"\nwrote {HERE / 'RESULTS.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
