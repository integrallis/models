#!/usr/bin/env python3
"""Derive the kernel-staleness result from the committed raw artifacts.

Every number in RESULTS.md comes from this script. Nothing is hand-typed.

    python3 experiments/kernel-staleness-delta/analyse.py

Two stages:

1. The paired run in `raw/` -- same box, same payload, same model, two kernels, each arm twice in
   opposite order -- gives the effect of the kernel on the two inputs the relative gate reads.
2. `raw/campaign-verdicts/` holds every verdict the campaign produced. For each QUALIFIED one it
   computes headroom: how far the measured ratio sat from its threshold. A verdict is only at risk
   if its headroom is smaller than the effect from stage 1.

The gate reads `decodeThroughputRatio` against `minimumDecodeThroughputRatio` and
`endToEndLatencyRatio` against `maximumEndToEndLatencyRatio` -- field names taken from the
artifacts, not from the policy source, so this script describes what was actually recorded.
"""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "raw")


def paired_effect():
    cells = {}
    for path in sorted(glob.glob(os.path.join(RAW, "*.json"))):
        base = os.path.basename(path)
        if base == "kernels.json":
            continue
        mid, arm, pas = base[:-5].rsplit(".", 2)
        rep = json.load(open(path))
        s = rep["summary"]
        cells.setdefault(mid, {})[(arm, pas)] = {
            "decode": s["p50DecodeTokensPerSecond"],
            "e2e95": s["endToEndMillis"]["p95"],
            "outTok": s["totalOutputTokens"],
            "correct": s["correctAnswerRate"],
            "mAR": s["modelAnswerRate"],
            "tier": rep.get("performanceTier"),
        }
    out = {}
    for mid, c in cells.items():
        # H1 first: if the arms disagree on output, no throughput number means anything.
        identical = (len({c[k]["outTok"] for k in c}) == 1
                     and len({c[k]["correct"] for k in c}) == 1
                     and len({c[k]["mAR"] for k in c}) == 1)
        res = {"identical": identical, "tiers": sorted({c[k]["tier"] for k in c})}
        for key in ("decode", "e2e95"):
            a = [c[k][key] for k in c if k[0] == "A"]
            b = [c[k][key] for k in c if k[0] == "B"]
            ma, mb = sum(a) / len(a), sum(b) / len(b)
            res[key] = {
                "sept": ma, "released": mb,
                "pct": (mb - ma) / ma * 100.0,
                "spread": max((max(a) - min(a)) / ma, (max(b) - min(b)) / mb) * 100.0,
            }
        out[mid] = res
    return out


def headroom():
    rows = []
    for path in sorted(glob.glob(os.path.join(RAW, "campaign-verdicts", "**", "*.verdict.json"),
                                 recursive=True)):
        q = json.load(open(path))["qualification"]
        for c in (q.get("comparisons") or []):
            dr, fl = c.get("decodeThroughputRatio"), c.get("minimumDecodeThroughputRatio")
            er, ec = c.get("endToEndLatencyRatio"), c.get("maximumEndToEndLatencyRatio")
            if None in (dr, fl, er, ec):
                continue
            rows.append({
                "modelId": q["modelId"], "verdict": q["verdict"],
                "decodeRatio": dr, "decodeFloor": fl,
                "decodeHeadroomPct": (dr - fl) / fl * 100.0,
                "e2eRatio": er, "e2eCeiling": ec,
                "e2eHeadroomPct": (ec - er) / ec * 100.0,
            })
    return rows


def main():
    eff = paired_effect()
    print("STAGE 1 -- effect of the kernel, paired on one box")
    worst_decode_against = 0.0
    worst_e2e_against = 0.0
    for mid, r in sorted(eff.items()):
        print(f"  {mid}")
        print(f"      outputs identical across all four runs: {r['identical']}"
              f"   performanceTier: {r['tiers']}")
        d, e = r["decode"], r["e2e95"]
        print(f"      p50 decode tok/s   {d['sept']:9.3f} -> {d['released']:9.3f}  "
              f"{d['pct']:+6.2f}%   own-arm spread {d['spread']:.2f}%")
        print(f"      endToEnd p95 ms    {e['sept']:9.1f} -> {e['released']:9.1f}  "
              f"{e['pct']:+6.2f}%   own-arm spread {e['spread']:.2f}%")
        worst_decode_against = min(worst_decode_against, d["pct"])
        worst_e2e_against = max(worst_e2e_against, e["pct"])
    if not all(r["identical"] for r in eff.values()):
        print()
        print("  OUTPUTS DIFFER BETWEEN KERNELS. Direction and magnitude are irrelevant: every")
        print("  verdict from the campaign is void and must be re-run. Stopping.")
        return 1

    print()
    print(f"  worst case against us: decode {worst_decode_against:+.2f}%, "
          f"endToEnd p95 {worst_e2e_against:+.2f}%")
    print()
    rows = headroom()
    qual = [r for r in rows if r["verdict"] == "QUALIFIED"]
    print(f"STAGE 2 -- headroom of every QUALIFIED verdict ({len(qual)} of {len(rows)} "
          f"comparison records)")
    at_risk = []
    for r in sorted(qual, key=lambda x: x["decodeHeadroomPct"]):
        flags = []
        if r["decodeHeadroomPct"] < abs(worst_decode_against):
            flags.append("DECODE")
        if r["e2eHeadroomPct"] < worst_e2e_against:
            flags.append("E2E")
        if flags:
            at_risk.append((r["modelId"], "+".join(flags)))
        print(f"  {r['modelId'][:46]:<46} decode {r['decodeRatio']:.3f} vs "
              f"{r['decodeFloor']:.2f} ({r['decodeHeadroomPct']:+5.1f}%)   "
              f"e2e {r['e2eRatio']:.3f} vs {r['e2eCeiling']:.2f} "
              f"({r['e2eHeadroomPct']:+5.1f}%){'  <-- ' + ','.join(flags) if flags else ''}")
    print()
    if at_risk:
        print("  AT RISK -- headroom smaller than the measured effect, must be re-run:")
        for mid, why in at_risk:
            print(f"    {mid}  ({why})")
        return 1
    tightest_d = min(r["decodeHeadroomPct"] for r in qual)
    tightest_e = min(r["e2eHeadroomPct"] for r in qual)
    print(f"  NONE AT RISK. Tightest headroom: decode {tightest_d:.1f}% "
          f"(effect {worst_decode_against:+.2f}%), endToEnd {tightest_e:.1f}% "
          f"(effect {worst_e2e_against:+.2f}%).")
    print("  No QUALIFIED verdict changes sign under the measured kernel effect, so the stale")
    print("  kernel does not require requalifying anything.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
