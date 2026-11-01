#!/usr/bin/env python3
"""Apply the frozen gates to a derive.py report. Nothing hand-typed: every number is read from
report.json, every threshold is the README's.

usage: verdict.py REPORT_JSON [--timing-is-gate] [--sanity SANITY_REPORT_JSON] > verdict.json

Without --timing-is-gate the throughput ratio is reported but marked "not measured as a gate"
(the dev-box passes ran under other sessions' build load). The intelligibility gate, the kill
criteria and completion are decided from the report either way.
"""
import json, sys

WER_MARGIN = 0.005          # 0.5 points absolute over the comparator's worst pass
COMPARATOR_SPREAD_KILL = 0.010  # comparator passes must agree within 1.0 point
CARD_WER, CARD_TOLERANCE = 0.058, 0.020  # af_heart on wikitext103-heldout-200 vs the card's 5.8%
THROUGHPUT_RATIO_FLOOR = 0.80
PASSES = 3


def main(argv):
    report_path = argv[1]
    timing_is_gate = "--timing-is-gate" in argv
    sanity = None
    if "--sanity" in argv:
        sanity = json.load(open(argv[argv.index("--sanity") + 1]))
    report = json.load(open(report_path))
    by_key = {(s["arm"], s["corpus"], s["voice"], s["threads"]): s for s in report["summary"]}
    corpora = sorted({s["corpus"] for s in report["summary"]})
    voices = sorted({s["voice"] for s in report["summary"]})
    out = {"report": report_path, "timingIsGate": timing_is_gate, "thresholds": {
        "werMarginAbsolute": WER_MARGIN, "comparatorSpreadKill": COMPARATOR_SPREAD_KILL,
        "cardWer": CARD_WER, "cardTolerance": CARD_TOLERANCE, "throughputRatioFloor": THROUGHPUT_RATIO_FLOOR,
        "passesPerArm": PASSES}, "cells": [], "kill": [], "incomplete": []}
    for corpus in corpora:
        for voice in voices:
            comp = by_key.get(("comparator", corpus, voice, 1))
            if comp is None:
                continue
            cell = {"corpus": corpus, "voice": voice,
                    "comparator": {"passes": comp["passes"], "werMin": comp["werMin"], "werMax": comp["werMax"],
                                   "realTimeMultipleMin": comp["realTimeMultipleMin"], "realTimeMultipleMax": comp["realTimeMultipleMax"]}}
            complete = comp["passes"] >= PASSES and comp["werMax"] is not None
            if comp["werMax"] is not None and comp["werMin"] is not None:
                spread = comp["werMax"] - comp["werMin"]
                cell["comparatorSpread"] = spread
                if spread > COMPARATOR_SPREAD_KILL:
                    out["kill"].append(f"{corpus}/{voice}: comparator WER spread {spread*100:.2f} pt > {COMPARATOR_SPREAD_KILL*100:.1f}; the floor cannot be supported")
                if corpus == "wikitext103-heldout-200" and voice == "af_heart":
                    if comp["werMin"] > CARD_WER + CARD_TOLERANCE or comp["werMax"] < CARD_WER - CARD_TOLERANCE:
                        out["kill"].append(f"anchor: comparator af_heart WER {comp['werMin']*100:.2f}..{comp['werMax']*100:.2f}% is more than {CARD_TOLERANCE*100:.0f} pt from the card's {CARD_WER*100:.1f}%")
                floor = comp["werMax"] + WER_MARGIN
                cell["werFloor"] = floor
            else:
                floor = None
            for arm, label in (("candidate", "ourPhonemes"), ("candidate-publisher-phonemes", "publisherPhonemes")):
                cand = by_key.get((arm, corpus, voice, 1))
                if cand is None:
                    continue
                entry = {"passes": cand["passes"], "werMin": cand["werMin"], "werMax": cand["werMax"],
                         "realTimeMultipleMin": cand["realTimeMultipleMin"], "realTimeMultipleMax": cand["realTimeMultipleMax"]}
                if floor is not None and cand["werMax"] is not None and cand["passes"] >= PASSES:
                    entry["intelligibility"] = "PASS" if cand["werMax"] <= floor else "FAIL"
                    entry["worstPassMinusFloor"] = cand["werMax"] - floor
                else:
                    entry["intelligibility"] = "INCOMPLETE"
                    complete = False
                if cand["realTimeMultipleMin"] and comp["realTimeMultipleMax"]:
                    ratio = cand["realTimeMultipleMin"] / comp["realTimeMultipleMax"]
                    entry["throughputRatioMinOverComparatorBest"] = ratio
                    entry["throughput"] = ("PASS" if ratio >= THROUGHPUT_RATIO_FLOOR else "FAIL") if timing_is_gate else "NOT A GATE ON THIS MACHINE"
                cell[label] = entry
            if not complete:
                out["incomplete"].append(f"{corpus}/{voice}")
            out["cells"].append(cell)
    # Completion rule: every pass finite.
    nonfinite = [g["path"] for g in report["groups"] if not g["allFinite"]]
    out["nonFinitePasses"] = nonfinite
    if nonfinite:
        out["kill"].append(f"non-finite PCM in {len(nonfinite)} pass(es)")
    out["phonemizer"] = report.get("phonemizer", [])
    if sanity:
        out["judgeSanity"] = [{"arm": g["arm"], "corpus": g["corpus"], "werPublisherNormalizer": g["werPublisherNormalizer"],
                               "werWhisperNormalizer": g["werWhisperNormalizer"], "sentences": g["sentences"]} for g in sanity["groups"]]
    # Human-readable table on stderr; JSON on stdout.
    print(f"{'corpus':26s} {'voice':12s} {'comp WER min..max':20s} {'ours max':>9s} {'gate':>6s} {'pub max':>9s} {'gate':>6s} {'x ours/comp':>12s}", file=sys.stderr)
    for c in out["cells"]:
        comp = c["comparator"]; ours = c.get("ourPhonemes", {}); pub = c.get("publisherPhonemes", {})
        f = lambda v: "-" if v is None else f"{v*100:.2f}"
        print(f"{c['corpus']:26s} {c['voice']:12s} {f(comp['werMin'])+'..'+f(comp['werMax']):20s} {f(ours.get('werMax')):>9s} {ours.get('intelligibility','-')[:6]:>6s} {f(pub.get('werMax')):>9s} {pub.get('intelligibility','-')[:6]:>6s} {ours.get('throughputRatioMinOverComparatorBest', float('nan')):>12.3f}", file=sys.stderr)
    for k in out["kill"]:
        print("KILL:", k, file=sys.stderr)
    for k in out["incomplete"]:
        print("INCOMPLETE:", k, file=sys.stderr)
    json.dump(out, sys.stdout, indent=1); print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
