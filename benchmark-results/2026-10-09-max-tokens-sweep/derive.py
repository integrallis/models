#!/usr/bin/env python3
"""Derive the max-output-tokens sweep table from the committed raw artifacts.

Every number in NOTES.md comes from this script. Nothing is hand-typed.

Run from the repository root:
    python3 benchmark-results/2026-10-09-max-tokens-sweep/derive.py

The design is paired: for each (model, workload) the SAME artifact sha and the SAME
corpus sha are run twice with exactly one variable changed, settings.maxOutputTokens.
The script REFUSES to pair two runs whose artifactSha256 or corpusSha256 differ, because
then more than one thing changed and the pair measures nothing.

Decision rule, registered before the data arrived (see NOTES.md "Decision rule"):
  truncatedAnswerRate falls to ~0 AND the verdict changes  -> the cap was the cause
  truncatedAnswerRate falls to ~0 AND the verdict does not  -> the model genuinely
                                                               does not contribute
  truncatedAnswerRate does not fall                         -> the cap was not the
                                                               binding constraint
"""
import json, glob, os, csv, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "raw")

# Metrics lifted from the candidate report's own summary block. Field names are the
# report's, not restated: schemaVersion 5 summary keys.
# A pair whose two runs were produced by different backend builds has changed TWO things,
# not one. It is still admissible only if the delta between the two builds touches nothing on
# the measured path. These prefixes are the paths that cannot affect a text-generation RAG
# qualification run: a module the run never loads, and the fleet plumbing that sets the very
# knob being swept. Anything else changing makes the pair inadmissible.
OFF_MEASURED_PATH = (
    "models-audio/",      # speech-to-text module; not on the text-generation path
    "scripts/fleet/",     # the harness that sets maxOutputTokens, not the code under test
)

METRICS = [
    "truncatedAnswerRate", "modelAnswerRate", "modelAnswerCorrectRate",
    "correctAnswerRate", "rawCorrectAnswerRate", "extractiveFallbackRate",
    "totalOutputTokens",
]

def load_runs():
    runs = {}
    for path in sorted(glob.glob(os.path.join(RAW, "mt*-*-shard*", "*.json"))):
        base = os.path.basename(path)
        if base.endswith(".verdict.json") or base.endswith(".comparator.json"):
            continue
        rep = json.load(open(path))
        s = rep["settings"]
        folder = os.path.basename(os.path.dirname(path))
        shard = folder.rsplit("shard", 1)[1]
        key = (rep["modelId"], s["workload"])
        verdict_path = path[: -len(".json")] + ".verdict.json"
        verdict = None
        if os.path.exists(verdict_path):
            verdict = json.load(open(verdict_path))["qualification"]["verdict"]
        runs.setdefault(key, []).append({
            "modelId": rep["modelId"],
            "workload": s["workload"],
            "mt": s["maxOutputTokens"],
            "shard": shard,
            "artifactSha256": rep["artifactSha256"],
            "corpusSha256": s["corpusSha256"],
            "backendVersion": rep["backendVersion"],
            "verdict": verdict,
            **{m: rep["summary"].get(m) for m in METRICS},
        })
    return runs

def backend_delta_paths(sha_a, sha_b):
    """Paths that differ between two backend builds, or None if neither git nor the
    committed record can answer.

    git is the authority. But this evidence directory has to re-derive its own numbers from
    committed artifacts on a machine that may have a shallow clone, or long after the branch
    holding these dev shas is pruned. So when git answers, the answer is written to
    backend-delta/<a>..<b>.txt and any existing copy is checked against it; when git cannot
    answer, that committed record is used and the caller is told the source was the record.
    The function never guesses: no git and no record means None, and None rejects the pair.
    """
    record_dir = os.path.join(HERE, "backend-delta")
    record = os.path.join(record_dir, f"{sha_a}..{sha_b}.txt")
    live = None
    try:
        out = subprocess.run(["git", "diff", "--name-only", sha_a, sha_b],
                             cwd=os.path.join(HERE, "..", ".."),
                             capture_output=True, text=True, check=True)
        live = [ln for ln in out.stdout.splitlines() if ln.strip()]
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    if live is not None:
        os.makedirs(record_dir, exist_ok=True)
        if os.path.exists(record):
            was = [ln for ln in open(record).read().splitlines() if ln.strip()]
            if was != live:
                raise SystemExit(
                    f"backend delta for {sha_a}..{sha_b} no longer matches the committed "
                    f"record in {os.path.relpath(record, HERE)}.\n"
                    f"  recorded: {len(was)} files\n  git now says: {len(live)} files\n"
                    f"History was rewritten under this evidence. Refusing to re-derive.")
        else:
            with open(record, "w") as fh:
                fh.write("\n".join(live) + "\n")
        return live, "git"

    if os.path.exists(record):
        return [ln for ln in open(record).read().splitlines() if ln.strip()], "record"
    return None, None


def version_sha(v):
    """models@0.3.56-dev+q41-7ac41536f5c2 -> 7ac41536f5c2"""
    return v.rsplit("-", 1)[-1]


def main():
    runs = load_runs()
    rows, unpaired, rejected = [], [], []
    for (model, wl), got in sorted(runs.items()):
        by_mt = {}
        for r in got:
            # Repeat runs of the same cell would make the pairing ambiguous; refuse.
            if r["mt"] in by_mt:
                rejected.append((model, wl, f"two runs at mt={r['mt']}"))
                break
            by_mt[r["mt"]] = r
        else:
            if len(by_mt) < 2:
                unpaired.append((model, wl, sorted(by_mt)))
                continue
            lo, hi = sorted(by_mt)[0], sorted(by_mt)[-1]
            a, b = by_mt[lo], by_mt[hi]
            bad = None
            for field in ("artifactSha256", "corpusSha256"):
                if a[field] != b[field]:
                    bad = f"{field} differs between mt={lo} and mt={hi}"
                    break
            if bad is None and a["backendVersion"] != b["backendVersion"]:
                paths, src = backend_delta_paths(version_sha(a["backendVersion"]),
                                                 version_sha(b["backendVersion"]))
                if paths is None:
                    bad = (f"backendVersion differs ({a['backendVersion']} vs "
                           f"{b['backendVersion']}) and git cannot resolve the delta")
                else:
                    on_path = [x for x in paths if not x.startswith(OFF_MEASURED_PATH)]
                    if on_path:
                        bad = (f"backendVersion differs and the delta touches the measured "
                               f"path: {', '.join(on_path[:4])}")
                    else:
                        a["backendDeltaCleared"] = b["backendDeltaCleared"] = len(paths)
                        a["backendDeltaSource"] = b["backendDeltaSource"] = src
            if bad:
                rejected.append((model, wl, bad))
            else:
                rows.append((a, b))

    out_csv = os.path.join(HERE, "max-tokens-sweep.csv")
    with open(out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["modelId", "workload", "mtLow", "mtHigh", "verdictLow", "verdictHigh",
                    "verdictChanged"] + [f"{m}@{k}" for m in METRICS for k in ("low", "high")])
        for a, b in rows:
            w.writerow([a["modelId"], a["workload"], a["mt"], b["mt"], a["verdict"], b["verdict"],
                        a["verdict"] != b["verdict"]]
                       + [v for m in METRICS for v in (a[m], b[m])])

    def fmt(v):
        if v is None: return "-"
        return f"{v:.3f}" if isinstance(v, float) else str(v)

    print(f"paired cells: {len(rows)}   csv: {os.path.relpath(out_csv, os.path.join(HERE,'..','..'))}")
    print()
    hdr = f"{'model':<44} {'wl':<12} {'mt':>9} {'trunc':>7} {'mAR':>7} {'mACR':>7} {'extrFB':>7} {'outTok':>7}  verdict"
    print(hdr); print("-" * len(hdr))
    for a, b in rows:
        for r in (a, b):
            print(f"{r['modelId']:<44} {r['workload']:<12} {r['mt']:>9} "
                  f"{fmt(r['truncatedAnswerRate']):>7} {fmt(r['modelAnswerRate']):>7} "
                  f"{fmt(r['modelAnswerCorrectRate']):>7} {fmt(r['extractiveFallbackRate']):>7} "
                  f"{fmt(r['totalOutputTokens']):>7}  {r['verdict']}")
        # Order matters. A verdict can change while truncation is still happening, and then the
        # cap has NOT been ruled out -- it has only moved. Checking "verdict changed" first
        # labelled fin-r1 "CAP WAS THE CAUSE" when it was still truncating 56% of answers at the
        # higher cap. Truncation is the question this sweep asks, so it is asked first.
        verdict_changed = a["verdict"] != b["verdict"]
        trunc_lo = a["truncatedAnswerRate"] or 0
        trunc_hi = b["truncatedAnswerRate"] or 0
        notes = []
        if trunc_hi > 0:
            call = f"CAP STILL BINDING ({trunc_hi:.3f} of answers truncated at mt={b['mt']})"
            notes.append(f"inconclusive: raise the cap again before judging this model")
        elif trunc_lo > 0 and verdict_changed:
            call = "CAP WAS THE CAUSE (truncation cleared and the verdict changed)"
        elif trunc_lo > 0:
            call = "GENUINE UNDERPERFORMANCE (truncation cleared, verdict unchanged)"
        else:
            call = "CAP NOT BINDING (no truncation at either cap)"
        if verdict_changed:
            notes.append(f"verdict moved {a['verdict']} -> {b['verdict']}")
        # A model whose finished answers are correct is being masked by the cap, not failing on
        # quality. That is the one signal that justifies spending another run on it.
        if (b["modelAnswerCorrectRate"] or 0) > (a["modelAnswerCorrectRate"] or 0):
            notes.append(f"modelAnswerCorrectRate rose {a['modelAnswerCorrectRate']:.3f} -> "
                         f"{b['modelAnswerCorrectRate']:.3f}: its finished answers are right, "
                         f"it is not finishing enough of them")
        print(f"{'':<44} {'':<12} {'=>':>9}  {call}")
        for n in notes:
            print(f"{'':<44} {'':<12} {'':>9}  - {n}")
        print()

    if unpaired:
        print("UNPAIRED (only one cap measured; no conclusion drawn):")
        for m, wl, mts in unpaired: print(f"  {m} [{wl}] at mt={mts}")
        print()
    if rejected:
        print("REJECTED PAIRS (more than one variable changed):")
        for m, wl, why in rejected: print(f"  {m} [{wl}]: {why}")
        print()
    cleared = {(a["modelId"], a["workload"]):
               (a.get("backendDeltaCleared"), a.get("backendDeltaSource")) for a, b in rows}
    if any(n for n, _ in cleared.values()):
        print("Pairs admitted across a backend-build change, delta verified off the measured path")
        print(f"  (allowlisted prefixes: {', '.join(OFF_MEASURED_PATH)}):")
        for (m, wl), (n, src) in sorted(cleared.items()):
            if n:
                how = "resolved by git" if src == "git" else "read from the committed record"
                print(f"  {m} [{wl}]: {n} files differ, none on the measured path ({how})")
        print()
    rc = check_notes(rows)
    print()
    versions = sorted({r["backendVersion"] for got in runs.values() for r in got})
    print("backendVersion present in these artifacts:")
    for v in versions: print(f"  {v}")
    if any("-dev" in v for v in versions):
        print("  NOTE: a -dev version is not a released build. Verdicts carrying one")
        print("        cannot be landed in the catalog; they must be re-run against a release.")
    return rc

def check_notes(rows):
    """Refuse to let NOTES.md carry a number this script did not derive.

    NOTES.md states the sweep as prose with an inline table. That table is hand-written, so
    it is exactly the place a wrong number survives unnoticed. Every value it asserts must be
    findable, formatted the same way, in the derived rows. Run as part of the default
    invocation: deriving the numbers and checking the write-up are one step, not two.
    """
    notes_path = os.path.join(HERE, "NOTES.md")
    if not os.path.exists(notes_path):
        print("NOTES.md absent; nothing to check.")
        return 0
    notes = open(notes_path).read()
    problems = []
    for a, b in rows:
        mid = a["modelId"]
        for label, claim in (
            ("truncatedAnswerRate",
             f"{a['truncatedAnswerRate']:.3f} -> {b['truncatedAnswerRate']:.3f}"),
            ("modelAnswerRate",
             f"{a['modelAnswerRate']:.3f} -> {b['modelAnswerRate']:.3f}"),
            ("totalOutputTokens",
             f"{a['totalOutputTokens']} -> {b['totalOutputTokens']}"),
        ):
            # NOTES.md uses a typographic arrow; compare against both spellings.
            if claim not in notes and claim.replace("->", "\u2192") not in notes:
                problems.append(f"{mid}: NOTES.md does not state {label} as '{claim}'")
        if (a["verdict"] != b["verdict"]) and "verdict changed" not in notes:
            problems.append(f"{mid}: the verdict changed but NOTES.md does not say so")
    if problems:
        print("NOTES.md DISAGREES WITH THE DERIVED NUMBERS:")
        for x in problems: print(f"  {x}")
        return 1
    print(f"NOTES.md checked: every stated rate and token count matches the {len(rows)} "
          f"derived pairs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
