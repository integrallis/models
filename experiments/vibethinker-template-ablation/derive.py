#!/usr/bin/env python3
"""Derive the cap x template grid from the raw candidate reports. Nothing is hand-typed.

Fail-closed on admissibility. Two axes vary by design -- `maxOutputTokens` and `promptTemplate` --
and nothing else may: not the artifact, the corpus, the cases, the context, the threads, the
grounding policy, the retrieval top-k, the minimum retrieval score or the generation controls. A
grid that differs on anything else is rejected rather than reported, because a difference would
then be attributable to an uncontrolled variable.

Shard 703 is printed as a REFERENCE row, never as a control: it ran `threads = 16` against this
experiment's 8, and `iterations = 3` against 1. Thread count changes floating-point reduction
order, which under greedy decoding can change the tokens chosen. The control for the template
question is this experiment's own `chatml-direct` cell at the same cap.
"""
import json, pathlib, re, sys

HERE = pathlib.Path(__file__).parent
EVIDENCE = HERE / "evidence"
REFERENCE = (HERE / "../../benchmark-results/2026-10-21-candidate-qualification/raw/shard-703"
             / "vibethinker_1_5b_q4_k_m.json").resolve()

# Everything that must be identical in every cell. The two deliberately varied axes are absent.
HELD = ["corpusSha256", "workload", "caseIds", "retrievalTopK", "warmups", "iterations",
        "contextLength", "threads", "groundingPolicy", "minimumRetrievalScore",
        "generationControls"]
TEMPLATES = ["chatml-direct", "chatml-answer", "chatml-no-think"]
FLOOR_RATE, FLOOR_CORRECT = 1 / 3, 0.90
TRUNCATION_CLEAR = 0.05   # "~0" in the registered decision rule


def load(path):
    d = json.loads(path.read_text())
    return {"id": d["modelId"], "tpl": d["settings"]["promptTemplate"],
            "cap": d["settings"]["maxOutputTokens"], "artifact": d["artifactSha256"],
            "settings": d["settings"], "s": d["summary"],
            "failures": d.get("failures") or [],
            "backend": d.get("backend"), "backendVersion": d.get("backendVersion")}


def check_results_md(by, caps, templates):
    """Every figure quoted in RESULTS.md must match the derived grid, or this exits 1.

    The 2026-10-09 max-tokens sweep established this: prose drifts from data silently, and a
    results document that cannot be checked against its own artifacts is an assertion rather than
    a report.
    """
    doc = HERE / "RESULTS.md"
    if not doc.exists():
        return
    text = doc.read_text()
    problems = []

    # The combined grid: "truncated / **modelAnswerRate** / correct" per cell.
    for cap in caps:
        row = re.search(rf"^\| \*\*{cap}\*\* \|(.+)\|\s*$", text, re.M)
        if not row:
            problems.append(f"RESULTS.md has no combined-grid row for cap {cap}")
            continue
        quoted = [c.strip() for c in row.group(1).split("|")]
        if len(quoted) != len(templates):
            problems.append(f"cap {cap} row has {len(quoted)} cells, expected {len(templates)}")
            continue
        for tpl, cell in zip(templates, quoted):
            nums = re.findall(r"\d\.\d{3}", cell)
            if len(nums) != 3:
                problems.append(f"cap {cap} / {tpl}: expected 3 rates, found {nums}")
                continue
            c = by[(cap, tpl)]["s"]
            want = [c["truncatedAnswerRate"], c["modelAnswerRate"], c["modelAnswerCorrectRate"]]
            for got, w, name in zip(nums, want, ("truncated", "modelAnswerRate", "correct")):
                if abs(float(got) - w) > 5e-4:
                    problems.append(f"cap {cap} / {tpl} / {name}: RESULTS.md says {got}, "
                                    f"derived {w:.3f}")

    # The token table.
    for cap in caps:
        rows = re.findall(rf"^\| \*\*{cap}\*\* \|([^|]*\|[^|]*\|[^|]*)\|\s*$", text, re.M)
        for raw in rows:
            cells = [c.strip().replace("**", "") for c in raw.split("|")]
            if not all(re.fullmatch(r"\d+", c) for c in cells):
                continue        # not the token row
            for tpl, got in zip(templates, cells):
                w = by[(cap, tpl)]["s"]["totalOutputTokens"]
                if int(got) != w:
                    problems.append(f"cap {cap} / {tpl} / totalOutputTokens: RESULTS.md says "
                                    f"{got}, derived {w}")

    if problems:
        print("\nRESULTS.md DISAGREES WITH THE ARTIFACTS:")
        for x in problems:
            print("  " + x)
        sys.exit(1)
    print("RESULTS.md agrees with every derived figure it quotes.")


def main():
    cells = [load(p) for p in sorted(EVIDENCE.glob("*.json"))]
    if not cells:
        sys.exit(f"no reports in {EVIDENCE}; run run_on_isolated_host.sh first")

    # A cell that did not run every case is not a measurement of that cell. Its rates are computed
    # over the attempts that survived, and when the failures correlate with the thing being
    # measured -- here, long generations overrunning the context -- the survivors are a biased
    # subset and the rate reads better than the truth. Such cells are reported and excluded, never
    # averaged in. Found by a cap of 2048 against a context of 2048: generation ran past the KV
    # window and raised "position out of range: 2048".
    broken = [c for c in cells
              if c["s"]["successfulAttempts"] != c["s"]["totalAttempts"]]
    for c in broken:
        errs = {f.get("error", "?").split(";")[-1].strip() for f in c["failures"]}
        print(f"EXCLUDED {c['tpl']} @ cap {c['cap']}: "
              f"{c['s']['successfulAttempts']}/{c['s']['totalAttempts']} attempts succeeded")
        for e in sorted(errs):
            print(f"         {e}")
    if broken:
        print("  These cells are excluded: their rates are over a biased subset of cases.\n")
    cells = [c for c in cells if c not in broken]
    if not cells:
        sys.exit("every cell was excluded; nothing admissible to report")

    base = cells[0]
    for c in cells[1:]:
        if c["artifact"] != base["artifact"]:
            sys.exit(f"INADMISSIBLE: {c['id']} artifact {c['artifact'][:12]} "
                     f"!= {base['artifact'][:12]}")
        for k in HELD:
            if c["settings"][k] != base["settings"][k]:
                sys.exit(f"INADMISSIBLE: {c['id']} differs on held variable {k}: "
                         f"{c['settings'][k]!r} != {base['settings'][k]!r}")
    seen = [(c["cap"], c["tpl"]) for c in cells]
    if len(set(seen)) != len(seen):
        sys.exit(f"INADMISSIBLE: duplicate (cap, template) cells: {seen}")

    caps = sorted({c["cap"] for c in cells})
    print(f"artifact sha256  : {base['artifact']}")
    print(f"corpus sha256    : {base['settings']['corpusSha256']}")
    print(f"backend          : {base['backend']} {base['backendVersion']}")
    print(f"held             : threads={base['settings']['threads']}, "
          f"context={base['settings']['contextLength']}, "
          f"cases={len(base['settings']['caseIds'])}, "
          f"warmups={base['settings']['warmups']}, "
          f"iterations={base['settings']['iterations']}")
    print(f"grounding policy : {base['settings']['groundingPolicy']}")
    print(f"grid             : {len(cells)} cells, caps {caps}")
    print()

    by = {(c["cap"], c["tpl"]): c for c in cells}
    templates = [t for t in TEMPLATES if any(c["tpl"] == t for c in cells)]

    def grid(metric, fmt="{:.3f}"):
        print(f"### {metric}\n")
        print("| cap | " + " | ".join(f"`{t}`" for t in templates) + " |")
        print("| --- " * (len(templates) + 1) + "|")
        for cap in caps:
            row = [fmt.format(by[(cap, t)]["s"][metric]) if (cap, t) in by else "-"
                   for t in templates]
            print(f"| **{cap}** | " + " | ".join(row) + " |")
        print()

    grid("truncatedAnswerRate")
    grid("modelAnswerRate")
    grid("modelAnswerCorrectRate")
    grid("extractiveFallbackRate")
    grid("correctAnswerRate")
    grid("totalOutputTokens", "{:.0f}")

    if REFERENCE.exists():
        r = load(REFERENCE)
        s = r["s"]
        print(f"_reference_ shard 703, `{r['tpl']}` at cap {r['cap']}, "
              f"threads={r['settings']['threads']}, iterations={r['settings']['iterations']}: "
              f"truncated {s['truncatedAnswerRate']:.3f}, "
              f"modelAnswerRate {s['modelAnswerRate']:.3f}, "
              f"outputTokens {s['totalOutputTokens']}\n")

    print("## Registered decision rule applied\n")
    print(f"gate: modelAnswerRate >= {FLOOR_RATE:.3f} AND "
          f"modelAnswerCorrectRate >= {FLOOR_CORRECT:.2f}")
    print(f"truncation counts as cleared below {TRUNCATION_CLEAR:.2f}\n")

    top = max(caps)
    truncating_at_top = [c for c in cells
                         if c["cap"] == top and c["s"]["truncatedAnswerRate"] >= TRUNCATION_CLEAR]
    if len(truncating_at_top) == len([c for c in cells if c["cap"] == top]):
        print(f"REGISTERED RULE: cap still binding at {top} in every cell -- INCONCLUSIVE.")
        print("Truncation is asked first, so the aggregate rates cannot judge a template here.")
        print(f"{top} is the highest cap this harness can run: the worker pins --context 2048 and")
        print("the rendered prompts are 156-172 tokens, so the rule has no higher cap to ask for.")
        print("See RESULTS.md: the per-case replay is a declared post-hoc refinement, not this rule.")
        check_results_md(by, caps, templates)
        return

    for c in truncating_at_top:
        print(f"note: `{c['tpl']}` at cap {top} still truncates "
              f"({c['s']['truncatedAnswerRate']:.3f}); that cell is inconclusive.")

    passing = [c for c in cells
               if c["s"]["modelAnswerRate"] >= FLOOR_RATE
               and c["s"]["modelAnswerCorrectRate"] >= FLOOR_CORRECT
               and c["s"]["truncatedAnswerRate"] < TRUNCATION_CLEAR]
    if passing:
        # Lowest cap that clears, then the higher correct rate; chatml-answer breaks a tie.
        order = {"chatml-answer": 0, "chatml-no-think": 1, "chatml-direct": 2}
        winner = min(passing, key=lambda c: (c["cap"], -c["s"]["modelAnswerCorrectRate"],
                                             order.get(c["tpl"], 9)))
        print("\nCLEARS THE CONTRIBUTION GATE:")
        for c in passing:
            print(f"  {c['tpl']} @ {c['cap']}: rate {c['s']['modelAnswerRate']:.3f}, "
                  f"correct {c['s']['modelAnswerCorrectRate']:.3f}")
        ctrl = by.get((winner["cap"], "chatml-direct"))
        if ctrl is not None and ctrl is not winner:
            print(f"  control at the same cap (`chatml-direct` @ {winner['cap']}): "
                  f"rate {ctrl['s']['modelAnswerRate']:.3f} -- "
                  + ("the template is the cause" if ctrl["s"]["modelAnswerRate"] < FLOOR_RATE
                     else "the control clears it too, so the cap is the cause"))
        print(f"\nSELECTED: {winner['tpl']} at cap {winner['cap']}")
        print("Next: shard 911 on the fleet, two-arm, this template and cap, canonical model id.")
    else:
        print("\nNO CELL CLEARS THE GATE with truncation cleared.")
        print("Per the registered kill criterion this is the result: VibeThinker 1.5B is not")
        print("qualified on `math`. No fourth template, no stop sequence, no cap beyond 1792.")
    print()
    check_results_md(by, caps, templates)


if __name__ == "__main__":
    main()
