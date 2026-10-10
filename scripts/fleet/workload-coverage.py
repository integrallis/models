#!/usr/bin/env python3
"""Report which RAG workloads have qualified models, and which cannot yet be served.

The workload list is parsed out of ``RagWorkload.java`` rather than written here, because a
hard-coded copy would drift from the enum and a coverage report that silently omits a workload is
worse than no report. Coverage comes from the ModelJars catalog's own qualification manifest.

It also separates two states that look identical in a count of zero and are not the same thing:

* **no corpus** -- the workload's documents or cases resource is missing or empty, so no model
  *could* qualify there. That is "no data".
* **no qualified model** -- the corpus exists and has cases, but nothing has passed the gate on it
  yet. That is a gap a campaign can close.

Conflating them is the mistake the house rules name directly: a flat line from a dataset that
cannot exercise a feature is "no data", not "no effect".

Usage:
    scripts/fleet/workload-coverage.py --catalog ../model-jars/catalog [--fail-on-zero]
"""

import argparse
import collections
import json
import pathlib
import re
import sys

ENUM = "models-rag-bench/src/main/java/com/integrallis/models/rag/RagWorkload.java"
RESOURCES = "models-rag-bench/src/main/resources"

# GENERAL("general", "/rag/documents.json", "/rag/cases.json"),
ENTRY = re.compile(
    r'^\s*[A-Z_]+\(\s*$|^\s*[A-Z_]+\(\s*"(?P<id1>[a-z]+)"\s*,\s*"(?P<docs1>[^"]+)"\s*,'
    r'\s*"(?P<cases1>[^"]+)"', re.MULTILINE)


def parse_workloads(repo_root):
    """Workload id -> (documents resource, cases resource), straight from the enum.

    The enum wraps long constants across lines, so the whole declaration block is flattened before
    matching. Anything that does not yield all three fields is reported rather than skipped: a
    workload this parser cannot read is a hole in the report, and a silent hole is the failure mode
    this function exists to avoid.
    """
    text = (pathlib.Path(repo_root) / ENUM).read_text()
    body = text.split("public enum RagWorkload {", 1)[1].split(";", 1)[0]
    # Drop comments, then flatten so a constant split across lines matches as one.
    body = re.sub(r"//[^\n]*", "", body)
    flat = " ".join(body.split())
    found, names = {}, []
    for m in re.finditer(r'([A-Z][A-Z_]*)\s*\(\s*"([a-z]+)"\s*,\s*"([^"]+)"\s*,\s*"([^"]+)"\s*\)',
                         flat):
        names.append(m.group(1))
        found[m.group(2)] = (m.group(3), m.group(4))
    declared = re.findall(r'(?:^|,)\s*([A-Z][A-Z_]*)\s*\(', flat)
    unreadable = [n for n in declared if n not in names]
    return found, unreadable


def corpus_state(repo_root, docs_resource, cases_resource):
    base = pathlib.Path(repo_root) / RESOURCES
    out = {}
    for label, resource in (("documents", docs_resource), ("cases", cases_resource)):
        path = base / resource.lstrip("/")
        if not path.exists():
            out[label] = None
            continue
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            out[label] = None
            continue
        out[label] = len(data) if isinstance(data, list) else None
    return out


def check_doc(repo_root, doc_path, table, workloads):
    """Refuse to let the coverage doc carry a claim this tool did not derive.

    docs/WORKLOAD-COVERAGE.md pastes the table above and quotes the corpora directly, so it is
    exactly where a stale number survives unnoticed. Markdown blockquote markers are stripped
    before comparing, because a quoted case question legitimately wraps across lines and `> ` in
    the middle of a sentence is formatting, not content.
    """
    doc = pathlib.Path(doc_path)
    if not doc.exists():
        return [f"{doc_path} does not exist"]
    text = doc.read_text()
    flat = " ".join(re.sub(r"(?m)^\s*>\s?", "", text).split())
    problems = []
    if table not in text:
        problems.append("the pasted coverage table is not the one this tool now produces")
    for wl, (_, cases_resource) in sorted(workloads.items()):
        path = pathlib.Path(repo_root) / RESOURCES / cases_resource.lstrip("/")
        if not path.exists():
            continue
        try:
            cases = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if not isinstance(cases, list) or not cases:
            continue
        answerable = [c for c in cases if c.get("answerable")]
        unanswerable = [c for c in cases if not c.get("answerable")]
        # Only workloads the doc actually discusses in prose are checked for quoted text.
        if f"`{wl}` does **not**" in text or f"`{wl}` asks for" in text:
            q = " ".join(cases[0].get("question", "").split())
            if q and q not in flat:
                problems.append(f"{wl}: the quoted case question is not the corpus's")
        if f"**{len(answerable)} answerable plus {len(unanswerable)}" in text:
            counts = {len(c.get("requiredFacts", [])) for c in answerable}
            if len(counts) != 1:
                problems.append(f"{wl}: answerable cases carry {sorted(counts)} required facts, "
                                f"so a single claimed count would be wrong")
            if any(c.get("requiredFacts") for c in unanswerable):
                problems.append(f"{wl}: an unanswerable case carries required facts, which the "
                                f"doc says it does not")
    return problems


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--catalog", required=True, help="path to the ModelJars catalog/ directory")
    ap.add_argument("--repo-root", default=str(pathlib.Path(__file__).resolve().parents[2]))
    ap.add_argument("--fail-on-zero", action="store_true",
                    help="exit 1 if a workload with a usable corpus has no qualified model")
    ap.add_argument("--check-doc", metavar="PATH",
                    help="verify that this document's pasted table and quoted corpus text still "
                         "match what is derived here; exit 1 if not")
    args = ap.parse_args(argv)

    workloads, unreadable = parse_workloads(args.repo_root)
    if unreadable:
        print(f"WARNING: {len(unreadable)} enum constant(s) could not be parsed and are missing "
              f"from this report: {', '.join(unreadable)}")
        print()

    catalog = pathlib.Path(args.catalog)
    raw = json.loads((catalog / "models.json").read_bytes())
    models = raw["models"] if isinstance(raw, dict) and "models" in raw else raw
    idx = {m["id"]: m for m in models}
    entries = json.loads((catalog / "qualifications.json").read_bytes())["entries"]

    # `entries` holds BOTH outcomes: the manifest header counts qualifiedModels and
    # rejectedModels separately and `entries` is their sum. Counting every row as coverage
    # inflated general from 51 to 52 and the catalog's qualified total from 101 to 102, because
    # h2o-danube3-500m sits there with verdict FAILED_MODEL_CONTRIBUTION_GATE. A rejection is
    # evidence, but it is not coverage, so it is counted and reported separately.
    qualified = [e for e in entries if e.get("qualified") is True]
    rejected = [e for e in entries if e.get("qualified") is not True]

    by_wl = collections.defaultdict(list)
    for e in qualified:
        by_wl[e["workload"]].append(e)

    unknown = sorted(set(by_wl) - set(workloads))
    rows, zero_with_corpus, no_corpus = [], [], []
    for wl in sorted(workloads):
        docs, cases = workloads[wl]
        state = corpus_state(args.repo_root, docs, cases)
        got = by_wl.get(wl, [])
        usable = bool(state["cases"])
        if not usable:
            no_corpus.append(wl)
        elif not got:
            zero_with_corpus.append(wl)
        rows.append((wl, len(got), state, got))

    width = max(len(w) for w in workloads)
    table_lines = [f"{'workload':<{width}}  {'qualified':>9}  {'docs':>5}  {'cases':>5}  state",
                   "-" * (width + 40)]
    print(table_lines[0]); print(table_lines[1])
    for wl, n, state, got in rows:
        d = state["documents"] if state["documents"] is not None else "-"
        c = state["cases"] if state["cases"] is not None else "-"
        if not state["cases"]:
            note = "NO CORPUS (no model could qualify here)"
        elif n == 0:
            note = "NO QUALIFIED MODEL (corpus is usable; this is a closable gap)"
        elif n == 1:
            note = "thin: one model"
        else:
            note = ""
        line = f"{wl:<{width}}  {n:>9}  {d:>5}  {c:>5}  {note}"
        table_lines.append(line)
        print(line)

    print()
    for wl, n, state, got in rows:
        if not got:
            continue
        sizes = [round((idx.get(e["modelId"], {}).get("sizeBytes") or 0) / 1e9, 2) for e in got]
        print(f"{wl}:")
        for e, gb in sorted(zip(got, sizes), key=lambda x: x[1]):
            print(f"    {gb:>5} GB  {e['modelId']:<46} tpl={e.get('promptTemplate')}")

    if rejected:
        print()
        print(f"rejected, and therefore NOT counted as coverage ({len(rejected)}):")
        for e in rejected:
            print(f"    {e['modelId']:<46} [{e['workload']}]  {e.get('verdict', 'no verdict')}")

    if unknown:
        print()
        print(f"qualifications naming a workload the enum does not declare: {', '.join(unknown)}")

    print()
    if no_corpus:
        print(f"no corpus ({len(no_corpus)}): {', '.join(no_corpus)}")
        print("   these are 'no data', not 'no effect' -- a corpus has to exist first.")
    if zero_with_corpus:
        print(f"no qualified model but a usable corpus ({len(zero_with_corpus)}): "
              f"{', '.join(zero_with_corpus)}")
        print("   a campaign can close these today; nothing is missing but a passing verdict.")
    if not no_corpus and not zero_with_corpus:
        print("every declared workload with a corpus has at least one qualified model.")

    rc = 0
    if args.check_doc:
        print()
        problems = check_doc(args.repo_root, args.check_doc,
                             "\n".join(table_lines), workloads)
        if problems:
            print(f"{args.check_doc} DISAGREES WITH THE DERIVED NUMBERS:")
            for x in problems:
                print(f"  {x}")
            rc = 1
        else:
            print(f"{args.check_doc} checked: its table and quoted corpus text match this run.")

    if args.fail_on_zero and zero_with_corpus:
        rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
