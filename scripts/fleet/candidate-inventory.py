#!/usr/bin/env python3
"""Classify every catalog candidate: qualified, rejected, evaluated-not-landed, or never evaluated.

The catalog carries 457 candidates and 101 qualifications. Until this script existed, the other
356 were undocumented as a set -- there was no way to answer "what has been tried and failed" as
distinct from "what has never been looked at", and those are different facts with different
consequences. Publishing nulls is a house rule, and a null nobody can enumerate is not published.

Four buckets, in descending order of evidence:

* QUALIFIED            -- a manifest row with `qualified` true.
* REJECTED             -- a manifest row with `qualified` false. The verdict is the finding.
* EVALUATED_NOT_LANDED -- committed evidence under benchmark-results/ names it, but no manifest row
                          claims it. Either the run failed and was never recorded, or it passed on
                          an unreleased build and is waiting for a release.
* NOT_EVALUATED        -- no manifest row or matching committed JSON artifact found. Aliases,
                          non-JSON notes and external evidence are outside this inventory.

    python3 scripts/fleet/candidate-inventory.py --catalog ../model-jars/catalog \
        [--check-doc docs/CANDIDATE-INVENTORY.md] [--format md]

Evidence is matched by model id against committed report filenames and their `modelId` fields, so
a candidate counts as evaluated only when an artifact in this repository names it.
"""
import argparse
import collections
import io
import json
import pathlib
import subprocess
import sys

MANIFESTS = ("qualifications", "embedding-qualifications", "tool-qualifications",
             "reranking-qualifications", "speech-qualifications", "component-qualifications")


def read(path):
    return json.loads(pathlib.Path(path).read_bytes())


def manifest_rows(catalog_dir):
    """modelId -> (bucket, manifest, verdict). Both outcomes, kept apart."""
    out = {}
    for name in MANIFESTS:
        p = pathlib.Path(catalog_dir) / f"{name}.json"
        if not p.exists():
            continue
        j = read(p)
        for key in ("entries", "qualifiedModels", "rejectedModels"):
            rows = j.get(key)
            if not isinstance(rows, list):
                continue
            for e in rows:
                if not isinstance(e, dict):
                    continue
                mid = e.get("modelId") or e.get("id")
                if not mid:
                    continue
                ok = (e.get("qualified") is True
                      or (e.get("summary") or {}).get("qualified") is True)
                verdict = e.get("verdict") or e.get("reason") or "no verdict recorded"
                prior = out.get(mid)
                # A model can appear in several manifests. Qualified anywhere wins, because the
                # catalog's claim is that it qualified for something.
                if prior and prior[0] == "QUALIFIED":
                    continue
                out[mid] = ("QUALIFIED" if ok else "REJECTED", name, verdict)
    return out


def evidence_ids(repo_root, candidate_ids):
    """Match complete IDs in valid JSON at HEAD, never uncommitted or filename substrings.

    Payload modelId fields take precedence. Older oracle reports lack that field, so their exact
    filename stem is accepted. Only underscore/hyphen spelling is normalized; aliases and model
    display names are deliberately not guessed. A matching artifact is evidence to inspect, not
    proof of a successful run, a paired qualification, or a released runtime.
    """
    found, where = set(), collections.defaultdict(set)
    by_slug = {}
    for mid in candidate_ids:
        slug = mid.replace("_", "-")
        if slug in by_slug and by_slug[slug] != mid:
            raise ValueError(f"ambiguous normalized candidate ID: {mid}")
        by_slug[slug] = mid

    def payload_ids(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if key == "modelId" and isinstance(child, str):
                    yield child
                else:
                    yield from payload_ids(child)
        elif isinstance(value, list):
            for child in value:
                yield from payload_ids(child)

    paths = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", "-z", "HEAD", "--", "benchmark-results"],
        cwd=repo_root).decode().split("\0")
    paths = [path for path in paths if path.endswith(".json")]
    # One Git process for the evidence tree: hundreds of individual git-show processes repeatedly
    # inflate the same pack and make this otherwise small CI guard unnecessarily slow.
    batch = subprocess.run(["git", "cat-file", "--batch"], cwd=repo_root,
                           input="".join(f"HEAD:{path}\n" for path in paths).encode(),
                           stdout=subprocess.PIPE, check=True)
    objects = io.BytesIO(batch.stdout)
    for path in paths:
        _, kind, size = objects.readline().split()
        if kind != b"blob":
            raise ValueError(f"evidence is not a Git blob: {path}")
        raw = objects.read(int(size))
        if objects.read(1) != b"\n":
            raise ValueError(f"incomplete Git evidence object: {path}")
        try:
            report = json.loads(raw)
        except (ValueError, UnicodeDecodeError):
            continue
        if not isinstance(report, (dict, list)) or not report:
            continue
        ids = list(payload_ids(report))
        if not ids:
            ids = [pathlib.PurePosixPath(path).stem]
        for value in ids:
            mid = by_slug.get(value.replace("_", "-"))
            if mid is not None:
                found.add(mid)
                where[mid].add(str(pathlib.PurePosixPath(path).relative_to("benchmark-results")))
    return found, where


def classify(catalog_dir, repo_root):
    raw = read(pathlib.Path(catalog_dir) / "models.json")
    models = raw["models"] if isinstance(raw, dict) and "models" in raw else raw
    ids = [m["id"] for m in models]
    rows = manifest_rows(catalog_dir)
    seen, where = evidence_ids(repo_root, ids)

    by = {}
    for m in models:
        mid = m["id"]
        if mid in rows:
            bucket, manifest, verdict = rows[mid]
            by[mid] = {"bucket": bucket, "manifest": manifest, "verdict": verdict, "model": m}
        elif mid in seen:
            by[mid] = {"bucket": "EVALUATED_NOT_LANDED", "manifest": None,
                       "verdict": f"evidence: {', '.join(sorted(where[mid])[:2])}",
                       "model": m}
        else:
            by[mid] = {"bucket": "NOT_EVALUATED", "manifest": None, "verdict": None, "model": m}
    return by


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--catalog", required=True)
    ap.add_argument("--repo-root", default=str(pathlib.Path(__file__).resolve().parents[2]))
    ap.add_argument("--format", choices=("text", "md"), default="text")
    ap.add_argument("--check-doc", metavar="PATH",
                    help="verify this document's counts still match; exit 1 if not")
    args = ap.parse_args(argv)

    by = classify(args.catalog, args.repo_root)
    counts = collections.Counter(v["bucket"] for v in by.values())
    order = ["QUALIFIED", "REJECTED", "EVALUATED_NOT_LANDED", "NOT_EVALUATED"]
    total = sum(counts.values())

    if args.format == "md":
        print("| bucket | models | meaning |")
        print("| --- | --- | --- |")
        meaning = {
            "QUALIFIED": "a manifest row with `qualified` true",
            "REJECTED": "a manifest row with `qualified` false; the verdict is the finding",
            "EVALUATED_NOT_LANDED": "committed evidence names it, no manifest row claims it",
            "NOT_EVALUATED": "no manifest row or matching committed JSON artifact found",
        }
        for b in order:
            print(f"| **{b}** | {counts[b]} | {meaning[b]} |")
        print(f"| | **{total}** | catalog candidates |")
    else:
        print(f"catalog candidates: {total}")
        for b in order:
            print(f"  {counts[b]:>4}  {b}")

    print()
    for b in ("REJECTED", "EVALUATED_NOT_LANDED"):
        rows = [(k, v) for k, v in sorted(by.items()) if v["bucket"] == b]
        if not rows:
            continue
        print(f"{b} ({len(rows)}):")
        for mid, v in rows:
            print(f"  {mid:<56} {v['verdict']}")
        print()

    rc = 0
    if args.check_doc:
        doc = pathlib.Path(args.check_doc)
        if not doc.exists():
            print(f"{args.check_doc} does not exist")
            return 1
        text = doc.read_text()
        problems = []
        for b in order:
            if f"| **{b}** | {counts[b]} |" not in text:
                problems.append(f"{b} is not stated as {counts[b]}")
        if f"**{total}** | catalog candidates" not in text:
            problems.append(f"total is not stated as {total}")
        if problems:
            print(f"{args.check_doc} DISAGREES WITH THE CATALOG:")
            for p in problems:
                print(f"  {p}")
            rc = 1
        else:
            print(f"{args.check_doc} checked: every bucket count matches the catalog.")
    return rc


if __name__ == "__main__":
    sys.exit(main())
