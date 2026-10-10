#!/usr/bin/env python3
"""Every evidence path a document cites must exist.

A number in a document is only evidence if the artifact it points at is real. On 2026-10-09 a
CHANGELOG entry for 0.3.56 cited `benchmark-results/2026-10-08-q4-1-exact-path/NOTES.md`, which
never existed -- the real directory was `2026-10-09-q4-1-support`. It was caught by hand before
release, and nothing would have caught it otherwise.

Scope is deliberately narrow: references to this repository's own evidence trees,
`benchmark-results/` and `experiments/`. Those are the paths that back measurements, so a broken
one is an integrity failure rather than a stale link. Cross-repository references (the ModelJars
`catalog/`, its `tools/`) and paths that lived on a remote host are not checked here, because a
check that reports things nobody can fix gets ignored, and an ignored check is worse than none.

A reference resolves if it exists relative to the repository root or relative to the citing
document. Both are legitimate ways to cite a sibling artifact.

    python3 scripts/integrity/check-evidence-references.py [--root .]

Exits 1 if any reference is unresolvable.
"""
import argparse
import collections
import json
import pathlib
import re
import sys

REFERENCE = re.compile(r"((?:benchmark-results|experiments)/[A-Za-z0-9_][A-Za-z0-9_./-]*)")
# A reference whose last segment is a bare prefix is prose describing a family of directories
# ("benchmark-results/certified-*"), not a citation of one file.
PROSE_TAIL = re.compile(r"[-/]$")
DOC_GLOBS = ("*.md", "docs/**/*.md", "benchmark-results/**/*.md", "experiments/**/*.md",
             "scripts/**/*.md")


def references(root: pathlib.Path):
    docs = sorted({p for g in DOC_GLOBS for p in root.glob(g)})
    for doc in docs:
        try:
            text = doc.read_text()
        except (OSError, UnicodeDecodeError):
            continue
        for match in REFERENCE.finditer(text):
            path = match.group(1).rstrip(".,);:`'\"")
            if PROSE_TAIL.search(path) or "*" in path:
                continue
            yield doc, path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=".")
    ap.add_argument("--baseline",
                    default="scripts/integrity/evidence-reference-baseline.json",
                    help="known-unresolvable references, enumerated. The check stays live for "
                         "everything else; this list may shrink and must never grow.")
    args = ap.parse_args(argv)
    root = pathlib.Path(args.root).resolve()

    baseline_path = root / args.baseline
    baseline = {}
    if baseline_path.exists():
        baseline = {e["document"]: set(e["references"])
                    for e in json.loads(baseline_path.read_text())["known"]}

    missing = collections.defaultdict(set)
    waived = collections.defaultdict(set)
    checked = 0
    for doc, path in references(root):
        checked += 1
        if (root / path).exists() or (doc.parent / path).exists():
            continue
        rel = str(doc.relative_to(root))
        if path in baseline.get(rel, ()):
            waived[rel].add(path)
        else:
            missing[rel].add(path)

    print(f"evidence references checked: {checked}")
    waived_total = sum(len(v) for v in waived.values())
    if waived_total:
        print(f"waived by the baseline: {waived_total} (pre-existing, enumerated, must not grow)")

    # A baseline entry that now resolves has been fixed, and leaving it listed would let a future
    # break hide behind it. Shrinking the baseline is mandatory, not optional.
    stale = []
    for rel, paths in baseline.items():
        doc = root / rel
        for path in paths:
            if (root / path).exists() or (doc.parent / path).exists():
                stale.append((rel, path))
    if stale:
        print()
        print(f"BASELINE IS STALE: {len(stale)} waived reference(s) now resolve and must be "
              f"removed from {args.baseline}:")
        for rel, path in sorted(stale):
            print(f"  {rel}  ->  {path}")
        return 1

    if not missing:
        print("every cited evidence path exists, or is a baselined pre-existing gap.")
        return 0
    total = sum(len(v) for v in missing.values())
    print()
    print(f"UNRESOLVABLE AND NOT BASELINED: {total} reference(s) in {len(missing)} document(s).")
    print("A number is only evidence if the artifact it cites is real. Either commit the")
    print("artifact, correct the path, or -- if it genuinely cannot be produced -- add it to the")
    print("baseline with a reason, which is a statement that the document is not self-contained.")
    for rel, paths in sorted(missing.items()):
        print(f"  {rel}")
        for p in sorted(paths):
            print(f"      {p}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
