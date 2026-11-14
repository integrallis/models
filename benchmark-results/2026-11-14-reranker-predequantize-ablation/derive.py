#!/usr/bin/env python3
"""Derive the #360 ablation table from the committed per-run artifacts.

Three arms, interleaved A,B,C per round so drift hits each arm equally:

  A-off     pre-dequantization disabled (predicate forced false)
  B-narrow  the fix: projections only, and only when no kernel takes the type
  C-main    main as released-candidate: the original #360 condition

Every figure printed is read from a file; none is typed. Latency comes from the experiment's
own JSON (medians across rounds of its per-run p50/p95), memory from /usr/bin/time -v around
the forked run, and the numerical check from the experiment's scoreChecksum, which must agree
across arms to the printed precision or the arms are not computing the same thing.

    python3 derive360.py <ab-directory>
"""
import json
import pathlib
import re
import statistics
import sys

ORDER = ["V-0520", "A-off", "B-narrow", "C-main"]
ARMS: list = []  # filled from the files present
KEYS = ["coldLoadMillis", "pairP50Millis", "pairP95Millis", "batchP50Millis", "batchP95Millis",
        "batchP50DocumentsPerSecond"]


def runs(root, arm):
    out = []
    for path in sorted(root.glob(f"{arm}-*.out"), key=lambda p: int(p.stem.split("-")[-1])):
        text = path.read_text()
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end < 0:
            raise SystemExit(f"{path}: no JSON block")
        record = json.loads(text[start:end + 1])
        err = path.with_suffix(".err").read_text()
        match = re.search(r"Maximum resident set size \(kbytes\): (\d+)", err)
        record["maxRssMiB"] = int(match.group(1)) / 1024 if match else None
        record["_run"] = path.name
        out.append(record)
    return out


def main():
    root = pathlib.Path(sys.argv[1])
    print("host:", " / ".join(l.split(":", 1)[1].strip() for l in (root / "cpu.txt").read_text().splitlines()
                            if l.startswith(("Model name", "Core(s) per socket", "Thread(s) per core"))))
    print("java:", (root / "java-version.txt").read_text().splitlines()[0])
    for arm in ORDER:
        sha = root / f"sha-{arm}.txt"
        if sha.is_file():
            print(f"{arm:9} {sha.read_text().strip()}")
    if list(root.glob("V-0520-*.out")):
        print("V-0520    tag v0.5.20 (released), reference arm")
    print()
    ARMS.extend(a for a in ORDER if list(root.glob(f"{a}-*.out")))
    data = {arm: runs(root, arm) for arm in ARMS}
    counts = {arm: len(v) for arm, v in data.items()}
    print("runs per arm:", counts)
    first = data["A-off"][0]
    print(f"artifact: {first['model']}  sha256={first['artifactSha256']}  warmups={first['warmups']} "
          f"pairIterations={first['pairIterations']} batchIterations={first['batchIterations']} "
          f"processors={first['processors']}")
    print()
    ratios = [r for r, need in [("B/A", ("A-off", "B-narrow")), ("C/B", ("B-narrow", "C-main")), ("A/V", ("V-0520", "A-off"))]
              if all(a in ARMS for a in need)]
    header = "| metric | " + " | ".join(ARMS) + " | " + " | ".join(ratios) + " |"
    print(header)
    print("| --- " * (header.count("|") - 1) + "|")
    for key in KEYS + ["maxRssMiB"]:
        values = {arm: [r[key] for r in data[arm] if r[key] is not None] for arm in ARMS}
        med = {arm: (statistics.median(v) if v else None) for arm, v in values.items()}
        spread = {arm: ((min(v), max(v)) if v else None) for arm, v in values.items()}
        cells = [f"{med[arm]:.2f} [{spread[arm][0]:.2f}..{spread[arm][1]:.2f}]" if med[arm] is not None
                 else "n/a (no /usr/bin/time wrapper)" for arm in ARMS]
        def ratio(num, den):
            return f"{med[num] / med[den]:.3f}" if med.get(num) and med.get(den) else "n/a"
        cols = {"B/A": ("B-narrow", "A-off"), "C/B": ("C-main", "B-narrow"), "A/V": ("A-off", "V-0520")}
        print(f"| {key} | " + " | ".join(cells) + " | " + " | ".join(ratio(*cols[r]) for r in ratios) + " |")
    print()
    checks = {arm: sorted({round(r["scoreChecksum"], 9) for r in data[arm]}) for arm in ARMS}
    print("scoreChecksum per arm (distinct values, 9dp):", checks)
    if "V-0520" in checks and len(checks["V-0520"]) == 1 and len(checks["A-off"]) == 1:
        print("A-off reproduces released v0.5.20 scores to 9dp:", checks["A-off"] == checks["V-0520"])
    if "B-narrow" in checks:
        print("B-narrow equals A-off to 9dp:", checks["B-narrow"] == checks["A-off"])
    if "C-main" in checks and "B-narrow" in checks:
        print("C-main equals B-narrow to 9dp:", checks["C-main"] == checks["B-narrow"])
    tops = {arm: sorted({tuple(r["topTwoOriginalIndexes"]) for r in data[arm]}) for arm in ARMS}
    print("topTwoOriginalIndexes per arm:", tops)


if __name__ == "__main__":
    main()
