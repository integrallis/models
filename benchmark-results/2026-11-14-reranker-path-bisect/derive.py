#!/usr/bin/env python3
"""Derive the reranker-path bisect table from the committed per-run artifacts.

Arms are commits in v0.5.20..main order, each built with #360 reverted (where present) and the
#456 fix applied (where scoreAll would otherwise throw), so every arm runs the arithmetic that
ships. Interleaved V,T,F,W,X,A per round for five rounds. Each metric is the median of the
experiment's own per-run figure, with [min..max]; every ratio is against released v0.5.20.

Every figure printed is read from a file; none is typed.

    python3 derive.py <runs-directory>
"""
import json
import pathlib
import re
import statistics
import sys

# (arm, what it is) in commit order.
ARMS = [
    ("V-0520", "released v0.5.20"),
    ("T-tokenizer", "63b46a83: #435/#437 tokenizer allocation work"),
    ("F-436", "4819990c: #436 tokenizer EI_EXPOSE_REP + BERT F32 projection path; #360 reverted"),
    ("W-440", "3b37bc3c: #440 batched BERT + #438 compact F16 KV + ...; #360 reverted; #456 applied"),
    ("X-vectors", "5c72d535: vectors 0.1.34 -> 0.1.35; #360 reverted; #456 applied"),
    ("A-off", "main (cdbc9edc) + #456; #360 off - what 0.5.21 ships"),
]
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
        out.append(record)
    return out


def main():
    root = pathlib.Path(sys.argv[1])
    print("host:", " / ".join(l.split(":", 1)[1].strip() for l in (root / "cpu.txt").read_text().splitlines()
                            if l.startswith(("Model name", "Core(s) per socket", "Thread(s) per core"))))
    print("java:", (root / "java-version.txt").read_text().splitlines()[0])
    present = [(a, d) for a, d in ARMS if list(root.glob(f"{a}-*.out"))]
    for arm, what in present:
        sha = root / f"sha-{arm}.txt"
        print(f"{arm:12} {sha.read_text().strip() if sha.is_file() else '?':40}  {what}")
    data = {arm: runs(root, arm) for arm, _ in present}
    print()
    print("runs per arm:", {arm: len(v) for arm, v in data.items()})
    first = data["V-0520"][0]
    print(f"artifact: {first['model']}  sha256={first['artifactSha256']}  warmups={first['warmups']} "
          f"pairIterations={first['pairIterations']} batchIterations={first['batchIterations']}")
    print()
    names = [a for a, _ in present]
    print("| metric | " + " | ".join(names) + " |")
    print("| --- " * (len(names) + 1) + "|")
    for key in KEYS + ["maxRssMiB"]:
        cells = []
        base = None
        for arm in names:
            values = [r[key] for r in data[arm] if r[key] is not None]
            if not values:
                cells.append("n/a")
                continue
            med = statistics.median(values)
            if arm == "V-0520":
                base = med
            ratio = f" ({med / base:.3f}x)" if base and arm != "V-0520" else ""
            cells.append(f"{med:.2f} [{min(values):.2f}..{max(values):.2f}]{ratio}")
        print(f"| {key} | " + " | ".join(cells) + " |")
    print()
    print("ratios in parentheses are median(arm) / median(V-0520).")
    checks = {arm: sorted({round(r["scoreChecksum"], 9) for r in data[arm]}) for arm in names}
    print("scoreChecksum per arm (distinct, 9dp):", checks)
    print("every arm reproduces released v0.5.20 scores to 9dp:",
          all(v == checks["V-0520"] for v in checks.values()))


if __name__ == "__main__":
    main()
