#!/usr/bin/env python3
"""Derive the reranker-path fix tables from the committed per-run artifacts.

Two measurement sessions, each interleaved five rounds on the registered host class against the
pinned MS MARCO MiniLM-L6 Q4_K reranker, each run one forked JVM under /usr/bin/time -v:

  runs/         V-0520 (released v0.5.20) | A-main (main 9db11325 + #456) | P-fix (packed batch + code-point trie)
  runs-notrie/  V-0520 | P-fix | Q-notrie (packed batch, trie removed)

Every figure printed is read from a file; none is typed. Ratios are median(arm) / median(V-0520).

    python3 derive.py runs
    python3 derive.py runs-notrie
"""
import json, pathlib, re, statistics, sys

ORDER = [("V-0520", "released v0.5.20"), ("A-main", "main 9db11325 + #456 (the 0.5.21 candidate before this branch)"),
         ("P-fix", "packed batch + code-point trie"), ("Q-notrie", "packed batch, WordPiece trie removed")]
KEYS = ["coldLoadMillis", "pairP50Millis", "pairP95Millis", "batchP50Millis", "batchP95Millis", "batchP50DocumentsPerSecond", "maxRssMiB"]

def runs(root, arm):
    out = []
    for path in sorted(root.glob(f"{arm}-*.out"), key=lambda p: int(p.stem.split("-")[-1])):
        text = path.read_text(); s, e = text.find("{"), text.rfind("}")
        if s < 0: raise SystemExit(f"{path}: no JSON block")
        record = json.loads(text[s:e + 1])
        m = re.search(r"Maximum resident set size \(kbytes\): (\d+)", path.with_suffix(".err").read_text())
        record["maxRssMiB"] = int(m.group(1)) / 1024 if m else None
        out.append(record)
    return out

def main():
    root = pathlib.Path(sys.argv[1])
    present = [(a, d) for a, d in ORDER if list(root.glob(f"{a}-*.out"))]
    names = [a for a, _ in present]
    print("host:", " / ".join(l.split(":", 1)[1].strip() for l in (root / "cpu.txt").read_text().splitlines() if l.startswith(("Model name", "Core(s) per socket", "Thread(s) per core"))))
    for a, d in present:
        sha = root / f"sha-{a}.txt"; print(f"{a:9} {sha.read_text().strip().splitlines()[0][:40]:40}  {d}")
    data = {a: runs(root, a) for a in names}
    first = data[names[0]][0]
    print(f"runs per arm: { {a: len(v) for a, v in data.items()} }; warmups={first['warmups']} pairIterations={first['pairIterations']} batchIterations={first['batchIterations']}; artifact sha256={first['artifactSha256'][:16]}…")
    print()
    print("| metric | " + " | ".join(names) + " |"); print("| --- " * (len(names) + 1) + "|")
    for key in KEYS:
        cells = []; base = None
        for a in names:
            v = [r[key] for r in data[a] if r[key] is not None]
            if not v: cells.append("n/a"); continue
            med = statistics.median(v)
            if a == "V-0520": base = med
            ratio = f" ({med / base:.3f}x)" if base and a != "V-0520" else ""
            cells.append(f"{med:.2f} [{min(v):.2f}..{max(v):.2f}]{ratio}")
        print(f"| {key} | " + " | ".join(cells) + " |")
    checks = {a: sorted({round(r["scoreChecksum"], 9) for r in data[a]}) for a in names}
    print(); print("scoreChecksum per arm (distinct, 9dp):", checks)
    print("every arm reproduces released v0.5.20 scores to 9dp:", all(v == checks["V-0520"] for v in checks.values()) and all(len(v) == 1 for v in checks.values()))

if __name__ == "__main__":
    main()
