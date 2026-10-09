#!/usr/bin/env python3
"""Derives the Q5_1 matmul share from the committed tensor-table dumps in raw/.

Hand-typing this table got the denominator wrong once: token_embd.weight is a lookup, not a
matmul, so it must be excluded, and F32 norms are vectors rather than matrices.
"""
import pathlib, re, sys

RAW = pathlib.Path(__file__).parent / "raw"
EXCLUDED = {"F32"}  # per-channel norms, not projections


def shares(path):
    rows = {}
    for line in path.read_text().splitlines():
        match = re.match(r"\s+(\S+)\s+tensors=\s*(\d+)\s+weights=\s*([\d,]+)", line)
        if match:
            rows[match.group(1)] = (int(match.group(2)), int(match.group(3).replace(",", "")))
    # token_embd.weight is the single widest tensor and is read by lookup; the dump names the
    # Q5_1 tensors but not the Q8_0 one, so it is identified by being one tensor of embedding width.
    matmul = {t: v for t, v in rows.items() if t not in EXCLUDED}
    embed = max(matmul.items(), key=lambda item: item[1][1])
    del matmul[embed[0]]
    total = sum(weights for _, weights in matmul.values())
    tensors = sum(count for count, _ in matmul.values())
    q51 = matmul.get("Q5_1", (0, 0))
    return q51[0], tensors, q51[1], total, (100.0 * q51[1] / total if total else 0.0)


def main():
    print(f"{'artifact':34s} {'Q5_1 tensors':>14s} {'Q5_1 weights':>14s} {'matmul weights':>15s} {'share':>8s}")
    for path in sorted(RAW.glob("weight-share-*.txt")):
        q51_tensors, tensors, q51_weights, total, pct = shares(path)
        name = path.stem.replace("weight-share-", "")
        print(f"{name:34s} {q51_tensors:>6d} of {tensors:<4d} {q51_weights:>14,d} {total:>15,d} {pct:>7.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
