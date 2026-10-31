"""Per-layer residual-sum comparison for maple, ours against llama.cpp.

Reads `l_out-<n> <sum>` lines from each arm and prints the relative delta per layer. No tolerance
constant lives here: the number is printed and the reader decides, because the question this
answers ("did the computation fix close the gap?") is about an order of magnitude and not about a
threshold somebody picked.

Usage: python3 compare_ladder.py <oracle.txt> <ours.txt> [<ours-prefix.txt>]
"""
import re
import sys


def read(path):
    sums = {}
    for line in open(path):
        match = re.match(r"l_out-(\d+)\s+(-?[\d.]+)", line.strip())
        if match:
            sums[int(match.group(1))] = float(match.group(2))
    return sums


def main(argv):
    oracle = read(argv[0])
    ours = read(argv[1])
    before = read(argv[2]) if len(argv) > 2 else {}

    header = f"{'L':>3} {'llama.cpp':>14} {'ours':>14} {'rel':>11}"
    if before:
        header += f" | {'ours (pre-fix)':>15} {'rel':>11}"
    print(header)

    worst = worst_before = 0.0
    for layer in sorted(oracle):
        relative = abs(ours[layer] - oracle[layer]) / abs(oracle[layer])
        worst = max(worst, relative)
        row = f"{layer:>3} {oracle[layer]:>14.6f} {ours[layer]:>14.6f} {relative * 100:>10.5f}%"
        if layer in before:
            was = abs(before[layer] - oracle[layer]) / abs(oracle[layer])
            worst_before = max(worst_before, was)
            row += f" | {before[layer]:>15.6f} {was * 100:>10.5f}%"
        print(row)

    print(f"\nworst relative delta, {len(oracle)} layers: {worst * 100:.6f}%")
    if before:
        print(f"worst relative delta before the fix:    {worst_before * 100:.6f}%")


if __name__ == "__main__":
    main(sys.argv[1:])
