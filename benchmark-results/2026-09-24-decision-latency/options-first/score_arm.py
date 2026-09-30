"""Score one arm's runner output with JevBench's own scoring code.

The runner writes `id<TAB>latency<TAB>label=p...` per task. Everything that turns those into an
axis -- validity, argmax, ECE, the tier weights, the speed and cost curves -- comes from the
benchmark's own modules, imported rather than reimplemented. Reimplementing them is how three
published numbers went wrong before.
"""
import json
import sys
from collections import defaultdict

sys.path.insert(0, "/Users/briansam-bodden/Code/java-ai/references/openjev/gh-jevbench")
from jevbench import composite_v12 as cv  # noqa: E402
from jevbench import metrics, scoring  # noqa: E402


class Task:
    def __init__(self, row):
        self.labels = row["labels"]
        self.expected = row["expected"]
        self.question = row["question"]
        self.id = row["id"]
        self.family = row["family"]


def tier_of(task_id):
    if task_id.startswith("easy-"):
        return "easy"
    if task_id.startswith("hard-"):
        return "hard"
    return "judge"


def main(cohort, results, label):
    tasks = {}
    for line in open(cohort):
        row = json.loads(line)
        tasks[row["id"]] = Task(row)

    per_tier = defaultdict(lambda: {"n": 0, "correct": 0})
    pairs = []
    latencies = []
    per_family = defaultdict(lambda: {"n": 0, "correct": 0})
    predictions = {}

    for line in open(results):
        fields = line.rstrip("\n").split("\t")
        task = tasks[fields[0]]
        latencies.append(float(fields[1]))
        probs = {}
        for cell in fields[2:]:
            name, value = cell.rsplit("=", 1)
            probs[name] = float(value)
        scored = scoring.score_task(probs, task)
        tier = tier_of(task.id)
        per_tier[tier]["n"] += 1
        per_tier[tier]["correct"] += 1 if scored["correct"] else 0
        per_family[task.family]["n"] += 1
        per_family[task.family]["correct"] += 1 if scored["correct"] else 0
        predictions[task.id] = scored["predicted"]
        if scored["valid"]:
            pairs.append((scoring.top_label_confidence(scored["probs"]), bool(scored["correct"])))

    latencies.sort()
    p50 = latencies[len(latencies) // 2]
    p95 = latencies[min(len(latencies) - 1, int(0.95 * len(latencies)))]
    accuracy = {t: v["correct"] / v["n"] for t, v in per_tier.items()}
    ece = metrics.ece_top_label(pairs)["ece"]

    axes = {
        "intelligence": cv.intelligence(accuracy),
        "calibration": cv.calibration(ece),
        "speed": cv.speed(p50, p95, "cpu"),
    }
    print(f"\n  == {label} ==  {len(latencies)} items")
    for tier in ("easy", "judge", "hard"):
        if tier in per_tier:
            v = per_tier[tier]
            print(f"  {tier:<8} {v['correct']:>3}/{v['n']:<4} {v['correct'] / v['n']:.4f}")
    total_n = sum(v["n"] for v in per_tier.values())
    total_c = sum(v["correct"] for v in per_tier.values())
    print(f"  {'overall':<8} {total_c:>3}/{total_n:<4} {total_c / total_n:.4f}")
    print(f"  Intelligence {axes['intelligence']:.1f}   Calibration {axes['calibration']:.1f} "
          f"(ECE {ece:.4f})   Speed {axes['speed']:.1f}  (p50 {p50:.3f}s p95 {p95:.3f}s)")
    return predictions, per_family


if __name__ == "__main__":
    cohort = sys.argv[1]
    first, first_family = main(cohort, sys.argv[2], sys.argv[3])
    if len(sys.argv) > 5:
        second, second_family = main(cohort, sys.argv[4], sys.argv[5])
        # Compared against each other, because a difference smaller than what an unrelated
        # implementation change produces is not a result. Merely swapping the matrix kernel moves
        # this model's logits by 6e-2 on average, which flips whichever items were close.

        flipped = [k for k in first if first[k] != second.get(k)]
        print(f"\n  winners differ on {len(flipped)}/{len(first)} items")
        moved = []
        for family in sorted(set(first_family) | set(second_family)):
            a = first_family[family]
            b = second_family[family]
            if a["n"] and (a["correct"] != b["correct"]):
                moved.append((family, a["correct"] / a["n"], b["correct"] / b["n"], a["n"]))
        if moved:
            print(f"\n  {'family':<22} {sys.argv[3]:<14} {sys.argv[5]:<14} n")
            for family, x, y, n in sorted(moved, key=lambda m: m[2] - m[1]):
                print(f"  {family:<22} {x:<14.4f} {y:<14.4f} {n}")
        print()
