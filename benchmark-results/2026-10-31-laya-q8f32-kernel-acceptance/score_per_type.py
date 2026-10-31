"""Per-question-type accuracy for one arm, with JevBench's own scoring.

#292's acceptance criterion names three per-type cells as well as the overall accuracy, and
`../2026-09-24-decision-latency/options-first/score_arm.py` reports per TIER and per FAMILY, not
per type. This adds the one breakdown that criterion needs and nothing else -- it imports
`jevbench.scoring.score_task`, the same function that scorer uses, rather than deciding
correctness itself.

That distinction is not pedantic. An argmax over the probability columns reproduces `choice` and
`noul` exactly and reports 0/12 on `score`, because a `score`-type item is not a pick-the-label
question. Scoring it by hand would have reported a catastrophic per-type regression that does not
exist.

Usage: python3 score_per_type.py <cohort.jsonl> <arm.tsv>
"""
import collections
import json
import sys

sys.path.insert(0, "/Users/briansam-bodden/Code/java-ai/references/openjev/gh-jevbench")
from jevbench import scoring  # noqa: E402


class Task:
    def __init__(self, row):
        self.labels = row["labels"]
        self.expected = row["expected"]
        self.question = row["question"]
        self.id = row["id"]
        self.family = row["family"]


def main(cohort, arm):
    tasks = {}
    for line in open(cohort):
        row = json.loads(line)
        tasks[row["id"]] = Task(row)

    per_type = collections.defaultdict(lambda: [0, 0])
    for line in open(arm):
        fields = line.rstrip("\n").split("\t")
        task = tasks[fields[0]]
        probs = {}
        for cell in fields[2:]:
            name, value = cell.rsplit("=", 1)
            probs[name] = float(value)
        scored = scoring.score_task(probs, task)
        bucket = per_type[task.question.get("type")]
        bucket[1] += 1
        bucket[0] += 1 if scored["correct"] else 0

    published = {"choice": 0.791667, "noul": 0.777778, "score": 0.750000}
    print(f"{'type':<8} {'measured':>18} {'published':>10}  match")
    for question_type in sorted(per_type):
        correct, n = per_type[question_type]
        expected = published.get(question_type)
        verdict = "-" if expected is None else ("YES" if abs(correct / n - expected) < 5e-7 else "NO")
        shown = f"{expected:.6f}" if expected is not None else "-"
        print(f"{question_type:<8} {correct}/{n} = {correct / n:.6f} {shown:>10}  {verdict}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
