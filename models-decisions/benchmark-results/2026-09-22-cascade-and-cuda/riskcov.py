"""Risk-coverage: is the cheap model right when it is confident? Nothing else decides a cascade."""
import json, math, hashlib
BENCH = "/Users/briansam-bodden/Code/java-ai/references/openjev/gh-jevbench/datasets/public"

def load_gold():
    g = {}
    for s in ("easy","original","hard"):
        for line in open(f"{BENCH}/{s}.jsonl"):
            t = json.loads(line); g[t["id"]] = (str(t["expected"]), s, t["question"]["type"])
    return g

def load_arm(p):
    a = {}
    for line in open(p):
        parts = line.rstrip("\n").split("\t"); probs = {}
        for kv in parts[2:]:
            if kv:
                k, v = kv.rsplit("=",1); probs[k]=float(v)
        a[parts[0]] = (float(parts[1]), probs)
    return a

def argmax(p): return max(p.items(), key=lambda kv: kv[1])[0]

gold = load_gold()
arms = {
    "qwen0.6b-letters": load_arm("/tmp/cascade/qwen-letters.tsv"),
    "qwen0.6b-scores":  load_arm("/tmp/cascade/qwen-scores.tsv"),
    "granite3b":        load_arm("/tmp/cascade/granite-scores.tsv"),
}
ids = sorted(set(gold) & set(arms["granite3b"]))

for name, arm in arms.items():
    # margin = top1 - top2, generally a better selective score than raw max
    rows = []
    for i in ids:
        p = sorted(arm[i][1].values(), reverse=True)
        margin = p[0] - (p[1] if len(p) > 1 else 0.0)
        rows.append((p[0], margin, argmax(arm[i][1]) == gold[i][0], arm[i][0]))
    base = sum(r[2] for r in rows)/len(rows)
    print(f"\n=== {name}  overall acc {base:.4f} ===")
    for score_name, idx in (("p_max", 0), ("margin", 1)):
        srt = sorted(rows, key=lambda r: -r[idx])
        print(f"  selective by {score_name}:")
        for cov in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0):
            k = max(1, int(cov*len(srt)))
            acc = sum(r[2] for r in srt[:k])/k
            thr = srt[k-1][idx]
            print(f"    coverage {cov:4.0%}  n={k:3d}  acc {acc:.4f}  threshold {thr:.4f}")
