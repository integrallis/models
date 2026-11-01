"""Cheap proposer + gated expensive verifier, measured offline on stored per-item outputs.

No new inference: joins two arms already run on the same 231 JevBench items on the same host.
Threshold is fitted on one half and reported on the other, because every in-sample gate this
project has fitted so far has evaporated when held out.
"""
import json, math, hashlib, statistics, sys

BENCH = "/Users/briansam-bodden/Code/java-ai/references/openjev/gh-jevbench/datasets/public"

def load_gold():
    gold = {}
    for split in ("easy", "original", "hard"):
        for line in open(f"{BENCH}/{split}.jsonl"):
            t = json.loads(line)
            gold[t["id"]] = (str(t["expected"]), split, t["question"]["type"], t["family"])
    return gold

def load_arm(path):
    arm = {}
    for line in open(path):
        parts = line.rstrip("\n").split("\t")
        probs = {}
        for kv in parts[2:]:
            if kv:
                k, v = kv.rsplit("=", 1)
                probs[k] = float(v)
        arm[parts[0]] = (float(parts[1]), probs)
    return arm

def argmax(probs):
    return max(probs.items(), key=lambda kv: kv[1])[0]

def retemper(p, T):
    z = {k: math.log(max(v, 1e-12)) / T for k, v in p.items()}
    m = max(z.values())
    e = {k: math.exp(v - m) for k, v in z.items()}
    s = sum(e.values())
    return {k: v / s for k, v in e.items()}

def fit_T(items, arm, gold):
    def nll(T):
        total = 0.0
        for i in items:
            q = retemper(arm[i][1], T)
            total -= math.log(max(q.get(gold[i][0], 1e-12), 1e-12))
        return total / len(items)
    lo, hi = math.log(1/64), math.log(64)
    for _ in range(200):
        a, b = lo + (hi-lo)/3, hi - (hi-lo)/3
        if nll(math.exp(a)) < nll(math.exp(b)): hi = b
        else: lo = a
    return math.exp((lo+hi)/2)

def accuracy(items, arm, gold):
    return sum(1 for i in items if argmax(arm[i][1]) == gold[i][0]) / len(items)

def cascade(items, prop, ver, gold, tau, T):
    correct = 0; deferred = 0; lat = []
    for i in items:
        p = retemper(prop[i][1], T)
        if max(p.values()) >= tau:
            correct += argmax(p) == gold[i][0]
            lat.append(prop[i][0])
        else:
            deferred += 1
            correct += argmax(ver[i][1]) == gold[i][0]
            lat.append(prop[i][0] + ver[i][0])
    lat.sort()
    def pct(q):
        return lat[min(len(lat)-1, int(math.ceil(q*len(lat)))-1)]
    return correct/len(items), deferred/len(items), pct(0.50), pct(0.95), sum(lat)/len(lat)

gold = load_gold()
prop = load_arm("/tmp/cascade/qwen-letters.tsv")      # Qwen3-0.6B Q4_K_M, letter-logit scoring
ver  = load_arm("/tmp/cascade/granite-scores.tsv")    # Granite 4.1 3B Q4_K_M
ids  = sorted(set(gold) & set(prop) & set(ver))
assert len(ids) == 231, len(ids)

# Deterministic stratified halves: md5 parity within each (split, type) stratum.
fit_ids, eval_ids = [], []
for i in ids:
    h = int(hashlib.md5((gold[i][1] + gold[i][2] + i).encode()).hexdigest(), 16)
    (fit_ids if h % 2 == 0 else eval_ids).append(i)

print(f"items {len(ids)}  fit {len(fit_ids)}  eval {len(eval_ids)}")
print()
print("=== ARM ACCURACY (whole 231, for reference) ===")
for name, arm in (("proposer Qwen3-0.6B", prop), ("verifier Granite-3B", ver)):
    lats = sorted(a[0] for a in arm.values())
    p50 = lats[len(lats)//2]; p95 = lats[int(0.95*len(lats))]
    print(f"  {name:22s} acc {accuracy(ids, arm, gold):.4f}  p50 {p50:.3f}s  p95 {p95:.3f}s")
    for split in ("easy", "original", "hard"):
        sub = [i for i in ids if gold[i][1] == split]
        print(f"      {split:9s} n={len(sub):3d}  acc {accuracy(sub, arm, gold):.4f}")
print()

T = fit_T(fit_ids, prop, gold)
print(f"=== GATE CALIBRATION ===\n  proposer temperature fitted on fit half: T = {T:.4f}")
print()

print("=== THRESHOLD SWEEP ON THE FIT HALF ===")
print(f"  {'tau':>5} {'acc':>7} {'defer':>7} {'p50':>7} {'p95':>7}")
rows = []
for tau in [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 1.01]:
    a, d, p50, p95, mean = cascade(fit_ids, prop, ver, gold, tau, T)
    rows.append((tau, a, d))
    print(f"  {tau:5.3f} {a:7.4f} {d:7.3f} {p50:7.3f} {p95:7.3f}")

ver_fit = accuracy(fit_ids, ver, gold)
print(f"\n  verifier-alone accuracy on fit half: {ver_fit:.4f}")
# Rule fixed before looking at the eval half: smallest deferral whose fit accuracy is not below
# verifier-alone. Ties broken toward lower tau (less deferral).
candidates = [(d, tau) for tau, a, d in rows if a >= ver_fit]
chosen = min(candidates)[1] if candidates else 1.01
print(f"  selected tau (no accuracy loss vs verifier on fit half): {chosen:.3f}")

print("\n=== HELD-OUT EVAL HALF ===")
pa = accuracy(eval_ids, prop, gold); va = accuracy(eval_ids, ver, gold)
plat = sorted(prop[i][0] for i in eval_ids); vlat = sorted(ver[i][0] for i in eval_ids)
pp50 = plat[len(plat)//2]; pp95 = plat[int(0.95*len(plat))]
vp50 = vlat[len(vlat)//2]; vp95 = vlat[int(0.95*len(vlat))]
print(f"  proposer alone   acc {pa:.4f}  defer 0.000  p50 {pp50:.3f}s  p95 {pp95:.3f}s")
print(f"  verifier alone   acc {va:.4f}  defer 1.000  p50 {vp50:.3f}s  p95 {vp95:.3f}s")
a, d, p50, p95, mean = cascade(eval_ids, prop, ver, gold, chosen, T)
print(f"  cascade t={chosen:.3f}  acc {a:.4f}  defer {d:.3f}  p50 {p50:.3f}s  p95 {p95:.3f}s")
n = len(eval_ids)
se = math.sqrt(a*(1-a)/n)
print(f"  eval n={n}, SE on accuracy +/- {se:.4f}; cascade - verifier = {a-va:+.4f}")

print("\n=== FULL EVAL-HALF CURVE (reported, not selected on) ===")
print(f"  {'tau':>5} {'acc':>7} {'defer':>7} {'p50':>7} {'p95':>7}")
for tau in [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 1.01]:
    a, d, p50, p95, mean = cascade(eval_ids, prop, ver, gold, tau, T)
    print(f"  {tau:5.3f} {a:7.4f} {d:7.3f} {p50:7.3f} {p95:7.3f}")
