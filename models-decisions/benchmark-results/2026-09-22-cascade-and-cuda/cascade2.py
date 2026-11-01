"""Cascade with the sequence-scoring 0.6B as proposer. Threshold fitted on one half only."""
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
            if kv: k,v = kv.rsplit("=",1); probs[k]=float(v)
        a[parts[0]] = (float(parts[1]), probs)
    return a
def argmax(p): return max(p.items(), key=lambda kv: kv[1])[0]
def pct(xs,q):
    s=sorted(xs); return s[min(len(s)-1, int(math.ceil(q*len(s)))-1)]

gold = load_gold()
prop = load_arm("/tmp/cascade/qwen-scores.tsv")     # Qwen3-0.6B, sequence scoring
alt  = load_arm("/tmp/cascade/qwen-letters.tsv")    # Qwen3-0.6B, letter logits
ver  = load_arm("/tmp/cascade/granite-scores.tsv")  # Granite 4.1 3B
ids  = sorted(set(gold) & set(prop) & set(ver) & set(alt))

fit_ids, eval_ids = [], []
for i in ids:
    h = int(hashlib.md5((gold[i][1]+gold[i][2]+i).encode()).hexdigest(), 16)
    (fit_ids if h % 2 == 0 else eval_ids).append(i)

def run(items, tau, gate):
    corr=0; defer=0; lat=[]; acc_on_kept=[]; acc_on_defer=[]
    for i in items:
        if gate(i) >= tau:
            ok = argmax(prop[i][1]) == gold[i][0]
            corr += ok; acc_on_kept.append(ok); lat.append(prop[i][0])
        else:
            defer += 1
            ok = argmax(ver[i][1]) == gold[i][0]
            corr += ok; acc_on_defer.append(ok); lat.append(prop[i][0] + ver[i][0])
    n=len(items)
    return dict(acc=corr/n, defer=defer/n,
                p50=pct(lat,0.5), p95=pct(lat,0.95), mean=sum(lat)/n,
                kept_acc=(sum(acc_on_kept)/len(acc_on_kept)) if acc_on_kept else float('nan'),
                defer_acc=(sum(acc_on_defer)/len(acc_on_defer)) if acc_on_defer else float('nan'))

gates = {
  "p_max":    lambda i: max(prop[i][1].values()),
  "margin":   lambda i: (lambda v: v[0]-(v[1] if len(v)>1 else 0))(sorted(prop[i][1].values(), reverse=True)),
  # cheap ensemble: two scorings of the SAME 0.6B must agree, then gate on p_max
  "agree":    lambda i: max(prop[i][1].values()) if argmax(prop[i][1])==argmax(alt[i][1]) else -1.0,
}

TAUS = [i/100 for i in range(0,101,5)] + [0.97,0.98,0.99,0.995,1.01]
ver_fit = sum(argmax(ver[i][1])==gold[i][0] for i in fit_ids)/len(fit_ids)
ver_ev  = sum(argmax(ver[i][1])==gold[i][0] for i in eval_ids)/len(eval_ids)
vlat_ev = [ver[i][0] for i in eval_ids]
plat_ev = [prop[i][0] for i in eval_ids]
prop_ev = sum(argmax(prop[i][1])==gold[i][0] for i in eval_ids)/len(eval_ids)

print(f"fit n={len(fit_ids)}  eval n={len(eval_ids)}")
print(f"verifier alone  fit acc {ver_fit:.4f}   eval acc {ver_ev:.4f}  p50 {pct(vlat_ev,.5):.3f}s p95 {pct(vlat_ev,.95):.3f}s mean {sum(vlat_ev)/len(vlat_ev):.3f}s")
print(f"proposer alone                eval acc {prop_ev:.4f}  p50 {pct(plat_ev,.5):.3f}s p95 {pct(plat_ev,.95):.3f}s mean {sum(plat_ev)/len(plat_ev):.3f}s")

for gname, g in gates.items():
    # Selection rule fixed in advance: on the FIT half, the tau minimising mean latency subject to
    # accuracy >= verifier-alone accuracy. No peeking at the eval half.
    best = None
    for t in TAUS:
        r = run(fit_ids, t, g)
        if r["acc"] >= ver_fit and (best is None or r["mean"] < best[1]["mean"]):
            best = (t, r)
    print(f"\n=== gate: {gname} ===")
    if best is None:
        print("  no tau on the fit half reaches verifier-alone accuracy -> gate rejected")
    else:
        t, rf = best
        re_ = run(eval_ids, t, g)
        n = len(eval_ids); se = math.sqrt(re_['acc']*(1-re_['acc'])/n)
        print(f"  selected tau={t:.3f} (fit: acc {rf['acc']:.4f} defer {rf['defer']:.3f} mean {rf['mean']:.3f}s)")
        print(f"  HELD OUT: acc {re_['acc']:.4f} (SE {se:.4f})  defer {re_['defer']:.3f}"
              f"  p50 {re_['p50']:.3f}s  p95 {re_['p95']:.3f}s  mean {re_['mean']:.3f}s")
        print(f"            vs verifier alone: acc {re_['acc']-ver_ev:+.4f}  mean latency "
              f"{re_['mean']/(sum(vlat_ev)/len(vlat_ev)):.2f}x")
        print(f"            accuracy on kept {re_['kept_acc']:.4f} / on deferred {re_['defer_acc']:.4f}")
    print(f"  full eval curve ({gname}):")
    print(f"    {'tau':>6} {'acc':>7} {'defer':>7} {'p50':>7} {'p95':>7} {'mean':>7}")
    for t in [0.0,0.5,0.7,0.8,0.9,0.95,0.97,0.99,1.01]:
        r = run(eval_ids, t, g)
        print(f"    {t:6.3f} {r['acc']:7.4f} {r['defer']:7.3f} {r['p50']:7.3f} {r['p95']:7.3f} {r['mean']:7.3f}")
