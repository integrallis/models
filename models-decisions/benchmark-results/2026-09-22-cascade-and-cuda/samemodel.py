"""Same-model cascade: letter-logits (1 forward) proposes, candidate scoring (k forwards) verifies.
Both share the frozen prefix, so the proposer costs one extra logit read, not a second prefill."""
import json, math, hashlib
BENCH="/Users/briansam-bodden/Code/java-ai/references/openjev/gh-jevbench/datasets/public"
def load_gold():
    g={}
    for s in ("easy","original","hard"):
        for line in open(f"{BENCH}/{s}.jsonl"):
            t=json.loads(line); g[t["id"]]=(str(t["expected"]),s,t["question"]["type"])
    return g
def load_arm(p):
    a={}
    for line in open(p):
        parts=line.rstrip("\n").split("\t"); pr={}
        for kv in parts[2:]:
            if kv: k,v=kv.rsplit("=",1); pr[k]=float(v)
        a[parts[0]]=(float(parts[1]),pr)
    return a
def argmax(p): return max(p.items(),key=lambda kv:kv[1])[0]
def pct(xs,q):
    s=sorted(xs); return s[min(len(s)-1,int(math.ceil(q*len(s)))-1)]
gold=load_gold()
prop=load_arm("/tmp/cascade/qwen-letters.tsv")   # 1 forward pass
ver =load_arm("/tmp/cascade/qwen-scores.tsv")    # k forward passes, same 0.6B
ids=sorted(set(gold)&set(prop)&set(ver))
fit,ev=[],[]
for i in ids:
    h=int(hashlib.md5((gold[i][1]+gold[i][2]+i).encode()).hexdigest(),16)
    (fit if h%2==0 else ev).append(i)
# Proposer is free by construction (shares the prefill); verifier latency is the measured
# candidate-scoring latency. Deferred item costs the scoring pass only.
def run(items,tau):
    c=0;d=0;lat=[]
    for i in items:
        v=sorted(prop[i][1].values(),reverse=True); s=v[0]-(v[1] if len(v)>1 else 0)
        if s>=tau:
            c+=argmax(prop[i][1])==gold[i][0]
            lat.append(prop[i][0])          # measured single-pass latency
        else:
            d+=1; c+=argmax(ver[i][1])==gold[i][0]
            lat.append(ver[i][0])           # scoring pass replaces it, prefix already frozen
    n=len(items)
    return c/n,d/n,pct(lat,.5),pct(lat,.95),sum(lat)/n
ver_ev=sum(argmax(ver[i][1])==gold[i][0] for i in ev)/len(ev)
pro_ev=sum(argmax(prop[i][1])==gold[i][0] for i in ev)/len(ev)
vlat=[ver[i][0] for i in ev]; plat=[prop[i][0] for i in ev]
print(f"eval n={len(ev)}")
print(f"  letter-logits alone   acc {pro_ev:.4f}  p50 {pct(plat,.5):.3f} p95 {pct(plat,.95):.3f} mean {sum(plat)/len(plat):.3f}")
print(f"  candidate-scoring alone acc {ver_ev:.4f}  p50 {pct(vlat,.5):.3f} p95 {pct(vlat,.95):.3f} mean {sum(vlat)/len(vlat):.3f}")
# tau chosen on fit half: min mean latency s.t. fit acc >= scoring-alone fit acc
fit_ver=sum(argmax(ver[i][1])==gold[i][0] for i in fit)/len(fit)
TAUS=[i/100 for i in range(0,101,2)]+[0.99,0.995,1.01]
best=None
for t in TAUS:
    a,d,p50,p95,m=run(fit,t)
    if a>=fit_ver and (best is None or m<best[2]): best=(t,a,m,d)
print(f"\n  fit half scoring-alone acc {fit_ver:.4f}")
if best:
    t=best[0]; a,d,p50,p95,m=run(ev,t)
    se=math.sqrt(a*(1-a)/len(ev))
    print(f"  selected tau={t:.3f} on fit half (fit acc {best[1]:.4f}, defer {best[3]:.3f})")
    print(f"  HELD OUT: acc {a:.4f} (SE {se:.4f})  delta {a-ver_ev:+.4f}  defer {d:.3f}")
    print(f"            p50 {p50:.3f}s ({pct(vlat,.5)/p50:.2f}x)  p95 {p95:.3f}s ({pct(vlat,.95)/p95:.2f}x)  mean {m:.3f}s ({sum(vlat)/len(vlat)/m:.2f}x)")
else:
    print("  no tau on the fit half reaches scoring-alone accuracy")
print(f"\n  full eval curve:")
print(f"    {'tau':>6} {'acc':>7} {'delta':>8} {'defer':>7} {'p50':>7} {'p95':>7} {'mean':>7} {'mean-x':>7}")
for t in [0.0,0.2,0.4,0.6,0.8,0.9,0.95,0.99,1.01]:
    a,d,p50,p95,m=run(ev,t)
    print(f"    {t:6.3f} {a:7.4f} {a-ver_ev:+8.4f} {d:7.3f} {p50:7.3f} {p95:7.3f} {m:7.3f} {sum(vlat)/len(vlat)/m:6.2f}x")
