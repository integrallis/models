"""Upper bound: what would the cascade give if the proposer were FREE (shared prefill)?
If even a free proposer cannot hold accuracy, the idea dies regardless of engineering."""
import json, math, hashlib
BENCH = "/Users/briansam-bodden/Code/java-ai/references/openjev/gh-jevbench/datasets/public"
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
prop=load_arm("/tmp/cascade/qwen-scores.tsv")
ver =load_arm("/tmp/cascade/granite-scores.tsv")
ids=sorted(set(gold)&set(prop)&set(ver))
fit,ev=[],[]
for i in ids:
    h=int(hashlib.md5((gold[i][1]+gold[i][2]+i).encode()).hexdigest(),16)
    (fit if h%2==0 else ev).append(i)

def run(items,tau,prop_cost):
    c=0;d=0;lat=[]
    for i in items:
        v=sorted(prop[i][1].values(),reverse=True)
        s=v[0]-(v[1] if len(v)>1 else 0)
        if s>=tau:
            c+=argmax(prop[i][1])==gold[i][0]; lat.append(prop[i][0]*prop_cost)
        else:
            d+=1; c+=argmax(ver[i][1])==gold[i][0]; lat.append(prop[i][0]*prop_cost+ver[i][0])
    n=len(items)
    return c/n,d/n,pct(lat,.5),pct(lat,.95),sum(lat)/n

ver_ev=sum(argmax(ver[i][1])==gold[i][0] for i in ev)/len(ev)
vlat=[ver[i][0] for i in ev]
print(f"eval n={len(ev)}  verifier alone: acc {ver_ev:.4f}  p50 {pct(vlat,.5):.3f}  p95 {pct(vlat,.95):.3f}  mean {sum(vlat)/len(vlat):.3f}")
print(f"\nproposer cost multiplier 1.0 (real 0.6B pass) vs 0.0 (hypothetical free/shared-prefill head)")
for mult,label in ((1.0,"REAL"),(0.0,"FREE")):
    print(f"\n  --- proposer cost = {label} ---")
    print(f"    {'tau':>6} {'acc':>7} {'d(acc)':>8} {'defer':>7} {'p50':>7} {'p95':>8} {'mean':>7} {'speedup':>8}")
    for t in [0.0,0.3,0.5,0.7,0.8,0.9,0.95,0.99,1.01]:
        a,d,p50,p95,m=run(ev,t,mult)
        print(f"    {t:6.3f} {a:7.4f} {a-ver_ev:+8.4f} {d:7.3f} {p50:7.3f} {p95:8.3f} {m:7.3f} {(sum(vlat)/len(vlat)/m if m>0 else float("inf")):8.2f}x")
n=len(ev)
print(f"\n  binomial SE at these accuracies: +/-{math.sqrt(0.65*0.35/n):.4f}  (n={n})")
