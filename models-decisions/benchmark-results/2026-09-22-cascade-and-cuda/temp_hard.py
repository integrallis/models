"""Does a held-out temperature move the hard-tier Calibration axis? Post-processing only."""
import json, math, sys, hashlib
BENCH = "/Users/briansam-bodden/Code/java-ai/references/openjev/gh-jevbench"
sys.path.insert(0, BENCH)
from jevbench import composite_v12 as c
from jevbench import metrics as m

def load_gold():
    g = {}
    for tier, f in (("easy","easy"),("standard","original"),("hard","hard")):
        for line in open(f"{BENCH}/datasets/public/{f}.jsonl"):
            t = json.loads(line); g[t["id"]] = (str(t["expected"]), tier)
    return g
def load(path):
    r = {}
    for line in open(path):
        p = line.rstrip("\n").split("\t"); pr = {}
        for kv in p[2:]:
            if kv: k,v = kv.rsplit("=",1); pr[k]=float(v)
        r[p[0]] = (float(p[1]), pr)
    return r
def argmax(p): return max(p.items(), key=lambda kv: kv[1])[0]
def retemper(p, T):
    z = {k: math.log(max(v,1e-12))/T for k,v in p.items()}
    mx = max(z.values()); e = {k: math.exp(v-mx) for k,v in z.items()}; s = sum(e.values())
    return {k: v/s for k,v in e.items()}
def ece_of(items, res, gold, T):
    pairs = [(max(retemper(res[i][1],T).values()), argmax(res[i][1])==gold[i][0]) for i in items]
    return m.ece_top_label(pairs)["ece"]

gold = load_gold()
for path in sys.argv[1:]:
    res = load(path)
    hard = sorted(i for i in set(gold)&set(res) if gold[i][1]=="hard")
    # Stratified halves so the temperature is never fitted on what reports it.
    fit = [i for i in hard if int(hashlib.md5(i.encode()).hexdigest(),16)%2==0]
    ev  = [i for i in hard if i not in set(fit)]
    def nll(T):
        tot=0.0
        for i in fit:
            q=retemper(res[i][1],T); tot -= math.log(max(q.get(gold[i][0],1e-12),1e-12))
        return tot/len(fit)
    lo,hi = math.log(1/64), math.log(64)
    for _ in range(200):
        a,b = lo+(hi-lo)/3, hi-(hi-lo)/3
        if nll(math.exp(a)) < nll(math.exp(b)): hi=b
        else: lo=a
    T = math.exp((lo+hi)/2)
    e1, e2 = ece_of(ev,res,gold,1.0), ece_of(ev,res,gold,T)
    print(f"{path}  hard n={len(hard)} (fit {len(fit)} / eval {len(ev)})  T={T:.4f}")
    print(f"   held-out hard ECE  {e1:.4f} -> {e2:.4f}"
          f"   Calibration {c.calibration(e1,None):.1f} -> {c.calibration(e2,None):.1f}")
    acc = sum(1 for i in ev if argmax(res[i][1])==gold[i][0])/len(ev)
    print(f"   (accuracy unchanged at {acc:.4f}; temperature is monotone, it cannot move argmax)")
