"""Fit one temperature on the held-out calibration split, apply it to the stored cohort."""
import json, math, sys

calib_tasks, calib_scores, cohort_results, out_path = sys.argv[1:5]

def retemper(p, T):
    z = {k: math.log(max(v, 1e-12)) / T for k, v in p.items()}
    m = max(z.values()); e = {k: math.exp(v - m) for k, v in z.items()}; s = sum(e.values())
    return {k: v / s for k, v in e.items()}

tasks = {json.loads(l)["id"]: json.loads(l) for l in open(calib_tasks)}
calib = []
for line in open(calib_scores):
    parts = line.rstrip("\n").split("\t")
    probs = {}
    for kv in parts[2:]:
        if kv:
            k, v = kv.rsplit("=", 1); probs[k] = float(v)
    t = tasks[parts[0]]
    if t.get("expected") is not None and probs:
        calib.append((probs, str(t["expected"])))
print(f"calibration items usable: {len(calib)}")

def nll(T):
    total = 0.0
    for p, gold in calib:
        q = retemper(p, T)
        total -= math.log(max(q.get(gold, 1e-12), 1e-12))
    return total / len(calib)

# Ternary search over log T, same shape as the Java fitter, so the result is comparable.
lo, hi = math.log(1/64), math.log(64)
for _ in range(200):
    a, b = lo + (hi - lo)/3, hi - (hi - lo)/3
    if nll(math.exp(a)) < nll(math.exp(b)): hi = b
    else: lo = a
T = math.exp((lo + hi) / 2)
print(f"fitted temperature on held-out split: T = {T:.4f}  (NLL {nll(T):.4f}, vs {nll(1.0):.4f} at T=1)")

n = 0
with open(out_path, "w") as out:
    for line in open(cohort_results):
        r = json.loads(line)
        if r.get("probs"):
            r["probs"] = retemper(r["probs"], T)
            r["probs_source"] = f"logprob_then_temperature_scaled_heldout_T={T:.4f}"
            r["predicted"] = max(r["probs"], key=lambda k: r["probs"][k])
        out.write(json.dumps(r) + "\n"); n += 1
print(f"re-tempered {n} cohort results -> {out_path}")
