"""Score one of our result TSVs on JevBench v1.2 by calling the harness's own functions.

Three numbers in this project have been assembled by hand and been wrong: Speed from a mean
instead of p50/p95, a Cost borrowed from another system, and an Intelligence taken from one model
with the latency of another. This imports jevbench.composite_v12 so none of the formulas are
retyped, and it prints what the public data cannot supply rather than filling it in.

usage: jevscore.py RESULTS.tsv TOKENS_PER_DECISION USD_PER_M_IN [endpoint_kind] [--calibrate]

Cost follows the benchmark's own rule for self-hosted entrants, which is NOT hourly rate times
wall time. Every self-hosted row in results/v1.2/jevbench-v1.2-results.json is priced at a
hosted-provider list price for the same size class times measured input tokens per decision,
e.g. "DeepInfra Qwen3-Embedding-0.6B size-class reference list price $0.01/M in, $0.0/M out
(a <=0.6B one-pass model with no generated output) x 279 input and 0 output tokens per decision".
Pricing our runs by our host's hourly rate gave Cost 7.1 where their rule gives about 75, so the
rule matters more than any measurement it consumes.
"""
import json, math, sys, os, hashlib

BENCH = "/Users/briansam-bodden/Code/java-ai/references/openjev/gh-jevbench"
sys.path.insert(0, BENCH)
from jevbench import composite_v12 as c  # noqa: E402
from jevbench import metrics as m  # noqa: E402

TIER_FILE = {"easy": "easy", "standard": "original", "hard": "hard"}

def load_gold():
    g = {}
    for tier, fname in TIER_FILE.items():
        for line in open(f"{BENCH}/datasets/public/{fname}.jsonl"):
            t = json.loads(line)
            g[t["id"]] = (str(t["expected"]), tier, t["question"]["type"])
    return g

def load_results(path):
    r = {}
    for line in open(path):
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 3:
            continue
        probs = {}
        for kv in parts[2:]:
            if kv:
                k, v = kv.rsplit("=", 1)
                probs[k] = float(v)
        r[parts[0]] = (float(parts[1]), probs)
    return r

def argmax(p):
    return max(p.items(), key=lambda kv: kv[1])[0]

def ece(items, res, gold):
    """Harness's own top-label ECE, so the binning rule is not retyped."""
    pairs = [(max(res[i][1].values()), argmax(res[i][1]) == gold[i][0]) for i in items]
    return m.ece_top_label(pairs)["ece"] if pairs else None

def retemper(p, T):
    z = {k: math.log(max(v, 1e-12)) / T for k, v in p.items()}
    mx = max(z.values())
    e = {k: math.exp(v - mx) for k, v in z.items()}
    t = sum(e.values())
    return {k: v / t for k, v in e.items()}

def fit_temperature(items, res, gold):
    def nll(T):
        return -sum(math.log(max(retemper(res[i][1], T).get(gold[i][0], 1e-12), 1e-12))
                    for i in items) / len(items)
    lo, hi = math.log(1 / 64), math.log(64)
    for _ in range(200):
        a, b = lo + (hi - lo) / 3, hi - (hi - lo) / 3
        if nll(math.exp(a)) < nll(math.exp(b)):
            hi = b
        else:
            lo = a
    return math.exp((lo + hi) / 2)

def ece_at(items, res, gold, T):
    pairs = [(max(retemper(res[i][1], T).values()), argmax(res[i][1]) == gold[i][0]) for i in items]
    return m.ece_top_label(pairs)["ece"] if pairs else None

def pct(xs, q):
    s = sorted(xs)
    return s[min(len(s) - 1, int(math.ceil(q * len(s))) - 1)]

def main():
    argv = [a for a in sys.argv[1:] if a != "--calibrate"]
    calibrate = "--calibrate" in sys.argv
    path, tokens_per_decision, usd_per_m_in = argv[0], float(argv[1]), float(argv[2])
    kind = argv[3] if len(argv) > 3 else "cpu"
    gold, res = load_gold(), load_results(path)
    ids = sorted(set(gold) & set(res))
    print(f"{os.path.basename(path)}: {len(ids)} scored of {len(gold)} public items, endpoint kind '{kind}'")

    tiers = {}
    for tier in TIER_FILE:
        sub = [i for i in ids if gold[i][1] == tier]
        tiers[tier] = (sum(1 for i in sub if argmax(res[i][1]) == gold[i][0]) / len(sub)) if sub else None
        if sub:
            print(f"  {tier:9s} n={len(sub):3d}  accuracy {tiers[tier]:.4f}")
    tiers["judge"] = None  # not in the public release; weight is renormalised away
    print("  judge     NOT IN PUBLIC DATA -> weight renormalised away by the harness")

    intel = c.intelligence(tiers)

    hard = [i for i in ids if gold[i][1] == "hard"]
    if calibrate:
        # Temperature fitted on a stratified half of the hard tier and reported on the other
        # half. Monotone, so accuracy and therefore Intelligence are provably untouched.
        fit = [i for i in hard if int(hashlib.md5(i.encode()).hexdigest(), 16) % 2 == 0]
        ev = [i for i in hard if i not in set(fit)]
        T = fit_temperature(fit, res, gold)
        hard_ece = ece_at(ev, res, gold, T)
        print(f"  calibrated: T={T:.4f} fitted on {len(fit)} hard items, ECE reported on the "
              f"held-out {len(ev)}")
    else:
        hard_ece = ece(hard, res, gold) if hard else None
    # The public sets carry no gold distributions, so the TVD half of Calibration is uncomputable.
    calib = c.calibration(hard_ece, None) if hard_ece is not None else None
    print(f"  hard-tier ECE {hard_ece:.4f}  -> Calibration {calib:.1f} (ECE half only; no gold "
          f"distributions in the public release, so the TVD half is missing)")

    # Speed is defined on the standard+judge run. Judge is unavailable, so standard alone.
    std = [i for i in ids if gold[i][1] == "standard"]
    lat = [res[i][0] for i in std]
    p50, p95 = pct(lat, 0.50), pct(lat, 0.95)
    spd = c.speed(p50, p95, kind)
    print(f"  standard-tier latency raw p50 {p50:.3f}s p95 {p95:.3f}s -> adjusted "
          f"{c.adjusted_latency(p50, kind):.3f}s / {c.adjusted_latency(p95, kind):.3f}s  Speed {spd:.1f}")

    usd = tokens_per_decision * 1000.0 * usd_per_m_in / 1e6
    cst = c.cost(usd)
    mean_latency = sum(res[i][0] for i in ids) / len(ids)
    print(f"  {tokens_per_decision:.0f} input tokens/decision at ${usd_per_m_in}/M in "
          f"-> ${usd:.5f}/1000 -> Cost {cst:.1f}")
    print(f"  (measured mean wall latency {mean_latency:.3f}s over {len(ids)} items, reported "
          f"for the record; it is not what the Cost axis consumes)")

    axes = {"intelligence": intel, "calibration": calib, "speed": spd, "cost": cst}
    print(f"\n  Intelligence {intel:.1f}   Calibration {calib:.1f}   Speed {spd:.1f}   Cost {cst:.1f}")
    print(f"  JevBench Score (25:25:25:25) = {c.geometric(axes):.1f}")
    print("\n  Caveat: judge tier and the TVD half of Calibration are absent from the public "
          "release, so this is not comparable to a published leaderboard row.")

if __name__ == "__main__":
    main()
