"""Side-by-side summary. Accuracy is scored against the contract, not against each other."""
import json, sys

jev = json.load(open(sys.argv[1]))
ours = json.load(open(sys.argv[2]))
gold = [int(x) for x in sys.argv[3].split()]

def hits(probs):
    return sum(1 for p, g in zip(probs, gold) if (p >= 0.5) == bool(g))

jev_hits = hits(jev["probabilities"])
our_probs = ours.get("probabilities")
G, Y, R, B = "\033[1;32m", "\033[1;33m", "\033[1;31m", "\033[0m"

print()
print(f"  {'':28}{Y}{'TypeSafe Jev':>18}{B}{G}{'Integrallis':>18}{B}")
print(f"  {'─'*64}")
print(f"  {'correct of 10':28}{jev_hits:>18}"
      + (f"{hits(our_probs):>18}" if our_probs else f"{'see above':>18}"))
print(f"  {'total wall time':28}{jev['total_s']:>17.2f}s{ours['total_s']:>17.2f}s")
print(f"  {'one-off prefill':28}{'n/a':>18}{ours['prefill_s']:>17.2f}s")
m = ours["marginal_s"]
print(f"  {'marginal per question':28}{sum(jev['latency_s'])/len(jev['latency_s']):>17.3f}s"
      f"{sum(m)/len(m):>17.3f}s")
print(f"  {'document sent to a server':28}{jev['questions']:>15}x{'0':>18}")
print(f"  {'input tokens billed':28}{jev['input_tokens_billed']:>18}{'0':>18}")
print(f"  {'prefix physically shared':28}{'n/a':>18}{str(ours['prefix_shared']):>18}")
print(f"  {'runs offline':28}{'no':>18}{'yes':>18}")
print(f"  {'kernel':28}{'hosted':>18}{ours['kernel']:>18}")
print()
