"""Side-by-side summary. Accuracy is scored against the contract, not against each other.

The disclosure at the bottom is part of the output rather than part of the README, because the
README does not appear in the recording and the thing being disclosed is the reason a viewer should
not read the wall-clock column as a like-for-like hardware comparison. We do not know what Jev runs
on. Printing that next to the numbers is the only way the numbers are honest on their own.
"""
import json
import sys

jev = json.load(open(sys.argv[1]))
ours = json.load(open(sys.argv[2]))
gold = [int(x) for x in sys.argv[3].split()]


def hits(probs):
    return sum(1 for p, g in zip(probs, gold) if (p >= 0.5) == bool(g))


jev_hits = hits(jev["probabilities"])
our_probs = ours.get("probabilities")
our_hits = hits(our_probs) if our_probs else None
G, Y, R, D, B = "\033[1;32m", "\033[1;33m", "\033[1;31m", "\033[2m", "\033[0m"

jev_marginal = sum(jev["latency_s"]) / len(jev["latency_s"])
our_marginal = sum(ours["marginal_s"]) / len(ours["marginal_s"])
# Both arms' comparable total: everything it costs to answer ten questions from a cold start.
our_total = ours["prefill_s"] + sum(ours["marginal_s"])


def row(label, left, right, winner=None):
    lc = G if winner == "them" else B
    rc = G if winner == "us" else B
    print(f"  {label:28}{Y if winner == 'them' else lc}{left:>18}{B}{rc}{right:>18}{B}")


print()
print(f"  {'':28}{Y}{'TypeSafe Jev':>18}{B}{G}{'Harriet':>18}{B}")
print(f"  {'─' * 64}")
row("correct of 10", str(jev_hits), str(our_hits), "them" if jev_hits > our_hits else "us")
row("total, cold start", f"{jev['total_s']:.2f}s", f"{our_total:.2f}s", "them")
row("one-off prefill", "n/a", f"{ours['prefill_s']:.2f}s")
row("marginal per question", f"{jev_marginal:.3f}s", f"{our_marginal:.3f}s", "them")
print(f"  {'─' * 64}")
row("document sent to a server", f"{jev['questions']}x", "0", "us")
row("input tokens billed", str(jev["input_tokens_billed"]), "0", "us")
row("runs offline", "no", "yes", "us")
row("prefix physically shared", "n/a", str(ours["prefix_shared"]))
row("model", jev.get("model", "?"), ours.get("model", "?"))
row("kernel", "hosted", ours["kernel"])

print()
print(f"  {R}We lose on both measured axes: three answers and {our_total / jev['total_s']:.1f}x on wall clock.{B}")
print()
print(f"  {D}DISCLOSURE{B}")
print(f"  {D}We do not know what hardware Jev runs on. Not the device, not how many, not{B}")
print(f"  {D}whether requests were batched. Its timings include the network round trip from{B}")
print(f"  {D}this machine, so they flatter it on network and penalise it on nothing.{B}")
print(f"  {D}Harriet's host is stated and it is a CPU: {ours.get('host', 'unstated')}.{B}")
print(f"  {D}Read the wall-clock column as two systems on unknown-vs-known hardware, not as{B}")
print(f"  {D}a like-for-like comparison. The columns below the line are the ones that do not{B}")
print(f"  {D}depend on whose machine is bigger.{B}")
print()
print(f"  {D}Ten questions about one contract, gold labels written by us. A demo, not a{B}")
print(f"  {D}benchmark -- the measured benchmark position is JevBench, in ../2026-09-21-jevbench-baseline.{B}")
print()
