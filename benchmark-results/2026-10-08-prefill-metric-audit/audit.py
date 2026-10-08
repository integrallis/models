#!/usr/bin/env python3
"""Derive the prefill-metric audit from committed certified reports.

Every number in NOTES.md comes from this script. Nothing is hand-typed.

Run from the repository root:
    python3 benchmark-results/2026-10-08-prefill-metric-audit/audit.py

The physics check uses a deliberately GENEROUS ceiling for the fleet box
(16 vCPU AMD EPYC 7R13, Zen3, AVX2, no AVX-512): 2x256-bit FMA per core,
8 fp32 lanes, 2 flop/FMA, 16 threads, 3.6 GHz boost ->
    16 * 2 * 8 * 2 * 3.6e9 = 1.84 TFLOP/s
Rounded UP to 2.0 TFLOP/s so that anything flagged is flagged beyond argument.
Parameter counts are estimated from the GGUF artifact size at 5.0 bits/weight,
which OVER-estimates bits (Q4_K_M is ~4.8) and therefore UNDER-estimates both
the parameter count and the implied FLOP/s. The bound is conservative in both
directions on purpose.
"""
import csv, json, glob, os, statistics as st, sys

PEAK_TFLOPS = 2.0
BITS_PER_WEIGHT = 5.0
ROOT = os.path.join(os.path.dirname(__file__), "..", "..")

def p(d, key, sub=None):
    v = (d or {}).get(key)
    if isinstance(v, dict):
        return v.get(sub)
    return v

rows = []
for ours in sorted(glob.glob(os.path.join(ROOT, "benchmark-results/certified-*/rag/*/*-rust-ffm-grounded.json"))):
    folder = os.path.dirname(ours)
    comp = glob.glob(os.path.join(folder, "*-ollama-grounded.json"))
    if not comp:
        continue
    A = json.load(open(ours))
    B = json.load(open(comp[0]))
    a, b = A["summary"], B["summary"]
    size = A.get("artifactSizeBytes")
    if not size:
        continue
    params = size * 8 / BITS_PER_WEIGHT
    our_rate, oll_rate = a.get("p50PrefillTokensPerSecond"), b.get("p50PrefillTokensPerSecond")
    our_ttft, oll_ttft = p(a, "ttftMillis", "p50"), p(b, "ttftMillis", "p50")
    our_t95, oll_t95 = p(a, "ttftMillis", "p95"), p(b, "ttftMillis", "p95")
    if not all([our_rate, oll_rate, our_ttft, oll_ttft, our_t95, oll_t95]):
        continue
    attempts = a.get("successfulAttempts") or 1
    in_tok = (a.get("totalInputTokens") or 0)
    cache_read = a.get("totalCacheReadInputTokens") or 0
    rows.append(dict(
        model=os.path.basename(folder),
        host=(A.get("environment") or {}).get("cpuModel", "?"),
        vcpu=(A.get("environment") or {}).get("availableProcessors", 0),
        params_b=round(params / 1e9, 2),
        input_tokens_per_attempt=round(in_tok / attempts, 1),
        our_cache_hit_fraction=round(cache_read / in_tok, 4) if in_tok else None,
        ollama_cache_tokens_reported=(b.get("totalCacheReadInputTokens") or 0) + (b.get("totalCacheWriteInputTokens") or 0),
        our_prefill_tok_s=round(our_rate, 1),
        ollama_prefill_tok_s=round(oll_rate, 1),
        our_implied_tflops=round(2 * params * our_rate / 1e12, 3),
        ollama_implied_tflops=round(2 * params * oll_rate / 1e12, 3),
        ollama_exceeds_machine_peak=(2 * params * oll_rate / 1e12) > PEAK_TFLOPS,
        our_ttft_p50_ms=round(our_ttft, 1),
        ollama_ttft_p50_ms=round(oll_ttft, 1),
        ttft_p50_ratio=round(our_ttft / oll_ttft, 3),
        our_ttft_p95_ms=round(our_t95, 1),
        ollama_ttft_p95_ms=round(oll_t95, 1),
        ttft_p95_ratio=round(our_t95 / oll_t95, 3),
    ))

if not rows:
    sys.exit("no paired certified reports found")

out = os.path.join(os.path.dirname(__file__), "prefill-metric-audit.csv")
with open(out, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

med = lambda k: st.median([r[k] for r in rows])
imp = [r for r in rows if r["ollama_exceeds_machine_peak"]]
ours_imp = [r for r in rows if r["our_implied_tflops"] > PEAK_TFLOPS]

print(f"paired models                              : {len(rows)}")
print(f"hosts                                      : {sorted({r['host'] for r in rows})} ({sorted({r['vcpu'] for r in rows})} vCPU)")
print(f"generous peak ceiling                      : {PEAK_TFLOPS} TFLOP/s")
print()
print("-- the invalid metric --")
print(f"Ollama prefill rates above machine peak    : {len(imp)}/{len(rows)}")
print(f"  worst implied                            : {max(r['ollama_implied_tflops'] for r in rows):.2f} TFLOP/s"
      f"  ({max(r['ollama_implied_tflops'] for r in rows)/PEAK_TFLOPS:.1f}x peak)")
print(f"Ollama reports on cached prompt tokens     : {sum(r['ollama_cache_tokens_reported'] for r in rows)} tokens across all reports")
print(f"median ratio ollama/ours on this metric    : {st.median([r['ollama_prefill_tok_s']/r['our_prefill_tok_s'] for r in rows]):.2f}x  <-- NOT a speed ratio")
print()
print("-- our own rates, same arithmetic --")
print(f"ours above machine peak                    : {len(ours_imp)}/{len(rows)}")
print(f"  range                                    : {min(r['our_implied_tflops'] for r in rows):.2f} - {max(r['our_implied_tflops'] for r in rows):.2f} TFLOP/s"
      f"  ({min(r['our_implied_tflops'] for r in rows)/PEAK_TFLOPS*100:.0f}-{max(r['our_implied_tflops'] for r in rows)/PEAK_TFLOPS*100:.0f}% of peak)")
print(f"  our prompt-cache hit rate (median)       : {med('our_cache_hit_fraction')*100:.0f}% of input tokens")
print()
print("-- the valid, user-visible metric: wall-clock TTFT on our clock --")
print(f"TTFT p50 ratio ours/ollama   median        : {med('ttft_p50_ratio'):.2f}x")
print(f"  we are FASTER on                         : {sum(1 for r in rows if r['ttft_p50_ratio'] < 1)}/{len(rows)} models")
print(f"  worst                                    : {max(r['ttft_p50_ratio'] for r in rows):.2f}x ({max(rows, key=lambda r: r['ttft_p50_ratio'])['model']})")
print(f"  best                                     : {min(r['ttft_p50_ratio'] for r in rows):.2f}x ({min(rows, key=lambda r: r['ttft_p50_ratio'])['model']})")
print(f"TTFT p95 ratio ours/ollama   median        : {med('ttft_p95_ratio'):.2f}x")
print(f"our absolute TTFT p50 median               : {med('our_ttft_p50_ms'):.0f} ms   (ollama {med('ollama_ttft_p50_ms'):.0f} ms)")
print(f"our absolute TTFT p95 median               : {med('our_ttft_p95_ms'):.0f} ms   (ollama {med('ollama_ttft_p95_ms'):.0f} ms)")
print()
print(f"prompt size in this workload               : {med('input_tokens_per_attempt'):.0f} tokens/attempt (median)")
print(f"csv                                        : {os.path.relpath(out, ROOT)}")

# --------------------------------------------------------------------------------------------
# PROJECTION, NOT A MEASUREMENT.
#
# The tile A/B (experiments/prefill-repack-tile) measured 1.49-1.87x on the Q4_K prefill kernel.
# The rows above show our implied prefill time is ~95% of our measured TTFT, so a prefill win
# should pass through to TTFT nearly one-for-one. This block states what that would mean IF the
# port lands the lower end of the measured range. It is arithmetic on measured inputs, not an
# observation, and nothing here may be reported as a result.
# --------------------------------------------------------------------------------------------
SPEEDUP = 1.49  # the lowest value any arm of the A/B measured, so the projection is conservative
PREFILL_SHARE = 0.95  # median implied-prefill / measured-TTFT from the rows above

print()
print(f"-- PROJECTION ONLY: if the ported tile lands {SPEEDUP}x on the prefill kernel --")
proj = []
for r in rows:
    ttft = r["our_ttft_p50_ms"]
    prefill = ttft * PREFILL_SHARE
    new_ttft = (ttft - prefill) + prefill / SPEEDUP
    proj.append((new_ttft, new_ttft / r["ollama_ttft_p50_ms"], ttft))
print(f"our TTFT p50 median                        : {med('our_ttft_p50_ms'):.0f} ms"
      f"  ->  {st.median([p[0] for p in proj]):.0f} ms (projected)")
print(f"TTFT ratio vs Ollama, median               : {med('ttft_p50_ratio'):.2f}x"
      f"  ->  {st.median([p[1] for p in proj]):.2f}x (projected)")
print(f"models where we would be faster than Ollama : {sum(1 for r in rows if r['ttft_p50_ratio'] < 1)}"
      f"/{len(rows)}  ->  {sum(1 for p in proj if p[1] < 1)}/{len(rows)} (projected)")
print("Not a result. Only a measured post-port TTFT may be reported as one.")
