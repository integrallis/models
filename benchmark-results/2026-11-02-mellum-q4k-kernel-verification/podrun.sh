set -e
export HOME=/root
export OLLAMA_MODELS=/root/ollama
ART=/root/mellum/m.gguf
MID=mellum2_12b_a2_5b_thinking_q4_k_m
# Recorded config for mellum, not guessed: benchmark-results/2026-11-01-mellum-final-qualification
# ran wl=coding tpl=chatml-no-think topk=1.
TPL=chatml-no-think
WL=coding
TOPK=1
export THREADS=16 CONTEXT=2048 MAXTOK=256 COMPARATOR=ollama JAVA_BIN=/root/jdk/bin/java
echo "######## ARM SET 1: MAIN kernel (baseline on this box) ########"
KERNEL=/root/libkernels-main.so BACKEND_VERSION=main-33700300 \
  bash /root/bench/fleet/two-arm-pod.sh "$ART" "$MID" "$TPL" "$WL" "$TOPK" /root/out-main 2>&1 | tail -30
echo
echo "######## ARM SET 2: FIXED kernel (Q4_K per-block unpack) ########"
KERNEL=/root/libkernels-fixed.so BACKEND_VERSION=q4kfix-760f7776 \
  bash /root/bench/fleet/two-arm-pod.sh "$ART" "$MID" "$TPL" "$WL" "$TOPK" /root/out-fixed 2>&1 | tail -30
echo
echo "######## SUMMARY ########"
python3 - <<'EOP'
import json
for tag in ('main','fixed'):
    try:
        c=json.load(open(f'/root/out-{tag}/candidate.json'))['summary']
        o=json.load(open(f'/root/out-{tag}/comparator.json'))['summary']
        print(f"  {tag:6s} candidate decode {c['p50DecodeTokensPerSecond']:.2f}  prefill {c['p50PrefillTokensPerSecond']:.2f}"
              f"   | comparator decode {o['p50DecodeTokensPerSecond']:.2f}"
              f"   | ratio {c['p50DecodeTokensPerSecond']/o['p50DecodeTokensPerSecond']:.4f}")
    except Exception as e:
        print(f"  {tag}: {e}")
EOP
for tag in main fixed; do echo "--- verdict ($tag) ---"; cat /root/out-$tag/verdict.log 2>/dev/null | tail -12; done
