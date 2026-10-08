#!/bin/bash
# A/B: llama.cpp Q4_K prefill with the 8x8 repacked-interleaved GEMM ON vs OFF.
#
# Why this is the measurement: with GGML_CPU_REPACK=OFF, llama.cpp's Q4_K matmul falls back to a
# per-output-row vec_dot loop over row-major weights -- structurally what
# compute_q4_k_batched_row_range_avx2 does. With it ON, weights are repacked to block_q4_Kx8 at
# load and matmul runs ggml_gemm_q4_K_8x8_q8_K, 8 output rows x 4 batch rows per pass. The delta is
# therefore an estimate, in their own code on one box, of what porting that tile is worth to us --
# before writing any Rust.
#
# tinyBLAS is irrelevant here and BLAS is compiled out of both builds: llamafile/sgemm.cpp
# implements no K-quant, so it never handles Q4_K in either arm.
#
# Arms are INTERLEAVED because this host is a thermally limited laptop; a first-all-ON then
# all-OFF ordering would attribute throttling to the flag.
set -u
LC=/Users/briansam-bodden/Code/llama.cpp
OUT="${OUT:-$(dirname "$0")/results}"
MODEL="${MODEL:?set MODEL to a Q4_K_M gguf}"
PROMPT="${PROMPT:-1024}"
THREADS="${THREADS:-6}"
REPS="${REPS:-5}"
ALTERNATIONS="${ALTERNATIONS:-3}"
mkdir -p "$OUT"
TAG=$(basename "$MODEL" .gguf)
for b in build-rp-on build-rp-off; do
  [ -x "$LC/$b/bin/llama-bench" ] || { echo "missing $LC/$b/bin/llama-bench"; exit 1; }
done
echo "model      : $TAG"
echo "prompt     : $PROMPT tokens, -n 0 (prefill only)"
echo "threads    : $THREADS   reps/run: $REPS   alternations: $ALTERNATIONS"
echo "host       : $(sysctl -n machdep.cpu.brand_string 2>/dev/null || uname -m)"
echo
for i in $(seq 1 "$ALTERNATIONS"); do
  for arm in on off; do
    f="$OUT/$TAG-p$PROMPT-$arm-$i.json"
    "$LC/build-rp-$arm/bin/llama-bench" -m "$MODEL" -p "$PROMPT" -n 0 -r "$REPS" \
      -t "$THREADS" -o json > "$f" 2>/dev/null
    v=$(python3 -c "import json,sys;d=json.load(open('$f'));print(f\"{d[0]['avg_ts']:.2f} +/- {d[0]['stddev_ts']:.2f}\")" 2>/dev/null || echo "PARSE FAIL")
    printf '  pass %d  repack=%-3s  %s tok/s\n' "$i" "$arm" "$v"
  done
done
echo
python3 - "$OUT" "$TAG" "$PROMPT" <<'PY'
import json, glob, os, statistics as st, sys
out, tag, prompt = sys.argv[1], sys.argv[2], sys.argv[3]
res={}
for arm in ("on","off"):
    vals=[]
    for f in sorted(glob.glob(os.path.join(out, f"{tag}-p{prompt}-{arm}-*.json"))):
        try: vals.append(json.load(open(f))[0]["avg_ts"])
        except Exception: pass
    res[arm]=vals
if not res["on"] or not res["off"]:
    sys.exit("incomplete arms")
mon, moff = st.median(res["on"]), st.median(res["off"])
print(f"  repack ON  (8x8 tile)  : {[round(v,1) for v in res['on']]}  median {mon:.2f} tok/s")
print(f"  repack OFF (vec_dot)   : {[round(v,1) for v in res['off']]}  median {moff:.2f} tok/s")
print(f"  speedup from the tile  : {mon/moff:.2f}x")
spread = lambda v: (max(v)-min(v))/st.median(v)
print(f"  within-arm spread      : ON {spread(res['on'])*100:.1f}%   OFF {spread(res['off'])*100:.1f}%")
if mon/moff - 1 < max(spread(res['on']), spread(res['off'])):
    print("  VERDICT: effect is NOT larger than run-to-run spread on this host -- inconclusive here.")
else:
    print("  VERDICT: effect exceeds run-to-run spread on this host.")
PY
