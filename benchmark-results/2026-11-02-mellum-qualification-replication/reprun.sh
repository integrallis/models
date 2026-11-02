set -e
export HOME=/root
export OLLAMA_MODELS=/root/ollama
export THREADS=16 CONTEXT=2048 MAXTOK=256 COMPARATOR=ollama JAVA_BIN=/root/jdk/bin/java
export KERNEL=/root/libkernels-fixed.so BACKEND_VERSION=q4kfix-dcd8b172
ART=/root/mellum/m.gguf
MID=mellum2_12b_a2_5b_thinking_q4_k_m
# Same protocol the 2026-11-01 replication used.
for n in 1 2 3; do
  echo "######## PAIRED RUN $n of 3 ########"
  bash /root/bench/fleet/two-arm-pod.sh "$ART" "$MID" chatml-no-think coding 1 /root/rep$n 2>&1 | tail -12
done
echo
echo "######## REPLICATED RESULT ########"
python3 - <<'EOP'
import json, statistics
rows=[]
for n in (1,2,3):
    try:
        v=json.load(open(f'/root/rep{n}/verdict.json'))['qualification']
        c=v['comparisons'][0]
        cand=json.load(open(f'/root/rep{n}/candidate.json'))['summary']['p50DecodeTokensPerSecond']
        comp=json.load(open(f'/root/rep{n}/comparator.json'))['summary']['p50DecodeTokensPerSecond']
        rows.append((n,cand,comp,c['decodeThroughputRatio'],v['verdict']))
    except Exception as e:
        rows.append((n,None,None,None,f'ERROR {e}'))
print(f"{'run':>4} {'ours':>8} {'ollama':>8} {'ratio':>8}  verdict")
for n,cand,comp,r,verd in rows:
    if r is None: print(f"{n:>4} {'-':>8} {'-':>8} {'-':>8}  {verd}"); continue
    print(f"{n:>4} {cand:8.2f} {comp:8.2f} {r:8.4f}  {verd}")
good=[x for x in rows if x[3] is not None]
if len(good)>=2:
    ratios=[x[3] for x in good]; ours=[x[1] for x in good]; comps=[x[2] for x in good]
    m=statistics.mean(ratios)
    sd=statistics.stdev(ratios) if len(ratios)>1 else 0.0
    print()
    print(f"  ours      mean {statistics.mean(ours):.2f} sd {statistics.stdev(ours) if len(ours)>1 else 0:.2f}")
    print(f"  ollama    mean {statistics.mean(comps):.2f} sd {statistics.stdev(comps) if len(comps)>1 else 0:.2f}")
    print(f"  RATIO     mean {m:.4f} sd {sd:.4f}   floor 0.80")
    print(f"  passes    {sum(1 for x in good if x[3]>=0.80)} of {len(good)}")
    print(f"  DECISION  {'EARNED - mean clears the floor' if m>=0.80 else 'NOT EARNED - entry must be withdrawn'}")
EOP
echo "######## DONE ########"
