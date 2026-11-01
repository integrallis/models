#!/usr/bin/env bash
# CPU cost attribution only; these shortened cases are not qualification evidence.
set -euo pipefail
trap 'touch /work/STOP' EXIT
cd /work/payload/experiment
export PATH=/work/jdk/bin:/root/.cargo/bin:$PATH TOKENIZERS_PARALLELISM=false
/work/python/bin/python - <<'PY'
import collections,json,pathlib
counts=collections.Counter(); chosen=[]
for line in pathlib.Path('performance-cases.jsonl').read_text().splitlines():
    row=json.loads(line)
    if counts[row['bucket']] < 8: chosen.append(row)
    counts[row['bucket']]+=1
pathlib.Path('/work/out/profile-cases.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in chosen))
PY
sha256sum /work/payload/lib/*.jar /work/payload/rust/target/release/libjmodels_kernels.so > /work/out/runtime-sha256.txt
/work/python/bin/python run_worker.py --cases /work/out/profile-cases.jsonl --output /work/out/profile-models.jsonl -- java -Xlog:jfr*=off -XX:StartFlightRecording=filename=/work/out/deberta-profile.jfr,settings=profile,dumponexit=true --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED -Xmx8g -Dmodels.deberta.threads=8 -Djdk.virtualThreadScheduler.parallelism=8 -cp '/work/payload/classes:/work/payload/lib/*' GlinerWorker /work/snapshot /work/payload/rust/target/release/libjmodels_kernels.so snapshot.json
jfr print --json --events jdk.ExecutionSample,jdk.NativeMethodSample,jdk.GarbageCollection /work/out/deberta-profile.jfr > /work/out/deberta-profile.json
