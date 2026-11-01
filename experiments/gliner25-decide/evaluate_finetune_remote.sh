#!/usr/bin/env bash
# Runs only on the disposable CPU qualification worker after its pinned bootstrap.
set -euo pipefail
case "${1:-}" in
  ''|--parity-only) ;;
  *) echo 'Unsupported evaluation phase' >&2; exit 2 ;;
esac
trap 'touch /work/STOP' EXIT
cd /work/payload/experiment
export PATH=/work/jdk/bin:/root/.cargo/bin:$PATH TOKENIZERS_PARALLELISM=false
aws s3 cp "$GLINER_URI/evaluation-overlay.tar.gz" /work/evaluation-overlay.tar.gz --only-show-errors
printf '%s  %s\n' "$GLINER_EVAL_OVERLAY_SHA" /work/evaluation-overlay.tar.gz | sha256sum -c -
tar -xzf /work/evaluation-overlay.tar.gz -C /work/payload/experiment
aws s3 cp "$GLINER_TRAINED_RESULTS/snapshot.json" /work/variant-manifest.json --only-show-errors
/work/python/bin/python - <<'PY'
import json,os,pathlib,subprocess,sys
sys.path.insert(0,'/work/payload/experiment')
from common import verify_snapshot
manifest=json.load(open('/work/variant-manifest.json'))
assert manifest['id']==json.load(open('/work/payload/experiment/finetune-policy.json'))['candidate']
for entry in manifest['files']:
    path=pathlib.Path('/work/variant')/entry['path'];path.parent.mkdir(parents=True,exist_ok=True)
    subprocess.run(['aws','s3','cp',os.environ['GLINER_TRAINED_RESULTS']+'/snapshot/'+entry['path'],str(path),'--only-show-errors'],check=True)
verify_snapshot('/work/variant','/work/variant-manifest.json')
PY
cp /work/variant-manifest.json /work/out/variant-manifest.json
sha256sum /work/payload/lib/*.jar /work/payload/rust/target/release/libjmodels_kernels.so > /work/out/runtime-sha256.txt
reference() {
  local cases=$1 output=$2 snapshot=$3 manifest=$4
  /work/python/bin/python run_worker.py --cases "$cases" --output "/work/out/$output.jsonl" -- /work/python/bin/python reference_worker.py --snapshot "$snapshot" --manifest "$manifest" --source /work/reference --threads 8
}
models() {
  local cases=$1 output=$2
  /work/python/bin/python run_worker.py --cases "$cases" --output "/work/out/$output.jsonl" -- java --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED -Xmx8g -Dmodels.deberta.threads=8 -Djdk.virtualThreadScheduler.parallelism=8 -cp '/work/payload/classes:/work/payload/lib/*' GlinerWorker /work/variant /work/payload/rust/target/release/libjmodels_kernels.so /work/variant-manifest.json
}
reference parity-cases.jsonl variant-parity-reference /work/variant /work/variant-manifest.json
models parity-cases.jsonl variant-parity-models
/work/python/bin/python compare_parity.py --cases parity-cases.jsonl --candidate /work/out/variant-parity-models.jsonl --reference /work/out/variant-parity-reference.jsonl --output /work/out/variant-parity-report.json
reference performance-parity-cases.jsonl variant-shape-reference /work/variant /work/variant-manifest.json
models performance-parity-cases.jsonl variant-shape-models
/work/python/bin/python compare_parity.py --cases performance-parity-cases.jsonl --candidate /work/out/variant-shape-models.jsonl --reference /work/out/variant-shape-reference.jsonl --output /work/out/variant-shape-report.json
if [[ "${1:-}" == --parity-only ]]; then exit 0; fi
reference finetune-qualification-cases.jsonl fresh-base /work/snapshot snapshot.json
reference finetune-qualification-cases.jsonl fresh-reference /work/variant /work/variant-manifest.json
models finetune-qualification-cases.jsonl fresh-models
models quality-cases.jsonl original-regression-models
/work/python/bin/python assess_finetune.py --candidate /work/out/fresh-models.jsonl --reference /work/out/fresh-reference.jsonl --base /work/out/fresh-base.jsonl --regression /work/out/original-regression-models.jsonl --output /work/out/fine-tuned-quality-report.json
