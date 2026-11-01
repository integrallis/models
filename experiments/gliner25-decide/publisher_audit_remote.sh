#!/usr/bin/env bash
# Accuracy only: concurrent workers save rental time; their timings are not performance evidence.
set -euo pipefail
trap 'touch /work/STOP' EXIT
cd /work/payload/experiment
export PATH=/work/jdk/bin:/root/.cargo/bin:$PATH TOKENIZERS_PARALLELISM=false
aws s3 cp "$GLINER_URI/evaluation-overlay.tar.gz" /work/evaluation-overlay.tar.gz --only-show-errors
printf '%s  %s\n' "$GLINER_EVAL_OVERLAY_SHA" /work/evaluation-overlay.tar.gz | sha256sum -c -
tar -xzf /work/evaluation-overlay.tar.gz -C /work/payload/experiment
bash evaluate_finetune_remote.sh --parity-only
cp publisher-audit-manifest.json /work/out/
suites=(publisher go-full)
if [[ "${GLINER_SINGLE_HEAD:-0}" == 1 ]]; then suites=(publisher); fi
for suite in "${suites[@]}"; do
  if [[ "$suite" == publisher ]]; then cases=publisher-development-cases.jsonl; else cases=go-full-test-cases.jsonl; fi
  /work/python/bin/python run_worker.py --cases "$cases" --output "/work/out/$suite-reference.jsonl" -- /work/python/bin/python reference_worker.py --snapshot /work/variant --manifest /work/variant-manifest.json --source /work/reference --threads 8 --api classify-text &
  reference_pid=$!
  /work/python/bin/python run_worker.py --cases "$cases" --output "/work/out/$suite-models.jsonl" -- java --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED -Xmx8g -Dmodels.deberta.threads=8 -Djdk.virtualThreadScheduler.parallelism=8 -cp '/work/payload/classes:/work/payload/lib/*' GlinerWorker /work/variant /work/payload/rust/target/release/libjmodels_kernels.so /work/variant-manifest.json &
  models_pid=$!
  reference_status=0;models_status=0
  wait "$reference_pid" || reference_status=$?
  wait "$models_pid" || models_status=$?
  printf '%s %s %s\n' "$suite" "$reference_status" "$models_status" >> /work/out/audit-phase-exits.txt
  [[ "$reference_status" == 0 && "$models_status" == 0 ]] || exit 1
done
report_args=()
if [[ "${GLINER_SINGLE_HEAD:-0}" == 1 ]]; then report_args=(--publisher-only); fi
/work/python/bin/python report_publisher_audit.py "${report_args[@]}" --directory /work/out --output /work/out/publisher-and-emotion-audit.json
