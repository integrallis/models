#!/usr/bin/env bash
# Full released accuracy repeat: six concurrent workers on a dedicated 64-vCPU host.
# These timings are deliberately excluded from performance qualification.
set -euo pipefail
trap 'touch /work/STOP' EXIT
cd /work/payload/experiment
export PATH=/work/jdk/bin:/root/.cargo/bin:$PATH TOKENIZERS_PARALLELISM=false
[[ -f /work/payload/released-runtime.json ]]
aws s3 cp "$GLINER_URI/evaluation-overlay.tar.gz" /work/evaluation-overlay.tar.gz --only-show-errors
printf '%s  %s\n' "$GLINER_EVAL_OVERLAY_SHA" /work/evaluation-overlay.tar.gz | sha256sum -c -
tar -xzf /work/evaluation-overlay.tar.gz -C /work/payload/experiment
bash evaluate_finetune_remote.sh --parity-only
cp publisher-audit-manifest.json publisher-single-head-manifest.json /work/out/
java --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED -Xmx8g -Dmodels.deberta.threads=8 -cp '/work/payload/classes:/work/payload/lib/*' ReleasedGlinerSmoke /work/variant > /work/out/released-default-discovery.json
pids=(); names=()
for suite in publisher publisher-single go-full; do
  case "$suite" in
    publisher) cases=publisher-development-cases.jsonl;;
    publisher-single) cases=publisher-single-head-cases.jsonl;;
    go-full) cases=go-full-test-cases.jsonl;;
  esac
  /work/python/bin/python run_worker.py --cases "$cases" --output "/work/out/$suite-reference.jsonl" -- /work/python/bin/python reference_worker.py --snapshot /work/variant --manifest /work/variant-manifest.json --source /work/reference --threads 8 --api classify-text &
  pids+=("$!"); names+=("$suite-reference")
  /work/python/bin/python run_worker.py --cases "$cases" --output "/work/out/$suite-models.jsonl" -- java --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED -Xmx8g -Dmodels.deberta.threads=8 -Djdk.virtualThreadScheduler.parallelism=8 -cp '/work/payload/classes:/work/payload/lib/*' GlinerWorker /work/variant /work/payload/rust/target/release/libjmodels_kernels.so /work/variant-manifest.json &
  pids+=("$!"); names+=("$suite-models")
done
failed=0
for i in "${!pids[@]}"; do
  status=0
  wait "${pids[$i]}" || status=$?
  printf '%s %s\n' "${names[$i]}" "$status" >> /work/out/audit-phase-exits.txt
  if [[ "$status" != 0 ]]; then failed=1; fi
done
exit "$failed"
