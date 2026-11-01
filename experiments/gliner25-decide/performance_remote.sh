#!/usr/bin/env bash
# Complete the registered matrix in both arms; never report partial buckets as a full pair.
set -euo pipefail
trap 'touch /work/STOP' EXIT
cd /work/payload/experiment
export PATH=/work/jdk/bin:/root/.cargo/bin:$PATH TOKENIZERS_PARALLELISM=false
aws s3 cp "$GLINER_URI/evaluation-overlay.tar.gz" /work/evaluation-overlay.tar.gz --only-show-errors
printf '%s  %s\n' "$GLINER_EVAL_OVERLAY_SHA" /work/evaluation-overlay.tar.gz | sha256sum -c -
tar -xzf /work/evaluation-overlay.tar.gz -C /work/payload/experiment
bash evaluate_finetune_remote.sh --parity-only
bash download_reranker_fixture.sh /work/reranker
java --add-modules jdk.incubator.vector -Xmx3g -Dmodels.deberta.threads=8 -cp '/work/payload/classes:/work/payload/lib/*' RerankerRegression /work/reranker > /work/out/reranker-regression.json
/work/python/bin/python run_worker.py --cases performance-cases.jsonl --output /work/out/performance-reference.jsonl -- /work/python/bin/python reference_worker.py --snapshot /work/variant --manifest /work/variant-manifest.json --source /work/reference --threads 8
/work/python/bin/python run_worker.py --cases performance-cases.jsonl --output /work/out/performance-models.jsonl -- java --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED -Xmx8g -Dmodels.deberta.threads=8 -Djdk.virtualThreadScheduler.parallelism=8 -cp '/work/payload/classes:/work/payload/lib/*' GlinerWorker /work/variant /work/payload/rust/target/release/libjmodels_kernels.so /work/variant-manifest.json
/work/python/bin/python compare_performance.py --cases performance-cases.jsonl --candidate /work/out/performance-models.jsonl --reference /work/out/performance-reference.jsonl --output /work/out/performance-report.json
