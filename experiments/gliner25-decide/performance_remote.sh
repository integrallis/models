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
if [[ -f /work/payload/released-runtime.json ]]; then
  java --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED -Xmx8g -Dmodels.deberta.threads=8 -cp '/work/payload/classes:/work/payload/lib/*' ReleasedGlinerSmoke /work/variant > /work/out/released-default-discovery.json
fi
bash download_reranker_fixture.sh /work/reranker
java --add-modules jdk.incubator.vector -Xmx3g -Dmodels.deberta.threads=8 -cp '/work/payload/classes:/work/payload/lib/*' RerankerRegression /work/reranker > /work/out/reranker-regression.json
/work/python/bin/python run_performance_pairs.py --replications "${GLINER_REPLICATIONS:-1}" --output /work/out
