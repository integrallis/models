#!/usr/bin/env bash
# Isolated dense candidate: retain failed quality evidence and run at most one performance pair.
set -euo pipefail
trap 'touch /work/STOP' EXIT
cd /work/payload/experiment
export PATH=/work/jdk/bin:/root/.cargo/bin:$PATH TOKENIZERS_PARALLELISM=false
# The overlay is verified again by the shared quality runner.
aws s3 cp "$GLINER_URI/evaluation-overlay.tar.gz" /work/evaluation-overlay.tar.gz --only-show-errors
printf '%s  %s\n' "$GLINER_EVAL_OVERLAY_SHA" /work/evaluation-overlay.tar.gz | sha256sum -c -
tar -xzf /work/evaluation-overlay.tar.gz -C /work/payload/experiment
bash evaluate_finetune_remote.sh --parity-only
# Screen correctness and speed before repeating the larger quality suites. A losing speed
# candidate must not consume the remaining rental budget on already-known accuracy failures.
/work/python/bin/python - <<'PY'
import json,pathlib
root=pathlib.Path('/work/out')
for name in ['variant-parity-report.json','variant-shape-report.json']:
    assert json.loads((root/name).read_text())['passed']
rows=[json.loads(line) for line in (root/'variant-shape-models.jsonl').read_text().splitlines()]
assert rows[0]['backend']=='java-deberta-rust-dense-tokenizer'
previous=0
for row in rows[1:]:
    current=row['denseProjectionCalls'];assert current>previous;previous=current
(root/'native-call-evidence.json').write_text(json.dumps(dict(stage='shape-parity',cases=len(rows)-1,completedProjectionCalls=previous,catalogQualified=False))+'\n')
PY
bash download_reranker_fixture.sh /work/reranker
java --add-modules jdk.incubator.vector -Xmx3g -Dmodels.deberta.threads=8 -cp '/work/payload/classes:/work/payload/lib/*' RerankerRegression /work/reranker > /work/out/dense-reranker-regression.json
sha256sum /work/reranker/* > /work/out/dense-reranker-snapshot-sha256.txt
/work/python/bin/python run_worker.py --cases performance-cases.jsonl --output /work/out/dense-performance-reference.jsonl -- /work/python/bin/python reference_worker.py --snapshot /work/variant --manifest /work/variant-manifest.json --source /work/reference --threads 8
/work/python/bin/python run_worker.py --cases performance-cases.jsonl --output /work/out/dense-performance-models.jsonl -- java --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED -Xmx8g -Dmodels.deberta.threads=8 -Djdk.virtualThreadScheduler.parallelism=8 -cp '/work/payload/classes:/work/payload/lib/*' GlinerWorker /work/variant /work/payload/rust/target/release/libjmodels_kernels.so /work/variant-manifest.json
/work/python/bin/python compare_performance.py --cases performance-cases.jsonl --candidate /work/out/dense-performance-models.jsonl --reference /work/out/dense-performance-reference.jsonl --output /work/out/dense-performance-report.json
# Only a passing screen justifies repeating full task quality with this runtime. Even if
# these all pass, one performance pair is not the required final replication sequence.
bash evaluate_finetune_remote.sh
