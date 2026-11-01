#!/usr/bin/env bash
# Development cohort only. No fresh or original qualification examples are evaluated.
set -euo pipefail
trap 'touch /work/STOP' EXIT
bash /work/payload/experiment/evaluate_finetune_remote.sh --parity-only
cd /work/payload/experiment
export PATH=/work/jdk/bin:/root/.cargo/bin:$PATH TOKENIZERS_PARALLELISM=false
/work/python/bin/python run_worker.py --cases development-diagnostic-cases.jsonl --output /work/out/development-reference.jsonl -- /work/python/bin/python reference_worker.py --snapshot /work/variant --manifest /work/variant-manifest.json --source /work/reference --threads 8
/work/python/bin/python run_worker.py --cases development-diagnostic-cases.jsonl --output /work/out/development-models.jsonl -- java --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED -Xmx8g -Dmodels.deberta.threads=8 -Djdk.virtualThreadScheduler.parallelism=8 -cp '/work/payload/classes:/work/payload/lib/*' GlinerWorker /work/variant /work/payload/rust/target/release/libjmodels_kernels.so /work/variant-manifest.json
/work/python/bin/python compare_parity.py --cases development-diagnostic-cases.jsonl --candidate /work/out/development-models.jsonl --reference /work/out/development-reference.jsonl --output /work/out/development-parity-report.json
/work/python/bin/python development_diagnostics.py --cases development-diagnostic-cases.jsonl --predictions /work/out/development-models.jsonl --output /work/out/development-diagnostics.json
