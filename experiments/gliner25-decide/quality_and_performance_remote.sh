#!/usr/bin/env bash
# Run both complete evaluations on the exact same checkpoint and runtime, even if a gate fails.
set -uo pipefail
trap 'touch /work/STOP' EXIT
cd /work/payload/experiment
export PATH=/work/jdk/bin:/root/.cargo/bin:$PATH TOKENIZERS_PARALLELISM=false
aws s3 cp "$GLINER_URI/evaluation-overlay.tar.gz" /work/evaluation-overlay.tar.gz --only-show-errors || exit 2
printf '%s  %s\n' "$GLINER_EVAL_OVERLAY_SHA" /work/evaluation-overlay.tar.gz | sha256sum -c - || exit 2
tar -xzf /work/evaluation-overlay.tar.gz -C /work/payload/experiment || exit 2
bash performance_remote.sh
performance_rc=$?
# Infrastructure/parity failures have no complete performance report: stop instead of compounding them.
[[ -f /work/out/performance-replications-report.json ]] || exit "$performance_rc"
bash evaluate_finetune_remote.sh
quality_rc=$?
printf '{"performanceExitCode":%d,"qualityExitCode":%d}\n' "$performance_rc" "$quality_rc" > /work/out/evaluation-phases.json
(( performance_rc == 0 && quality_rc == 0 ))
