#!/usr/bin/env bash
set -euo pipefail

export JAVA_HOME=/opt/temurin-25
export PATH="$JAVA_HOME/bin:/root/.cargo/bin:$PATH"
export RUSTUP_AUTO_INSTALL=0

readonly RUN_ROOT=/opt/modeljars-runs/flat-prefill-20260908
readonly RESULT_ROOT="$RUN_ROOT/results"
readonly DEADLINE_EPOCH=1788878700 # 2026-09-08T14:45:00Z
readonly -a ORDER=(control candidate candidate control control candidate)

mkdir -p "$RESULT_ROOT"
printf 'started_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee "$RESULT_ROOT/lifecycle.txt"
hostnamectl | tee "$RESULT_ROOT/host.txt"
lscpu | tee -a "$RESULT_ROOT/host.txt"
free -h | tee -a "$RESULT_ROOT/host.txt"
"$JAVA_HOME/bin/java" -version 2>&1 | tee "$RESULT_ROOT/java-version.txt"
printf '%s  %s\n' \
  81b64d05a23b17b34c475f42b3e72fbde62d4b92cc34541f7a8031d0752deafa \
  "$RUN_ROOT/models/MiniCPM5-1B-Q4_K_M.gguf" | sha256sum -c -
printf '%s  %s\n' \
  da2572f16c06133561ce56accaa822216f2391ef4d37fba427801cd6736417d4 \
  "$RUN_ROOT/models/Qwen3-0.6B-Q4_0.gguf" | sha256sum -c -

run_model() {
  local label=$1
  local model=$2
  local control_run=0
  local candidate_run=0
  local arm
  local ordinal
  local output

  for arm in "${ORDER[@]}"; do
    if (( $(date +%s) >= DEADLINE_EPOCH )); then
      printf 'benchmark deadline reached before %s/%s\n' "$label" "$arm" >&2
      return 124
    fi
    if [[ $arm == control ]]; then
      ((++control_run))
      ordinal=$control_run
    else
      ((++candidate_run))
      ordinal=$candidate_run
    fi
    output="$RESULT_ROOT/${label}-${arm}-$(printf '%02d' "$ordinal")"
    printf 'run_start=%s model=%s arm=%s ordinal=%d\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$label" "$arm" "$ordinal" \
      | tee -a "$RESULT_ROOT/lifecycle.txt"
    (
      cd "$RUN_ROOT/$arm"
      timeout --signal=TERM --kill-after=30s 20m \
        ./gradlew --no-daemon :models-bench:run --console=plain \
          --args="profile-ragged-prefill --model $model --mode ragged --context 512 --concurrency 4 --warmups 2 --iterations 5 --prompt-file $RUN_ROOT/prompt.txt --output $output.json"
    ) 2>&1 | tee "$output.log"
    printf 'run_finish=%s model=%s arm=%s ordinal=%d\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$label" "$arm" "$ordinal" \
      | tee -a "$RESULT_ROOT/lifecycle.txt"
  done
}

run_model minicpm5 "$RUN_ROOT/models/MiniCPM5-1B-Q4_K_M.gguf"
run_model qwen3-0.6b "$RUN_ROOT/models/Qwen3-0.6B-Q4_0.gguf"

printf 'finished_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$RESULT_ROOT/lifecycle.txt"
pgrep -af 'java|gradle|cargo|rustc' > "$RESULT_ROOT/processes-after.txt" || true
sha256sum "$RESULT_ROOT"/*.json > "$RESULT_ROOT/SHA256SUMS"
