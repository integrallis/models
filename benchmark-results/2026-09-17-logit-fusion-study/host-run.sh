#!/usr/bin/env bash
# Bounded-host runner for the logit-fusion control study (PREREGISTRATION.md in this directory).
# Runs on a fresh Ubuntu 24.04 x86-64 host as root, one arm per host.
#
# Everything the run needs is pinned: the Models commit (MODELS_COMMIT), the datasets (Hugging Face
# revisions + sha256 in prepare_fusion_data.py and data-manifest.json), the GGUFs (models.json), and
# the arm definitions (arms.json). All of those files are read from the Models checkout at
# MODELS_COMMIT, never from the machine that launched the run. Every download is hash-checked, the
# Java runtime owns all inference, and Python is used only for data preparation and the G2 NumPy
# reference.
#
# Usage: MODELS_COMMIT=<sha> bash host-run.sh <phase> [args]
#   bootstrap                      JDK 25, uv, clone + build models-bench, host baseline
#   data                           prepare datasets, verify against the committed data-manifest.json
#   models <q8|q4|big-q8>          download + hash-check one model set
#   gates                          G0, G1, G2 (dump + NumPy reference), G4 (every dataset), G6
#   pilot <arm-label>              S2: PILOT_LIMIT GSM8K test items, tuned -> uniform, tie-break B
#   arm <dataset> <arm-label>      full run; dataset = gsm8k | gsm8k-dev | arc | math500
#   tune                           S3: dev member arms, weight grids, calibration, freeze
#   summarize <pilot|full|dev> [dataset]
#   results-log                    render results-log.jsonl to RESULTS-LOG.generated.md
#
# Grep monitoring: every phase ends with exactly one `PHASE-DONE <phase>` or `PHASE-FAILED <phase>`.
set -euo pipefail

MODELS_COMMIT="${MODELS_COMMIT:?MODELS_COMMIT is required (40-hex Models commit)}"
RUN_ROOT="${RUN_ROOT:-/opt/modeljars-runs/logit-fusion-20260917}"
BACKEND="${BACKEND:-pure-java}"
MEMBER_THREADS="${MEMBER_THREADS:-}"
JAVA_OPTS="${JAVA_OPTS:--Xms2g -Xmx12g -XX:+UseG1GC -XX:+AlwaysPreTouch}"
PILOT_LIMIT="${PILOT_LIMIT:-50}"
SC_K="${SC_K:-2}"
ALLOW_PROPOSED_PINS="${ALLOW_PROPOSED_PINS:-0}"
FROZEN="${FROZEN:-}"
FROZEN_SHA256="${FROZEN_SHA256:-}"
MAX_TOKENS_OFF="${MAX_TOKENS_OFF:-1024}"
MAX_TOKENS_THINK="${MAX_TOKENS_THINK:-8192}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-10240}"
ARC_SCORING="${ARC_SCORING:-generative}"
PHASE_TIMEOUT="${PHASE_TIMEOUT:-36h}"
UV_VERSION="${UV_VERSION:-0.11.25}"
SEED="${SEED:-20260917}"
HF_TOKEN="${HF_TOKEN:-}"

STUDY_REL="benchmark-results/2026-09-17-logit-fusion-study"
EVIDENCE="$RUN_ROOT/evidence"
MODELS="$RUN_ROOT/models"
STUDY="$MODELS/$STUDY_REL"
STORE="$RUN_ROOT/store"
DATA="$RUN_ROOT/data"
WORK="$RUN_ROOT/work"
BIN="$MODELS/models-bench/build/install/models-bench/bin/models-bench"

mkdir -p "$EVIDENCE" "$STORE/models" "$DATA" "$WORK"
export JAVA_HOME=/opt/java/current
export PATH="/opt/java/current/bin:$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
export JAVA_OPTS

PHASE="${1:-}"
PHASE_ARM=""
PHASE_DATASET=""
PHASE_EVIDENCE="$EVIDENCE"
PHASE_NOTE=""

log() { printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$EVIDENCE/host-run.log" >&2; }

record() { # phase arm dataset evidencePath outcome [extra-json-object]
  local extra="${6:-}"
  [ -n "$extra" ] || extra='{}'
  jq -nc --arg date "$(date -u +%Y-%m-%dT%H:%M:%SZ)" --arg commit "$MODELS_COMMIT" \
    --arg host "$(hostname)" --arg phase "$1" --arg arm "$2" --arg dataset "$3" \
    --arg evidence "$4" --arg outcome "$5" --argjson extra "$extra" \
    '{date:$date, modelsCommit:$commit, host:$host, phase:$phase, arm:$arm, dataset:$dataset,
      evidencePath:$evidence, outcome:$outcome, provenance:"measured"} + $extra' \
    >> "$EVIDENCE/results-log.jsonl"
}

write_sums() {
  ( cd "$EVIDENCE" && find . -type f ! -name SHA256SUMS -print0 | LC_ALL=C sort -z \
      | xargs -0 -r sha256sum > SHA256SUMS.tmp && mv SHA256SUMS.tmp SHA256SUMS )
}

finish() {
  local rc=$?
  trap - EXIT
  [ -n "$PHASE" ] || exit "$rc"
  local extra='{}'
  [ -n "$PHASE_NOTE" ] && extra="$(jq -nc --arg note "$PHASE_NOTE" '{pilotSubstitution:$note}')"
  if [ "$rc" -eq 0 ]; then
    record "$PHASE" "$PHASE_ARM" "$PHASE_DATASET" "$PHASE_EVIDENCE" "done" "$extra" || true
    write_sums || true
    log "PHASE-DONE $PHASE${PHASE_ARM:+ $PHASE_ARM}${PHASE_DATASET:+ $PHASE_DATASET}"
  else
    record "$PHASE" "$PHASE_ARM" "$PHASE_DATASET" "$PHASE_EVIDENCE" "failed rc=$rc" "$extra" || true
    write_sums || true
    log "PHASE-FAILED $PHASE${PHASE_ARM:+ $PHASE_ARM}${PHASE_DATASET:+ $PHASE_DATASET} rc=$rc"
  fi
  exit "$rc"
}
trap finish EXIT

fetch() { # url dest sha size
  local url="$1" dest="$2" sha="$3" size="$4"
  if [ -f "$dest" ] && [ "$(stat -c %s "$dest")" = "$size" ] \
      && [ "$(sha256sum "$dest" | cut -d' ' -f1)" = "$sha" ]; then
    log "present $dest"; return 0
  fi
  local attempt auth=()
  [ -n "$HF_TOKEN" ] && auth=(-H "Authorization: Bearer $HF_TOKEN")
  rm -f "$dest.part"
  for attempt in 1 2 3 4 5 6; do
    if curl -fsSL --retry 0 "${auth[@]}" -o "$dest.part" "$url"; then break; fi
    log "fetch attempt $attempt failed for $url; backing off"
    rm -f "$dest.part"; sleep $((attempt * 30))
  done
  [ -f "$dest.part" ] || { log "GIVING UP on $url"; return 2; }
  local actual_size actual_sha
  actual_size="$(stat -c %s "$dest.part")"
  actual_sha="$(sha256sum "$dest.part" | cut -d' ' -f1)"
  if [ "$actual_size" != "$size" ] || [ "$actual_sha" != "$sha" ]; then
    log "HASH MISMATCH $dest expected $sha/$size got $actual_sha/$actual_size"
    rm -f "$dest.part"; return 2
  fi
  mv "$dest.part" "$dest"; log "fetched $dest $sha"
}

require_checkout() {
  [ -x "$BIN" ] || { log "models-bench launcher missing; run bootstrap first"; return 3; }
  local head; head="$(git -C "$MODELS" rev-parse HEAD)"
  [ "$head" = "$MODELS_COMMIT" ] || { log "checkout is $head, expected $MODELS_COMMIT"; return 3; }
  [ -z "$(git -C "$MODELS" status --short)" ] || { log "checkout is dirty"; return 3; }
}

# Runs one logit-fusion CLI action under the watchdog. Output goes to <log>; returns the CLI exit code.
cli() { # log-file action args...
  local out="$1"; shift
  local rc
  log "run logit-fusion $*"
  set +e
  # Run from the checkout so every report records `git rev-parse HEAD` and the dirty flag.
  ( cd "$MODELS" && timeout --kill-after=5m "$PHASE_TIMEOUT" "$BIN" logit-fusion "$@" ) > "$out" 2>&1
  rc=$?
  set -e
  if [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ]; then log "WATCHDOG timeout ($PHASE_TIMEOUT) for $*"; fi
  log "exit $rc: $(grep -E '^(PASS|FAIL|DONE) ' "$out" | tail -1 || true)"
  return "$rc"
}

# ----------------------------------------------------------------------------------------------
# Model sets and datasets
# ----------------------------------------------------------------------------------------------

set_keys() { # set -> "name=key ..."
  case "$1" in
    q8) echo "A=A-q8 B=B-q8 C=C-q8 BIG=BIG-q4" ;;
    q4) echo "A=A-q4 B=B-q4 C=C-q4 BIG=BIG-q4" ;;
    big-q8) echo "BIG=BIG-q8" ;;
    *) log "unknown model set: $1"; return 64 ;;
  esac
}

model_path() { # key
  local file; file="$(jq -er --arg k "$1" '.models[] | select(.key==$k) | .file' "$STUDY/models.json")"
  echo "$STORE/models/$1--$file"
}

# members_arg <set> <name...> -> "A=/path,B=/path"
members_arg() {
  local set="$1"; shift
  local pairs; pairs="$(set_keys "$set")"
  local out="" name pair key path
  for name in "$@"; do
    key=""
    for pair in $pairs; do [ "${pair%%=*}" = "$name" ] && key="${pair#*=}"; done
    [ -n "$key" ] || { log "member $name is not in model set $set"; return 64; }
    path="$(model_path "$key")"
    [ -f "$path" ] || { log "missing $path; run: host-run.sh models $set"; return 3; }
    out="${out:+$out,}$name=$path"
  done
  echo "$out"
}

spec_members() { # spec -> member names, space separated
  local kind="${1%%:*}" rest="${1#*:}"
  local field="${rest%%:*}"
  case "$kind" in
    member|sc) echo "$field" ;;
    fuse|vote|vote-consist|vote-conf|rerank) echo "${field//+/ }" ;;
    *) log "unknown arm kind in spec: $1"; return 64 ;;
  esac
}

dataset_cli() { case "$1" in gsm8k|gsm8k-dev) echo gsm8k ;; arc) echo arc ;; math500) echo math500 ;; *) return 64 ;; esac; }
dataset_file() {
  case "$1" in
    gsm8k) echo "$DATA/gsm8k-test.jsonl" ;;
    gsm8k-dev) echo "$DATA/gsm8k-dev.jsonl" ;;
    arc) echo "$DATA/arc-challenge-test.jsonl" ;;
    math500) echo "$DATA/math500-test.jsonl" ;;
    *) log "unknown dataset: $1"; return 64 ;;
  esac
}

require_data() { # dataset
  local file; file="$(dataset_file "$1")"
  [ -f "$file" ] || { log "missing $file; run: host-run.sh data"; return 3; }
}

arm_field() { # label jq-field
  # `jq -e` exits 1 when the value is false or null, and `set -e` would then kill the run for every
  # arm whose boolean field is false. Read with -r, and fail only on a genuinely absent field.
  local value
  value="$(jq -r --arg l "$1" ".arms[] | select(.label==\$l) | $2" "$STUDY/arms.json")" || return 1
  [ -n "$value" ] && [ "$value" != null ] || return 1
  printf '%s\n' "$value"
}

# ----------------------------------------------------------------------------------------------
# Phases
# ----------------------------------------------------------------------------------------------

bootstrap() {
  {
    echo "== hostname"; hostname
    echo "== uname"; uname -a
    echo "== os-release"; cat /etc/os-release
    echo "== nproc"; nproc
    echo "== lscpu"; lscpu
    echo "== cpu flags"; grep -m1 '^flags' /proc/cpuinfo
    echo "== free -b"; free -b
    echo "== df"; df -B1 /
    echo "== uptime"; uptime
    echo "== competing processes"; pgrep -af 'java|gradle|llama|ollama|cargo|python' || true
  } > "$EVIDENCE/host-baseline.txt" 2>&1
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq
  apt-get install -y -qq git curl jq ca-certificates coreutils python3 unzip > "$EVIDENCE/apt.log" 2>&1
  if ! command -v java >/dev/null || ! java -version 2>&1 | grep -q '"25'; then
    mkdir -p /opt/java
    curl -fsSL 'https://api.adoptium.net/v3/binary/latest/25/ga/linux/x64/jdk/hotspot/normal/eclipse' -o /tmp/jdk25.tar.gz
    sha256sum /tmp/jdk25.tar.gz | tee "$EVIDENCE/jdk-archive.sha256"
    tar -xzf /tmp/jdk25.tar.gz -C /opt/java
    ln -sfn "$(find /opt/java -maxdepth 1 -type d -name 'jdk-25*' | sort | head -1)" /opt/java/current
  fi
  java -version > "$EVIDENCE/java-version.txt" 2>&1
  if ! command -v uv >/dev/null || ! uv --version | grep -q "$UV_VERSION"; then
    curl -LsSf "https://astral.sh/uv/$UV_VERSION/install.sh" | sh > "$EVIDENCE/uv-install.log" 2>&1
  fi
  uv --version > "$EVIDENCE/uv-version.txt"
  local gradle_flags=()
  if [ "$BACKEND" = "rust-ffm" ]; then
    apt-get install -y -qq build-essential pkg-config >> "$EVIDENCE/apt.log" 2>&1
    command -v cargo >/dev/null || curl -fsSL https://sh.rustup.rs | sh -s -- -y --profile minimal > "$EVIDENCE/rustup.log" 2>&1
    gradle_flags=(-PmodelsBenchNative=true)
  fi
  if [ ! -d "$MODELS/.git" ]; then
    git clone --quiet https://github.com/integrallis/models.git "$MODELS"
  fi
  git -C "$MODELS" fetch --quiet origin "$MODELS_COMMIT"
  git -C "$MODELS" switch --detach --quiet "$MODELS_COMMIT"
  [ -z "$(git -C "$MODELS" status --short)" ]
  log "models checkout $(git -C "$MODELS" rev-parse HEAD)"
  if [ "$BACKEND" = "rust-ffm" ]; then
    ( cd "$MODELS" && ./gradlew :backend-native:cargoBuildRelease :backend-native:prepareNativePlatformResources --console=plain -q ) > "$EVIDENCE/build-native.log" 2>&1
    find "$MODELS/backend-native/build" -name native.properties -exec cat {} \; > "$EVIDENCE/native-library.txt"
    find "$MODELS/backend-native/build" \( -name '*.so' -o -name '*.dylib' \) -exec sha256sum {} \; >> "$EVIDENCE/native-library.txt"
  fi
  ( cd "$MODELS" && ./gradlew :models-bench:installDist "${gradle_flags[@]}" --console=plain -q ) > "$EVIDENCE/build.log" 2>&1
  [ -x "$BIN" ]
  { echo "BACKEND=$BACKEND"; echo "JAVA_OPTS=$JAVA_OPTS"; echo "MEMBER_THREADS=${MEMBER_THREADS:-<member count>}"; } > "$EVIDENCE/run-settings.txt"
  log "bootstrap complete"
}

data() {
  require_checkout
  local venv="$RUN_ROOT/venv-data"
  [ -x "$venv/bin/python" ] || uv venv --quiet --python 3.12 "$venv"
  uv pip install --quiet --python "$venv/bin/python" -r "$STUDY/requirements-data.txt"
  "$venv/bin/python" -m unittest discover -s "$STUDY" -p 'prepare_fusion_data_test.py' > "$EVIDENCE/data-unittest.log" 2>&1
  "$venv/bin/python" "$STUDY/prepare_fusion_data.py" --out-dir "$DATA" --cache-dir "$RUN_ROOT/hf-cache" \
    > "$EVIDENCE/data-prepare.log" 2>&1
  cp "$DATA/data-manifest.json" "$EVIDENCE/data-manifest.produced.json"
  # The produced files must be byte-identical to the committed expectation.
  local name expected actual
  for name in $(jq -r '.datasets | keys[]' "$STUDY/data-manifest.json"); do
    expected="$(jq -er --arg n "$name" '.datasets[$n].outputSha256' "$STUDY/data-manifest.json")"
    actual="$(sha256sum "$DATA/$(jq -er --arg n "$name" '.datasets[$n].output' "$STUDY/data-manifest.json")" | cut -d' ' -f1)"
    if [ "$expected" != "$actual" ]; then log "DATA MISMATCH $name expected $expected got $actual"; return 2; fi
    log "data $name $actual"
  done
  PHASE_EVIDENCE="$EVIDENCE/data-manifest.produced.json"
}

models() { # set
  require_checkout
  local set="$1" pair key entry status
  for pair in $(set_keys "$set"); do
    key="${pair#*=}"
    entry="$(jq -ec --arg k "$key" '.models[] | select(.key==$k)' "$STUDY/models.json")"
    status="$(jq -r .status <<<"$entry")"
    if [ "$status" != "official" ] && [ "$ALLOW_PROPOSED_PINS" != "1" ]; then
      log "REFUSING $key: status=$status (set ALLOW_PROPOSED_PINS=1 only after the substitute is approved)"
      return 4
    fi
    fetch "$(jq -r .url <<<"$entry")" "$(model_path "$key")" "$(jq -r .sha256 <<<"$entry")" "$(jq -r .sizeBytes <<<"$entry")"
    jq -c --arg path "$(model_path "$key")" '. + {localPath:$path}' <<<"$entry" >> "$EVIDENCE/models-$set.jsonl"
  done
  PHASE_DATASET="$set"
  PHASE_EVIDENCE="$EVIDENCE/models-$set.jsonl"
}

gates() {
  require_checkout
  require_data gsm8k
  local failed=0 gate
  local fused; fused="$(members_arg q8 A B C)"
  local all; all="$(members_arg q8 A B C BIG)"
  local common=(--backend "$BACKEND" --models-revision "$MODELS_COMMIT")

  run_gate() { # name report cli-args...
    local name="$1" report="$2"; shift 2
    if cli "$EVIDENCE/gate-$name.log" gate "$@" --report "$report"; then
      record "gate-$name" "" "" "$report" "pass"
    else
      record "gate-$name" "" "" "$report" "fail"; failed=1
    fi
  }

  run_gate g0 "$EVIDENCE/gate-g0.json" g0 --members "$all"
  run_gate g1 "$EVIDENCE/gate-g1.json" g1 --members "$fused" --dataset gsm8k --data "$DATA/gsm8k-test.jsonl" \
    --limit 20 --max-tokens "$MAX_TOKENS_OFF" --thinking off --context-length "$CONTEXT_LENGTH" "${common[@]}"

  rm -rf "$WORK/g2-dump"
  if cli "$EVIDENCE/gate-g2-dump.log" gate g2-dump --members "$fused" --dataset gsm8k --data "$DATA/gsm8k-test.jsonl" \
      --limit 20 --steps 50 --out "$WORK/g2-dump" --report "$EVIDENCE/gate-g2-dump.json" "${common[@]}"; then
    local venv="$RUN_ROOT/venv-reference"
    [ -x "$venv/bin/python" ] || uv venv --quiet --python 3.12 "$venv"
    uv pip install --quiet --python "$venv/bin/python" -r "$STUDY/requirements-reference.txt"
    cp "$WORK/g2-dump/manifest.json" "$EVIDENCE/gate-g2-dump-manifest.json"
    set +e
    "$venv/bin/python" "$STUDY/reference_fusion.py" --dump "$WORK/g2-dump" --tolerance 1e-4 \
      --report "$EVIDENCE/gate-g2.json" > "$EVIDENCE/gate-g2.log" 2>&1
    local rc=$?
    set -e
    if [ "$rc" -eq 0 ]; then record gate-g2 "" gsm8k "$EVIDENCE/gate-g2.json" pass
    else record gate-g2 "" gsm8k "$EVIDENCE/gate-g2.json" fail; failed=1; fi
  else
    record gate-g2 "" gsm8k "$EVIDENCE/gate-g2-dump.log" "fail (dump)"; failed=1
  fi

  for gate in gsm8k gsm8k-dev arc math500; do
    require_data "$gate"
    run_gate "g4-$gate" "$EVIDENCE/gate-g4-$gate.json" g4 --dataset "$(dataset_cli "$gate")" --data "$(dataset_file "$gate")"
  done
  run_gate g6 "$EVIDENCE/gate-g6.json" g6

  PHASE_EVIDENCE="$EVIDENCE"
  if [ "$failed" -ne 0 ]; then log "one or more gates failed; the stage stops here"; return 5; fi
}

# run_arm <mode: pilot|full|dev> <dataset> <label>
run_arm() {
  local mode="$1" dataset="$2" label="$3"
  require_checkout
  require_data "$dataset"
  local spec thinking set temperature requires_frozen
  spec="$(arm_field "$label" .spec)" || { log "unknown arm label: $label"; return 64; }
  thinking="$(arm_field "$label" .thinking)"
  set="$(arm_field "$label" .modelSet)"
  temperature="$(arm_field "$label" .temperature)"
  requires_frozen="$(arm_field "$label" .requiresFrozen)"
  spec="${spec//\$\{SC_K\}/$SC_K}"

  # The pilot (S2) precedes S3 tuning. The Java runner performs the substitutions itself under
  # --pilot true (tuned -> uniform, tie-break best -> B, vote-conf temperature 1) and records them
  # in the report's config.arm.pilotSubstitution; the note here only mirrors them into the log.
  local substitution=""
  local pilot_args=()
  if [ "$mode" = "pilot" ]; then
    [[ "$spec" == *:tuned ]] && substitution="tuned weights -> uniform (S2 precedes S3 tuning)"
    [[ "$spec" == vote*:best ]] && substitution="${substitution:+$substitution; }tie-break best -> B (dev best member not chosen yet)"
    [[ "$spec" == vote-conf:* ]] && substitution="${substitution:+$substitution; }uncalibrated confidence temperature 1.0 (G7 happens at S3)"
    pilot_args=(--pilot true)
    requires_frozen=false
  fi

  local frozen_args=()
  if [ "$requires_frozen" = "true" ]; then
    [ -n "$FROZEN" ] && [ -f "$FROZEN" ] || { log "arm $label requires FROZEN=<frozen.json>"; return 6; }
    [ -n "$FROZEN_SHA256" ] || { log "arm $label requires FROZEN_SHA256"; return 6; }
    [ "$(sha256sum "$FROZEN" | cut -d' ' -f1)" = "$FROZEN_SHA256" ] || { log "FROZEN sha256 mismatch"; return 6; }
    frozen_args=(--frozen "$FROZEN" --frozen-sha256 "$FROZEN_SHA256")
  fi

  local names; names="$(spec_members "$spec")"
  local members
  # shellcheck disable=SC2086  # names is a space-separated member list
  members="$(members_arg "$set" $names)"

  local tag="$mode-$dataset"
  local extra_args=()
  if [ "$dataset" = "arc" ]; then
    extra_args+=(--arc-scoring "$ARC_SCORING")
    [ "$ARC_SCORING" = "generative" ] || tag="$tag-$ARC_SCORING"
  fi

  local dep deps=()
  for dep in $(jq -r --arg l "$label" '.arms[] | select(.label==$l) | .dependsOn[]' "$STUDY/arms.json"); do
    local dep_report="$EVIDENCE/$tag-$dep.json"
    [ -s "$dep_report" ] || { log "arm $label needs $dep_report; run arm $dep first"; return 7; }
    deps+=("$dep_report")
  done
  if [ "${#deps[@]}" -gt 0 ]; then
    extra_args+=(--member-reports "$(IFS=,; echo "${deps[*]}")")
  fi

  local limit=0
  [ "$mode" = "pilot" ] && limit="$PILOT_LIMIT"
  local max_tokens="$MAX_TOKENS_OFF"
  [ "$thinking" = "on" ] && max_tokens="$MAX_TOKENS_THINK"
  [ -n "$MEMBER_THREADS" ] && extra_args+=(--member-threads "$MEMBER_THREADS")

  local report="$EVIDENCE/$tag-$label.json"
  PHASE_ARM="$label"; PHASE_DATASET="$dataset"; PHASE_EVIDENCE="$report"; PHASE_NOTE="$substitution"
  if [ -s "$report" ]; then log "present $report"; return 0; fi
  [ -n "$substitution" ] && log "pilot substitution for $label: $substitution"
  rm -f "$report.part"
  cli "$EVIDENCE/$tag-$label.log" run \
    --arm "$spec" --arm-label "$label" --members "$members" \
    --dataset "$(dataset_cli "$dataset")" --data "$(dataset_file "$dataset")" \
    --data-manifest "$DATA/data-manifest.json" \
    --thinking "$thinking" --temperature "$temperature" --seed "$SEED" \
    --max-tokens "$max_tokens" --context-length "$CONTEXT_LENGTH" --limit "$limit" \
    --backend "$BACKEND" --models-revision "$MODELS_COMMIT" \
    "${pilot_args[@]}" "${frozen_args[@]}" "${extra_args[@]}" --report "$report.part"
  mv "$report.part" "$report"
}

tune() {
  require_checkout
  require_data gsm8k-dev
  local label
  # Dev-split member outputs: best-member choice (non-thinking) and G7 calibration (thinking).
  for label in A B C A-think B-think C-think; do
    run_arm dev gsm8k-dev "$label"
  done
  PHASE_ARM=""; PHASE_NOTE=""
  local common=(--dataset gsm8k --data "$DATA/gsm8k-dev.jsonl" --data-manifest "$DATA/data-manifest.json" --grid-step 0.1 --thinking off
    --max-tokens "$MAX_TOKENS_OFF" --temperature 0 --seed "$SEED" --context-length "$CONTEXT_LENGTH"
    --backend "$BACKEND" --models-revision "$MODELS_COMMIT")
  local tunings=() combo names rule tag
  for combo in "A+B+C:poe" "A+B+C:mixture" "A+B:poe" "A+C:poe" "B+C:poe"; do
    names="${combo%%:*}"; rule="${combo#*:}"; tag="${names//+/}-$rule"
    # shellcheck disable=SC2086
    [ -s "$EVIDENCE/tuning-$tag.json" ] || cli "$EVIDENCE/tune-$tag.log" tune-weights \
      --members "$(members_arg q8 ${names//+/ })" --arm-members "$names" --rule "$rule" \
      --work-dir "$WORK/tune-$tag" --report "$EVIDENCE/tuning-$tag.json" "${common[@]}"
    tunings+=("$EVIDENCE/tuning-$tag.json")
  done
  [ -s "$EVIDENCE/calibration.json" ] || cli "$EVIDENCE/calibrate.log" calibrate --dataset gsm8k \
    --member-reports "$EVIDENCE/dev-gsm8k-dev-A-think.json,$EVIDENCE/dev-gsm8k-dev-B-think.json,$EVIDENCE/dev-gsm8k-dev-C-think.json" \
    --report "$EVIDENCE/calibration.json"
  [ -s "$EVIDENCE/frozen.json" ] || cli "$EVIDENCE/freeze.log" freeze --tuning "$(IFS=,; echo "${tunings[*]}")" \
    --calibration "$EVIDENCE/calibration.json" --dataset gsm8k --data "$DATA/gsm8k-dev.jsonl" \
    --data-manifest "$DATA/data-manifest.json" \
    --best-member-reports "$EVIDENCE/dev-gsm8k-dev-A.json,$EVIDENCE/dev-gsm8k-dev-B.json,$EVIDENCE/dev-gsm8k-dev-C.json" \
    --report "$EVIDENCE/frozen.json"
  log "FROZEN-READY $(sha256sum "$EVIDENCE/frozen.json" | cut -d' ' -f1) — commit frozen.json to the study directory before any test-set tuned run"
  PHASE_DATASET="gsm8k-dev"
  PHASE_EVIDENCE="$EVIDENCE/frozen.json"
}

summarize() { # mode [dataset]
  require_checkout
  local mode="$1" dataset="${2:-}"
  local pattern="$EVIDENCE/$mode-${dataset:-*}-*.json"
  local reports=()
  local f
  # shellcheck disable=SC2206
  for f in $pattern; do [ -f "$f" ] && reports+=("$f"); done
  [ "${#reports[@]}" -gt 0 ] || { log "no reports match $pattern"; return 3; }
  local out="$EVIDENCE/summary-$mode${dataset:+-$dataset}.json"
  rm -f "$out"
  cli "$EVIDENCE/summary-$mode${dataset:+-$dataset}.log" summarize --reports "$(IFS=,; echo "${reports[*]}")" \
    --member-labels A,B,C --fused-label F-tuned --report "$out"
  PHASE_DATASET="$dataset"; PHASE_EVIDENCE="$out"
}

results_log() {
  require_checkout
  cli "$EVIDENCE/results-log-render.log" results-log --in "$EVIDENCE/results-log.jsonl" \
    --report "$EVIDENCE/RESULTS-LOG.generated.md"
  PHASE_EVIDENCE="$EVIDENCE/RESULTS-LOG.generated.md"
}

case "$PHASE" in
  bootstrap) bootstrap ;;
  data) data ;;
  models) models "${2:?model set: q8 | q4 | big-q8}" ;;
  gates) gates ;;
  pilot) PHASE_DATASET=gsm8k; run_arm pilot gsm8k "${2:?arm label from arms.json}" ;;
  arm) run_arm full "${2:?dataset: gsm8k | gsm8k-dev | arc | math500}" "${3:?arm label from arms.json}" ;;
  tune) tune ;;
  summarize) summarize "${2:?mode: pilot | full | dev}" "${3:-}" ;;
  results-log) results_log ;;
  *) PHASE=""; echo "unknown phase: ${1:-<none>}" >&2; exit 64 ;;
esac
