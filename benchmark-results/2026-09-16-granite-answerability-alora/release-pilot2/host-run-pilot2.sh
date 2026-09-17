#!/usr/bin/env bash
# Java qualification of the Integrallis answerability aLoRA (pilot 2) on Granite 4.1 3B, at the
# Models release commit v0.3.42 so the report, the Central artifacts and the clean-host run pin
# one revision. Derived from ../../2026-09-15-granite-4.1-3b-alora-hybrid/host-run.sh; the only
# differences are the adapter source (our weights, copied to $STORE/adapter-src and hash-checked),
# the provenance written by the packager, window v2, and the phases below.
#
# Usage: bash host-run-pilot2.sh <phase> [args]
#   bootstrap | artifacts | smoke | identity | arm <suite> <arm> <backend> | rust-window |
#   junit | longcontext <label> | crossover <label>   (these two CLIs take no --backend; label as in host-run.sh)
set -euo pipefail

MODELS_COMMIT=6063076e476ba7a477e3f6dd966a77b571352d29   # v0.3.42
PROVENANCE_REVISION=ec56bc99a04457df8274c84c1f4ff72dc4167d9d # tag adapter-provenance/granite-4.1-3b-answerability-pilot2
RUN_ROOT="${RUN_ROOT:-/opt/modeljars-runs/granite41-answerability-pilot2}"
EVIDENCE="$RUN_ROOT/evidence"
MODELS="$RUN_ROOT/models"
STORE="$RUN_ROOT/store"
CANDIDATE_DIR="benchmark-results/2026-09-15-granite-4.1-3b-alora-hybrid"
WINDOW="$MODELS/$CANDIDATE_DIR/qualification-window-v2.json"
WINDOW_SHA=dfb8cd7203418c79bae27306ffc4857f0ab02be7f9f8ddcae793af556e9cab37

BASE_GGUF_URL='https://huggingface.co/ibm-granite/granite-4.1-3b-GGUF/resolve/ab4701481089b58a082ef63cc1cee738887293ff/granite-4.1-3b-Q4_K_M.gguf'
BASE_GGUF_SHA='662b0626cd58f443baea23559b469df6576a81d349649c59413b36a9fb32eb29'
TOKENIZER_URL='https://huggingface.co/ibm-granite/granite-4.1-3b/resolve/c0650403e44e78ec0262dab1c90914c65b196c4e'
declare -A TOKENIZER_SHA=(
  [tokenizer.json]='e2bad66439538cb4d5a7580680932432ed9ece9d3b8577e675512bdf11599253'
  [tokenizer_config.json]='a5ec5daab12ba090a90f3dd169c8f9c275557013a87b9c1258dc7cb497a35c86'
  [vocab.json]='8af71076de8b0b626eed0f4c984faf0a7c062479164b2a31308a948524d4f69c'
  [merges.txt]='b6fe424e334903f7fb84d3a106d9730455f4744b9fe3c21ee136d97a00e72502'
  [special_tokens_map.json]='c08676c49fd7969a3130f72be6d4bf34da66aa484a6e21dffe359893a1bd5f2e'
  [chat_template.jinja]='fed2756d2d24e127b951dcf139d0b03ab7db8ef23a456128ebc9c2db4901d476'
)
declare -A ADAPTER_SRC_SHA=(
  [adapter_model.safetensors]='67533dff14cd0cfa9ae0d00e83a9955226426ff1f98f1fd28790b2fd8757eea6'
  [adapter_config.json]='7b559cf39b86084d92363ad6bce1ed92a9ef3c2de78e512c65b693dd6e1f7209'
)
PROVENANCE_URL="https://raw.githubusercontent.com/integrallis/models/$PROVENANCE_REVISION/benchmark-results/2026-09-16-granite-answerability-alora/release-pilot2"

export JAVA_HOME=/opt/java/current PATH=/opt/java/current/bin:$HOME/.cargo/bin:$PATH
mkdir -p "$EVIDENCE" "$STORE"
log() { printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$EVIDENCE/host-run.log"; }

fetch() { # url dest sha
  local url="$1" dest="$2" sha="$3"
  if [ -f "$dest" ] && [ "$(sha256sum "$dest" | cut -d' ' -f1)" = "$sha" ]; then log "present $dest"; return; fi
  local attempt
  for attempt in 1 2 3 4 5 6; do
    if curl -fsSL -o "$dest.part" "$url"; then break; fi
    log "fetch attempt $attempt failed for $url"; rm -f "$dest.part"; sleep $((attempt * 30))
  done
  test -f "$dest.part" || { log "GIVING UP on $url"; exit 2; }
  local actual; actual="$(sha256sum "$dest.part" | cut -d' ' -f1)"
  [ "$actual" = "$sha" ] || { log "HASH MISMATCH $dest expected $sha got $actual"; rm -f "$dest.part"; exit 2; }
  mv "$dest.part" "$dest"; log "fetched $dest $sha"
}

bootstrap() {
  { hostname; uname -a; lscpu; free -h; df -h /; uptime; } > "$EVIDENCE/host-baseline.txt" 2>&1
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq && apt-get install -y -qq git curl jq build-essential pkg-config python3 unzip > "$EVIDENCE/apt.log" 2>&1
  if ! java -version 2>&1 | grep -q '"25'; then
    mkdir -p /opt/java
    curl -fsSL 'https://api.adoptium.net/v3/binary/latest/25/ga/linux/x64/jdk/hotspot/normal/eclipse' -o /tmp/jdk25.tar.gz
    tar -xzf /tmp/jdk25.tar.gz -C /opt/java
    ln -sfn "$(ls -d /opt/java/jdk-25* | head -1)" /opt/java/current
  fi
  java -version 2>&1 | tee "$EVIDENCE/java-version.txt"
  command -v cargo >/dev/null || curl -fsSL https://sh.rustup.rs | sh -s -- -y --profile minimal > "$EVIDENCE/rustup.log" 2>&1
  [ -d "$MODELS/.git" ] || git clone --quiet https://github.com/integrallis/models.git "$MODELS"
  git -C "$MODELS" fetch --quiet origin "$MODELS_COMMIT"
  git -C "$MODELS" switch --detach --quiet "$MODELS_COMMIT"
  test -z "$(git -C "$MODELS" status --short)"
  [ "$(sha256sum "$WINDOW" | cut -d' ' -f1)" = "$WINDOW_SHA" ] || { log "WINDOW HASH MISMATCH"; exit 2; }
  log "models checkout $(git -C "$MODELS" rev-parse HEAD)"
  ( cd "$MODELS" && ./gradlew :backend-native:cargoBuildRelease :backend-native:prepareNativePlatformResources :models-bench:classes --console=plain -q ) > "$EVIDENCE/build.log" 2>&1
  log "bootstrap complete"
}

artifacts() {
  mkdir -p "$STORE/tokenizer" "$STORE/models" "$STORE/provenance"
  fetch "$BASE_GGUF_URL" "$STORE/models/granite-4.1-3b-Q4_K_M.gguf" "$BASE_GGUF_SHA"
  for f in "${!TOKENIZER_SHA[@]}"; do fetch "$TOKENIZER_URL/$f" "$STORE/tokenizer/$f" "${TOKENIZER_SHA[$f]}"; done
  for f in "${!ADAPTER_SRC_SHA[@]}"; do
    [ "$(sha256sum "$STORE/adapter-src/$f" | cut -d' ' -f1)" = "${ADAPTER_SRC_SHA[$f]}" ] || { log "ADAPTER SOURCE HASH MISMATCH $f"; exit 2; }
  done
  curl -fsSL -o "$STORE/provenance/MODEL_CARD.md" "$PROVENANCE_URL/MODEL_CARD.md"
  curl -fsSL -o "$STORE/provenance/NOTICE" "$PROVENANCE_URL/NOTICE"
  curl -fsSL -o "$STORE/provenance/adapter_config.json" "$PROVENANCE_URL/adapter_config.json"
  cmp "$STORE/provenance/adapter_config.json" "$STORE/adapter-src/adapter_config.json" || { log "PROVENANCE CONFIG DIFFERS"; exit 2; }
  rm -rf "$STORE/adapter"
  python3 "$MODELS/$CANDIDATE_DIR/package_upstream_activated_adapter.py" \
    --adapter-directory "$STORE/adapter-src" --base-artifact "$STORE/models/granite-4.1-3b-Q4_K_M.gguf" \
    --base-model ibm-granite/granite-4.1-3b --base-revision ab4701481089b58a082ef63cc1cee738887293ff \
    --tokenizer-directory "$STORE/tokenizer" --tokenizer-file tokenizer.json --tokenizer-file tokenizer_config.json \
    --tokenizer-file special_tokens_map.json --tokenizer-file vocab.json --tokenizer-file merges.txt --tokenizer-file chat_template.jinja \
    --publisher Integrallis --repository integrallis/models --revision "$PROVENANCE_REVISION" \
    --license Apache-2.0 --license-file "$MODELS/LICENSE" --notice-file "$STORE/provenance/NOTICE" \
    --invocation-tokens '[100264,78191,100265]' --invocation-text '<|start_of_role|>assistant<|end_of_role|>' \
    --model-card "$STORE/provenance/MODEL_CARD.md" --output-directory "$STORE/adapter" | tee "$EVIDENCE/package.log"
  sha256sum "$STORE/adapter/"* | tee "$EVIDENCE/adapter-bundle.sha256"
  cp "$STORE/adapter/models-activated-lora.json" "$EVIDENCE/"
  log "artifacts complete"
}

run_arm() { # suite arm backend limit tag
  local suite="$1" arm="$2" backend="$3" limit="$4" tag="$5"
  local report="$EVIDENCE/$tag-$suite-$arm-$backend.json"
  if [ -s "$report" ]; then log "present $report"; return 0; fi
  log "start $tag $suite $arm $backend limit=$limit"
  ( cd "$MODELS" && ./gradlew :models-bench:run -PmodelsBenchNative=true --console=plain -q --args="activated-answerability \
      --model $STORE/models/granite-4.1-3b-Q4_K_M.gguf --adapter $STORE/adapter --models-revision $MODELS_COMMIT \
      --window $WINDOW --suite $suite --arm $arm --backend $backend --report $report --limit $limit" ) \
      > "$EVIDENCE/$tag-$suite-$arm-$backend.log" 2>&1 || log "NONZERO exit for $tag $suite $arm $backend"
  log "done $tag $suite $arm $backend: $(grep -E '^(EXECUTED|EXECUTION-FAILED)' "$EVIDENCE/$tag-$suite-$arm-$backend.log" | tail -1)"
}

smoke() {
  run_arm squad-v2-dev specialist rust-ffm 3 smoke3
  run_arm squad-v2-dev specialist pure-java 3 smoke3
  python3 - "$EVIDENCE" <<'PY'
import json, sys
E = sys.argv[1]
for b in ("rust-ffm", "pure-java"):
    d = json.load(open(f"{E}/smoke3-squad-v2-dev-specialist-{b}.json"))
    print("SMOKE", b, [(c["id"], c["output"], c["structured"], c["physicallyShared"]) for c in d["cases"]])
PY
}

identity() {
  for suite in squad-v2-dev msmarco-v2.1-validation; do for arm in specialist base; do for backend in pure-java rust-ffm; do
    run_arm "$suite" "$arm" "$backend" 10 identity10
  done; done; done
}

rust_window() {
  for suite in squad-v2-dev msmarco-v2.1-validation; do for arm in specialist base; do
    run_arm "$suite" "$arm" rust-ffm 0 window
  done; done
}

junit() {
  # Gates 1-3 mechanics: the two adapter-agnostic tests decide. The third test pins IBM's adapter
  # output on one case and is run separately for the record only.
  ( cd "$MODELS" && ./gradlew :backend-java:granite41AloraIntegrationTest --console=plain \
      -Dmodels.fixtures.granite41AloraBase="$STORE/models/granite-4.1-3b-Q4_K_M.gguf" \
      -Dmodels.fixtures.granite41AloraAdapter="$STORE/adapter" \
      --tests '*Granite41AloraIntegrationTest.matchesTheLlamaCppOracleAndPinsTheMarkerToTheGraniteTokenizer' \
      --tests '*Granite41AloraIntegrationTest.opensTheUpstreamAnswerabilityAdapterAtTheAssistantMarker' ) \
      > "$EVIDENCE/junit.log" 2>&1 || log "NONZERO junit"
  mkdir -p "$EVIDENCE/junit"; cp "$MODELS"/backend-java/build/test-results/granite41AloraIntegrationTest/*.xml "$EVIDENCE/junit/" 2>/dev/null || true
  ( cd "$MODELS" && ./gradlew :backend-java:granite41AloraIntegrationTest --console=plain \
      -Dmodels.fixtures.granite41AloraBase="$STORE/models/granite-4.1-3b-Q4_K_M.gguf" \
      -Dmodels.fixtures.granite41AloraAdapter="$STORE/adapter" \
      --tests '*Granite41AloraIntegrationTest.matchesThePeftReferenceWhereTheBaseWantsToAnswerInstead' ) \
      > "$EVIDENCE/junit-ibm-pin.log" 2>&1 || log "IBM-pinned test did not pass (record only)"
  mkdir -p "$EVIDENCE/junit-ibm-pin"; cp "$MODELS"/backend-java/build/test-results/granite41AloraIntegrationTest/*.xml "$EVIDENCE/junit-ibm-pin/" 2>/dev/null || true
  log "junit done: $(grep -E 'BUILD (SUCCESSFUL|FAILED)' "$EVIDENCE/junit.log" | tail -1)"
}

longcontext() { # backend
  local backend="$1" report="$EVIDENCE/gate5-long-context-$1.json"
  [ -s "$report" ] && { log "present $report"; return 0; }
  ( cd "$MODELS" && ./gradlew :models-bench:run -PmodelsBenchNative=true --console=plain -q --args="activated-answerability-long-context \
      --model $STORE/models/granite-4.1-3b-Q4_K_M.gguf --adapter $STORE/adapter --models-revision $MODELS_COMMIT --report $report" ) \
      > "$EVIDENCE/gate5-long-context-$backend.log" 2>&1 || log "NONZERO gate5 $backend"
  log "done gate5 $backend: $(grep -E '^(PASS|FAIL) ' "$EVIDENCE/gate5-long-context-$backend.log" | tail -1)"
}

crossover() { # backend
  local backend="$1" report="$EVIDENCE/gate6-crossover-$1.json"
  [ -s "$report" ] && { log "present $report"; return 0; }
  ( cd "$MODELS" && ./gradlew :models-bench:run -PmodelsBenchNative=true --console=plain -q --args="activated-prefix-sharing \
      --model $STORE/models/granite-4.1-3b-Q4_K_M.gguf --adapter $STORE/adapter --models-revision $MODELS_COMMIT \
      --template granite-documents --warmups 1 --trials 3 --report $report" ) \
      > "$EVIDENCE/gate6-crossover-$backend.log" 2>&1 || log "NONZERO gate6 $backend"
  log "done gate6 $backend: $(grep -E '^(PASS|FAIL) ' "$EVIDENCE/gate6-crossover-$backend.log" | tail -1)"
}

case "${1:?phase}" in
  bootstrap) bootstrap ;;
  artifacts) artifacts ;;
  smoke) smoke ;;
  identity) identity ;;
  arm) run_arm "${2:?suite}" "${3:?arm}" "${4:?backend}" 0 window ;;
  rust-window) rust_window ;;
  junit) junit ;;
  longcontext) longcontext "${2:?backend}" ;;
  crossover) crossover "${2:?backend}" ;;
  *) echo "unknown phase $1" >&2; exit 64 ;;
esac
