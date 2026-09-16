#!/usr/bin/env bash
# Bounded-host runner for the Granite 4.1 3B answerability hybrid, gate 4 (and the Rust identity
# precondition from the preflight amendment). Runs on a fresh Ubuntu 24.04 x86-64 host as root.
#
# Everything the run needs is pinned: the Models commit, the base GGUF, the tokenizer files, the
# adapter files, and the frozen window. Every download is hash-checked before use, the Java runtime
# owns all inference, and nothing external to Models is installed for inference.
#
# Usage: MODELS_COMMIT=<sha> HF_TOKEN=<optional> bash host-run.sh <phase>
#   phase = bootstrap | artifacts | identity | window | window-arm <backend> | all
set -euo pipefail

MODELS_COMMIT="${MODELS_COMMIT:?MODELS_COMMIT is required}"
RUN_ROOT="${RUN_ROOT:-/opt/modeljars-runs/granite41-answerability-20260915}"
EVIDENCE="$RUN_ROOT/evidence"
MODELS="$RUN_ROOT/models"
CANDIDATE_DIR="benchmark-results/2026-09-15-granite-4.1-3b-alora-hybrid"
STORE="$RUN_ROOT/store"

BASE_GGUF_URL='https://huggingface.co/ibm-granite/granite-4.1-3b-GGUF/resolve/ab4701481089b58a082ef63cc1cee738887293ff/granite-4.1-3b-Q4_K_M.gguf'
BASE_GGUF_SHA='662b0626cd58f443baea23559b469df6576a81d349649c59413b36a9fb32eb29'
TOKENIZER_URL='https://huggingface.co/ibm-granite/granite-4.1-3b/resolve/c0650403e44e78ec0262dab1c90914c65b196c4e'
LIBRARY_URL='https://huggingface.co/ibm-granite/granitelib-rag-r1.0/resolve/2f0b2c79c6731068625aca8045c2eb2e8912b353'
ADAPTER_SHA='765e85650d39b89e055634c8da0c2d469c3bfa436df271df17f516ad7466d6ca'
ADAPTER_CONFIG_SHA='79409b5fc9702a59d89b55d827cb9fa5d1c41fe1af01ddafceae4349ae227698'
CARD_SHA='010d3765af42b4c69a906bca1762b236b1db9e1f13592c4c7b78dc96db8381b8'
declare -A TOKENIZER_SHA=(
  [tokenizer.json]='e2bad66439538cb4d5a7580680932432ed9ece9d3b8577e675512bdf11599253'
  [tokenizer_config.json]='a5ec5daab12ba090a90f3dd169c8f9c275557013a87b9c1258dc7cb497a35c86'
  [vocab.json]='8af71076de8b0b626eed0f4c984faf0a7c062479164b2a31308a948524d4f69c'
  [merges.txt]='b6fe424e334903f7fb84d3a106d9730455f4744b9fe3c21ee136d97a00e72502'
  [special_tokens_map.json]='c08676c49fd7969a3130f72be6d4bf34da66aa484a6e21dffe359893a1bd5f2e'
  [chat_template.jinja]='fed2756d2d24e127b951dcf139d0b03ab7db8ef23a456128ebc9c2db4901d476'
)

log() { printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$EVIDENCE/host-run.log"; }

fetch() { # url dest sha
  local url="$1" dest="$2" sha="$3"
  if [ -f "$dest" ] && [ "$(sha256sum "$dest" | cut -d' ' -f1)" = "$sha" ]; then
    log "present $dest"; return
  fi
  local attempt
  for attempt in 1 2 3 4 5 6; do
    if curl -fsSL --retry 0 -o "$dest.part" "$url"; then break; fi
    log "fetch attempt $attempt failed for $url; backing off"
    rm -f "$dest.part"; sleep $((attempt * 30))
  done
  test -f "$dest.part" || { log "GIVING UP on $url"; exit 2; }
  local actual; actual="$(sha256sum "$dest.part" | cut -d' ' -f1)"
  if [ "$actual" != "$sha" ]; then log "HASH MISMATCH $dest expected $sha got $actual"; rm -f "$dest.part"; exit 2; fi
  mv "$dest.part" "$dest"; log "fetched $dest $sha"
}

bootstrap() {
  mkdir -p "$EVIDENCE" "$STORE"
  { hostname; uname -a; lscpu; free -h; df -h /; uptime; pgrep -af 'java|gradle|llama|ollama|cargo|python' || true; } > "$EVIDENCE/host-baseline.txt" 2>&1
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq && apt-get install -y -qq git curl jq build-essential pkg-config python3 unzip > "$EVIDENCE/apt.log" 2>&1
  if ! command -v java >/dev/null || ! java -version 2>&1 | grep -q '"25'; then
    mkdir -p /opt/java
    curl -fsSL 'https://api.adoptium.net/v3/binary/latest/25/ga/linux/x64/jdk/hotspot/normal/eclipse' -o /tmp/jdk25.tar.gz
    tar -xzf /tmp/jdk25.tar.gz -C /opt/java
    ln -sfn "$(ls -d /opt/java/jdk-25* | head -1)" /opt/java/current
  fi
  export JAVA_HOME=/opt/java/current PATH=/opt/java/current/bin:$PATH
  java -version 2>&1 | tee "$EVIDENCE/java-version.txt"
  if ! command -v cargo >/dev/null; then
    curl -fsSL https://sh.rustup.rs | sh -s -- -y --profile minimal > "$EVIDENCE/rustup.log" 2>&1
  fi
  export PATH="$HOME/.cargo/bin:$PATH"
  if [ ! -d "$MODELS/.git" ]; then
    git clone --quiet https://github.com/integrallis/models.git "$MODELS"
  fi
  git -C "$MODELS" fetch --quiet origin "$MODELS_COMMIT"
  git -C "$MODELS" switch --detach --quiet "$MODELS_COMMIT"
  test -z "$(git -C "$MODELS" status --short)"
  log "models checkout $(git -C "$MODELS" rev-parse HEAD)"
  ( cd "$MODELS" && ./gradlew :backend-native:cargoBuildRelease :backend-native:prepareNativePlatformResources :models-bench:classes --console=plain -q ) > "$EVIDENCE/build.log" 2>&1
  find "$MODELS/backend-native/build" -name native.properties -exec cat {} \; | tee "$EVIDENCE/native-library.txt"
  log "bootstrap complete"
}

artifacts() {
  mkdir -p "$STORE/tokenizer" "$STORE/adapter-src" "$STORE/models"
  fetch "$BASE_GGUF_URL" "$STORE/models/granite-4.1-3b-Q4_K_M.gguf" "$BASE_GGUF_SHA"
  for f in "${!TOKENIZER_SHA[@]}"; do fetch "$TOKENIZER_URL/$f" "$STORE/tokenizer/$f" "${TOKENIZER_SHA[$f]}"; done
  fetch "$LIBRARY_URL/answerability/granite-4.1-3b/alora/adapter_model.safetensors" "$STORE/adapter-src/adapter_model.safetensors" "$ADAPTER_SHA"
  fetch "$LIBRARY_URL/answerability/granite-4.1-3b/alora/adapter_config.json" "$STORE/adapter-src/adapter_config.json" "$ADAPTER_CONFIG_SHA"
  fetch "$LIBRARY_URL/answerability/README.md" "$STORE/answerability-README.md" "$CARD_SHA"
  rm -rf "$STORE/adapter"
  python3 "$MODELS/$CANDIDATE_DIR/package_upstream_activated_adapter.py" \
    --adapter-directory "$STORE/adapter-src" --base-artifact "$STORE/models/granite-4.1-3b-Q4_K_M.gguf" \
    --base-model ibm-granite/granite-4.1-3b --base-revision ab4701481089b58a082ef63cc1cee738887293ff \
    --tokenizer-directory "$STORE/tokenizer" --tokenizer-file tokenizer.json --tokenizer-file tokenizer_config.json \
    --tokenizer-file special_tokens_map.json --tokenizer-file vocab.json --tokenizer-file merges.txt --tokenizer-file chat_template.jinja \
    --publisher "IBM Granite" --repository ibm-granite/granitelib-rag-r1.0 --revision 2f0b2c79c6731068625aca8045c2eb2e8912b353 \
    --license Apache-2.0 --license-file "$MODELS/LICENSE" --notice-file "$MODELS/$CANDIDATE_DIR/NOTICE" \
    --invocation-tokens '[100264,78191,100265]' --invocation-text '<|start_of_role|>assistant<|end_of_role|>' \
    --model-card "$STORE/answerability-README.md" --output-directory "$STORE/adapter" | tee "$EVIDENCE/package.log"
  sha256sum "$STORE/adapter/"* | tee "$EVIDENCE/adapter-bundle.sha256"
  log "artifacts complete"
}

run_arm() { # suite arm backend limit tag
  local suite="$1" arm="$2" backend="$3" limit="$4" tag="$5"
  export JAVA_HOME=/opt/java/current PATH=/opt/java/current/bin:$HOME/.cargo/bin:$PATH
  local report="$EVIDENCE/$tag-$suite-$arm-$backend.json"
  if [ -s "$report" ]; then log "present $report"; return 0; fi
  log "start $tag $suite $arm $backend limit=$limit"
  ( cd "$MODELS" && ./gradlew :models-bench:run -PmodelsBenchNative=true --console=plain -q --args="activated-answerability \
      --model $STORE/models/granite-4.1-3b-Q4_K_M.gguf --adapter $STORE/adapter --models-revision $MODELS_COMMIT \
      --window $MODELS/$CANDIDATE_DIR/qualification-window.json --suite $suite --arm $arm --backend $backend \
      --report $report --limit $limit" ) > "$EVIDENCE/$tag-$suite-$arm-$backend.log" 2>&1 || log "NONZERO exit for $tag $suite $arm $backend"
  log "done $tag $suite $arm $backend: $(grep -E '^(EXECUTED|EXECUTION-FAILED)' "$EVIDENCE/$tag-$suite-$arm-$backend.log" | tail -1)"
}

identity() {
  for suite in mtrag-human-rag squad-v2-dev; do for arm in specialist base; do for backend in pure-java rust-ffm; do
    run_arm "$suite" "$arm" "$backend" 10 identity10
  done; done; done
  python3 - "$EVIDENCE" <<'PY' | tee "$EVIDENCE/identity10-summary.txt"
import json,sys,glob,os
E=sys.argv[1]; ok=True
for suite in ("mtrag-human-rag","squad-v2-dev"):
    for arm in ("specialist","base"):
        try:
            j=json.load(open(f"{E}/identity10-{suite}-{arm}-pure-java.json")); r=json.load(open(f"{E}/identity10-{suite}-{arm}-rust-ffm.json"))
        except FileNotFoundError as e: print("MISSING",e); ok=False; continue
        pairs=list(zip(j["cases"],r["cases"])); same=all(a["output"]==b["output"] and a["sharedPrefixTokens"]==b["sharedPrefixTokens"] for a,b in pairs)
        print(suite,arm,"cases",len(pairs),"identical" if same else "DIVERGENT", "java ms",sum(a["millis"] for a,_ in pairs),"rust ms",sum(b["millis"] for _,b in pairs))
        ok = ok and same and len(pairs)==10
print("IDENTITY10", "PASS" if ok else "FAIL")
PY
}

window() {
  if ! grep -q "IDENTITY10 PASS" "$EVIDENCE/identity10-summary.txt" 2>/dev/null; then log "identity precondition not satisfied; refusing the Rust window"; exit 3; fi
  window_arm rust-ffm
}

# The MT-RAG identity screen failed on 1 of 10 cases (recorded in preflight.md), so the window
# runs on both backends at one frozen commit: pure Java is the reference the amendment names, Rust
# FFM is the backend ModelJars ships. This phase records the identity summary next to the arm it
# runs instead of refusing, and the report states the per-case agreement between the arms.
window_arm() { # backend [suites] [arms]  (space-separated lists; defaults = every suite, every arm)
  local backend="$1" suites="${2:-squad-v2-dev mtrag-human-rag}" arms="${3:-specialist base}"
  log "window-arm $backend suites=[$suites] arms=[$arms] at $MODELS_COMMIT; identity10 summary: $(tail -1 "$EVIDENCE/identity10-summary.txt" 2>/dev/null || echo absent)"
  for suite in $suites; do for arm in $arms; do
    run_arm "$suite" "$arm" "$backend" 0 window
  done; done
  sha256sum "$EVIDENCE"/*.json | tee "$EVIDENCE/SHA256SUMS"
  log "window-arm $backend suites=[$suites] arms=[$arms] complete"
}

# Later phases run at a newer Models commit that adds the answerability long-context and
# crossover runners. The checkout is moved forward explicitly and both revisions are recorded.
checkout() { # commit
  export JAVA_HOME=/opt/java/current PATH=/opt/java/current/bin:$HOME/.cargo/bin:$PATH
  git -C "$MODELS" fetch --quiet origin "$1"
  git -C "$MODELS" switch --detach --quiet "$1"
  test -z "$(git -C "$MODELS" status --short)"
  ( cd "$MODELS" && ./gradlew :backend-native:cargoBuildRelease :backend-native:prepareNativePlatformResources :models-bench:classes --console=plain -q ) >> "$EVIDENCE/build.log" 2>&1
  log "models checkout now $(git -C "$MODELS" rev-parse HEAD)"
}

longcontext() { # backend
  local backend="${1:-rust-ffm}"
  export JAVA_HOME=/opt/java/current PATH=/opt/java/current/bin:$HOME/.cargo/bin:$PATH
  local rev; rev="$(git -C "$MODELS" rev-parse HEAD)"
  local report="$EVIDENCE/gate5-long-context-$backend.json"
  if [ -s "$report" ]; then log "present $report"; return 0; fi
  log "start gate5 long-context $backend at $rev"
  ( cd "$MODELS" && ./gradlew :models-bench:run -PmodelsBenchNative=true --console=plain -q --args="activated-answerability-long-context \
      --model $STORE/models/granite-4.1-3b-Q4_K_M.gguf --adapter $STORE/adapter --models-revision $rev --report $report" ) > "$EVIDENCE/gate5-long-context-$backend.log" 2>&1 || log "NONZERO exit for gate5 $backend"
  log "done gate5 long-context $backend: $(grep -E '^(PASS|FAIL) ' "$EVIDENCE/gate5-long-context-$backend.log" | tail -1)"
}

crossover() { # backend
  local backend="${1:-pure-java}"
  export JAVA_HOME=/opt/java/current PATH=/opt/java/current/bin:$HOME/.cargo/bin:$PATH
  local rev; rev="$(git -C "$MODELS" rev-parse HEAD)"
  local report="$EVIDENCE/gate6-crossover-$backend.json"
  if [ -s "$report" ]; then log "present $report"; return 0; fi
  log "start gate6 crossover $backend at $rev"
  ( cd "$MODELS" && ./gradlew :models-bench:run -PmodelsBenchNative=true --console=plain -q --args="activated-prefix-sharing \
      --model $STORE/models/granite-4.1-3b-Q4_K_M.gguf --adapter $STORE/adapter --models-revision $rev \
      --template granite-documents --warmups 1 --trials 3 --report $report" ) > "$EVIDENCE/gate6-crossover-$backend.log" 2>&1 || log "NONZERO exit for gate6 $backend"
  log "done gate6 crossover $backend: $(grep -E '^(PASS|FAIL) ' "$EVIDENCE/gate6-crossover-$backend.log" | tail -1)"
}

case "${1:-all}" in
  checkout) checkout "$2" ;;
  longcontext) longcontext "${2:-rust-ffm}" ;;
  crossover) crossover "${2:-pure-java}" ;;
  bootstrap) bootstrap ;;
  artifacts) artifacts ;;
  identity) identity ;;
  window) window ;;
  window-arm) window_arm "${2:?backend}" "${3:-}" "${4:-}" ;;
  all) bootstrap; artifacts; identity; window ;;
  *) echo "unknown phase: $1" >&2; exit 64 ;;
esac
