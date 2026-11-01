#!/usr/bin/env bash
# The candidate arm over every corpus, voice and pass, on our phonemes and (af_heart) on the
# publisher's, into EVIDENCE_ROOT/devbox/. Finished passes are skipped. A publisher-phonemes pass
# waits for the comparator pass it reads, so this can run beside run_comparator_passes.sh writing
# EVIDENCE_ROOT/devbox/comparator. Load averages are written before every pass (load.txt).
#
#   ./run_candidate_passes.sh [EVIDENCE_ROOT]      # default ../evidence
set -uo pipefail
cd "$(dirname "$0")"
COMP=$PWD
ROOT=$(cd ../../.. && pwd)
EXP=$(cd .. && pwd)
EV="$(cd "${1:-../evidence}" && pwd)/devbox"
ART=$COMP/.artifacts
MISAKI=$COMP/.venv/lib/python3.12/site-packages/misaki/data
log(){ echo "$(date -u +%FT%TZ) $*"; }
load(){ uptime | sed 's/.*load averages*: *//'; }

candidate(){ # corpus voice pass arm [phonemesFrom]
  local corpus=$1 voice=$2 pass=$3 arm=$4 from=${5:-}
  local out="$EV/$arm/$corpus/$voice/t1/pass$pass"
  if [ -f "$out/pass.json" ]; then log "skip $out"; return 0; fi
  if [ -n "$from" ]; then
    until [ -f "$from" ]; do log "waiting for $from"; sleep 60; done
  fi
  mkdir -p "$out"; load > "$out/load.txt"
  log "$arm $corpus $voice pass$pass (load $(cat "$out/load.txt"))"
  ( cd "$ROOT" && env MODELS_PARADEE_ONNX_DIR="$ART" MODELS_MISAKI_DATA_DIR="$MISAKI" \
      MODELS_PARADEE_CORPUS="$EXP/corpus/$corpus.json" MODELS_PARADEE_VOICE="$voice" \
      MODELS_PARADEE_PASS="$pass" MODELS_PARADEE_ARM_OUT="$out" \
      ${from:+MODELS_PARADEE_PHONEMES_FROM=$from} \
      ./gradlew :models-audio:integrationTest --tests '*ParadeeCandidateArmIntegrationTest' --rerun --console=plain -q ) > "$out/gradle.log" 2>&1
  local rc=$?
  cp "$ROOT"/models-audio/build/test-results/integrationTest/TEST-*ParadeeCandidateArm*.xml "$out/junit.xml" 2>/dev/null || true
  if [ ! -f "$out/pass.json" ]; then log "FAILED $arm $corpus $voice pass$pass (rc=$rc, see $out/gradle.log)"; return 1; fi
  [ $rc -eq 0 ] || log "note: gradle rc=$rc after pass.json was written (see $out/gradle.log)"
  grep -o 'PARADEE_ARM [a-z0-9-]* [a-z_]* pass[0-9]*: [^<]*realtime' "$out/junit.xml" 2>/dev/null | tail -1
  return 0
}

for pass in ${PASSES:-1 2 3}; do
  for corpus in ${CORPORA:-wikitext103-heldout-200 harvard-200}; do
    for voice in ${VOICES:-af_heart bf_isabella am_puck bm_fable}; do
      candidate "$corpus" "$voice" "$pass" candidate || exit 1
      if [ "$voice" = af_heart ]; then
        candidate "$corpus" "$voice" "$pass" candidate-publisher-phonemes "$EV/comparator/$corpus/$voice/t1/pass$pass/pass.json" || exit 1
      fi
    done
  done
done
log "all candidate passes present under $EV"
