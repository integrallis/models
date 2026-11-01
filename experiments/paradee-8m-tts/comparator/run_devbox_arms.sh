#!/usr/bin/env bash
# Both arms on the dev box, interleaved pass by pass so that each comparator pass and the candidate
# pass that follows it see the same background load. Evidence goes under EVIDENCE_ROOT/devbox/.
# The load averages before every pass are written beside its pass.json (load.txt) because this
# machine is a desktop; the WER gate does not depend on load, the throughput gate reads these.
#
#   ./run_devbox_arms.sh [EVIDENCE_ROOT]      # default ../evidence
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

comparator(){ # corpus voice pass
  local out="$EV/comparator/$1/$2/t1/pass$3"
  if [ -f "$out/pass.json" ]; then log "skip $out"; return 0; fi
  mkdir -p "$out"; load > "$out/load.txt"
  log "comparator $1 $2 pass$3 (load $(cat "$out/load.txt"))"
  .venv/bin/python comparator_arm.py --corpus "../corpus/$1.json" --voice "$2" --threads 1 --pass "$3" --out "$out" 2> >(tail -1 >&2)
}
candidate(){ # corpus voice pass arm [phonemesFrom]
  local corpus=$1 voice=$2 pass=$3 arm=$4 from=${5:-}
  local out="$EV/$arm/$corpus/$voice/t1/pass$pass"
  if [ -f "$out/pass.json" ]; then log "skip $out"; return 0; fi
  mkdir -p "$out"; load > "$out/load.txt"
  log "$arm $corpus $voice pass$pass (load $(cat "$out/load.txt"))"
  ( cd "$ROOT" && env MODELS_PARADEE_ONNX_DIR="$ART" MODELS_MISAKI_DATA_DIR="$MISAKI" \
      MODELS_PARADEE_CORPUS="$EXP/corpus/$corpus.json" MODELS_PARADEE_VOICE="$voice" \
      MODELS_PARADEE_PASS="$pass" MODELS_PARADEE_ARM_OUT="$out" \
      ${from:+MODELS_PARADEE_PHONEMES_FROM=$from} \
      ./gradlew :models-audio:integrationTest --tests '*ParadeeCandidateArmIntegrationTest' --rerun --console=plain -q ) > "$out/gradle.log" 2>&1
  local rc=$?
  cp "$ROOT"/models-audio/build/test-results/integrationTest/TEST-*ParadeeCandidateArm*.xml "$out/junit.xml" 2>/dev/null || true
  [ $rc -eq 0 ] && [ -f "$out/pass.json" ] || { log "FAILED $arm $corpus $voice pass$pass (rc=$rc, see $out/gradle.log)"; return 1; }
  grep -o 'PARADEE_ARM [a-z0-9-]* [a-z_]* pass[0-9]*: [^<]*realtime' "$out/junit.xml" | tail -1
}

for pass in ${PASSES:-1 2 3}; do
  for corpus in ${CORPORA:-wikitext103-heldout-200 harvard-200}; do
    for voice in ${VOICES:-af_heart bf_isabella am_puck bm_fable}; do
      comparator "$corpus" "$voice" "$pass" || { log "comparator failed; stopping"; exit 1; }
      candidate "$corpus" "$voice" "$pass" candidate || exit 1
      if [ "$voice" = af_heart ]; then
        candidate "$corpus" "$voice" "$pass" candidate-publisher-phonemes "$EV/comparator/$corpus/$voice/t1/pass$pass/pass.json" || exit 1
      fi
    done
  done
done
log "all dev-box passes present under $EV"
