#!/usr/bin/env bash
# Judge completed pass directories as they appear while the arms are still running, one judge
# process at a time. Ends when the arms runner has exited and nothing is left to judge.
#
#   ./run_judge_loop.sh [EVIDENCE_ROOT]      # default ../evidence
set -uo pipefail
cd "$(dirname "$0")"
EV="$(cd "${1:-../evidence}" && pwd)"
log(){ echo "$(date -u +%FT%TZ) $*"; }
while true; do
  judged_something=0
  for root in "$EV/devbox/comparator" "$EV/devbox/candidate" "$EV/devbox/candidate-publisher-phonemes" "$EV/judge-sanity-soprano"; do
    [ -d "$root" ] || continue
    out=$(./run_judge.sh "$root" 2>&1)
    if ! grep -q "nothing to judge" <<<"$out"; then
      judged_something=1
      grep -E "WER|judging|Error|Traceback" <<<"$out" | sed "s|$EV/||"
    fi
  done
  if ! pgrep -f run_devbox_arms.sh > /dev/null && [ $judged_something -eq 0 ]; then
    log "arms finished and nothing left to judge"
    break
  fi
  sleep 120
done
