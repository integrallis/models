#!/usr/bin/env bash
# Keep up to PARALLEL judge processes busy on completed, unjudged pass directories under the given
# roots, pass-1 directories first, until the roots are fully judged and no runner is alive.
# Written for the bash 3.2 macOS ships: no associative arrays, no mapfile.
#
#   PARALLEL=4 ./judge_dispatch.sh ROOT [ROOT ...]
set -u
cd "$(dirname "$0")"
PARALLEL=${PARALLEL:-4}
log(){ echo "$(date -u +%FT%TZ) $*"; }
running(){ jobs -rp | wc -l | tr -d ' '; }
while true; do
  launched=0
  for d in $(for root in "$@"; do find "$root" -type d -path '*/t1/pass*' 2>/dev/null; done | awk -F/ '{print $NF, $0}' | sort | awk '{print $2}'); do
    [ "$(running)" -ge "$PARALLEL" ] && break
    [ -f "$d/pass.json" ] && [ ! -f "$d/judge.json" ] && [ ! -f "$d/.judging" ] || continue
    touch "$d/.judging"
    ( THREADS=3 ./judge_dirs.sh "$d" > "$d/judge.log" 2>&1; rm -f "$d/.judging" ) &
    launched=$((launched+1))
    log "judging ${d#*/devbox/} ($(running) running)"
  done
  remaining=0
  for p in $(for root in "$@"; do find "$root" -name pass.json 2>/dev/null; done); do
    [ -f "$(dirname "$p")/judge.json" ] || remaining=$((remaining+1))
  done
  if [ "$remaining" -eq 0 ] && [ "$(running)" -eq 0 ] && ! pgrep -f run_candidate_passes.sh > /dev/null; then
    log "all judged"; break
  fi
  sleep 60
done
