#!/usr/bin/env bash
# Keep up to PARALLEL judge processes busy on completed, unjudged pass directories under the given
# roots, pass-1 directories first, until the roots are fully judged and no runner is alive.
#
#   PARALLEL=4 ./judge_dispatch.sh ROOT [ROOT ...]
set -uo pipefail
cd "$(dirname "$0")"
PARALLEL=${PARALLEL:-4}
log(){ echo "$(date -u +%FT%TZ) $*"; }
declare -A busy=()
while true; do
  for pid in "${!busy[@]}"; do kill -0 "$pid" 2>/dev/null || unset "busy[$pid]"; done
  mapfile -t todo < <(for root in "$@"; do find "$root" -type d -path '*/t1/pass*' 2>/dev/null; done | sort -t/ -k1,1 | awk -F/ '{print $NF, $0}' | sort | awk '{print $2}')
  launched=0
  for d in "${todo[@]}"; do
    [ ${#busy[@]} -ge $PARALLEL ] && break
    [ -f "$d/pass.json" ] && [ ! -f "$d/judge.json" ] && [ ! -f "$d/.judging" ] || continue
    touch "$d/.judging"
    ( THREADS=3 ./judge_dirs.sh "$d" > "$d/judge.log" 2>&1; rm -f "$d/.judging" ) &
    busy[$!]=$d; launched=$((launched+1))
    log "judging $d (${#busy[@]} running)"
  done
  remaining=$(for root in "$@"; do find "$root" -name pass.json 2>/dev/null; done | while read p; do [ -f "$(dirname $p)/judge.json" ] || echo x; done | wc -l | tr -d ' ')
  if [ "$remaining" = "0" ] && [ ${#busy[@]} -eq 0 ] && ! pgrep -f run_candidate_passes.sh > /dev/null; then
    log "all judged"; break
  fi
  sleep 60
done
