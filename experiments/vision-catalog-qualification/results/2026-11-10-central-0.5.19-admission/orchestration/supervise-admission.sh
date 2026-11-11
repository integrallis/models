#!/usr/bin/env bash
set -uo pipefail

output=${1:?Provide the staged runtime directory}
test -f "$output/runtime-receipt.json" || exit 2
mkdir "$output/supervisor-started" || exit 2
# Leave thirty minutes before the inherited worker's existing deletion deadline.
remaining=$(( $(date -u -d '2026-11-11T05:30:00Z' +%s) - $(date -u +%s) ))
limit=14400
if (( remaining < limit )); then limit=$remaining; fi
if (( limit <= 0 )); then
  printf 'Worker deadline reached; no measurement launched.\n' >&2
  exit 2
fi
printf 'Started %s; timeout %s seconds\n' "$(date -u --iso-8601=seconds)" "$limit"
timeout --signal=TERM --kill-after=30s "${limit}s" \
  python3 /home/ubuntu/run-admission.py /home/ubuntu/models-0519 "$output" /home/ubuntu/qual
result=$?
printf '%s\n' "$result" > "$output/exit-code"
printf 'Finished %s; exit %s\n' "$(date -u --iso-8601=seconds)" "$result"
exit "$result"
