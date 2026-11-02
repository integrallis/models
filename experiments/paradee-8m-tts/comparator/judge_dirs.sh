#!/usr/bin/env bash
# Judge an explicit, prioritized list of pass directories in one process, skipping any that are
# already judged or not yet complete. Several of these run side by side on disjoint lists.
#
#   THREADS=4 ./judge_dirs.sh DIR [DIR ...]
set -uo pipefail
cd "$(dirname "$0")"
todo=()
for d in "$@"; do
  [ -f "$d/pass.json" ] && [ ! -f "$d/judge.json" ] && todo+=("$d")
done
if [ ${#todo[@]} -eq 0 ]; then echo "nothing to judge"; exit 0; fi
echo "$(date -u +%FT%TZ) judging ${#todo[@]} directories with ${THREADS:-4} threads"
OMP_NUM_THREADS=${THREADS:-4} MKL_NUM_THREADS=${THREADS:-4} .venv/bin/python judge.py "${todo[@]}"
echo "$(date -u +%FT%TZ) done"
