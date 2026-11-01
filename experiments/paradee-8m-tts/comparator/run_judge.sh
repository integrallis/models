#!/usr/bin/env bash
# Judge every single-thread pass directory under an evidence root that has no judge.json yet.
# The judge is slow (Whisper base on CPU); thread count does not change what is said, so the
# six-thread passes are timing evidence only and are not judged.
#
#   ./run_judge.sh [EVIDENCE_ROOT]      # default ../evidence/comparator
set -euo pipefail
cd "$(dirname "$0")"
root="${1:-../evidence/comparator}"
dirs=()
while IFS= read -r d; do
  # A pass directory counts once its pass.json exists; before that it is still being written.
  [ -f "$d/pass.json" ] && [ ! -f "$d/judge.json" ] && dirs+=("$d")
done < <(find "$root" -type d -path '*/t1/pass*' | sort)
if [ ${#dirs[@]} -eq 0 ]; then
  echo "nothing to judge under $root"
  exit 0
fi
echo "judging ${#dirs[@]} pass directories"
.venv/bin/python judge.py "${dirs[@]}"
