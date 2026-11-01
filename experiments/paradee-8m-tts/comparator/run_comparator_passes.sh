#!/usr/bin/env bash
# Run the comparator arm over every corpus, voice, thread count and pass the protocol names.
#
# Timing is the measurement, so run this with nothing else on the machine. Finished passes are
# skipped, so an interrupted run resumes where it stopped.
#
#   ./run_comparator_passes.sh [EVIDENCE_ROOT]      # default ../evidence/comparator
#   THREADS="1" PASSES="1 2 3" ./run_comparator_passes.sh
set -euo pipefail
cd "$(dirname "$0")"
root="${1:-../evidence/comparator}"
for threads in ${THREADS:-1 6}; do
  for pass in ${PASSES:-1 2 3}; do
    for corpus in ${CORPORA:-wikitext103-heldout-200 harvard-200}; do
      for voice in ${VOICES:-af_heart bf_isabella am_puck bm_fable}; do
        out="$root/$corpus/$voice/t$threads/pass$pass"
        if [ -f "$out/pass.json" ]; then
          echo "skip $out (done)"
          continue
        fi
        echo "=== $corpus $voice threads=$threads pass=$pass -> $out"
        mkdir -p "$out" && uptime | sed 's/.*load averages*: *//' > "$out/load.txt"
        .venv/bin/python comparator_arm.py --corpus "../corpus/$corpus.json" --voice "$voice" \
          --threads "$threads" --pass "$pass" --out "$out" 2> >(tail -1 >&2)
      done
    done
  done
done
echo "all passes present under $root"
