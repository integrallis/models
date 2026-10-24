#!/usr/bin/env bash
# Evidence fields must be measured, not asserted.
#
# Three separate times on 2026-10-23 a field that reads as evidence turned out to be a constant:
#
#   qual-worker-two-arm.sh  `[ "1" = "1" ]`          HOLD_ON_FAIL was named in the comment and in
#                                                    the log line and read nowhere, so every failed
#                                                    shard held 45 minutes of billed time.
#   HarrietBriefing.java    `"prefix_shared":true`   unfalsifiable, and ambiguous between two
#                                                    different reuse mechanisms -- which is exactly
#                                                    how it was misread.
#   LayaLatency.java        `"prefix_shared": false` written while criticising the pattern.
#
# A constant in an evidence file is worse than a missing field: it reads as a measurement and
# cannot be falsified by any run. This refuses the shape, in the files whose job is to record
# measurements. It is deliberately narrow -- fixtures, stubs and canned tool replies are inputs,
# not evidence, so only evidence writers and worker scripts are scanned.
set -euo pipefail
cd "$(dirname "$0")/.."

status=0

# 1. Boolean literals in a JSON field position, in evidence writers and fleet workers.
while IFS= read -r hit; do
  echo "hardcoded boolean in an evidence field: $hit" >&2
  status=1
done < <(
  grep -rnE '\\"[A-Za-z_][A-Za-z_0-9]*\\" *: *(true|false)' \
    --include='*.java' benchmark-results scripts/fleet models-decisions/benchmark-results 2>/dev/null \
    | grep -vE '/build/|Stub[A-Za-z]*\.java' || true
)

# 2. Tautological conditions in the fleet workers, which is how HOLD_ON_FAIL hid.
while IFS= read -r hit; do
  echo "tautological condition (a toggle that cannot be off): $hit" >&2
  status=1
done < <(
  grep -rnE '\[ *"(1|0|true|false|yes|no)" *= *"(1|0|true|false|yes|no)" *\]|\[ *1 *-eq *1 *\]' \
    scripts/fleet 2>/dev/null | grep -v '^\s*#' | grep -vE ':\s*#' || true
)

if [ "$status" -eq 0 ]; then
  echo "evidence-literal check: no hardcoded evidence fields or tautological toggles"
fi
exit "$status"
