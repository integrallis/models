#!/bin/bash
# Exercises the guards inside qual-worker-two-arm.sh against the real code.
#
# The worker is fetched from S3 as a single self-contained file, so it cannot source a helper and
# the guard cannot live in its own module. Instead the guard is bracketed by
# `# >>> BEGIN payload_label_version` / `# <<< END payload_label_version` markers in the worker and
# extracted here, so this test runs the shipped implementation rather than a copy that can drift
# from it.
#
# Run: bash scripts/fleet/qual-worker-guards-test.sh
set -u
WORKER="$(cd "$(dirname "$0")" && pwd)/qual-worker-two-arm.sh"
[ -f "$WORKER" ] || { echo "worker not found at $WORKER"; exit 1; }

BLOCK=$(sed -n '/^# >>> BEGIN payload_label_version$/,/^# <<< END payload_label_version$/p' "$WORKER")
if [ -z "$BLOCK" ]; then
  echo "FAIL: could not extract the payload_label_version block from the worker."
  echo "      The markers were renamed or removed -- this test is now testing nothing, which is"
  echo "      worse than failing, so it fails."
  exit 1
fi
eval "$BLOCK"

fails=0
pass() { printf '  ok   %s\n' "$1"; }
fail() { printf '  FAIL %s\n' "$1"; fails=$((fails+1)); }

expect_accept() {
  if payload_matches_label "$1" "$2"; then pass "payload $1 accepted for label $2"
  else fail "payload $1 rejected for label $2, which agree"; fi
}
expect_reject() {
  if payload_matches_label "$1" "$2"; then fail "payload $1 ACCEPTED for label $2, which disagree"
  else pass "payload $1 rejected for label $2"; fi
}
expect_unparseable() {
  payload_matches_label "$1" "$2"
  if [ $? -eq 2 ]; then pass "refuses unparseable label $2"
  else fail "did not refuse unparseable label $2"; fi
}

echo "a payload whose name carries the labelled version is accepted:"
expect_accept "models-rag-bench-0.3.56-v24.tar" "models@0.3.56+v24-7ac41536f5c2"
expect_accept "models-rag-bench-0.3.50-v23.tar" "models@0.3.50+v23-08d9b5e1cef8"
expect_accept "models-rag-bench-0.3.56-dev.tar" "models@0.3.56-dev+q41-7ac41536f5c2"

echo "a mismatch is rejected -- this is the case that actually happened on 2026-10-09,"
echo "when the committed worker named a 0.3.50 payload while reports were stamped 0.3.56-dev:"
expect_reject "models-rag-bench-0.3.50-v23.tar" "models@0.3.56+v24-7ac41536f5c2"
expect_reject "models-rag-bench-0.3.56-v24.tar" "models@0.3.50+v23-08d9b5e1cef8"

echo "a near miss is still a miss:"
expect_reject "models-rag-bench-0.3.5-v23.tar" "models@0.3.56+v24-abc"

echo "a label that names no version at all cannot be checked, so it is refused rather than passed:"
expect_unparseable "models-rag-bench-0.3.56-v24.tar" "local"
expect_unparseable "models-rag-bench-0.3.56-v24.tar" "models@"

echo "the kernels jar is checked by the same rule, because backend-native's published JAR carries"
echo "no .so at all and a September kernel loaded cleanly against a 0.3.56 library for want of"
echo "this check -- both declare abi=6:"
expect_accept "models-kernels-linux-x86_64-0.3.56.jar" "models@0.3.56+v25-666bb48c61e0"
expect_reject "models-kernels-linux-x86_64.jar"        "models@0.3.56+v25-666bb48c61e0"
expect_reject "models-kernels-linux-x86_64-0.3.54.jar" "models@0.3.56+v25-666bb48c61e0"

echo
echo "the worker must require all three inputs and refuse to guess any of them:"
for required in QUAL_PAYLOAD QUAL_KERNELS QUAL_BACKEND_VERSION; do
  if grep -q "z \"\${$required:-}\"" "$WORKER"; then pass "refuses without $required"
  else fail "does not require $required"; fi
done
if grep -q 'CP="\$DIST/lib/\*:/work/\$KERNELS"' "$WORKER"; then
  pass "classpath uses the named kernels jar, not a fixed name"
else
  fail "classpath does not use \$KERNELS -- a fixed name can go stale unnoticed"
fi

echo
if [ "$fails" -eq 0 ]; then echo "all guard checks passed"; exit 0; fi
echo "$fails guard check(s) failed"; exit 1
