#!/bin/bash
# Side-by-side: same document, same ten questions, timings measured in-process on each side.
#
# Both arms are re-run every time; nothing here is a cached number. The Jev arm runs on this machine
# so TYPESAFE_API_KEY never leaves it. The Harriet arm runs on the host named in HARRIET_HOST_LABEL
# and the document never leaves that box.
#
# What changed from the 2026-09-21 recording: that one ran Granite 4.1 3B plus an 82 KB squad2-fitted
# Noul head. Neither ships. Harriet is frozen Qwen3.5-4B read through letter logits with no trained
# head, so the old arm could not be re-run -- the old video advertised a product that does not exist.
set -u

: "${SSH_HOST:?set SSH_HOST, e.g. root@1.2.3.4}"
: "${SSH_PORT:=22}"
: "${HARRIET_HOST_LABEL:?set HARRIET_HOST_LABEL, the host being demonstrated}"
: "${TYPESAFE_API_KEY:?set TYPESAFE_API_KEY (stays on this machine)}"
THREADS="${THREADS:-16}"
GOLD="1 1 1 1 1 1 1 1 0 0"   # which questions the contract actually answers
HERE="$(cd "$(dirname "$0")" && pwd)"

c() { printf "\033[%sm%s\033[0m" "$1" "$2"; }
rule() { printf '  %s\n' "────────────────────────────────────────────────────────────────────"; }

clear
echo
c "1;36" "  HARRIET  vs  TYPESAFE JEV"; echo
echo "  Master services agreement, 404 tokens, 10 questions, identical prompts."
echo "  Every timing is measured inside the process that answers."
c "2" "  We do not know what hardware Jev runs on. Ours is named below."; echo
rule
echo

c "1;33" "  ── TypeSafe Jev (hosted API, hardware unknown) ──"; echo
python3 "$HERE/jev_arm.py" "$HERE/document.txt" "$HERE/questions.txt" /tmp/sbs-jev.json

c "1;32" "  ── Harriet (frozen Qwen3.5-4B, letter logits, CPU) ──"; echo
c "2" "     $HARRIET_HOST_LABEL"; echo
ssh -o ConnectTimeout=20 -i ~/.ssh/id_ed25519 -p "$SSH_PORT" "$SSH_HOST" \
  "cd /root/demo && /root/jdk/bin/java -Xmx24g --add-modules jdk.incubator.vector \
     --enable-native-access=ALL-UNNAMED \
     -Dmodels.purejava.maxContextLength=4096 \
     -Dmodels.native.kernels.threads=$THREADS -Dvectors.gguf.threads=$THREADS \
     -Dharriet.demo.host='$HARRIET_HOST_LABEL' \
     -cp 'classes:lib/*' demo.HarrietBriefing document.txt questions.txt /root/demo/harriet.json" \
  2>/dev/null | grep -vE "^WARNING|^INFO|^$"
# Explicit host and port. An earlier version derived the scp target from SSH_TARGET with a string
# substitution, which produced a host:path spec, hung under a pty, and left a truncated recording
# that was missing the entire comparison table.
scp -q -o ConnectTimeout=20 -i ~/.ssh/id_ed25519 -P "$SSH_PORT" \
  "$SSH_HOST:/root/demo/harriet.json" /tmp/sbs-harriet.json

rule
python3 "$HERE/compare.py" /tmp/sbs-jev.json /tmp/sbs-harriet.json "$GOLD"
