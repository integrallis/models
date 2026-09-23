#!/bin/bash
# Side-by-side: same document, same ten questions, timings measured in-process on each side.
set -u
BOX=178.156.173.194
GOLD="1 1 1 1 1 1 1 1 0 0"   # which questions the contract actually answers

c() { printf "\033[%sm%s\033[0m" "$1" "$2"; }
rule() { printf '  %s\n' "────────────────────────────────────────────────────────────────────"; }

clear
echo
c "1;36" "  INTEGRALLIS DECISIONS  vs  TYPESAFE JEV"; echo
echo "  Master Services Agreement, 404 tokens, 10 questions, identical prompts."
echo "  Every timing is measured inside the process that answers."
rule
echo

c "1;33" "  ── TypeSafe Jev (jev-1.13.0, hosted API) ──"; echo
python3 /tmp/demo/jev_arm.py /tmp/demo/document.txt /tmp/demo/questions.txt /tmp/demo/jev-timing.json

c "1;32" "  ── Integrallis Decisions (Granite 4.1 3B + 82 KB head, 8-core CPU) ──"; echo
ssh -o ConnectTimeout=20 root@$BOX '/root/jdk/bin/java --add-modules jdk.incubator.vector \
  --enable-native-access=ALL-UNNAMED -cp "/root/demo/demo-dist/*:/root/demo/demo-dist:/root/demo/native" \
  com.integrallis.models.decisions.BriefingDemo \
  /root/demo/squad2-noul-v0.2.idsn /root/models/granite.gguf \
  /root/demo/document.txt /root/demo/questions.txt /root/demo/decisions-timing-rust.json' 2>/dev/null \
  | grep -vE "^WARNING|^INFO|^$"
scp -q -o ConnectTimeout=20 root@$BOX:/root/demo/decisions-timing-rust.json /tmp/demo/decisions-timing-rust.json

rule
python3 /tmp/demo/compare.py /tmp/demo/jev-timing.json /tmp/demo/decisions-timing-rust.json "$GOLD"
