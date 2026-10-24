#!/usr/bin/env bash
# Report the external oracle's revision, and refuse a stale one.
#
# Every cross-implementation claim in this tree names a llama.cpp revision, because the answer to
# "does it implement this architecture" changes week to week. MEASURED 2026-10-24: the local clone
# was pinned at a58222229 (2026-07-15), and at that revision `maple`, `mellum`, `laguna` and
# `bailingmoe3` are ABSENT -- `mellum` appears only as a pretokenizer in llama-vocab.cpp. Upstream
# that day implemented all four. Checking the checkout on disk instead of upstream would have
# recorded four false negatives and plausibly cancelled four ports.
#
# So: read upstream, not the clone, and if the clone is what you are about to compare against,
# bring it current first and record the revision beside the result.
#
#   scripts/oracle-revision.sh            report, and fail if behind upstream
#   scripts/oracle-revision.sh --update   fast-forward first, then report
set -euo pipefail

ORACLE="${LLAMA_CPP_DIR:-$HOME/Code/llama.cpp}"
[ -d "$ORACLE/.git" ] || { echo "No llama.cpp clone at $ORACLE; set LLAMA_CPP_DIR." >&2; exit 1; }

dirty=$(git -C "$ORACLE" status --porcelain | wc -l | tr -d ' ')
if [ "$dirty" != "0" ]; then
  echo "$ORACLE has $dirty uncommitted change(s); refusing to touch it." >&2
  echo "An oracle with local edits is not the revision its sha names." >&2
  exit 1
fi

if [ "${1:-}" = "--update" ]; then
  git -C "$ORACLE" fetch origin --quiet
  git -C "$ORACLE" merge --ff-only origin/master --quiet
fi

git -C "$ORACLE" fetch origin --quiet
local_rev=$(git -C "$ORACLE" rev-parse HEAD)
upstream_rev=$(git -C "$ORACLE" rev-parse origin/master)
behind=$(git -C "$ORACLE" rev-list --count HEAD..origin/master)

echo "oracle:   $ORACLE"
echo "revision: $local_rev"
echo "date:     $(git -C "$ORACLE" log -1 --format=%cd --date=short)"

if [ "$local_rev" != "$upstream_rev" ]; then
  echo "STALE: $behind commit(s) behind origin/master ($upstream_rev)." >&2
  echo "Run with --update, or record this revision deliberately and say why." >&2
  exit 1
fi
echo "current: matches origin/master"
