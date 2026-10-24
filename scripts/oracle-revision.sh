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

# Both checks below report, and the script exits once at the end. Gating the binary check behind
# the revision check would have made it unreachable on exactly the clones that need it most, and a
# check that cannot run is worse than no check -- it reads as a passing one.
failed=0

if [ "$local_rev" != "$upstream_rev" ]; then
  echo "revision: STALE -- $behind commit(s) behind origin/master ($upstream_rev)." >&2
  echo "Run with --update, or record this revision deliberately and say why." >&2
  failed=1
else
  echo "current:  matches origin/master"
fi

# A current checkout is not a current ORACLE. The binaries are what actually answers a question,
# and they are built, not checked out.
#
# MEASURED 2026-10-24: with the clone at today's master, build/bin/llama-simple was still the
# August build. It aborted on maple with a ggml ABI mismatch -- 0.16.0 against the tree's 0.27.0 --
# and that abort was filed as issue #296, "maple has no usable oracle: llama.cpp aborts loading the
# published artifact", against the artifact and upstream. Both were fine. Rebuilding fixed it, and
# the issue was invalid. Two stale dylibs were sitting beside each other in build/bin at the time
# and I read them as cosmetic.
#
# So the revision check alone is not enough to clear an oracle for use.
BIN="$ORACLE/build/bin"
head_epoch=$(git -C "$ORACLE" log -1 --format=%ct)
stale_bins=""
if [ -d "$BIN" ]; then
  for binary in "$BIN"/llama-* "$BIN"/libggml*.dylib "$BIN"/libggml*.so; do
    [ -f "$binary" ] || continue
    built=$(stat -f %m "$binary" 2>/dev/null || stat -c %Y "$binary" 2>/dev/null || echo 0)
    if [ "$built" -lt "$head_epoch" ]; then
      stale_bins="$stale_bins  $(basename "$binary")  built $(date -r "$built" +%Y-%m-%d 2>/dev/null)
"
    fi
  done
fi

if [ -n "$stale_bins" ]; then
  echo "binaries: STALE -- older than the checked-out revision:" >&2
  printf '%s' "$stale_bins" >&2
  echo "A binary built from another revision is not this oracle. Rebuild before comparing:" >&2
  echo "  cmake -B '$ORACLE/build' -S '$ORACLE' && cmake --build '$ORACLE/build' -j" >&2
  failed=1
elif [ -d "$BIN" ]; then
  echo "binaries: built at or after this revision"
else
  echo "binaries: none built yet at $BIN"
fi

exit "$failed"
