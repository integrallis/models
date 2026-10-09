#!/bin/bash
# Bump every version-bearing file for a release, then let the build verify it.
#
# `verifyReleaseMetadata` checks fourteen places and reports them one at a time, so cutting 0.3.55
# by hand meant eight failed runs to discover the list. The list lives here now. It is derived
# from the task's own file set in build.gradle.kts -- if that task gains a file, add it here and
# the next release will not rediscover it the slow way.
#
# Deliberately NOT a git operation: it edits the working tree and verifies. Committing, branching
# and tagging stay with the caller, because a release branch must carry only the release and that
# is a judgement this script should not make.
#
# Usage: scripts/prepare-release.sh <from-version> <to-version> [--changelog-date YYYY-MM-DD]
set -u

FROM=${1:-}
TO=${2:-}
if [ -z "$FROM" ] || [ -z "$TO" ]; then
  echo "usage: $0 <from-version> <to-version> [--changelog-date YYYY-MM-DD]" >&2
  exit 2
fi
DATE=$(date -u +%Y-%m-%d)
if [ "${3:-}" = "--changelog-date" ] && [ -n "${4:-}" ]; then
  DATE=$4
fi
cd "$(dirname "$0")/.." || exit 1

# Every file verifyReleaseMetadata checks, plus the two published-coordinate READMEs it validates.
# benchmark-results/** and scripts/fleet/** are deliberately absent: they name the version that was
# MEASURED and the payloads that were DEPLOYED, and rewriting either falsifies a record.
FILES=(
  gradle.properties
  docs/content/antora.yml
  docs/package.json
  docs/package-lock.json
  docs/landing/index.html
  notebooks/.env.example
  notebooks/docker-compose.yml
  notebooks/jupyter/prepare-classpath.sh
  notebooks/README.md
  backend-native/src/main/rust/model-kernels/Cargo.toml
  backend-native/src/main/rust/model-kernels/Cargo.lock
  models-backend-apple/README.md
  models-rag/README.md
)

escaped_from=${FROM//./\\.}
changed=0
for file in "${FILES[@]}"; do
  if [ ! -f "$file" ]; then
    echo "  MISSING $file -- the file set is stale, fix this script" >&2
    exit 1
  fi
  before=$(grep -c "$escaped_from" "$file" 2>/dev/null || echo 0)
  if [ "$before" -eq 0 ]; then
    printf '  %-56s no %s reference\n' "$file" "$FROM"
    continue
  fi
  perl -pi -e "s/\Q$FROM\E/$TO/g" "$file"
  after=$(grep -c "$(printf '%s' "$TO" | sed 's/\./\\./g')" "$file" 2>/dev/null || echo 0)
  printf '  %-56s %s -> %s\n' "$file" "$before" "$after"
  changed=$((changed+1))
done

# The changelog is a roll, not a substitution: Unreleased stays and a dated section appears below it.
if grep -q "^## \[$TO\]" CHANGELOG.md; then
  echo "  CHANGELOG.md already carries [$TO]"
else
  perl -pi -e "s/^## \[Unreleased\]\n/## [Unreleased]\n\n## [$TO] - $DATE\n/ if \$. < 40" CHANGELOG.md
  grep -q "^## \[$TO\] - $DATE" CHANGELOG.md \
    && echo "  CHANGELOG.md                                             rolled Unreleased -> [$TO] - $DATE" \
    || { echo "  CHANGELOG.md roll FAILED -- is there an '## [Unreleased]' heading?" >&2; exit 1; }
fi

echo
echo "  $changed files bumped; verifying"
./gradlew verifyReleaseMetadata -q
rc=$?
if [ "$rc" -eq 0 ]; then
  echo "  verifyReleaseMetadata passed"
else
  echo "  verifyReleaseMetadata FAILED -- it names the file it wants; add it to FILES above" >&2
fi
echo
echo "  remaining references to $FROM outside the records that must keep it:"
grep -rln "$escaped_from" --include="*.md" --include="*.json" --include="*.yml" --include="*.html" \
  --include="*.sh" --include="*.kts" --include="*.toml" --include="*.example" --include="*.properties" . 2>/dev/null \
  | grep -vE "/build/|node_modules|^\./benchmark-results|^\./release-evidence|^\./scripts/fleet|^\./\.claude|CHANGELOG" \
  | sed 's/^/    /' || true
exit $rc
