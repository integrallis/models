set -euo pipefail
STUDY="$PWD"
eval "$(sed -n '/^arm_field() {/,/^}/p' host-run.sh)"
for f in .requiresFrozen .spec .thinking .modelSet .temperature; do printf 'A %s -> %s\n' "$f" "$(arm_field A "$f")"; done
printf 'F-tuned .requiresFrozen -> %s\n' "$(arm_field F-tuned .requiresFrozen)"
if arm_field NOPE .spec >/dev/null 2>&1; then echo "BUG: unknown arm accepted"; else echo "unknown arm rejected"; fi
if arm_field A .nosuchfield >/dev/null 2>&1; then echo "BUG: missing field accepted"; else echo "missing field rejected"; fi
echo ALL-OK
