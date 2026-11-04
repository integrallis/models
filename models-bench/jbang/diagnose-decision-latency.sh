#!/usr/bin/env bash
# Why does one typed decision cost what it costs ON THIS MACHINE?
#
# Reports the CPU topology, then measures the decision at several thread counts and, on a
# heterogeneous CPU, pinned to each class of core separately. The pinned arms are the point: if a
# machine mixes fast and slow cores, a pool that spreads equal chunks over all of them waits at
# every barrier for the slowest one, and the only way to see that is to run the fast cores alone.
#
# Usage:  ./diagnose-decision-latency.sh [path-to-DecideLatency.java]
set -uo pipefail
SCRIPT=${1:-$(dirname "$0")/DecideLatency.java}
OUT=${OUT:-./decision-diagnosis}
mkdir -p "$OUT"
REPORT="$OUT/report.txt"
: > "$REPORT"

say() { echo "$*" | tee -a "$REPORT"; }
have() { command -v "$1" >/dev/null 2>&1; }

[ -f "$SCRIPT" ] || { echo "cannot find DecideLatency.java at $SCRIPT" >&2; exit 2; }
have jbang || { echo "jbang is required: curl -Ls https://sh.jbang.dev | bash -s - app setup" >&2; exit 2; }

say "=============================================================="
say " decision latency diagnosis — $(date -u +%Y-%m-%dT%H:%M:%SZ)"
say "=============================================================="
say ""
say "--- topology ---"
say "nproc:                 $(nproc 2>/dev/null || echo '?')"
if have lscpu; then
  lscpu > "$OUT/lscpu.txt" 2>&1
  say "model:                 $(awk -F: '/Model name/{gsub(/^ +/,"",$2); print $2; exit}' "$OUT/lscpu.txt")"
  say "sockets:               $(awk -F: '/^Socket\(s\)/{gsub(/ /,"",$2); print $2; exit}' "$OUT/lscpu.txt")"
  say "cores per socket:      $(awk -F: '/^Core\(s\) per socket/{gsub(/ /,"",$2); print $2; exit}' "$OUT/lscpu.txt")"
  say "threads per core:      $(awk -F: '/^Thread\(s\) per core/{gsub(/ /,"",$2); print $2; exit}' "$OUT/lscpu.txt")"
  lscpu -e > "$OUT/lscpu-e.txt" 2>&1 || true
fi

# Intel hybrid parts publish one directory per core type; ARM big.LITTLE publishes cpu_capacity.
HYBRID=no
P_LIST=""
E_LIST=""
if [ -d /sys/devices/system/cpu/types ]; then
  say ""
  say "core types under /sys/devices/system/cpu/types:"
  for t in /sys/devices/system/cpu/types/*; do
    [ -d "$t" ] || continue
    name=$(basename "$t")
    list=$(cat "$t/cpulist" 2>/dev/null)
    say "  $name -> $list"
    case "$name" in
      *core*) P_LIST="$list" ;;
      *atom*) E_LIST="$list" ;;
    esac
  done
  [ -n "$P_LIST" ] && [ -n "$E_LIST" ] && HYBRID=yes
fi

# Max frequency per logical cpu: a second, independent read on heterogeneity.
if ls /sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq >/dev/null 2>&1; then
  say ""
  say "distinct max frequencies (kHz -> how many logical cpus):"
  for f in /sys/devices/system/cpu/cpu*/cpufreq/cpuinfo_max_freq; do cat "$f" 2>/dev/null; done \
    | sort -n | uniq -c | awk '{printf "  %s kHz -> %s cpus\n", $2, $1}' | tee -a "$REPORT"
fi
if ls /sys/devices/system/cpu/cpu0/cpu_capacity >/dev/null 2>&1; then
  say ""
  say "distinct cpu_capacity values (ARM big.LITTLE):"
  for f in /sys/devices/system/cpu/cpu*/cpu_capacity; do cat "$f" 2>/dev/null; done \
    | sort -n | uniq -c | awk '{printf "  capacity %s -> %s cpus\n", $2, $1}' | tee -a "$REPORT"
fi
say ""
say "heterogeneous CPU: $HYBRID"
[ "$HYBRID" = yes ] && { say "  performance cores: $P_LIST"; say "  efficiency  cores: $E_LIST"; }

say ""
say "--- thermal / power state before we start ---"
if have sensors; then sensors 2>/dev/null | head -20 | tee -a "$REPORT" >/dev/null; say "(sensors captured to report)"; fi
say "governor:              $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo '?')"
say "load average:          $(cut -d' ' -f1-3 /proc/loadavg 2>/dev/null || echo '?')"

run() {
  local tag="$1"; shift
  echo "" | tee -a "$REPORT" >/dev/null
  printf '%-26s ' "$tag" | tee -a "$REPORT"
  local log="$OUT/$tag.log"
  if ! "$@" > "$log" 2>&1; then
    say "FAILED (see $log)"
    return
  fi
  local p50 p95 first
  p50=$(sed -nE 's/.*warm: p50 ([0-9.]+) ms.*/\1/p' "$log" | head -1)
  p95=$(sed -nE 's/.*warm: p50 [0-9.]+ ms, p95 ([0-9.]+) ms.*/\1/p' "$log" | head -1)
  first=$(sed -nE 's/.*first decision: ([0-9]+) ms.*/\1/p' "$log" | head -1)
  say "p50=${p50:-?} ms  p95=${p95:-?} ms  first=${first:-?} ms"
}

say ""
say "--- priming (downloads ~1 GB once, builds the script) ---"
jbang -Dvectors.gguf.threads=4 -Dmodels.native.kernels.threads=4 "$SCRIPT" > "$OUT/prime.log" 2>&1
say "prime: $(sed -nE 's/.*warm: p50 ([0-9.]+) ms.*/p50=\1 ms/p' "$OUT/prime.log" | head -1)"

say ""
say "--- thread count sweep (both pools together) ---"
TOTAL=$(nproc 2>/dev/null || echo 8)
for t in $TOTAL 24 16 12 8 6 4 2 1; do
  [ "$t" -gt "$TOTAL" ] && continue
  run "threads-$t" jbang -Dvectors.gguf.threads=$t -Dmodels.native.kernels.threads=$t "$SCRIPT" \
      --json "$OUT/threads-$t.json"
done

if [ "$HYBRID" = yes ] && have taskset; then
  P_COUNT=$(taskset -c "$P_LIST" nproc 2>/dev/null || echo 0)
  E_COUNT=$(taskset -c "$E_LIST" nproc 2>/dev/null || echo 0)
  say ""
  say "--- pinned to one class of core (P=$P_COUNT logical, E=$E_COUNT logical) ---"
  say "If P-only beats everything unpinned, the mixed pool is barrier-bound on its slowest core."
  for t in 8 4; do
    [ "$P_COUNT" -ge "$t" ] && run "pcores-$t" taskset -c "$P_LIST" \
        jbang -Dvectors.gguf.threads=$t -Dmodels.native.kernels.threads=$t "$SCRIPT" --json "$OUT/pcores-$t.json"
    [ "$E_COUNT" -ge "$t" ] && run "ecores-$t" taskset -c "$E_LIST" \
        jbang -Dvectors.gguf.threads=$t -Dmodels.native.kernels.threads=$t "$SCRIPT" --json "$OUT/ecores-$t.json"
  done
  # All P logical threads, which is the count a topology-aware pool should pick on this machine.
  [ "$P_COUNT" -ge 1 ] && run "pcores-all-$P_COUNT" taskset -c "$P_LIST" \
      jbang -Dvectors.gguf.threads=$P_COUNT -Dmodels.native.kernels.threads=$P_COUNT "$SCRIPT" \
      --json "$OUT/pcores-all.json"
fi

say ""
say "--- repeat of the best and the default, to size run-to-run variance ---"
say "(a single run per arm cannot tell a 20% difference from noise)"
run "repeat-default-$TOTAL" jbang -Dvectors.gguf.threads=$TOTAL -Dmodels.native.kernels.threads=$TOTAL "$SCRIPT"
run "repeat-half-$((TOTAL/2))" jbang -Dvectors.gguf.threads=$((TOTAL/2)) -Dmodels.native.kernels.threads=$((TOTAL/2)) "$SCRIPT"

say ""
say "--- after the run ---"
say "load average:          $(cut -d' ' -f1-3 /proc/loadavg 2>/dev/null || echo '?')"
if ls /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq >/dev/null 2>&1; then
  say "current freqs (kHz, first 8 cpus):"
  for c in 0 1 2 3 4 5 6 7; do
    f=/sys/devices/system/cpu/cpu$c/cpufreq/scaling_cur_freq
    [ -r "$f" ] && say "  cpu$c $(cat $f)"
  done
fi
say ""
say "=============================================================="
say "report:   $REPORT"
say "raw logs: $OUT/"
say "=============================================================="
