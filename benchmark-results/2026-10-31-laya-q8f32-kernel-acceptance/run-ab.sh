#!/bin/zsh
# Both Laya arms back to back on an otherwise idle host. Sequential on purpose: the earlier
# attempt ran them concurrently with a download and the load average hit 328 on a 12-thread box,
# which makes a latency ratio meaningless. Accuracy is unaffected by contention; p50 is not.
set -e
S="$1"; CP="$2"; COHORT="$3"; ART="$4"
echo "### load before: $(sysctl -n vm.loadavg)"
echo "### ARM A: Rust Q8_0 x F32-activation kernel (exact activations)"
java --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED \
  -cp "$S/probe:$CP" LayaLatency "$ART" "$COHORT" "$S/ab-kernel.tsv" native 2>&1 | grep -v -E "WARNING|INFO|^Oct "
echo "### load between: $(sysctl -n vm.loadavg)"
echo "### ARM B: pure-Java exact path (the shipped default)"
java --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED \
  -cp "$S/probe:$CP" LayaLatency "$ART" "$COHORT" "$S/ab-java.tsv" 2>&1 | grep -v -E "WARNING|INFO|^Oct "
echo "### load after: $(sysctl -n vm.loadavg)"
