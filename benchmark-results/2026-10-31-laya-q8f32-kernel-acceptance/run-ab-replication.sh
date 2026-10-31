#!/bin/zsh
# Replication of the A/B. Same order, same host, nothing else running: the first pass ran arm A
# while the load average was still decaying from an earlier thrash, which biases against arm A, so
# a second pass says whether the ratio is stable or whether that bias produced it.
set -e
S="$1"; CP="$2"; COHORT="$3"; ART="$4"
echo "### load before: $(sysctl -n vm.loadavg)"
echo "### ARM A2: Rust Q8_0 x F32-activation kernel"
java --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED \
  -cp "$S/probe:$CP" LayaLatency "$ART" "$COHORT" "$S/ab2-kernel.tsv" native 2>&1 | grep -v -E "WARNING|INFO|^Oct "
echo "### ARM B2: pure-Java exact path"
java --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED \
  -cp "$S/probe:$CP" LayaLatency "$ART" "$COHORT" "$S/ab2-java.tsv" 2>&1 | grep -v -E "WARNING|INFO|^Oct "
echo "### load after: $(sysctl -n vm.loadavg)"
