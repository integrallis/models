#!/bin/bash
# usage: run.sh <label> <mode> <model> <decode> [jvm flags...]
S=$SCRATCH
label=$1; mode=$2; model=$3; decode=$4; shift 4
CP="$S/exp/classes:$(cat $S/cp.txt)"
LIB=$HOME/Code/java-ai/projects/models-loader-cliffs/backend-native/build/rust-target/release/libjmodels_kernels.dylib
out=$S/exp/$label
mkdir -p $out
java --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED -Xmx8g \
  -XX:StartFlightRecording:filename=$out/run.jfr,settings=default \
  -Dprobe.library=$LIB -Dmodels.native.kernels.library=$LIB "$@" \
  -cp "$CP" CliffProbe $mode $model $decode > $out/stdout.txt 2> $out/stderr.txt
echo "exit=$?" >> $out/stdout.txt
echo "flags: $*" > $out/config.txt
echo "mode=$mode model=$model decode=$decode" >> $out/config.txt
jfr print --events com.integrallis.models.PerformanceCliff $out/run.jfr > $out/events.txt 2>&1
echo "== $label ($*)"
grep PROBE $out/stdout.txt | grep -v greedy
grep exit= $out/stdout.txt
echo "jfr-events: $(grep -c 'com.integrallis.models.PerformanceCliff {' $out/events.txt)"; grep 'reason = ' $out/events.txt | sort | uniq -c
