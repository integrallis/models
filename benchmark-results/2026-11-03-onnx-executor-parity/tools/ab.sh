#!/bin/bash
# usage: ab.sh <label> [shadow-jar...]  -- runs the released 0.5.12 arm on 50 Harvard sentences, af_heart
W=/private/tmp/claude-501/-Users-briansam-bodden-Code-java-ai/b2264b88-5cfa-4127-acd9-1f3058f0b59a/scratchpad/ortprof
A=/Users/briansam-bodden/Code/java-ai/projects/models-paradee/experiments/paradee-8m-tts/comparator/.artifacts
label=$1; shift; pre=""; for j in "$@"; do pre="$pre$j:"; done
out=$W/run-$label-$RANDOM; mkdir -p $out
java --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED -Dmodels.version=0.5.12 $JOPTS \
  -Donnx.prof.out=$out/prof.json -cp "$pre$(cat $W/cp.txt | tr : "\n" | grep -v "${DROP:-NOTHING}" | paste -sd: -)" ParadeeLatencyArm \
  $A/onnx_paradee_int8.onnx $A/config.json $W/harvard-50.json af_heart 1 $out 2>&1 | grep PARADEE_ARM | sed "s/^/$label /"
echo "$out"
