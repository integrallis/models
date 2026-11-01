# The two-arm measurement protocol, in one place, because an ad-hoc runner drifted from it.
#
# Source this from any script that measures a candidate against a comparator. It exists because
# `benchmark-results/2026-10-31-maple-qualification/run-two-arm.sh` was written by hand to drive a
# RunPod box (the fleet worker bootstraps itself from S3 and cannot run standalone), and it got two
# things wrong that nothing checked:
#
#   1. It passed `--model` but not `--artifact` to the comparator arm. A non-in-process backend only
#      hashes an artifact it was given, so the report carried a null `artifactSha256`, and
#      RagProductionQualificationPolicy excludes any comparator whose sha does not equal the
#      candidate's -- "artifact SHA-256 differs". Every such run can only answer
#      NO_COMPARABLE_BASELINE. The reports looked complete, so three runs' worth of throughput
#      ratios were read straight out of the two files by hand and reported as though a gate had
#      accepted them. None had.
#   2. It used `--warmups 0 --iterations 1`. Every comparator arm in the certified catalog -- 80
#      ollama, 35 llama.cpp, 1 transformers -- ran at `--warmups 1 --iterations 3`, so a 0/1 ratio
#      is not comparable with the 0.84-1.00 figures the qualified entries carry. It also leaves JIT
#      compilation inside the candidate's measured window while the comparator, a C++ server, pays
#      no such cost: on lfm2moe the first case took 25067 ms against an 8-11 s steady state.
#
# Keep these values equal to the ones in qual-worker-two-arm.sh. That worker is deliberately
# self-contained (it is fetched from S3 as a single file and cannot source anything), so the two
# cannot share code -- TwoArmProtocolTest fails the build if they disagree.
TWO_ARM_WARMUPS=1
TWO_ARM_ITERATIONS=3

# Both arms must be told the thread count explicitly. sameWorkload() compares `threads`, so letting
# the comparator default to the host core count rejects every comparison with
# "benchmark workload differs".
two_arm_common_args() {
  local workload="$1" template="$2" topk="$3" context="$4" threads="$5" maxtok="$6"
  printf '%s' "--workload $workload --prompt-template $template --top-k $topk"
  printf '%s' " --context $context --threads $threads --max-tokens $maxtok"
  printf '%s' " --warmups $TWO_ARM_WARMUPS --iterations $TWO_ARM_ITERATIONS"
}
