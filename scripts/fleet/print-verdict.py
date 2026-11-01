#!/usr/bin/env python3
"""Print a qualification verdict, so a runner never reports a ratio it computed itself.

Split out of two-arm-pod.sh because the numbers that matter are the policy's: the gate excludes a
comparator for reasons (artifact sha, host, workload, prompt coverage) that are invisible in the two
reports, and dividing their throughput by hand produces a plausible figure no gate accepted.
"""
import json
import sys

with open(sys.argv[1]) as handle:
    qualification = json.load(handle)["qualification"]

print("### verdict:", qualification["verdict"], "qualified=", qualification["qualified"])
print(
    "### absoluteTier: %s" % qualification.get("absoluteTier"),
)
# modelAnswerRate and modelAnswerCorrectRate, never correctAnswerRate: the latter scores the
# pipeline, because GroundedAnswerPolicy substitutes text extracted from the retrieved document
# when a generation is screened out, so a model that never answers can still read as 1.000.
print(
    "### contribution: answerRate=%.4f (min %.4f)  correctRate=%.4f (min %.4f)"
    % (
        qualification["modelAnswerRate"],
        qualification["minimumModelAnswerRate"],
        qualification["modelAnswerCorrectRate"],
        qualification["minimumModelAnswerCorrectRate"],
    )
)
for comparison in qualification.get("comparisons", []):
    print(
        "### vs %s: decodeRatio=%.4f e2eRatio=%.4f qualified=%s"
        % (
            comparison["comparatorBackend"],
            comparison["decodeThroughputRatio"],
            comparison["endToEndLatencyRatio"],
            comparison["qualified"],
        )
    )
for backend, reason in (qualification.get("exclusions") or {}).items():
    print("### EXCLUDED %s: %s" % (backend, reason))
if not qualification.get("comparisons"):
    print("### NO COMPARISON WAS ADMITTED -- this run qualifies nothing")
