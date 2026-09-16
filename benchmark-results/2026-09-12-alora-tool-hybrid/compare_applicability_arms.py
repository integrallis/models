#!/usr/bin/env python3
"""Apply the frozen V13 teacher-forced control/treatment gates."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


TRAINING_CONTRACT_KEYS = (
    "epochs",
    "screenSteps",
    "optimizerStepsCompleted",
    "learningRate",
    "gradientAccumulation",
    "maxLength",
    "seed",
)


def _finite_number(value: Any, label: str, failures: list[str]) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        failures.append(f"{label} must be finite")
        return math.nan
    return float(value)


def _evaluation(manifest: dict[str, Any], label: str, failures: list[str]) -> dict[str, Any]:
    try:
        evaluation = manifest["training"]["teacherForcedEvaluation"]
        call = evaluation["call"]
        no_call = evaluation["noCall"]
    except (KeyError, TypeError):
        failures.append(f"{label} has no complete teacher-forced evaluation")
        return {
            "callCorrect": 0,
            "noCallCorrect": 0,
            "callAccuracy": math.nan,
            "noCallAccuracy": math.nan,
            "balancedAccuracy": math.nan,
            "languageModelLoss": math.nan,
        }
    for name, values, expected_total in (("call", call, 390), ("no-call", no_call, 81)):
        if values.get("total") != expected_total:
            failures.append(f"{label} {name} total must be {expected_total}")
        correct = values.get("correct")
        if (
            isinstance(correct, bool)
            or not isinstance(correct, int)
            or correct < 0
            or correct > expected_total
        ):
            failures.append(f"{label} {name} correct count is invalid")
    call_correct = call.get("correct", 0)
    no_call_correct = no_call.get("correct", 0)
    call_accuracy = call_correct / 390 if isinstance(call_correct, int) else math.nan
    no_call_accuracy = no_call_correct / 81 if isinstance(no_call_correct, int) else math.nan
    balanced = (call_accuracy + no_call_accuracy) / 2
    declared_balanced = _finite_number(
        evaluation.get("balancedAccuracy"), f"{label} balanced accuracy", failures
    )
    if math.isfinite(declared_balanced) and not math.isclose(
        declared_balanced, balanced, rel_tol=0.0, abs_tol=1e-12
    ):
        failures.append(f"{label} balanced accuracy does not match its counts")
    language_model_loss = _finite_number(
        evaluation.get("languageModelLoss"), f"{label} language-model loss", failures
    )
    return {
        "callCorrect": call_correct,
        "noCallCorrect": no_call_correct,
        "callAccuracy": call_accuracy,
        "noCallAccuracy": no_call_accuracy,
        "balancedAccuracy": balanced,
        "languageModelLoss": language_model_loss,
    }


def compare_arms(
    control: dict[str, Any], treatment: dict[str, Any], *, expected_steps: int
) -> dict[str, Any]:
    failures: list[str] = []
    if expected_steps <= 0:
        raise ValueError("expected optimizer steps must be positive")
    if control.get("experiment") != "qwen3-17b-applicability-decision-v13" or treatment.get(
        "experiment"
    ) != "qwen3-17b-applicability-decision-v13":
        failures.append("both manifests must be V13 experiments")
    if control.get("arm") != "control" or control.get("decision", {}).get("coefficient") != 0.0:
        failures.append("the control arm must have coefficient 0.0")
    if treatment.get("arm") != "treatment" or treatment.get("decision", {}).get(
        "coefficient"
    ) != 0.1:
        failures.append("the treatment arm must have coefficient 0.1")

    for identity in ("base", "continuedFrom", "data", "environment"):
        if control.get(identity) != treatment.get(identity):
            failures.append(f"control and treatment {identity} differ")
    control_decision = {key: value for key, value in control.get("decision", {}).items() if key != "coefficient"}
    treatment_decision = {
        key: value for key, value in treatment.get("decision", {}).items() if key != "coefficient"
    }
    if control_decision != treatment_decision:
        failures.append("control and treatment decision-token contracts differ")
    control_training = control.get("training", {})
    treatment_training = treatment.get("training", {})
    if {key: control_training.get(key) for key in TRAINING_CONTRACT_KEYS} != {
        key: treatment_training.get(key) for key in TRAINING_CONTRACT_KEYS
    }:
        failures.append("control and treatment training contracts differ")
    if control_training.get("optimizerStepsCompleted") != expected_steps or treatment_training.get(
        "optimizerStepsCompleted"
    ) != expected_steps:
        failures.append(f"both arms must complete exactly {expected_steps} optimizer steps")
    if control_training.get("initialLogitsSha256") != treatment_training.get(
        "initialLogitsSha256"
    ):
        failures.append("control and treatment initial logits differ")
    for label, training in (("control", control_training), ("treatment", treatment_training)):
        base_logits = training.get("exactBaseLogitsSha256")
        if not isinstance(base_logits, str) or len(base_logits) != 64:
            failures.append(f"{label} exact-base logits hash is invalid")
        if training.get("initialDisabledAdapterLogitsSha256") != base_logits:
            failures.append(f"{label} initial adapter-disabled logits differ from the base")
        if training.get("finalDisabledAdapterLogitsSha256") != base_logits:
            failures.append(f"{label} final adapter-disabled logits differ from the base")
        if training.get("reloadedLogitsSha256") != training.get("finalLogitsSha256"):
            failures.append(f"{label} saved/reloaded adapter logits differ")
        checks = training.get("finiteGradientChecks")
        if isinstance(checks, bool) or not isinstance(checks, int) or checks <= 0:
            failures.append(f"{label} has no finite-gradient checks")

    control_eval = _evaluation(control, "control", failures)
    treatment_eval = _evaluation(treatment, "treatment", failures)
    no_call_gain = treatment_eval["noCallCorrect"] - control_eval["noCallCorrect"]
    call_loss = control_eval["callCorrect"] - treatment_eval["callCorrect"]
    if no_call_gain < 9:
        failures.append("treatment must make at least nine additional no-call decisions")
    if call_loss > 7:
        failures.append("treatment may lose no more than seven correct call decisions")
    if not treatment_eval["balancedAccuracy"] > control_eval["balancedAccuracy"]:
        failures.append("treatment balanced applicability accuracy must exceed control")
    if not treatment_eval["languageModelLoss"] <= control_eval["languageModelLoss"] * 1.10:
        failures.append("treatment language-model loss may not exceed control by more than 10%")
    return {
        "schemaVersion": 1,
        "experiment": "qwen3-17b-applicability-decision-v13",
        "expectedOptimizerSteps": expected_steps,
        "passed": not failures,
        "failures": failures,
        "noCallCorrectGain": no_call_gain,
        "callCorrectLoss": call_loss,
        "control": control_eval,
        "treatment": treatment_eval,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", required=True, type=Path)
    parser.add_argument("--treatment", required=True, type=Path)
    parser.add_argument("--expected-steps", required=True, type=int)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = compare_arms(
        json.loads(args.control.read_text()),
        json.loads(args.treatment.read_text()),
        expected_steps=args.expected_steps,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.out is not None:
        if args.out.exists():
            parser.error(f"output already exists: {args.out}")
        args.out.write_text(rendered)
    print(rendered, end="")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
