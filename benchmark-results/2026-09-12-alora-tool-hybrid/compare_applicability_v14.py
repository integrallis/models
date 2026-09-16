#!/usr/bin/env python3
"""Apply the frozen V14 applicability development gates."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


V13 = "qwen3-17b-applicability-decision-v13"
V14 = "qwen3-17b-applicability-decision-v14"
TRAINING_CONTRACT_KEYS = (
    "epochs",
    "screenSteps",
    "optimizerStepsCompleted",
    "learningRate",
    "gradientAccumulation",
    "maxLength",
    "seed",
)


def finite(value: Any, label: str, failures: list[str]) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        failures.append(f"{label} must be finite")
        return math.nan
    return float(value)


def evaluation(manifest: dict[str, Any], label: str, failures: list[str]) -> dict[str, Any]:
    try:
        value = manifest["training"]["teacherForcedEvaluation"]
        call = value["call"]
        no_call = value["noCall"]
    except (KeyError, TypeError):
        failures.append(f"{label} has no complete teacher-forced evaluation")
        return {
            "callCorrect": 0,
            "noCallCorrect": 0,
            "balancedAccuracy": math.nan,
            "languageModelLoss": math.nan,
            "applicabilityLoss": math.nan,
        }
    for name, item, total in (("call", call, 390), ("no-call", no_call, 81)):
        correct = item.get("correct")
        if item.get("total") != total:
            failures.append(f"{label} {name} total must be {total}")
        if isinstance(correct, bool) or not isinstance(correct, int) or not 0 <= correct <= total:
            failures.append(f"{label} {name} correct count is invalid")
    call_correct = call.get("correct", 0)
    no_call_correct = no_call.get("correct", 0)
    balanced = (call_correct / 390 + no_call_correct / 81) / 2
    declared = finite(value.get("balancedAccuracy"), f"{label} balanced accuracy", failures)
    if math.isfinite(declared) and not math.isclose(declared, balanced, rel_tol=0, abs_tol=1e-12):
        failures.append(f"{label} balanced accuracy does not match its counts")
    return {
        "callCorrect": call_correct,
        "noCallCorrect": no_call_correct,
        "balancedAccuracy": balanced,
        "languageModelLoss": finite(
            value.get("languageModelLoss"), f"{label} language-model loss", failures
        ),
        "applicabilityLoss": finite(
            value.get("applicabilityLoss"), f"{label} applicability loss", failures
        ),
    }


def compare_arms(
    control: dict[str, Any], treatment: dict[str, Any], *, expected_steps: int
) -> dict[str, Any]:
    if expected_steps <= 0:
        raise ValueError("expected optimizer steps must be positive")
    failures: list[str] = []
    allowed_control = {V14} if expected_steps != 32 else {V13, V14}
    if control.get("experiment") not in allowed_control:
        failures.append("the control is not an allowed V13 screen or V14 endpoint baseline")
    if treatment.get("experiment") != V14:
        failures.append("the treatment must be a V14 experiment")
    if control.get("arm") != "control" or control.get("decision", {}).get("coefficient") != 0.0:
        failures.append("the control arm must have coefficient 0.0")
    if treatment.get("arm") != "treatment" or treatment.get("decision", {}).get("coefficient") != 0.1:
        failures.append("the treatment arm must have coefficient 0.1")

    for identity in ("base", "continuedFrom", "data", "environment"):
        if control.get(identity) != treatment.get(identity):
            failures.append(f"control and treatment {identity} differ")
    control_decision = {
        key: value for key, value in control.get("decision", {}).items() if key != "coefficient"
    }
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
    if any(
        training.get("optimizerStepsCompleted") != expected_steps
        for training in (control_training, treatment_training)
    ):
        failures.append(f"both arms must complete exactly {expected_steps} optimizer steps")
    if control_training.get("initialLogitsSha256") != treatment_training.get(
        "initialLogitsSha256"
    ):
        failures.append("control and treatment initial logits differ")

    expected_gradient_checks = min(expected_steps * 16, 2840)
    for label, training in (("control", control_training), ("treatment", treatment_training)):
        base_hash = training.get("exactBaseLogitsSha256")
        if not isinstance(base_hash, str) or len(base_hash) != 64:
            failures.append(f"{label} exact-base logits hash is invalid")
        if training.get("initialDisabledAdapterLogitsSha256") != base_hash:
            failures.append(f"{label} initial adapter-disabled logits differ from the base")
        if training.get("finalDisabledAdapterLogitsSha256") != base_hash:
            failures.append(f"{label} final adapter-disabled logits differ from the base")
        final_hash = training.get("finalLogitsSha256")
        if training.get("postSaveLogitsSha256") != final_hash:
            failures.append(f"{label} saving changed the live adapter logits")
        if training.get("reloadedLogitsSha256") != final_hash:
            failures.append(f"{label} saved/reloaded adapter logits differ")
        if training.get("reloadedAdapterStateSha256") != training.get(
            "trainedAdapterStateSha256"
        ):
            failures.append(f"{label} saved/reloaded adapter state differs")
        if training.get("finiteGradientChecks") != expected_gradient_checks:
            failures.append(
                f"{label} must record exactly {expected_gradient_checks} finite-gradient checks"
            )

    control_eval = evaluation(control, "control", failures)
    treatment_eval = evaluation(treatment, "treatment", failures)
    no_call_gain = treatment_eval["noCallCorrect"] - control_eval["noCallCorrect"]
    call_loss = control_eval["callCorrect"] - treatment_eval["callCorrect"]
    if no_call_gain < 0:
        failures.append("treatment may not regress correct no-call decisions")
    if call_loss > 7:
        failures.append("treatment may lose no more than seven correct call decisions")
    if not treatment_eval["balancedAccuracy"] > control_eval["balancedAccuracy"]:
        failures.append("treatment balanced applicability accuracy must exceed control")
    if not treatment_eval["applicabilityLoss"] <= control_eval["applicabilityLoss"] * 0.95:
        failures.append("treatment must reduce applicability loss by at least 5%")
    if not treatment_eval["languageModelLoss"] <= control_eval["languageModelLoss"] * 1.10:
        failures.append("treatment language-model loss may not exceed control by more than 10%")
    return {
        "schemaVersion": 1,
        "experiment": V14,
        "baselineExperiment": control.get("experiment"),
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
