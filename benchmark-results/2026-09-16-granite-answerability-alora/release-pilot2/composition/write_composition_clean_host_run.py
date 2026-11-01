#!/usr/bin/env python3
"""Writes composition-clean-host-run.json and the published-artifacts record.

The composition record is the component clean-host record (release-pilot2/clean-host/
write_clean_host_run.py) plus the facts only a composition run can establish:

  pass, freshMachine, externalInference, javaMajor, exitCode, modelsVersion, startedAt,
  completedAt, command, resolvedClasspathSha256, outputLog {uri, sha256, sizeBytes}
  + modeljarsVersion, backend, backendSelection, casesPerSuite, armOrder, suites, casesPerArm,
    controlMedianMillis, compositeMedianMillis, latencyImprovement,
    controlMedianUniqueInferenceStateBytes, compositeMedianUniqueInferenceStateBytes,
    publicApi, publishedArtifacts

publishedArtifacts is also written on its own, as the bare JSON list that
assemble_composition_report.py --published-artifacts consumes.

Nothing here trusts the caller's opinion of success; every boolean is derived from a recorded file:

* freshMachine   - the fresh-check file lists every cache as "absent" (none as "present").
* javaMajor      - parsed from the recorded `java -version` output.
* resolvedClasspathSha256 - sha256 of the resolved-classpath file bytes (see
                   run-composition-clean-host.sh for how that file is produced).
* publishedArtifacts - copied from the program's own report, then re-checked: every member must
                   carry a group:artifact:version coordinate, a 64-hex artifact sha256, a 64-hex
                   marker-jar sha256 with the URI it was hashed from, and both resolvedFromCentral
                   and runViaPublicApi exactly true. A member the runtime did not resolve from a
                   Maven-layout marker JAR fails the record, it does not silently become a passing
                   artifact.
* marker cross-check - each member's marker-jar sha256 must also appear in the wrapper's own
                   resolved-classpath fingerprint. The program hashes the jar it actually loaded the
                   marker from; the wrapper hashes every jar the resolver put on the classpath. A
                   member that only one of the two saw is not evidence of anything.
* publicApi      - the program's derivation that the run went through
                   org.modeljars.ModelJars.openActivatedToolRuntime, re-checked here: the entry
                   point must be that method, the selected backend must be the one the base is
                   qualified on, every underlying derivation fact must be true, runViaPublicApi must
                   equal their conjunction, and every member must agree with it.
* medians        - copied from the program's report, then re-derived: latencyImprovement must equal
                   (control - composite) / control to within 1e-12, and the composite arm must be
                   the faster one.
* pass           - exitCode == 0, freshMachine, javaMajor == 25, the program report's own pass, the
                   measurement count the configuration implies, the artifact, marker, public-API and
                   median checks above, and the output log's last non-empty line being exactly the
                   program's PASS verdict for every measurement.

outputLog.uri is written with the literal revision placeholder <EVIDENCE_REVISION>; the
coordinator replaces it with the integrallis/models commit that contains the log.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

ARMS = ("control", "composite")
SUITES = ("squad-v2-dev", "msmarco-v2.1-validation")
REVISION_PLACEHOLDER = "<EVIDENCE_REVISION>"
UTC_TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")
JAVA_VERSION = re.compile(r'^\S+ version "(\d+)(?:[.+\-"][^"]*)?"', re.MULTILINE)
SHA256 = re.compile(r"^[a-f0-9]{64}$")
ARM_ORDERS = ("composite-first", "control-first")
TOLERANCE = 1e-12

#: The only backend the Granite 4.1 3B base is qualified on, and the one the component evidence
#: binds. A record produced on any other backend is a different measurement, not a weaker one.
REQUIRED_BACKEND = "rust-ffm"

#: The core ModelJars entry point this run must have gone through. The supported facade,
#: org.modeljars.composite:granite-answerability, is not published yet - its publication is gated on
#: the very catalog entry this run qualifies - so the run calls what that facade calls.
PUBLIC_API_ENTRY_POINT = (
    "org.modeljars.ModelJars.openActivatedToolRuntime(ModelJar,ModelJar,ModelLoadOptions)"
)

#: The observations the program derives runViaPublicApi from. All must hold; see the program's
#: CompositionCleanHost#publicApi for what each one observes and what it cannot observe.
PUBLIC_API_FACTS = (
    "publicApiClassFromCentral",
    "runtimeTypeOwnedByModelJars",
    "membersResolvedByRuntime",
    "backendSelectedByCatalog",
)


def pass_line(measurements: int) -> str:
    return f"PASS measurements={measurements} passed={measurements}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def java_major(version_output: str) -> int | None:
    match = JAVA_VERSION.search(version_output)
    return int(match.group(1)) if match else None


def fresh_machine(fresh_check: str) -> bool:
    """Lines are "absent <path>" or "present <path>"; fresh only if at least one line and all absent."""
    lines = [line.split(None, 1) for line in fresh_check.splitlines() if line.strip()]
    return bool(lines) and all(len(parts) == 2 and parts[0] == "absent" for parts in lines)


def log_passed(log_text: str, measurements: int) -> bool:
    lines = [line.rstrip("\r") for line in log_text.splitlines() if line.strip()]
    return bool(lines) and lines[-1] == pass_line(measurements)


def utc(value: str) -> str:
    if UTC_TIMESTAMP.fullmatch(value) is None:
        raise ValueError(f"not a UTC second timestamp: {value}")
    datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    return value


def published_artifacts(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Re-checks the program's artifact record the way the composition gate will read it."""
    artifacts = report.get("publishedArtifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("program report carries no publishedArtifacts list")
    seen: set[str] = set()
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            raise ValueError("every published artifact must be an object")
        for key in ("modelId", "coordinate", "sha256", "markerJarSha256", "markerJarUri"):
            if not isinstance(artifact.get(key), str) or not artifact[key]:
                raise ValueError(f"published artifact is missing {key}")
        if artifact["coordinate"].count(":") != 2:
            raise ValueError(
                f"{artifact['modelId']} coordinate must be group:artifact:version"
            )
        for key in ("sha256", "markerJarSha256"):
            if SHA256.fullmatch(artifact[key]) is None:
                raise ValueError(f"{artifact['modelId']} {key} must be a lowercase SHA-256")
        if artifact["modelId"] in seen:
            raise ValueError(f"duplicate published artifact {artifact['modelId']}")
        seen.add(artifact["modelId"])
    return artifacts


def artifacts_resolved(artifacts: list[dict[str, Any]]) -> bool:
    return all(
        artifact.get("resolvedFromCentral") is True and artifact.get("runViaPublicApi") is True
        for artifact in artifacts
    )


def classpath_digests(classpath_file: Path) -> set[str]:
    """The sha256 of every jar the wrapper recorded on the resolved classpath."""
    digests: set[str] = set()
    for line in classpath_file.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if parts and SHA256.fullmatch(parts[0]) is not None:
            digests.add(parts[0])
    return digests


def markers_on_classpath(artifacts: list[dict[str, Any]], classpath_file: Path) -> bool:
    """Whether every member marker the program loaded is a jar the resolver actually put there."""
    digests = classpath_digests(classpath_file)
    return all(artifact["markerJarSha256"] in digests for artifact in artifacts)


def public_api(report: dict[str, Any]) -> dict[str, Any]:
    """Re-checks the program's derivation that the hybrid was opened through the public API."""
    record = report.get("publicApi")
    if not isinstance(record, dict):
        raise ValueError("program report carries no publicApi derivation")
    if record.get("entryPoint") != PUBLIC_API_ENTRY_POINT:
        raise ValueError(
            f"public API entry point must be {PUBLIC_API_ENTRY_POINT}, "
            f"not {record.get('entryPoint')!r}"
        )
    if record.get("selectedBackend") != REQUIRED_BACKEND:
        raise ValueError(
            f"the composition runs on {REQUIRED_BACKEND}, not {record.get('selectedBackend')!r}"
        )
    if SHA256.fullmatch(str(record.get("jarSha256", ""))) is None:
        raise ValueError("publicApi.jarSha256 must be a lowercase SHA-256")
    for fact in PUBLIC_API_FACTS:
        if fact not in record:
            raise ValueError(f"publicApi derivation is missing {fact}")
    return record


def public_api_consistent(record: dict[str, Any], artifacts: list[dict[str, Any]]) -> bool:
    derived = all(record.get(fact) is True for fact in PUBLIC_API_FACTS)
    if record.get("runViaPublicApi") is not derived:
        # A flag that does not follow from the facts it claims to summarise is not a derivation.
        return False
    return derived and all(
        artifact.get("runViaPublicApi") is derived for artifact in artifacts
    )


def medians_consistent(report: dict[str, Any]) -> bool:
    control = report.get("controlMedianMillis")
    composite = report.get("compositeMedianMillis")
    improvement = report.get("latencyImprovement")
    for value in (control, composite, improvement):
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError("medians and improvement must be numbers")
    if control <= 0 or composite <= 0:
        raise ValueError("medians must be positive")
    if composite >= control:
        return False
    return abs(improvement - (control - composite) / control) <= TOLERANCE


def expected_measurements(report: dict[str, Any]) -> int:
    cases_per_suite = report.get("casesPerSuite")
    if not isinstance(cases_per_suite, int) or isinstance(cases_per_suite, bool):
        raise ValueError("casesPerSuite must be an integer")
    if cases_per_suite <= 0:
        raise ValueError("casesPerSuite must be > 0")
    return cases_per_suite * len(SUITES) * len(ARMS)


def measurements_complete(report: dict[str, Any], expected: int) -> bool:
    measurements = report.get("measurements")
    if not isinstance(measurements, list):
        raise ValueError("program report carries no measurements list")
    if len(measurements) != expected:
        return False
    per_arm = {arm: 0 for arm in ARMS}
    for measurement in measurements:
        arm = measurement.get("arm")
        if arm not in per_arm:
            raise ValueError(f"unknown arm {arm!r} in measurements")
        per_arm[arm] += 1
        unique_bytes = measurement.get("uniqueInferenceStateBytes")
        if not isinstance(unique_bytes, int) or isinstance(unique_bytes, bool):
            raise ValueError("every measurement must record uniqueInferenceStateBytes")
        if unique_bytes < 0:
            # -1 is the program's record of OptionalLong.empty(): the backend could not measure the
            # inference state. That is a missing observation, so the record cannot claim one.
            return False
        shared_tokens = measurement.get("sharedPrefixTokens")
        if not isinstance(shared_tokens, int) or isinstance(shared_tokens, bool):
            raise ValueError("every measurement must record sharedPrefixTokens")
        if measurement.get("pass") is not True:
            return False
        if arm == "composite" and (
            measurement.get("physicallySharesPrefix") is not True or shared_tokens <= 0
        ):
            return False
        # sharedPrefixTokens counts PHYSICALLY SHARED tokens, so a recomputed turn reports 0 by
        # construction. A control measurement claiming shared tokens would mean the arm switch did
        # not take, which is exactly the silent-no-op failure this record has to catch.
        if arm == "control" and (
            measurement.get("physicallySharesPrefix") is not False or shared_tokens != 0
        ):
            return False
    return all(count == expected // len(ARMS) for count in per_arm.values())


def build(
    *,
    started_at: str,
    completed_at: str,
    exit_code: int,
    models_version: str,
    modeljars_version: str,
    command: list[str],
    java_version_output: str,
    fresh_check: str,
    classpath_file: Path,
    output_log: Path,
    output_log_repo_path: str,
    program_report: dict[str, Any],
) -> dict[str, Any]:
    if not command or any(not isinstance(part, str) or not part for part in command):
        raise ValueError("command must be a nonempty list of nonempty strings")
    if output_log_repo_path.startswith("/"):
        raise ValueError("output log repository path must be relative")
    arm_order = program_report.get("armOrder")
    if arm_order not in ARM_ORDERS:
        raise ValueError(f"unknown armOrder {arm_order!r}")
    if program_report.get("modelsVersion") != models_version:
        raise ValueError("program report modelsVersion does not match the wrapper")
    if program_report.get("modeljarsVersion") != modeljars_version:
        raise ValueError("program report modeljarsVersion does not match the wrapper")
    if program_report.get("backend") != REQUIRED_BACKEND:
        raise ValueError(
            f"the composition runs on {REQUIRED_BACKEND}, "
            f"not {program_report.get('backend')!r}"
        )

    artifacts = published_artifacts(program_report)
    api = public_api(program_report)
    expected = expected_measurements(program_report)
    major = java_major(java_version_output)
    fresh = fresh_machine(fresh_check)
    log_bytes = output_log.read_bytes()
    passed = (
        exit_code == 0
        and fresh
        and major == 25
        and program_report.get("pass") is True
        and artifacts_resolved(artifacts)
        and markers_on_classpath(artifacts, classpath_file)
        and public_api_consistent(api, artifacts)
        and medians_consistent(program_report)
        and measurements_complete(program_report, expected)
        and log_passed(log_bytes.decode("utf-8", errors="replace"), expected)
        and utc(completed_at) > utc(started_at)
    )
    return {
        "pass": passed,
        "freshMachine": fresh,
        "externalInference": False,
        "javaMajor": major,
        "exitCode": exit_code,
        "modelsVersion": models_version,
        "modeljarsVersion": modeljars_version,
        "backend": program_report["backend"],
        "backendSelection": program_report.get("backendSelection"),
        "startedAt": utc(started_at),
        "completedAt": utc(completed_at),
        "command": command,
        "resolvedClasspathSha256": sha256_file(classpath_file),
        "suites": list(SUITES),
        "casesPerSuite": program_report["casesPerSuite"],
        "casesPerArm": expected // len(ARMS),
        "armOrder": arm_order,
        "windowSha256": program_report.get("windowSha256"),
        "controlMedianMillis": program_report["controlMedianMillis"],
        "compositeMedianMillis": program_report["compositeMedianMillis"],
        "latencyImprovement": program_report["latencyImprovement"],
        "controlMedianUniqueInferenceStateBytes": program_report.get(
            "controlMedianUniqueInferenceStateBytes"
        ),
        "compositeMedianUniqueInferenceStateBytes": program_report.get(
            "compositeMedianUniqueInferenceStateBytes"
        ),
        "publicApi": api,
        "publishedArtifacts": artifacts,
        "outputLog": {
            "uri": f"https://raw.githubusercontent.com/integrallis/models/{REVISION_PLACEHOLDER}/{output_log_repo_path}",
            "sha256": hashlib.sha256(log_bytes).hexdigest(),
            "sizeBytes": len(log_bytes),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--started-at", required=True)
    parser.add_argument("--completed-at", required=True)
    parser.add_argument("--exit-code", required=True, type=int)
    parser.add_argument("--models-version", required=True)
    parser.add_argument("--modeljars-version", required=True)
    parser.add_argument("--command-json", required=True, help="JSON array of the exact argv run")
    parser.add_argument("--java-version-file", required=True, type=Path)
    parser.add_argument("--fresh-check-file", required=True, type=Path)
    parser.add_argument("--classpath-file", required=True, type=Path)
    parser.add_argument("--output-log", required=True, type=Path)
    parser.add_argument("--output-log-repo-path", required=True)
    parser.add_argument(
        "--program-report", required=True, type=Path, help="CompositionCleanHost --report output"
    )
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--published-artifacts-out",
        type=Path,
        help="also write the bare publishedArtifacts list for assemble_composition_report.py",
    )
    args = parser.parse_args(argv)
    run = build(
        started_at=args.started_at,
        completed_at=args.completed_at,
        exit_code=args.exit_code,
        models_version=args.models_version,
        modeljars_version=args.modeljars_version,
        command=json.loads(args.command_json),
        java_version_output=args.java_version_file.read_text(encoding="utf-8"),
        fresh_check=args.fresh_check_file.read_text(encoding="utf-8"),
        classpath_file=args.classpath_file,
        output_log=args.output_log,
        output_log_repo_path=args.output_log_repo_path,
        program_report=json.loads(args.program_report.read_text(encoding="utf-8")),
    )
    args.out.write_text(json.dumps(run, indent=2) + "\n", encoding="utf-8")
    if args.published_artifacts_out is not None:
        args.published_artifacts_out.write_text(
            json.dumps(run["publishedArtifacts"], indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps(run, indent=2))
    return 0 if run["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
