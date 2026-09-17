#!/usr/bin/env python3
"""Writes clean-host-run.json in the shape ModelJars' requireCleanHostRun gate reads.

Fields (tools/component-evidence-gate.mjs, mirrored by validate_clean_host_run in
benchmark-results/2026-09-15-granite-4.1-3b-alora-hybrid/assemble_component_report.py):

  pass, freshMachine, externalInference, javaMajor, exitCode, modelsVersion, startedAt,
  completedAt, command, resolvedClasspathSha256, outputLog {uri, sha256, sizeBytes}

plus two fields this run adds beyond the gate's minimum, so a record says which kernel runtime
produced it (the gate ignores unknown keys; it never inferred a backend on its own):

  backend        - "pure-java" or "rust-ffm", cross-checked against the program log
  nativeLibrary  - null on pure-java; on rust-ffm the identity of the loaded native kernel
                   {platform, abi, library, sha256, source, kernelPlan}, read out of the log

Nothing here trusts the caller's opinion of success; every boolean is derived from a recorded file:

* freshMachine   - the fresh-check file lists every cache as "absent" (none as "present").
* javaMajor      - parsed from the recorded `java -version` output.
* resolvedClasspathSha256 - sha256 of the resolved-classpath file bytes (see run-clean-host.sh
                   for how that file is produced).
* backend        - the caller declares it, but the record only passes when the log's own
                   "loaded backend=" line, all six "CASE ... backend=" lines and the presence or
                   absence of a "native-library" line agree with the declaration. A pure-Java log
                   relabelled as a Rust run therefore fails instead of being believed.
* nativeLibrary  - parsed from the program's "native-library" line, never supplied by the caller.
* pass           - exitCode == 0, freshMachine, javaMajor == 25, the backend agreement above, the
                   kernel observably injected on the Rust arm (and observably absent on the
                   pure-Java arm), and the output log's last non-empty line being exactly the
                   program's PASS verdict for all expected cases.

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

EXPECTED_CASES = 6
PASS_LINE = f"PASS cases={EXPECTED_CASES} passed={EXPECTED_CASES}"
REVISION_PLACEHOLDER = "<EVIDENCE_REVISION>"
BACKENDS = ("pure-java", "rust-ffm")
NATIVE_BACKEND = "rust-ffm"
UTC_TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")
JAVA_VERSION = re.compile(r'^\S+ version "(\d+)(?:[.+\-"][^"]*)?"', re.MULTILINE)

# Emitted once by AnswerabilityCleanHost after the backend opens; see its `run` method.
LOADED_LINE = re.compile(
    r"^loaded backend=(?P<backend>\S+)\b.*?"
    r"\binjectedGroupedProjections=(?P<injected>true|false)\b.*?"
    r"\bmatrixKernel=(?P<kernel>\S+)",
    re.MULTILINE,
)
# Emitted only on the Rust arm, before the backend opens.
NATIVE_LINE = re.compile(
    r"^native-library backend=(?P<backend>\S+) platform=(?P<platform>\S+) abi=(?P<abi>\d+)"
    r" file=(?P<library>\S+) sha256=(?P<sha256>[0-9a-f]{64}) source=(?P<source>\S+)"
    r" kernelPlan=(?P<plan>\S+)",
    re.MULTILINE,
)
CASE_LINE = re.compile(r"^CASE (?:PASS|FAIL) backend=(\S+)\b", re.MULTILINE)
# The injected kernel identifies itself as rust-ffm-quantized-vNN; anything else means the
# native toggle was selected but the transformer still ran the Vector API path.
RUST_KERNEL = re.compile(r"^rust-ffm-")


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


def logged_backend(log_text: str) -> str | None:
    """Returns the backend the program itself reported loading, or None if it reported none."""
    match = LOADED_LINE.search(log_text)
    return match.group("backend") if match else None


def logged_kernel(log_text: str) -> tuple[bool, str] | None:
    """Returns (injectedGroupedProjections, matrixKernel) as the program observed them."""
    match = LOADED_LINE.search(log_text)
    return (match.group("injected") == "true", match.group("kernel")) if match else None


def native_library(log_text: str) -> dict[str, Any] | None:
    """Returns the native kernel identity the program printed, or None when it printed none."""
    match = NATIVE_LINE.search(log_text)
    if match is None:
        return None
    return {
        "platform": match.group("platform"),
        "abi": int(match.group("abi")),
        "library": match.group("library"),
        "sha256": match.group("sha256"),
        "source": match.group("source"),
        "kernelPlan": match.group("plan"),
        "_backend": match.group("backend"),
    }


def case_backends(log_text: str) -> list[str]:
    return CASE_LINE.findall(log_text)


def log_passed(log_text: str) -> bool:
    lines = [line.rstrip("\r") for line in log_text.splitlines() if line.strip()]
    return bool(lines) and lines[-1] == PASS_LINE


def backend_agrees(log_text: str, backend: str, native: dict[str, Any] | None) -> bool:
    """Every backend statement the program made must name the declared backend, and only it."""
    if logged_backend(log_text) != backend:
        return False
    if case_backends(log_text) != [backend] * EXPECTED_CASES:
        return False
    if (native is not None) != (backend == NATIVE_BACKEND):
        return False
    if native is not None and native["_backend"] != NATIVE_BACKEND:
        return False
    kernel = logged_kernel(log_text)
    if kernel is None:
        return False
    injected, implementation = kernel
    # An ablation switch that silently does nothing is worse than no switch: require the
    # kernel choice to be visible in the loaded execution plan, both ways.
    if backend == NATIVE_BACKEND:
        return injected and RUST_KERNEL.match(implementation) is not None
    return not injected and RUST_KERNEL.match(implementation) is None


def utc(value: str) -> str:
    if UTC_TIMESTAMP.fullmatch(value) is None:
        raise ValueError(f"not a UTC second timestamp: {value}")
    datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    return value


def build(
    *,
    started_at: str,
    completed_at: str,
    exit_code: int,
    models_version: str,
    backend: str,
    command: list[str],
    java_version_output: str,
    fresh_check: str,
    classpath_file: Path,
    output_log: Path,
    output_log_repo_path: str,
) -> dict[str, Any]:
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}: {backend!r}")
    if not command or any(not isinstance(part, str) or not part for part in command):
        raise ValueError("command must be a nonempty list of nonempty strings")
    if output_log_repo_path.startswith("/"):
        raise ValueError("output log repository path must be relative")
    major = java_major(java_version_output)
    fresh = fresh_machine(fresh_check)
    log_bytes = output_log.read_bytes()
    log_text = log_bytes.decode("utf-8", errors="replace")
    native = native_library(log_text)
    passed = (
        exit_code == 0
        and fresh
        and major == 25
        and backend_agrees(log_text, backend, native)
        and log_passed(log_text)
        and utc(completed_at) > utc(started_at)
    )
    return {
        "pass": passed,
        "freshMachine": fresh,
        "externalInference": False,
        "javaMajor": major,
        "exitCode": exit_code,
        "modelsVersion": models_version,
        "backend": backend,
        "nativeLibrary": None if native is None else {k: v for k, v in native.items() if k != "_backend"},
        "startedAt": utc(started_at),
        "completedAt": utc(completed_at),
        "command": command,
        "resolvedClasspathSha256": sha256_file(classpath_file),
        "outputLog": {
            "uri": f"https://raw.githubusercontent.com/integrallis/models/{REVISION_PLACEHOLDER}/{output_log_repo_path}",
            "sha256": hashlib.sha256(log_bytes).hexdigest(),
            "sizeBytes": len(log_bytes),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--started-at", required=True)
    parser.add_argument("--completed-at", required=True)
    parser.add_argument("--exit-code", required=True, type=int)
    parser.add_argument("--models-version", required=True)
    parser.add_argument("--backend", required=True, choices=list(BACKENDS))
    parser.add_argument("--command-json", required=True, help="JSON array of the exact argv run")
    parser.add_argument("--java-version-file", required=True, type=Path)
    parser.add_argument("--fresh-check-file", required=True, type=Path)
    parser.add_argument("--classpath-file", required=True, type=Path)
    parser.add_argument("--output-log", required=True, type=Path)
    parser.add_argument("--output-log-repo-path", required=True)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    run = build(
        started_at=args.started_at,
        completed_at=args.completed_at,
        exit_code=args.exit_code,
        models_version=args.models_version,
        backend=args.backend,
        command=json.loads(args.command_json),
        java_version_output=args.java_version_file.read_text(encoding="utf-8"),
        fresh_check=args.fresh_check_file.read_text(encoding="utf-8"),
        classpath_file=args.classpath_file,
        output_log=args.output_log,
        output_log_repo_path=args.output_log_repo_path,
    )
    args.out.write_text(json.dumps(run, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(run, indent=2))
    return 0 if run["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
