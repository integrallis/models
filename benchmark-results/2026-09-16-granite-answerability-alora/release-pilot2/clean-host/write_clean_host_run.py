#!/usr/bin/env python3
"""Writes clean-host-run.json in the shape ModelJars' requireCleanHostRun gate reads.

Fields (tools/component-evidence-gate.mjs, mirrored by validate_clean_host_run in
benchmark-results/2026-09-15-granite-4.1-3b-alora-hybrid/assemble_component_report.py):

  pass, freshMachine, externalInference, javaMajor, exitCode, modelsVersion, startedAt,
  completedAt, command, resolvedClasspathSha256, outputLog {uri, sha256, sizeBytes}

Nothing here trusts the caller's opinion of success; every boolean is derived from a recorded file:

* freshMachine   - the fresh-check file lists every cache as "absent" (none as "present").
* javaMajor      - parsed from the recorded `java -version` output.
* resolvedClasspathSha256 - sha256 of the resolved-classpath file bytes (see run-clean-host.sh
                   for how that file is produced).
* pass           - exitCode == 0, freshMachine, javaMajor == 25, and the output log's last
                   non-empty line is exactly the program's PASS verdict for all expected cases.

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
UTC_TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")
JAVA_VERSION = re.compile(r'^\S+ version "(\d+)(?:[.+\-"][^"]*)?"', re.MULTILINE)


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


def log_passed(log_text: str) -> bool:
    lines = [line.rstrip("\r") for line in log_text.splitlines() if line.strip()]
    return bool(lines) and lines[-1] == PASS_LINE


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
    command: list[str],
    java_version_output: str,
    fresh_check: str,
    classpath_file: Path,
    output_log: Path,
    output_log_repo_path: str,
) -> dict[str, Any]:
    if not command or any(not isinstance(part, str) or not part for part in command):
        raise ValueError("command must be a nonempty list of nonempty strings")
    if output_log_repo_path.startswith("/"):
        raise ValueError("output log repository path must be relative")
    major = java_major(java_version_output)
    fresh = fresh_machine(fresh_check)
    log_bytes = output_log.read_bytes()
    passed = (
        exit_code == 0
        and fresh
        and major == 25
        and log_passed(log_bytes.decode("utf-8", errors="replace"))
        and utc(completed_at) > utc(started_at)
    )
    return {
        "pass": passed,
        "freshMachine": fresh,
        "externalInference": False,
        "javaMajor": major,
        "exitCode": exit_code,
        "modelsVersion": models_version,
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
