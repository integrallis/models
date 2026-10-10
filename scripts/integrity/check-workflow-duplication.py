#!/usr/bin/env python3
"""No workflow may inline a setup step that a shared action already owns.

A step copied into N workflows gets fixed in N-1 of them. The PTX toolchain install was copied into
five: ci, codeql, model-determinism and release had it, docs did not. So Deploy Docs failed on main
for every run from 2026-10-09 with rust-src absent and `:backend-cuda:compilePtx FAILED`, while the
other four stayed green and nothing connected the two facts. It is now one composite action at
.github/actions/ptx-toolchain, and this check keeps it that way.

    python3 scripts/integrity/check-workflow-duplication.py

Exits 1 if a workflow installs the pinned nightly itself instead of using the action.
"""
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
WORKFLOWS = ROOT / ".github" / "workflows"
# Workflows reference it as ./models/.github/actions/... because `uses: ./` resolves from the
# workspace root and every workflow here checks the repo out into models/. The check matches the
# repo-relative tail so it is independent of that prefix.
ACTION = ".github/actions/ptx-toolchain"

# Markers that mean a workflow is doing the action's job by hand.
INLINE = [
    (re.compile(r"rustup\s+component\s+add[^\n]*rust-src"), "installs rust-src itself"),
    (re.compile(r"rustup\s+target\s+add[^\n]*nvptx64-nvidia-cuda"),
     "adds the nvptx64 target itself"),
    (re.compile(r"rustup\s+toolchain\s+install\s+nightly-"), "installs the pinned nightly itself"),
]


def main():
    if not (ROOT / ACTION / "action.yml").exists():
        print(f"FAIL: the shared action is missing at {ACTION}/action.yml")
        return 1

    problems, users = [], []
    for wf in sorted(WORKFLOWS.glob("*.yml")):
        text = wf.read_text()
        if ACTION in text:
            users.append(wf.name)
        for pattern, why in INLINE:
            if pattern.search(text):
                problems.append(f"{wf.name}: {why}; use the shared action at {ACTION} instead")

    print(f"workflows using the shared PTX toolchain action: {len(users)}")
    for u in users:
        print(f"  {u}")
    print()
    if problems:
        print(f"FAIL: {len(problems)} workflow(s) duplicate what the shared action owns.")
        print("A step copied into N places gets fixed in N-1 of them -- that is how Deploy Docs")
        print("became the one workflow without rust-src while four others had it.")
        for p in problems:
            print(f"  {p}")
        return 1
    print("no workflow installs the pinned nightly by hand.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
