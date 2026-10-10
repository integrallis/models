#!/usr/bin/env python3
"""Prove every integrity guard can actually fail.

A guard that cannot detect the defect it was written for is worse than no guard: it reports green
and everyone stops looking. This has already happened here. On 2026-10-09 a check meant to catch a
fixed-name kernel reference used the character class `[a-z0-9_]+`, which cannot match
`models-kernels-linux-x86_64.jar` because the name contains hyphens. It passed while the defect was
present, and it was only caught by deliberately reintroducing the bug.

So every guard is run twice: once on the clean tree, where it must pass, and once with its defect
injected, where it must fail. A guard that passes under mutation is reported as disarmed.

Mutations are written out explicitly rather than generated, because each one has to be the defect
that guard exists to catch -- a generic perturbation would prove nothing. The table is the
reviewable part of this script.

    python3 scripts/integrity/mutation-check.py [--only NAME]

Exits 1 if any guard passes on a mutated tree, or fails on a clean one.
"""
import argparse
import pathlib
import shutil
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[2]


def sub(path, old, new):
    """A mutation that replaces exact text, and refuses if the anchor is not there."""
    def apply(root):
        p = root / path
        s = p.read_text()
        if s.count(old) < 1:
            raise AssertionError(f"mutation anchor absent in {path}: {old[:60]!r}")
        p.write_text(s.replace(old, new, 1))
    return apply


def append(path, text):
    def apply(root):
        p = root / path
        p.write_text(p.read_text() + text)
    return apply


# name, guard command, the defect to inject, what the guard is supposed to notice
MUTATIONS = [
    (
        "evidence-references/unresolvable-path",
        ["python3", "scripts/integrity/check-evidence-references.py"],
        sub("CHANGELOG.md",
            "benchmark-results/2026-10-09-q4-1-support/NOTES.md",
            "benchmark-results/2026-10-08-q4-1-exact-path/NOTES.md"),
        "a changelog citing an evidence directory that does not exist -- shipped on 2026-10-09",
    ),
    (
        "fleet-guards/fixed-name-kernel-in-classpath",
        ["bash", "scripts/fleet/qual-worker-guards-test.sh"],
        sub("scripts/fleet/qual-worker-two-arm.sh",
            'CP="$DIST/lib/*:/work/$KERNELS"',
            'CP="$DIST/lib/*:/work/models-kernels-linux-x86_64.jar"'),
        "a worker naming the kernels JAR literally, which ran a September kernel for a whole "
        "campaign with no symptom",
    ),
    (
        "fleet-guards/fixed-name-kernel-in-fetch",
        ["bash", "scripts/fleet/qual-worker-guards-test.sh"],
        sub("scripts/fleet/parity-worker.sh",
            'aws s3 cp "s3://$BUCKET/payload/$KERNELS" . --only-show-errors',
            'aws s3 cp "s3://$BUCKET/payload/models-kernels-linux-x86_64.jar" . --only-show-errors'),
        "the same defect in the second worker, which is the copy I missed the first time",
    ),
    (
        "workflow-duplication/step-copied-back-inline",
        ["python3", "scripts/integrity/check-workflow-duplication.py"],
        sub(".github/workflows/docs.yml",
            "      - name: Install the PTX toolchain\n        uses: ./.github/actions/ptx-toolchain",
            "      - name: Install pinned Rust nightly for the PTX kernels\n"
            "        run: |\n"
            "          rustup toolchain install nightly-2026-09-17 --profile minimal\n"
            "          rustup component add --toolchain nightly-2026-09-17 rust-src"),
        "a workflow installing the pinned nightly itself again, which is how docs ended up as "
        "the one of five without rust-src",
    ),
    (
        "sweep-derive/number-drifts-from-data",
        ["python3", "benchmark-results/2026-10-09-max-tokens-sweep/derive.py"],
        sub("benchmark-results/2026-10-09-max-tokens-sweep/NOTES.md",
            "0.778 → 0.000", "0.778 → 0.100"),
        "a hand-written rate in the write-up disagreeing with the derived artifacts",
    ),
    (
        "sweep-derive/backend-delta-record-tampered",
        ["python3", "benchmark-results/2026-10-09-max-tokens-sweep/derive.py"],
        append("benchmark-results/2026-10-09-max-tokens-sweep/backend-delta/"
               "48caca0dafed..7ac41536f5c2.txt",
               "backend-java/src/main/java/Injected.java\n"),
        "the recorded backend delta no longer matching git, which would admit a pair whose two "
        "arms differ on the measured path",
    ),
]


def run(cmd, cwd):
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    return p.returncode, (p.stdout + p.stderr)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", help="run just the mutations whose name contains this")
    args = ap.parse_args(argv)

    selected = [m for m in MUTATIONS if not args.only or args.only in m[0]]
    if not selected:
        print(f"no mutation matches {args.only!r}")
        return 1

    failures = []
    # Clean-tree pass first. A guard that is already red makes its mutation result meaningless.
    print("clean tree -- every guard must pass:")
    for name, cmd, _, _ in selected:
        rc, out = run(cmd, ROOT)
        label = " ".join(cmd[-1:])
        if rc == 0:
            print(f"  ok     {name}")
        else:
            print(f"  RED    {name} -- guard already fails on a clean tree, so its mutation "
                  f"result would mean nothing")
            print("".join(f"           {l}\n" for l in out.strip().splitlines()[-4:]))
            failures.append((name, "red on clean tree"))

    print()
    print("mutated tree -- every guard must FAIL:")
    for name, cmd, mutate, defect in selected:
        if any(f[0] == name for f in failures):
            continue
        with tempfile.TemporaryDirectory() as tmp:
            work = pathlib.Path(tmp) / "tree"
            # A copy, so a mutation can never be left behind in the real tree even on a crash.
            shutil.copytree(ROOT, work, symlinks=True,
                            ignore=shutil.ignore_patterns("build", ".gradle", "node_modules",
                                                          ".git", "*.gguf", "*.tar"))
            try:
                mutate(work)
            except AssertionError as e:
                print(f"  BROKEN {name} -- {e}")
                failures.append((name, "mutation anchor missing"))
                continue
            rc, _ = run(cmd, work)
            if rc != 0:
                print(f"  ok     {name}  (guard caught: {defect})")
            else:
                print(f"  DISARMED {name}")
                print(f"           the guard PASSED with this defect injected: {defect}")
                failures.append((name, "passed under mutation"))

    print()
    if failures:
        print(f"FAIL: {len(failures)} guard(s) are not doing their job.")
        for name, why in failures:
            print(f"  {name}: {why}")
        return 1
    print(f"every one of {len(selected)} guards passes clean and fails on its own defect.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
