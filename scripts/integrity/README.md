# Integrity checks

Each check here exists because a specific defect shipped. The script that catches it names that
defect, so nobody has to guess why the check is there or whether it still matters.

| check | the defect it exists for |
| --- | --- |
| `check-evidence-references.py` | A 0.3.56 CHANGELOG entry cited an evidence directory dated `2026-10-08-q4-1-exact-path`, which never existed — the real one is dated `2026-10-09-q4-1-support`. A number is only evidence if the artifact it points at is real. |
| `check-workflow-duplication.py` | The PTX toolchain install was copied into five workflows. Four had it, `docs` did not, so Deploy Docs failed on main for every run from 2026-10-09 with `rust-src` absent while the other four stayed green. |
| `../fleet/candidate-inventory.py` | 356 of 457 catalog candidates were undocumented as a set, so "tried and failed" could not be told from "never looked at". |
| `mutation-check.py` | A check meant to catch a fixed-name kernel reference used `[a-z0-9_]+`, which cannot match `models-kernels-linux-x86_64.jar` because the name has hyphens. It passed while the defect was present. |

## The mutation check is the important one

Every other guard is only worth what it detects. `mutation-check.py` runs each guard twice: once
on a clean tree where it must pass, and once with its own defect injected into a throwaway copy,
where it must fail. A guard that passes under mutation is reported as **DISARMED**.

It paid for itself on the first run, finding three further defects in guards that had looked
green:

1. `derive.py` trusted its committed backend-delta record whenever `git` could not answer — so on
   a shallow clone or a tarball, a tampered record would admit a pair whose two arms differed on
   the measured path. Its tamper detection only worked when git worked.
2. With that fixed, `derive.py` **exited 0 having rejected every pair.** A derivation that
   produced no numbers was reporting success.
3. And its NOTES check passed vacuously against zero derived rows, meaning it would have blessed
   any prose at all.

None of those were visible from a clean run. All three are the same shape: a check that reports
green because it had nothing to check.

## Writing about a bad path

`check-evidence-references.py` cannot tell a citation from an illustration, so **do not write a
literal non-existent path into a document**, not even as an example of what went wrong. Name the
directory without the `benchmark-results/` prefix, as the table above does. This was found by the
check failing on this very file, which is the check working.

Teaching it to recognise examples would mean a marker comment the author has to remember, and a
guard whose correctness depends on being remembered is the kind that quietly stops working.

## Adding a guard

Add the guard, then add a mutation for it to the table in `mutation-check.py`. The mutation must
be the defect the guard exists to catch — a generic perturbation proves nothing. If you cannot
write a mutation that makes your guard fail, the guard does not work yet.

## The baseline

`evidence-reference-baseline.json` enumerates references that were already broken before the check
existed, each with a reason. The check stays live for everything else. The list may shrink and must
never grow: a new unresolvable reference fails, and a baselined entry that *starts* resolving also
fails, so the baseline cannot quietly hide a later break.
