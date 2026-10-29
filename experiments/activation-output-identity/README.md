# activation-output-identity

Does making the x86 activation path deterministic change what the model chooses, and what does it
cost?

## Status

Measurements complete except the baseline self-difference control, which was running when this was
written. Findings in `RESULTS.md`; where the defect actually lives in `SCOPE.md`.

## Where the protocol was fixed

Honestly stated, because the house style asks for a pre-registered protocol and this experiment did
not start with one in this directory. The protocol was fixed in the driver headers and committed to
those files before each run, not in a README written first:

- `articles/models-deep-dive/tools/activation_cost_on_isolated_host.sh` — throughput. Header states
  the two arms, the alternating A B A B schedule, one dedicated-vCPU host with nothing else on it, a
  fresh JVM per run, warmups inside each run, the median estimator, and the decision rule: a
  two-percentage-point precision target, with a per-pair spread straddling 1.00 ± 0.02 reported
  UNRESOLVED and never as a result.
- `articles/models-deep-dive/tools/cohort_identity_on_isolated_host.sh` — output identity. Header
  states the arms, the single fixed template, the 120 JevBench cases, exact token comparison against
  a common baseline, and why a bound is measured rather than a rubric reconstructed.

Both drivers abort rather than report on an empty arm, a degenerate baseline, or a projected runtime
that cannot finish inside the budget. The route is asserted identical across arms before any ratio
is attributed.

One protocol choice was made **after** seeing data and is called out in `RESULTS.md` rather than
presented as foresight: the pre-registered median estimator assumed one mode per arm, and the
released baseline turned out bimodal. The median is still reported, alongside the mean-to-mean ratio
and per-mode ratios, with the reason the median alone is the wrong summary.

## Kill criterion

A candidate is rejected if it costs throughput the user will not accept, or if it changes decision
labels on the cohort with a net quality loss. `b3c1245d` was rejected on the first: ~9% of expected
decode throughput. It was **not** rejected on the second — 2 of 120 labels differ, flipping in
opposite directions, zero net change — which contradicts the "76/120 vs 75/120" figure recorded in
the tree, whose run returns 404 and whose harness is gone.

## Files

| file | what it is |
| --- | --- |
| `README.md` | this |
| `SCOPE.md` | where the defect lives: three activation sites, one predicate, catalog exposure |
| `RESULTS.md` | measured costs and output impact of every candidate, with what is not established |
| `CohortIdentity.java` | the harness: loads the model once, greedy-decodes N tokens per cohort case |

Raw artifacts live in `articles/models-deep-dive/evidence/activation-cost-*` and
`cohort-identity-*`. The analyzers that derive every reported number from them are
`articles/models-deep-dive/tools/analyze_activation_cost.py` and `analyze_cohort_identity.py`; no
number in `RESULTS.md` is hand-typed.

## Reproduce

    cd articles/models-deep-dive
    python3 tools/analyze_activation_cost.py evidence/activation-cost-<stamp> <baseline> <candidate>
    python3 tools/analyze_cohort_identity.py evidence/cohort-identity-<stamp> baseline <arm>...

Re-running a measurement provisions a paid host and refuses to start without `--confirm`. See
`VPS-BENCHMARK-OPERATIONS.md`.

## Candidate revisions

All on the vectors repository. `44211d46`, `b8ad9fcf` and `35732bb2` are experiment arms and are
labelled EXPERIMENT ONLY in their own source comments.

| revision | arm |
| --- | --- |
| `v0.1.30` / `v0.1.31` | released baselines |
| `b3c1245d` | `ReproducibleActivationExp` substitution; rejected on cost |
| `44211d46` | published polynomial at its hardcoded `SPECIES_128`; refuted at 12% |
| `b8ad9fcf` | same polynomial at native vector width; leading candidate, changes no arithmetic |
| `35732bb2` | sigmoid-specialised exp; correct and bit-stable but breaks recipe parity |
