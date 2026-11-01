# Prospective fixed Qwen full-metric phase

**Authorized for launch on the isolated13f0bd51 BF16 artifact.** The bounded comparison removed
48B/op from the ARM serial case; x86 serial and both parallel latency intervals remain unresolved.
They are retained as limitations, not reclassified as passing gates. This separate final-artifact
whole-model cohort measures the published comparison directly.

Keep all observations from run38321278750 and its failed precision verdicts unchanged. A separate
fixed full-metric Qwen cohort can resolve effect-size precision more cheaply than separate latency,
memory and allocation experiments: each existing seven-request process already measures all of
these. It must retain the same two public datasets, actual immutable artifacts, fresh-process
fixtures, full-logit repeatability checks, workload, settings and unchanged acceptance limits.

Using the standard deviation of the ten existing paired log ratios, the planning approximation
`n = ceil((3.92 * sd(log ratio) * observed ratio / 0.02)^2)` gives:

| Architecture | Metric | Minimum n for expected interval width 0.02 |
|---|---|---:|
| x86 | whole request | 214 |
| x86 | decode | 185 |
| x86 | cold request | 104 |
| x86 | RSS | 33 |
| x86 | allocation | 28 |
| ARM | RSS | 95 |
| ARM | allocation | 41 |

**Authorized fixed counts: 320 x86 pairs and 160 ARM pairs**, approximately 50% above the largest
planning minimum per architecture. Expected worst interval width is approximately 0.0163, with no
guarantee that host variation will match the first cohort. Fix these counts before launching; no
optional stopping, data removal, replacement of measured failures, or pooling with the original run.
The separately launched 400-pair load-only phase addresses startup and remains its own estimator.

Four x86 shards of80 pairs and two ARM shards of80 minimize duplicated setup while keeping expected
wall time near one hour. Existing completed process wall means are21.318s x86 and24.857s ARM. At two
processes per pair this forecasts about3.79 x86 runner-hours plus2.21 ARM runner-hours, before setup
and fixtures. Each shard is approximately57–66minutes plus those costs. Keep incremental raw
checkpoints outside timed regions. The frozen30 measured tasks cycle in subsequent pairs: this
increases performance sampling, not the number of independent quality cases.

The Qwen phase is authorized as a separate fixed cohort. Bonsai remains held; derive any Bonsai
count from its completed evidence before launching a separately frozen phase. No further load-only
experiment is added; model-load measurements already inherent in each unchanged process are retained. Do not borrow
Qwen noise estimates for the much larger Bonsai model.

## Interpretation

Existing Qwen intervals wholly below their limit but wider than0.02 support the direction of
non-regression while failing the required effect-size precision. ARM model-load startup also crosses
1.01, so it fails to resolve non-regression itself. Neither category is a passing frozen gate.

The startup count was chosen from the earlier CI-implied log SD≈0.029. Direct sample SD of those same
ten ARM load ratios is0.03199; that predicts400-pair width0.00627 and about87% normal-approximation
power at true ratio1.005, versus the original CI-derived planning estimate≈93%. Both are uncertain
planning approximations; the fixed400 count and the original1.01/0.02 limits remain unchanged.

The candidate is91dd64ae / SHA25613f0bd51, with the bounded supported-host findings above.
Full JAR comparison proves only `BFloat16Matrix.class` differs from the measured combined e3e7
artifact; no added or removed entries. The fixed cohort size and all metrics/limits are unchanged.
New core correspondence is retained in `candidate-correspondence.json`.
