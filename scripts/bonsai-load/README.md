# Prospective model-load startup follow-up

This separate phase resolves model-load startup only. It does not replace, pool with, or relabel
run 38321278750. The original complete Qwen results remain failed/unresolved under their frozen
precision gates. Bonsai whole-model results remain a separate prerequisite for any Bonsai release.

The fixed sample count is **400 matched fresh-JVM pairs per model and architecture**, prospectively
partitioned into two independent runners of 200 pairs. Every pair executes both exact artifacts on
the same runner, alternating order. Qwen x86 and ARM launch first; Bonsai uses the same independently
specified protocol only when authorized. No optional stopping, outlier removal, measured-process
retries, sample-count changes, or additional pairs based on the observed confidence interval.

The completed 10-pair ARM Qwen model-load interval was [0.9896796, 1.0262218]. Its log-scale width
implies approximate paired log standard deviation 0.029. At 400 pairs, the expected 95% interval
width is about 0.006, below the unchanged 0.02 requirement. If the true ratio is 1.005 and that noise
estimate transfers, a normal approximation gives about 93% probability that the upper 95% endpoint
falls below 1.01. These are planning approximations, not guarantees: host effects, non-normal tails,
or an actual ratio near 1.01 can leave the fixed result unresolved. The same fixed count is used on
both architectures and both models before observing any follow-up data.

The candidate/baseline geometric ratio uses the existing host-stratified 20,000-resample bootstrap,
seed 716309. Acceptance still requires upper 95% endpoint <= 1.01 AND interval width <= 0.02. Missing
pairs or failed processes fail acceptance. The original observations never enter this estimator.

The exact published baseline JAR, combined candidate JAR, all model/dependency JAR entry hashes,
architecture-specific native binary, Models and Vectors commits, public model weight hashes, data
files, JDK 25.0.3, default inference route, context budget, and 2g/10g heap settings match the macro.
CPU, JDK, runner, commands and full artifact provenance are captured anew. This measures fresh JVM
model loading after the same runtime-origin output and frozen task JSON setup, with no inference.
It does not flush filesystem caches or claim a disk-cold result. Closing the model and reporting
are outside the timed load region. No RSS, allocation, request latency, or quality gate is inferred
from this phase; GNU time output is diagnostic only.

Every completed pair emits a live progress line and durable receipt. Every fixed block of 20 pairs
uploads an immutable checkpoint outside timed regions. Each process has a 120-second failure bound;
each block and whole job are also bounded. A timeout retains raw partial logs and fails the phase.
Checkpoint uploads may change the delay between blocks, equally for both arms; no upload runs while
a timed process is active. Re-running failed jobs creates a new separately identified experiment,
not a repair of missing observations in this cohort.

## Meaning of existing unresolved results

An interval entirely below the non-regression limit but wider than 0.02 establishes the requested
non-regression direction at the reported confidence level, while failing the prospectively required
precision of the effect-size estimate. This applies to complete x86 Qwen decode, whole request,
cold request, load, RSS and allocation, and ARM Qwen RSS/allocation. Their original gate still fails.

ARM Qwen model load also crosses 1.01; that is actual uncertainty about non-regression as well as
insufficient precision. The extra prospective startup cohort targets that uncertainty. It cannot
make unrelated precision-only misses disappear or turn the original macro into a passing run.
