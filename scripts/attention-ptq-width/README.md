# Reachable fixed-width component diagnosis

The original frozen diagnosis at revision `762ecb3d` measured existing grouped attention scores/softmax and Java PTQ1_0×Q8_0 row products and the published float PTQ API. The follow-up measures the exact native-width correction at the candidate revision pinned in `protocol.json`; it does not claim whole-model qualification. Grouped attention is the Granite/default-fallback or explicit-plan path; PTQ is the Java/unsupported-native route, not the normal native PTQ projection.

Sixteen fresh-JVM same-host three-arm blocks per component compare the frozen production class, candidate and independent primitive reference. All three orders reverse on alternating blocks. Candidate/current parity also runs over 2,328 shape/offset/tail/extreme-value cases with default and forced128 physical widths before timing. The control keeps eight logical score/sum lanes, the exact fold, true FMA, integer PTQ grouping and scale-FMA order. Thirty-one changing-input output signatures and cold/hot signatures must match exactly. Warmup is fixed at 2,000 calls; five measured blocks each run at least 250 ms, checking time only after a 992-call score/softmax batch or 1020-call PTQ batch. Each batch contains full 31-input or 255-input mutation cycles. Actual call counts/durations are retained, and no block is discarded. All inputs/output buffers are retained. Softmax resets its input using the same arraycopy in both arms. Cold allocation includes each arm's own first-use initialization; warmed allocation and latency are recorded separately. These synthetic shapes are diagnosis, not model performance claims.

JDK 25.0.3, actual CPU/VM flags, frozen source/classes and downloaded dependency hashes are retained. Every four process triplets upload before the next group; all partial records are retained on failure. Separate JFR processes attribute allocations; their latency is never pooled with uninstrumented measurements. No failed process or partial run becomes a passing result.

The first native128 PTQ prototype removed warm allocation but increased cold allocation. The final candidate uses primitive arithmetic on hosts without native256 support to satisfy both constraints, retaining the eight float chains and fixed integer group/scale folds. Wider hosts retain their existing vector implementation. The public float API remains available. Softmax separates its exponent preparation from the pinned sum through array-only phases, keeping vector values inside each compiled phase. No warmup counters, caches or phase-dependent dispatch are used.

## Running it from main, 2026-11-01

The five GitHub workflows that drove this originally are **not** carried into `main`. Each triggered
only on `push: branches: [fix/attention-ptq-width-proof]`, a branch that no longer exists, so on
`main` they would have been five files that could never fire. `run.py` and `attention-final.py` are
the entry points; they take a source tree per arm and compile each with `javac --release 25
--add-modules jdk.incubator.vector`.

The probes were verified against `main` before this harness was landed. `WidthParityProbe`, run with
`main`'s own compiled `backend-java` classes as both arms, reports:

```
{"cases":2328,"exact":true,"preferredBits":256}
```

so the reflective contract it asserts — `GroupedQueryAttentionKernel.scoreGroup` with its
thirteen-parameter signature, `softmax(float[],int,int)` and
`TernaryDequantizer.dotProductPtq1_0Q8_0` — still matches `main`. All three probes compile against
`main` with no changes.

The source baselines each `*-protocol.json` pins live on no branch. They are tagged
`archive/evidence-20261101-{61a47f548d1f,7633a70cc24e,7ad622a40451,d72380a39d9b}` so the protocols
keep resolving; `c95e243d` is on `main` already.
