# Final composed GDN correctness and task quality

Prepared, not dispatched. Both model sizes and both architectures use the exact
Models and Vectors revisions in protocol.json; their actual current class entries
and native-library hashes must match the final Bonsai artifact manifests. The
published arm is the immutable historical published-runtime payload from the
previous comparison, preserved by whole-archive checksum. It is not relabeled
as the current composed runtime.

The Java recurrence control changes only the actual current provider's
`supportsGatedDeltaNet()` result to false. Native quantized projections stay
active. Exactly one class entry changes; the compiled native library is identical
between current and Java-recurrence controls. All 148 cases retain all nine full
logit hashes, input tokens, greedy tokens and label predictions. Those controls
must agree exactly. Published differences remain listed individually; each
dataset's current correct count must not decrease. Known historical published
routing failures remain failures but do not prevent collecting its independent
cohort. Current routing failures block the corresponding cohort.

This phase does not qualify latency, cold allocation, RSS or startup. Those
remain separate frozen paired measurements. A successful kernel screen or the
previous dense-Qwen fixtures cannot replace this cohort. Each arm uploads its
outputs immediately; no failed observation is overwritten. Data derives from
the existing pinned JevBench/SNIPS 148-case cohort, with both licenses included.
The secret GDN_REFERENCE_URL must contain a valid read-only URL for the exact
checksum-bound historical source/runtime archive. It contains no model weights;
weights are downloaded remotely from their pinned publisher revisions.
