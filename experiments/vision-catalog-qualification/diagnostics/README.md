# Disagreement localization

This diagnostic is separate from admission. It does not replace any measured outputs
or change thresholds. Run only after the paired campaign and its queued repeats finish.

For each original question whose generated tokens differ, retain the existing reference
encoder's embeddings and export the candidate encoder's embeddings with the shipping
`VisionQualificationCli`. Decode both embeddings with both decoders, using the same
saved prefix/suffix token IDs, 8 threads, 1024 context, F32 KV, and greedy 32-token cap.

The four cells identify whether the discrepancy follows the embeddings, the decoder,
or their interaction. First reproduce the diagonal cells against the original outputs.
If they do not reproduce, investigate nondeterminism before interpreting the cross cells.
The diagnostic does not establish performance or full-corpus qualification.

The diagnostic classpath selects valid executable JARs and checks their hashes against
the measurement receipt. macOS AppleDouble `._*.jar` metadata is excluded explicitly:
the JVM ignored it during execution, but `javac` rejects it as an invalid archive. The
first diagnostic compilation failure is retained separately; no inference ran in that
failed diagnostic attempt, and correcting its classpath does not alter the measured
payload or repeat any qualification timings. Future macOS payload archives should use
`COPYFILE_DISABLE=1` when invoking `tar`.

## Exact reference attention substitution

`attention-substitution.py` compiles `oracle-attention.cpp` against the pinned
independent ggml CPU library and injects `OracleAttention` through the existing
attention SPI in the standalone decoder diagnostic. It preserves the Java graph,
weights, projections, tokenizer and greedy generation, replacing only attention.
Both candidate and independent encoder embeddings are crossed with both attention
implementations and the complete reference decoder. The previous candidate tokens
must reproduce before interpreting the result.

This tests the reference's actual F16 casts, tiled prefill, online softmax,
half-precision accumulation and split-KV decode together. Merely rounding operands
in our existing F32 kernel does not reproduce that calculation. Analytic bridge
checks cover tensor layout, causal masking and split-KV lengths before model cases.
The probe is external research code and must never enter a runtime artifact or a
qualification performance result. Its eight selected cases localize the problem;
only the complete frozen corpus can establish admission.

On the 2026-11-08 attention-dispatch worker this probe is queued after the timing
worker exits, capped at 15 minutes and skipped if fewer than 15 minutes remain
before the 19:40 UTC collection safety window. The independent 19:50 UTC deletion
deadline is unchanged. Capture partial reports and failures in the raw archive.
