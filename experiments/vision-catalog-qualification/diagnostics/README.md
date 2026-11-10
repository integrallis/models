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

`run.py` requires both collection-complete markers and checks the actual native binary
and executable JARs against the measurement receipt. Completion of separately queued
diagnostics is still an operator precondition; do not overlap workloads.

`LayerTrace` accepts an optional forced token after the output directory;
`layer-oracle` accepts it after the attention mode (`auto` or `off`). Both feed it
through a separate one-token decode call so a second-token disagreement can be
localized without changing the prefill shape. The Java observer reads scalar
workspace during decode and batched workspace during prefill. Observing layers
disables the final-layer query-pruning optimization, so reproduce the unobserved
diagonal first and compare the traced prediction before interpreting layer deltas.
