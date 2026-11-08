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
