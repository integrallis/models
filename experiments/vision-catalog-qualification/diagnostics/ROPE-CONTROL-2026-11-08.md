# Remaining Q4_K_S disagreement: rotary arithmetic experiment

Recorded before diagnostic execution, 2026-11-08 21:50:51 UTC. This supplements the
arithmetic campaign; the frozen corpus and gates are unchanged.

The complete fresh reference pass at candidate
`c7acc590147032ae78adad01ae1e869b64922274` agrees on all 80 Q4_K_M cases and
79/80 Q4_K_S cases. Q4_K_S VQAv2 `454129000` returns `Rectangle`
(`[22914,2521]`) versus the reference's `Rectangles` (`[22914,19536]`).
The final confirmation/performance verdict is still pending at registration.

Static inspection identifies a distinct numerical hypothesis: the pinned Linux
reference's interleaved rotary loop uses `vfmaddsub231ps`; Java's `rotatePair`
performs separately rounded products and additions. The reference also calls
float `sincosf`/`powf`, whereas Java rounds double `Math` results to float.
These are differences to measure, not yet an explanation of the token mismatch.

Use the already-owned Q4_K_S m6a.4xlarge worker after its full qualification and
archive complete. No workload overlaps timing. Existing batch authorization,
combined USD 6.50 ceiling, collector and 2026-11-09 00:13:33 UTC deletion deadline
remain in force. No additional machine is provisioned.

1. Cross candidate/reference image embeddings with both decoders. Require the
   unobserved diagonal outputs to reproduce before interpreting the cross cells.
2. Trace the first prediction and a separate one-token forward call forced with
   shared token 22914. Capture every layer and the final logits, plus layer-zero
   projections/attention. Record that observers disable final-layer query pruning;
   compare an unobserved no-pruning control as well as traced predictions.
3. In external classpath overrides only, compare the current rotary path with
   (a) FMA rotation preserving the reference's vector operation order, and
   (b) the same rotation plus Linux float libm factors through an external FFM
   diagnostic. The libm diagnostic is not a proposed runtime dependency or source
   import. Capture hashes, commands, complete tokens and layer/logit errors.
4. A promising candidate must improve the independently compared arithmetic,
   reproduce the complete answer, and pass all five earlier answer regressions.
   Do not select a different formula per prompt, token, label, or quantization.
5. Any retained production change gets independent arithmetic fixtures, normal
   tests/CI, then a complete frozen qualification repeat on both quantizations.
   Neither this single case nor an observer-assisted match establishes admission.

The scripts and all successful/failed outputs are retained in the worker evidence
archive. No alternate experimental production path is retained if it loses.
