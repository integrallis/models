# Declared integration confirmation — 2026-11-08

The first new integration attempt used one warmup and three measured repetitions.
All three encoder probes passed; the F32 decoder reference passed. The F16 reference
decode gate failed narrowly. The raw result is retained and will not be overwritten.

Inspection of the prior performance archive showed that the earlier passing PR result
used **three warmups and seven measured repetitions**, with the same JDK version.
The first new run therefore did not repeat the earlier warmup protocol. This does not
invalidate its failure or prove warmup is the cause.

Before collecting further integration timings, declare one repeat using the prior
three/seven protocol on **both arms**, after the entire task campaign finishes. The
runtime, artifacts, context, threads, prompts, original-weight kernels, KV choices,
comparison script and 1.10 ceiling remain unchanged. Run all three image probes and
both F32/F16 decoder references. Save to `integration-confirmation-3w7r`, a new
directory. Keep both attempts visible in the report; no best-run selection or
averaging across incompatible warmup schedules. If the confirmation fails, do not
claim the integration gate passed. The task-set one/three protocol remains frozen.

This repeat diagnoses the short warmup miss; it is not an optimization or proof that
the model qualifies. Dataset accuracy, task performance, image contribution and the
ModelJars route still have independent gates.
