# Flint vector attention performance diagnosis — decode floor not met

Runtime and worker revision: `9c453e9de85f77b398e9ad57eba13e305db96f72`.
This is a single-case speed diagnosis, not catalog qualification. The unchanged short-reference
parity gate precedes sequential native/Ollama arms at 512 tokens, one warmup, three iterations,
context 2048 and 16 threads on one m6a.4xlarge. The original full qualification is still required.

Instance `i-0c61157d84757049d` launched at 2026-10-30T19:08:14Z, with independent
90-minute termination and root-volume deletion. The instance is terminated and its root volume is deleted.

Both arms completed: native median decode 9.37 tokens/s, Ollama 14.53 tokens/s (64.45%).
This misses the unchanged 80% target. Native p95 end-to-end was 15.29 s versus 31.67 s;
native emitted 143 tokens and Ollama 454. All visible answers were complete and correct.
The native raw answer matches the previous runtime. Exact prompt hashes match across arms.
The one repeated question does not establish workload qualification. `FAILED_QUALITY` in
the single-case reports reflects missing abstention coverage, not an incorrect answer.
No full campaign is launched on this result; worker-count diagnosis is next.
