# Flint full qualification — RAG policy passed, held-out run continuing

Frozen runtime/worker: `fb5d7a68248ca14da832ae886016b86d62060382`.
Includes ordered Q4_K, vector attention, and bounded immutable F32 router snapshots.
Candidate public native pool: 8 workers; default smoke and Ollama control: 16 workers.
The unchanged `production-rag-model-contribution-v6` policy returned **QUALIFIED**.
Native median decode is 11.9275 tokens/second versus Ollama's 14.3025: ratio 0.83395,
above the 0.80 floor. Native p95 end-to-end latency is 37,910.1 ms versus 26,138.7 ms:
ratio 1.45035, below the 1.50 ceiling. Absolute tier is `PERFORMANCE_REDUCED`; this is
not a low-latency or frontier-equivalence claim.

All 27 visible answers pass completion, facts, citations, and polarity checks across nine
distinct questions. The model contributes 18/27 answers, with all accepted model answers
correct; six use extractive fallback and three abstain. All raw answers match the reviewed
512-token pilot. The original 256-token default smoke passes the pipeline but truncates four
answers; it does not establish complete visible-answer accuracy at that budget.

Fresh held-out answers, released-library verification, and catalog admission are still pending.
The six-worker diagnostic is separate and does not alter this campaign's frozen profile.

Instance `i-0ddc4e4b6b6ffae04`, root `vol-07ed6544dd1caa0d4`, launched 2026-10-30T20:01:35Z.
Root: 120 GiB gp3, 500 MiB/s, 3000 IOPS, DeleteOnTermination. Independent four-hour shutdown,
terminate-on-shutdown; bootstrap worker timeout is 14100 seconds. Cleanup pending.

Full Models build/compliance/installDist passed in 10m58s (399 tasks). All Flint fixture tests
passed separately. ModelJars branch checks passed in 1m47s, and all 357 Node tests passed.
The local synthetic router timing uses random arrays, not model inference, and is not whole-model
speed evidence. Real Flint inference occurs only on the remote host.
