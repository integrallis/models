# Flint full qualification — running

Frozen runtime/worker: `fb5d7a68248ca14da832ae886016b86d62060382`.
Includes ordered Q4_K, vector attention, and bounded immutable F32 router snapshots.
Candidate public native pool: 8 workers; default smoke and Ollama control: 16 workers.
All original admission gates remain. No qualification verdict is asserted until reports finish.

Instance `i-0ddc4e4b6b6ffae04`, root `vol-07ed6544dd1caa0d4`, launched 2026-10-30T20:01:35Z.
Root: 120 GiB gp3, 500 MiB/s, 3000 IOPS, DeleteOnTermination. Independent four-hour shutdown,
terminate-on-shutdown; bootstrap worker timeout is 14100 seconds. Cleanup pending.

Full Models build/compliance/installDist passed in 10m58s (399 tasks). All Flint fixture tests
passed separately. ModelJars branch checks passed in 1m47s, and all 357 Node tests passed.
The local synthetic router timing uses random arrays, not model inference, and is not whole-model
speed evidence. Real Flint inference occurs only on the remote host.
