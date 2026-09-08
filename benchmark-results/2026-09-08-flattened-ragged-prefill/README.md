# Flattened ragged-prefill experiment

## Preflight control record

- Experiment: flatten compatible prompt rows from independent sessions into physical projection
  batches while retaining separate per-session KV state.
- Provider / region / exact plan: Hetzner Cloud, `ash`, `ccx33`, 8 dedicated x86 vCPU,
  32 GiB RAM, 240 GiB disk, Ubuntu 24.04 x86-64 image `161547269`.
- Live price checked: USD 0.266/hour gross on 2026-09-08 at 12:32 UTC.
- Maximum authorized spend: USD 0.532 (two hours); delete earlier as soon as evidence is copied
  and verified.
- UTC creation deadline: 2026-09-08T12:45:00Z.
- UTC deletion deadline: 2026-09-08T14:45:00Z.
- Control commit: `96ee1c20907d72f1557a1756f387d5042dca8f3a` (Models 0.3.35).
- Candidate commit: `e869ae507aeb73faf1b53891d49dbb713fa92ba9`.
- Correctness gate: the pinned MiniCPM5 real-weight ragged-prefill test must match independent
  final logits, complete KV buffers, checkpoints, and the next-token continuation exactly. Every
  measured arm must produce the same trace hash as its control.
- Performance gate: accept only if the candidate improves median aggregate ragged prompt
  throughput by more than 5% without regressing either pinned model materially. Otherwise retain
  the evidence and reject or revise the implementation.
- Models: pinned MiniCPM5 1B Q4_K_M SHA-256
  `81b64d05a23b17b34c475f42b3e72fbde62d4b92cc34541f7a8031d0752deafa` and Qwen3 0.6B Q4_0
  SHA-256 `da2572f16c06133561ce56accaa822216f2391ef4d37fba427801cd6736417d4`.
- Measurement design: fresh JVM per arm; counterbalanced control/candidate order; identical
  prompts, context, concurrency, warmups, iterations, JVM flags, model bytes, and host.
- Local evidence destination: this directory. Remote evidence must be copied back and hashed before
  decommissioning.

## Lifecycle record

- Provisioned server `165142090`, `modeljars-bench-flat-prefill-20260908`, at approximately
  2026-09-08T12:34Z. Hetzner assigned recycled IPv4 `5.161.100.228`; the run therefore uses an
  experiment-local `known_hosts` file and verifies the new host key independently.
- Host key: ED25519 `SHA256:UCzXQ2Nhr7I41LsqCwg8NNns+97gjsFDLOvcS4rFrPo`.
- Environment: Ubuntu 24.04.4, Linux `6.8.0-138-generic`, AMD EPYC Milan, eight logical processors
  presented as four cores / eight threads, 32 GiB RAM, Temurin 25.0.4.1, 256-bit Panama vectors,
  eight GGUF threads.
- The pinned real-weight MiniCPM5 test matched independent inference exactly for final logits, the
  complete key and value buffers, session checkpoints, and the next-token continuation.
- Twelve fresh-JVM reports ran from 2026-09-08T12:46:19Z through 2026-09-08T12:51:09Z in
  control-candidate-candidate-control-control-candidate order for each model.
- The 12 copied JSON reports matched the remote SHA-256 manifest before deletion.
- Server deletion was requested at 2026-09-08T12:52:44Z, approximately 19 minutes after
  provisioning. The compute-charge upper bound is therefore USD 0.266.
- Post-deletion audit: zero servers, volumes, floating IPs, and snapshots. Two automatically
  assigned primary-IP records were visible while server deletion was settling; a follow-up query
  returned zero primary IPs.

## Result

Every run produced the same output/continuation hash as its model control. Median peak RSS did not
increase: it fell from 1,814,974,464 to 1,774,239,744 bytes for MiniCPM5 and from 1,242,341,376 to
1,233,125,376 bytes for Qwen3.

| Model | Models 0.3.35 control | Flattened candidate | Change |
| --- | ---: | ---: | ---: |
| MiniCPM5 1B Q4_K_M | 22.51 prompt tok/s | 23.82 prompt tok/s | +5.81% |
| Qwen3 0.6B Q4_0 | 45.98 prompt tok/s | 52.48 prompt tok/s | +14.12% |

The geometric-mean throughput improvement is 9.89%. Both models cleared the predeclared 5% gate,
so this implementation is accepted for broader source verification. This result concerns
multi-session prompt/catch-up throughput; it does not by itself remove the first-switch latency of
a single hybrid conversation.

Raw JSON reports, JVM logs, hardware metadata, lifecycle timestamps, and the remote digest manifest
are under `remote-results/`. Derived values are retained in `comparison.json`.
