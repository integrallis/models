# Projected Qwen hybrid qualification preflight

- Experiment: compare one Qwen3 1.7B Q8_0 control with a Qwen3 0.6B Q4_0 chat / Qwen3 1.7B
  Q8_0 tool-selection virtual model using capability-specific context projection.
- Provider / region / exact plan: Hetzner Cloud, `ash`, `ccx33`; eight dedicated x86 vCPUs,
  32 GB RAM, 240 GB disk, Ubuntu 24.04 x86-64.
- Live price: USD 0.266/hour; hourly billing.
- Maximum authorized spend: USD 0.798 (three hours).
- UTC creation deadline: 2026-09-11T19:15:00Z.
- UTC deletion deadline: 2026-09-11T21:58:44Z.
- Correctness gate: all six protocol turns pass in every one of three fresh control and three
  fresh hybrid JVM processes.
- Performance gate: hybrid median end-to-end generation time improves on control by at least 5%.
- Memory evidence: record per-turn process RSS and after-load JVM/native memory; a memory increase
  does not silently become a memory win.
- Run order: control 1, hybrid 1, hybrid 2, control 2, control 3, hybrid 3.
- Exact Models revision: `e4d130dd8c5986e6cef6d7ff5cb7d3533a5ceb6b`.
- Local evidence destination: this directory.
- SSH key: Hetzner key `114663461` (`vectors-bench-v2`), verified against local ED25519 MD5
  fingerprint `60:4e:57:38:96:48:c5:45:73:10:9d:37:f7:4f:74:cf`.

The production comparison uses Qwen3 1.7B because it is the smallest currently qualified local
generative member that passed the complete tool-calling suite and can narrate tool results. The
hybrid routes prose and tool-result narration to Qwen3 0.6B, and exposes only the current selection
turn to the 1.7B tool member. Model weights remain independent and no KV state is copied.
