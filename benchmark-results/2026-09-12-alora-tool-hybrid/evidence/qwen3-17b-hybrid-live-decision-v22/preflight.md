# Qwen3 1.7B hybrid live-decision V22 remote preflight

Status: **frozen before V22 provisioning or model scores**.

V21 began on the local Intel i7-9750H Mac and completed its first in-process live case in
389,649 ms. At that rate its 75-case screen would monopolize the workstation for roughly eight
hours. V21 was deliberately interrupted after that first mechanics observation; it produced no
complete report and cannot pass. V22 retains the exact V21 scorer, inputs, policy, and gates, but
uses a temporary Linux CPU host so infrastructure does not block qualification progress.

## Frozen inputs and gate

- Models scorer revision: `e2dab1e2749199792237bd388e70fba04076335b`.
- Base Qwen3 1.7B Q4_K_M GGUF SHA-256:
  `061b54daade076b5d3362dac252678d17da8c68f07560be70818cace6590cb1a`.
- Activated adapter SHA-256:
  `f1a4a1bc15e3942ed5a3fbe2c4750f39a8b24804e20566242ec740c5ce851c21`.
- BFCL-live records SHA-256:
  `4d70a040341480ccbae9af5a00d83bc4e7365ebead67e390cb5c3763ed83b30a`.
- BFCL-live manifest SHA-256:
  `f1815fa298e8b4e205b061a26ed8a1043e7aa688f7a4b40d034e7c329d14507a`.
- Contextual call/no-call IDs: `4913` / `19536`.
- Policy: primary specialist margin `> 4.5`; otherwise the V20 rescue conditions remain
  specialist `> 1.3521204`, base-minus-specialist `> 33.705593`, and base `< 39.42147`.
- Pass: exactly 75 cases with a 50/25 class split, at least 48/50 correct calls, at least 24/25
  correct no-calls, and 75/75 physically shared KV prefixes.

The BFCL-live window is previously exposed development evidence. A V22 pass only permits a
separately frozen generation screen; it does not qualify or publish a hybrid model and does not
open the sealed 300-case window.

## Bounded infrastructure

- Provider / location / plan: Hetzner Cloud `ash`, `cpx51` (16 shared vCPU, 32 GB RAM, 360 GB).
- Image / SSH key: Ubuntu 24.04 image `161547269`; key `vectors-bench-v2` ID `114663461`.
- Reference hourly price: USD 0.4479/hour from the immediately preceding V19 run.
- Creation deadline: `2026-09-14T16:40:00Z`.
- Hard deletion deadline: `2026-09-14T18:30:00Z`.
- Maximum upper-bound compute cost: USD 0.90.
- Firewall: SSH TCP/22 from operator IP `68.227.243.139/32` only; no public inference ports.

The host receives a fresh detached checkout at the exact scorer revision, checksum-pinned model,
runtime adapter, and records. It runs `ActivatedHybridLiveDecisionCli` directly with Java 25.
No Ollama, llama.cpp server, Python inference runtime, Docker inference service, or Rust inference
kernel may be installed or used. The host is deleted after the output report is copied and its hash
is verified, or at the hard deadline. Every server, firewall, volume, floating IP, primary IP, and
snapshot is audited after deletion.
