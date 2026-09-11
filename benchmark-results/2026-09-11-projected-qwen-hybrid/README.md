# Projected Qwen hybrid qualification

This gate compares a single Qwen3 1.7B Q8_0 model with a virtual model that routes prose and
tool-result narration to Qwen3 0.6B Q4_0 and tool selection to the same Qwen3 1.7B Q8_0 model.
Both arms use the same six-turn conversation, tools, system instruction, greedy sampling, output
limit, pinned artifact bytes, Models revision, JVM, and dedicated host.

The hybrid does not copy KV state between models. Each physical member keeps an independent exact
prompt/KV lineage. Its qualified context policy sends translated tool results and normal prose
history to the chat member, while the stateless tool specialist receives only the current tool
selection turn. That removes irrelevant cross-model catch-up without claiming portable KV.

## Result

Every turn passed in three fresh control JVMs and three fresh hybrid JVMs. The counterbalanced run
order was C1, H1, H2, C2, C3, H3.

| Arm | Fresh-process totals | Median | Median peak RSS |
| --- | --- | ---: | ---: |
| Qwen3 1.7B control | 57.119 / 53.166 / 52.698 s | 53.166 s | 2,404,032 KiB |
| Projected Qwen hybrid | 35.103 / 35.151 / 35.252 s | 35.151 s | 3,270,444 KiB |

The hybrid improves median end-to-end generation time by **33.88%**, clearing the predeclared 5%
gate. Its median peak RSS is **36.04% higher** because both weight sets remain resident. This is a
latency-qualified composition with an explicit memory tradeoff, not a memory optimization.

The result is driven by capability-aware prompt composition: the control reflects the usual
application contract in which registered tool schemas remain visible on every turn, while the
hybrid keeps that protocol burden on the tool member and runs ordinary chat on the smaller member.
The tool member's first switch is cold; its switch-back reuses only its own retained prefix.

## Environment and lifecycle

- Models revision: `e4d130dd8c5986e6cef6d7ff5cb7d3533a5ceb6b`
- Host: Hetzner `ccx33`, Ashburn; eight dedicated AMD EPYC-Milan vCPUs, 32 GB RAM
- Runtime: Ubuntu 24.04, Temurin 25.0.4.1, 256-bit Panama vectors, eight GGUF workers
- Artifacts: Qwen3 0.6B Q4_0
  `da2572f16c06133561ce56accaa822216f2391ef4d37fba427801cd6736417d4` and Qwen3 1.7B
  Q8_0 `061b54daade076b5d3362dac252678d17da8c68f07560be70818cace6590cb1a`
- Server `165547873` was created at 2026-09-11T18:59:20Z and confirmed deleted at
  2026-09-11T19:18:22Z. With hourly price USD 0.266, the compute-charge upper bound is USD 0.266.
- Post-deletion audit: the server returned HTTP 404 and no experiment-owned volume, primary IP,
  floating IP, or snapshot remained. The unrelated `sota-hippo-20260911` server was not touched.

`comparison.json` is the machine-readable qualification decision. The six arm reports retain exact
outputs, routing boundaries, token/cache metrics, JVM memory, process memory, and artifact evidence.
The matching logs include `/usr/bin/time -v` peak RSS. `SHA256SUMS.txt` verifies the files copied
from the VPS before it was deleted.
