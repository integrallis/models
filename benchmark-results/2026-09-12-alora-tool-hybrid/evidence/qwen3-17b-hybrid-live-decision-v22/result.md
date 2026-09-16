# Qwen3 1.7B hybrid live-decision V22 remote result — rejected early

V22 ran the checksum-pinned Java-only scorer described in `preflight.md` on the bounded Hetzner
CPU host. It is a **rejected development replication**, not qualification: the applicable-call
gate became mathematically impossible after case 37, so the scorer was deliberately stopped
rather than consuming the remaining host budget. The sealed 300-case qualification window was
not opened. No hybrid artifact or release is authorized.

| Frozen gate | Required | Observed when stopped | Result |
| --- | ---: | ---: | --- |
| Correct applicable-call decisions | at least 48/50 | 9/12 | fail; three misses already |
| Correct irrelevant/no-call decisions | at least 24/25 | 24/25 | pass; completed |
| Physical KV-prefix sharing | 75/75 | 37/37 observed | incomplete; no failure observed |

The three applicable misses were `live_multiple_1051-278-0`,
`live_multiple_32-10-2`, and `live_multiple_673-162-15`; each produced `NO_CALL` where a call
was expected. Because the frozen call floor permits at most two misses, case 37 made a pass
impossible. The host process was sent SIGINT, then TERM after it did not exit promptly; the
partial run log was copied before host deletion.

Partial log SHA-256:
`9b641ad35b2e56914f6b8e147189a63b569be9c430c67cf2acd99fe9ae7e8c5a`.

The log records the same pinned inputs and Java-only execution described in the preflight:
Models scorer revision `e2dab1e2749199792237bd388e70fba04076335b`, base GGUF
`061b54daade076b5d3362dac252678d17da8c68f07560be70818cace6590cb1a`, adapter
`f1a4a1bc15e3942ed5a3fbe2c4750f39a8b24804e20566242ec740c5ce851c21`, BFCL-live records
`4d70a040341480ccbae9af5a00d83bc4e7365ebead67e390cb5c3763ed83b30a`, and manifest
`f1815fa298e8b4e205b061a26ed8a1043e7aa688f7a4b40d034e7c329d14507a`.

Per-case wall time varied materially (18,121 ms to 336,401 ms among completed cases). That is
useful performance evidence but does not alter the frozen correctness gate. Every completed case
reported `shared=true`, demonstrating physical sharing mechanically without establishing the
required routing accuracy.

## Next experiment boundary

V22 is rejected unchanged. The exposed BFCL-live evidence may not be used to tune the policy and
then claim an independent pass. Any next attempt must introduce a separately versioned training
or policy hypothesis, freeze it before execution, and still pass a separate sealed qualification
screen along with product, provenance, memory, and performance gates.
