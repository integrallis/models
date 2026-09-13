# Qwen3 1.7B applicability-decision V14 preflight

Status: **frozen at 2026-09-13T21:14:22Z, before any V14 model output**.

V13 was rejected because its relative no-call count gate was impossible after
the control scored 77/81: only four errors remained, while the treatment was
required to correct nine. V14 does not reinterpret or advance V13. It defines a
new development screen around the quantity optimized by the treatment—the
class-balanced decision loss—while preserving all correctness, generation,
sealed-data, JVM, sharing, performance, and publication gates.

## Frozen ancestry and screen baseline

- Base: `Qwen/Qwen3-1.7B` revision
  `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`.
- V9 adapter SHA-256:
  `f1a4a1bc15e3942ed5a3fbe2c4750f39a8b24804e20566242ec740c5ce851c21`.
- V9 training manifest SHA-256:
  `d0620df4860c879f6d3f6e5573168bb08afc0b48d954040825e1317972f7de47`.
- Exact V8 hard-mixed prepared manifest SHA-256:
  `6abf29b6040a2831f52ad0b77a7276a2213e5c6bc775c4c68b40e3a3dc9b788c`.
- V13 corrected 32-step control manifest SHA-256:
  `6da2e2bddb7d635d51c0f17913e2549e86eb7a95d55e163bf6f5090bc891912f`.

The immutable V13 control is reused only as V14's 32-step development baseline.
This is valid because the V13 treatment completed zero optimizer steps and the
V14 treatment retains the exact base, initial adapter, corpus, row order,
optimizer, full-horizon scheduler, seed, environment, and 32-step stop point.
The sole causal difference remains the predeclared applicability coefficient.
The screen baseline scored 362/390 call decisions, 77/81 no-call decisions,
0.9394112060778728 balanced accuracy, 0.18326023501677952 applicability loss,
and 0.047281891635664736 ordinary language-model loss.

## Frozen treatment and artifact gates

The treatment is unchanged from the unrun V13 treatment:

```text
L = L_SFT + 0.1 * w_y * CE([z_call, z_none], y)
```

It starts from the exact V9 bytes and runs 32 optimizer steps through the stop
callback while retaining the 178-step one-epoch scheduler. Rank/alpha/dropout
remain 32/64/0; batch size is one with accumulation 16; maximum length is 1,024;
learning rate is `2e-5`; seed is `20260918`; BF16/SDPA and gradient checkpointing
remain enabled; TF32 remains disabled. No prompt, corpus, decoder, threshold,
policy, classifier, or runtime rule changes.

The screen passes only if every condition holds:

- applicability loss is at least 5% lower than the control;
- correct no-call decisions do not decrease from 77/81;
- no more than seven correct call decisions are lost from 362/390;
- balanced applicability accuracy exceeds 0.9394112060778728;
- ordinary language-model loss is no more than 10% above control;
- exactly 32 optimizer steps and 512 finite-gradient checks complete;
- initial active logits match the control exactly;
- base and disabled-adapter logits match before and after training;
- saving does not change live logits; and
- every adapter tensor and sampled logit is bit-identical after reload.

Failure rejects V14. A pass requires fresh V14 control and treatment runs from
the V9 bytes to the fixed 178-step endpoint, with the same directional and
artifact gates. The endpoint cannot reuse the V13 screen control.

Only a passing endpoint treatment may run the already exposed 25 simple, 25
multiple, and 25 irrelevance generation screen: 75/75 syntax, 75/75 schema, at
least 43/50 exact positive calls, no worse than a freshly rerun same-prompt base,
and at most 1/25 false calls. The sealed 300-case set remains closed until that
complete pass. The existing Java/oracle, Spring AI, LangChain4j, long-context,
physical shared-KV identity, crossover, NMT, memory, clean-host, packaging, and
published-artifact gates remain mandatory and unchanged.

## Frozen implementation

- Trainer SHA-256:
  `5105cb8185da270dacab449041cde8d3cea51c623bd165a6e333c92b294e0011`
- Trainer test SHA-256:
  `4355dc68d394562a73e52e469f5f29f0d953cf4b0581c772cc9bc6cf0da60a58`
- V14 gate SHA-256:
  `9627c1a7c76051a984db3246d5a7766f03d721ec4c7a03e70a8a86e0b6ca4d29`
- V14 gate test SHA-256:
  `2b923728041972a8285f3837def80b995ac99f0d0471e1037c037e9550eb0df6`

At freeze time the focused 21 tests passed. The treatment may use the already
provisioned, preflighted A16 host. Its identity-bound deletion watchdog is fixed
at 2026-09-14T00:05:00Z; no generation or JVM performance work is permitted on
that training host.
