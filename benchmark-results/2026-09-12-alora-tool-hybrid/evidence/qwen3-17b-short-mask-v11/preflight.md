# Qwen3 1.7B short function-masked rank-32 aLoRA preflight

Status: **preflight was frozen before training; V11 was subsequently rejected by its exposed
smoke gate**.

V11 corrects the input-only retention failure discovered at V10 startup. Callable names are short
opaque ordinals such as `f0`; parameters use `p0`. Their assignment is shuffled by a deterministic
content hash within each record, so ordinals do not encode source names or declaration order. The
same original name always maps consistently across duplicate declarations, schemas, and expected
calls in that record.

## Frozen model and training contract

- Base: `Qwen/Qwen3-1.7B`
- Base revision: `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`
- Invocation tokens: `[151644, 77091, 198]` for `<|im_start|>assistant\n`
- Target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, and
  `down_proj`
- Rank / alpha / dropout: 32 / 64 / 0
- Epochs: 1
- Learning rate: `1e-4`, cosine schedule, 3% warm-up
- Batch size: 1; gradient accumulation: 16
- Maximum sequence length: 1,024
- Seed: `20260917`
- Precision: BF16 with SDPA; TF32 disabled
- Gradient checkpointing: enabled
- Trainer SHA-256:
  `bcf5942e8689aa92c02998ce858b56580c728c1fce7e23451578b5c75e05ad8c`

These values are identical to V9 and V10. V11 starts from the pinned unmodified base and does not
consume any rejected or aborted adapter.

## Frozen and reproduced preparation

- Preparation seed: `20260912`
- Function-mask fraction: `0.6666666666666666`
- Masked training / validation rows: 7,955 / 644
- No-tool share: 19% in both splits
- Identifier assignment: short ordinals ranked by
  `SHA-256(seed:record-fingerprint:namespace:source-name)`
- Preparation script SHA-256:
  `05d4f219b45c951911b07713499371e2cfb8d4bb2b8135a16ea1fb21fc3e2968`
- Preparation manifest SHA-256:
  `281098c3b031758a2ff3b40d08b72def709698a3aa5c285657640e5db485c801`
- Training split SHA-256:
  `996b88d8b547e4377e469228f1632e05aa8eaa9de144fb13bc88b72835192980`
- Validation split SHA-256:
  `1d1b977295ff6e301337c15cb03c8c94bf85a6503e55b05e359f1b087229d43f`

Two independent preparations reproduced the splits and manifest byte-for-byte. Host-side hashes
match the local hashes. Tokenization with the pinned Qwen3 tokenizer proves that V11 retains the
exact V9 usable-row counts:

| Split | Usable | Over 1,024 | No-call | Multiple-call | Source-line hash |
| --- | ---: | ---: | ---: | ---: | --- |
| Train | 11,827 | 173 | 2,263 | 5,149 | `079dd38a3e6b35ef21fe6da769d70fab9e79cb8a9b868e1a7649c644f5f2cb6d` |
| Validation | 979 | 21 | 188 | 396 | `1c1c1fa35b750e342f33a1420c36cf8fb793184bdf80c132b68fd4e66f0a96e2` |

The counts exactly match V9. The training identity hash differs because the 21 structurally
ambiguous selected rows are replaced; the ordered validation identity hash is identical to V9.

## Unchanged gates

The exposed 25-simple / 25-multiple / 25-irrelevance smoke runs first and remains fixed at 100%
syntax, 100% schema validity, at least 85% exact calls, no worse exact calls than the unadapted base,
and at most 5% false calls. Failure rejects V11 immediately.

Only a complete smoke pass permits scoring the still-unscored V9 300-case window. Its manifest hash
is `5fee70a0d18801ac9541b2cfa3b504c0fd06e98f5bf1adeb196adfc89dffa1d2`, its case-set hash is
`94c951aa277cee1e3f5ba454323e28b81cbfa357eb3d25d23a2e6ca581d1f132`, and the query-bound
evaluator hash is `d60e5f96f0dd3c65f0dc56001dda803026d7950dac92493670643ff5b9415e14`.

The Java oracle, Spring AI, LangChain4j, tool-result synthesis, multi-turn behavior, disabled
adapter, exact physical KV-block identity, complete memory accounting, 4,096-token retention,
recompute equivalence, TTFT crossover, clean-host, packaging, and published-artifact gates are
unchanged and remain mandatory after both Python gates.

## Recorded outcome

Training completed without changing the frozen inputs. The fixed exposed set then produced 86%
exact calls, equal to the base and above the 85% floor, but only 97.33% valid syntax, 94.67% valid
schema, and a 20% irrelevance false-tool rate. Those results fail three immutable gates, so the
candidate is rejected. The sealed 300-case set was not opened, no JVM or packaging gate was used
to override the failure, and no V11 artifact is eligible for release.
