# maple cannot be qualified: no authoritative engine will load it

**Not a performance result.** maple's candidate arm ran cleanly and well. The comparator arm never
started, because ollama refuses the artifact:

```
ollama create qual-maple_preview_tq1_0 -f Modelfile
  copying file sha256:54016e4d... 100%
  parsing GGUF
  Error: unsupported tensor "blk.0.attn_k.weight" size overflows
```

The runner aborted rather than recording an arm it could not produce (`### ollama create failed`),
which is the behaviour the campaign runbook requires: a shard that records one arm produces reports
that look like progress and qualify nothing.

## What the candidate arm did show

Measured 2026-11-01, pod `9lpalxaf8vzgpg`, 16 vCPU of an AMD EPYC 7713, `--warmups 1
--iterations 3`, workload `general`, template `chatml-no-think`, 27 of 27 attempts successful.

| | value |
|---|---:|
| p50 decode tok/s | 20.71 |
| p50 prefill tok/s | 14.17 |
| `modelAnswerRate` | **0.7778** |
| `modelAnswerCorrectRate` | **1.0000** |

7 of 9 cases answered by the model and all 7 correct — the same contribution as mellum, and well
above lfm2moe's 0.4444. On the evidence available this is a capable model; the TQ1_0 ternary decode
path works and produces correct grounded answers.

## Why there is no comparator

`AUTHORITATIVE_ENGINES` is `{ollama, transformers}`:

- **ollama** cannot parse the artifact, as above. Its GGUF reader rejects the tensor shapes that
  TQ1_0 ternary packing produces.
- **transformers** would need the publisher's original weights rather than this GGUF, which are not
  held here.
- **llama.cpp** serves it (our oracle ladder used it), but llama.cpp is not an authoritative engine:
  it is admitted at 0.45/2.0 and can only ever reach `FAILED_RELATIVE_GATE`. The
  `--publisher-reference` route that lets llama.cpp qualify a model is reviewed per artifact and
  exists for publishers shipping their own llama.cpp; deepgrove ships no fork, so it does not apply.

So the verdict for maple is `NO_COMPARABLE_BASELINE` **by construction**, and no amount of
optimisation changes that. It is an ecosystem gap, not a throughput shortfall, and it is the honest
reason this port cannot be qualified today.

## Disposition

Unqualifiable, so it cannot be catalogued. Recorded here rather than inferred from the earlier
withdrawn figures: the off-protocol 0.2086 that once appeared against this model was computed by
hand from an arm the policy had excluded, and is not evidence of anything.

If ollama gains TQ1_0 support, this becomes a one-run question and the candidate arm above is
already the protocol-matched half of it.
