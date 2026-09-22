# Attribution

Harriet is a method and a release format over other people's weights. Everything it stands on is
named here, with what was taken and under which licence.

## Weights — not ours

| what | source | licence |
| --- | --- | --- |
| Qwen3.5-4B | `Qwen/Qwen3.5-4B`, GGUF Q4_K_M quantised by Unsloth | Apache-2.0 |
| Granite 4.1 3B | IBM | Apache-2.0 |
| Qwen3-0.6B | Qwen | Apache-2.0 |

The weights are **frozen**. Harriet trains nothing, fine-tunes nothing, and distils nothing.

## Method — substantially reproduced from SemIf

The prompting and readout in Harriet's letter arm are a reimplementation of **SemIf** (formerly
OpenJev) by **Theodore Lee (@TheoLeeCJ)**, MIT licensed, <https://github.com/TheoLeeCJ/openjev>.

Specifically taken, after reading `src/semif_phase1/core.py`, `src/semif_phase1/direct.py` and
`jevbench/adapters/semif_direct.py`:

* applying the model's own chat template with a system instruction and a JSON user payload
  (`evidence` / `criterion` / `options`) rather than feeding raw text;
* answer slots as **bare single-token uppercase letters**, refusing any slot that is not one exact
  round-trip token, rather than letters with a leading space;
* prefixing every option description with its own id;
* reading last-position logits restricted to those slots and softmaxing over them.

Measured effect of adopting it, our engine, same frozen Qwen3.5-4B Q4_K_M, standard tier n=72:
**0.8056 -> 0.9028**. A first attempt that reconstructed the prompt from the paper-level
description rather than the adapter scored 0.8472; the difference was three details listed above.
No SemIf code is vendored; the behaviour was reimplemented in Java.

## Benchmark

**JevBench** by Florian Standhartinger, MIT, <https://github.com/fstandhartinger/jevbench>.
Public datasets (`datasets/public/{easy,original,hard}.jsonl`, 231 decisions) and the scoring
functions in `jevbench/composite_v12.py` and `jevbench/metrics.py`, which our scorer imports
rather than reimplements.

## Not used

**Nothing in Harriet is derived from Jev or TypeSafe.** No Jev output was used for training,
distillation, fitting, prompt search, or model selection. Jev is a closed commercial service and
appears here only as a benchmark entrant measured by the benchmark's authors.

## Name

**Harriet Ann Jevons** (née Taylor, m. 1867) selected and edited *Letters & Journal of W. Stanley
Jevons* (Macmillan, 1886) after his death. Her husband built the Logic Piano; she is the reason
his work was published. An open-weights release is named for the publisher, not the inventor.

Independent project. Not affiliated with or endorsed by TypeSafe, Qwen, IBM, or SemIf. Jev,
TypeSafe and other marks belong to their owners.
