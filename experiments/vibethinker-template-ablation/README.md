# Does VibeThinker 1.5B fail `math` because of the output cap, or because of the prompt template?

Registered before data collection. Nothing in this directory was written after seeing a result.

## What is already measured

`benchmark-results/2026-10-21-candidate-qualification/raw/shard-703/` ran this exact artifact on
`math` at `maxOutputTokens = 768` with `chatml-direct`:

| field | value |
| --- | --- |
| `verdict` | `FAILED_MODEL_CONTRIBUTION_GATE` |
| `modelAnswerRate` | 0.000 (floor 0.333) |
| `truncatedAnswerRate` | **0.889** |
| `extractiveFallbackRate` | 0.889 |
| `correctAnswerRate` | 1.000 (the *pipeline*, on extractive fallback) |
| `totalOutputTokens` | 20175 over 27 generations (747 mean against a 768 cap) |

A later run, shard 910, used `mt: 64`. That value is documented as known-bad at
`models-rag-bench/src/main/java/com/integrallis/models/rag/RagBenchmarkCli.java:164` -- "a thinking
model spent the whole budget on its reasoning trace" -- so shard 910 measures the cap and not the
model, and is superseded by this experiment rather than reported.

## What the raw text shows

From shard 703, `runs[0].rawText` (3086 characters, cut off at the cap):

- no `<think>` tags and no `\boxed{}`: the harness system prompt does override this model's own
  chat-template defaults, which the GGUF header shows would otherwise request step-by-step
  reasoning with the answer in `\boxed{}`
- the model finds the right facts and the right source id, and recites the output rules
- it then deliberates untagged -- "My answer: ...", "But to be precise..." -- rehearsing candidate
  answers without committing, until the cap cuts it off
- it opens mid-sentence (`" for the Amber matrix..."`), continuing `chatml-direct`'s prefill
  `"The context states that "` (`ChatTemplate.java:244`)

## Why the cap alone is not expected to fix it

`GroundedAnswerPolicy.apply` strips a reasoning trace only when it is closed:
`REASONING_TRACE` requires `<think>`/`<thinking>`/`<reasoning>` tags
(`models-rag/src/main/java/com/integrallis/models/rag/GroundedAnswerPolicy.java:47-48`, used at
`:257`). This model's deliberation carries no tags, so no cap makes it strippable; the whole blob
becomes the validation candidate and fails `hasOnlySupportedClaims`, giving
`EXTRACTIVE_FALLBACK`.

So the cap is a necessary fix to a defect, and the template is the candidate treatment.

## Hypothesis

The prefill decides whether this model commits or narrates. Three templates are available on this
path and all three are tested rather than one being picked by guess
(`RagPromptTemplate.java:72-89`):

- `chatml-direct` prefills `"The context states that "` -- a sentence-continuation, which is what
  shard 703 ran and what the raw text above is drifting out of
- `chatml-answer` prefills `"Answer: "` -- a commit-to-an-answer form
- `chatml-no-think` prefills an empty closed `<think>\n\n</think>\n\n` block -- the Qwen3
  convention for "thinking is finished, answer now"

The `no-think` arm is only meaningful if those are real tokens for this model, so that was checked
before including it: in this GGUF's vocabulary `<think>` is token 151665 and `</think>` is 151666
(read from the range-fetched header, 151936 tokens). They are single tokens, so the prefill is a
control signal and not literal text the model has to interpret.

Each arm was confirmed to be observable on the production path before running, because a toggle
that silently does nothing yields "this stage does not matter" when the truth is "this stage never
ran". The bench renders through `RagPromptRenderer` -> `applyGroundedPrompt`
(`RagPromptTemplate.java:309`), which for every non-Granite template delegates to
`applyPrompt(instructions, request)`; that path emits the three prefills distinctly at
`RagPromptTemplate.java:185-193`. `chatml-no-think` emits its block through `control()` rather
than `text()`, so the think markers are tokenized as trusted template tokens -- which is why their
presence in this model's vocabulary was worth checking.

That same path renders the harness's own system turn (`<|im_start|>system`), which is why this
model's chat-template default -- step-by-step reasoning with the answer in `\boxed{}` -- does not
appear in the shard 703 output.

Citations are not required of the model: a validated answer with no citation is accepted as
`MODEL_ANSWER_WITH_DERIVED_CITATIONS` (`GroundedAnswerPolicy.java:276-280`), and that
decision counts as a model answer (`GroundingDecision.java:21`).

## Design

Two variables, fully crossed: **cap x template**, 6 cells, everything else held. A one-cap design
cannot separate the two explanations -- if no template commits at a single cap, "the template does
not help" and "the cap is still binding" are indistinguishable. Crossing them makes both
observable in one protocol.

| | `chatml-direct` (control) | `chatml-answer` | `chatml-no-think` |
| --- | --- | --- | --- |
| **cap 768** | `..._tpl_direct_mt768` | `..._tpl_answer_mt768` | `..._tpl_nothink_mt768` |
| **cap 1792** | `..._tpl_direct_mt1792` | `..._tpl_answer_mt1792` | `..._tpl_nothink_mt1792` |

All ids are prefixed `vibethinker_1_5b_q4_k_m`. 768 is shard 703's cap, so the control cell at 768
is the closest thing to a like-for-like repeat of what is already measured; 1792 is the highest cap the fleet's pinned
2048 context can actually run (see the amendment below).

Cell ids are deliberately not the catalog id. This selects a configuration; a verdict eligible for
the catalog is produced afterwards under the canonical id `vibethinker_1_5b_q4_k_m` using whichever
configuration wins here.

`warmups = 0`, `iterations = 1`. Generation is greedy and pinned -- `temperature 0`, `topK 1`,
`seed 42` -- so repeated iterations of a case re-generate the same text, and the quantities this
experiment reads are rates over cases, which iteration count therefore does not move. Iterations
buy timing stability, and no timing is reported here. **What this cannot test:** run-to-run text
stability. It is not needed for a choice between templates, and any cell that lands marginal
against the floor is re-run rather than read as-is.

## Where this runs, and what it is not

The ablation runs on a Hetzner `ccx33` (dedicated x86_64 vCPU) via
`run_on_isolated_host.sh`, which sources the shared `isolated_host.sh` so the host is firewalled to
this workstation and deleted on every exit path.

It does not run on this workstation: the native kernel ships per platform and there is no
`macos-x86_64` build, so `--backend rust-ffm` cannot load here -- it fails in
`BundledNativeKernelLibrary.resolve` with "no Models native-kernel artifact for macos-x86_64".
Falling back to `pure-java` would measure a route the fleet never runs. The Linux host takes the
same payload tar (`models-rag-bench-0.5.7-vibethinker1.tar`) and the same kernels jar
(`models-kernels-linux-x86_64-0.5.7.jar`) the qualification fleet uses, and the script checks the
downloaded GGUF's sha256 against the artifact shard 703 measured before running anything.

**This is not a qualification.** One arm, no comparator, and a different machine than the fleet.
It selects a prompt template. Every field the decision rule below reads comes from the candidate
report alone, which is why one arm is enough to choose between templates. The verdict that can
reach the catalog is produced afterwards by `scripts/fleet/run-shards.sh` on shard 911 under the
two-arm protocol, with the selected template and the canonical model id.

## Decision rule — registered before the data

Read `truncatedAnswerRate` first, per the 2026-10-09 sweep's own ordering: a cell still truncating
is inconclusive regardless of what its verdict did. The gate is `modelAnswerRate >= 0.333` **and**
`modelAnswerCorrectRate >= 0.90` (`RagProductionQualificationPolicy`).

| what the grid shows | conclusion |
| --- | --- |
| a treatment clears the gate, control does not, at the same cap | the template was the cause; select that template |
| control clears the gate too, at the higher cap | the cap was the cause; template is free |
| truncation ~0 everywhere, no cell clears the gate | the model does not contribute on this corpus; publish the null |
| truncation still high at 1792 in every cell | the cap is still binding at the highest cap this harness can run; inconclusive |

If more than one cell clears the gate, the winner is the lowest cap that clears it, and within that
cap the higher `modelAnswerCorrectRate`; on a tie, `chatml-answer`, as the simpler prefill. Lowest
cap first because the cap is a cost paid on every request in production, not a free parameter.

## Amendment: the 2048 cap was not runnable, and 1792 replaces it

The first grid's three `cap 2048` cells failed mechanically, not on the measured quantity. Every
failure was `IllegalArgumentException: position out of range: 2048`: generation ran past the KV
window, because the cap equalled the context. `--context 2048` with `--max-tokens 2048` leaves no
room for the prompt.

Measured prompt lengths in the valid cells are 156-172 tokens across all nine cases, so against a
2048 context the usable ceiling is **1876**. The replacement cap is **1792**, which keeps 84 tokens
of headroom and is 2.3x the 768 arm.

This matters past this experiment: `scripts/fleet/qual-worker-two-arm.sh:427` pins
`--context 2048` for both arms, so **no fleet shard can run a cap at or near 2048**. The shard 911
manifest originally carried `mt: 2048` and would have failed on a fleet box exactly this way.

The cells are excluded rather than reported, and `derive.py` now refuses any cell whose
`successfulAttempts` is below `totalAttempts`: rates computed over the surviving attempts are
biased precisely when the failures correlate with what is being measured, which is the case here --
the cases that overran the window are the long-generating ones, so the survivors are the short
ones and every rate reads better than the truth. Replacing an unrunnable cap with a runnable one
repairs the instrument; the decision rule above is unchanged.

## Kill criterion

If every cell shows `modelAnswerRate` below 0.333 with `truncatedAnswerRate` at ~0,
VibeThinker 1.5B is not qualified on `math` and that null is the result. No fourth template is
tried, no stop sequence is added, and the cap is not raised beyond 1792 to chase the gate -- a
per-model stop sequence or a cap chosen after seeing the score would be tuning the harness to the
answer, which is what the campaign runbook forbids.
