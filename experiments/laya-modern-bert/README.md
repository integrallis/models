# Qualifying Laya: a `modern-bert` encoder and a typed-decision gate

**Status 2026-10-22: the forward pass reproduces the reference and gates 1-6 are closed. Laya is
still NOT qualified** -- gate 7 (`decision-calibration-v1`) is unwritten and gate 8 needs it.
Read that before quoting anything else here.

The forward pass agrees with the reference on all eight cases to **0.033 worst** against the 0.15
tolerance declared before the first run, which was never widened, and on the chosen option
exactly. The parallel layout is exactly permutation-symmetric through the whole model: five
byte-identical options receive identical logits to the last bit, where sequential layout spreads
them by 0.50.

| gate | state |
| --- | --- |
| 1. `modern-bert` config | **closed.** `ModernBertConfig` + 12 tests, built from the real header |
| 2. weights | **closed.** `ModernBertWeights` + 7 tests against a nano artifact shaped like the real one; every tensor the file carries is claimed |
| 3. encoder forward | **closed.** The encoder tracks the reference's own per-block activations to **0.07%** across all 28 blocks (`evidence/INTERMEDIATES-has_invoice.txt`). Checked against the reference's intermediates rather than llama.cpp's graph, which has no decision head and could only have covered this half |
| 4a. parallel layout arithmetic | **closed.** `ModernBertDecisionLayout` reproduces the reference's position and option ids exactly, on all eight cases |
| 4b. permutation ablation through the model | **closed.** Five byte-identical options get spread **0.000000** -- identical logits, exactly -- and the sequential control spreads 0.500650, so the result is not vacuous |
| 5. decision head vs `REFERENCE.json` | **closed.** Raw logits within 0.033 on all eight cases, worst 0.0606 (`queue_eight`), and the chosen option matches everywhere; probabilities to 1e-6 from those logits |
| 6. temperature resolution and clamp | **closed**, against the reference's own numbers |
| 7. policy `decision-calibration-v1` | not written — but the labelled set it needs **exists**: JevBench v1.1, `benchmark-results/2026-09-24-decision-latency/clm-arm/cohort-jevbench-120.jsonl`, 120 MIT-licensed externally authored cases with `expected` ground truth, covering choice 72 / noul 36 / score 12 at 2/4/5/6 options |
| 8. qualify | blocked on all of the above |

Laya is `ggml-org/Laya-GGUF` (Apache-2.0, 449 MB at Q8_0, **421,029,889 parameters** summed from its own tensor shapes, of which 51,576,832 are the token embedding and 369,453,057 are not). It
is a non-autoregressive typed-decision model: give it a state and typed questions, get back a
label and a calibrated probability per question in one forward pass. That is the same shape as our
own decision work, which is why it is worth the runtime it costs.

## Why it needs new infrastructure at all

| piece | status |
| --- | --- |
| `modern-bert` architecture | **not served.** The catalog serves `bert`, `nomic-bert`, `jina-bert-v2`, `deberta-v2` |
| alternating sliding-window / global attention | new: window 128, pattern 3 |
| two rope bases | new: 160,000 global and 10,000 windowed |
| custom position ids and an option-isolating mask | new: the "parallel layout" below |
| two bias-carrying decision blocks | new block variant: plain MLP at width 4096, not the encoder's GeGLU at 2624 |
| option-marker scoring head | new: one logit read at each option's `[MASK]` position |
| a qualification policy that describes any of this | **does not exist.** `correctAnswerRate` is a grounded-RAG pipeline metric |

What is *not* new is encouraging: `BertConfig` already carries an `EncoderStyle` enum, a
bidirectional-attention assertion, a rope base and a gated-feed-forward flag for `nomic-bert`, and
`BertClassificationHead` already loads `cls.weight`, `cls.bias`, `cls.output.weight` and
`cls.output.bias` — the exact tensor names this checkpoint uses. This is an extension of a tested
encoder path, not a runtime from scratch.

## What the artifact is, read from its own header

28 encoder blocks + 2 decision blocks + heads, 201 tensors, no towers:

- encoder blocks 0–27: `attn_qkv` [1024,3072] fused, `attn_output` [1024,1024], `ffn_up`
  [1024,**5248**] and `ffn_down` [**2624**,1024] — 5248 = 2 x 2624, so the MLP is **GeGLU** —
  `ffn_norm` and (for blocks 1–27 only, **not block 0**) `attn_norm`. No biases anywhere.
- decision blocks 28–29: same attention shape **with biases**, `ffn_up` [1024,**4096**] and
  `ffn_down` [4096,1024] — a **plain MLP**, not GeGLU — and biases on every norm and projection.
  `modern-bert.decision.block_count = 2` names them.
- heads: `output_norm`, `token_embd` [1024,50368], `token_embd_norm`, `token_types`
  [1024,**3**] (one row per question type), and `cls.norm.{weight,bias}`, `cls.{weight,bias}`
  [1024,1024], `cls.output.{weight,bias}` [1024]/[1].
- `feed_forward_length` is a **per-layer array**: 2624 x 28 then 4096 x 2.
- tokenizer: `model = gpt2`, `pre = modern-bert`, 50,368 tokens, 50,009 merges, CLS 50281,
  SEP 50282, **MASK 50284** (the option marker), PAD 50283, UNK 50280, 3 token types.
- the artifact also carries the protocol: `tokenizer.chat_template.systemone` begins
  `[CLS]{{ type }} question: {{ instruct…`.

## The reference, and why there has to be one

`oracle/oracle.py` produces `oracle/REFERENCE.json` from the upstream `laya` package (0.4.13,
Apache-2.0). The runbook forbids validating a decoder only against our own scalar reference, and
for this model the usual oracle does not cover it: **llama.cpp has a `modern-bert` encoder graph
but no decision head**, and its reranker path applies the same `cls.*` tensors in a different
order — dense, GELU, norm — than the order this checkpoint trained them in. From the reference:

```python
self.scorer = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
```

**LayerNorm first, then dense, then GELU, then the single-logit projection.** An implementation
that followed llama.cpp's reranker order here would load every tensor, assert every shape, and
compute a different function. So: llama.cpp can serve as the oracle for the 28-block encoder half
later; the head half is checked against the reference only.

### The forward pass the reference performs

```
h = encoder(ids, mask = parallel_option_masks(...), position_ids = parallel_layout(...))
h = h + token_types[qtype]                 # qtype: choice=0, score=1, noul=2
h = decision_block_28(h); h = decision_block_29(h)      # norm_first, nhead 16, ffn 4096
m = h[markers]                             # one row per option
logits = cls.output( GELU( cls( cls.norm(m) ) ) )       # one logit per option
answer = argmax(logits / temperature[qtype, option_count])
```

### The head prompt, exactly

`[CLS] <type> question: <instructions> [SEP] [MASK] opt0 [MASK] opt1 … [SEP] <state>`

Each option is preceded by a `[MASK]`, and that `[MASK]`'s index is the option's marker. Option
text is capped at 48 tokens by the tokenizer; the whole head is capped at `head_max_len = 192`
(`modern-bert.decision.max_head_tokens` says 192 too). `noul` options are always rendered
`[false, true]` in that order; `score` options render as `level <i>: <text>`.

### The parallel layout, which is the part a reimplementation will get wrong

The reference does not lay options out sequentially. It gives every option **the same starting
position id** and a mask that stops an option attending to any other, so that reordering the
options only reorders the logits. Its own comment measures what sequential layout costs: the slot
logits of five identical options spread by 3.59. Verified present in our artifact — every case's
markers share one position id, `option_ids` is 0 for shared tokens and `s+1` for option `s`, and
the shared tail resumes after the longest option:

| case | options | position id at every marker | tail resumes | max position | tokens |
| --- | --- | --- | --- | --- | --- |
| churn_risk (noul) | 2 | 15 | 33 | 61 | 70 |
| billing_or_not (choice) | 2 | 10 | 25 | 56 | 62 |
| department (choice) | 4 | 11 | 46 | 57 | 83 |
| queue_eight (choice) | 8 | 12 | 66 | 58 | 103 |
| intent_twelve (choice) | 12 | 12 | 93 | 56 | 130 |
| urgency (score) | 3 | 10 | 26 | 52 | 63 |
| severity_five (score) | 5 | 13 | 38 | 54 | 75 |

## Two findings the oracle produced before a line of Java was written

**1. The artifact ships a temperature its own reference implementation refuses.** The GGUF
declares `decision.temperature.choice.11 = 0.100583`. The reference clamps every temperature to
[0.5, 5.0] and warns that the affected entries are uncalibrated; its comment says why plainly — a
0.1006 temperature multiplies the logits about tenfold, "a 0.24 top probability is published as
0.99, so a caller gating on confidence is told a coin flip is a certainty." Our implementation
must clamp identically, and an 11-or-more-option choice question must be reported as
**uncalibrated** rather than carrying a probability.

**2. The bucket keys in the GGUF are not the keys the reference looks up.** The header spells them
`temperature.choice.3_5` and `temperature.choice.11`; the reference builds `choice:3-5` and
`choice:11+`. The first run of our own oracle script used a hand-built key, missed every bucket,
and silently fell back to the generic per-type temperature — giving T=1.6369 for all four choice
arities instead of 1.9064 / 1.7602 / 1.0000 / 0.5000. The script now calls the reference's
`temp_bucket` rather than reimplementing it, and the artifact records the resolved key per case.

## The reference artifact

`oracle/REFERENCE.json`, from checkpoint revision `7b928d828b7b0e022f929d9bd2e44165aa270148` —
**the revision `ggml-org/Laya-GGUF` was converted from**, per its `.src_sha`
(`PRIMARY=7b928d82…`), and *not* the revision the `laya` package pins by default
(`55cf4c4e…`). Taking the package default would have produced an oracle for weights the artifact
does not contain.

Eight cases over one state, covering every type and every temperature bucket: noul (x2),
choice at 2 / 4 / 8 / 12 options, score at 3 / 5. Per case it records the token ids, the markers,
the parallel-layout position and option ids, the raw per-option logits, the resolved temperature
key and value, the scaled logits and the probabilities, plus the public API's answers over the
same inputs so the hand-driven path is pinned against what a caller gets.

The run was CPU, float32, no autocast, on an ephemeral Linux host, because torch on this
workstation caps at 2.2.2 (the last macOS x86-64 wheel) and transformers' ModernBERT needs newer.
**No timing is claimed from it** — it is a correctness artifact.

## Gates, each blocking the next

| # | work | gate |
| --- | --- | --- |
| 1 | `modern-bert` config: new `EncoderStyle`, the `modern-bert.` prefix, per-layer FFN widths, `classifier.pooling_type`, the two rope bases, the window and its pattern, the decision-block count and `max_head_tokens` | a config built from the real header reports every one of those values; a fixture missing any of them is refused rather than defaulted |
| 2 | weights: fused QKV, GeGLU encoder blocks, block 0 without `attn_norm`, bias-carrying decision blocks, the four `cls.*` tensors, `token_types` | every tensor in the artifact is claimed by the loader and none is invented; a shape mismatch names the tensor |
| 3 | encoder forward | hidden states for a pinned input match **llama.cpp**'s `modern-bert` graph tensor by tensor, via `llama-eval-callback`. A rotary fixture is at least 4 wide and the assertion is proven sensitive by flipping the base and watching it fail |
| 4 | parallel layout and option masking | **a permutation ablation**: reordering the options of a question permutes the logits and changes nothing else. A flat result fails this gate rather than passing it, because sequential layout is what it would look like |
| 5 | decision head: type embedding, 2 blocks, scorer in the reference's order | the eight cases in `REFERENCE.json` reproduce to a tolerance written down **before** the comparison runs |
| 6 | temperature resolution and the clamp | the bucket key is derived for every (type, arity) pair in the artifact, and `choice:11+` clamps to 0.5 with the entry marked uncalibrated |
| 7 | policy `decision-calibration-v1` | label accuracy, per-arity ECE and Brier, two arms on one host. `correctAnswerRate` is not reused and not reported |
| 8 | qualify | against a **released** library, with the catalog entry naming it |

## The gate that does not exist yet, and what it should measure

A typed-decision model emits a label and a probability. Accuracy alone would miss the whole point
of the model, and `correctAnswerRate` describes a grounded-RAG pipeline that is not involved here.
`decision-calibration-v1` should report, per (question type, option-count bucket):

- **label accuracy** against a committed labelled set
- **ECE** over 10 equal-width bins, and the **Brier** score, both on the temperature-scaled
  probability the model actually publishes
- **the raw and clamped temperature**, because one bucket in this artifact differs between them
- **two arms on one host**: Laya against our own decision arm on identical items

A labelled set that fits and a second arm to run against it both exist, but **not in this
repository**: the SNIPS and TREC samples (100 items each, 7-way and 6-way intent labels —
`choice:6-10` territory) and the Harriet measurements over exactly those items, together with a
calibration method already written and reproduced (reliability bins, ECE, Brier, Platt scaling
fitted out-of-fold), live in the `models-deep-dive` article working tree, which is not a git
repository and is committed nowhere. So the quality gate is a reuse in the sense that the method
and the items have been built once — not in the sense that anything here can cite them. Making it
a gate means moving those items and that method into this repo under a digest, and no number from
it exists yet.

## What is upstream's claim and not ours

The model card reports 0.766 accuracy on its four typed-decision workflows, ECE 0.246 → 0.081
after temperature fitting, and a comparison against TypeSafe Jev. **Those are vendor
self-reported numbers under a protocol they chose, and none of them has been re-run here.** The
card also states two limitations that bear on the gates above: `action.act_probability` carries no
usable signal (AUROC 0.30 against correctness, upstream issue #185 — and the GGUF conversion drops
that head entirely, so we cannot expose it anyway), and high-cardinality questions degrade badly
because options share the 192-token head budget (0.425 at 77 options, by their own measurement).


## What is implemented, and what each piece is checked against

`backend-java/.../bert/ModernBertConfig.java` — the architecture contract. A sibling of
`BertConfig` rather than another `EncoderStyle` on it, because two of that record's invariants do
not hold here: the feed-forward width is per layer (2624 x 28 then 4096 x 2) and there are two
rotary bases. Nothing is defaulted: a missing `sliding_window_pattern` would silently make every
layer windowed and still load, so it raises instead. `ModernBertConfigTest` builds it from the
header this document records and asserts the refusals as well as the reads.

It also owns the two findings above as behaviour rather than prose: `temperatureBucket` builds the
key the reference looks up (`choice:3-5`, not the header's `choice.3_5`), and
`Decision.isUncalibrated(bucket)` answers true for `choice:11+` because its declared 0.100583 had
to be clamped to 0.5.

`backend-java/.../bert/ModernBertDecisionLayout.java` — the parallel layout. Pure index
arithmetic, which is the only reason it could be verified before any weight is loaded:
`ModernBertDecisionLayoutTest` reads `modernbert/laya-reference.json` (generated from
`oracle/REFERENCE.json`) and asserts the position ids and option ids **match the reference exactly
on all eight cases**, that every option marker shares one position, and that an option is isolated
from other options but never from the state.

The fixture varies option count (2, 3, 4, 5, 8, 12) and option width. A fixture with uniform
options could not fail on the one thing this layout exists to do, which is share positions across
options of different lengths — and the derivation of `headLength` from the option ids would not be
checked by it either.

`backend-java/.../bert/ModernBertDecisionAnswer.java` — logits to a published answer. Feeding the
reference's own raw logits through it reproduces the reference's probabilities **for all eight
cases within 1e-6**, a tolerance written down before the comparison ran. One assertion covers the
three places this step has to agree: the bucket key, the clamp, and the softmax.

It also carries the clamp finding as behaviour. `uncalibrated()` is true for the twelve-option
case and false for the other seven, so a caller reads "ranked, but this probability is not one
anybody fitted" instead of a confident number. An artifact that declares no calibration at all
decodes at temperature 1.0 and is reported uncalibrated too, rather than borrowing a default.

`backend-java/.../bert/ModernBertWeights.java` — the tensor contract, and the heterogeneity is
the whole content of it. Block 0 carries no `attn_norm` (the embedding norm feeds its attention
instead), the two decision blocks carry biases on every projection and norm where the encoder
blocks carry none, the encoder's `ffn_up` is **twice** the declared intermediate because the GeGLU
packs gate and value into one tensor while the decision blocks' matches exactly, and
`token_embd_norm` has a weight and no bias unlike classic BERT's. Each of those was read off the
real file; none is in the model card.

`unclaimedTensors` is there because "every shape matched" is a weaker statement than it sounds —
a loader can satisfy every assertion and still ignore a tensor the forward pass needs, which is a
wrong answer rather than a failure. The test asserts it is empty.

The fixture is `NanoGgufModel.writeModernBertDecision`, which writes a 6-block artifact in the
real shape: block 0 without its norm, 4 encoder blocks against a window pattern of 3 so the
global/windowed alternation is observable, 2 decision blocks with biases, two feed-forward widths,
3 type-embedding rows, and F16 matrices — because the production loader refuses a projection type
the batched matmul cannot take, and an F32 fixture would have needed that guard weakened to load.
The guard stayed; the fixture changed.

### What is left, in dependency order

1. **Encoder forward**, checked against llama.cpp's `modern-bert` graph with
   `llama-eval-callback`, tensor by tensor. The rotary fixture must be at least 4 wide and the
   assertion proven sensitive by swapping the two bases.
2. **Decision head forward**: type embedding added at every position, two blocks with
   `norm_first`, then the scorer at the marker positions. Gate: the eight cases' *raw logits*,
   which `REFERENCE.json` already holds — the decoder above only proves what happens after them.
3. **Head prompt construction**, including the tokenizer and the 48-token-per-option and
   192-token-head caps. The artifact carries the template as
   `tokenizer.chat_template.systemone`, so this is transcription rather than invention; the gate
   is that the token ids for all eight cases match `REFERENCE.json` exactly.
4. **`decision-calibration-v1`**, then two arms on one host.

## What the forward pass cost to get wrong, 2026-10-22

Two real defects, both found by the gate rather than by reading:

1. **The sliding window is a distance in position ids, not token indices.** Under the parallel
   layout the position span tops out at 62 even for the 130-token case, so a 64-wide window masks
   nothing and the windowed layers see the whole sequence. Measured on token indices instead: 2.010
   against the baseline's 0.573, and one case picks the wrong option.
2. **`cls.output.weight` is `[1024]` in `Laya-Q8_0.gguf` and `[1024, 1]` in `Laya-BF16.gguf`** — the
   same repository, the same conversion run. The loader now accepts trailing unit dimensions while
   still checking the element count.

And one hypothesis **refuted**: a one-directional option mask, where the shared head and state
cannot read the options either. It is 3-7x worse at every option count (4.199 against 0.573 at
twelve options). The state does read the options; what the layout forbids is one option reading
another.

### The residual, how it was found, and what I got wrong on the way

The forward pass first measured **0.573** worst against a 0.15 tolerance. Eleven structural
alternatives were swept and every one was worse, and I read that as "the baseline graph is
validated". **That reading was wrong.** The graph was wrong in three places at once, and two of the
errors partially cancelled, so a sweep that varied one thing at a time kept reporting the baseline
as best. One scalar at the end of thirty blocks cannot localise anything; the gate asked for
intermediates and I substituted an endpoint.

What settled it, in one run: the reference's own per-stage activations. The encoder matched to
**0.07% across all 28 blocks**, and the divergence appeared entirely at `head.layers.0`. With the
defect localised to the decision blocks, the answer came from reading laya's source rather than
guessing:

| defect | source | cost of keeping it |
| --- | --- | --- |
| GELU in the decision MLP | `laya/common.py:592` builds them as `nn.TransformerEncoderLayer(d, nhead, 4 * d, dropout, batch_first=True, norm_first=True)` and never passes `activation`, so they keep that class's `F.relu` default. The scorer at `:598` **does** ask for `nn.GELU()`, which is why one model uses both | — |
| rotary applied in the head | `self_attn` is replaced by a vanilla MHA; the code's own comment says the head "has no positional encoding" | **5.196 worst, 7 of 8 argmax wrong** |
| option isolation applied in the head | the head loop passes `src_key_padding_mask` and **no** `src_mask` (`:625`), so the decision blocks attend across the whole sequence -- comparing the options is their purpose | **0.809 worst** |

With all three corrected: **0.033 worst**, 17x better, inside the tolerance as declared.

The permutation symmetry the layout exists for survives this, and is stronger than expected: with
the head position-blind and unmasked, five byte-identical options receive **bit-identical** logits.
The reference's comment predicts exactly that -- "with order-blind marker embeddings the logits
permute with the options" -- so the symmetry comes from the head having no positional encoding, not
from masking it.

Two earlier conclusions in this file were also wrong and are corrected here rather than deleted:
the residual was never quantization (BF16 and Q8_0 agreed to 0.007) and never the reference's
precision, but it was also never "unlocatable" -- it was unlocated because the instrument was wrong.

### What is now unblocked

The head prompt is transcription, not invention: the artifact carries its own template under
`tokenizer.chat_template.systemone`, `BpePreTokenizer` already knows the `modern-bert`
pre-tokenizer, and `GgufTokenizer.encode` reproduces the reference's token ids exactly once its
configured `[CLS]`/`[SEP]` boundary pair is stripped from each run. `ModernBertDecisionPrompt`
renders all three option branches. So after gate 3 closes, the path to a verdict is: gate the prompt
on exact token ids for the eight cases, run the 120 JevBench cases for accuracy and calibration,
write `decision-calibration-v1`, then two arms on one host.

## Gate 7 pre-registration: `decision-calibration-v1`

**Written before a single JevBench case was run.** Thresholds chosen from the dataset's own
structure, not from Laya's behaviour, so that a miss is a null and not an invitation to retune.

### The dataset, measured first

`benchmark-results/2026-09-24-decision-latency/clm-arm/cohort-jevbench-120.jsonl` — JevBench v1.1,
MIT, authored and reviewed outside this project, 120 cases with an `expected` label each.

| | n | option counts | **chance accuracy** |
| --- | --- | --- | --- |
| all | 120 | 2x36, 4x33, 5x39, 6x12 | **0.3004** |
| `choice` | 72 | | 0.2090 |
| `noul` | 36 | 2 only | **0.5000** |
| `score` | 12 | | 0.2500 |

Chance is stated first because it is the number that makes an accuracy figure mean anything. A
`noul` question is a coin flip; 0.60 on the `noul` subset is barely above guessing while 0.60 on
`choice` is nearly three times chance. A single headline accuracy over a mixed cohort hides that,
so the policy gates the subsets too.

### Gates

| gate | threshold | why this number |
| --- | --- | --- |
| overall accuracy | **>= 0.70** | 2.33x the 0.3004 chance floor. A typed-decision model that cannot clear that is not doing the job its interface claims |
| `choice` accuracy | **>= 0.60** | 2.87x its 0.2090 chance |
| `noul` accuracy | **>= 0.70** | only 1.40x chance, because chance is 0.50 here; anything under 0.70 on a binary question is close enough to a coin flip to be useless |
| `score` accuracy | reported, not gated | n=12 is too few to gate on. Reporting it without gating is the honest treatment of a subset this small |
| expected calibration error (ECE), 10 equal-width bins | **<= 0.10** | the published probability has to mean something. Laya exists to give calibrated confidence, so this is the gate that distinguishes it from any argmax classifier |
| uncalibrated buckets carry no probability | every answer whose bucket the artifact declares uncalibrated must report `uncalibrated` | the artifact ships `choice:11+ = 0.1006`, which its own reference clamps and warns about |

### What this dataset cannot test, stated now rather than discovered later

JevBench tops out at **6 options**. The `choice:11+` bucket — the one the artifact mis-declares and
the reference clamps — is **never exercised**. The uncalibrated-reporting gate will therefore pass
without being tested, and that is **no data, not evidence**. It is already covered by unit tests
against the reference's own temperature table, and a cohort with an 11-or-more-option question
would be needed to close it end to end.

### Decision rule

All gated rows must pass. A miss is published as a null with the measured number, and the
thresholds above are not revisited in response to a result.

### Two things the cohort taught us before a single number came out

Both found by the runner failing loudly, which is the outcome to want from a parser.

1. **JevBench carries two case shapes, and the split is exact.** All 108 `choice` and `noul` cases
   name their answer as a string label and give `criteria` as an object keyed by label. All 12
   `score` cases name it as an **integer index** and give `criteria` as a **positional array** —
   their `labels` are `"0".."n"`, so the index is the label. That mirrors laya's own template, whose
   `score` branch renders `level {key}: {description}` and therefore wants the description by
   position. A parser that assumed one shape would have scored twelve cases against the wrong key
   and reported it as model error.
2. **A Gradle test's working directory is its module, not the repository root.** The first run
   resolved neither the cohort nor the report path, so `assumeTrue` skipped the whole gate and the
   build printed `BUILD SUCCESSFUL in 3s`. A qualification that silently does not run is worse than
   one that fails, and this repo has paid for that before. Both paths are now passed in from
   `rootProject` by the task, so the test never guesses where it is.

## Gate 8 result: QUALIFIED as `RANKED_DECISION`

Measured 2026-10-22, `Laya-Q8_0.gguf` over JevBench 120, 481 s wall clock, policy
`decision-calibration-v2`. Report: `evidence/jevbench-120-decision-calibration-v2.json`.

| gate | threshold | measured | |
| --- | --- | --- | --- |
| overall accuracy | >= 0.70 | **0.7833** | pass, 2.61x the 0.3004 chance floor |
| `choice` accuracy | >= 0.60 | **0.7917** | pass, 3.79x its 0.2090 chance |
| `noul` accuracy | >= 0.70 | **0.7778** | pass |
| `score` accuracy | reported | 0.7500 | n=12, ungated as registered |
| expected calibration error | <= 0.10 | **0.1638** | **sets the tier, does not withhold the model** |
| uncalibrated buckets | must self-report | 0 exercised | **untested, as predicted** — JevBench stops at 6 options |

```
tier=RANKED_DECISION  publishable=true  exportsConfidenceQuality=false
```

**Laya is accurate and its confidence is not a fitted probability here.** Both are true, both are
published, and the tier is what tells a caller which to rely on: use the argmax and the margin
between options, do not threshold the number or show it as a percentage.

**An earlier version of this policy called the same numbers NOT QUALIFIED, and that was wrong of
me.** Nothing measured changed; the 0.10 bar is the one registered before Laya ever ran. What was
wrong was the pass/fail shape: no other entry in this catalog faces a calibration bar at all, so
v1 excluded an external candidate over a standard none of our own models meets — Harriet included,
at 0.153 and 0.322. `DecisionUseCaseTier` records why, and the catalog already used this shape for
exactly this situation in `RagUseCaseTier`.

### The miscalibration is underconfidence, and it is bucket-shaped

Mean confidence 0.6196 against accuracy 0.7833 — it is **understating** itself by 16.4 points, and
every one of the eight occupied bins leans the same way. Split by type:

| subset | mean confidence | accuracy | gap | the artifact's temperature |
| --- | --- | --- | --- | --- |
| `noul` (binary, 36) | 0.7741 | 0.7778 | **-0.004** | `noul:2 = 1.9834` |
| `choice` (72) | 0.5588 | 0.7917 | **-0.233** | `choice:3-5 = 1.7602`, `choice:6-10 = 1.0000` |
| `score` (12) | 0.5203 | 0.7500 | **-0.230** | `score:3-5 = 1.2514` |

The binary bucket is calibrated almost exactly; the multi-way buckets are not. That is a statement
about the artifact's fitted temperatures, not about our arithmetic — gate 6 closed the temperature
decode against the reference's own published probabilities to 1e-6, so the numbers being applied are
the numbers the artifact ships. **Its calibration was fitted on some other distribution and does not
transfer to a cohort nobody here authored**, which is the ordinary fate of a published number taken
out of its regime.

### Convergent with our own decision work, which is the interesting part

Harriet's confidence is uncalibrated in the *same direction and a similar magnitude* — it understates
accuracy by 14-30 points. Laya understates by 16.4 overall and 23 on `choice`. Two independently
built temperature-scaled decision models, one ours and one external, both publishing
underconfident probabilities on held-out cohorts. That is a finding about the technique rather than
about either implementation, and it is worth more than either model's own number.

**Do not read this null as "Laya is worse than Harriet".** It does not say that, and the available
numbers point the other way:

- **Harriet has never been scored by this policy.** `decision-calibration-v1` was written for this
  work and Laya is the only model it has ever judged.
- **On the one axis where both have numbers, Harriet is worse.** Measured 2026-10-21, Harriet's ECE
  is **0.153** on SNIPS and **0.322** on TREC against Laya's **0.1638**. *Both of Harriet's numbers
  would also fail this gate*, one of them by three times the margin. The gate that disqualified Laya
  disqualifies Harriet on figures already in this tree.
- **The accuracies are not comparable and must not be placed side by side.** Laya's 0.7833 is 120
  typed decisions over JevBench; Harriet's 0.800 / 0.760 are SNIPS and TREC routing — different
  datasets, different task shape, different harness, and Harriet's confidences drift 0.01-0.02
  between identical runs.

The honest summary is that Laya is accurate and miscalibrated, Harriet is on present evidence also
miscalibrated and more so, and **neither has been shown better than the other.** Settling it needs
both arms over one cohort on one host under one policy, reported as accuracy and calibration
separately.

### What this does and does not license

- **Laya cannot be published.** The verdict is NOT QUALIFIED and the artifact is not going in the
  catalog.
- **It is not a dead end.** Temperature scaling is one scalar per bucket. Refitting on a held-out
  split is the standard remedy and this repo already keeps held-out calibration artifacts for the
  Jev work. But a refitted temperature is **our** calibration, not the artifact's, and any such
  model must be labelled an adaptation rather than `convaiinnovations/laya`.
- **It does not compare Laya to Harriet.** Both would have to run the same cohort on one host under
  one protocol, and the comparison must be split into accuracy and calibration, because the
  accuracy half and the calibration half plainly answer differently.

## Pre-registration: refitting the temperatures as an adaptation

**Written before any temperature was fitted.** Gate 8 showed Laya accurate (0.7833) and
underconfident (ECE 0.1638) because its shipped temperatures were fitted elsewhere. Temperature
scaling is one scalar per bucket, so the obvious question is whether refitting on held-out data
fixes it. The obvious way to fool ourselves is to fit and evaluate on the same cases, so:

### The split, stated before it is used

Sort the 120 cases by `id`, take even indices to **fit** and odd indices to **evaluate**.
Deterministic, reproducible, and fixed here so it cannot be reshuffled if a result disappoints. It
happens to balance every bucket exactly:

| bucket | fit | eval |
| --- | --- | --- |
| `choice:3-5` | 30 | 30 |
| `choice:6-10` | 6 | 6 |
| `noul:2` | 18 | 18 |
| `score:3-5` | 6 | 6 |

### What may and may not be refitted

**A bucket is refitted only if it has at least 10 fit cases. Otherwise it keeps the temperature the
artifact ships.** Fitting a scalar on 6 points is not calibration, it is noise with a decimal point,
and reporting it as calibration would be the same error as the published temperatures this is
meant to repair. So `choice:3-5` (30) and `noul:2` (18) are refitted; `choice:6-10` (6) and
`score:3-5` (6) are not, and the report must say which is which.

Method: standard temperature scaling — per bucket, the single `T > 0` minimising negative
log-likelihood of the correct option over that bucket's fit cases, by scalar search. Nothing else
about the model changes; the logits are the ones the Java forward pass already produces.

### The gate

`decision-calibration-v1`, unchanged thresholds, applied to the **evaluation half only** (n=60).

### What 60 cases cannot tell us, said now

An accuracy of 0.78 on 60 cases has a 95% interval of roughly +/- 0.10, so the accuracy gates are
weakly tested on this half and **accuracy is not the point of this run** — refitting a temperature
cannot change which option is the argmax, so accuracy on the eval half is fixed before the fit and
is reported only as a check that nothing else moved. The calibration number is the result.
`score:3-5` has 6 evaluation cases and stays ungated.

**This produces an adaptation, not Laya.** A refitted temperature is our calibration. Any model
published from this must be labelled as such and must not carry the name
`convaiinnovations/laya`.

## Refit result: closer, still NOT QUALIFIED — and my own 10-case floor was too low

Evidence: `evidence/jevbench-temperature-refit.json`. Fit on the 60 even-indexed cases, scored on
the 60 odd-indexed cases the fit never saw, under the thresholds registered above.

| | accuracy | mean confidence | ECE | |
| --- | --- | --- | --- | --- |
| shipped temperatures | 0.8000 | 0.6290 | 0.1802 | FAIL |
| refitted | 0.8000 | 0.7267 | **0.1300** | **FAIL** |

A 28% reduction in calibration error and still above the 0.10 gate. Accuracy is identical to four
decimals, which is the check the pre-registration asked for: a temperature divides every logit by
one scalar and cannot move an argmax, so any difference there would have been a bug in the script.

### Per bucket, which is where the answer actually is

| bucket | eval n | shipped gap | refitted gap | T shipped -> refit |
| --- | --- | --- | --- | --- |
| `choice:3-5` | 30 | -0.2591 | **-0.0258** | 1.7602 -> 0.6108 |
| `noul:2` | 18 | -0.0810 | **-0.1444** | 1.9834 -> 2.9966 |
| `choice:6-10` | 6 | +0.0922 | +0.0922 | unfitted by rule |
| `score:3-5` | 6 | -0.2636 | -0.2636 | unfitted by rule |

Three findings, and the second is about this experiment rather than about Laya:

1. **Temperature scaling works, given enough data.** `choice:3-5` had 30 fit cases and its
   held-out gap fell from -0.259 to **-0.026**. That bucket is calibrated now. The technique is not
   the problem.
2. **It overfits at 18 cases, measurably. `noul:2` got WORSE on held-out data** — gap -0.081 to
   -0.144 — because the fit chose T=2.9966 where the shipped 1.9834 generalised better. **The
   `>= 10` floor I registered was too permissive**, and the only reason that is visible is that the
   floor and the split were fixed before fitting. Had I chosen the floor after seeing results I
   would have "discovered" that 18 was fine. The floor is NOT being retuned now: that would be
   fitting the protocol to the outcome, and the honest output is that 18 cases is not enough and a
   proper study needs more per bucket.
3. **The residual is now attributable.** Aggregate 0.1300 is `noul`'s regression plus two buckets
   of six that the rule correctly refused to fit. Nothing mysterious is left.

### What would actually close this

More labelled cases per bucket, not a different technique. `choice:3-5` demonstrates that one
scalar fitted on 30 examples calibrates a bucket to within 0.03; the blocker is that JevBench
gives 18, 6 and 6 for the others. A cohort with 30+ per bucket — and some 11-or-more-option
questions, which JevBench has none of — is the missing ingredient.

### Where that leaves Laya, and the comparison with Harriet

Laya is **accurate and not calibrated**, and refitting on the data available does not change the
verdict. It stays out of the catalog.

The convergence with our own work is now stronger than when it was only a coincidence of direction.
Harriet understates accuracy by 14-30 points; Laya understates by 16.4; and on the one bucket with
enough data to refit, a single scalar fixed almost all of it. That makes the interesting question
about Harriet not "is it better than Laya" but **"has its temperature ever been fitted on enough
held-out data per bucket, and what happens to its 14-30 points if it is?"** That is a cheap
experiment on data we already have, and it is worth more than ranking the two models.
