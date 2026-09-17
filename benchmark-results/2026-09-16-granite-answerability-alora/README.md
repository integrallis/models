# Granite 4.1 3B answerability aLoRA — our own adapter (Track B)

Started 2026-09-16 after IBM's `granitelib-rag-r1.0` answerability aLoRA measured 0.600 balanced
accuracy on the MT-RAG suite of the frozen window (unanswerable recall 14/40) with the runtime
matching IBM's PEFT reference 8/8, i.e. the adapter's own behaviour on multi-turn RAG. The user's
direction: fix the adapter rather than reject.

- `prepare_answerability_training.py` — balanced training set from published **train** splits
  only (QuAC train for multi-turn with `CANNOTANSWER`, SQuAD v2 train, MS MARCO v2.1 train shard
  0), documents = grounding text plus distractors, validation held out by dialogue/article/query.
  `prepared-manifest.json` binds the sources and the two split files by SHA-256; the splits are
  regenerated deterministically (salt `20260916-answerability-alora`, seed 20260916).
- `train_answerability_alora.py` — IBM's adapter shape (rank 16, alpha 32, seven projections,
  invocation `<|start_of_role|>assistant<|end_of_role|>`), completion-only loss, greedy strict
  validation under the runner's contract; emits `training-manifest.json`.
- `evaluate_alora_window.py` — PEFT reference over one window suite (CPU), strict and lenient
  readings; used on the reference host for early determinations.

Protocol: adapter versions are selected on the held-out validation split (QuAC validation is the
multi-turn proxy) and scored on the frozen qualification window only once, at the end, under the
pre-registered rule in `../2026-09-15-granite-4.1-3b-alora-hybrid/preflight.md`.

## Pilot 1 (2026-09-16, Vultr A16 8 GB, nf4 base kept bf16)

Measured: 2,400 training records sampled, 1,941 used (459 over 2,048 tokens dropped, mostly long
QuAC dialogues), rank 16 / alpha 32, 145 optimizer steps (micro-batch 1 × accumulation 8), lr 2e-4
cosine, 107 minutes. Held-out validation (300 records, strict contract):

| step | balanced | answerable | unanswerable | SQuAD | QuAC | MS MARCO |
|-----:|---------:|-----------:|-------------:|------:|-----:|---------:|
| 60   | 0.684    | 122/139    | 79/161       | 0.763 | 0.575| 0.689    |
| 120  | 0.699    | 70/139     | 144/161      | 0.814 | 0.602| 0.744    |
| 145  | 0.742    | 94/139     | 130/161      | 0.825 | 0.699| 0.722    |

`pilot1/training-manifest.json`, `pilot1/training-log.jsonl`, and `pilot1/adapter.sha256` bind the
run; the adapter itself is on the reference host at `/opt/ref/pilot1-adapter` for the window read.
The validation split is not the qualification window; the window is scored once, on the
reference host, next.

### Pilot 1 on the frozen window (PEFT reference, bf16 base, CPU; window ea9e4a0c)

Measured once, after training finished; reports in `pilot1/window-*-peft.json`.

| suite                   | pilot 1 | answerable | unanswerable | IBM adapter (same path) | gate |
|-------------------------|--------:|-----------:|-------------:|------------------------:|-----:|
| squad-v2-dev            | **0.840** | 93/100   | 75/100       | 0.795                   | 0.8 ✓ |
| msmarco-v2.1-validation | 0.730   | 85/100     | 61/100       | 0.720                   | 0.8 ✗ |
| mtrag-human-rag         | 0.682   | 26/55      | 49/55        | 0.582                   | (out of scope, reported) |

Better than IBM's adapter on all three suites, above the gate on SQuAD, below it on MS MARCO.
The failure shape differs by suite: on MS MARCO the pilot still misses unanswerable queries
(61/100), on MT-RAG it now over-calls unanswerable (answerable 26/55). Both point at the same
training gap: the 2,048-token cut removed 459 of 2,400 sampled records, mostly long QuAC dialogues
and MS MARCO queries with many passages, which are exactly the long-context cases the window tests.
Pilot 2 (4,924 records, same cut) is running; the next lever after it is the length cut itself
(3k-token sequences, which need a 24 GB GPU).

## Pilot 2 (2026-09-17, same host and settings as pilot 1, lr 1.5e-4)

Measured: 6,000 records sampled, 4,924 used (1,076 over 2,048 tokens dropped), 364 optimizer steps,
238 minutes. Held-out validation (300 records, strict contract):

| step | balanced | answerable | unanswerable | SQuAD | QuAC | MS MARCO |
|-----:|---------:|-----------:|-------------:|------:|-----:|---------:|
| 150  | 0.699    | 127/139    | 78/161       | 0.794 | 0.540| 0.744    |
| 300  | 0.684    | 127/139    | 73/161       | 0.763 | 0.566| 0.689    |
| 364  | **0.775** | 105/139   | 128/161      | 0.835 | 0.735| 0.767    |

The balance between the two labels swings by step in both pilots (under-calling unanswerable at
the mid-run checks, recovering at the final low-learning-rate steps), so only the final step is
comparable across runs: pilot 2 0.775 against pilot 1 0.742, with the multi-turn QuAC slice
0.735 against 0.699. The window read on the reference host follows.

A 3,072-token smoke run on the same 8 GB card started after pilot 2 (records over the cut in
its 300-record sample: 6, against about 20 % at 2,048 tokens) to check whether longer sequences
fit before committing a full run to them.

### Pilot 2 on the frozen window (PEFT reference, bf16 base, CPU; window ea9e4a0c)

| suite                   | pilot 2 | answerable | unanswerable | pilot 1 | IBM adapter | gate |
|-------------------------|--------:|-----------:|-------------:|--------:|------------:|-----:|
| squad-v2-dev            | **0.860** | 92/100   | 80/100       | 0.840   | 0.795       | 0.8 ✓ |
| msmarco-v2.1-validation | 0.750   | 82/100     | 68/100       | 0.730   | 0.720       | 0.8 ✗ |
| mtrag-human-rag         | 0.673   | 27/55      | 47/55        | 0.682   | 0.582       | (out of scope, reported) |

2.5× the data moved both single-turn suites up two points and left multi-turn flat; the held-out
QuAC gain (0.70 → 0.73) did not transfer to MT-RAG, whose answerable recall stays at half. MS MARCO
is still the blocking suite. Pilot 3 (same records, 3,072-token cut) tests the length lever next.

### Why pilot 3 was stopped and pilot 4 is MS MARCO-weighted (2026-09-17T05:35Z)

Measured on pilot 2's MS MARCO window report: the 32 unanswerable cases it misses average 938
prompt tokens and 10 passages, the 68 it catches 979 tokens and 10 passages. MS MARCO prompts are
far below the 2,048-token cut, so raising the cut (pilot 3) changes no MS MARCO training record; it
only restores long QuAC dialogues, which bear on MT-RAG, and MT-RAG is outside the gate. Pilot 3 was
stopped after ~40 minutes; no result is claimed for it.

Pilot 4 changes the mix toward the blocking suite: per label, QuAC 1,500 / SQuAD 1,500 / MS MARCO
4,500 (`prepared-mm-manifest.json`, train sha256 3edcf3dc…), 7,000 records sampled, 3,072-token
cut kept, same seed, learning rate 1.5e-4. It therefore changes two things at once relative to
pilot 2 (mix and cut); that confound is accepted because the gate turns on MS MARCO and the frozen
window read decides. Validation uses all 900 held-out records (same ids as pilots 1–2, different
order in the file), so its held-out numbers are not directly comparable with the 300-record numbers
above; the window numbers are.

## Pilot 4 (2026-09-17, MS MARCO-weighted mix, 3,072-token cut, 376 steps)

Held-out validation (all 900 records, final step only): balanced 0.796; by source SQuAD 0.830,
MS MARCO 0.787, QuAC 0.770. Adapter sha256 in `pilot4/adapter.sha256`
(`adapter_model.safetensors` b17e8424…).

### Pilot 4 on the frozen window (PEFT reference, bf16 base, CPU; window ea9e4a0c)

| suite | answerable | unanswerable | balanced | pilot 2 | IBM adapter |
|---|---|---|---|---|---|
| SQuAD v2 dev | 95/100 | 70/100 | **0.825** | 0.860 | 0.795 |
| MS MARCO v2.1 validation | 87/100 | 61/100 | **0.740** | 0.750 | 0.720 |
| MT-RAG human | 30/55 | 41/55 | 0.645 | 0.673 | 0.582 |

Strict and lenient agree; structured rate 1.0 on every suite. **Gate (≥ 0.80 on SQuAD and MS MARCO):
not met — MS MARCO 0.740.** Tripling MS MARCO's share of training moved its held-out training
split but not the window's validation split (0.750 → 0.740, within noise at n=200), and cost SQuAD
and MT-RAG a few points. Measured conclusion: data re-weighting is not the lever for MS MARCO.
Believed, to be tested next: MS MARCO's "No Answer Present." label is noisy (annotators could mark
no answer while a passage does answer), which would cap any model's unanswerable recall on that
suite below the gate.

## MS MARCO label audit (2026-09-17T11:27Z; diagnostic, not a gate)

**Question.** Are the MS MARCO window's "unanswerable" misses model errors or label errors?

**Groups.** Every window case was grouped by the three adapters' predictions (pilot 2, pilot 4, IBM):
- 27 unanswerable-labelled cases missed by all three;
- 51 caught by all three;
- 22 mixed.

**Sample.** All 27 shared misses, plus 20 random caught-by-all unanswerables and 20 random
correct-by-all answerables (seed 20260917). The 67 cases were shuffled and stripped of labels and
predictions (`msmarco-label-audit/msm-audit-blind.json`; key in `msm-audit-key.json`).

**Judges.** Two independent model judges (Claude Opus, Claude Sonnet) read each query and its
passages and labelled it answerable, unanswerable or ambiguous, quoting evidence.

| group (by adapters) | n | both judges "answerable" | both "unanswerable" | judges agree |
|---|---|---|---|---|
| labelled unanswerable, missed by all 3 adapters | 27 | **19** | 3 | 23 |
| labelled unanswerable, caught by all 3 | 20 | 1 | 11 | 15 |
| labelled answerable, correct for all 3 | 20 | 20 | 0 | 20 |

The controls show that the judges separate the classes: 20/20 on answerables, and 1/20 answerable
among caught unanswerables. Yet on the shared misses both judges found an answering passage in 19
of 27 (for example, "A simple majority — that is, one more in favor than opposed", labelled no-answer).

**Reading (believed; model judges are not ground truth).** At least ~19 of the window's 100
MS MARCO unanswerable labels appear to be wrong. That caps a correct model's strict unanswerable
recall near 0.81. With pilot 4's measured answerable recall (0.87), even perfect unanswerable
judgement would score about 0.84 balanced, so label noise alone eats most of the margin above the
0.80 gate. Correcting only those 19 labels would move pilot 4 from 0.740 to 0.81 if
they are dropped, or 0.82 if they are relabelled answerable (arithmetic, not a measurement). The noise explains most, not all, of the shortfall:
pilot 4 still misses unanswerables the judges agree are unanswerable.

**Not done.** The gate is not re-scored on relabelled cases. Changing the instrument after
seeing results is post-hoc; any relabelled or replacement suite must be pre-registered and applied
to every arm (IBM, pilot 2, pilot 4) unchanged.

## Pre-registration: adjudicated MS MARCO suite (written 2026-09-17T11:28Z, before any adjudication of the remaining cases)

**Flag.** This instrument change is motivated by the audit above, which saw adapter results. It
is post-hoc in origin, and it is reported as such beside the original gate result. The original
result stands: all three adapters fail the pre-registered MS MARCO suite.

**Labelling.**
- The remaining 133 MS MARCO window cases are judged blind, exactly as in the audit: same prompt,
  same two judges, shuffled, no labels or predictions.
- The 67 audit judgements are reused unchanged.

**Adjudicated label** for each case:
- the dataset label, if at least one judge agrees with it;
- the opposite label, if both judges contradict the dataset label with a definite judgement;
- otherwise (ambiguous from both, or one ambiguous and one contradicting) the case is **excluded**.

The adjudicated suite is fixed before any arm is re-scored. Its case ids, labels and sha256 are
committed first.

**Scoring.**
- No model is re-run: the existing window predictions are re-scored for every arm (IBM adapter,
  pilot 1, pilot 2, pilot 4 PEFT reference).
- The adjudicated suite also re-scores the Java arms already recorded (Rust and pure-Java
  specialist/base).

**Rule, unchanged.** A candidate passes if balanced accuracy is ≥ 0.80 on SQuAD v2 dev and on the
adjudicated MS MARCO suite. The report shows original and adjudicated numbers side by side, with
the count of flipped and excluded cases.

**Stated in advance.**
- Model judges are not ground truth. They can share systematic biases with model-based adapters.
- The audit's controls (20/20 answerables, 1/20 caught unanswerables) bound but do not remove that
  risk.
- If the adjudicated suite excludes more than 25 % of cases, it is reported as unusable and no
  pass is claimed from it.

## Pilot 5: no MS MARCO training data (pre-registered 2026-09-17T12:20Z, at launch, before any result)

**Why.** Pilots 1–4 all trained on MS MARCO v2.1 train records. MS MARCO's terms
(microsoft.github.io/msmarco, read 2026-09-17) state: "The MS MARCO datasets are intended for
non-commercial research purposes only." Publishing those adapters on ModelJars needs a licensing
decision. Pilot 5 removes the question: MS MARCO is used only as an evaluation window, never as
training data.

**Recipe.** Identical to pilot 2 except for the data:
- Data: the same prepared set with every `msmarco-v2.1-train` record removed from train, leaving
  SQuAD v2 train and QuAC train, 4,000 per label each (`prepared-nomsm/train.jsonl`, sha256
  2000c773…). Validation is unchanged.
- Training: 6,000 sampled, 4,378 used at ≤ 2,048 tokens, 318 steps, lr 1.5e-4, nf4 base, final
  validation only.

**Licences of what remains.** SQuAD v2.0 and QuAC are CC BY-SA 4.0 (believed from their project
pages; to be recorded with pinned citations before any publication). Attribution and share-alike
implications for adapter weights need the same licensing review, but neither is non-commercial.

**Rule.** Unchanged:
- **Reference path:** confirmed-label window through the PEFT reference, with ≥ 0.80 on
  confirmed SQuAD and confirmed MS MARCO.
- **Java path, only if the reference path passes:** the unchanged Java gate at the Models
  release commit v0.3.42 (6063076e), pure Java deciding.
