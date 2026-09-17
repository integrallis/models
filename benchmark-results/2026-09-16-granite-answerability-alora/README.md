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
