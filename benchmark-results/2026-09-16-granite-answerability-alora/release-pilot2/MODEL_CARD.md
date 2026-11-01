# Granite 4.1 3B answerability aLoRA (Integrallis, pilot 2)

An activated LoRA (aLoRA) for `ibm-granite/granite-4.1-3b`. At the assistant marker
`<|start_of_role|>assistant<|end_of_role|>` (tokens `[100264, 78191, 100265]`), it answers one
question: can the user's last question be answered from the supplied documents? The output is
`"answerable"` or `"unanswerable"`.

The adapter activates only after the marker. Everything before the marker (system prompt,
documents, conversation) runs with base weights, so a Models runtime can share that KV prefix
physically with the base chat model.

## Shape

- LoRA rank 16, alpha 32, dropout 0.05 during training.
- Targets `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`.
- Same shape and invocation contract as IBM's `granitelib-rag-r1.0` answerability adapter, which
  it replaces.

## Training

- **Data:** SQuAD v2.0 train, QuAC v0.2 train and MS MARCO v2.1 train.
  - 4,000 records per label per source, with up to 4 distractor passages per record; salt
    `20260916-answerability-alora`.
  - 6,000 sampled, 4,924 used at ≤ 2,048 tokens.
  - Prepared-set manifest: `prepared-manifest.json`.
- **Base:** loaded in nf4 (QLoRA) and kept bf16 for compute. Loss on the completion only.
- **Schedule:** one epoch, 364 optimizer steps, lr 1.5e-4, seed 20260916.
- **Tooling:** `prepare_answerability_training.py` and `train_answerability_alora.py` in this
  directory, with the run manifest in `pilot2/training-manifest.json`.

## Evaluation

Reference path: Transformers + PEFT, bf16, greedy.
- **Window:** qualification window v2 (sha256 ea9e4a0c…).
- **Labels:** evidence-confirmed (`label-confirmation/`, see preflight).

| suite | balanced accuracy (confirmed labels) | original dataset labels |
|---|---|---|
| SQuAD v2 dev (200) | 0.873 | 0.860 |
| MS MARCO v2.1 validation (200) | 0.829 | 0.750 |
| MT-RAG human turns (110) | not qualified | 0.673 |

Java-runtime qualification numbers are published with the ModelJars component report. They are
the numbers that decide qualification.

## Limitations

- Scope is single-turn answerability. Multi-turn conversations (MT-RAG) score well below the
  qualification floor.
- Labels on public answerability datasets are noisy. MS MARCO "No Answer Present." labels were
  wrong on a large share of audited cases, so original-label numbers understate accuracy there.

## License and data

- Adapter weights: Apache-2.0. Base model: Apache-2.0 (IBM Granite).
- Training data:
  - SQuAD v2.0 (CC BY-SA 4.0)
  - QuAC (CC BY-SA 4.0)
  - MS MARCO v2.1: Microsoft terms; use confirmed by the publisher.
