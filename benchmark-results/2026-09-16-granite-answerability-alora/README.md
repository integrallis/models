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
