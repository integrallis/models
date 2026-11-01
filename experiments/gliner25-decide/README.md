# GLiNER2.5-Decide qualification work

Started 2026-11-01 from Models main `5d4de4262045afe825f295e022c9a64ce1e1e6e7`.
Candidate: `fastino_gliner2_5_decide_f32`. Nothing here grants catalog admission.

`policy.json` registers the comparator, quality/parity gates, final replication rule and
aggregate two-hour rented-compute cap before model runs. Ollama is not a comparator for this
schema classifier: the user explicitly authorized a dedicated publisher-runtime harness.
Both arms run on CPU with the same thread budget and verified immutable snapshot.

## Implementation decision

Reuse the existing DeBERTa encoder, adding F32 and explicit encoder namespaces. GLiNER's
classifier uses the contextual label markers and a 1024/2048/1 ReLU head. The stored span/count
heads are not executed for classification. Existing reranker execution is retained.

The pinned tokenizer differs materially from the reranker's tokenizer: Replace/NFC/right-Strip
normalization versus precompiled SentencePiece normalization. Use a narrow FFM shim around
Hugging Face `tokenizers=0.22.2` rather than an approximate second implementation. Rust owns
opaque handles; Java owns input/output buffers. Calls are serialized with close; Rust catches
panics and rejects stale handles. Cargo.lock and the Linux Python dependency lock are committed.
The runtime is Java encoder plus Rust tokenizer; it is not advertised as a pure-Java backend.

## Harness

- `snapshot.json`: all six immutable files, byte sizes and digests.
- `reference_worker.py`: pinned publisher inference API; optional read-only hooks capture IDs
  and classifier logits for parity. No replacement inference loop.
- `GlinerWorker.java`: production Java encoder/head plus FFM tokenizer. Diagnostic requests
  record IDs, structural marker positions and raw logits. Normal requests measure classify().
- `run_worker.py`: bounded subprocess driver, case identity, readiness, errors and wall times.
- `compare_parity.py`: fail-closed parity report. Passing parity is never catalog qualification.
- `quality-cases.jsonl` / `cohorts.json`: 529 frozen cases (120 JevBench, 150 domain-scoped
  CLINC banking, 259 GoEmotions), with deterministic selection and source hashes/licenses.
- `assess_quality.py`: absolute accuracy/F1, coverage, per-label precision/recall, single-label
  ECE and paired oracle loss. Calibration is reported without granting a confidence claim.
- `GeneratePerformanceCases.java`: metadata-only generator for 13 fixed shape buckets,
  32/128/256/512 encoded tokens, 2/8/32 labels where they fit, and one/two tasks. Each bucket
  has three warmups and thirty timed requests. These synthetic shapes are not accuracy data.
- `performance-cohort.json`: frozen performance matrix and its digest; batch/concurrency one.
- `compare_performance.py`: same-host CPU pair comparison with wall latency and peak RSS.
  One passing pair cannot satisfy the required three final paired replications.
- `remote-worker.sh`: disposable CPU bootstrap; 110-minute shutdown, incremental result upload,
  isolated Python environment, immutable snapshot and one source checkout. No weights or real
  model inference run on the developer laptop.

The first evidence stage is correctness/parity. `parity-cases.jsonl` is a diagnostic suite, not
an accuracy benchmark. Both held-out cohorts and performance buckets are frozen before their
respective runs. A library built from this branch is development evidence; final catalog
admission still requires a released-runtime repeat and ModelJars integration.

The initial remote parity run passed all nine diagnostic requests on Linux x86_64 CPU, with
maximum absolute logit error 0.0000038316. This does not establish quality or performance.
The inference payload is source revision `6add46ee740f8acd80295741fc8e22bae34f845f`; later
harness/reporting changes do not silently replace that measured runtime.

## Local checks (no real weights)

```
python3 -m unittest discover -s experiments/gliner25-decide -p 'test_*.py'
./gradlew :backend-java:test --tests '*GlinerClassificationTest' --tests '*DebertaV2EncoderTest' --tests '*DebertaV2ForwardPassTest'
./gradlew :backend-native:test --tests '*NativeHuggingFaceTokenizerTest'
cargo test --manifest-path backend-native/src/main/rust/model-kernels/Cargo.toml hf_tokenizer --lib
```

Native tokenizer reference API: https://docs.rs/tokenizers/0.22.2/tokenizers/tokenizer/struct.TokenizerImpl.html
Publisher source: https://github.com/fastino-ai/GLiNER2/tree/ca4f68235b50fe4c3ecf3876851d748fe4b15c6c

## General classification accuracy experiment

The base checkpoint failed the frozen JevBench yes/no and GoEmotions floors in both runtimes.
The user authorized fine-tuning for qualification **and general usage**. The resulting candidate
has a separate identity, `modeljars-gliner25-decide-general-v1`; no trained artifact or improved
accuracy is claimed yet. Its scope remains English schema classification, not generative chat.

`finetune-policy.json` freezes the training recipe and evaluation requirements before training.
The dataset mix covers all ten CLINC intent domains, all 28 GoEmotions labels, and BoolQ
passage-conditioned yes/no decisions. `finetune-sources.json` pins source revisions, downloads,
hashes and licenses: CLINC CC-BY-3.0, GoEmotions Apache-2.0, and BoolQ CC-BY-SA-3.0. Any derived
artifact publication must carry its own provenance and applicable attribution; the original
publisher checkpoint and qualification record are not replaced.

`prepare_finetune.py` only reads dataset files and tokenizer metadata. It verifies the pinned
tokenizer against real publisher input-ID fixtures, rejects inputs beyond 512 tokens before
selection, and prevents normalized text overlap across training, development, qualification,
and the original 529 cases. Its deterministic split contains 65,656 training cases, 990
development cases, and 1,001 fresh qualification cases. Exclusion counts and every split digest
are in `finetune-splits.json`. The fresh holdout is fresh to this fine-tuning experiment; the
base model's pretraining exposure to these public datasets is unknown.

On an ephemeral Linux BF16-capable GPU, install `requirements-training.lock` with hash checking
and run the pinned publisher trainer through the bounded wrapper:

```sh
uv pip sync --python /work/venv/bin/python --require-hashes experiments/gliner25-decide/requirements-training.lock
/work/venv/bin/python experiments/gliner25-decide/train_general.py \
  --snapshot /work/base-snapshot --source /work/publisher \
  --data /work/training-data --output /work/training-output
```

Prepare the source checkout, verified base snapshot and the two hash-verified training/development
JSONL files before starting. Never copy qualification cases into the trainer's data directory.
The output directory must be empty, so a previous run's `best` checkpoint cannot be selected.
The 20-minute training deadline is checked between optimizer steps; a final development evaluation
and export can extend beyond it. Provisioning, training, export and evaluation all count against
the original aggregate two-hour rental cap, enforced separately by the remote lifecycle controller.
No GPU worker is provisioned by this script, and no local real-model inference is permitted.

The selected adapter minimizes development loss. The wrapper reloads that adapter on the original
base, merges to F32, validates all tensor names/shapes/dtypes, copies the original tokenizer and
configuration, and writes a new snapshot manifest and training receipt. Both CPU runtimes must
then load that exact manifest (`reference_worker.py --manifest` and the third positional
argument to `GlinerWorker`).

`assess_finetune.py` requires fresh holdout accuracy, per-domain intent F1, emotion micro/macro F1,
non-regression against the original base, and matched-checkpoint publisher parity in quality.
The original 529 cases remain regression evidence and must still pass all original absolute
floors. Checkpoint selection never uses either qualification suite. Numerical parity, three
passing performance pairs, ModelJars integration and a released-runtime repeat remain separate
requirements. A passing fine-tune quality report never grants catalog admission by itself.
