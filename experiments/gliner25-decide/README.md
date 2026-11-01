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
- `remote-worker.sh`: disposable CPU bootstrap; 110-minute shutdown, incremental result upload,
  isolated Python environment, immutable snapshot and one source checkout. No weights or real
  model inference run on the developer laptop.

The first evidence stage is correctness/parity. `parity-cases.jsonl` is a diagnostic suite, not
an accuracy benchmark. External held-out cohorts and final performance buckets must be frozen
before those gates run. A library built from this branch is development evidence; final catalog
admission still requires a released-runtime repeat and ModelJars integration.

## Local checks (no real weights)

```
python3 -m unittest discover -s experiments/gliner25-decide -p 'test_*.py'
./gradlew :backend-java:test --tests '*GlinerClassificationTest' --tests '*DebertaV2EncoderTest' --tests '*DebertaV2ForwardPassTest'
./gradlew :backend-native:test --tests '*NativeHuggingFaceTokenizerTest'
cargo test --manifest-path backend-native/src/main/rust/model-kernels/Cargo.toml hf_tokenizer --lib
```

Native tokenizer reference API: https://docs.rs/tokenizers/0.22.2/tokenizers/tokenizer/struct.TokenizerImpl.html
Publisher source: https://github.com/fastino-ai/GLiNER2/tree/ca4f68235b50fe4c3ecf3876851d748fe4b15c6c
