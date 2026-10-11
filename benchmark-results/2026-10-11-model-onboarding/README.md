# October 11 model onboarding evidence

The protocol is [frozen in findings](../../docs/findings/2026-10-11-model-onboarding.md).
`downloads.json` records verified local bytes, not just Hugging Face metadata.

## Text embedding

`embeddinggemma2-red.xml` is the public API failure before the encoder existed.
`embeddinggemma2-integration.xml` records dimension, normalization and sequence isolation.
`embeddinggemma2-oracle.stdout` retains the complete original stdout, including token IDs before
its vector array.
The parsed immutable reference is in `models-bench/src/main/resources/embedding-equivalence/`.
`embeddinggemma2-equivalence.json` is the Java gate result, not a retrieval-quality benchmark.

Reference command (paths resolved locally):

```sh
llama-embedding -m embeddinggemma2.gguf -f probes.txt --pooling mean \
  --attention non-causal --embd-output-format array --embd-normalize 2 \
  --device none -c 512 --verbose-prompt
```

## Bonsai text compatibility

`bonsai2-red.xml` records the previously unsupported PTQ1_0 matrix type.
`bonsai2-scalar-smoke.xml` records the first unoptimized Java compatibility check.
`bonsai2-oracle-tokens.json` captures the independent reference's eight token IDs and top logits.
`bonsai2-integration.xml` checks all eight IDs and a separate session on the same exact weights.
The standalone reference driver links the external Prism build only; no product links it.

```sh
c++ -std=c++17 -O2 -I include -I ggml/include bonsai2-oracle-driver.cpp \
  -L build/bin -Wl,-rpath,/absolute/path/to/prism/build/bin \
  -lllama -lggml -lggml-base -o bonsai2-oracle-tokens
./bonsai2-oracle-tokens /absolute/path/to/bonsai2.gguf
```

The raw completion prompt is `The quick brown fox`, temperature zero, context 512,
12 CPU threads. The Java integration task uses context capacity 64, enough for this short
compatibility check; this is not a matched performance comparison. Builds and diagnostics were
active on this host. Do not derive throughput claims from the test durations.

## Publication constraint

As of 2026-10-11, Ollama v0.40.3 (`eab97e9f92b9a25c2d52d2cc6c1b1c99bd9fae21`)
`fs/gguf/tensor.go` has no type 143 sizing/decoding contract. The open upstream tracker is
https://github.com/ollama/ollama/issues/18521. Prism's backend-support documentation at
`d5ea901004ae983f5bea86dd2e3f5fa1fe915e2c` requires its fork for these weights.
The existing `production-rag-model-contribution-v6` policy cannot qualify a model using only
that fork as comparator. No authoritative baseline was fabricated, no policy floor was changed,
and no performance or RAG-quality qualification is claimed for Bonsai here.

Remaining Bonsai publication gates: supported authoritative engine over the exact artifact;
complete default smoke and controlled workload; model-contribution, absolute and relative gates;
released-runtime evidence; marker staging/finalization and Central verification. Image understanding
also requires its separate projector and modality gates. EmbeddingGemma's other modalities likewise
remain separate work.

## Local release checks

`build-verification.json` records the complete 0.3.58 build/compliance result and test counts.
The refreshed embedding report retains every Java vector and token sequence; all eight token
sequences match the independent oracle exactly. The original report records the working tree. The separate released-artifact report below
provides the catalog publication evidence.

## Released runtime verification

Models 0.3.58 was published by [release run 38156648214](https://github.com/integrallis/models/actions/runs/38156648214)
from `fab199754d219f53623e7810ccc05be68f8186df`. All ten release jobs, including
independent Maven and Gradle consumers, completed successfully. Vectors remains 0.1.29.

`verify-released-runtime.py` fetches each runtime JAR from Maven Central into an isolated
classpath and checks its Central checksum. Only the unpublished measurement harness comes
from the local build. `released-classpath.json` records every coordinate, URL, SHA-256,
command and exit code; `embeddinggemma2-released.log` is its execution log.
`embeddinggemma2-released-equivalence.json` retains all eight Java vectors and token sequences.
All eight probes pass: minimum cosine 0.9998087162015021, maximum norm deviation
7.675834412523841e-9, and 768 output dimensions. These are numerical conformance measurements,
not a retrieval-quality or performance comparison. Other workloads were active on the host.

```sh
python3 benchmark-results/2026-10-11-model-onboarding/verify-released-runtime.py \
  --models /absolute/path/to/models --output /tmp/models-0358-published \
  --model /absolute/path/to/embeddinggemma2.gguf
```
