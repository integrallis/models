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
controlled paired workload; model-contribution, absolute and relative gates;
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

## Bonsai default RAG smoke result

`bonsai2-default.json` and its unmodified `.log` retain all nine cases from the
predeclared default workload. `bonsai2-default-provenance.json` records file hashes
and runtime identity. The source build is `3e4f0c56471bd0f4217f9ead6853ed737664e030`;
the five runtime/harness source trees listed in the receipt are unchanged at the
0.3.58 release commit. This is still a source-build run, not a published-artifact run.

All nine attempts completed without failure. Four returned `MODEL_ANSWER`, four
returned `EXTRACTIVE_FALLBACK`, and one returned `RETRIEVAL_ABSTENTION`. All returned
answers and the abstention passed the workload checks. Model-answer rate was 4/9
(44.44%); correctness among those four answers was 100%; no answer was truncated.
The raw-output evaluator also marked all nine correct. This small synthetic smoke
set does not establish general answer quality, and pipeline correctness does not
mean the model supplied every returned answer.

The observed median decode rate was 0.2399 tokens/second and p95 end-to-end time
was 1028.6 seconds. Builds and recordings shared this Intel host, there was one
iteration and no warmup, and there was no paired comparator. These timings are
retained observations, not controlled performance claims. `peakRssBytes: 0` means
no RSS observation was captured; it must not be interpreted as zero memory use.
Diagnostics report that PTQ1_0 has no retained batched-prefill kernel.

The absolute model-contribution smoke floors are met, but this does not satisfy
the full production qualification gate. Authoritative same-artifact comparison,
controlled relative gates and released-runtime workload evidence remain open.
Bonsai remains an unpublished catalog candidate.

Reproduce the declared source-build smoke (the complete command is also reflected
in the report settings):

```sh
JAVA_OPTS='--enable-native-access=ALL-UNNAMED -Xmx8g -XX:ActiveProcessorCount=12' \
  models-rag-bench/build/install/models-rag-bench/bin/models-rag-bench \
  --framework plain-java --backend pure-java \
  --backend-version models-0.3.58-source-3e4f0c56 \
  --model /absolute/path/to/bonsai2.gguf \
  --model-id prism_ml_ternary_bonsai_2_27b_gguf_ptq1_0 \
  --workload general --prompt-template chatml-no-think \
  --context 2048 --threads 12 --top-k 1 --max-tokens 64 \
  --warmups 0 --iterations 1 --output /tmp/bonsai2-default.json
```
