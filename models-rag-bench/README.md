# Production RAG benchmark

`models-rag-bench` is one controlled RAG workload with three Java application
paths:

- plain Java using the Models runtime directly
- LangChain4j using `DefaultRetrievalAugmentor`
- Spring AI using `RetrievalAugmentationAdvisor`

The Python baseline uses the same committed corpus and prompt contract with
BM25S, the official Ollama client, revision-matched llama.cpp server HTTP, or an
optional direct `llama-cpp-python` binding.

## Workload contract

The synthetic Northstar policy corpus has 12 documents, eight answerable
questions, and one unanswerable question. Every answerable case declares its
required facts and source IDs. Answers must cite the source, while the
unanswerable case must return exactly `INSUFFICIENT_CONTEXT`.

The controlled comparison uses top-1 BM25 retrieval because each case has one
relevant source. Lucene and BM25S agree on every top result and all nine
rendered prompt hashes. Higher `--top-k` values are supported for experiments,
but reports with different prompt hashes are not directly comparable.

The report records:

- artifact and corpus SHA-256
- retrieval and framework overhead p50/p95
- model load, TTFT, TPOT, end-to-end p50/p95, prefill and decode throughput
- process CPU and Linux peak RSS when `--pid` is supplied
- Recall@1, MRR, fact coverage, citation recall/precision, abstention accuracy,
  and complete-answer accuracy
- failures, backend controls, hardware, OS, JDK or Python identity

## Java applications

Build the executable distribution:

```shell
./gradlew :models-rag-bench:installDist
```

Run the pure-Java Models backend through each application path:

```shell
for framework in plain-java langchain4j spring-ai; do
  models-rag-bench/build/install/models-rag-bench/bin/models-rag-bench \
    --framework "$framework" \
    --backend pure-java \
    --model ~/.jvllm/models/Qwen3-0.6B-Q4_0.gguf \
    --model-id qwen3-0.6b-q4_0 \
    --context 2048 \
    --threads 8 \
    --max-tokens 64 \
    --warmups 1 \
    --iterations 3 \
    --output "build/reports/rag/qwen3-${framework}-pure-java.json"
done
```

Use the same Java application with a locally running native backend:

```shell
models-rag-bench/build/install/models-rag-bench/bin/models-rag-bench \
  --framework plain-java \
  --backend llama.cpp \
  --backend-version b10012-c71854292 \
  --model ~/.jvllm/models/Qwen3-0.6B-Q4_0.gguf \
  --artifact ~/.jvllm/models/Qwen3-0.6B-Q4_0.gguf \
  --endpoint http://127.0.0.1:8080 \
  --pid "$(pgrep -n llama-server)" \
  --threads 8
```

For Ollama, use `--backend ollama`, the Ollama model tag for `--model`, port
`11434`, and the `ollama serve` PID. Both HTTP modes send the canonical prompt
raw with temperature 0, top-k 1, top-p 1, repetition penalty 1, and seed 42.

## Python baselines

Install the locked baseline and run its tests:

```shell
cd models-rag-bench/python
uv sync --frozen --extra test
uv run --frozen pytest -q
```

Run Ollama or the same llama.cpp server used by Java:

```shell
uv run models-rag-python \
  --backend ollama \
  --model qwen3:0.6b \
  --endpoint http://127.0.0.1:11434 \
  --threads 8 \
  --iterations 3

uv run models-rag-python \
  --backend llama.cpp-server \
  --model Qwen3-0.6B-Q4_0.gguf \
  --artifact ~/.jvllm/models/Qwen3-0.6B-Q4_0.gguf \
  --endpoint http://127.0.0.1:8080 \
  --threads 8 \
  --iterations 3
```

The optional direct binding intentionally has a separate backend label because
its bundled llama.cpp revision can differ from the pinned server:

```shell
uv sync --frozen --extra llama
uv run models-rag-python \
  --backend llama.cpp-python \
  --model Qwen3-0.6B-Q4_0.gguf \
  --artifact ~/.jvllm/models/Qwen3-0.6B-Q4_0.gguf
```

## Project gates

`PRODUCTION_READY` requires every trial to succeed, at least 95% retrieval
recall, 0.90 MRR, 90% fact/citation/complete-answer accuracy, perfect
abstention, p95 retrieval at or below 100 ms, p95 TTFT at or below 1 second,
p95 TPOT at or below 100 ms, and p95 end-to-end latency at or below 5 seconds.
`USABLE` relaxes latency to 250 ms, 2 seconds, 200 ms, and 10 seconds without
relaxing quality.

These are ModelJars project SLOs, not universal industry thresholds. They are
documented and justified in `RAG_BENCHMARKS.md` together with controlled
results. Release qualification should use more iterations and realistic
concurrency than the short diagnostic commands above.
