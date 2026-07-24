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

Use `--prompt-template chatml` for ChatML-family instruction models such as
Qwen3, SmolLM2, and MiniCPM5. The benchmark applies the envelope itself and
sends raw requests to native servers, ensuring every backend receives the same
model-facing bytes. `--prompt-template raw` remains available for base models
and is the default so template selection is never hidden. Use
`--prompt-template chatml-no-think` for reasoning models whose GGUF template
supports `enable_thinking=false`; it prefills the template's empty reasoning
block so the measured output budget contains the answer.

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
    --modeljar qwen3_0_6b_q4_0 \
    --model-id qwen3-0.6b-q4_0 \
    --prompt-template chatml \
    --context 2048 \
    --threads 8 \
    --max-tokens 64 \
    --warmups 1 \
    --iterations 3 \
    --output "build/reports/rag/qwen3-${framework}-pure-java.json"
done
```

`--modeljar` resolves the exact artifact and applies only performance profiles
whose artifact, processor, JVM, vector width, and required launch arguments all
match. Use the catalog's recommended Java runtime and startup arguments before
launching the benchmark. `--model /path/to/model.gguf` remains available for an
unprofiled control and intentionally cannot select artifact-specific profiles.
The JSON report embeds the backend diagnostics so a published run proves which
profiles were enabled.

Use the same Java application with a locally running native backend:

```shell
models-rag-bench/build/install/models-rag-bench/bin/models-rag-bench \
  --framework plain-java \
  --backend llama.cpp \
  --backend-version b10012-c71854292 \
  --model ~/.jvllm/models/Qwen3-0.6B-Q4_0.gguf \
  --artifact ~/.jvllm/models/Qwen3-0.6B-Q4_0.gguf \
  --prompt-template chatml \
  --endpoint http://127.0.0.1:8080 \
  --pid "$(pgrep -n llama-server)" \
  --threads 8
```

For Ollama, use `--backend ollama`, the Ollama model tag for `--model`, port
`11434`, and the `ollama serve` PID. Both HTTP modes send the canonical prompt
raw with temperature 0, top-k 1, top-p 1, repetition penalty 1, and seed 42.

## Hosted API comparison

The same plain Java application can measure economical hosted controls. API
keys are read only from `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, or
`DEEPSEEK_API_KEY`; keys are not accepted as CLI options and are never written
to reports.

```shell
models-rag-bench/build/install/models-rag-bench/bin/models-rag-bench \
  --framework plain-java \
  --backend openai \
  --backend-version hosted-api-2026-07-23 \
  --model gpt-5.4-nano-2026-03-17 \
  --prompt-template raw \
  --max-tokens 64 \
  --warmups 1 \
  --iterations 3
```

The certified hosted controls are OpenAI `gpt-5.4-nano-2026-03-17`, Anthropic
`claude-haiku-4-5-20251001`, and DeepSeek `deepseek-v4-flash` with thinking
disabled. Reports embed the exact pricing snapshot and source URL, normalized
input/cache/output usage, measured request cost, and a projected API cost per
1,000 requests. Pricing is intentionally pinned: adding another provider model
requires an explicit reviewed pricing profile instead of silently applying an
unverified rate.

Hosted results are a developer-experience comparison, not an engine
microbenchmark. They include Internet transport, provider queueing, and
provider-side serving, while local Models, llama.cpp, and Ollama runs do not.

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
  --prompt-template chatml \
  --threads 8 \
  --iterations 3

uv run models-rag-python \
  --backend llama.cpp-server \
  --model Qwen3-0.6B-Q4_0.gguf \
  --artifact ~/.jvllm/models/Qwen3-0.6B-Q4_0.gguf \
  --endpoint http://127.0.0.1:8080 \
  --prompt-template chatml \
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
  --artifact ~/.jvllm/models/Qwen3-0.6B-Q4_0.gguf \
  --prompt-template chatml
```

## Project gates

`PRODUCTION_READY` requires every trial to succeed, at least 95% retrieval
recall, 0.90 MRR, 90% fact/citation/complete-answer accuracy, perfect
abstention, p95 retrieval at or below 100 ms, p95 TTFT at or below 1 second,
p95 TPOT at or below 100 ms, and p95 end-to-end latency at or below 5 seconds.
`USABLE` relaxes latency to 250 ms, 2 seconds, 200 ms, and 10 seconds without
relaxing quality.

These are ModelJars project SLOs, not universal industry thresholds. They are
documented and justified in [RAG_BENCHMARKS.md](../RAG_BENCHMARKS.md) together
with controlled results. Release qualification should use more iterations and
realistic concurrency than the short diagnostic commands above.
