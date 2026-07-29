# Production RAG benchmark

[![MFCQI](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/integrallis/models/main/models-rag-bench/.github/badges/mfcqi.json)](https://github.com/integrallis/mfcqi-java)

`models-rag-bench` provides versioned controlled RAG workloads through three
Java application paths:

- plain Java using the Models runtime directly
- LangChain4j using `DefaultRetrievalAugmentor`
- Spring AI using `RetrievalAugmentationAdvisor`

The Python general-workload baseline uses the same committed corpus and prompt
contract with BM25S, the official Ollama client, revision-matched llama.cpp
server HTTP, or an optional direct `llama-cpp-python` binding.

## Workload contract

`--workload general` selects the synthetic Northstar policy corpus.
`--workload coding` selects Java, Gradle, HTTP, database, and JSON implementation
notes intended to evaluate coding-specialized models. Each workload has 12
documents, eight answerable questions, and one unanswerable question. Every
answerable case declares its required facts and source IDs, while the
unanswerable case must return exactly `INSUFFICIENT_CONTEXT`. `general` remains
the default.

The versioned domain workloads are `finance`, `healthcare`, `legal`, `math`,
`multilingual`, `sql`, and `transportation`. They use the same 12-document,
nine-case contract and synthetic facts, so they measure retrieval, grounding,
abstention, model contribution, and latency without presenting benchmark
content as professional advice. Qualification reports pin each corpus SHA-256;
reports from different workloads or corpus revisions are not comparable.

Models are encouraged to emit source IDs, but source formatting is not treated
as generation quality. When an otherwise supported answer omits citations, the
grounding layer retains the model text, appends retrieved provenance, and records
`MODEL_ANSWER_WITH_DERIVED_CITATIONS`. Unsupported output still uses an
extractive fallback.

The controlled comparison uses top-1 BM25 retrieval because each case has one
relevant source. Lucene and BM25S agree on every top result and all nine
rendered prompt hashes. Higher `--top-k` values are supported for experiments,
but reports with different prompt hashes are not directly comparable.

Use `--prompt-template chatml` for ChatML-family instruction models such as
Qwen3 and SmolLM2. The benchmark applies the envelope itself and
sends raw requests to native servers, ensuring every backend receives the same
model-facing bytes. The grounding policy is a native system turn and the
retrieved evidence plus question is a native user turn. `--prompt-template raw`
remains available for base models and is the default so template selection is
never hidden. Use
`--prompt-template chatml-no-think` for reasoning models whose GGUF template
supports `enable_thinking=false`; it prefills the template's empty reasoning
block so the measured output budget contains the answer. Use
`--prompt-template zephyr` for Zephyr-family chat models such as TinyLlama
1.1B Chat; it places the grounding policy in the model's system turn and the
evidence plus question in its user turn, then emits EOS and assistant-generation
markers explicitly. The source-compatible `llama3`, `gemma`, `phi3`, and
`deepseek` templates cover Llama 3 Instruct, Gemma Instruct, Phi-3/3.5, and
DeepSeek-Coder models respectively. Gemma merges the system instructions into
its first user turn, as required by its chat format. Prompt rendering does not
emit BOS markers when the GGUF tokenizer owns that setting. MiniCPM5 is the
exception: use `--prompt-template minicpm5-no-think`, which emits its
template-owned `<s>` marker before the ChatML turns and suppresses reasoning.

The report records:

- artifact and corpus SHA-256 plus the workload ID
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
    --prompt-template chatml \
    --context 2048 \
    --threads 8 \
    --max-tokens 64 \
    --warmups 1 \
    --iterations 3 \
    --output "build/reports/rag/qwen3-${framework}-pure-java.json"
done
```

Run the Models-owned Rust/FFM projection backend through the same application
paths:

```shell
export JAVA_OPTS="--enable-native-access=ALL-UNNAMED \
  -Dmodels.native.kernels.library=/absolute/path/to/libjmodels_kernels"

for framework in plain-java langchain4j spring-ai; do
  models-rag-bench/build/install/models-rag-bench/bin/models-rag-bench \
    --framework "$framework" \
    --backend rust-ffm \
    --model ~/.jvllm/models/smollm2-360m-instruct-q8_0.gguf \
    --model-id smollm2-360m-instruct-q8_0 \
    --workload general \
    --prompt-template chatml \
    --context 2048 \
    --threads 8 \
    --max-tokens 64 \
    --warmups 1 \
    --iterations 3 \
    --output "build/reports/rag/smollm2-${framework}-rust-ffm.json"
done
```

Use the coding workload for a coding-specialized model:

```shell
models-rag-bench/build/install/models-rag-bench/bin/models-rag-bench \
  --framework plain-java \
  --backend rust-ffm \
  --model ~/.jvllm/models/qwen2.5-coder-0.5b-instruct-q8_0.gguf \
  --model-id qwen2.5-coder-0.5b-instruct-q8_0 \
  --workload coding \
  --prompt-template chatml \
  --context 2048 \
  --threads 8 \
  --max-tokens 64 \
  --warmups 1 \
  --iterations 3
```

In-process backends retain the longest exact token prefix between sequential
requests and report the reused and newly evaluated token counts separately.
The final prompt token is always replayed, so an identical prompt never relies
on a stale logits buffer. Backends without checkpoint/rewind support continue
to reset and prefill the complete prompt.

`--model /path/to/model.gguf` identifies the exact artifact under test. Catalog
and contribution tooling can resolve an independently published model artifact
before invoking the benchmark. The JSON report embeds backend diagnostics so a
published run proves which execution plan was enabled.

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
`11434`, and the `ollama serve` PID. Local runs default to greedy generation:
temperature 0, sampling top-k 1, top-p 1, repetition penalty 1, and seed 42.
The benchmark passes those controls identically to Models, Ollama, and
llama.cpp and records them in every report.

Use `--stop-sequence '\n\n'` to stop at the first paragraph. The CLI decodes
`\n`, `\r`, `\t`, and `\\` escapes, applies the sequence during in-process
generation, sends the same stop array to Ollama and llama.cpp, and records the
escaped value as a matched qualification control. The controlled runner accepts
the same value through `RAG_STOP_SEQUENCE`.

Models that require their published sampling profile can override the local
controls explicitly. For example, MiniCPM5 no-think mode recommends temperature
0.7 and top-p 0.95:

```shell
models-rag-bench/build/install/models-rag-bench/bin/models-rag-bench \
  --framework plain-java \
  --backend rust-ffm \
  --model ~/.jvllm/models/MiniCPM5-1B-Q4_K_M.gguf \
  --model-id minicpm5-1b-q4_k_m \
  --prompt-template chatml-no-think \
  --temperature 0.7 \
  --top-p 0.95 \
  --sampling-top-k 40 \
  --seed 42 \
  --repetition-penalty 1 \
  --threads 8
```

Use the same values for the corresponding Ollama and llama.cpp runs.
Qualification rejects reports whose generation controls differ, so a faster
but semantically different comparator cannot pass the relative gate.

Apply the production policy to retained reports and emit file-hashed evidence:

```shell
models-rag-bench/build/install/models-rag-bench/bin/models-rag-bench qualify \
  --candidate build/reports/rag/models-rust-ffm.json \
  --comparator build/reports/rag/llama.cpp.json \
  --comparator build/reports/rag/ollama.json \
  --output build/reports/rag/qualification.json \
  --require-qualified
```

On the controlled Linux host, the repository driver builds the Models-owned
native library, runs the three backends sequentially against the same artifact
and workload, verifies every artifact digest, and applies that gate:

```shell
scripts/run-controlled-rag-qualification.sh \
  ~/.jvllm/models/EuroLLM-1.7B-Instruct.Q4_K_M.gguf \
  eurollm-1.7b-q4-k-m \
  multilingual \
  chatml
```

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
