# decide-latency

One number, on your hardware: **how long does one typed decision take?**

An external Java decision library reports 0.22 s per message for a 4-bit Qwen2.5 1.5B and 0.98 s
for a 7B, measured on an M1 Pro through an HTTP server that returns log-probabilities. Our own
measurements of the same shape of work, on four physical CPU cores, put a decision at 1.4 to 2.3 s
p50. Those are different machines, different model sizes and different runtimes, so neither number
tells you what your hardware will do. This script finds out.

## Run it

```
jbang DecideLatency.java
```

That runs the in-process arm alone: it downloads Qwen2.5-1.5B-Instruct-Q4_K_M once (about 1.1 GB,
cached afterwards), then answers fifteen triage messages and reports where the time went.

To include the over-server arm, start a server on the **same GGUF** and point the script at it:

```
# llama.cpp
llama-server -m Qwen2.5-1.5B-Instruct-Q4_K_M.gguf --port 8080
jbang DecideLatency.java --server http://localhost:8080/v1 --server-model qwen2.5-1.5b-instruct

# or Ollama
ollama serve & ollama pull qwen2.5:1.5b
jbang DecideLatency.java --server http://localhost:11434/v1 --server-model qwen2.5:1.5b
```

| flag | meaning |
|---|---|
| `--server URL` | enable the over-server arm against an OpenAI-compatible `/v1` endpoint |
| `--server-model NAME` | the model name that endpoint expects |
| `--no-local` | skip the in-process arm (no download) |
| `--json FILE` | also write the results as JSON, for collecting across machines |

## Reading the output

Times are itemized on purpose, because a single average hides the thing people actually hit:

- **open** — resolving, verifying and mapping the weights. Paid once per process. The over-server
  arm's open is near zero because the server is already holding the weights; its cost was paid
  before you started timing.
- **first decision** — includes JIT warm-up, and is reported separately so a cold start does not
  get read as the steady-state speed.
- **p50 / p95 warm** — what a request actually costs once the process is warm. This is the number
  to compare.

The script prints the backend it resolved rather than assuming one, because the same model file can
run through different kernels and a latency figure whose route is unknown is not comparable to
anything.

## What it does not tell you

Agreement with the human label over fifteen messages is printed only to catch an arm answering
nonsense — fifteen cases cannot support an accuracy claim and the script says so. For accuracy, run
a real corpus through `models-bench`'s `triage-decisions` command.

The over-server comparison is only a comparison of *runtimes* if the server is serving the same
GGUF the script downloaded. Serve a different file and you have measured two models.

## Requirements

Java 25 or newer (the runtime uses the Vector API and the FFM API), and JBang. The script pins every
coordinate it uses, resolved from Maven Central on 2026-11-03, so two people running it are running
the same thing.
